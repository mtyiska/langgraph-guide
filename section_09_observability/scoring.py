import json
import os
import logging
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

import sys
sys.path.append("..")
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

primary_llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

JUDGE_SYSTEM = (
    "You are a precise factual evaluator. "
    "Quote exactly from the provided text. Never invent quotes. "
    "Respond with JSON only — no prose, no markdown fences."
)

JUDGE_PROMPT = """Evaluate this AI research assistant answer against expected facts.

ORIGINAL QUESTION:
{query}

AI ANSWER:
{answer}

EXPECTED FACTS (each must be present):
{expected_facts}

FORBIDDEN CLAIMS (hallucinations to detect):
{forbidden_claims}

For each expected fact, find any sentence or phrase in the answer that confirms
the same information, even if worded differently. If the answer conveys the same
meaning as the expected fact, mark as found. Only mark missing if the information
is genuinely absent from the answer.

Respond with exactly this JSON:
{{
  "fact_evaluations": [
    {{
      "expected_fact": "...",
      "status": "found" | "missing" | "contradicted",
      "supporting_quote": "exact short quote or null"
    }}
  ],
  "forbidden_claim_evaluations": [
    {{
      "forbidden_claim": "...",
      "present_in_answer": true | false,
      "evidence": "quote if present, null if absent"
    }}
  ],
  "overall_reasoning": "2-3 sentence summary"
}}"""


def score_with_llm_judge(
    query: str, answer: str, expected_facts: list[str], forbidden_claims: list[str]
) -> tuple[float, float, str, list[str], list[str]]:
    """
    Returns:
        factual_score (0-1),
        hallucination_score (0-1, higher = cleaner),
        judge_reasoning,
        facts_found (list),
        facts_missing (list)
    """
    if not expected_facts:
        return 1.0, 1.0, "No expected facts specified.", [], []

    prompt = JUDGE_PROMPT.format(
        query=query,
        answer=answer[:3000],
        expected_facts="\n".join(f"- {f}" for f in expected_facts),
        forbidden_claims="\n".join(f"- {c}" for c in forbidden_claims) or "None specified.",
    )

    try:
        response = primary_llm.invoke(
            [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)]
        )
        raw = response.content.strip()
        # Strip fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(l for l in lines if not l.strip().startswith("```"))

        data = json.loads(raw)

        fact_evals = data.get("fact_evaluations", [])
        found = [e["expected_fact"] for e in fact_evals if e.get("status") == "found"]
        missing = [e["expected_fact"] for e in fact_evals if e.get("status") != "found"]
        factual_score = len(found) / len(fact_evals) if fact_evals else 1.0

        forbidden_evals = data.get("forbidden_claim_evaluations", [])
        hallucinations_present = sum(
            1 for e in forbidden_evals if e.get("present_in_answer")
        )
        hallucination_score = (
            1.0 - (hallucinations_present / len(forbidden_evals))
            if forbidden_evals
            else 1.0
        )

        reasoning = data.get("overall_reasoning", "")
        return factual_score, hallucination_score, reasoning, found, missing

    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return 0.0, 1.0, f"Judge failed: {e}", [], expected_facts


def score_citations(
    answer_citations: list, expected_citation_files: list[str]
) -> tuple[float, list[str], list[str]]:
    """
    answer_citations: list of citation dicts from validated_answer
    expected_citation_files: list of expected source file names
    """
    if not expected_citation_files:
        return 1.0, [], []

    cited_sources = set()
    for c in answer_citations:
        if isinstance(c, dict):
            cited_sources.add(os.path.basename(c.get("source_path", "")))
        elif isinstance(c, str):
            cited_sources.add(os.path.basename(c))

    expected_set = set(os.path.basename(p) for p in expected_citation_files)
    valid = list(cited_sources & expected_set)
    missing = list(expected_set - cited_sources)
    score = len(valid) / len(expected_set) if expected_set else 1.0
    return score, valid, missing


def score_completeness(answer: str, expected_facts: list[str]) -> float:
    """
    Lightweight keyword-based completeness check.
    Not a substitute for LLM judging — used as a fast supplementary signal.
    """
    if not expected_facts or not answer:
        return 0.0 if expected_facts else 1.0

    answer_lower = answer.lower()
    hits = 0
    for fact in expected_facts:
        # Check if at least half the key words of each fact appear in the answer
        words = [w for w in fact.lower().split() if len(w) > 3]
        if not words:
            continue
        word_hits = sum(1 for w in words if w in answer_lower)
        if word_hits / len(words) >= 0.5:
            hits += 1
    return hits / len(expected_facts)


def score_trajectory(
    actual: list[str], expected: Optional[list[str]]
) -> float:
    if expected is None:
        return 1.0

    expected_set = set(expected)
    actual_set = set(actual)
    coverage = len(expected_set & actual_set) / len(expected_set) if expected_set else 1.0

    extra_steps = max(0, len(actual) - len(expected))
    efficiency_penalty = min(0.3, extra_steps * 0.05)

    return max(0.0, coverage - efficiency_penalty)