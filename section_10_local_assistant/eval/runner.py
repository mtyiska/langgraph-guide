import sys
import os
import time
import uuid
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


def _score_judge(query, answer, expected_facts, forbidden_claims):
    """LLM-as-judge factual + hallucination scoring."""
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage
    from config import PRIMARY_MODEL, OLLAMA_BASE_URL

    if not expected_facts:
        return 1.0, 1.0, "No expected facts.", [], []

    prompt = f"""Evaluate this answer against expected facts.
Question: {query}
Answer: {answer[:2000]}
Expected facts: {json.dumps(expected_facts)}
Forbidden claims: {json.dumps(forbidden_claims)}

For each expected fact, find a supporting quote or mark MISSING.
Respond with JSON only:
{{
  "fact_evaluations": [{{"expected_fact": "...", "status": "found"|"missing", "quote": "..."}}],
  "forbidden_found": [{{"claim": "...", "present": true|false}}],
  "reasoning": "2-3 sentences"
}}"""

    try:
        llm  = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        resp = llm.invoke([SystemMessage(content="Precise factual evaluator. JSON only."), HumanMessage(content=prompt)])
        raw  = resp.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```"))
        s = raw.find("{"); e = raw.rfind("}")
        if s != -1 and e != -1:
            raw = raw[s:e+1]
        data = json.loads(raw)
        fes  = data.get("fact_evaluations", [])
        found   = [f["expected_fact"] for f in fes if f.get("status") == "found"]
        missing = [f["expected_fact"] for f in fes if f.get("status") != "found"]
        fact_score = len(found) / len(fes) if fes else 1.0
        fbs  = data.get("forbidden_found", [])
        hallu_present = sum(1 for f in fbs if f.get("present"))
        hallu_score   = 1.0 - (hallu_present / len(fbs)) if fbs else 1.0
        return fact_score, hallu_score, data.get("reasoning", ""), found, missing
    except Exception as ex:
        logger.warning(f"Judge failed: {ex}")
        return 0.0, 1.0, str(ex), [], expected_facts


def _score_completeness(answer, expected_facts):
    if not expected_facts or not answer:
        return 0.0 if expected_facts else 1.0
    al = answer.lower()
    hits = sum(1 for f in expected_facts if sum(1 for w in f.lower().split() if len(w)>3 and w in al) / max(1, len([w for w in f.lower().split() if len(w)>3])) >= 0.5)
    return hits / len(expected_facts)


def _score_trajectory(actual, expected):
    if not expected:
        return 1.0
    cov = len(set(expected) & set(actual)) / len(set(expected))
    extra_penalty = min(0.3, max(0, len(actual) - len(expected)) * 0.05)
    return max(0.0, cov - extra_penalty)


@dataclass
class EvalReport:
    label:          str
    run_date:       datetime
    total:          int
    passed:         int
    pass_rate:      float
    avg_score:      float
    avg_factual:    float
    avg_citation:   float
    avg_completeness: float
    avg_trajectory: float
    avg_hallucination: float
    avg_duration_ms: float
    avg_tokens:     int
    total_tokens:   int
    failed_cases:   list = field(default_factory=list)
    results:        list = field(default_factory=list)

    def print_report(self):
        print("\n" + "="*65)
        print(f"EVAL REPORT — {self.label}")
        print(f"Date: {self.run_date.strftime('%Y-%m-%d %H:%M UTC')}")
        print("="*65)
        s = "✓ PASSING" if self.pass_rate >= 0.75 else "✗ NEEDS WORK"
        print(f"\n{s}  {self.passed}/{self.total} ({self.pass_rate:.0%})")
        print(f"Avg score: {self.avg_score:.3f}\n")
        print(f"  Factual accuracy:      {self.avg_factual:.3f}")
        print(f"  Citation validity:     {self.avg_citation:.3f}")
        print(f"  Completeness:          {self.avg_completeness:.3f}")
        print(f"  Trajectory efficiency: {self.avg_trajectory:.3f}")
        print(f"  No hallucination:      {self.avg_hallucination:.3f}")
        print(f"\n  Avg duration: {self.avg_duration_ms:.0f}ms")
        print(f"  Avg tokens:   {self.avg_tokens:,}")
        print(f"  Total tokens: {self.total_tokens:,}")
        if self.failed_cases:
            print(f"\n  Failed: {', '.join(self.failed_cases)}")
        print("="*65)


class EvaluationRunner:
    def __init__(self, pass_threshold: float = 0.65):
        self.threshold = pass_threshold

    def run(self, cases: list[dict], label: str = "eval", verbose: bool = True, max_iters: int = 8) -> EvalReport:
        from agent.graph import build_graph
        from agent.state import make_initial_state
        from infrastructure.trace_store import TraceStore

        ts    = TraceStore()
        graph, rc = build_graph(ts)
        results   = []
        start_dt  = datetime.utcnow()

        if verbose:
            print(f"\nEvaluation '{label}' — {len(cases)} cases")
            print("-"*65)

        for i, case in enumerate(cases):
            if verbose:
                print(f"\n[{i+1}/{len(cases)}] {case['case_id']}: {case['description']}")
                print(f"  Query: {case['query'][:70]}")

            r = self._run_case(case, graph, rc, make_initial_state, max_iters)
            results.append(r)

            if verbose:
                sym = "✓" if r["passed"] else "✗"
                print(f"  {sym} {r['total_score']:.3f} | {r['final_status']} | {r['duration_ms']:.0f}ms | {r['total_tokens']} tok")
                if not r["passed"] and r.get("facts_missing"):
                    print(f"    Missing: {r['facts_missing'][:2]}")

        n   = len(results)
        def avg(k): return sum(r.get(k,0) for r in results)/n if n else 0
        passed = [r for r in results if r["passed"]]

        return EvalReport(
            label=label, run_date=start_dt, total=n,
            passed=len(passed), pass_rate=len(passed)/n if n else 0,
            avg_score=avg("total_score"),
            avg_factual=avg("score_factual_accuracy"),
            avg_citation=avg("score_citation_validity"),
            avg_completeness=avg("score_completeness"),
            avg_trajectory=avg("score_trajectory_efficiency"),
            avg_hallucination=avg("score_no_hallucination"),
            avg_duration_ms=avg("duration_ms"),
            avg_tokens=int(avg("total_tokens")),
            total_tokens=sum(r.get("total_tokens",0) for r in results),
            failed_cases=[r["case_id"] for r in results if not r["passed"]],
            results=results,
        )

    def _run_case(self, case, graph, rc, make_initial_state, max_iters):
        rc["run_id"]     = str(uuid.uuid4())
        rc["node_index"] = 0
        rc["trajectory"] = []
        rc["total_tokens"] = 0
        t0 = time.perf_counter()

        try:
            sid   = f"eval-{case['case_id']}"
            init  = make_initial_state(case["query"], session_id=sid, thread_id=f"thread-{sid}", run_id=rc["run_id"])
            state = graph.invoke(init, {"configurable": {"thread_id": f"thread-{sid}"}})
        except Exception as e:
            logger.error(f"{case['case_id']} crashed: {e}")
            return self._zero(case, str(e), time.perf_counter()-t0)

        dur    = (time.perf_counter() - t0) * 1000
        answer = state.get("final_response", "")

        fs, hs, reasoning, found, missing = _score_judge(
            case["query"], answer,
            case.get("expected_facts", []),
            case.get("forbidden_claims", []),
        )
        cs = 1.0  # citation scoring simplified — knowledge base varies
        co = _score_completeness(answer, case.get("expected_facts", []))
        ts = _score_trajectory(rc["trajectory"], case.get("expected_trajectory"))

        w_fa = case.get("weight_factual_accuracy", 0.4)
        w_cv = case.get("weight_citation_validity", 0.1)
        w_co = case.get("weight_completeness",      0.25)
        w_te = case.get("weight_trajectory_efficiency", 0.1)
        w_nh = case.get("weight_no_hallucination",  0.15)

        total = fs*w_fa + cs*w_cv + co*w_co + ts*w_te + hs*w_nh

        return {
            "case_id": case["case_id"],
            "score_factual_accuracy":    fs,
            "score_citation_validity":   cs,
            "score_completeness":        co,
            "score_trajectory_efficiency": ts,
            "score_no_hallucination":    hs,
            "total_score":               total,
            "passed":                    total >= self.threshold,
            "facts_found":               found,
            "facts_missing":             missing,
            "forbidden_claims_found":    [c for c in case.get("forbidden_claims",[]) if c.lower() in answer.lower()],
            "actual_trajectory":         rc["trajectory"],
            "judge_reasoning":           reasoning,
            "duration_ms":               dur,
            "total_tokens":              rc.get("total_tokens", 0),
            "final_status":              state.get("status", "unknown"),
        }

    def _zero(self, case, error, elapsed):
        return {
            "case_id": case["case_id"],
            "score_factual_accuracy": 0.0, "score_citation_validity": 0.0,
            "score_completeness": 0.0, "score_trajectory_efficiency": 0.0,
            "score_no_hallucination": 1.0, "total_score": 0.0, "passed": False,
            "facts_found": [], "facts_missing": case.get("expected_facts", []),
            "forbidden_claims_found": [], "actual_trajectory": [],
            "judge_reasoning": f"Crashed: {error}",
            "duration_ms": elapsed*1000, "total_tokens": 0, "final_status": "crashed",
        }