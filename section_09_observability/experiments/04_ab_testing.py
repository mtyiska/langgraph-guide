"""
Experiment 4 — A/B prompt testing.
Compares the baseline Section 8 system prompt against a variant
with 3 few-shot examples added.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "section_08_reliability"))

from eval_cases import EVAL_CASES, create_knowledge_base, KB_DIR
import tools as tools_mod
tools_mod.DOCS_DIR = KB_DIR
create_knowledge_base()

from nodes import RESEARCH_SYSTEM
from trace_store import TraceStore
from eval_store import EvalStore
from golden_trajectories import GoldenTrajectoryStore
from ab_testing import run_ab_test

print("=== Experiment 4 — A/B Prompt Testing ===\n")

FEW_SHOT_ADDENDUM = """
EXAMPLES OF GOOD RESEARCH BEHAVIOUR:

Example 1:
User: "What is the project budget?"
Agent action: Call search_documents(query="budget") → find budget_overview.md → read it → report exact figures with citation.

Example 2:
User: "Who leads the frontend work?"
Agent action: Call search_documents(query="frontend engineer") → find team document → cite the specific name.

Example 3:
User: "What happened in Q3?"
Agent action: Call search_documents(query="Q3") → may find multiple docs → read the most relevant → synthesise key points with citations.
"""

PROMPT_A = RESEARCH_SYSTEM  # baseline
PROMPT_B = RESEARCH_SYSTEM + FEW_SHOT_ADDENDUM  # variant with few-shot examples

trace_store  = TraceStore("../data/traces.db")
eval_store   = EvalStore("../data/eval_results.db")
golden_store = GoldenTrajectoryStore("../data/eval_results.db")

# Run on first 10 cases to keep experiment time manageable
cases = EVAL_CASES[:10]

comparison = run_ab_test(
    cases=cases,
    prompt_a=PROMPT_A,
    prompt_b=PROMPT_B,
    label_a="baseline",
    label_b="few_shot",
    trace_store=trace_store,
    eval_store=eval_store,
    golden_store=golden_store,
    pass_threshold=0.65,
    max_iterations=8,
    verbose=True,
)

# ── Decision rationale ─────────────────────────────────────────────────────────
# Adopt variant B (few-shot) if:
#   1. score_delta > 0 (B scores higher on average)
#   2. token_delta is acceptable (few-shot adds ~150-300 tokens per run due to longer prompt)
#   3. No cases degraded that weren't already failing
#
# In practice for smaller Ollama models:
#   - Few-shot examples often improve JSON output format adherence (+0.05–0.15 factual score)
#   - Token cost increase is modest (~10-15% on a 2000-token prompt)
#   - Risk: if examples are poorly written they can confuse smaller models
#
# Verdict: adopt B if score_delta >= 0.02 and token_delta < 500.
adopt = comparison["score_delta"] >= 0.02 and comparison["token_delta"] < 500
print(f"\nDecision: {'ADOPT few-shot variant B' if adopt else 'KEEP baseline A'}")
print(f"  Rationale: score_delta={comparison['score_delta']:+.3f}, token_delta={comparison['token_delta']:+,}")

print("\nExperiment 4 complete.")