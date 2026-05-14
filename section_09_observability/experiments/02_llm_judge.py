"""
Experiment 2 — LLM judge consistency and accuracy.
"""

import sys
import os
import statistics

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "section_08_reliability"))

from scoring import score_with_llm_judge

print("=== Experiment 2 — LLM Judge ===\n")

EXPECTED_FACTS = [
    "total approved budget is $320,000",
    "total spend through Q3 is $161,000",
    "remaining budget is $159,000",
]
FORBIDDEN = ["budget is $500,000", "project is over budget"]

ANSWERS = {
    "correct": (
        "The Project Atlas budget is $320,000 total. Through Q3, the team has "
        "spent $161,000, leaving $159,000 remaining. The project is on track."
    ),
    "partial": (
        "Project Atlas has a budget of $320,000. Some amount has been spent "
        "but the exact remaining balance is not clear from the documents reviewed."
    ),
    "hallucinated": (
        "The Project Atlas budget is $500,000 total. The project is over budget "
        "with $161,000 spent and the team is at risk of exceeding capacity."
    ),
}

QUERY = "What is the total budget for Project Atlas and how much has been spent?"

print("Single-run scores per answer variant:")
print(f"  {'Variant':<14} {'Factual':>8} {'No-Halluc':>10} {'Reasoning'}")
print("  " + "-" * 70)

for label, answer in ANSWERS.items():
    fs, hs, reasoning, found, missing = score_with_llm_judge(
        QUERY, answer, EXPECTED_FACTS, FORBIDDEN
    )
    print(f"  {label:<14} {fs:>8.2f} {hs:>10.2f}  {reasoning[:50]}")

print()

# ── Judge consistency test ─────────────────────────────────────────────────────

print("Consistency test: score the correct answer 5 times")
print("(High variance > 0.1 range indicates ambiguous judge prompt)\n")

scores = []
for i in range(5):
    fs, _, _, _, _ = score_with_llm_judge(
        QUERY, ANSWERS["correct"], EXPECTED_FACTS, FORBIDDEN
    )
    scores.append(fs)
    print(f"  Run {i+1}: factual_score = {fs:.3f}")

rng = max(scores) - min(scores)
std = statistics.stdev(scores) if len(scores) > 1 else 0
print(f"\n  Range: {rng:.3f}  Std dev: {std:.3f}")
if rng <= 0.1:
    print("  ✓ Low variance — judge is consistent")
else:
    print("  ✗ High variance — tighten the judge prompt scoring criteria")

print("\nExperiment 2 complete.")