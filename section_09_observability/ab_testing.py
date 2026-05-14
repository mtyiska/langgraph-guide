import sys
import os
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


@contextmanager
def patched_system_prompt(new_prompt: str):
    """
    Context manager that temporarily replaces the RESEARCH_SYSTEM prompt
    in both nodes.py and graph.py (which import it).
    """
    # Import the modules whose globals we need to patch
    section08_path = os.path.join(
        os.path.dirname(__file__), "..", "section_08_reliability"
    )
    if section08_path not in sys.path:
        sys.path.insert(0, section08_path)

    import nodes as nodes_mod
    import graph as graph_mod
    import guardrails as guardrails_mod

    original_nodes = nodes_mod.RESEARCH_SYSTEM
    original_graph = getattr(graph_mod, "RESEARCH_SYSTEM", None)

    nodes_mod.RESEARCH_SYSTEM = new_prompt
    if hasattr(graph_mod, "RESEARCH_SYSTEM"):
        graph_mod.RESEARCH_SYSTEM = new_prompt

    try:
        yield
    finally:
        nodes_mod.RESEARCH_SYSTEM = original_nodes
        if original_graph is not None:
            graph_mod.RESEARCH_SYSTEM = original_graph


def run_ab_test(
    cases: list[dict],
    prompt_a: str,
    prompt_b: str,
    label_a: str = "prompt_a",
    label_b: str = "prompt_b",
    trace_store=None,
    eval_store=None,
    golden_store=None,
    pass_threshold: float = 0.65,
    max_iterations: int = 10,
    verbose: bool = True,
) -> dict:
    from eval_runner import EvaluationRunner

    runner = EvaluationRunner(
        trace_store=trace_store,
        eval_store=eval_store,
        golden_store=golden_store,
        pass_threshold=pass_threshold,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"A/B TEST: '{label_a}' vs '{label_b}'")
        print(f"{'='*60}")
        print(f"\nVariant A — {label_a}")

    with patched_system_prompt(prompt_a):
        report_a = runner.run_evaluation(
            cases, eval_label=label_a, verbose=verbose, max_iterations=max_iterations
        )

    if verbose:
        print(f"\nVariant B — {label_b}")

    with patched_system_prompt(prompt_b):
        report_b = runner.run_evaluation(
            cases, eval_label=label_b, verbose=verbose, max_iterations=max_iterations
        )

    # Compare
    cases_improved = []
    cases_degraded = []
    results_a = {r["case_id"]: r for r in report_a.results}
    results_b = {r["case_id"]: r for r in report_b.results}

    for cid in results_a:
        if cid in results_b:
            delta = results_b[cid]["total_score"] - results_a[cid]["total_score"]
            if delta > 0.02:
                cases_improved.append(
                    {"case_id": cid, "delta": round(delta, 3)}
                )
            elif delta < -0.02:
                cases_degraded.append(
                    {"case_id": cid, "delta": round(delta, 3)}
                )

    score_delta = report_b.avg_total_score - report_a.avg_total_score
    pass_rate_delta = report_b.pass_rate - report_a.pass_rate
    token_delta = report_b.avg_tokens_per_run - report_a.avg_tokens_per_run
    latency_delta = report_b.avg_duration_ms - report_a.avg_duration_ms
    recommendation = label_b if score_delta > 0 else label_a

    comparison = {
        "label_a": label_a,
        "label_b": label_b,
        "score_a": round(report_a.avg_total_score, 3),
        "score_b": round(report_b.avg_total_score, 3),
        "score_delta": round(score_delta, 3),
        "pass_rate_a": round(report_a.pass_rate, 3),
        "pass_rate_b": round(report_b.pass_rate, 3),
        "pass_rate_delta": round(pass_rate_delta, 3),
        "avg_tokens_a": report_a.avg_tokens_per_run,
        "avg_tokens_b": report_b.avg_tokens_per_run,
        "token_delta": token_delta,
        "avg_latency_a_ms": round(report_a.avg_duration_ms, 0),
        "avg_latency_b_ms": round(report_b.avg_duration_ms, 0),
        "latency_delta_ms": round(latency_delta, 0),
        "cases_improved": cases_improved,
        "cases_degraded": cases_degraded,
        "recommendation": recommendation,
    }

    if verbose:
        print(f"\n{'='*60}")
        print("A/B RESULT")
        print(f"{'='*60}")
        winner = "B is BETTER" if score_delta > 0 else "A is BETTER" if score_delta < 0 else "TIE"
        print(f"  {winner}: recommend '{recommendation}'")
        print(f"  Score:     A={comparison['score_a']}  B={comparison['score_b']}  Δ={score_delta:+.3f}")
        print(f"  Pass rate: A={comparison['pass_rate_a']:.0%}  B={comparison['pass_rate_b']:.0%}  Δ={pass_rate_delta:+.1%}")
        print(f"  Tokens:    A={comparison['avg_tokens_a']:,}  B={comparison['avg_tokens_b']:,}  Δ={token_delta:+,}")
        print(f"  Latency:   A={comparison['avg_latency_a_ms']:.0f}ms  B={comparison['avg_latency_b_ms']:.0f}ms  Δ={latency_delta:+.0f}ms")
        if cases_improved:
            print(f"  Cases improved in B: {[c['case_id'] for c in cases_improved]}")
        if cases_degraded:
            print(f"  Cases degraded in B: {[c['case_id'] for c in cases_degraded]}")

    return comparison