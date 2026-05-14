import sys
import os
import time
import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "section_08_reliability"))

from langchain_core.messages import HumanMessage, SystemMessage

from trace_store import TraceStore
from eval_store import EvalStore
from golden_trajectories import GoldenTrajectoryStore
from scoring import (
    score_with_llm_judge,
    score_citations,
    score_completeness,
    score_trajectory,
)
from instrumentation import traced_node

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    run_date: datetime
    eval_label: str
    total_cases: int
    passed: int
    failed: int
    pass_rate: float

    avg_factual_accuracy: float
    avg_citation_validity: float
    avg_completeness: float
    avg_trajectory_efficiency: float
    avg_no_hallucination: float
    avg_total_score: float

    avg_duration_ms: float
    avg_tokens_per_run: int
    total_tokens_used: int

    failed_cases: list = field(default_factory=list)
    most_common_failure_mode: str = ""
    results: list = field(default_factory=list)

    def print_report(self):
        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print(f"Label:  {self.eval_label}")
        print(f"Date:   {self.run_date.strftime('%Y-%m-%d %H:%M UTC')}")
        print("=" * 70)

        status = "✓ PASSING" if self.pass_rate >= 0.75 else "✗ NEEDS WORK"
        print(f"\nOverall: {status}")
        print(f"Pass rate: {self.passed}/{self.total_cases} ({self.pass_rate:.0%})")
        print(f"Avg score: {self.avg_total_score:.3f}\n")

        print("Scores by dimension:")
        print(f"  Factual accuracy:      {self.avg_factual_accuracy:.3f}")
        print(f"  Citation validity:     {self.avg_citation_validity:.3f}")
        print(f"  Completeness:          {self.avg_completeness:.3f}")
        print(f"  Trajectory efficiency: {self.avg_trajectory_efficiency:.3f}")
        print(f"  No hallucination:      {self.avg_no_hallucination:.3f}")

        print(f"\nPerformance:")
        print(f"  Avg duration:  {self.avg_duration_ms:.0f}ms per run")
        print(f"  Avg tokens:    {self.avg_tokens_per_run:,} per run")
        print(f"  Total tokens:  {self.total_tokens_used:,}")

        if self.failed_cases:
            print(f"\nFailed cases ({len(self.failed_cases)}): {', '.join(self.failed_cases)}")
            print(f"Most common failure mode: {self.most_common_failure_mode}")

        print("=" * 70)


def _determine_failure_mode(results: list[dict]) -> str:
    failed = [r for r in results if not r.get("passed")]
    if not failed:
        return "none"

    mode_counts = {
        "factual_accuracy": sum(1 for r in failed if r.get("score_factual_accuracy", 1) < 0.5),
        "hallucination": sum(1 for r in failed if r.get("score_no_hallucination", 1) < 0.5),
        "citation": sum(1 for r in failed if r.get("score_citation_validity", 1) < 0.5),
        "completeness": sum(1 for r in failed if r.get("score_completeness", 1) < 0.5),
        "trajectory": sum(1 for r in failed if r.get("score_trajectory_efficiency", 1) < 0.5),
    }
    return max(mode_counts, key=mode_counts.get)


class EvaluationRunner:
    def __init__(
        self,
        trace_store: TraceStore,
        eval_store: EvalStore,
        golden_store: Optional[GoldenTrajectoryStore] = None,
        pass_threshold: float = 0.65,
    ):
        self.trace_store = trace_store
        self.eval_store = eval_store
        self.golden_store = golden_store
        self.pass_threshold = pass_threshold

    def run_evaluation(
        self,
        cases: list[dict],
        eval_label: str = "default",
        verbose: bool = True,
        max_iterations: int = 10,
    ) -> EvaluationReport:

        # Import the section 8 graph builder here so it uses the patched DOCS_DIR
        from graph import build_graph, make_initial_state

        results = []
        start = datetime.utcnow()

        if verbose:
            print(f"\nStarting evaluation '{eval_label}' — {len(cases)} cases")
            print("-" * 70)

        for i, case in enumerate(cases):
            if verbose:
                print(f"\n[{i+1}/{len(cases)}] {case['case_id']}: {case['description']}")
                print(f"  Query: {case['query'][:80]}")

            result = self._run_single_case(
                case, build_graph, make_initial_state, eval_label, max_iterations
            )
            results.append(result)
            self.eval_store.save_result(result, eval_run_label=eval_label)

            if verbose:
                status_sym = "✓" if result["passed"] else "✗"
                print(
                    f"  {status_sym} Score: {result['total_score']:.3f} | "
                    f"Tokens: {result['total_tokens']} | "
                    f"Time: {result['duration_ms']:.0f}ms | "
                    f"Status: {result['final_status']}"
                )
                if not result["passed"] and result.get("facts_missing"):
                    print(f"    Missing facts: {result['facts_missing'][:2]}")
                if result.get("forbidden_claims_found"):
                    print(f"    Hallucinations: {result['forbidden_claims_found']}")

        # Build report
        passed = [r for r in results if r["passed"]]
        n = len(results)

        def avg(key):
            vals = [r.get(key, 0) for r in results]
            return sum(vals) / len(vals) if vals else 0.0

        report = EvaluationReport(
            run_date=start,
            eval_label=eval_label,
            total_cases=n,
            passed=len(passed),
            failed=n - len(passed),
            pass_rate=len(passed) / n if n else 0,
            avg_factual_accuracy=avg("score_factual_accuracy"),
            avg_citation_validity=avg("score_citation_validity"),
            avg_completeness=avg("score_completeness"),
            avg_trajectory_efficiency=avg("score_trajectory_efficiency"),
            avg_no_hallucination=avg("score_no_hallucination"),
            avg_total_score=avg("total_score"),
            avg_duration_ms=avg("duration_ms"),
            avg_tokens_per_run=int(avg("total_tokens")),
            total_tokens_used=sum(r.get("total_tokens", 0) for r in results),
            failed_cases=[r["case_id"] for r in results if not r["passed"]],
            most_common_failure_mode=_determine_failure_mode(results),
            results=results,
        )

        return report

    def _run_single_case(
        self,
        case: dict,
        build_graph_fn,
        make_initial_state_fn,
        eval_label: str,
        max_iterations: int,
    ) -> dict:
        run_context = {
            "run_id": str(uuid.uuid4()),
            "thread_id": f"eval-{case['case_id']}",
            "node_index": 0,
            "trajectory": [],
            "total_tokens": 0,
        }

        t0 = time.perf_counter()

        try:
            graph = build_graph_fn()
            initial = make_initial_state_fn(
                case["query"], max_iterations=max_iterations
            )
            final_state = graph.invoke(initial)
        except Exception as e:
            logger.error(f"Graph invocation failed for {case['case_id']}: {e}")
            return self._zero_result(case, str(e), time.perf_counter() - t0)

        duration_ms = (time.perf_counter() - t0) * 1000
        answer = final_state.get("final_answer", "")
        validated = final_state.get("validated_answer") or {}
        citations = validated.get("citations", [])
        actual_trajectory = final_state.get("status", "")  # fallback
        # Use trajectory from messages if available
        trajectory_from_state = []  # section 8 graph doesn't expose trajectory in state

        # Score
        factual_score, hallucination_score, reasoning, facts_found, facts_missing = (
            score_with_llm_judge(
                case["query"],
                answer,
                case.get("expected_facts", []),
                case.get("forbidden_claims", []),
            )
        )
        citation_score, valid_cites, missing_cites = score_citations(
            citations, case.get("expected_citation_files", [])
        )
        completeness_score = score_completeness(
            answer, case.get("expected_facts", [])
        )
        trajectory_score = score_trajectory(
            trajectory_from_state, case.get("expected_trajectory")
        )

        w_fa = case.get("weight_factual_accuracy", 0.4)
        w_cv = case.get("weight_citation_validity", 0.2)
        w_co = case.get("weight_completeness", 0.2)
        w_te = case.get("weight_trajectory_efficiency", 0.1)
        w_nh = case.get("weight_no_hallucination", 0.1)

        total = (
            factual_score * w_fa
            + citation_score * w_cv
            + completeness_score * w_co
            + trajectory_score * w_te
            + hallucination_score * w_nh
        )

        # Record golden if golden store present and no golden exists yet
        if self.golden_store and not self.golden_store.get_golden(case["case_id"]):
            if final_state.get("status") == "complete":
                self.golden_store.record_golden(
                    case["case_id"],
                    run_context["run_id"],
                    trajectory_from_state,
                    notes=f"Auto-recorded from eval run '{eval_label}'",
                )

        return {
            "case_id": case["case_id"],
            "run_id": run_context["run_id"],
            "score_factual_accuracy": factual_score,
            "score_citation_validity": citation_score,
            "score_completeness": completeness_score,
            "score_trajectory_efficiency": trajectory_score,
            "score_no_hallucination": hallucination_score,
            "total_score": total,
            "passed": total >= self.pass_threshold,
            "facts_found": facts_found,
            "facts_missing": facts_missing,
            "forbidden_claims_found": [
                c for c in case.get("forbidden_claims", [])
                if c.lower() in answer.lower()
            ],
            "citations_valid": valid_cites,
            "citations_missing": missing_cites,
            "actual_trajectory": trajectory_from_state,
            "judge_reasoning": reasoning,
            "duration_ms": duration_ms,
            "total_tokens": run_context.get("total_tokens", 0),
            "final_status": final_state.get("status", "unknown"),
        }

    def _zero_result(self, case: dict, error: str, elapsed: float) -> dict:
        return {
            "case_id": case["case_id"],
            "run_id": str(uuid.uuid4()),
            "score_factual_accuracy": 0.0,
            "score_citation_validity": 0.0,
            "score_completeness": 0.0,
            "score_trajectory_efficiency": 0.0,
            "score_no_hallucination": 1.0,
            "total_score": 0.0,
            "passed": False,
            "facts_found": [],
            "facts_missing": case.get("expected_facts", []),
            "forbidden_claims_found": [],
            "citations_valid": [],
            "citations_missing": case.get("expected_citation_files", []),
            "actual_trajectory": [],
            "judge_reasoning": f"Agent crashed: {error}",
            "duration_ms": elapsed * 1000,
            "total_tokens": 0,
            "final_status": "crashed",
        }