"""
Section 9 mini-project — eval harness entry point.
Run: python eval_harness.py [--cases N] [--label my_run] [--viewer]
"""

import sys
import os
import argparse
import logging

sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "section_08_reliability"))

from eval_cases import EVAL_CASES, create_knowledge_base, KB_DIR
from trace_store import TraceStore
from eval_store import EvalStore
from golden_trajectories import GoldenTrajectoryStore
from eval_runner import EvaluationRunner

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Point section 8 tools at our knowledge base
import tools as tools_mod
tools_mod.DOCS_DIR = KB_DIR


def main():
    parser = argparse.ArgumentParser(description="Section 9 Evaluation Harness")
    parser.add_argument("--cases", type=int, default=20, help="Number of cases to run (1-20)")
    parser.add_argument("--label", default="eval_run", help="Label for this evaluation run")
    parser.add_argument("--threshold", type=float, default=0.65, help="Pass threshold (0-1)")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--viewer", action="store_true", help="Launch trace viewer after run")
    parser.add_argument("--setup-kb", action="store_true", help="Create knowledge base docs and exit")
    args = parser.parse_args()

    if args.setup_kb:
        create_knowledge_base()
        return

    # Ensure knowledge base exists
    create_knowledge_base()

    trace_store = TraceStore("./data/traces.db")
    eval_store = EvalStore("./data/eval_results.db")
    golden_store = GoldenTrajectoryStore("./data/eval_results.db")

    runner = EvaluationRunner(
        trace_store=trace_store,
        eval_store=eval_store,
        golden_store=golden_store,
        pass_threshold=args.threshold,
    )

    cases = EVAL_CASES[: args.cases]
    report = runner.run_evaluation(
        cases,
        eval_label=args.label,
        verbose=True,
        max_iterations=args.max_iterations,
    )
    report.print_report()

    if args.viewer:
        import subprocess
        print("\nLaunching trace viewer at http://localhost:8001 ...")
        subprocess.Popen(
            ["python", "-m", "uvicorn", "trace_viewer:app", "--port", "8001"],
            cwd=os.path.dirname(__file__),
        )


if __name__ == "__main__":
    main()