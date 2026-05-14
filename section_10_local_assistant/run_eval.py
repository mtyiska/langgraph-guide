"""
Evaluation suite entry point.
Usage:
    python run_eval.py
    python run_eval.py --cases 5
    python run_eval.py --label my_run --threshold 0.70
    python run_eval.py --setup-kb
"""

import sys
import os
import argparse
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from eval.cases import EVAL_CASES, create_eval_kb, EVAL_KB_DIR
from eval.runner import EvaluationRunner
from config import EVAL_PASS_THRESHOLD

# Point document tools at the eval knowledge base
import agent.tools.document_tools as dtm
from infrastructure.vector_store import VectorStoreWrapper

_eval_vs = None


def get_eval_vector_store():
    global _eval_vs
    if _eval_vs is None:
        _eval_vs = VectorStoreWrapper(collection_name="eval_knowledge_base")
        _eval_vs.index_directory(EVAL_KB_DIR)
    return _eval_vs


def main():
    parser = argparse.ArgumentParser(description="Capstone evaluation suite")
    parser.add_argument("--cases",     type=int, default=20)
    parser.add_argument("--label",     default="capstone_eval")
    parser.add_argument("--threshold", type=float, default=EVAL_PASS_THRESHOLD)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--setup-kb",  action="store_true")
    parser.add_argument("--verbose",   action="store_true", default=True)
    args = parser.parse_args()

    if args.setup_kb:
        create_eval_kb()
        return

    create_eval_kb()

    # Redirect document tool searches to eval KB
    original_get = dtm.get_vector_store
    dtm.get_vector_store = get_eval_vector_store

    try:
        runner = EvaluationRunner(pass_threshold=args.threshold)
        cases  = EVAL_CASES[:args.cases]
        report = runner.run(cases, label=args.label, verbose=args.verbose, max_iters=args.max_iters)
        report.print_report()
    finally:
        dtm.get_vector_store = original_get


if __name__ == "__main__":
    main()