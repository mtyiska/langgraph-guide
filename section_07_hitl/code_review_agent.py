import sys
import os
import uuid
import argparse
sys.path.append("..")

from langgraph.checkpoint.sqlite import SqliteSaver

from graph import build_graph, get_checkpointer, CHECKPOINT_DB
from human_interface import present_change_for_review
from checkpoint_utils import print_sessions, print_state_history
from state import ReviewDecision

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def run_review_loop(graph, config: dict, initial_input):
    try:
        graph.invoke(initial_input, config)
    except Exception as e:
        print(f"\n[ERROR] Graph raised an exception: {e}")
        print("State has been saved. Resume with --session flag.")
        return

    while True:
        state = graph.get_state(config)

        if not state.next:
            _print_final_summary(state.values)
            return

        # ✅ Check for human_review, not apply_or_skip
        if "human_review" in state.next:
            try:
                decision: ReviewDecision = present_change_for_review(graph, config)
            except SystemExit:
                return

            if decision is None:
                graph.invoke(None, config)
                continue

            graph.update_state(
                config,
                {"review_decisions": [decision]},
                as_node="human_review",
            )

            try:
                graph.invoke(None, config)
            except Exception as e:
                print(f"\n[ERROR] {e}")
                print("State saved. Resume with --session flag.")
                return

        else:
            print(f"\n[INFO] Paused before: {state.next}. Resuming...")
            graph.invoke(None, config)

def _print_final_summary(values: dict):
    total = len(values.get("proposed_changes", []))
    decisions = values.get("review_decisions", [])
    approved = sum(1 for d in decisions if d["decision"] == "approved")
    edited = sum(1 for d in decisions if d["decision"] == "edited")
    rejected = sum(1 for d in decisions if d["decision"] == "rejected")
    modified = list(set(values.get("changes_applied", [])))

    print("\n" + "=" * 64)
    print("  REVIEW SESSION COMPLETE")
    print("=" * 64)
    print(f"  Files reviewed  : {len(values.get('files_read', []))}")
    print(f"  Issues proposed : {total}")
    print(f"  Approved        : {approved}")
    print(f"  Edited & applied: {edited}")
    print(f"  Rejected        : {rejected}")
    print(f"  Files modified  : {len(modified)}")
    if modified:
        for f in modified:
            print(f"    - {f}  (backup: {f}.bak)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Agentic code reviewer with human-in-the-loop approval."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Python files to review.",
    )
    parser.add_argument(
        "--session",
        help="Resume an existing session by its ID.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List all saved sessions and exit.",
    )
    parser.add_argument(
        "--history",
        metavar="SESSION_ID",
        help="Print checkpoint history for a session and exit.",
    )
    args = parser.parse_args()

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_graph(checkpointer)

        if args.list_sessions:
            print_sessions(graph)
            return

        if args.history:
            config = {"configurable": {"thread_id": args.history}}
            print_state_history(graph, config)
            return

        if args.session:
            session_id = args.session
            config = {"configurable": {"thread_id": session_id}}
            state = graph.get_state(config)

            if not state.values:
                print(f"Session '{session_id}' not found. Starting fresh.")
                if not args.files:
                    parser.error("Provide at least one file to review.")
                initial_input = _make_initial_state(args.files, session_id)
            elif not state.next:
                print(f"Session '{session_id}' is already complete.")
                _print_final_summary(state.values)
                return
            else:
                print(f"\nResuming session: {session_id}")
                print(f"Paused before: {state.next}")
                initial_input = None

        else:
            if not args.files:
                parser.error("Provide at least one file to review (or --session to resume).")
            session_id = str(uuid.uuid4())[:8]
            config = {"configurable": {"thread_id": session_id}}
            initial_input = _make_initial_state(args.files, session_id)
            print(f"\nNew session ID: {session_id}")
            print(f"Resume later with: python code_review_agent.py --session {session_id} <files>")

        run_review_loop(graph, config, initial_input)


def _make_initial_state(files: list[str], session_id: str) -> dict:
    return {
        "files_to_review": files,
        "files_read": [],
        "issues_found": [],
        "proposed_changes": [],
        "pending_change": None,
        "formatted_diff": None,
        "review_decisions": [],
        "changes_applied": [],
        "current_file_index": 0,
        "current_change_index": 0,
        "status": "analysing",
        "error": None,
        "session_id": session_id,
    }


if __name__ == "__main__":
    main()