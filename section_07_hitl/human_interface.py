import os
import sys
from typing import Optional

from state import ProposedChange, ReviewDecision
from tools import get_file_lines

SEVERITY_COLOUR = {
    "critical": "\033[91m",  # red
    "high": "\033[93m",      # yellow
    "medium": "\033[94m",    # blue
    "low": "\033[92m",       # green
}
RESET = "\033[0m"
BOLD = "\033[1m"


def _colour(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def print_banner(title: str):
    width = 64
    print("\n" + "=" * width)
    print(f"  {BOLD}{title}{RESET}")
    print("=" * width)


def present_change_for_review(graph, config: dict) -> Optional[ReviewDecision]:
    """
    Read the current pending change from graph state, display it to the
    human, collect their decision, and return a ReviewDecision.

    Returns None if the human quits the session.
    """
    state_snapshot = graph.get_state(config)
    values = state_snapshot.values

    change: ProposedChange = values.get("pending_change")
    diff: str = values.get("formatted_diff", "")
    total = len(values.get("proposed_changes", []))
    idx = values.get("current_change_index", 0)

    if not change:
        return None

    severity = change["severity"]
    colour = SEVERITY_COLOUR.get(severity, "")

    print_banner(f"PROPOSED CHANGE  [{idx + 1} of {total}]")
    print(
        f"  File:     {change['file_path']}\n"
        f"  Lines:    {change['line_start']}–{change['line_end']}\n"
        f"  Type:     {change['issue_type'].upper()}\n"
        f"  Severity: {_colour(severity.upper(), colour)}\n"
        f"  Reason:   {change['reason']}\n"
    )

    if diff and diff.strip() != "(no textual difference)":
        print("--- DIFF " + "-" * 55)
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                print(f"\033[92m{line}{RESET}")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"\033[91m{line}{RESET}")
            elif line.startswith("@@"):
                print(f"\033[96m{line}{RESET}")
            else:
                print(line)
        print("-" * 64)
    else:
        print("  (Diff unavailable — showing proposed code directly)")
        print(change["proposed_code"])
        print("-" * 64)

    while True:
        print(
            "\n  [a] Approve    [r] Reject    [e] Edit before applying\n"
            "  [v] View full file context    [q] Save & quit session\n"
        )
        choice = input("  Choice: ").strip().lower()

        if choice == "a":
            return ReviewDecision(
                change=change,
                decision="approved",
                edited_code=None,
                reviewer_note=None,
            )

        elif choice == "r":
            note = input("  Rejection note (optional, Enter to skip): ").strip()
            return ReviewDecision(
                change=change,
                decision="rejected",
                edited_code=None,
                reviewer_note=note or None,
            )

        elif choice == "e":
            print("\n  Current proposed code:")
            print("  " + "\n  ".join(change["proposed_code"].splitlines()))
            print("\n  Enter your replacement code.")
            print("  Type END on a line by itself when done:\n")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            edited = "\n".join(lines)
            confirm = input(f"\n  Apply this edited version? [y/n]: ").strip().lower()
            if confirm == "y":
                return ReviewDecision(
                    change=change,
                    decision="edited",
                    edited_code=edited,
                    reviewer_note=None,
                )
            else:
                print("  Edit cancelled — returning to options.")

        elif choice == "v":
            try:
                excerpt = get_file_lines(
                    change["file_path"],
                    change["line_start"],
                    change["line_end"],
                    context=5,
                )
                print(f"\n--- {change['file_path']} ---")
                print(excerpt)
            except Exception as e:
                print(f"  Could not read file: {e}")

        elif choice == "q":
            print(
                "\n  Session saved. Resume with:\n"
                f"  python code_review_agent.py --session <session_id> <files...>"
            )
            raise SystemExit(0)

        else:
            print("  Unrecognised choice — enter a, r, e, v, or q.")