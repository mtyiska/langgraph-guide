import sqlite3
import os
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DB = os.path.join(DATA_DIR, "checkpoints.db")


def list_sessions(graph=None) -> list[dict]:
    """
    Query the checkpoints database and return a list of sessions
    with their last-active timestamp and (if graph is provided) completion status.
    """
    if not os.path.exists(CHECKPOINT_DB):
        return []

    conn = sqlite3.connect(CHECKPOINT_DB)
    try:
        cursor = conn.execute("""
            SELECT thread_id, MAX(ts) as last_active, COUNT(*) as checkpoint_count
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY last_active DESC
        """)
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return []
    finally:
        conn.close()

    sessions = []
    for thread_id, last_active, count in rows:
        entry = {
            "thread_id": thread_id,
            "last_active": last_active,
            "checkpoint_count": count,
            "status": "unknown",
        }
        if graph:
            try:
                config = {"configurable": {"thread_id": thread_id}}
                state = graph.get_state(config)
                if state.next:
                    entry["status"] = f"paused (next: {', '.join(state.next)})"
                else:
                    entry["status"] = state.values.get("status", "complete")
            except Exception:
                pass
        sessions.append(entry)

    return sessions


def print_sessions(graph=None):
    sessions = list_sessions(graph)
    if not sessions:
        print("No saved sessions found.")
        return

    print(f"\n{'Thread ID':<44} {'Last Active':<22} {'Checkpoints':<12} Status")
    print("-" * 100)
    for s in sessions:
        print(
            f"{s['thread_id']:<44} "
            f"{s['last_active']:<22} "
            f"{s['checkpoint_count']:<12} "
            f"{s['status']}"
        )


def print_state_history(graph, config: dict):
    """
    Print a human-readable history of all checkpoints for a thread,
    showing what each node changed.
    """
    history = list(graph.get_state_history(config))
    if not history:
        print("No history found for this thread.")
        return

    history.reverse()  # oldest first

    print(f"\nCheckpoint history for thread: {config['configurable']['thread_id']}")
    print(f"Total checkpoints: {len(history)}\n")

    prev_values = {}
    for i, snapshot in enumerate(history):
        source = snapshot.metadata.get("source", "unknown")
        step = snapshot.metadata.get("step", i)
        ts = getattr(snapshot, "created_at", "")

        print(f"Step {step:>3} | Node: {source:<30} | {ts}")

        # Show what changed
        current_values = snapshot.values
        changes = []
        for key, val in current_values.items():
            prev = prev_values.get(key)
            if prev != val:
                if isinstance(val, list) and isinstance(prev, list):
                    added = len(val) - len(prev)
                    if added > 0:
                        changes.append(f"  + {key}: +{added} items (total {len(val)})")
                elif isinstance(val, str) and len(val) > 60:
                    changes.append(f"  ~ {key}: (string, {len(val)} chars)")
                elif val != prev:
                    changes.append(f"  ~ {key}: {repr(prev)!s:.40} → {repr(val)!s:.40}")
        for c in changes:
            print(c)

        prev_values = dict(current_values)
        print()


def state_diff(previous: dict, current: dict) -> list[str]:
    """Return a list of human-readable differences between two state dicts."""
    diffs = []
    all_keys = set(previous) | set(current)
    for key in sorted(all_keys):
        p = previous.get(key)
        c = current.get(key)
        if p == c:
            continue
        if isinstance(c, list) and isinstance(p, list):
            delta = len(c) - len(p)
            diffs.append(f"{key}: list grew by {delta} (now {len(c)} items)")
        else:
            diffs.append(f"{key}: {repr(p)!s:.50} → {repr(c)!s:.50}")
    return diffs


def get_last_good_checkpoint(graph, config: dict, failed_status: str = "failed"):
    """
    Walk the checkpoint history backwards and return the StateSnapshot
    just before the status became `failed_status`.
    Useful for time-travel debugging.
    """
    history = list(graph.get_state_history(config))
    for snapshot in history:
        if snapshot.values.get("status") != failed_status:
            return snapshot
    return None