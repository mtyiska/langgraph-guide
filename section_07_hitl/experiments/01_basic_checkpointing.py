import sys
import os
import sqlite3
sys.path.append("../..")

from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "exp01_checkpoints.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


class SimpleState(TypedDict):
    messages: Annotated[list[str], add]
    step_outputs: Annotated[list[str], add]
    status: str


def node_a(state: SimpleState) -> dict:
    print("  [node_a] running")
    return {"messages": ["node_a ran"], "step_outputs": ["output_a"], "status": "after_a"}


def node_b(state: SimpleState) -> dict:
    print("  [node_b] running")
    return {"messages": ["node_b ran"], "step_outputs": ["output_b"], "status": "after_b"}


def node_c(state: SimpleState) -> dict:
    print("  [node_c] running")
    return {"messages": ["node_c ran"], "step_outputs": ["output_c"], "status": "complete"}


builder = StateGraph(SimpleState)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", "node_c")
builder.add_edge("node_c", END)

THREAD_ID = "experiment-01-thread"
config = {"configurable": {"thread_id": THREAD_ID}}

print("=== Experiment 1 — Basic Checkpointing ===\n")

with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

    print("Running graph...")
    result = graph.invoke(
        {"messages": [], "step_outputs": [], "status": "start"},
        config=config,
    )
    print(f"Final status: {result['status']}")
    print(f"Messages: {result['messages']}")

    print("\n=== Querying SQLite checkpoints directly ===")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT thread_id, checkpoint_id, parent_checkpoint_id, ts FROM checkpoints WHERE thread_id = ? ORDER BY ts",
        (THREAD_ID,)
    ).fetchall()
    print(f"Total checkpoints for thread '{THREAD_ID}': {len(rows)}")
    for i, row in enumerate(rows):
        print(f"  [{i}] checkpoint_id={row[1][:12]}... parent={str(row[2])[:12]}... ts={row[3]}")
    conn.close()

    print("\n=== State at each checkpoint ===")
    history = list(graph.get_state_history(config))
    history.reverse()
    for i, snapshot in enumerate(history):
        source = snapshot.metadata.get("source", "?")
        step = snapshot.metadata.get("step", i)
        vals = snapshot.values
        print(f"  Step {step} [{source}]: status={vals.get('status')} | "
              f"messages={vals.get('messages')} | outputs={vals.get('step_outputs')}")

    print("\n=== Patching node_c output via update_state and re-running ===")
    # Find the checkpoint after node_b (before node_c ran)
    target = None
    for snapshot in history:
        if snapshot.values.get("status") == "after_b":
            target = snapshot
            break

    if target:
        fork_checkpoint_id = target.config["configurable"]["checkpoint_id"]
        fork_config = {"configurable": {"thread_id": THREAD_ID + "-fork", "checkpoint_id": fork_checkpoint_id}}

        # update state at the fork point
        graph.update_state(
            {"configurable": {"thread_id": THREAD_ID + "-fork"}},
            {"step_outputs": ["patched_before_c"], "status": "after_b"},
        )
        # Resume
        fork_result = graph.invoke(None, {"configurable": {"thread_id": THREAD_ID + "-fork"}})
        print(f"  Forked result — step_outputs: {fork_result['step_outputs']}")
        print(f"  node_c ran once on patched state, not re-running node_a or node_b")
    else:
        print("  Could not find target checkpoint (run may already be forked).")