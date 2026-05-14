import sys
import os
sys.path.append("../..")

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "exp04_checkpoints.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


class FragileState(TypedDict):
    value: int
    history: list[str]
    trigger_failure_at: int   # node index that will fail
    status: str
    error: Optional[str]


def make_fragile_node(index: int, name: str):
    def node_fn(state: FragileState) -> dict:
        if state["trigger_failure_at"] == index:
            raise RuntimeError(f"Simulated failure at node '{name}' (index {index})")
        new_val = state["value"] * (index + 2)
        return {
            "value": new_val,
            "history": state["history"] + [f"{name}:{new_val}"],
            "status": "running",
        }
    node_fn.__name__ = name
    return node_fn


node_names = ["setup", "transform", "enrich", "finalise"]
nodes = [make_fragile_node(i, name) for i, name in enumerate(node_names)]

builder = StateGraph(FragileState)
prev = START
for n in nodes:
    builder.add_node(n.__name__, n)
    builder.add_edge(prev, n.__name__)
    prev = n.__name__
builder.add_edge(prev, END)

THREAD_ID = "exp04-fragile"
config = {"configurable": {"thread_id": THREAD_ID}}

print("=== Experiment 4 — Time-Travel Debugging ===\n")

with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

    # Run with failure at node index 2 ("enrich")
    initial = {
        "value": 3,
        "history": [],
        "trigger_failure_at": 2,
        "status": "running",
        "error": None,
    }

    print("1. Running graph with failure at 'enrich' node...")
    failed_thread = THREAD_ID + "-failed"
    failed_config = {"configurable": {"thread_id": failed_thread}}
    try:
        graph.invoke(initial, config=failed_config)
    except RuntimeError as e:
        print(f"   Exception caught: {e}")

    # Retrieve history
    history = list(graph.get_state_history(failed_config))
    history.reverse()

    print(f"\n2. Checkpoint history ({len(history)} checkpoints):")
    for i, snap in enumerate(history):
        source = snap.metadata.get("source", "?")
        vals = snap.values
        print(f"   [{i}] source={source:<12} value={vals.get('value'):<6} "
              f"status={vals.get('status')} history={vals.get('history')}")

    # Find last good checkpoint (before the failure)
    last_good = None
    for snap in reversed(history):
        if snap.values.get("status") == "running" and snap.values.get("error") is None:
            if "enrich" not in snap.metadata.get("source", ""):
                last_good = snap
                break

    if not last_good:
        # Fall back to the checkpoint just before the failed node
        last_good = history[-2] if len(history) >= 2 else history[-1]

    last_good_checkpoint_id = last_good.config["configurable"]["checkpoint_id"]
    print(f"\n3. Last good checkpoint: source={last_good.metadata.get('source')} "
          f"value={last_good.values.get('value')}")

    # Fork from that checkpoint into a new thread, fixing the failure condition
    fork_thread = THREAD_ID + "-fork"
    fork_config = {
        "configurable": {
            "thread_id": fork_thread,
            "checkpoint_id": last_good_checkpoint_id,
        }
    }

    # Patch the state so trigger_failure_at no longer hits
    graph.update_state(
        {"configurable": {"thread_id": fork_thread}},
        {
            **last_good.values,
            "trigger_failure_at": 99,  # never trigger
            "status": "running",
            "error": None,
        },
    )

    print(f"\n4. Forking from checkpoint, disabling failure condition...")
    fork_result = graph.invoke(None, {"configurable": {"thread_id": fork_thread}})

    print(f"\n5. Fork completed successfully!")
    print(f"   Final value : {fork_result['value']}")
    print(f"   History     : {fork_result['history']}")
    print(f"   Status      : {fork_result['status']}")

    print(f"""
=== Debugging Report ===
Original thread  : {failed_thread}
  Failure point  : node 'enrich' (index 2)
  Last good state: value={last_good.values['value']}, history={last_good.values['history']}

Fork thread      : {fork_thread}
  Fix applied    : trigger_failure_at set to 99 (never fires)
  Resumed from   : checkpoint_id={last_good_checkpoint_id[:16]}...
  Final result   : value={fork_result['value']}, status={fork_result['status']}

Nodes re-run after fork: {[n for n in fork_result['history'] if n not in last_good.values.get('history', [])]}
Nodes NOT re-run (saved by checkpointing): {last_good.values.get('history', [])}
""")