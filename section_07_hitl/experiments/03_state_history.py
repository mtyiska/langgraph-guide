import sys
import os
sys.path.append("../..")

from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "exp03_checkpoints.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


class RichState(TypedDict):
    counter: int
    tags: Annotated[list[str], add]
    last_node: str
    accumulated_value: float
    status: str


def make_node(name: str, tag: str, delta: float):
    def node_fn(state: RichState) -> dict:
        return {
            "counter": state["counter"] + 1,
            "tags": [tag],
            "last_node": name,
            "accumulated_value": round(state["accumulated_value"] + delta, 4),
        }
    node_fn.__name__ = name
    return node_fn


nodes = [
    make_node("alpha", "α", 1.5),
    make_node("beta",  "β", 2.3),
    make_node("gamma", "γ", 0.7),
    make_node("delta", "δ", 3.1),
    make_node("epsilon","ε", 1.1),
    make_node("zeta",  "ζ", 4.0),
]

builder = StateGraph(RichState)
prev = START
for n in nodes:
    builder.add_node(n.__name__, n)
    builder.add_edge(prev, n.__name__)
    prev = n.__name__
builder.add_edge(prev, END)

THREAD_ID = "exp03-main"
config = {"configurable": {"thread_id": THREAD_ID}}

print("=== Experiment 3 — State History ===\n")


def state_diff(prev_vals: dict, curr_vals: dict) -> list[str]:
    diffs = []
    for key in sorted(set(prev_vals) | set(curr_vals)):
        p, c = prev_vals.get(key), curr_vals.get(key)
        if p == c:
            continue
        if isinstance(c, list) and isinstance(p, list):
            added = [x for x in c if x not in p]
            diffs.append(f"  {key}: +{added}")
        else:
            diffs.append(f"  {key}: {p!r} → {c!r}")
    return diffs


with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

    initial = {
        "counter": 0,
        "tags": [],
        "last_node": "START",
        "accumulated_value": 0.0,
        "status": "running",
    }
    result = graph.invoke(initial, config=config)
    print(f"Final: counter={result['counter']}, value={result['accumulated_value']}, tags={result['tags']}\n")

    print("=== Full checkpoint history with diffs ===\n")
    history = list(graph.get_state_history(config))
    history.reverse()

    prev_vals: dict = {}
    for i, snapshot in enumerate(history):
        source = snapshot.metadata.get("source", "?")
        step = snapshot.metadata.get("step", i)
        curr_vals = snapshot.values
        diffs = state_diff(prev_vals, curr_vals)

        print(f"Checkpoint {i} | Step {step} | Source: {source}")
        if diffs:
            for d in diffs:
                print(d)
        else:
            print("  (no changes)")
        prev_vals = dict(curr_vals)
        print()

    print("=== Second run on same thread — checkpoints append ===")
    result2 = graph.invoke(initial, config=config)
    all_history = list(graph.get_state_history(config))
    print(f"Total checkpoints now: {len(all_history)}")
    print(f"(First run: {len(history)}, after second run: {len(all_history)})")
    print("Both runs are preserved — checkpoints accumulate per thread.")