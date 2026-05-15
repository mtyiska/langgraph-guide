import sys
sys.path.append("../..")

import random
import time
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class ParallelState(TypedDict):
    numbers: list[int]
    worker_results: Annotated[list[dict], add]
    overwrite_field: Annotated[list[str], add]  # changed from str to list with add
    aggregated_total: float


def fan_out(state: ParallelState) -> list[Send]:
    return [
        Send("worker", {**state, "current_number": n, "worker_id": i})
        for i, n in enumerate(state["numbers"])
    ]


class WorkerState(ParallelState):
    current_number: int
    worker_id: int


def worker_node(state: WorkerState) -> dict:
    n = state["current_number"]
    multiplier = random.uniform(1.0, 3.0)
    result = round(n * multiplier, 2)
    time.sleep(random.uniform(0.01, 0.05))

    return {
        "worker_results": [{"worker_id": state["worker_id"], "input": n, "result": result}],
        "overwrite_field": [f"written_by_worker_{state['worker_id']}"],  # now a list
    }


def aggregate_node(state: ParallelState) -> dict:
    total = sum(r["result"] for r in state["worker_results"])
    return {"aggregated_total": total}


builder = StateGraph(ParallelState)
builder.add_node("worker", worker_node)
builder.add_node("aggregate", aggregate_node)
builder.add_conditional_edges(START, fan_out, ["worker"])
builder.add_edge("worker", "aggregate")
builder.add_edge("aggregate", END)
graph = builder.compile()

print("=== Parallel Send — 5 runs ===\n")
for run in range(5):
    initial: ParallelState = {
        "numbers": [10, 20, 30],
        "worker_results": [],
        "overwrite_field": [],
        "aggregated_total": 0.0,
    }
    result = graph.invoke(initial)
    completion_order = [r["worker_id"] for r in result["worker_results"]]
    print(f"Run {run + 1}:")
    print(f"  Completion order: workers {completion_order}")
    print(f"  overwrite_field: '{result['overwrite_field']}' — all workers captured (older LangGraph silently dropped all but one)")
    print(f"  aggregated_total: {result['aggregated_total']:.2f}")
    print(f"  All results: {result['worker_results']}")
    print()

print("=== Proving the overwrite_field data loss ===")
print("In older LangGraph versions, overwrite_field with no reducer would silently")
print("keep only one worker's value. Newer versions raise InvalidUpdateError instead.")
print("Fix: use Annotated[list[str], add] to capture all values.\n")

# Fix demonstration
class FixedState(TypedDict):
    numbers: list[int]
    worker_results: Annotated[list[dict], add]
    all_writes: Annotated[list[str], add]   # fixed with add reducer
    aggregated_total: float


class FixedWorkerState(FixedState):
    current_number: int
    worker_id: int


def fixed_worker_node(state: FixedWorkerState) -> dict:
    multiplier = random.uniform(1.0, 3.0)
    result = round(state["current_number"] * multiplier, 2)
    return {
        "worker_results": [{"worker_id": state["worker_id"], "result": result}],
        "all_writes": [f"written_by_worker_{state['worker_id']}"],
    }


def fixed_fan_out(state: FixedState) -> list[Send]:
    return [Send("worker", {**state, "current_number": n, "worker_id": i})
            for i, n in enumerate(state["numbers"])]


def fixed_aggregate(state: FixedState) -> dict:
    total = sum(r["result"] for r in state["worker_results"])
    return {"aggregated_total": total}


fixed_builder = StateGraph(FixedState)
fixed_builder.add_node("worker", fixed_worker_node)
fixed_builder.add_node("aggregate", fixed_aggregate)
fixed_builder.add_conditional_edges(START, fixed_fan_out, ["worker"])
fixed_builder.add_edge("worker", "aggregate")
fixed_builder.add_edge("aggregate", END)
fixed_graph = fixed_builder.compile()

fixed_initial: FixedState = {
    "numbers": [10, 20, 30],
    "worker_results": [],
    "all_writes": [],
    "aggregated_total": 0.0,
}
fixed_result = fixed_graph.invoke(fixed_initial)
print("Fixed version — all_writes captures every worker:")
print(f"  all_writes: {fixed_result['all_writes']}")
print(f"  Count: {len(fixed_result['all_writes'])} — all 3 workers captured")