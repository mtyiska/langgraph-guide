import sys
sys.path.append("../..")

from typing import TypedDict, Annotated, Optional
from operator import add
from langgraph.graph import StateGraph, START, END


# ── Shared state between parent and subgraph ──────────────────────────────────

class SharedKeys(TypedDict):
    items: Annotated[list[str], add]
    status: str
    result: Optional[str]


# ── Subgraph ──────────────────────────────────────────────────────────────────

def subgraph_node_a(state: SharedKeys) -> dict:
    print(f"  [subgraph] node_a running — items so far: {state['items']}")
    return {"items": ["from_subgraph_node_a"], "status": "subgraph_running"}


def subgraph_node_b(state: SharedKeys) -> dict:
    print(f"  [subgraph] node_b running — items so far: {state['items']}")
    return {"items": ["from_subgraph_node_b"], "result": "subgraph complete"}


sub_builder = StateGraph(SharedKeys)
sub_builder.add_node("sub_a", subgraph_node_a)
sub_builder.add_node("sub_b", subgraph_node_b)
sub_builder.add_edge(START, "sub_a")
sub_builder.add_edge("sub_a", "sub_b")
sub_builder.add_edge("sub_b", END)
subgraph = sub_builder.compile()

# ── Parent graph ──────────────────────────────────────────────────────────────

def parent_node_before(state: SharedKeys) -> dict:
    print(f"[parent] before_node running")
    return {"items": ["from_parent_before"], "status": "parent_running"}


def parent_node_after(state: SharedKeys) -> dict:
    print(f"[parent] after_node running — result from subgraph: {state.get('result')}")
    return {"items": ["from_parent_after"], "status": "complete"}


parent_builder = StateGraph(SharedKeys)
parent_builder.add_node("before", parent_node_before)
parent_builder.add_node("subgraph", subgraph)
parent_builder.add_node("after", parent_node_after)
parent_builder.add_edge(START, "before")
parent_builder.add_edge("before", "subgraph")
parent_builder.add_edge("subgraph", "after")
parent_builder.add_edge("after", END)
parent_graph = parent_builder.compile()

print("=== Graph Structure ===")
parent_graph.get_graph(xray=True).print_ascii()

print("\n=== Execution with stream_mode='updates' ===")
initial = {"items": [], "status": "pending", "result": None}
final_result = None
for step in parent_graph.stream(initial, stream_mode="updates"):
    node_name = list(step.keys())[0]
    updates = step[node_name]
    print(f"  [{node_name}] changed: {updates}")

# invoke separately with a fresh initial state — don't reuse after stream
result = parent_graph.invoke({"items": [], "status": "pending", "result": None})
print(f"\nFinal items: {result['items']}")
print(f"Final status: {result['status']}")
print(f"Final result: {result['result']}")

# ── Break the interface — mismatched key name ─────────────────────────────────

print("\n=== Deliberate interface break: mismatched key ===")


class BrokenSubState(TypedDict):
    items: Annotated[list[str], add]
    status: str
    output: Optional[str]  # 'output' instead of 'result'


def broken_sub_node(state: BrokenSubState) -> dict:
    return {"items": ["from_broken_sub"], "output": "this key won't reach parent"}


broken_sub_builder = StateGraph(BrokenSubState)
broken_sub_builder.add_node("broken", broken_sub_node)
broken_sub_builder.add_edge(START, "broken")
broken_sub_builder.add_edge("broken", END)
broken_subgraph = broken_sub_builder.compile()


class ParentWithBrokenSub(TypedDict):
    items: Annotated[list[str], add]
    status: str
    result: Optional[str]  # parent expects 'result', subgraph writes 'output'


broken_parent_builder = StateGraph(ParentWithBrokenSub)
broken_parent_builder.add_node("subgraph", broken_subgraph)
broken_parent_builder.add_edge(START, "subgraph")
broken_parent_builder.add_edge("subgraph", END)
broken_parent = broken_parent_builder.compile()

broken_result = broken_parent.invoke({"items": [], "status": "pending", "result": None})
print(f"  result field in parent: {broken_result.get('result')}")
print(f"  items: {broken_result.get('items')}")
print("  → 'output' key from subgraph silently lost — parent never sees it")
print("  → Shared key names between parent and subgraph are a CONTRACT")