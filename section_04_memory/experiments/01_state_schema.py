

from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END


# Schema A — all overwrite
class SchemaA(TypedDict):
    items: list

def node_a1(state: SchemaA) -> dict:
    return {"items": ["from_node_1"]}

def node_a2(state: SchemaA) -> dict:
    return {"items": ["from_node_2"]}

builder_a = StateGraph(SchemaA)
builder_a.add_node("n1", node_a1)
builder_a.add_node("n2", node_a2)
builder_a.add_edge(START, "n1")
builder_a.add_edge("n1", "n2")
builder_a.add_edge("n2", END)
graph_a = builder_a.compile()
result_a = graph_a.invoke({"items": []})
print("=== Schema A (overwrite) ===")
print(f"items: {result_a['items']} — only last writer survives")


# Schema B — all add
class SchemaB(TypedDict):
    items: Annotated[list, add]

def node_b1(state: SchemaB) -> dict:
    return {"items": ["from_node_1"]}

def node_b2(state: SchemaB) -> dict:
    return {"items": ["from_node_2"]}

builder_b = StateGraph(SchemaB)
builder_b.add_node("n1", node_b1)
builder_b.add_node("n2", node_b2)
builder_b.add_edge(START, "n1")
builder_b.add_edge("n1", "n2")
builder_b.add_edge("n2", END)
graph_b = builder_b.compile()
result_b = graph_b.invoke({"items": []})
print("\n=== Schema B (add reducer) ===")
print(f"items: {result_b['items']} — both writers' data survives")


# Schema C — custom keep_latest_n reducer
def keep_latest_n(n: int):
    def reducer(existing: list, new: list) -> list:
        combined = existing + new
        return combined[-n:]
    return reducer

class SchemaC(TypedDict):
    items: Annotated[list, keep_latest_n(3)]

def make_node(label: str):
    def node(state: SchemaC) -> dict:
        return {"items": [label]}
    return node

builder_c = StateGraph(SchemaC)
for i in range(1, 9):
    builder_c.add_node(f"n{i}", make_node(f"item_{i}"))

builder_c.add_edge(START, "n1")
for i in range(1, 8):
    builder_c.add_edge(f"n{i}", f"n{i+1}")
builder_c.add_edge("n8", END)

graph_c = builder_c.compile()
result_c = graph_c.invoke({"items": []})
print("\n=== Schema C (keep_latest_n(3)) ===")
print(f"items after 8 appends: {result_c['items']} — only last 3 kept")