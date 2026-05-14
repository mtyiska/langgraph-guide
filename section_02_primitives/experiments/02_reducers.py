import sys
sys.path.append("../..")

from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END

# With add reducer on append_list
class ReducerState(TypedDict):
    overwrite_list: list          # default: overwrites every time
    append_list: Annotated[list, add]  # appends every time

def node_a(state: ReducerState) -> dict:
    return {"overwrite_list": ["from_a"], "append_list": ["from_a"]}

def node_b(state: ReducerState) -> dict:
    return {"overwrite_list": ["from_b"], "append_list": ["from_b"]}

def node_c(state: ReducerState) -> dict:
    return {"overwrite_list": ["from_c"], "append_list": ["from_c"]}

builder = StateGraph(ReducerState)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", "c")
builder.add_edge("c", END)

graph = builder.compile()
result = graph.invoke({"overwrite_list": [], "append_list": []})

print("=== Reducer Comparison ===")
print(f"overwrite_list (default reducer): {result['overwrite_list']}")
print(f"  → Only last node's value survives")
print(f"append_list    (add reducer):     {result['append_list']}")
print(f"  → All three nodes' values accumulate")

print("\n=== Now breaking it — removing Annotated from append_list ===")

class BrokenState(TypedDict):
    overwrite_list: list
    append_list: list   # no Annotated — will silently overwrite

def node_a2(state: BrokenState) -> dict:
    return {"overwrite_list": ["from_a"], "append_list": ["from_a"]}

def node_b2(state: BrokenState) -> dict:
    return {"overwrite_list": ["from_b"], "append_list": ["from_b"]}

def node_c2(state: BrokenState) -> dict:
    return {"overwrite_list": ["from_c"], "append_list": ["from_c"]}

builder2 = StateGraph(BrokenState)
builder2.add_node("a", node_a2)
builder2.add_node("b", node_b2)
builder2.add_node("c", node_c2)
builder2.add_edge(START, "a")
builder2.add_edge("a", "b")
builder2.add_edge("b", "c")
builder2.add_edge("c", END)

graph2 = builder2.compile()
result2 = graph2.invoke({"overwrite_list": [], "append_list": []})

print(f"append_list without Annotated: {result2['append_list']}")
print(f"  → Data silently lost, no error thrown. This is the bug to remember.")