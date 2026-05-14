import sys
sys.path.append("../..")

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

class NumberState(TypedDict):
    number: int
    category: Optional[str]
    message: Optional[str]

def classify(state: NumberState) -> dict:
    n = state["number"]
    if n < 0:
        return {"category": "negative"}
    elif n < 10:
        return {"category": "small"}
    elif n < 100:
        return {"category": "medium"}
    else:
        return {"category": "large"}

def route_by_category(state: NumberState) -> str:
    cat = state["category"]
    if cat == "negative":
        return END
    return cat

def handle_small(state: NumberState) -> dict:
    return {"message": f"{state['number']} is small"}

def handle_medium(state: NumberState) -> dict:
    return {"message": f"{state['number']} is medium"}

def handle_large(state: NumberState) -> dict:
    return {"message": f"{state['number']} is large"}

builder = StateGraph(NumberState)
builder.add_node("classify", classify)
builder.add_node("small", handle_small)
builder.add_node("medium", handle_medium)
builder.add_node("large", handle_large)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_by_category)
builder.add_edge("small", END)
builder.add_edge("medium", END)
builder.add_edge("large", END)

graph = builder.compile()
initial = {"number": 42, "category": None, "message": None}

print("=== stream_mode='updates' — only what changed each step ===")
for step in graph.stream(initial, stream_mode="updates"):
    node_name = list(step.keys())[0]
    updates = step[node_name]
    print(f"  [{node_name}] changed: {updates}")

print("\n=== stream_mode='values' — full state after each step ===")
for step in graph.stream(initial, stream_mode="values"):
    print(f"  full state: {step}")

print("\n=== Deliberate bug: node returns unknown key ===")

def buggy_node(state: NumberState) -> dict:
    return {"nonexistent_field": "this key is not in the TypedDict"}

builder2 = StateGraph(NumberState)
builder2.add_node("buggy", buggy_node)
builder2.add_edge(START, "buggy")
builder2.add_edge("buggy", END)

graph2 = builder2.compile()
result = graph2.invoke({"number": 1, "category": None, "message": None})
print(f"  Result after buggy node: {result}")
print(f"  'nonexistent_field' in result: {'nonexistent_field' in result}")
print(f"  → Unknown keys are silently ignored. Typos in return dicts produce silent no-ops.")