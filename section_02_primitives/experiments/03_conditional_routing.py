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
    if n < 10:
        category = "small"
    elif n < 100:
        category = "medium"
    else:
        category = "large"
    return {"category": category}

def route_by_category(state: NumberState) -> str:
    if state["number"] < 0:
        return END
    return state["category"]  # "small", "medium", or "large"

def handle_small(state: NumberState) -> dict:
    return {"message": f"{state['number']} is a small number (< 10)"}

def handle_medium(state: NumberState) -> dict:
    return {"message": f"{state['number']} is a medium number (10–99)"}

def handle_large(state: NumberState) -> dict:
    return {"message": f"{state['number']} is a large number (100+)"}

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

print("=== Routing Test ===")
for number in [5, 42, 200, 7, -3]:
    result = graph.invoke({"number": number, "category": None, "message": None})
    msg = result["message"] if result["message"] else f"Negative number ({number}) — routed to END, no handler."
    print(f"  {number:>5} → {msg}")