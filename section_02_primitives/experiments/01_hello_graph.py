import sys
sys.path.append("../..")

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class SimpleState(TypedDict):
    input_text: str
    output_text: str

def reverse_node(state: SimpleState) -> dict:
    reversed_text = state["input_text"][::-1]
    return {"output_text": reversed_text}

graph_builder = StateGraph(SimpleState)
graph_builder.add_node("reverse", reverse_node)
graph_builder.add_edge(START, "reverse")
graph_builder.add_edge("reverse", END)

graph = graph_builder.compile()

# Print the graph structure
print("=== Graph Structure ===")
graph.get_graph().print_ascii()

# Invoke
result = graph.invoke({"input_text": "Hello LangGraph", "output_text": ""})

print("\n=== Result ===")
print(f"Input:  {result['input_text']}")
print(f"Output: {result['output_text']}")