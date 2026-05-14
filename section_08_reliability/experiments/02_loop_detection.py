import sys
import time
sys.path.append("../..")

from typing import TypedDict, Annotated, Optional
from operator import add
import json
import hashlib

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

print("=== Experiment 2 — Loop Detection ===\n")


class LoopTestState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    iteration_count: int
    max_iterations: int
    loop_detected: bool
    loop_reason: Optional[str]
    final_answer: Optional[str]
    status: str


@tool
def always_same_tool(query: str) -> str:
    """A tool that always returns the same result."""
    return json.dumps({"status": "success", "result": f"Result for: {query}"})


@tool
def different_tool(query: str) -> str:
    """A different tool."""
    return json.dumps({"status": "success", "result": f"Different result: {query}"})


def _detect_loops(state: LoopTestState) -> tuple[bool, str]:
    messages = state.get("messages", [])
    call_hashes = []
    for msg in messages[-16:]:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                h = hashlib.md5(
                    json.dumps({"name": tc["name"], "args": tc["args"]}, sort_keys=True).encode()
                ).hexdigest()
                call_hashes.append(h)

    if len(call_hashes) < 4:
        return False, ""

    recent = set(call_hashes[-2:])
    earlier = call_hashes[:-2]
    if any(h in earlier for h in recent):
        return True, "Repeated tool calls detected in recent history."
    return False, ""


def check_limit(state: LoopTestState) -> dict:
    count = state["iteration_count"] + 1
    if count > state["max_iterations"]:
        return {
            "iteration_count": count,
            "loop_detected": True,
            "loop_reason": f"Hit max iterations ({state['max_iterations']}).",
            "status": "failed",
        }
    detected, reason = _detect_loops(state)
    if detected:
        return {
            "iteration_count": count,
            "loop_detected": True,
            "loop_reason": reason,
            "status": "failed",
        }
    return {"iteration_count": count, "loop_detected": False}


def route_limit(state: LoopTestState) -> str:
    return "degrade" if state.get("loop_detected") else "agent"


def route_agent(state: LoopTestState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "check_limit"
    return "done"


# ── Scenario 1: Stuck agent always calling same tool ──────────────────────────

print("Scenario 1: Agent always calls the same tool (simulated stuck agent)")

call_counter = {"n": 0}

def stuck_agent(state: LoopTestState) -> dict:
    call_counter["n"] += 1
    # Always call always_same_tool with the same args
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "always_same_tool",
            "args": {"query": "budget total"},
            "id": f"call_{call_counter['n']}",
            "type": "tool_call",
        }],
    )
    return {"messages": [msg]}


def tool_exec(state: LoopTestState) -> dict:
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        result = always_same_tool.invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}


def degrade(state: LoopTestState) -> dict:
    return {
        "final_answer": (
            f"Partial answer (stopped due to: {state.get('loop_reason', 'unknown')}). "
            f"Ran {state['iteration_count']} iterations."
        ),
        "status": "degraded",
    }


builder = StateGraph(LoopTestState)
builder.add_node("check_limit", check_limit)
builder.add_node("agent", stuck_agent)
builder.add_node("tool_exec", tool_exec)
builder.add_node("degrade", degrade)
builder.add_node("done", lambda s: {"final_answer": "done", "status": "complete"})

builder.add_edge(START, "check_limit")
builder.add_conditional_edges("check_limit", route_limit, {"degrade": "degrade", "agent": "agent"})
builder.add_conditional_edges("agent", route_agent, {"check_limit": "check_limit", "done": "done"})
builder.add_edge("tool_exec", "check_limit")
builder.add_edge("degrade", "done")
builder.add_edge("done", END)

graph = builder.compile()

initial: LoopTestState = {
    "messages": [HumanMessage(content="What is the budget?")],
    "iteration_count": 0,
    "max_iterations": 10,
    "loop_detected": False,
    "loop_reason": None,
    "final_answer": None,
    "status": "running",
}

# Wire tool_exec into agent — rebuild with tool execution
builder2 = StateGraph(LoopTestState)
builder2.add_node("check_limit", check_limit)
builder2.add_node("agent", stuck_agent)
builder2.add_node("tool_exec", tool_exec)
builder2.add_node("degrade", degrade)
builder2.add_node("done", lambda s: {"status": "complete"})
builder2.add_edge(START, "check_limit")
builder2.add_conditional_edges("check_limit", route_limit, {"degrade": "degrade", "agent": "agent"})
builder2.add_edge("agent", "tool_exec")
builder2.add_edge("tool_exec", "check_limit")
builder2.add_edge("degrade", "done")
builder2.add_edge("done", END)
graph2 = builder2.compile()

result = graph2.invoke(initial)
print(f"  Loop detected: {result['loop_detected']}")
print(f"  Loop reason:   {result.get('loop_reason')}")
print(f"  Iterations:    {result['iteration_count']}")
print(f"  Status:        {result['status']}")
assert result["loop_detected"], "Should have detected a loop"
print("  ✓ Loop correctly detected and stopped\n")

# ── Loop analysis report ───────────────────────────────────────────────────────

print("Loop analysis report:")
tool_calls_made = [
    tc["args"]
    for msg in result["messages"]
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)
    for tc in msg.tool_calls
]
from collections import Counter
call_freq = Counter(str(c) for c in tool_calls_made)
print(f"  Total tool calls: {len(tool_calls_made)}")
print(f"  Unique call signatures: {len(call_freq)}")
for sig, count in call_freq.most_common(3):
    print(f"    {count}× {sig[:60]}")

print("\nAll loop detection tests passed.")