import sys
sys.path.append("../..")

import time
from typing import TypedDict, Annotated, Optional
from operator import add
from langgraph.graph import StateGraph, START, END


FAKE_RESEARCH = "Project Atlas has a $320,000 budget. Launch: November 1 2024. Team lead: Jordan Kim."
FAKE_DRAFT = "Project Atlas will launch November 1 2024 with a $320,000 budget, led by Jordan Kim."


# ── Pattern A — Shared State ──────────────────────────────────────────────────

class PatternAState(TypedDict):
    request: str
    research: Optional[str]
    draft: Optional[str]
    worker_results: Annotated[list[str], add]
    status: str


def pattern_a_supervisor(state: PatternAState) -> dict:
    if state.get("research") is None:
        return {"status": "researching"}
    return {"status": "writing"}


def pattern_a_researcher(state: PatternAState) -> dict:
    time.sleep(0.01)
    return {"research": FAKE_RESEARCH, "worker_results": ["researcher_done"]}



def pattern_a_writer(state: PatternAState) -> dict:
    # Writer reads research directly from shared state
    research = state.get("research", "")
    return {"draft": FAKE_DRAFT, "worker_results": ["writer_done"], "status": "complete"}


def route_a(state: PatternAState) -> str:
    if state["status"] == "researching":
        return "researcher"
    if state["status"] == "writing":
        return "writer"
    return END


a_builder = StateGraph(PatternAState)
a_builder.add_node("supervisor", pattern_a_supervisor)  # single supervisor
a_builder.add_node("researcher", pattern_a_researcher)
a_builder.add_node("writer", pattern_a_writer)
a_builder.add_edge(START, "supervisor")
a_builder.add_conditional_edges("supervisor", route_a)
a_builder.add_edge("researcher", "supervisor")
a_builder.add_edge("writer", END)
graph_a = a_builder.compile()


# ── Pattern B — Message Passing Through Supervisor ────────────────────────────

class PatternBState(TypedDict):
    request: str
    supervisor_inbox: Optional[str]   # supervisor reads results here
    worker_inbox: Optional[str]       # workers read tasks here
    final_output: Optional[str]
    status: str
    turn: int


def pattern_b_supervisor(state: PatternBState) -> dict:
    turn = state["turn"]
    if turn == 0:
        return {
            "worker_inbox": f"Research: {state['request']}",
            "status": "researcher",
            "turn": 1,
        }
    if turn == 1:
        research = state.get("supervisor_inbox", "")
        return {
            "worker_inbox": f"Write a report using this research: {research}",
            "status": "writer",
            "turn": 2,
        }
    final = state.get("supervisor_inbox", "")
    return {"final_output": final, "status": "complete", "turn": 3}


def pattern_b_researcher(state: PatternBState) -> dict:
    return {"supervisor_inbox": FAKE_RESEARCH}


def pattern_b_writer(state: PatternBState) -> dict:
    task = state.get("worker_inbox", "")
    return {"supervisor_inbox": FAKE_DRAFT}


def route_b(state: PatternBState) -> str:
    status = state["status"]
    if status == "researcher":
        return "researcher"
    if status == "writer":
        return "writer"
    return END


b_builder = StateGraph(PatternBState)
b_builder.add_node("supervisor", pattern_b_supervisor)
b_builder.add_node("researcher", pattern_b_researcher)
b_builder.add_node("writer", pattern_b_writer)
b_builder.add_edge(START, "supervisor")
b_builder.add_conditional_edges("supervisor", route_b)
b_builder.add_edge("researcher", "supervisor")
b_builder.add_edge("writer", "supervisor")
graph_b = b_builder.compile()


# ── Pattern C — Blackboard ────────────────────────────────────────────────────

class Blackboard(TypedDict):
    request: str
    research: Optional[str]
    outline: Optional[str]
    draft: Optional[str]
    final: Optional[str]
    status: str


def pattern_c_researcher(state: Blackboard) -> dict:
    return {"research": FAKE_RESEARCH}


def pattern_c_writer(state: Blackboard) -> dict:
    research = state.get("research", "")
    outline = f"1. Budget overview\n2. Timeline\n3. Team"
    draft = FAKE_DRAFT
    return {"outline": outline, "draft": draft}


def pattern_c_supervisor(state: Blackboard) -> dict:
    draft = state.get("draft", "")
    return {"final": draft, "status": "complete"}


def route_c(state: Blackboard) -> str:
    if not state.get("research"):
        return "researcher"
    if not state.get("draft"):
        return "writer"
    if not state.get("final"):
        return "supervisor"
    return END


c_builder = StateGraph(Blackboard)
c_builder.add_node("researcher", pattern_c_researcher)
c_builder.add_node("writer", pattern_c_writer)
c_builder.add_node("supervisor", pattern_c_supervisor)
c_builder.add_conditional_edges(START, route_c)
c_builder.add_edge("researcher", "supervisor")
c_builder.add_edge("writer", "supervisor")
c_builder.add_edge("supervisor", END)
graph_c = c_builder.compile()


# ── Run and compare ───────────────────────────────────────────────────────────

print("=== Pattern A — Shared State ===")
result_a = graph_a.invoke({
    "request": "Write about Project Atlas",
    "research": None,
    "draft": None,
    "worker_results": [],
    "status": "pending",
})
print(f"  Draft produced: {bool(result_a.get('draft'))}")
print(f"  Research visible to writer: YES (direct shared state read)")
print(f"  Coupling: HIGH — writer knows research field name directly")

print("\n=== Pattern B — Message Passing ===")
result_b = graph_b.invoke({"request": "Write about Project Atlas", "supervisor_inbox": None, "worker_inbox": None, "final_output": None, "status": "start", "turn": 0})
print(f"  Final output produced: {bool(result_b.get('final_output'))}")
print(f"  Research visible to writer: via supervisor only")
print(f"  Coupling: LOW — researcher and writer never touch each other's fields")

print("\n=== Pattern C — Blackboard ===")
result_c = graph_c.invoke({"request": "Write about Project Atlas", "research": None, "outline": None, "draft": None, "final": None, "status": "running"})
print(f"  Final produced: {bool(result_c.get('final'))}")
print(f"  Work product IS the communication medium")
print(f"  Coupling: MEDIUM — workers share field names but don't call each other")

print("""
=== Comparison Summary ===

Pattern A (Shared State):
  + Simplest to implement and debug
  + Any node can read any field
  - Tight schema coupling — renaming a field breaks all consumers
  - Adding a 4th agent requires all agents to know about new fields
  Best for: small stable pipelines designed together

Pattern B (Message Passing):
  + Cleanest separation — workers are fully decoupled from each other
  + Swap researcher without touching writer at all
  - More supervisor LLM calls — supervisor mediates everything
  - Harder to debug — data flow is less visible in state
  Best for: pipelines needing maximum worker independence

Pattern C (Blackboard):
  + Work product is self-documenting — the blackboard tells you everything
  + Natural fit for document creation workflows
  - Workers must know blackboard field names (medium coupling)
  - Concurrent writes to same field need careful reducer design
  Best for: collaborative document workflows, creative pipelines
""")