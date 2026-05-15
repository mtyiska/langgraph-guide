import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))      # section_06_multi_agent/
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))   # langgraph_guide/

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from typing import Literal
from config import PRIMARY_MODEL, OLLAMA_BASE_URL
from shared_state import SharedState, build_supervisor_context, make_initial_state

llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


@tool
def route_to_worker(
    worker: Literal["researcher", "writer", "FINISH"],
    task_description: str,
    context_for_worker: str
) -> str:
    """Route to a worker agent or finish. Always call this tool."""
    return f"Routing to {worker}"


llm_with_tools = llm.bind_tools([route_to_worker])

SUPERVISOR_PROMPT = """You coordinate a research and writing pipeline.
You have two workers: researcher (finds facts) and writer (drafts content).

Routing rules — follow these exactly:
- If no worker_results exist: route to researcher
- If only researcher has run: route to writer
- If both researcher and writer have run: route to FINISH
- Never route to the same worker twice in one turn

Always call the route_to_worker tool. Never respond with plain text.
The 'worker' argument must be exactly one of: researcher, writer, FINISH"""


def call_supervisor(state: SharedState) -> dict:
    context = build_supervisor_context(state)
    response = llm_with_tools.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=context),
    ])
    if response.tool_calls:
        tc = response.tool_calls[0]
        return {
            "worker": tc["args"].get("worker"),
            "task": tc["args"].get("task_description"),
            "context": str(tc["args"].get("context_for_worker", ""))[:200],
        }
    return {"worker": "researcher", "task": "fallback", "context": ""}

test_states = [
    {
        "label": "Empty state — no work done yet",
        "state": make_initial_state("Write a summary of Project Atlas budget and timeline"),
    },
    {
        "label": "Researcher ran — should route to writer",
        "state": {
            **make_initial_state("Write a summary of Project Atlas budget and timeline"),
            "worker_results": [{
                "worker": "researcher",
                "task": "Find budget and timeline info",
                "result": "Budget: $320,000. Timeline: Launch November 1 2024. Q3 spend: $161,000.",
                "timestamp": "2024-01-01T00:00:00",
            }],
            "iteration_count": 1,
        },
    },
    {
        "label": "Writer ran — should FINISH",
        "state": {
            **make_initial_state("Write a summary of Project Atlas budget and timeline"),
            "worker_results": [
                {
                    "worker": "researcher",
                    "task": "Find budget and timeline info",
                    "result": "Budget: $320,000. Timeline: November 1 2024.",
                    "timestamp": "2024-01-01T00:00:00",
                },
                {
                    "worker": "writer",
                    "task": "Write summary",
                    "result": "Project Atlas has a $320,000 budget and launches November 1 2024.",
                    "timestamp": "2024-01-01T00:01:00",
                },
            ],
            "iteration_count": 2,
        },
    },
    {
        "label": "Contradictory state — writer ran but status still running",
        "state": {
            **make_initial_state("Write a brief report on team structure"),
            "worker_results": [{
                "worker": "writer",
                "task": "Write report",
                "result": "The team consists of 8 members.",
                "timestamp": "2024-01-01T00:00:00",
            }],
            "status": "running",
            "iteration_count": 1,
        },
    },
    {
        "label": "Multi-part request — researcher ran for part 1 only",
        "state": {
            **make_initial_state("Compare the budget status and technical architecture of Project Atlas"),
            "worker_results": [{
                "worker": "researcher",
                "task": "Find budget status",
                "result": "Budget $320,000 total. $161,000 spent. ON TRACK.",
                "timestamp": "2024-01-01T00:00:00",
            }],
            "iteration_count": 1,
        },
    },
]

print("=== Supervisor Routing Test ===\n")
for test in test_states:
    print(f"Scenario: {test['label']}")
    decision = call_supervisor(test["state"])
    print(f"  → Route to: {decision['worker']}")
    print(f"  → Task:     {decision['task']}")
    print(f"  → Context:  {decision['context'][:100]}...")
    print()