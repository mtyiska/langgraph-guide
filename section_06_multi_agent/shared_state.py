from typing import TypedDict, Annotated, Optional
from operator import add
from langchain_core.messages import BaseMessage


class SharedState(TypedDict):
    # The original user request
    original_request: str

    # Supervisor coordination
    current_task: Optional[str]
    next_worker: Optional[str]
    last_worker: Optional[str]
    worker_results: Annotated[list[dict], add]

    # Final output
    final_output: Optional[str]
    status: str  # "running", "complete", "failed"
    iteration_count: int
    max_iterations: int
    max_retrievals: int  


class ResearcherState(SharedState):
    research_query: str
    search_results: Annotated[list[str], add]
    sources_cited: Annotated[list[dict], add]
    research_summary: Optional[str]


class WriterState(SharedState):
    research_input: str
    draft: Optional[str]
    revision_notes: Optional[str]
    needs_revision: bool
    word_count: int


def make_initial_state(request: str, max_iterations: int = 10) -> SharedState:
    return {
        "original_request": request,
        "current_task": None,
        "next_worker": None,
        "last_worker": None,
        "worker_results": [],
        "final_output": None,
        "status": "running",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "max_retrievals": 5,  # add this
    }


def build_supervisor_context(state: SharedState) -> str:
    parts = [f"Original request: {state['original_request']}"]

    if state["worker_results"]:
        parts.append("\nWork completed so far:")
        for r in state["worker_results"]:
            worker = r.get("worker", "unknown")
            task = r.get("task", "")
            result_preview = str(r.get("result", ""))[:300]
            parts.append(f"  [{worker.upper()}] Task: {task}\n  Result: {result_preview}...")
    else:
        parts.append("\nNo work completed yet.")

    parts.append(f"\nIteration: {state['iteration_count']} of {state['max_iterations']}")
    return "\n".join(parts)