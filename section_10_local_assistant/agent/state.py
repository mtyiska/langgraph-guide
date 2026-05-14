from typing import TypedDict, Annotated, Optional, Literal
from operator import add
from langchain_core.messages import BaseMessage


class AssistantState(TypedDict):
    # Conversation
    messages:     Annotated[list[BaseMessage], add]
    session_id:   str
    thread_id:    str
    turn_number:  int

    # User memory
    user_name:             Optional[str]
    user_preferences:      dict
    relevant_past_context: list[str]
    conversation_summary:  Optional[str]

    # Supervisor routing
    intent:                  Optional[str]
    route:                   Optional[Literal["researcher", "writer", "task_manager", "direct_answer"]]
    supervisor_instructions: Optional[str]
    worker_results:          Annotated[list[dict], add]
    supervisor_rounds:       int

    # Control flow
    iteration_count: int
    loop_detected:   bool
    status:          Literal["running", "complete", "degraded", "blocked", "awaiting_approval"]

    # HITL
    pending_write_operation: Optional[dict]
    write_approved:          Optional[bool]
    write_decision_note:     Optional[str]

    # Output
    final_response:     Optional[str]
    response_type:      Optional[Literal["answer", "task_confirmation", "clarification", "error"]]
    sources_used:       Annotated[list[str], add]
    guardrail_triggered: bool
    output_valid:        bool

    # Internal tracing (stripped before delivery)
    run_id:        Optional[str]
    _llm_response: Optional[object]
    _tool_calls:   Optional[list]


def make_initial_state(
    user_message: str,
    session_id:   str = "default",
    thread_id:    str = "default",
    run_id:       str = "",
) -> AssistantState:
    from langchain_core.messages import HumanMessage
    return {
        "messages":               [HumanMessage(content=user_message)],
        "session_id":             session_id,
        "thread_id":              thread_id,
        "turn_number":            0,
        "user_name":              None,
        "user_preferences":       {},
        "relevant_past_context":  [],
        "conversation_summary":   None,
        "intent":                 None,
        "route":                  None,
        "supervisor_instructions": None,
        "worker_results":         [],
        "supervisor_rounds":      0,
        "iteration_count":        0,
        "loop_detected":          False,
        "status":                 "running",
        "pending_write_operation": None,
        "write_approved":         None,
        "write_decision_note":    None,
        "final_response":         None,
        "response_type":          None,
        "sources_used":           [],
        "guardrail_triggered":    False,
        "output_valid":           False,
        "run_id":                 run_id,
        "_llm_response":          None,
        "_tool_calls":            None,
    }


def clean_for_delivery(state: dict) -> dict:
    """Strip internal fields before serialising to the user."""
    return {k: v for k, v in state.items() if not k.startswith("_")}