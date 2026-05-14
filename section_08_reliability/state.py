from typing import TypedDict, Annotated, Optional, Literal
from operator import add
from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    # Conversation
    messages: Annotated[list[BaseMessage], add]
    system_prompt: str

    # Control flow
    iteration_count: int
    max_iterations: int
    loop_detected: bool
    loop_reason: Optional[str]

    # Guardrail
    guardrail_triggered: bool
    guardrail_decision: Optional[str]   # "allow", "block", "clarify"

    # Output validation
    raw_answer: Optional[str]
    validated_answer: Optional[dict]
    output_valid: bool
    validation_error: Optional[str]

    # Injection defence
    injection_detected: bool
    injection_reason: Optional[str]

    # Final delivery
    final_answer: Optional[str]
    answer_complete: bool
    failure_reason: Optional[str]
    status: Literal["running", "blocked", "degraded", "complete", "failed"]