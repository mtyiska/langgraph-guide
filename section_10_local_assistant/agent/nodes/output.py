import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AssistantState
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


def validate_output_node(state: AssistantState) -> dict:
    response = state.get("final_response", "")
    if response and len(response.strip()) > 10:
        return {"output_valid": True}
    return {"output_valid": False}


def graceful_degradation_node(state: AssistantState) -> dict:
    failure = _determine_failure(state)
    worker_results = state.get("worker_results", [])
    partial = "\n\n".join(r["result"] for r in worker_results[-3:]) if worker_results else ""

    try:
        llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        user_q = _get_last_user_message(state)
        prompt = f"""The assistant encountered a problem: {failure}

Partial information gathered:
{partial or "Nothing gathered."}

Original question: {user_q}

Produce a response that answers as much as possible, clearly states what
could not be answered and why, and does not fabricate information."""

        response = llm.invoke([
            SystemMessage(content="You produce honest partial answers when research is incomplete."),
            HumanMessage(content=prompt),
        ])
        answer = response.content
    except Exception as e:
        logger.error(f"Graceful degradation LLM call failed: {e}")
        answer = f"I encountered a problem and could not complete the request.\n\nReason: {failure}"

    return {
        "final_response": answer,
        "output_valid":   True,
        "status":         "degraded",
    }


def deliver_answer_node(state: AssistantState) -> dict:
    response = state.get("final_response", "No response generated.")
    sources  = state.get("sources_used", [])

    if sources:
        source_list = ", ".join(set(s.split("/")[-1] for s in sources))
        response = f"{response}\n\n_Sources: {source_list}_"

    return {"final_response": response, "status": "complete"}


def _determine_failure(state: AssistantState) -> str:
    if state.get("loop_detected"):
        return "Agent exceeded iteration limit or repeated tool calls."
    if not state.get("output_valid", True):
        return "Output validation failed after retry."
    if state.get("guardrail_triggered"):
        return "Request was outside permitted scope."
    return state.get("status", "Unknown failure.")


def _get_last_user_message(state: AssistantState) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.__class__.__name__ == "HumanMessage":
            return msg.content
    return ""