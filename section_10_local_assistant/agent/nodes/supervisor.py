import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from typing import Literal
from agent.state import AssistantState
from config import FAST_MODEL, OLLAMA_BASE_URL, PROMPTS_DIR, ENABLE_MULTI_AGENT, MAX_ITERATIONS

logger = logging.getLogger(__name__)
_llm_instance = None


def _llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOllama(model=FAST_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    return _llm_instance


@tool
def routing_decision(
    intent:       str,
    route:        Literal["researcher", "writer", "task_manager", "direct_answer"],
    instructions: str,
    reasoning:    str,
) -> str:
    """Make a routing decision for this conversation turn."""
    return "Decision recorded."


def supervisor_node(state: AssistantState) -> dict:
    if not ENABLE_MULTI_AGENT:
        return {
            "route":                  "direct_answer",
            "intent":                 "direct",
            "supervisor_instructions": _get_last_user_message(state),
            "supervisor_rounds":      0,
        }

    # Safety: prevent supervisor from looping indefinitely
    rounds = state.get("supervisor_rounds", 0)
    if rounds >= 3:
        return {
            "route":                  "direct_answer",
            "intent":                 "synthesise",
            "supervisor_instructions": "Synthesise all worker results into a final answer.",
            "supervisor_rounds":      rounds,
        }

    llm_with_tools = _llm().bind_tools([routing_decision])

    memory_ctx   = _format_memory_context(state)
    worker_hist  = _format_worker_history(state)
    user_message = _get_last_user_message(state)
    system       = (PROMPTS_DIR / "supervisor.txt").read_text()

    response = llm_with_tools.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""Memory context:
{memory_ctx}

Previous worker results this turn:
{worker_hist or "None yet."}

User message: {user_message}
"""),
    ])

    if not response.tool_calls:
        return {
            "route":                  "direct_answer",
            "intent":                 "unclear",
            "supervisor_instructions": user_message,
            "supervisor_rounds":      rounds + 1,
            "_llm_response":          response,
        }

    args = response.tool_calls[0]["args"]
    return {
        "route":                  args["route"],
        "intent":                 args["intent"],
        "supervisor_instructions": args["instructions"],
        "supervisor_rounds":      rounds + 1,
        "_llm_response":          response,
    }


def route_after_supervisor(state: AssistantState) -> str:
    if state.get("loop_detected") or state.get("status") == "blocked":
        return "graceful_degradation"
    route = state.get("route", "direct_answer")
    valid = {"researcher", "writer", "task_manager", "direct_answer"}
    return route if route in valid else "direct_answer"


def _format_memory_context(state: AssistantState) -> str:
    parts = []
    if state.get("user_name"):
        parts.append(f"User name: {state['user_name']}")
    prefs = {k: v for k, v in (state.get("user_preferences") or {}).items() if k != "name"}
    if prefs:
        parts.append(f"Preferences: {json.dumps(prefs)}")
    if state.get("conversation_summary"):
        parts.append(f"Conversation so far: {state['conversation_summary']}")
    if state.get("relevant_past_context"):
        parts.append("Past context:\n" + "\n".join(f"- {c}" for c in state["relevant_past_context"][:3]))
    return "\n".join(parts) or "No memory loaded."


def _format_worker_history(state: AssistantState) -> str:
    results = state.get("worker_results", [])
    if not results:
        return ""
    return "\n".join(f"[{r['worker']}]: {str(r['result'])[:400]}" for r in results)


def _get_last_user_message(state: AssistantState) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.__class__.__name__ == "HumanMessage":
            return msg.content
    return ""