import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent.state import AssistantState
from agent.tools.memory_tools import MEMORY_TOOLS
from config import PRIMARY_MODEL, OLLAMA_BASE_URL, PROMPTS_DIR

logger = logging.getLogger(__name__)
_llm_instance = None


def _llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    return _llm_instance


def direct_answer_node(state: AssistantState) -> dict:
    system   = (PROMPTS_DIR / "direct_answer.txt").read_text()
    user_msg = _get_last_user_message(state)

    context_parts = []
    if state.get("user_name"):
        context_parts.append(f"User name: {state['user_name']}")
    if state.get("user_preferences"):
        prefs = {k: v for k, v in state["user_preferences"].items() if k != "name"}
        if prefs:
            context_parts.append(f"Preferences: {json.dumps(prefs)}")
    if state.get("conversation_summary"):
        context_parts.append(f"Conversation summary: {state['conversation_summary']}")

    worker_results = state.get("worker_results", [])
    if worker_results:
        context_parts.append(
            "Information gathered:\n"
            + "\n\n".join(f"[{r['worker']}]: {r['result']}" for r in worker_results)
        )

    context = "\n\n".join(context_parts) if context_parts else "No prior context."

    response = _llm().bind_tools(MEMORY_TOOLS).invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Context:\n{context}\n\nUser: {user_msg}"),
    ])

    # If model wants to save a preference, handle it inline
    if response.tool_calls:
        from langchain_core.messages import ToolMessage
        tool_map = {t.name: t for t in MEMORY_TOOLS}
        tool_results = []
        for tc in response.tool_calls:
            fn = tool_map.get(tc["name"])
            if fn:
                result = fn.invoke(tc["args"])
                tool_results.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # Get final answer after tool use
        follow_up = _llm().invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Context:\n{context}\n\nUser: {user_msg}"),
            response,
            *tool_results,
        ])
        final_text = follow_up.content
        return {
            "messages":         [response, *tool_results, follow_up],
            "final_response":   final_text,
            "worker_results":   [{"worker": "direct_answer", "result": final_text[:300]}],
            "output_valid":     True,
            "status":           "complete",
            "_llm_response":    follow_up,
        }

    return {
        "messages":       [response],
        "final_response": response.content,
        "worker_results": [{"worker": "direct_answer", "result": response.content[:300]}],
        "output_valid":   True,
        "status":         "complete",
        "_llm_response":  response,
    }


def _get_last_user_message(state: AssistantState) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.__class__.__name__ == "HumanMessage":
            return msg.content
    return ""