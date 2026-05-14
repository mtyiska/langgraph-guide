import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from agent.state import AssistantState
from agent.tools.task_tools import ALL_TASK_TOOLS, TASK_WRITE_TOOLS
from config import PRIMARY_MODEL, OLLAMA_BASE_URL, PROMPTS_DIR, ENABLE_HITL

logger  = logging.getLogger(__name__)
TOOL_MAP = {t.name: t for t in ALL_TASK_TOOLS}
WRITE_TOOL_NAMES = {t.name for t in TASK_WRITE_TOOLS}


def task_agent_node(state: AssistantState) -> dict:
    llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0).bind_tools(ALL_TASK_TOOLS)
    system       = (PROMPTS_DIR / "task_manager.txt").read_text()
    instructions = state.get("supervisor_instructions", "Help with task management.")

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=instructions),
    ])

    pending = _extract_write_op(response)
    return {
        "messages":               [response],
        "pending_write_operation": pending,
        "_llm_response":          response,
        "_tool_calls":            response.tool_calls or [],
    }


def execute_tools_node(state: AssistantState) -> dict:
    last    = state["messages"][-1]
    results = []

    for tc in getattr(last, "tool_calls", []):
        fn = TOOL_MAP.get(tc["name"])
        result = fn.invoke(tc["args"]) if fn else json.dumps({"status": "error", "message": f"Unknown tool: {tc['name']}"})
        results.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    summary = results[-1].content[:300] if results else "No tools executed."
    return {
        "messages":                results,
        "worker_results":          [{"worker": "task_manager", "result": summary}],
        "pending_write_operation": None,
    }


def summarise_task_result_node(state: AssistantState) -> dict:
    """Final node — produce a clean summary of what was done."""
    llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    instructions = state.get("supervisor_instructions", "")
    worker_results = state.get("worker_results", [])
    results_text = "\n".join(r["result"] for r in worker_results[-3:])

    response = llm.invoke([
        SystemMessage(content="Summarise task management actions clearly and briefly."),
        HumanMessage(content=f"Task requested: {instructions}\n\nActions taken:\n{results_text}"),
    ])
    return {
        "messages":       [response],
        "worker_results": [{"worker": "task_manager", "result": response.content[:300]}],
        "_llm_response":  response,
    }


def _extract_write_op(response) -> dict | None:
    if not response.tool_calls:
        return None
    for tc in response.tool_calls:
        if tc["name"] in WRITE_TOOL_NAMES:
            return {"tool": tc["name"], "args": tc["args"], "call_id": tc["id"]}
    return None


def route_task_agent(state: AssistantState) -> str:
    pending = state.get("pending_write_operation")
    last    = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "summarise"
    if pending and ENABLE_HITL:
        return "await_approval"
    return "execute_tools"


def build_task_manager_subgraph():
    builder = StateGraph(AssistantState)
    builder.add_node("task_agent",    task_agent_node)
    builder.add_node("execute_tools", execute_tools_node)
    builder.add_node("summarise",     summarise_task_result_node)

    builder.add_edge(START, "task_agent")
    builder.add_conditional_edges(
        "task_agent",
        route_task_agent,
        {
            "execute_tools":  "execute_tools",
            "await_approval": "execute_tools",   # HITL is handled at supervisor graph level
            "summarise":      "summarise",
        },
    )
    builder.add_edge("execute_tools", "summarise")
    builder.add_edge("summarise", END)

    return builder.compile()