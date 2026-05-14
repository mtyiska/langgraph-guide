import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from agent.state import AssistantState
from agent.tools.document_tools import DOCUMENT_TOOLS
from config import PRIMARY_MODEL, OLLAMA_BASE_URL, PROMPTS_DIR, MAX_RETRIEVAL_STEPS

logger = logging.getLogger(__name__)
TOOL_MAP = {t.name: t for t in DOCUMENT_TOOLS}


def researcher_agent_node(state: AssistantState) -> dict:
    llm  = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0).bind_tools(DOCUMENT_TOOLS)
    system    = (PROMPTS_DIR / "researcher.txt").read_text()
    instructions = state.get("supervisor_instructions", "Research the user's question.")

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=instructions),
    ])
    return {"messages": [response], "_llm_response": response}


def researcher_tools_node(state: AssistantState) -> dict:
    last = state["messages"][-1]
    results = []
    sources = []

    for tc in getattr(last, "tool_calls", []):
        fn = TOOL_MAP.get(tc["name"])
        if not fn:
            result = json.dumps({"status": "error", "message": f"Unknown tool: {tc['name']}"})
        else:
            result = fn.invoke(tc["args"])
            try:
                data = json.loads(result)
                if data.get("status") == "success":
                    sources.append(tc["name"])
            except Exception:
                pass
        results.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    return {"messages": results, "sources_used": sources}


def researcher_synthesise_node(state: AssistantState) -> dict:
    """Produce final researcher answer from gathered tool results."""
    llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    gathered = "\n\n---\n\n".join(m.content[:600] for m in tool_messages[-4:])

    instructions = state.get("supervisor_instructions", "")
    response = llm.invoke([
        SystemMessage(content=(PROMPTS_DIR / "researcher.txt").read_text()),
        HumanMessage(content=f"Based on these search results:\n\n{gathered}\n\nAnswer: {instructions}"),
    ])

    return {
        "messages":       [response],
        "worker_results": [{"worker": "researcher", "result": response.content[:400]}],
        "_llm_response":  response,
    }


def route_researcher(state: AssistantState) -> str:
    last  = state["messages"][-1]
    iters = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None) and iters < MAX_RETRIEVAL_STEPS:
        return "tools"
    return "synthesise"


def build_researcher_subgraph():
    builder = StateGraph(AssistantState)
    builder.add_node("agent",      researcher_agent_node)
    builder.add_node("tools",      researcher_tools_node)
    builder.add_node("synthesise", researcher_synthesise_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_researcher, {"tools": "tools", "synthesise": "synthesise"})
    builder.add_edge("tools", "agent")
    builder.add_edge("synthesise", END)
    return builder.compile()