import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import TypedDict, Annotated
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from config import PRIMARY_MODEL, OLLAMA_BASE_URL
from tools import all_tools

RESEARCH_SYSTEM = """You are a research assistant with access to a knowledge base of project documents.

You have access to these tools:
- list_available_files: call this first to see what files exist
- read_file: read a specific file by name
- search_documents: search across all files for a keyword or phrase

Rules:
1. Always use tools to find information — never guess.
2. If you don't know which file to read, call search_documents first.
3. If search_documents finds nothing, call list_available_files then read the most relevant file.
4. When you answer, cite which file the information came from.
5. If the information is genuinely not in any document, say so clearly.
"""

llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
llm_with_tools = llm.bind_tools(all_tools)
tool_registry = {t.name: t for t in all_tools}


class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    iteration_count: int
    max_iterations: int
    final_answer: str
    status: str


def agent_node(state: ResearchState) -> dict:
    new_count = state["iteration_count"] + 1
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "iteration_count": new_count,
    }


def validate_node(state: ResearchState) -> dict:
    last = state["messages"][-1]

    if state["iteration_count"] >= state["max_iterations"]:
        forced = HumanMessage(
            content=(
                f"You have made {state['iteration_count']} tool calls. "
                "Stop calling tools and provide your best answer now "
                "using the information gathered so far."
            )
        )
        return {"messages": [forced]}

    for tc in last.tool_calls:
        if tc["name"] not in tool_registry:
            error_msg = HumanMessage(
                content=(
                    f"Tool '{tc['name']}' does not exist. "
                    f"Available tools: {', '.join(tool_registry.keys())}. "
                    "Please call one of the available tools."
                )
            )
            return {"messages": [error_msg]}

    return {}


def deliver_answer_node(state: ResearchState) -> dict:
    final_answer = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final_answer = msg.content
            break
    return {"final_answer": final_answer, "status": "complete"}


def route_after_agent(state: ResearchState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "validate"
    return "deliver_answer"


def route_after_validate(state: ResearchState) -> str:
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return "agent"
    return "tools"


def build_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("agent", agent_node)
    builder.add_node("validate", validate_node)
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_node("deliver_answer", deliver_answer_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_conditional_edges("validate", route_after_validate)
    builder.add_edge("tools", "agent")
    builder.add_edge("deliver_answer", END)

    return builder.compile()


def make_initial_state(query: str, max_iterations: int = 10) -> dict:
    return {
        "messages": [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=query),
        ],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "final_answer": "",
        "status": "running",
    }

def build_traced_graph(trace_store, run_context):
    from instrumentation import traced_node as _traced_node

    builder = StateGraph(ResearchState)

    builder.add_node("agent", _traced_node(agent_node, trace_store, run_context))
    builder.add_node("validate", _traced_node(validate_node, trace_store, run_context))
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_node("deliver_answer", _traced_node(deliver_answer_node, trace_store, run_context))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_conditional_edges("validate", route_after_validate)
    builder.add_edge("tools", "agent")
    builder.add_edge("deliver_answer", END)

    return builder.compile()