import sys
import os
sys.path.append("..")

from typing import TypedDict, Annotated, Optional
from operator import add
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from shared_state import SharedState
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename: str) -> str:
    with open(os.path.join(PROMPTS_DIR, filename)) as f:
        return f.read()


# ── Researcher internal state ─────────────────────────────────────────────────

class ResearcherInternalState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    current_task: str
    retrieval_count: int
    max_retrievals: int
    sources_cited: Annotated[list[dict], add]
    research_summary: Optional[str]
    iteration_count: int
    max_iterations: int
    # Passthrough fields from SharedState
    original_request: str
    worker_results: Annotated[list[dict], add]
    status: str
    next_worker: Optional[str]
    last_worker: Optional[str]
    final_output: Optional[str]


def build_researcher_graph(tools: list):
    tool_registry = {t.name: t for t in tools}
    llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    researcher_prompt = load_prompt("researcher.txt")

    def parse_task_node(state: ResearcherInternalState) -> dict:
        task = state.get("current_task", state.get("original_request", ""))
        system_msg = SystemMessage(content=researcher_prompt)
        human_msg = HumanMessage(content=f"Research task: {task}")
        return {
            "messages": [system_msg, human_msg],
            "retrieval_count": 0,
            "max_retrievals": 2,
            "iteration_count": 0,
            "max_iterations": 4,
        }

    def check_budget_node(state: ResearcherInternalState) -> dict:
        if state["retrieval_count"] >= state["max_retrievals"]:
            forced = HumanMessage(
                content=(
                    f"You have used {state['retrieval_count']} retrieval calls. "
                    "Stop searching now and produce your research summary using "
                    "the information already gathered. Include any gaps clearly."
                )
            )
            return {"messages": [forced]}
        if state["iteration_count"] >= state["max_iterations"]:
            forced = HumanMessage(
                content="Iteration limit reached. Produce your research summary now."
            )
            return {"messages": [forced]}
        return {}

    def agent_node(state: ResearcherInternalState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {
            "messages": [response],
            "iteration_count": state["iteration_count"] + 1,
        }

    def tools_node(state: ResearcherInternalState) -> dict:
        last = state["messages"][-1]
        tool_messages = []
        new_retrieval_count = state["retrieval_count"]
        new_sources = []

        for tc in last.tool_calls:
            name = tc["name"]
            args = tc["args"]
            tid = tc["id"]

            if name not in tool_registry:
                tool_messages.append(
                    ToolMessage(content=f"Error: tool '{name}' not found.", tool_call_id=tid)
                )
                continue

            result = tool_registry[name].invoke(args)

            if name == "search_documents":
                new_retrieval_count += 1
                new_sources.append({
                    "source": args.get("source_filter", "collection"),
                    "query": args.get("query", ""),
                })

            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tid))

        return {
            "messages": tool_messages,
            "retrieval_count": new_retrieval_count,
            "sources_cited": new_sources,
        }

    def synthesise_node(state: ResearcherInternalState) -> dict:
        final_content = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_content = msg.content
                break

        result_record = {
            "worker": "researcher",
            "task": state.get("current_task", ""),
            "result": final_content,
            "retrieval_count": state["retrieval_count"],
            "sources_cited": state["sources_cited"],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return {
            "research_summary": final_content,
            "worker_results": [result_record],
            "last_worker": "researcher",
        }

    # ── Routing ───────────────────────────────────────────────────────────────

    def route_after_budget(state: ResearcherInternalState) -> str:
        last = state["messages"][-1]
        if isinstance(last, HumanMessage) and "limit" in last.content.lower():
            return "agent"
        return "agent"

    def route_after_agent(state: ResearcherInternalState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "check_budget"
        return "synthesise"

    def route_after_check(state: ResearcherInternalState) -> str:
        last = state["messages"][-1]
        if isinstance(last, HumanMessage):
            return "agent"
        return "tools"

    # ── Build graph ───────────────────────────────────────────────────────────

    builder = StateGraph(ResearcherInternalState)

    builder.add_node("parse_task", parse_task_node)
    builder.add_node("check_budget", check_budget_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("synthesise", synthesise_node)

    builder.add_edge(START, "parse_task")
    builder.add_edge("parse_task", "agent")
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_conditional_edges("check_budget", route_after_check)
    builder.add_edge("tools", "agent")
    builder.add_edge("synthesise", END)

    return builder.compile()