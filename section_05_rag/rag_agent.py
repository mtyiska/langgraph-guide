import sys
import os
sys.path.append("..")

from typing import TypedDict, Annotated, Optional
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from rich.console import Console
from rich.panel import Panel

from ingestion import ingest_documents, get_collection
from retrieval_tools import build_retrieval_tools
from citation_formatter import format_citations, parse_search_results_for_citations
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

console = Console()

# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    question: str
    retrieval_count: int
    max_retrievals: int
    sources_used: Annotated[list[dict], add]
    answer: Optional[str]
    iteration_count: int
    max_iterations: int

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a document QA assistant with access to a local document collection.

RETRIEVAL STRATEGY:
1. Start with list_document_sources to understand what documents are available.
2. Search with specific, targeted queries — use concrete nouns and key terms.
3. Evaluate result relevance scores:
   - score < 0.5: high relevance — likely sufficient, proceed to answer
   - score 0.5-1.0: medium relevance — consider one refinement search
   - score > 1.0: low relevance — reformulate query with different terminology
4. For multi-part questions, break into sub-questions and search each separately.
5. Use source_filter once you identify the right document for targeted follow-up.
6. After 3 searches with poor results, answer with what you have and state gaps clearly.

ANSWERING RULES:
- Only state facts that appear in retrieved chunks — never use training knowledge for domain facts.
- If information is not in the documents, say so explicitly.
- Never guess or fabricate figures, names, or dates.
- Keep answers concise and cite your sources.
"""

# ── Build model and tools ─────────────────────────────────────────────────────

def build_agent(collection=None):
    if collection is None:
        collection = get_collection()

    tools = build_retrieval_tools(collection)
    tool_registry = {t.name: t for t in tools}

    llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def agent_node(state: RAGState) -> dict:
        new_iter = state["iteration_count"] + 1
        response = llm_with_tools.invoke(state["messages"])
        return {
            "messages": [response],
            "iteration_count": new_iter,
        }

    def tools_node_fn(state: RAGState) -> dict:
        last = state["messages"][-1]
        tool_messages = []
        new_sources = []
        new_retrieval_count = state["retrieval_count"]

        for tc in last.tool_calls:
            name = tc["name"]
            args = tc["args"]
            tid = tc["id"]

            if name not in tool_registry:
                from langchain_core.messages import ToolMessage
                tool_messages.append(
                    ToolMessage(content=f"Error: tool '{name}' not found.", tool_call_id=tid)
                )
                continue

            result = tool_registry[name].invoke(args)

            if name == "search_documents":
                new_retrieval_count += 1
                query_used = args.get("query", "")
                citations = parse_search_results_for_citations(str(result), query_used)
                new_sources.extend(citations)

            from langchain_core.messages import ToolMessage
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tid))

        return {
            "messages": tool_messages,
            "retrieval_count": new_retrieval_count,
            "sources_used": new_sources,
        }

    def check_budget_node(state: RAGState) -> dict:
        if state["retrieval_count"] >= state["max_retrievals"]:
            forced = HumanMessage(
                content=(
                    f"You have used {state['retrieval_count']} retrieval calls "
                    f"(maximum is {state['max_retrievals']}). "
                    "Stop searching and provide your best answer now using only "
                    "the information already retrieved. Clearly state any gaps."
                )
            )
            return {"messages": [forced]}
        if state["iteration_count"] >= state["max_iterations"]:
            forced = HumanMessage(
                content=(
                    f"You have reached the iteration limit ({state['max_iterations']}). "
                    "Provide your best answer now using what you have gathered."
                )
            )
            return {"messages": [forced]}
        return {}

    def format_answer_node(state: RAGState) -> dict:
        final_answer = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break

        citations_text = format_citations(state["sources_used"])
        full_answer = final_answer + citations_text if citations_text else final_answer

        return {"answer": full_answer}

    # ── Routing ───────────────────────────────────────────────────────────────

    def route_after_budget(state: RAGState) -> str:
        last = state["messages"][-1]
        if isinstance(last, HumanMessage):
            return "agent"
        return "agent"

    def route_after_agent(state: RAGState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "check_budget"
        return "format_answer"

    def route_after_check(state: RAGState) -> str:
        last = state["messages"][-1]
        if isinstance(last, HumanMessage):
            return "agent"
        return "tools"

    # ── Build graph ───────────────────────────────────────────────────────────

    builder = StateGraph(RAGState)

    builder.add_node("check_budget", check_budget_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node_fn)
    builder.add_node("format_answer", format_answer_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_conditional_edges("check_budget", route_after_check)
    builder.add_edge("tools", "agent")
    builder.add_edge("format_answer", END)

    return builder.compile(), tool_registry

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold blue]Agentic Document QA[/bold blue]")
    console.print("Ingesting documents...")
    ingest_documents(verbose=True)

    collection = get_collection()
    graph, _ = build_agent(collection)

    console.print("\nGraph structure:")
    graph.get_graph().print_ascii()

    console.print("\nType [bold]/exit[/bold] to quit, [bold]/graph[/bold] to see structure\n")

    while True:
        try:
            user_input = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/graph":
            graph.get_graph().print_ascii()
            continue

        initial_state: RAGState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_input),
            ],
            "question": user_input,
            "retrieval_count": 0,
            "max_retrievals": 5,
            "sources_used": [],
            "answer": None,
            "iteration_count": 0,
            "max_iterations": 15,
        }

        console.print("[yellow]Searching...[/yellow]")

        for step in graph.stream(initial_state, stream_mode="updates"):
            node_name = list(step.keys())[0]
            if node_name not in ("format_answer",):
                console.print(f"[dim]  → {node_name}[/dim]")

        result = graph.invoke(initial_state)

        console.print(Panel(
            result["answer"] or "No answer produced.",
            title="Answer",
            border_style="green"
        ))
        console.print(
            f"[dim]Retrievals: {result['retrieval_count']} | "
            f"Iterations: {result['iteration_count']} | "
            f"Sources cited: {len(result['sources_used'])}[/dim]\n"
        )


if __name__ == "__main__":
    main()