# agent.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import TypedDict, Annotated
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from rich.console import Console
from rich.panel import Panel

from tools import all_tools
from knowledge_base import init_db
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

console = Console()

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    iteration_count: int
    max_iterations: int

# ── Model ─────────────────────────────────────────────────────────────────────

llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
llm_with_tools = llm.bind_tools(all_tools)
tool_registry = {t.name: t for t in all_tools}

# ── Nodes ─────────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    new_count = state["iteration_count"] + 1
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "iteration_count": new_count,
    }


def validate_node(state: AgentState) -> dict:
    last = state["messages"][-1]

    # Mode 4: iteration limit reached — force a final answer
    if state["iteration_count"] >= state["max_iterations"]:
        forced = HumanMessage(
            content=(
                f"You have made {state['iteration_count']} tool calls. "
                "Stop calling tools and provide your best answer now "
                "using the information you have gathered so far."
            )
        )
        return {"messages": [forced]}

    # Mode 3: unknown tool name — correct and route back to agent
    for tc in last.tool_calls:
        if tc["name"] not in tool_registry:
            error_msg = HumanMessage(
                content=(
                    f"Error: tool '{tc['name']}' does not exist. "
                    f"Available tools: {', '.join(tool_registry.keys())}. "
                    "Please call one of the available tools."
                )
            )
            return {"messages": [error_msg]}

    return {}  # validation passed — no change, route to tools

# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "validate"
    return END


def route_after_validate(state: AgentState) -> str:
    last = state["messages"][-1]
    # If validate injected a HumanMessage (error or forced answer), loop back
    if isinstance(last, HumanMessage):
        return "agent"
    return "tools"

# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("validate", validate_node)
    builder.add_node("tools", ToolNode(all_tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_conditional_edges("validate", route_after_validate)
    builder.add_edge("tools", "agent")

    return builder.compile()

# ── Run summary ───────────────────────────────────────────────────────────────

def print_run_summary(result: dict):
    messages = result["messages"]
    tool_calls_made = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(tc["name"])

    console.print(
        f"\n[dim]Iterations: {result['iteration_count']} | "
        f"Tool calls: {len(tool_calls_made)} | "
        f"Tools used: {', '.join(tool_calls_made) if tool_calls_made else 'none'}[/dim]"
    )

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    init_db()

    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    with open(prompt_path) as f:
        system_prompt = f.read()

    graph = build_graph()

    console.print("\n[bold blue]Project Orion Research Assistant[/bold blue]")
    console.print(f"Model: {PRIMARY_MODEL} | Max iterations: 10")
    console.print("Type [bold]/exit[/bold] to quit, [bold]/graph[/bold] to see structure\n")

    base_messages = [SystemMessage(content=system_prompt)]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("Goodbye.")
            break
        if user_input == "/graph":
            graph.get_graph().print_ascii()
            continue

        state = {
            "messages": base_messages + [HumanMessage(content=user_input)],
            "iteration_count": 0,
            "max_iterations": 10,
        }

        console.print("\n[yellow]Thinking...[/yellow]")
        result = graph.invoke(state)

        final_answer = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break

        console.print(Panel(final_answer, title="Assistant", border_style="green"))
        print_run_summary(result)
        print()


if __name__ == "__main__":
    main()