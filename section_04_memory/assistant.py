import sys
import os
import uuid
sys.path.append("..")

from typing import TypedDict, Annotated, Optional
from operator import add

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage
)
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel

from memory_store import (
    init_db, load_profile, create_profile, update_profile,
    retrieve_memories, save_memory, get_memory_count,
    consolidate_memories, save_session, check_duplicate
)
from summariser import estimate_tokens, maybe_summarise
from memory_extractor import extract_memories
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

console = Console()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "section_03_react_agent", "test_docs")
CONTEXT_WINDOW = 8192
SUMMARISE_THRESHOLD = 0.70
CONSOLIDATE_THRESHOLD = 50

# ── State ─────────────────────────────────────────────────────────────────────

class AssistantState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    session_id: str
    user_id: str
    turn_count: int
    needs_summarisation: bool
    summary_so_far: Optional[str]
    tokens_used_estimate: int
    user_profile: Optional[dict]
    relevant_memories: Annotated[list[str], add]
    iteration_count: int
    max_iterations: int

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def list_available_files() -> str:
    """List all available project document files."""
    try:
        files = os.listdir(DOCS_DIR)
        return "Files: " + ", ".join(sorted(files))
    except Exception as e:
        return f"Error: {e}"

@tool
def read_file(path: str) -> str:
    """Read a project document. Pass just the filename."""
    filename = os.path.basename(path)
    full_path = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(full_path):
        return f"File '{filename}' not found. Use list_available_files to see options."
    with open(full_path) as f:
        content = f.read()
    return f"[{filename}]\n{content}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Example: '180000 - 42000'"""
    import math
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["math"] = math
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"

_current_user_id: str = "default_user"  # set at session start

@tool
def update_preference(key: str, value: str) -> str:
    """
    Save a user preference or personal detail to long-term memory.
    Use when the user states their name, communication preference, or any persistent fact.
    Keys: 'name', 'style' (concise/detailed/formal/casual), 'timezone', 'interests'
    Example: update_preference('name', 'Alex') or update_preference('style', 'concise')
    """
    update_profile(_current_user_id, key, value)
    return f"Preference saved: {key} = {value}"

all_tools = [list_available_files, read_file, calculate, update_preference]

# ── System prompt builder ─────────────────────────────────────────────────────

BASE_PROMPT = """You are a helpful personal assistant with memory across sessions.
You help the user find information from project documents and remember their preferences.

Tools available:
- list_available_files: see what documents exist
- read_file: read a document (filename only)
- calculate: evaluate math expressions
- update_preference: save user preferences or personal details to memory

Rules:
1. When a user tells you their name, preference, or personal info — call update_preference immediately.
2. Always use tools for factual questions. Never guess.
3. Keep responses aligned with the user's preferred style.
4. Cite your sources (file or memory) in answers."""


def build_system_message(state: AssistantState) -> str:
    parts = [BASE_PROMPT]

    if state.get("user_profile"):
        p = state["user_profile"]
        name = p.get("name") or "unknown"
        style = p.get("style") or "default"
        interests = p.get("interests", [])
        interests_str = ", ".join(interests) if interests else "none noted"
        parts.append(f"\nUser context:\n- Name: {name}\n- Style: {style}\n- Interests: {interests_str}")

    if state.get("summary_so_far"):
        parts.append(f"\nEarlier in this conversation (summarised):\n{state['summary_so_far']}")

    if state.get("relevant_memories"):
        mem_lines = "\n".join(f"- {m}" for m in state["relevant_memories"])
        parts.append(f"\nContext from past sessions:\n{mem_lines}")

    return "\n\n".join(parts)

# ── Nodes ─────────────────────────────────────────────────────────────────────

def load_memory_node(state: AssistantState) -> dict:
    user_id = state["user_id"]
    profile = load_profile(user_id)
    if not profile:
        profile = create_profile(user_id)

    memories = retrieve_memories(user_id, limit=5)
    memory_strings = [m["content"] for m in memories]

    return {
        "user_profile": profile,
        "relevant_memories": memory_strings,
    }


def inject_context_node(state: AssistantState) -> dict:
    system_text = build_system_message(state)
    system_msg = SystemMessage(content=system_text)
    token_estimate = estimate_tokens([system_msg])
    return {
        "messages": [system_msg],
        "turn_count": 0,
        "tokens_used_estimate": token_estimate,
    }


def agent_node(state: AssistantState) -> dict:
    llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    llm_with_tools = llm.bind_tools(all_tools)

    new_turn = state["turn_count"] + 1
    new_iter = state["iteration_count"] + 1
    response = llm_with_tools.invoke(state["messages"])

    new_token_est = estimate_tokens(state["messages"] + [response])
    needs_sum = new_token_est > CONTEXT_WINDOW * SUMMARISE_THRESHOLD

    return {
        "messages": [response],
        "turn_count": new_turn,
        "iteration_count": new_iter,
        "tokens_used_estimate": new_token_est,
        "needs_summarisation": needs_sum,
    }


def check_memory_node(state: AssistantState) -> dict:
    return {}  # routing only — no state change


def summarise_node(state: AssistantState) -> dict:
    updated_msgs, updated_summary, did_summarise = maybe_summarise(
        state["messages"],
        state.get("summary_so_far") or "",
        context_window=CONTEXT_WINDOW,
        threshold=SUMMARISE_THRESHOLD,
    )
    if did_summarise:
        console.print("[dim yellow]  ↻ Summarised older conversation turns[/dim yellow]")
    return {
        "messages": updated_msgs,
        "summary_so_far": updated_summary,
        "needs_summarisation": False,
        "tokens_used_estimate": estimate_tokens(updated_msgs),
    }


def save_memory_node(state: AssistantState) -> dict:
    user_id = state["user_id"]
    session_id = state["session_id"]
    messages = state["messages"]

    # Extract memories
    candidates = extract_memories(messages)
    saved = 0
    for mem in candidates:
        if not check_duplicate(user_id, mem["content"]):
            save_memory(user_id, session_id, mem["content"], mem["category"], mem["importance"])
            saved += 1

    if saved:
        console.print(f"[dim green]  ✓ Saved {saved} new memory/memories[/dim green]")

    # Consolidate if over threshold
    count = get_memory_count(user_id)
    if count > CONSOLIDATE_THRESHOLD:
        archived = consolidate_memories(user_id)
        console.print(f"[dim]  ↓ Consolidated memories — archived {archived} stale entries[/dim]")

    return {}

# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_agent(state: AssistantState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "check_memory"


def route_after_check(state: AssistantState) -> str:
    if state.get("needs_summarisation"):
        return "summarise"
    return "save_memory"

# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AssistantState)

    builder.add_node("load_memory", load_memory_node)
    builder.add_node("inject_context", inject_context_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_node("check_memory", check_memory_node)
    builder.add_node("summarise", summarise_node)
    builder.add_node("save_memory", save_memory_node)

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "inject_context")
    builder.add_edge("inject_context", "agent")
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges("check_memory", route_after_check)
    builder.add_edge("summarise", "save_memory")
    builder.add_edge("save_memory", END)

    return builder.compile()

# ── Session end ───────────────────────────────────────────────────────────────

def end_session(state: AssistantState):
    from summariser import summarise_messages
    summary = summarise_messages(state["messages"])
    save_session(
        state["session_id"],
        state["user_id"],
        summary,
        state["turn_count"]
    )
    console.print(Panel(
        f"Session saved.\nTurns: {state['turn_count']} | "
        f"Tokens (est): {state['tokens_used_estimate']:,}\n"
        f"Summary: {summary[:200]}...",
        title="Session Complete",
        border_style="dim"
    ))

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    init_db()
    graph = build_graph()

    global _current_user_id
    user_id = os.getenv("ASSISTANT_USER_ID", "default_user")
    _current_user_id = user_id
    session_id = str(uuid.uuid4())

    console.print(f"\n[bold blue]Personal Assistant[/bold blue]")
    console.print(f"User: {user_id} | Session: {session_id[:8]}...")
    console.print("Commands: [bold]/exit[/bold] [bold]/memories[/bold] [bold]/graph[/bold]\n")

    # Build initial state (no messages yet — load_memory and inject_context add them)
    current_state: AssistantState = {
        "messages": [],
        "session_id": session_id,
        "user_id": user_id,
        "turn_count": 0,
        "needs_summarisation": False,
        "summary_so_far": None,
        "tokens_used_estimate": 0,
        "user_profile": None,
        "relevant_memories": [],
        "iteration_count": 0,
        "max_iterations": 10,
    }

    # Bootstrap: run memory and context nodes directly as functions
    memory_updates = load_memory_node(current_state)
    current_state["user_profile"] = memory_updates["user_profile"]
    current_state["relevant_memories"] = current_state["relevant_memories"] + memory_updates["relevant_memories"]
    context_updates = inject_context_node(current_state)
    current_state["messages"] = current_state["messages"] + context_updates["messages"]
    current_state["turn_count"] = context_updates["turn_count"]
    current_state["tokens_used_estimate"] = context_updates["tokens_used_estimate"]

    # Show welcome context
    profile = current_state.get("user_profile") or {}
    name = profile.get("name") or "there"
    memories = current_state.get("relevant_memories", [])
    if memories:
        console.print(f"[dim]Welcome back, {name}! I remember:[/dim]")
        for m in memories[:3]:
            console.print(f"[dim]  - {m}[/dim]")
    else:
        console.print(f"[dim]Hello {name}! No prior memories — this may be your first session.[/dim]")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            end_session(current_state)
            break

        if not user_input:
            continue

        if user_input == "/exit":
            end_session(current_state)
            break

        if user_input == "/graph":
            graph.get_graph().print_ascii()
            continue

        if user_input == "/memories":
            mems = retrieve_memories(user_id, limit=10)
            console.print("[bold]Your memories:[/bold]")
            for m in mems:
                console.print(f"  [{m['category']} | {m['importance']}] {m['content']}")
            continue

        # Add user message and run one conversation turn
        new_messages = current_state["messages"] + [HumanMessage(content=user_input)]
        turn_state = {**current_state, "messages": new_messages, "iteration_count": 0}

        console.print("[yellow]Thinking...[/yellow]")
        result = graph.invoke(turn_state)
        current_state = result

        # Find last AIMessage with content
        final_answer = ""
        for msg in reversed(current_state["messages"]):
            print(f"  {type(msg).__name__}: content={repr(str(msg.content)[:80])} tool_calls={getattr(msg, 'tool_calls', [])}")
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break

        console.print(Panel(final_answer, title="Assistant", border_style="green"))
        console.print(f"[dim]Tokens: ~{current_state['tokens_used_estimate']:,} | Turn: {current_state['turn_count']}[/dim]\n")


if __name__ == "__main__":
    main()