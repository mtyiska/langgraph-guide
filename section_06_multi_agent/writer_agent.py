import sys
import os
import json
sys.path.append("..")

from typing import TypedDict, Annotated, Optional
from operator import add
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from shared_state import SharedState
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename: str) -> str:
    with open(os.path.join(PROMPTS_DIR, filename)) as f:
        return f.read()


# ── Writer internal state ─────────────────────────────────────────────────────

class WriterInternalState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    current_task: str
    research_input: str
    draft: Optional[str]
    revision_notes: Optional[str]
    needs_revision: bool
    word_count: int
    iteration_count: int
    # Passthrough fields
    original_request: str
    worker_results: Annotated[list[dict], add]
    status: str
    next_worker: Optional[str]
    last_worker: Optional[str]
    final_output: Optional[str]


def build_writer_graph():
    llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)
    reviewer_llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    writer_prompt = load_prompt("writer.txt")
    reviewer_prompt = load_prompt("reviewer.txt")

    def parse_task_node(state: WriterInternalState) -> dict:
        task = state.get("current_task", "")
        research = state.get("research_input", "")

        # Extract research from worker_results if not set directly
        if not research and state.get("worker_results"):
            research_parts = []
            for r in state["worker_results"]:
                if r.get("worker") == "researcher":
                    research_parts.append(r.get("result", ""))
            research = "\n\n---\n\n".join(research_parts)

        system_msg = SystemMessage(content=writer_prompt)
        human_msg = HumanMessage(content=(
            f"Writing task: {task}\n\n"
            f"Original request: {state['original_request']}\n\n"
            f"Research provided:\n{research}"
        ))

        return {
            "messages": [system_msg, human_msg],
            "research_input": research,
            "iteration_count": 0,
            "needs_revision": False,
        }

    def plan_structure_node(state: WriterInternalState) -> dict:
        plan_prompt = HumanMessage(content=(
            "Before writing, briefly outline the structure you will use. "
            "List the main sections and what each will cover. "
            "Keep this outline concise — 5 to 8 bullet points maximum."
        ))
        response = llm.invoke(state["messages"] + [plan_prompt])
        return {
            "messages": [plan_prompt, response],
            "iteration_count": state["iteration_count"] + 1,
        }

    def draft_node(state: WriterInternalState) -> dict:
        draft_prompt = HumanMessage(content=(
            "Now write the full draft based on your outline and the research provided. "
            "Use only facts from the research. "
            "Include the source filename in parentheses after each key claim."
        ))
        response = llm.invoke(state["messages"] + [draft_prompt])
        draft_text = response.content
        word_count = len(draft_text.split())

        return {
            "messages": [draft_prompt, response],
            "draft": draft_text,
            "word_count": word_count,
            "iteration_count": state["iteration_count"] + 1,
        }

    def review_node(state: WriterInternalState) -> dict:
        review_messages = [
            SystemMessage(content=reviewer_prompt),
            HumanMessage(content=(
                f"Research provided:\n{state['research_input']}\n\n"
                f"Draft to review:\n{state['draft']}"
            ))
        ]
        response = reviewer_llm.invoke(review_messages)

        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            review_result = json.loads(raw.strip())
            needs_revision = review_result.get("needs_revision", False)
            issues = review_result.get("issues", [])
            revised_sections = review_result.get("revised_sections", {})
            notes = "; ".join(issues) if issues else "No issues found."
        except Exception:
            needs_revision = False
            notes = "Review parsing failed — proceeding with draft."
            revised_sections = {}

        return {
            "needs_revision": needs_revision,
            "revision_notes": notes,
            "iteration_count": state["iteration_count"] + 1,
        }

    def revise_node(state: WriterInternalState) -> dict:
        revise_prompt = HumanMessage(content=(
            f"Revise the draft to address these issues:\n{state['revision_notes']}\n\n"
            f"Current draft:\n{state['draft']}\n\n"
            "Fix only the identified issues. Do not rewrite sections that are fine."
        ))
        response = llm.invoke([
            SystemMessage(content=load_prompt("writer.txt")),
            revise_prompt
        ])
        revised = response.content
        return {
            "messages": [revise_prompt, response],
            "draft": revised,
            "word_count": len(revised.split()),
            "needs_revision": False,
            "iteration_count": state["iteration_count"] + 1,
        }

    def format_output_node(state: WriterInternalState) -> dict:
        result_record = {
            "worker": "writer",
            "task": state.get("current_task", ""),
            "result": state["draft"],
            "word_count": state["word_count"],
            "revision_notes": state.get("revision_notes", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }
        return {
            "worker_results": [result_record],
            "last_worker": "writer",
        }

    # ── Routing ───────────────────────────────────────────────────────────────

    def route_after_review(state: WriterInternalState) -> str:
        if state["needs_revision"] and state["iteration_count"] < 6:
            return "revise"
        return "format_output"

    # ── Build graph ───────────────────────────────────────────────────────────

    builder = StateGraph(WriterInternalState)

    builder.add_node("parse_task", parse_task_node)
    builder.add_node("plan_structure", plan_structure_node)
    builder.add_node("draft", draft_node)
    builder.add_node("review", review_node)
    builder.add_node("revise", revise_node)
    builder.add_node("format_output", format_output_node)

    builder.add_edge(START, "parse_task")
    builder.add_edge("parse_task", "plan_structure")
    builder.add_edge("plan_structure", "draft")
    builder.add_edge("draft", "review")
    builder.add_conditional_edges("review", route_after_review)
    builder.add_edge("revise", "format_output")
    builder.add_edge("format_output", END)

    return builder.compile()