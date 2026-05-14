import sys
import os
sys.path.append("..")

from typing import Literal, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from shared_state import SharedState, build_supervisor_context
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename: str) -> str:
    with open(os.path.join(PROMPTS_DIR, filename)) as f:
        return f.read()


@tool
def route_to_worker(
    worker: Literal["researcher", "writer", "FINISH"],
    task_description: str,
    context_for_worker: str
) -> str:
    """
    Route the next step to a specific worker agent or finish the pipeline.

    Args:
        worker: Which worker to call next. Use FINISH when the draft is complete.
        task_description: Brief description of what this worker should accomplish.
        context_for_worker: Full instructions and context the worker needs to do the task.
                            For the writer, include ALL research findings here.
    """
    return f"Routing to {worker}: {task_description}"


def build_supervisor_node(fast_model: str = None):
    model = fast_model or PRIMARY_MODEL
    llm = ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0)
    llm_with_tools = llm.bind_tools([route_to_worker])
    supervisor_prompt = load_prompt("supervisor.txt")

    def supervisor_node(state: SharedState) -> dict:
        new_iter = state["iteration_count"] + 1

        if new_iter >= state["max_iterations"]:
            last_result = ""
            if state["worker_results"]:
                last_result = str(state["worker_results"][-1].get("result", ""))
            return {
                "status": "complete",
                "final_output": last_result or "Pipeline reached iteration limit without completing.",
                "iteration_count": new_iter,
                "next_worker": None,
            }

        context = build_supervisor_context(state)
        messages = [
            SystemMessage(content=supervisor_prompt),
            HumanMessage(content=context),
        ]

        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            # Model produced text instead of a tool call — extract routing from content
            content = response.content.lower()
            if "finish" in content or "complete" in content:
                worker = "FINISH"
                task = "Pipeline complete"
                context_for_worker = ""
            elif "writer" in content:
                worker = "writer"
                task = "Write content from research"
                context_for_worker = context
            else:
                worker = "researcher"
                task = state["original_request"]
                context_for_worker = state["original_request"]
        else:
            tc = response.tool_calls[0]
            worker = tc["args"].get("worker", "researcher")
            task = tc["args"].get("task_description", "")
            context_for_worker = tc["args"].get("context_for_worker", "")

        if worker == "FINISH":
            final_result = ""
            for r in reversed(state["worker_results"]):
                if r.get("worker") == "writer":
                    final_result = r.get("result", "")
                    break
            if not final_result and state["worker_results"]:
                final_result = state["worker_results"][-1].get("result", "")

            return {
                "status": "complete",
                "final_output": final_result,
                "iteration_count": new_iter,
                "next_worker": None,
                "last_worker": "supervisor",
            }

        return {
            "current_task": context_for_worker,
            "next_worker": worker,
            "last_worker": "supervisor",
            "iteration_count": new_iter,
            "status": "running",
        }

    return supervisor_node


def route_after_supervisor(state: SharedState) -> str:
    if state["status"] == "complete":
        return "END"
    return state.get("next_worker", "END")