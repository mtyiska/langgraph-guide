import json
import logging
import hashlib
from typing import Optional

from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ToolMessage
)
from langchain_ollama import ChatOllama

import sys
sys.path.append("..")
from config import PRIMARY_MODEL, OLLAMA_BASE_URL
from state import ResearchState
from validators import (
    ResearchAnswer, extract_json_from_text, parse_tool_response, ToolError
)
from injection_defence import screen_for_injection, inject_security_addendum
from degradation import graceful_degradation_node   # re-exported for graph

logger = logging.getLogger(__name__)

primary_llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

RESEARCH_SYSTEM = inject_security_addendum("""
You are a local document research assistant. You answer questions using only
the documents available via your tools. You do not use training knowledge
for domain-specific facts — only retrieved document content.

TOOL USE:
1. Call list_available_documents first if you don't know what files exist.
2. Use search_documents for keyword search across all files.
3. Use read_document to get full contents of a specific file.
4. Use calculate for arithmetic.

TOOL RESPONSE FORMAT:
Tools return JSON with "status": "success" or "status": "error".
On error, read the "recoverable" field:
- true: try a different approach based on "suggestion"
- false: note the failure and continue with what you have

Never retry the exact same call that returned an error.

OUTPUT FORMAT:
When you have enough information, respond with ONLY a JSON object (no other text):
{
  "answer": "Your complete answer here — be descriptive, at least 30 words.",
  "confidence": "high" | "medium" | "low",
  "citations": [
    {
      "source_path": "budget_summary.md",
      "excerpt": "Total Approved Budget: $180,000",
      "relevance": "This document contains the exact budget amount for Project Orion"
    }
  ],
  "gaps": [],
  "answer_complete": true
}

CRITICAL RULES:
- If confidence is "high", you MUST include at least one citation
- If confidence is "medium" or "low", citations are optional but recommended
- Always include the source_path (just the filename, e.g., "budget_summary.md")
- Always include a direct excerpt from that source
- Make your answer detailed and at least 30 words long
""")


# ── Iteration limit ────────────────────────────────────────────────────────────

def check_iteration_limit_node(state: ResearchState) -> dict:
    """Increment iteration counter and detect loops."""
    count = state["iteration_count"] + 1
    limit = state["max_iterations"]

    # Hard limit
    if count > limit:
        return {
            "iteration_count": count,
            "loop_detected": True,
            "loop_reason": f"Reached maximum iterations ({limit}).",
            "status": "failed",
        }

    # Repetition detection
    if _detect_repeated_tool_calls(state):
        return {
            "iteration_count": count,
            "loop_detected": True,
            "loop_reason": "Agent is repeating the same tool calls. Stopping early.",
            "status": "failed",
        }

    return {"iteration_count": count, "loop_detected": False}


def _detect_repeated_tool_calls(state: ResearchState) -> bool:
    messages = state.get("messages", [])

    # Collect tool call hashes from recent AI messages
    call_hashes = []
    for msg in messages[-16:]:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                call_str = json.dumps(
                    {"name": tc["name"], "args": tc["args"]}, sort_keys=True
                )
                call_hashes.append(hashlib.md5(call_str.encode()).hexdigest())

    if len(call_hashes) < 4:
        return False

    recent = set(call_hashes[-2:])
    earlier = call_hashes[:-2]
    return any(h in earlier for h in recent)


def route_after_limit_check(state: ResearchState) -> str:
    if state.get("loop_detected"):
        return "graceful_degradation"
    return "tools"  


# ── Agent node ─────────────────────────────────────────────────────────────────

def agent_node(state: ResearchState, tools) -> dict:
    """Main reasoning node. Calls LLM with tools bound."""
    llm_with_tools = primary_llm.bind_tools(tools)

    # Prepend system message if not already present
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=RESEARCH_SYSTEM)] + list(messages)

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def route_after_agent(state: ResearchState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "check_iteration_limit"
    return "validate_output"


# ── Tools node ─────────────────────────────────────────────────────────────────

def tools_node_fn(state: ResearchState, tool_registry: dict) -> dict:
    """Execute tool calls from the last AI message."""
    last = state["messages"][-1]
    tool_messages = []

    for tc in getattr(last, "tool_calls", []):
        name = tc["name"]
        args = tc["args"]
        tid = tc["id"]

        if name not in tool_registry:
            tool_messages.append(
                ToolMessage(
                    content=json.dumps({
                        "status": "error",
                        "error_type": "invalid_input",
                        "message": f"Tool '{name}' does not exist.",
                        "recoverable": True,
                        "suggestion": (
                            "Use only these tools: "
                            + ", ".join(tool_registry.keys())
                        ),
                    }),
                    tool_call_id=tid,
                )
            )
            continue

        try:
            result = tool_registry[name].invoke(args)
        except Exception as e:
            result = json.dumps({
                "status": "error",
                "error_type": "unknown",
                "message": str(e),
                "recoverable": True,
                "suggestion": "Try a different approach.",
            })

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tid)
        )

    return {"messages": tool_messages}


# ── Output validation ──────────────────────────────────────────────────────────

def validate_output_node(state: ResearchState) -> dict:
    """Validate the agent's final answer against ResearchAnswer schema."""
    raw = _extract_final_answer(state)

    if not raw:
        return {
            "output_valid": False,
            "raw_answer": "",
            "validation_error": "No answer text found in agent messages.",
        }

    first_error = None  # declare before the try block so it's always in scope

    # First attempt
    try:
        cleaned = extract_json_from_text(raw)
        validated = ResearchAnswer.model_validate_json(cleaned)
        return {
            "raw_answer": raw,
            "validated_answer": validated.model_dump(),
            "output_valid": True,
        }
    except Exception as e:
        first_error = e

    # One retry with correction prompt
    correction_prompt = (
        f"Your previous answer had validation errors: {first_error}\n\n"
        f"The answer must be valid JSON matching this schema:\n"
        f"{json.dumps(ResearchAnswer.model_json_schema(), indent=2)}\n\n"
        f"Previous answer:\n{raw}\n\n"
        "Produce a corrected version in valid JSON only. "
        "Fix the validation errors without changing factual content."
    )

    try:
        retry_response = primary_llm.invoke([
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=correction_prompt),
        ])
        cleaned = extract_json_from_text(retry_response.content)
        validated = ResearchAnswer.model_validate_json(cleaned)
        return {
            "raw_answer": retry_response.content,
            "validated_answer": validated.model_dump(),
            "output_valid": True,
        }
    except Exception as second_error:
        return {
            "output_valid": False,
            "raw_answer": raw,
            "validation_error": str(second_error),
        }

def route_after_validation(state: ResearchState) -> str:
    if not state.get("output_valid"):
        return "graceful_degradation"
    return "injection_screen"


def _extract_final_answer(state: ResearchState) -> str:
    """Pull the last AIMessage content that isn't a tool call."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            if not getattr(msg, "tool_calls", None):
                return msg.content
    return ""


# ── Injection screen ───────────────────────────────────────────────────────────

def injection_screen_node(state: ResearchState) -> dict:
    """Screen the validated answer for prompt injection artifacts."""
    answer_data = state.get("validated_answer", {})
    answer_text = answer_data.get("answer", state.get("raw_answer", ""))

    # Find original user query
    original_query = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break

    detected, reason = screen_for_injection(answer_text, original_query)

    return {
        "injection_detected": detected,
        "injection_reason": reason if detected else None,
    }


def route_after_injection_screen(state: ResearchState) -> str:
    if state.get("injection_detected"):
        return "graceful_degradation"
    return "deliver_answer"


# ── Deliver answer ─────────────────────────────────────────────────────────────

def deliver_answer_node(state: ResearchState) -> dict:
    """Format and deliver the final validated answer."""
    validated = state.get("validated_answer") or {}
    answer = validated.get("answer", state.get("final_answer", "No answer produced."))
    confidence = validated.get("confidence", "low")
    citations = validated.get("citations", [])
    gaps = validated.get("gaps", [])
    complete = validated.get("answer_complete", True)

    lines = [answer, ""]

    if citations:
        lines.append("**Sources:**")
        for c in citations:
            lines.append(f"  - `{c['source_path']}`: {c['excerpt'][:80]}...")

    if gaps:
        lines.append("\n**Could not find:**")
        for g in gaps:
            lines.append(f"  - {g}")

    if not complete:
        lines.append("\n*(Answer may be incomplete due to retrieval limitations.)*")

    return {
        "final_answer": "\n".join(lines),
        "answer_complete": complete,
        "status": "complete",
    }