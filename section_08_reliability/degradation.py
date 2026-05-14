import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

import sys
sys.path.append("..")
from config import PRIMARY_MODEL, OLLAMA_BASE_URL
from state import ResearchState
from validators import parse_tool_response, ToolSuccess

logger = logging.getLogger(__name__)

primary_llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

DEGRADATION_SYSTEM = (
    "You produce honest partial answers when a research process is incomplete. "
    "You never fabricate information to fill gaps. "
    "You clearly distinguish what was found from what could not be found."
)

DEGRADATION_PROMPT_TEMPLATE = """The research process encountered a problem and could not complete.

Failure reason: {failure_reason}

Partial findings gathered before failure:
{partial_findings}

Original question: {original_question}

Produce a response that:
1. Answers as much of the question as the partial findings allow
2. Clearly states what could not be answered and why
3. Does NOT fabricate information to fill gaps
4. Is concise — do not pad the answer

If no useful findings were gathered, say so honestly."""


def determine_failure_reason(state: ResearchState) -> str:
    if state.get("loop_detected"):
        return (
            f"Agent exceeded iteration limit. "
            f"{state.get('loop_reason', 'Repeated tool calls detected.')}"
        )
    if not state.get("output_valid", True):
        return (
            f"Output validation failed after retry. "
            f"Error: {state.get('validation_error', 'unknown')}"
        )
    if state.get("guardrail_triggered"):
        return "Request was outside the agent's permitted scope."
    if state.get("injection_detected"):
        return (
            f"Potential prompt injection detected in retrieved content. "
            f"Reason: {state.get('injection_reason', 'unknown')}"
        )
    return state.get("failure_reason", "Unknown failure — check logs.")


def extract_partial_findings(state: ResearchState) -> str:
    tool_messages = [
        m for m in state.get("messages", [])
        if isinstance(m, ToolMessage)
    ]

    if not tool_messages:
        return "No findings were gathered before the failure."

    findings = []
    for msg in tool_messages[-5:]:
        try:
            parsed = parse_tool_response(str(msg.content))
            if isinstance(parsed, ToolSuccess):
                findings.append(parsed.result[:600])
            else:
                findings.append(f"[Tool error: {parsed.message}]")
        except Exception:
            findings.append(str(msg.content)[:600])

    return "\n\n---\n\n".join(findings)


def graceful_degradation_node(state: ResearchState) -> dict:
    """
    Assemble whatever was gathered before failure into a partial answer.
    Always produces a response — never crashes or returns empty.
    """
    failure_reason = determine_failure_reason(state)
    partial_findings = extract_partial_findings(state)

    # Find the original user question
    original_question = "Unknown question."
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            original_question = msg.content
            break

    prompt = DEGRADATION_PROMPT_TEMPLATE.format(
        failure_reason=failure_reason,
        partial_findings=partial_findings,
        original_question=original_question,
    )

    try:
        response = primary_llm.invoke([
            SystemMessage(content=DEGRADATION_SYSTEM),
            HumanMessage(content=prompt),
        ])
        answer = response.content
    except Exception as e:
        # Last-resort fallback — the degradation LLM call itself failed
        logger.error(f"Degradation LLM call failed: {e}")
        answer = (
            f"The research process failed and the answer could not be produced.\n\n"
            f"Failure reason: {failure_reason}\n\n"
            f"Partial findings:\n{partial_findings[:500]}"
        )

    return {
        "final_answer": answer,
        "answer_complete": False,
        "failure_reason": failure_reason,
        "status": "degraded",
    }