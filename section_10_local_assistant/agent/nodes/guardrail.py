import json
import logging
from typing import Optional
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from agent.state import AssistantState
from config import FAST_MODEL, OLLAMA_BASE_URL, PROMPTS_DIR

logger = logging.getLogger(__name__)
_fast_llm = None


def _llm():
    global _fast_llm
    if _fast_llm is None:
        _fast_llm = ChatOllama(model=FAST_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    return _fast_llm


class GuardrailDecision(BaseModel):
    decision:      str
    reason:        str
    safe_response: Optional[str] = None


def input_guardrail_node(state: AssistantState) -> dict:
    user_input = ""
    for msg in reversed(state.get("messages", [])):
        if msg.__class__.__name__ == "HumanMessage":
            user_input = msg.content
            break

    system = (PROMPTS_DIR / "guardrail.txt").read_text()
    prompt = f'Evaluate this user request: "{user_input}"'

    try:
        response = _llm().invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(l for l in lines if not l.strip().startswith("```"))
        start = raw.find("{"); end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]
        data     = json.loads(raw)
        decision = GuardrailDecision.model_validate(data)
    except Exception as e:
        logger.warning(f"Guardrail parse failed: {e} — failing open")
        decision = GuardrailDecision(decision="allow", reason="parse failure", safe_response=None)

    if decision.decision in ("block", "clarify"):
        return {
            "guardrail_triggered": True,
            "final_response": decision.safe_response or "I can't help with that request.",
            "status": "blocked",
            "output_valid": True,
        }

    return {"guardrail_triggered": False}


def route_after_guardrail(state: AssistantState) -> str:
    if state.get("guardrail_triggered"):
        return "deliver_answer"
    return "memory_load"