import sys
sys.path.append("..")

from langgraph.graph import StateGraph, START, END

from state import ResearchState
from tools import ALL_TOOLS
from guardrails import input_guardrail_node, route_after_guardrail
from nodes import (
    check_iteration_limit_node,
    route_after_limit_check,
    agent_node,
    route_after_agent,
    tools_node_fn,
    validate_output_node,
    route_after_validation,
    injection_screen_node,
    route_after_injection_screen,
    deliver_answer_node,
)
from degradation import graceful_degradation_node
from injection_defence import RESEARCH_SYSTEM


def build_graph():
    tool_registry = {t.name: t for t in ALL_TOOLS}

    # Bind tool_registry into closures
    def _agent_node(state):
        return agent_node(state, ALL_TOOLS)

    def _tools_node(state):
        return tools_node_fn(state, tool_registry)

    builder = StateGraph(ResearchState)

    builder.add_node("input_guardrail", input_guardrail_node)
    builder.add_node("check_iteration_limit", check_iteration_limit_node)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", _tools_node)
    builder.add_node("validate_output", validate_output_node)
    builder.add_node("injection_screen", injection_screen_node)
    builder.add_node("deliver_answer", deliver_answer_node)
    builder.add_node("graceful_degradation", graceful_degradation_node)

    # Edges
    builder.add_edge(START, "input_guardrail")
    builder.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {"deliver_answer": "deliver_answer", "check_iteration_limit": "check_iteration_limit"},
    )
    builder.add_conditional_edges(
        "check_iteration_limit",
        route_after_limit_check,
        {"graceful_degradation": "graceful_degradation", "agent": "agent"},
    )
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"check_iteration_limit": "check_iteration_limit", "validate_output": "validate_output"},
    )
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges(
        "validate_output",
        route_after_validation,
        {"graceful_degradation": "graceful_degradation", "injection_screen": "injection_screen"},
    )
    builder.add_conditional_edges(
        "injection_screen",
        route_after_injection_screen,
        {"graceful_degradation": "graceful_degradation", "deliver_answer": "deliver_answer"},
    )
    builder.add_edge("graceful_degradation", "deliver_answer")
    builder.add_edge("deliver_answer", END)

    return builder.compile()


def make_initial_state(user_query: str, max_iterations: int = 12) -> ResearchState:
    from langchain_core.messages import HumanMessage, SystemMessage
    return {
        "messages": [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=user_query),
        ],
        "system_prompt": RESEARCH_SYSTEM,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "loop_detected": False,
        "loop_reason": None,
        "guardrail_triggered": False,
        "guardrail_decision": None,
        "raw_answer": None,
        "validated_answer": None,
        "output_valid": False,
        "validation_error": None,
        "injection_detected": False,
        "injection_reason": None,
        "final_answer": None,
        "answer_complete": True,
        "failure_reason": None,
        "status": "running",
    }