import uuid
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from agent.state import AssistantState
from agent.nodes.guardrail     import input_guardrail_node, route_after_guardrail
from agent.nodes.memory        import memory_load_node, memory_save_node
from agent.nodes.supervisor    import supervisor_node, route_after_supervisor
from agent.nodes.direct_answer import direct_answer_node
from agent.nodes.output        import (
    validate_output_node,
    graceful_degradation_node,
    deliver_answer_node,
)
from agent.subgraphs.researcher   import build_researcher_subgraph
from agent.subgraphs.writer       import build_writer_subgraph
from agent.subgraphs.task_manager import build_task_manager_subgraph
from infrastructure.trace_store   import TraceStore, traced_node
from config import CHECKPOINTS_DB, ENABLE_TRACING, ENABLE_MULTI_AGENT

logger = logging.getLogger(__name__)


def build_graph(trace_store: TraceStore | None = None) -> tuple:
    run_context = {
        "run_id":       "",
        "thread_id":    "",
        "node_index":   0,
        "trajectory":   [],
        "total_tokens": 0,
    }

    def wrap(fn):
        if ENABLE_TRACING and trace_store is not None:
            return traced_node(fn, trace_store, run_context)
        return fn

    builder = StateGraph(AssistantState)

    # Infrastructure nodes
    builder.add_node("input_guardrail", wrap(input_guardrail_node))
    builder.add_node("memory_load",     wrap(memory_load_node))
    builder.add_node("supervisor",      wrap(supervisor_node))
    builder.add_node("direct_answer",   wrap(direct_answer_node))
    builder.add_node("validate_output", wrap(validate_output_node))
    builder.add_node("graceful_degradation", wrap(graceful_degradation_node))
    builder.add_node("deliver_answer",  wrap(deliver_answer_node))
    builder.add_node("memory_save",     wrap(memory_save_node))

    # Subgraph nodes (not wrapped — they have internal nodes)
    if ENABLE_MULTI_AGENT:
        builder.add_node("researcher",   build_researcher_subgraph())
        builder.add_node("writer",       build_writer_subgraph())
        builder.add_node("task_manager", build_task_manager_subgraph())

    # Wiring
    builder.add_edge(START, "input_guardrail")
    builder.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {"memory_load": "memory_load", "deliver_answer": "deliver_answer"},
    )
    builder.add_edge("memory_load", "supervisor")

    if ENABLE_MULTI_AGENT:
        builder.add_conditional_edges(
            "supervisor",
            route_after_supervisor,
            {
                "researcher":          "researcher",
                "writer":              "writer",
                "task_manager":        "task_manager",
                "direct_answer":       "direct_answer",
                "graceful_degradation":"graceful_degradation",
            },
        )
        for worker in ["researcher", "writer", "task_manager"]:
            builder.add_edge(worker, "validate_output")
    else:
        builder.add_conditional_edges(
            "supervisor",
            route_after_supervisor,
            {"direct_answer": "direct_answer", "graceful_degradation": "graceful_degradation"},
        )

    builder.add_edge("direct_answer",        "validate_output")
    builder.add_edge("validate_output",      "deliver_answer")
    builder.add_edge("graceful_degradation", "deliver_answer")
    builder.add_edge("deliver_answer",       "memory_save")
    builder.add_edge("memory_save",          END)

    
    conn = sqlite3.connect(CHECKPOINTS_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)

    return graph, run_context