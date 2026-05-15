import sys
import os
sys.path.append("..")

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import CodeReviewState
from nodes import (
    read_files_node,
    analyse_files_node,
    generate_changes_node,
    select_next_change_node,
    format_for_review_node,
    apply_or_skip_node,
    generate_summary_node,
    route_after_select,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DB = os.path.join(DATA_DIR, "checkpoints.db")
os.makedirs(DATA_DIR, exist_ok=True)


def human_review_node(state):
    """Passthrough node — decision is injected via graph.update_state()."""
    return {}


def build_graph(checkpointer=None):
    builder = StateGraph(CodeReviewState)

    # Add all nodes
    builder.add_node("read_files", read_files_node)
    builder.add_node("analyse_files", analyse_files_node)
    builder.add_node("generate_changes", generate_changes_node)
    builder.add_node("select_next_change", select_next_change_node)
    builder.add_node("format_for_review", format_for_review_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("apply_or_skip", apply_or_skip_node)
    builder.add_node("generate_summary", generate_summary_node)

    # Linear setup phase
    builder.add_edge(START, "read_files")
    builder.add_edge("read_files", "analyse_files")
    builder.add_edge("analyse_files", "generate_changes")
    builder.add_edge("generate_changes", "select_next_change")

    # Review loop
    builder.add_conditional_edges(
        "select_next_change",
        route_after_select,
        {
            "format_for_review": "format_for_review",
            "generate_summary": "generate_summary",
        }
    )
    builder.add_edge("format_for_review", "human_review")
    builder.add_edge("human_review", "apply_or_skip")
    builder.add_edge("apply_or_skip", "select_next_change")
    builder.add_edge("generate_summary", END)

    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
        compile_kwargs["interrupt_before"] = ["human_review"]

    return builder.compile(**compile_kwargs)


def get_checkpointer():
    return SqliteSaver.from_conn_string(CHECKPOINT_DB)