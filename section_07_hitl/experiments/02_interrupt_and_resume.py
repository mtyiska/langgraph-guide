import sys
import os
sys.path.append("../..")

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "exp02_checkpoints.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


class ActionState(TypedDict):
    action: str
    proposed_result: Optional[str]
    approved: Optional[bool]
    skip_reason: Optional[str]
    execution_result: Optional[str]


def propose_action_node(state: ActionState) -> dict:
    proposed = f"[Proposed] Run: {state['action'].upper()} — estimated impact: HIGH"
    print(f"  [propose_action] {proposed}")
    return {"proposed_result": proposed}


def execute_action_node(state: ActionState) -> dict:
    if state.get("approved") is False:
        print(f"  [execute_action] SKIPPED — reason: {state.get('skip_reason', 'rejected')}")
        return {"execution_result": f"SKIPPED: {state.get('skip_reason', 'rejected')}"}

    if state.get("approved") is True:
        edited = state.get("edited_result")
        result = edited if edited else state["proposed_result"]
        print(f"  [execute_action] EXECUTED — {result}")
        return {"execution_result": f"APPLIED: {result}"}

    # No decision — defensive skip
    print("  [execute_action] No decision found — skipping.")
    return {"execution_result": "SKIPPED: no decision"}


builder = StateGraph(ActionState)
builder.add_node("propose_action", propose_action_node)
builder.add_node("execute_action", execute_action_node)
builder.add_edge(START, "propose_action")
builder.add_edge("propose_action", "execute_action")
builder.add_edge("execute_action", END)

print("=== Experiment 2 — Interrupt and Resume ===\n")


def run_scenario(label: str, decision_updates: dict, thread_suffix: str):
    db = DB_PATH.replace(".db", f"_{thread_suffix}.db")
    thread_id = f"exp02-{thread_suffix}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"--- Scenario: {label} ---")

    with SqliteSaver.from_conn_string(db) as checkpointer:
        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["execute_action"],
        )

        # Initial invocation — stops before execute_action
        graph.invoke(
            {"action": "delete all temp files", "approved": None, "proposed_result": None,
             "skip_reason": None, "execution_result": None},
            config=config,
        )

        # Check state after interrupt
        state = graph.get_state(config)
        print(f"  Paused. Next: {state.next}")
        print(f"  Proposed: {state.values.get('proposed_result')}")
        assert "execute_action" in state.next, "Should be paused before execute_action"

        # Inject human decision
        graph.update_state(config, decision_updates, as_node="human_review")

        # Resume
        result = graph.invoke(None, config)
        print(f"  Result: {result['execution_result']}\n")

        # Verify no more pending nodes
        final_state = graph.get_state(config)
        assert not final_state.next, "Graph should be complete"

        # Show audit trail
        history = list(graph.get_state_history(config))
        sources = [s.metadata.get("source", "?") for s in reversed(history)]
        print(f"  Checkpoint sources (audit trail): {sources}\n")


run_scenario(
    label="Approve",
    decision_updates={"approved": True},
    thread_suffix="approve",
)

run_scenario(
    label="Reject",
    decision_updates={"approved": False, "skip_reason": "too risky right now"},
    thread_suffix="reject",
)

run_scenario(
    label="Edit before applying",
    decision_updates={
        "approved": True,
        "edited_result": "[Edited by human] Run: DELETE TEMP — restricted to /tmp only",
    },
    thread_suffix="edit",
)

print("All three resume paths verified.")