"""
Experiment 3 — Golden trajectory recording and diff.
"""

import sys
import os
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "section_08_reliability"))

from eval_cases import create_knowledge_base, KB_DIR
import tools as tools_mod
tools_mod.DOCS_DIR = KB_DIR
create_knowledge_base()

from golden_trajectories import GoldenTrajectoryStore
from graph import build_graph, make_initial_state

print("=== Experiment 3 — Golden Trajectories ===\n")

golden_store = GoldenTrajectoryStore("../data/eval_results.db")

TEST_QUERIES = [
    ("G01", "What is the Project Atlas budget?"),
    ("G02", "Who is the tech lead?"),
    ("G03", "What database was chosen?"),
    ("G04", "When are the standups?"),
    ("G05", "What were the Q3 retrospective action items?"),
]

# ── Phase 1: Record goldens ────────────────────────────────────────────────────

print("Phase 1: Running 5 queries and recording golden trajectories")

def run_and_get_trajectory(query: str) -> tuple[list[str], str]:
    graph = build_graph()
    initial = make_initial_state(query, max_iterations=8)
    result = graph.invoke(initial)
    # Derive trajectory from message types
    from langchain_core.messages import AIMessage, ToolMessage
    trajectory = []
    seen_agent = False
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage) and not seen_agent:
            trajectory.append("agent")
            seen_agent = True
        elif isinstance(msg, ToolMessage):
            trajectory.append("tools")
            seen_agent = False
    trajectory = ["input_guardrail", "check_iteration_limit"] + trajectory + ["validate_output", "deliver_answer"]
    return trajectory, result.get("status", "unknown")

for case_id, query in TEST_QUERIES:
    traj, status = run_and_get_trajectory(query)
    golden_store.record_golden(case_id, str(uuid.uuid4()), traj, notes="initial recording")
    print(f"  {case_id}: recorded {len(traj)}-node trajectory (status={status})")

# ── Phase 2: Re-run same queries and diff ─────────────────────────────────────

print("\nPhase 2: Re-running same queries and diffing against goldens")

for case_id, query in TEST_QUERIES:
    traj, _ = run_and_get_trajectory(query)
    diff = golden_store.diff_against_golden(case_id, traj)
    match = "✓ matches" if diff["matches_golden"] else "△ differs"
    added_str   = f" +{diff['nodes_added']}"   if diff["nodes_added"]   else ""
    removed_str = f" -{diff['nodes_removed']}" if diff["nodes_removed"] else ""
    print(f"  {case_id}: {match}{added_str}{removed_str}")

# ── Phase 3: Deliberate breaking change — add a mock extra node ───────────────

print("\nPhase 3: Simulate a breaking change (extra node in trajectory)")

for case_id, query in TEST_QUERIES[:2]:
    traj, _ = run_and_get_trajectory(query)
    modified_traj = traj + ["new_monitoring_node"]  # simulate new node added
    diff = golden_store.diff_against_golden(case_id, modified_traj)
    print(f"  {case_id}: nodes_added = {diff['nodes_added']} | matches_golden = {diff['matches_golden']}")

print("\n  → Update goldens after verifying the new node is intentional:")
for case_id, query in TEST_QUERIES[:2]:
    traj, _ = run_and_get_trajectory(query)
    modified_traj = traj + ["new_monitoring_node"]
    golden_store.record_golden(case_id, str(uuid.uuid4()), modified_traj, notes="after monitoring node added")
    diff = golden_store.diff_against_golden(case_id, modified_traj)
    print(f"  {case_id}: matches_golden after update = {diff['matches_golden']}")

print("\nExperiment 3 complete.")