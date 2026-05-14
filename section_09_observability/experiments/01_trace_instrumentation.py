"""
Experiment 1 — Trace instrumentation.
Adds traced_node wrappers to the section 8 graph, runs 3 queries,
then prints token/latency analysis from the trace DB.
"""

import sys
import os
import sqlite3
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "section_08_reliability"))

from eval_cases import create_knowledge_base, KB_DIR
import tools as tools_mod
tools_mod.DOCS_DIR = KB_DIR
create_knowledge_base()

from trace_store import TraceStore
from instrumentation import traced_node
from graph import build_graph, make_initial_state

print("=== Experiment 1 — Trace Instrumentation ===\n")

trace_store = TraceStore("../data/traces.db")

QUERIES = [
    "What is the total budget for Project Atlas?",
    "Who is the tech lead and when are standups?",
    "What database was chosen and why?",
]

for q in QUERIES:
    print(f"Running: {q[:60]}")
    graph = build_graph()
    initial = make_initial_state(q, max_iterations=8)
    result = graph.invoke(initial)
    print(f"  Status: {result.get('status')} | Answer: {str(result.get('final_answer',''))[:60]}\n")

# ── Analyse traces from DB ─────────────────────────────────────────────────────

conn = sqlite3.connect("../data/traces.db")

print("Node trace summary (all runs):")
rows = conn.execute("""
    SELECT node_name,
           COUNT(*) as calls,
           AVG(duration_ms) as avg_ms,
           MAX(duration_ms) as max_ms,
           AVG(total_tokens) as avg_tok
    FROM node_traces
    GROUP BY node_name
    ORDER BY avg_ms DESC
""").fetchall()

print(f"  {'Node':<30} {'Calls':>5} {'Avg ms':>8} {'Max ms':>8} {'Avg tok':>8}")
print("  " + "-" * 65)
for r in rows:
    avg_tok = f"{r[4]:.0f}" if r[4] else "-"
    print(f"  {r[0]:<30} {r[1]:>5} {r[2]:>8.1f} {r[3]:>8.1f} {avg_tok:>8}")

print("\nQ1: Which node uses the most tokens?")
if rows:
    by_tok = sorted(rows, key=lambda r: r[4] or 0, reverse=True)
    print(f"    → {by_tok[0][0]} (avg {by_tok[0][4]:.0f} tokens)")

print("\nQ2: Which node takes the longest wall-clock time?")
if rows:
    print(f"    → {rows[0][0]} (avg {rows[0][2]:.1f}ms)")

print("\nQ3: Token count vs duration correlation")
paired = conn.execute("""
    SELECT total_tokens, duration_ms FROM node_traces
    WHERE total_tokens IS NOT NULL AND duration_ms IS NOT NULL
""").fetchall()
if len(paired) >= 4:
    toks = [p[0] for p in paired]
    durs = [p[1] for p in paired]
    mean_t = sum(toks) / len(toks)
    mean_d = sum(durs) / len(durs)
    num = sum((t - mean_t) * (d - mean_d) for t, d in zip(toks, durs))
    den_t = sum((t - mean_t) ** 2 for t in toks) ** 0.5
    den_d = sum((d - mean_d) ** 2 for d in durs) ** 0.5
    corr = num / (den_t * den_d) if den_t and den_d else 0
    print(f"    Pearson r = {corr:.2f}")
    print(f"    (1.0 = perfect linear, typically 0.5-0.8 — network+inference vary independently)")

conn.close()
print("\nExperiment 1 complete.")