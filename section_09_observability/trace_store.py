import sqlite3
import json
import os
import logging
from datetime import datetime
from trace_schema import NodeTrace, RunTrace

logger = logging.getLogger(__name__)


def truncate_state(state: dict, max_str_len: int = 300) -> dict:
    """Truncate long string values so state snapshots don't bloat the DB."""
    result = {}
    for k, v in state.items():
        if isinstance(v, list) and v and hasattr(v[0], 'content'):
            serialized = []
            for msg in v[:10]:
                serialized.append({
                    "type": type(msg).__name__,
                    "content": str(getattr(msg, 'content', ''))[:max_str_len]
                })
            if len(v) > 10:
                serialized.append(f"... [{len(v)} items total]")
            result[k] = serialized
        elif isinstance(v, str) and len(v) > max_str_len:
            result[k] = v[:max_str_len] + f"... [{len(v)} chars total]"
        elif isinstance(v, list) and len(v) > 10:
            result[k] = v[:10] + [f"... [{len(v)} items total]"]
        else:
            result[k] = v
    return result


class TraceStore:
    def __init__(self, db_path: str = "./data/traces.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS node_traces (
                trace_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                thread_id TEXT,
                node_name TEXT NOT NULL,
                node_index INTEGER,
                start_time TEXT,
                end_time TEXT,
                duration_ms REAL,
                input_snapshot TEXT,
                output_snapshot TEXT,
                model_name TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                tool_calls TEXT,
                status TEXT,
                error_message TEXT,
                retry_count INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_traces (
                run_id TEXT PRIMARY KEY,
                thread_id TEXT,
                start_time TEXT,
                end_time TEXT,
                duration_ms REAL,
                user_query TEXT,
                trajectory TEXT,
                total_tokens INTEGER,
                final_status TEXT,
                output_valid INTEGER,
                guardrail_triggered INTEGER,
                loop_detected INTEGER,
                degraded INTEGER,
                final_answer_preview TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_node_run_id ON node_traces(run_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_run_start ON run_traces(start_time)"
        )
        conn.commit()
        conn.close()

    def save_node_trace(self, trace: NodeTrace):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                INSERT INTO node_traces VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """,
                (
                    trace.trace_id,
                    trace.run_id,
                    trace.thread_id,
                    trace.node_name,
                    trace.node_index,
                    trace.start_time.isoformat(),
                    trace.end_time.isoformat() if trace.end_time else None,
                    trace.duration_ms,
                    json.dumps(truncate_state(trace.input_snapshot)),
                    json.dumps(truncate_state(trace.output_snapshot)),
                    trace.model_name,
                    trace.input_tokens,
                    trace.output_tokens,
                    trace.total_tokens,
                    json.dumps(trace.tool_calls),
                    trace.status,
                    trace.error_message,
                    trace.retry_count,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save node trace: {e}")

    def save_run_trace(self, trace: RunTrace):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                INSERT OR REPLACE INTO run_traces VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """,
                (
                    trace.run_id,
                    trace.thread_id,
                    trace.start_time.isoformat(),
                    trace.end_time.isoformat() if trace.end_time else None,
                    trace.duration_ms,
                    trace.user_query,
                    json.dumps(trace.trajectory),
                    trace.total_tokens,
                    trace.final_status,
                    int(trace.output_valid),
                    int(trace.guardrail_triggered),
                    int(trace.loop_detected),
                    int(trace.degraded),
                    trace.final_answer_preview[:500],
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save run trace: {e}")

    def get_runs(self, limit: int = 50) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """
            SELECT run_id, thread_id, start_time, duration_ms, user_query,
                   trajectory, total_tokens, final_status, output_valid,
                   guardrail_triggered, loop_detected, degraded, final_answer_preview
            FROM run_traces
            ORDER BY start_time DESC
            LIMIT ?
        """,
            (limit,),
        ).fetchall()
        conn.close()
        cols = [
            "run_id", "thread_id", "start_time", "duration_ms", "user_query",
            "trajectory", "total_tokens", "final_status", "output_valid",
            "guardrail_triggered", "loop_detected", "degraded", "final_answer_preview",
        ]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            d["trajectory"] = json.loads(d["trajectory"] or "[]")
            result.append(d)
        return result

    def get_node_traces(self, run_id: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """
            SELECT node_name, node_index, start_time, duration_ms,
                   input_snapshot, output_snapshot, model_name,
                   input_tokens, output_tokens, tool_calls, status, error_message
            FROM node_traces
            WHERE run_id = ?
            ORDER BY node_index ASC
        """,
            (run_id,),
        ).fetchall()
        conn.close()
        cols = [
            "node_name", "node_index", "start_time", "duration_ms",
            "input_snapshot", "output_snapshot", "model_name",
            "input_tokens", "output_tokens", "tool_calls", "status", "error_message",
        ]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            d["input_snapshot"] = json.loads(d["input_snapshot"] or "{}")
            d["output_snapshot"] = json.loads(d["output_snapshot"] or "{}")
            d["tool_calls"] = json.loads(d["tool_calls"] or "[]")
            result.append(d)
        return result

    def get_token_stats(self) -> dict:
        """Aggregate token usage per node across all runs."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """
            SELECT node_name,
                   AVG(input_tokens) as avg_in,
                   AVG(output_tokens) as avg_out,
                   AVG(total_tokens) as avg_total,
                   MAX(total_tokens) as max_total,
                   COUNT(*) as calls
            FROM node_traces
            WHERE total_tokens IS NOT NULL
            GROUP BY node_name
            ORDER BY avg_total DESC
        """
        ).fetchall()
        conn.close()
        return [
            {
                "node_name": r[0],
                "avg_input_tokens": round(r[1] or 0, 1),
                "avg_output_tokens": round(r[2] or 0, 1),
                "avg_total_tokens": round(r[3] or 0, 1),
                "max_total_tokens": r[4] or 0,
                "call_count": r[5],
            }
            for r in rows
        ]