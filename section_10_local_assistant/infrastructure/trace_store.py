import sqlite3
import json
import os
import time
import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional, Callable
from langchain_core.messages import AIMessage
from config import TRACES_DB

logger = logging.getLogger(__name__)


@dataclass
class NodeTrace:
    trace_id:         str      = field(default_factory=lambda: str(uuid.uuid4()))
    run_id:           str      = ""
    thread_id:        str      = ""
    node_name:        str      = ""
    node_index:       int      = 0
    start_time:       datetime = field(default_factory=datetime.utcnow)
    end_time:         Optional[datetime] = None
    duration_ms:      Optional[float]   = None
    input_snapshot:   dict     = field(default_factory=dict)
    output_snapshot:  dict     = field(default_factory=dict)
    model_name:       Optional[str] = None
    input_tokens:     Optional[int] = None
    output_tokens:    Optional[int] = None
    total_tokens:     Optional[int] = None
    tool_calls:       list     = field(default_factory=list)
    status:           str      = "pending"
    error_message:    Optional[str] = None


@dataclass
class RunTrace:
    run_id:              str      = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id:           str      = ""
    start_time:          datetime = field(default_factory=datetime.utcnow)
    end_time:            Optional[datetime] = None
    duration_ms:         Optional[float]   = None
    user_query:          str      = ""
    trajectory:          list     = field(default_factory=list)
    total_tokens:        int      = 0
    final_status:        str      = ""
    output_valid:        bool     = False
    guardrail_triggered: bool     = False
    loop_detected:       bool     = False
    degraded:            bool     = False
    final_answer_preview: str     = ""


def _truncate_state(state: dict, max_len: int = 300) -> dict:
    result = {}
    for k, v in state.items():
        if k.startswith("_"):
            continue
        if isinstance(v, str) and len(v) > max_len:
            result[k] = v[:max_len] + f"... [{len(v)} chars]"
        elif isinstance(v, list) and len(v) > 8:
            result[k] = v[:8] + [f"... [{len(v)} items]"]
        else:
            result[k] = v
    return result


class TraceStore:
    def __init__(self, db_path: str = TRACES_DB):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS node_traces (
                trace_id      TEXT PRIMARY KEY,
                run_id        TEXT NOT NULL,
                thread_id     TEXT,
                node_name     TEXT NOT NULL,
                node_index    INTEGER,
                start_time    TEXT,
                end_time      TEXT,
                duration_ms   REAL,
                input_snapshot  TEXT,
                output_snapshot TEXT,
                model_name    TEXT,
                input_tokens  INTEGER,
                output_tokens INTEGER,
                total_tokens  INTEGER,
                tool_calls    TEXT,
                status        TEXT,
                error_message TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_traces (
                run_id               TEXT PRIMARY KEY,
                thread_id            TEXT,
                start_time           TEXT,
                end_time             TEXT,
                duration_ms          REAL,
                user_query           TEXT,
                trajectory           TEXT,
                total_tokens         INTEGER,
                final_status         TEXT,
                output_valid         INTEGER,
                guardrail_triggered  INTEGER,
                loop_detected        INTEGER,
                degraded             INTEGER,
                final_answer_preview TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_node_run ON node_traces(run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_run_time ON run_traces(start_time)")
        conn.commit()
        conn.close()

    def save_node_trace(self, trace: NodeTrace):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO node_traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    trace.trace_id, trace.run_id, trace.thread_id,
                    trace.node_name, trace.node_index,
                    trace.start_time.isoformat(),
                    trace.end_time.isoformat() if trace.end_time else None,
                    trace.duration_ms,
                    json.dumps(_truncate_state(trace.input_snapshot)),
                    json.dumps(_truncate_state(trace.output_snapshot)),
                    trace.model_name, trace.input_tokens,
                    trace.output_tokens, trace.total_tokens,
                    json.dumps(trace.tool_calls),
                    trace.status, trace.error_message,
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
                "INSERT OR REPLACE INTO run_traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    trace.run_id, trace.thread_id,
                    trace.start_time.isoformat(),
                    trace.end_time.isoformat() if trace.end_time else None,
                    trace.duration_ms, trace.user_query,
                    json.dumps(trace.trajectory), trace.total_tokens,
                    trace.final_status, int(trace.output_valid),
                    int(trace.guardrail_triggered), int(trace.loop_detected),
                    int(trace.degraded), trace.final_answer_preview[:500],
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save run trace: {e}")

    def get_runs(self, limit: int = 100) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT run_id, thread_id, start_time, duration_ms, user_query,
                      trajectory, total_tokens, final_status, output_valid,
                      guardrail_triggered, loop_detected, degraded, final_answer_preview
               FROM run_traces ORDER BY start_time DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        cols = ["run_id","thread_id","start_time","duration_ms","user_query",
                "trajectory","total_tokens","final_status","output_valid",
                "guardrail_triggered","loop_detected","degraded","final_answer_preview"]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            d["trajectory"] = json.loads(d["trajectory"] or "[]")
            result.append(d)
        return result

    def get_node_traces(self, run_id: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT node_name, node_index, start_time, duration_ms,
                      input_snapshot, output_snapshot, model_name,
                      input_tokens, output_tokens, tool_calls, status, error_message
               FROM node_traces WHERE run_id=? ORDER BY node_index""",
            (run_id,),
        ).fetchall()
        conn.close()
        cols = ["node_name","node_index","start_time","duration_ms",
                "input_snapshot","output_snapshot","model_name",
                "input_tokens","output_tokens","tool_calls","status","error_message"]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            d["input_snapshot"]  = json.loads(d["input_snapshot"]  or "{}")
            d["output_snapshot"] = json.loads(d["output_snapshot"] or "{}")
            d["tool_calls"]      = json.loads(d["tool_calls"]      or "[]")
            result.append(d)
        return result

    def get_token_stats(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT node_name,
                   AVG(input_tokens)  avg_in,
                   AVG(output_tokens) avg_out,
                   AVG(total_tokens)  avg_total,
                   MAX(total_tokens)  max_total,
                   COUNT(*)           calls
            FROM node_traces
            WHERE total_tokens IS NOT NULL
            GROUP BY node_name ORDER BY avg_total DESC
        """).fetchall()
        conn.close()
        return [
            {"node_name": r[0], "avg_input_tokens": round(r[1] or 0,1),
             "avg_output_tokens": round(r[2] or 0,1),
             "avg_total_tokens": round(r[3] or 0,1),
             "max_total_tokens": r[4] or 0, "call_count": r[5]}
            for r in rows
        ]


def traced_node(node_func: Callable, store: TraceStore, run_context: dict) -> Callable:
    @wraps(node_func)
    def wrapper(state: dict) -> dict:
        trace = NodeTrace(
            run_id=run_context.get("run_id", ""),
            thread_id=run_context.get("thread_id", ""),
            node_name=node_func.__name__,
            node_index=run_context["node_index"],
            start_time=datetime.utcnow(),
            input_snapshot=dict(state),
        )
        run_context["node_index"] += 1
        t0 = time.perf_counter()
        result = None
        try:
            result = node_func(state)
            trace.status = "success"
            trace.output_snapshot = dict(result) if result else {}

            msgs = (result or {}).get("messages", [])
            if msgs and isinstance(msgs[-1], AIMessage):
                meta  = getattr(msgs[-1], "response_metadata", {}) or {}
                usage = meta.get("usage", meta.get("usage_metadata", {}))
                if usage:
                    trace.input_tokens  = usage.get("prompt_tokens") or usage.get("input_tokens")
                    trace.output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
                    if trace.input_tokens and trace.output_tokens:
                        trace.total_tokens = trace.input_tokens + trace.output_tokens
                    trace.model_name = meta.get("model", meta.get("model_name"))
                tcs = getattr(msgs[-1], "tool_calls", []) or []
                trace.tool_calls = [{"name": tc.get("name"), "args": tc.get("args")} for tc in tcs]

            if trace.total_tokens:
                run_context["total_tokens"] = run_context.get("total_tokens", 0) + trace.total_tokens

            return result
        except Exception as e:
            trace.status = "error"
            trace.error_message = str(e)
            raise
        finally:
            trace.end_time   = datetime.utcnow()
            trace.duration_ms = (time.perf_counter() - t0) * 1000
            store.save_node_trace(trace)
            run_context["trajectory"].append(node_func.__name__)
    return wrapper