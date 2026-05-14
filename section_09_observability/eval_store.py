import sqlite3
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EvalStore:
    def __init__(self, db_path: str = "./data/eval_results.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                result_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                run_id TEXT,
                eval_run_label TEXT,
                timestamp TEXT,
                score_factual_accuracy REAL,
                score_citation_validity REAL,
                score_completeness REAL,
                score_trajectory_efficiency REAL,
                score_no_hallucination REAL,
                total_score REAL,
                passed INTEGER,
                facts_found TEXT,
                facts_missing TEXT,
                forbidden_claims_found TEXT,
                citations_valid TEXT,
                citations_missing TEXT,
                actual_trajectory TEXT,
                judge_reasoning TEXT,
                duration_ms REAL,
                total_tokens INTEGER,
                final_status TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_case ON eval_results(case_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_label ON eval_results(eval_run_label)"
        )
        conn.commit()
        conn.close()

    def save_result(self, result: dict, eval_run_label: str = "default"):
        import uuid
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO eval_results VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """,
            (
                str(uuid.uuid4()),
                result["case_id"],
                result.get("run_id", ""),
                eval_run_label,
                datetime.utcnow().isoformat(),
                result.get("score_factual_accuracy", 0),
                result.get("score_citation_validity", 0),
                result.get("score_completeness", 0),
                result.get("score_trajectory_efficiency", 0),
                result.get("score_no_hallucination", 0),
                result.get("total_score", 0),
                int(result.get("passed", False)),
                json.dumps(result.get("facts_found", [])),
                json.dumps(result.get("facts_missing", [])),
                json.dumps(result.get("forbidden_claims_found", [])),
                json.dumps(result.get("citations_valid", [])),
                json.dumps(result.get("citations_missing", [])),
                json.dumps(result.get("actual_trajectory", [])),
                result.get("judge_reasoning", ""),
                result.get("duration_ms", 0),
                result.get("total_tokens", 0),
                result.get("final_status", ""),
            ),
        )
        conn.commit()
        conn.close()

    def get_results_for_label(self, label: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM eval_results WHERE eval_run_label = ? ORDER BY case_id",
            (label,),
        ).fetchall()
        cols = [d[0] for d in conn.execute("PRAGMA table_info(eval_results)").fetchall()]
        conn.close()
        results = []
        for row in rows:
            d = dict(zip(cols, row))
            for field in [
                "facts_found", "facts_missing", "forbidden_claims_found",
                "citations_valid", "citations_missing", "actual_trajectory",
            ]:
                d[field] = json.loads(d.get(field) or "[]")
            results.append(d)
        return results

    def compare_labels(self, label_a: str, label_b: str) -> dict:
        a = {r["case_id"]: r for r in self.get_results_for_label(label_a)}
        b = {r["case_id"]: r for r in self.get_results_for_label(label_b)}

        shared = set(a.keys()) & set(b.keys())
        improved = [c for c in shared if b[c]["total_score"] > a[c]["total_score"] + 0.01]
        degraded = [c for c in shared if b[c]["total_score"] < a[c]["total_score"] - 0.01]

        avg_a = sum(a[c]["total_score"] for c in shared) / len(shared) if shared else 0
        avg_b = sum(b[c]["total_score"] for c in shared) / len(shared) if shared else 0

        return {
            "label_a": label_a,
            "label_b": label_b,
            "shared_cases": len(shared),
            "avg_score_a": round(avg_a, 3),
            "avg_score_b": round(avg_b, 3),
            "score_delta": round(avg_b - avg_a, 3),
            "cases_improved": improved,
            "cases_degraded": degraded,
            "recommendation": label_b if avg_b > avg_a else label_a,
        }