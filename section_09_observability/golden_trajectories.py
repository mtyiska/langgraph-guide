import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class GoldenTrajectoryStore:
    def __init__(self, db_path: str = "./data/eval_results.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS golden_trajectories (
                case_id TEXT PRIMARY KEY,
                run_id TEXT,
                trajectory TEXT NOT NULL,
                recorded_at TEXT,
                notes TEXT
            )
        """)
        conn.commit()
        conn.close()

    def record_golden(
        self,
        case_id: str,
        run_id: str,
        trajectory: list[str],
        notes: str = "",
    ):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO golden_trajectories
            (case_id, run_id, trajectory, recorded_at, notes)
            VALUES (?, ?, ?, ?, ?)
        """,
            (case_id, run_id, json.dumps(trajectory), datetime.utcnow().isoformat(), notes),
        )
        conn.commit()
        conn.close()
        logger.info(f"Recorded golden trajectory for {case_id}: {trajectory}")

    def get_golden(self, case_id: str) -> Optional[list[str]]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT trajectory FROM golden_trajectories WHERE case_id = ?", (case_id,)
        ).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None

    def diff_against_golden(
        self, case_id: str, actual_trajectory: list[str]
    ) -> dict:
        golden = self.get_golden(case_id)

        if golden is None:
            return {"has_golden": False, "case_id": case_id}

        added = [n for n in actual_trajectory if n not in golden]
        removed = [n for n in golden if n not in actual_trajectory]
        same_nodes = not added and not removed
        reordered = same_nodes and golden != actual_trajectory

        return {
            "has_golden": True,
            "case_id": case_id,
            "golden_trajectory": golden,
            "actual_trajectory": actual_trajectory,
            "nodes_added": added,
            "nodes_removed": removed,
            "reordered": reordered,
            "matches_golden": golden == actual_trajectory,
            "golden_length": len(golden),
            "actual_length": len(actual_trajectory),
            "length_delta": len(actual_trajectory) - len(golden),
        }

    def list_goldens(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT case_id, run_id, trajectory, recorded_at, notes FROM golden_trajectories"
        ).fetchall()
        conn.close()
        return [
            {
                "case_id": r[0],
                "run_id": r[1],
                "trajectory": json.loads(r[2]),
                "recorded_at": r[3],
                "notes": r[4],
            }
            for r in rows
        ]

    def delete_golden(self, case_id: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "DELETE FROM golden_trajectories WHERE case_id = ?", (case_id,)
        )
        conn.commit()
        conn.close()