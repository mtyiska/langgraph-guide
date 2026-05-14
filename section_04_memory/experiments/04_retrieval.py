import sys
import sqlite3
from datetime import datetime, timedelta
import random
sys.path.append(".")

from memory_store import init_db, DB_PATH, get_conn

TEST_USER = "retrieval_test_user"

FAKE_MEMORIES = [
    ("User prefers concise answers.", "preference", 0.9, 1),
    ("User's name is Riley.", "preference", 0.95, 5),
    ("User is working on Project Zenith.", "fact", 0.85, 3),
    ("User dislikes long paragraphs.", "preference", 0.8, 0),
    ("User asked about Q1 budget on Jan 10.", "episode", 0.4, 10),
    ("User's timezone is EST.", "preference", 0.7, 2),
    ("User prefers Python over Go.", "preference", 0.75, 7),
    ("User asked for file list on Jan 15.", "episode", 0.35, 15),
    ("Project Zenith launches in December.", "fact", 0.7, 4),
    ("User mentioned team size is 6.", "fact", 0.6, 8),
    ("User asked for budget calculation Jan 16.", "episode", 0.35, 16),
    ("User prefers dark mode in tools.", "preference", 0.5, 20),
    ("User's manager is named Chen.", "fact", 0.55, 12),
    ("User dislikes meetings before 9am.", "preference", 0.65, 6),
    ("User asked about team contacts Jan 20.", "episode", 0.3, 25),
    ("User has 3 direct reports.", "fact", 0.6, 9),
    ("User wants weekly summaries.", "preference", 0.8, 1),
    ("User's project deadline is Dec 31.", "fact", 0.85, 3),
    ("User asked for status report Jan 22.", "episode", 0.3, 22),
    ("User enjoys async work, dislikes sync meetings.", "preference", 0.7, 11),
]


def seed_retrieval_test():
    init_db()
    conn = get_conn()
    conn.execute("DELETE FROM memories WHERE user_id = ?", (TEST_USER,))

    for content, category, importance, days_ago in FAKE_MEMORIES:
        last_accessed = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
        created_at = (datetime.utcnow() - timedelta(days=days_ago + random.randint(0, 5))).isoformat()
        conn.execute(
            """INSERT INTO memories (user_id, session_id, content, category, importance, created_at, last_accessed, access_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (TEST_USER, "seed", content, category, importance, created_at, last_accessed, random.randint(0, 10))
        )

    conn.commit()
    conn.close()
    print(f"Seeded {len(FAKE_MEMORIES)} memories for '{TEST_USER}'\n")


def strategy_a(user_id: str, limit: int = 5) -> list:
    """Most recent by last_accessed."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT content, category, importance, last_accessed FROM memories WHERE user_id = ? AND archived = 0 ORDER BY last_accessed DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def strategy_b(user_id: str, limit: int = 5) -> list:
    """Highest importance."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT content, category, importance, last_accessed FROM memories WHERE user_id = ? AND archived = 0 ORDER BY importance DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def strategy_c(user_id: str, limit: int = 5) -> list:
    """Combined score: 0.6 * importance + 0.4 * recency_score."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT content, category, importance, last_accessed,
           (0.6 * importance + 0.4 * (1.0 / (1.0 + CAST(
               (julianday('now') - julianday(last_accessed)) AS REAL
           )))) AS score
           FROM memories WHERE user_id = ? AND archived = 0
           ORDER BY score DESC LIMIT ?""",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def print_results(label: str, results: list):
    print(f"=== {label} ===")
    for i, r in enumerate(results, 1):
        days = r.get("last_accessed", "?")
        imp = r.get("importance", "?")
        score = r.get("score", "")
        score_str = f" | score={score:.3f}" if score else ""
        print(f"  {i}. [{r['category']} | imp={imp}{score_str}] {r['content'][:70]}")
    print()


if __name__ == "__main__":
    seed_retrieval_test()
    print_results("Strategy A — Most Recent", strategy_a(TEST_USER))
    print_results("Strategy B — Highest Importance", strategy_b(TEST_USER))
    print_results("Strategy C — Combined Score", strategy_c(TEST_USER))
    print("Observation: Strategy C balances freshness and importance.")
    print("Pure recency surfaces stale low-value episodes.")
    print("Pure importance misses recently-updated context.")