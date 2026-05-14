import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


# ── User Profile ──────────────────────────────────────────────────────────────

def load_profile(user_id: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    profile = dict(row)
    if profile.get("interests"):
        try:
            profile["interests"] = json.loads(profile["interests"])
        except Exception:
            profile["interests"] = []
    return profile


def create_profile(user_id: str) -> dict:
    conn = get_conn()
    conn.execute(
        """INSERT OR IGNORE INTO user_profiles
           (user_id, name, style, interests, timezone)
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, None, "default", "[]", "UTC")
    )
    conn.commit()
    conn.close()
    return load_profile(user_id)


def update_profile(user_id: str, key: str, value: str):
    allowed = {"name", "style", "interests", "timezone"}
    if key not in allowed:
        return f"Unknown profile field: {key}"
    conn = get_conn()
    if key == "interests":
        # append to interests list
        row = conn.execute(
            "SELECT interests FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        current = json.loads(row["interests"]) if row and row["interests"] else []
        if value not in current:
            current.append(value)
        value = json.dumps(current)
    conn.execute(
        f"UPDATE user_profiles SET {key} = ?, updated_at = ? WHERE user_id = ?",
        (value, datetime.utcnow().isoformat(), user_id)
    )
    conn.commit()
    conn.close()


# ── Memories ──────────────────────────────────────────────────────────────────

def save_memory(user_id: str, session_id: str, content: str,
                category: str, importance: float = 0.5):
    conn = get_conn()
    conn.execute(
        """INSERT INTO memories (user_id, session_id, content, category, importance)
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, session_id, content, category, importance)
    )
    conn.commit()
    conn.close()


def retrieve_memories(user_id: str, limit: int = 5) -> list[dict]:
    """Strategy C: combined score = 0.6 * importance + 0.4 * recency_score"""
    conn = get_conn()
    rows = conn.execute(
        """SELECT *, 
           (0.6 * importance + 0.4 * (1.0 / (1.0 + CAST(
               (julianday('now') - julianday(last_accessed)) AS REAL
           )))) AS combined_score
           FROM memories
           WHERE user_id = ? AND archived = 0
           ORDER BY combined_score DESC
           LIMIT ?""",
        (user_id, limit)
    ).fetchall()
    conn.close()
    results = [dict(r) for r in rows]

    # Update access count and last_accessed
    if results:
        ids = [r["id"] for r in results]
        conn = get_conn()
        conn.execute(
            f"""UPDATE memories SET
                access_count = access_count + 1,
                last_accessed = ?
                WHERE id IN ({','.join('?' * len(ids))})""",
            [datetime.utcnow().isoformat()] + ids
        )
        conn.commit()
        conn.close()

    return results


def get_memory_count(user_id: str) -> int:
    conn = get_conn()
    count = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE user_id = ? AND archived = 0",
        (user_id,)
    ).fetchone()[0]
    conn.close()
    return count


def check_duplicate(user_id: str, candidate: str, threshold: float = 0.5) -> bool:
    """Simple word-overlap duplicate check."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT content FROM memories WHERE user_id = ? AND archived = 0",
        (user_id,)
    ).fetchall()
    conn.close()

    candidate_words = set(candidate.lower().split())
    for row in rows:
        existing_words = set(row["content"].lower().split())
        if not existing_words:
            continue
        overlap = len(candidate_words & existing_words) / len(candidate_words | existing_words)
        if overlap >= threshold:
            return True
    return False


def consolidate_memories(user_id: str, threshold: int = 50):
    """Archive stale and deduplicate preference memories."""
    conn = get_conn()

    # Archive memories not accessed in 30 days
    cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
    conn.execute(
        """UPDATE memories SET archived = 1
           WHERE user_id = ? AND last_accessed < ? AND archived = 0""",
        (user_id, cutoff)
    )

    # Deduplicate preferences — keep only most recent per content cluster
    rows = conn.execute(
        """SELECT id, content FROM memories
           WHERE user_id = ? AND category = 'preference' AND archived = 0
           ORDER BY created_at DESC""",
        (user_id,)
    ).fetchall()

    seen_words = []
    to_archive = []
    for row in rows:
        words = set(row["content"].lower().split())
        is_dup = any(
            len(words & seen) / max(len(words | seen), 1) > 0.5
            for seen in seen_words
        )
        if is_dup:
            to_archive.append(row["id"])
        else:
            seen_words.append(words)

    if to_archive:
        conn.execute(
            f"UPDATE memories SET archived = 1 WHERE id IN ({','.join('?' * len(to_archive))})",
            to_archive
        )

    conn.commit()
    conn.close()
    return len(to_archive)


# ── Sessions ──────────────────────────────────────────────────────────────────

def save_session(session_id: str, user_id: str, summary: str, turn_count: int):
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO sessions
           (session_id, user_id, summary, turn_count, ended_at)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, user_id, summary, turn_count, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()