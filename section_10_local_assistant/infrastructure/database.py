import sqlite3
from config import TASKS_DB, MEMORY_DB


def init_tasks_db():
    conn = sqlite3.connect(TASKS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id    TEXT PRIMARY KEY,
            session_id TEXT,
            title      TEXT NOT NULL,
            description TEXT,
            status     TEXT DEFAULT 'pending',
            priority   TEXT DEFAULT 'medium',
            due_date   TEXT,
            created_at TEXT,
            updated_at TEXT,
            completed_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            reminder_id TEXT PRIMARY KEY,
            task_id     TEXT,
            session_id  TEXT,
            message     TEXT NOT NULL,
            remind_at   TEXT,
            created_at  TEXT,
            FOREIGN KEY (task_id) REFERENCES tasks(task_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status  ON tasks(status)")
    conn.commit()
    conn.close()


def init_memory_db():
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            session_id TEXT NOT NULL,
            key        TEXT NOT NULL,
            value      TEXT NOT NULL,
            updated_at TEXT,
            PRIMARY KEY (session_id, key)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodic_memory (
            memory_id  TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            summary    TEXT NOT NULL,
            topics     TEXT,
            created_at TEXT,
            importance REAL DEFAULT 0.5
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            session_id  TEXT PRIMARY KEY,
            summary     TEXT NOT NULL,
            turn_count  INTEGER,
            updated_at  TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_session ON episodic_memory(session_id)")
    conn.commit()
    conn.close()


def init_all_databases():
    init_tasks_db()
    init_memory_db()