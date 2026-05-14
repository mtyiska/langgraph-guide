CREATE TABLE IF NOT EXISTS user_profiles (
    user_id     TEXT PRIMARY KEY,
    name        TEXT,
    style       TEXT,
    interests   TEXT,
    timezone    TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    session_id    TEXT,
    content       TEXT NOT NULL,
    category      TEXT NOT NULL,
    importance    REAL DEFAULT 0.5,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    access_count  INTEGER DEFAULT 0,
    archived      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    summary     TEXT,
    turn_count  INTEGER DEFAULT 0,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    ended_at    DATETIME
);