import sqlite3
import json
from datetime import datetime
from langchain_core.tools import tool
from agent.tools.base import ToolSuccess, ToolError
from config import MEMORY_DB


@tool
def save_user_preference(key: str, value: str) -> str:
    """
    Save a user preference for future sessions.
    Use when the user states a preference explicitly — name, communication style,
    topics of interest, preferred formats, or any personal setting.
    Args:
        key: preference name (e.g. 'name', 'tone', 'interests')
        value: preference value
    """
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute(
        """INSERT OR REPLACE INTO user_preferences (session_id, key, value, updated_at)
           VALUES ('global', ?, ?, ?)""",
        (key, value, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return ToolSuccess(result=f"Preference '{key}' saved.", metadata={}).model_dump_json()


@tool
def get_user_preferences() -> str:
    """
    Retrieve all saved user preferences.
    Use at the start of a conversation to personalise the response.
    """
    conn = sqlite3.connect(MEMORY_DB)
    rows = conn.execute(
        "SELECT key, value FROM user_preferences WHERE session_id='global'"
    ).fetchall()
    conn.close()
    if not rows:
        return ToolSuccess(result="No preferences saved yet.", metadata={}).model_dump_json()
    prefs = {k: v for k, v in rows}
    return ToolSuccess(result=json.dumps(prefs, indent=2), metadata={}).model_dump_json()


MEMORY_TOOLS = [save_user_preference, get_user_preferences]