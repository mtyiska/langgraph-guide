import sqlite3
import uuid
import logging
from datetime import datetime
from agent.state import AssistantState
from config import MEMORY_DB, MAX_MEMORY_ITEMS, CONVERSATION_SUMMARY_THRESHOLD, FAST_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


def memory_load_node(state: AssistantState) -> dict:
    """Load user preferences and episodic memory into state."""
    conn = sqlite3.connect(MEMORY_DB)

    pref_rows = conn.execute(
        "SELECT key, value FROM user_preferences WHERE session_id='global'"
    ).fetchall()
    preferences = {k: v for k, v in pref_rows}

    memory_rows = conn.execute(
        """SELECT summary FROM episodic_memory
           WHERE session_id='global'
           ORDER BY importance DESC, created_at DESC LIMIT ?""",
        (MAX_MEMORY_ITEMS,),
    ).fetchall()
    past_context = [r[0] for r in memory_rows]

    summary_row = conn.execute(
        "SELECT summary FROM conversation_summaries WHERE session_id='global'"
    ).fetchone()
    conversation_summary = summary_row[0] if summary_row else None

    conn.close()

    return {
        "user_name":             preferences.get("name"),
        "user_preferences":      preferences,
        "relevant_past_context": past_context,
        "conversation_summary":  conversation_summary,
    }


def memory_save_node(state: AssistantState) -> dict:
    """After a turn, persist anything worth remembering."""
    messages = state.get("messages", [])

    if len(messages) >= CONVERSATION_SUMMARY_THRESHOLD:
        _update_conversation_summary(state)

    # Save a brief episodic memory of this turn if there's a final response
    final = state.get("final_response", "")
    if final and len(final) > 20:
        _save_episodic_snippet(state, final)

    return {}


def _update_conversation_summary(state: AssistantState):
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=FAST_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

        recent = state["messages"][-CONVERSATION_SUMMARY_THRESHOLD:]
        text   = "\n".join(f"{m.__class__.__name__}: {m.content[:200]}" for m in recent)

        resp = llm.invoke([{
            "role": "user",
            "content": f"Summarise this conversation in 3 sentences, noting key facts established:\n\n{text}",
        }])

        conn = sqlite3.connect(MEMORY_DB)
        conn.execute(
            """INSERT OR REPLACE INTO conversation_summaries
               (session_id, summary, turn_count, updated_at) VALUES ('global',?,?,?)""",
            (resp.content, len(state["messages"]), datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Conversation summary update failed: {e}")


def _save_episodic_snippet(state: AssistantState, final_response: str):
    try:
        conn = sqlite3.connect(MEMORY_DB)
        conn.execute(
            """INSERT INTO episodic_memory
               (memory_id, session_id, summary, topics, created_at, importance)
               VALUES (?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()),
                "global",
                final_response[:400],
                "[]",
                datetime.utcnow().isoformat(),
                0.5,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Episodic memory save failed: {e}")