import sqlite3
import uuid
import json
from datetime import datetime
from typing import Optional, Literal
from langchain_core.tools import tool
from agent.tools.base import ToolSuccess, ToolError
from config import TASKS_DB


def _conn():
    return sqlite3.connect(TASKS_DB)


@tool
def create_task(
    title: str,
    description: str,
    priority: Literal["low", "medium", "high"] = "medium",
    due_date: Optional[str] = None,
) -> str:
    """
    Create a new task. This is a write operation that requires human approval.
    Args:
        title: Short task title (required)
        description: Full task description
        priority: low, medium, or high (default: medium)
        due_date: ISO date string YYYY-MM-DD or null
    """
    task_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()
    try:
        conn = _conn()
        conn.execute(
            """INSERT INTO tasks
               (task_id, title, description, status, priority, due_date, created_at, updated_at)
               VALUES (?,?,?,'pending',?,?,?,?)""",
            (task_id, title, description, priority, due_date, now, now),
        )
        conn.commit()
        conn.close()
        return ToolSuccess(
            result=f"Task created: '{title}' (ID: {task_id})",
            metadata={"task_id": task_id},
        ).model_dump_json()
    except Exception as e:
        return ToolError(
            error_type="unknown", message=str(e),
            recoverable=False, suggestion="Database may be unavailable.",
        ).model_dump_json()


@tool
def list_tasks(
    status: Optional[Literal["pending", "in_progress", "complete"]] = None,
    priority: Optional[Literal["low", "medium", "high"]] = None,
) -> str:
    """
    List tasks, optionally filtered by status or priority.
    Use when the user asks about their tasks, what's pending, or what's due.
    """
    conn = _conn()
    q = "SELECT task_id, title, status, priority, due_date FROM tasks WHERE 1=1"
    params: list = []
    if status:
        q += " AND status=?"; params.append(status)
    if priority:
        q += " AND priority=?"; params.append(priority)
    q += " ORDER BY priority DESC, created_at DESC LIMIT 20"
    rows = conn.execute(q, params).fetchall()
    conn.close()

    if not rows:
        return ToolSuccess(result="No tasks found.", metadata={}).model_dump_json()

    lines = []
    for tid, title, st, pri, due in rows:
        due_str = f" (due: {due})" if due else ""
        lines.append(f"[{tid}] {title} — {st} | {pri}{due_str}")
    return ToolSuccess(
        result="\n".join(lines), metadata={"count": len(rows)}
    ).model_dump_json()


@tool
def complete_task(task_id: str) -> str:
    """
    Mark a task as complete. Write operation — requires human approval.
    Args:
        task_id: The short task ID shown in list_tasks
    """
    now = datetime.utcnow().isoformat()
    conn = _conn()
    conn.execute(
        "UPDATE tasks SET status='complete', completed_at=?, updated_at=? WHERE task_id=?",
        (now, now, task_id),
    )
    affected = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()

    if affected == 0:
        return ToolError(
            error_type="not_found",
            message=f"Task {task_id} not found.",
            recoverable=False,
            suggestion="Check task ID with list_tasks.",
        ).model_dump_json()
    return ToolSuccess(
        result=f"Task {task_id} marked complete.", metadata={}
    ).model_dump_json()


@tool
def update_task_status(
    task_id: str,
    new_status: Literal["pending", "in_progress", "complete"],
) -> str:
    """
    Update the status of a task. Write operation — requires approval.
    Args:
        task_id: Short task ID from list_tasks
        new_status: pending, in_progress, or complete
    """
    now = datetime.utcnow().isoformat()
    conn = _conn()
    conn.execute(
        "UPDATE tasks SET status=?, updated_at=? WHERE task_id=?",
        (new_status, now, task_id),
    )
    affected = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()

    if affected == 0:
        return ToolError(
            error_type="not_found",
            message=f"Task {task_id} not found.",
            recoverable=False,
            suggestion="Check task ID with list_tasks.",
        ).model_dump_json()
    return ToolSuccess(
        result=f"Task {task_id} status updated to '{new_status}'.", metadata={}
    ).model_dump_json()


TASK_READ_TOOLS  = [list_tasks]
TASK_WRITE_TOOLS = [create_task, complete_task, update_task_status]
ALL_TASK_TOOLS   = TASK_READ_TOOLS + TASK_WRITE_TOOLS