import sys
import os
import json
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

sys.path.append("..")
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

from state import CodeReviewState, ProposedChange, ReviewDecision
from tools import read_file, generate_unified_diff, apply_change_to_file

llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

# ── Analysis nodes ─────────────────────────────────────────────────────────────

def read_files_node(state: CodeReviewState) -> dict:
    """Read all target files into state."""
    files_read = []
    errors = []

    for path in state["files_to_review"]:
        try:
            file_data = read_file(path)
            files_read.append({
                "path": file_data["path"],
                "content": file_data["content"],
                "line_count": file_data["line_count"],
            })
        except FileNotFoundError as e:
            errors.append(str(e))

    if errors and not files_read:
        return {"status": "failed", "error": "; ".join(errors)}

    return {
        "files_read": files_read,
        "status": "analysing",
    }


def analyse_files_node(state: CodeReviewState) -> dict:
    """Use LLM to analyse each file and identify issues."""
    all_issues = []

    for file_info in state["files_read"]:
        prompt = f"""Analyse this Python file for issues. Look for:
- Bugs (incorrect logic, missing error handling, edge cases)
- Type errors (missing annotations, wrong types)
- Style problems (overly verbose code, non-idiomatic Python)
- Performance issues (algorithmic inefficiency, unnecessary work)
- Security concerns

File: {file_info['path']}

{file_info['content']}

Return a JSON array of issues. Each issue must have:
- "line_start": int (1-indexed start line)
- "line_end": int (1-indexed end line, same as line_start for single-line)
- "issue_type": one of "bug", "type_error", "style", "performance", "security"
- "severity": one of "critical", "high", "medium", "low"
- "description": string explaining the problem
- "original_code": the exact current code on those lines
- "fix_description": string explaining what should change

Return only valid JSON array, no markdown, no explanation."""

        response = llm.invoke([
            SystemMessage(content="You are a senior Python code reviewer. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            issues = json.loads(raw)
            if not isinstance(issues, list):
                issues = []
        except json.JSONDecodeError:
            issues = []

        for issue in issues:
            issue["file_path"] = file_info["path"]

        all_issues.append({
            "file": file_info["path"],
            "issues": issues,
        })

    return {"issues_found": all_issues}


def generate_changes_node(state: CodeReviewState) -> dict:
    """Convert identified issues into specific ProposedChange objects."""
    proposed_changes = []

    for file_issues in state["issues_found"]:
        file_path = file_issues["file"]
        issues = file_issues.get("issues", [])

        file_content = ""
        for fr in state["files_read"]:
            if fr["path"] == file_path:
                file_content = fr["content"]
                break

        for issue in issues:
            if not isinstance(issue, dict):
                continue

            line_start = issue.get("line_start", 1)
            line_end = issue.get("line_end", line_start)
            original_code = issue.get("original_code", "")

            prompt = f"""Given this Python issue, produce the exact replacement code.

File: {file_path}
Issue type: {issue.get('issue_type', 'bug')}
Severity: {issue.get('severity', 'medium')}
Problem: {issue.get('description', '')}
Fix description: {issue.get('fix_description', '')}

Current code (lines {line_start}-{line_end}):
{original_code}

Return only the replacement Python code — no explanation, no markdown, no backticks. Preserve indentation. The replacement must be syntactically valid."""

            response = llm.invoke([
                SystemMessage(content="You are a Python expert. Return only valid Python code."),
                HumanMessage(content=prompt),
            ])

            proposed_code = response.content.strip()
            if proposed_code.startswith("```"):
                lines = proposed_code.splitlines()
                proposed_code = "\n".join(
                    l for l in lines
                    if not l.strip().startswith("```")
                ).strip()

            change: ProposedChange = {
                "file_path": file_path,
                "line_start": line_start,
                "line_end": line_end,
                "original_code": original_code,
                "proposed_code": proposed_code,
                "reason": issue.get("description", ""),
                "issue_type": issue.get("issue_type", "bug"),
                "severity": issue.get("severity", "medium"),
            }
            proposed_changes.append(change)

    return {"proposed_changes": proposed_changes}


# ── Review loop nodes ──────────────────────────────────────────────────────────

def select_next_change_node(state: CodeReviewState) -> dict:
    """Pick the next unreviewed proposed change."""
    idx = state["current_change_index"]
    changes = state["proposed_changes"]

    if idx >= len(changes):
        return {"status": "complete", "pending_change": None, "formatted_diff": None}

    pending = changes[idx]
    return {
        "pending_change": pending,
        "status": "reviewing",
    }


def format_for_review_node(state: CodeReviewState) -> dict:
    """Format the pending change into a human-readable diff."""
    change = state["pending_change"]
    if not change:
        return {}

    diff = generate_unified_diff(
        original=change["original_code"],
        proposed=change["proposed_code"],
        filename=change["file_path"],
        line_start=change["line_start"],
    )

    return {"formatted_diff": diff}


def apply_or_skip_node(state: CodeReviewState) -> dict:
    """Apply or skip the pending change based on the latest review decision."""
    pending = state["pending_change"]
    decisions = state["review_decisions"]

    if not decisions or not pending:
        return {"current_change_index": state["current_change_index"] + 1}

    latest_decision = decisions[-1]
    decision_type = latest_decision["decision"]

    if decision_type == "approved":
        try:
            apply_change_to_file(
                file_path=pending["file_path"],
                line_start=pending["line_start"],
                line_end=pending["line_end"],
                new_code=pending["proposed_code"],
            )
            return {
                "changes_applied": [pending["file_path"]],
                "current_change_index": state["current_change_index"] + 1,
                "status": "applying",
            }
        except Exception as e:
            return {
                "current_change_index": state["current_change_index"] + 1,
                "error": f"Failed to apply change: {e}",
            }

    elif decision_type == "edited":
        edited_code = latest_decision.get("edited_code") or pending["proposed_code"]
        try:
            apply_change_to_file(
                file_path=pending["file_path"],
                line_start=pending["line_start"],
                line_end=pending["line_end"],
                new_code=edited_code,
            )
            return {
                "changes_applied": [pending["file_path"]],
                "current_change_index": state["current_change_index"] + 1,
                "status": "applying",
            }
        except Exception as e:
            return {
                "current_change_index": state["current_change_index"] + 1,
                "error": f"Failed to apply edited change: {e}",
            }

    else:  # rejected
        return {"current_change_index": state["current_change_index"] + 1}


def generate_summary_node(state: CodeReviewState) -> dict:
    """Generate a final summary of the review session."""
    total_proposed = len(state["proposed_changes"])
    decisions = state["review_decisions"]

    approved = [d for d in decisions if d["decision"] == "approved"]
    edited = [d for d in decisions if d["decision"] == "edited"]
    rejected = [d for d in decisions if d["decision"] == "rejected"]
    files_modified = list(set(state["changes_applied"]))

    lines = [
        "=" * 60,
        "CODE REVIEW SESSION COMPLETE",
        "=" * 60,
        f"Files reviewed:    {len(state['files_read'])}",
        f"Issues found:      {total_proposed}",
        f"Approved:          {len(approved)}",
        f"Edited & applied:  {len(edited)}",
        f"Rejected:          {len(rejected)}",
        f"Files modified:    {len(files_modified)}",
        "",
    ]

    if files_modified:
        lines.append("Modified files:")
        for f in files_modified:
            lines.append(f"  - {f}  (backup at {f}.bak)")
        lines.append("")

    if rejected:
        lines.append("Rejected changes:")
        for d in rejected:
            c = d["change"]
            note = d.get("reviewer_note") or "no note"
            lines.append(
                f"  - {c['file_path']} L{c['line_start']}: "
                f"{c['issue_type']} — {c['reason'][:60]} [{note}]"
            )

    summary = "\n".join(lines)
    print(summary)
    return {"status": "complete", "error": None}


# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_select(state: CodeReviewState) -> str:
    if state["status"] == "complete":
        return "generate_summary"
    return "format_for_review"