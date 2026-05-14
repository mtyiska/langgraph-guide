import os
import json
import logging
from typing import Optional
from langchain_core.tools import tool

import sys
sys.path.append("..")
from config import OLLAMA_BASE_URL
from retry import retry_with_backoff, retry_stats
from validators import ToolSuccess, ToolError
from injection_defence import wrap_retrieved_content

logger = logging.getLogger(__name__)

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "section_05_rag", "docs")


def _format_success(result: str, **metadata) -> str:
    return ToolSuccess(result=result, metadata=metadata).model_dump_json()


def _format_error(
    error_type: str,
    message: str,
    recoverable: bool,
    suggestion: str,
) -> str:
    return ToolError(
        error_type=error_type,
        message=message,
        recoverable=recoverable,
        suggestion=suggestion,
    ).model_dump_json()


@tool
def list_available_documents() -> str:
    """
    List all markdown documents available for research.
    Returns filenames with their sizes. Call this first to understand
    what sources are available before searching.
    """
    if not os.path.exists(DOCS_DIR):
        return _format_error(
            "not_found",
            f"Documents directory not found: {DOCS_DIR}",
            recoverable=False,
            suggestion="Ensure section_05_rag/docs exists and contains markdown files.",
        )

    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".md")]
    if not files:
        return _format_error(
            "not_found",
            "No markdown files found in documents directory.",
            recoverable=False,
            suggestion="Add markdown files to the docs directory.",
        )

    lines = ["Available documents:"]
    for f in sorted(files):
        path = os.path.join(DOCS_DIR, f)
        size = os.path.getsize(path)
        lines.append(f"  - {f}  ({size} bytes)")

    return _format_success("\n".join(lines), count=len(files))


@tool
@retry_with_backoff(max_attempts=3, base_delay=0.3, exceptions=(IOError, OSError))
def read_document(filename: str) -> str:
    """
    Read the full contents of a document from the local docs directory.
    Returns the document content wrapped for safe injection into context.

    Args:
        filename: The markdown filename (e.g. 'budget_overview.md').
                  Use list_available_documents to see valid filenames.
    """
    # Sanitise — no path traversal
    filename = os.path.basename(filename)
    if not filename.endswith(".md"):
        return _format_error(
            "invalid_input",
            f"Only .md files are supported. Got: {filename}",
            recoverable=True,
            suggestion="Check the filename with list_available_documents.",
        )

    path = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(path):
        return _format_error(
            "not_found",
            f"Document not found: {filename}",
            recoverable=True,
            suggestion="Use list_available_documents to see available files.",
        )

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Wrap in injection-safe container
    wrapped = wrap_retrieved_content(content, source=filename)
    return _format_success(wrapped, filename=filename, char_count=len(content))


@tool
def search_documents(query: str, max_results: int = 3) -> str:
    """
    Search across all documents for content relevant to the query.
    Uses simple keyword matching — for semantic search use the Section 5 RAG tools.

    Args:
        query: Search query. Use specific terms.
        max_results: Maximum number of matching excerpts to return (default 3).
    """
    if not os.path.exists(DOCS_DIR):
        return _format_error(
            "not_found",
            "Documents directory not found.",
            recoverable=False,
            suggestion="Ensure section_05_rag/docs exists.",
        )

    query_lower = query.lower()
    terms = [t.strip() for t in query_lower.split() if len(t.strip()) > 2]

    if not terms:
        return _format_error(
            "invalid_input",
            "Query too short or consists only of stop words.",
            recoverable=True,
            suggestion="Use more specific search terms.",
        )

    matches = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(DOCS_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except IOError:
            continue

        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(term in line_lower for term in terms):
                # Grab a small context window
                lo = max(0, i - 1)
                hi = min(len(lines), i + 3)
                excerpt = "".join(lines[lo:hi]).strip()
                matches.append({
                    "filename": filename,
                    "line": i + 1,
                    "excerpt": excerpt,
                })
                if len(matches) >= max_results * 3:
                    break

        if len(matches) >= max_results * 3:
            break

    if not matches:
        return _format_error(
            "not_found",
            f"No documents contain terms matching '{query}'.",
            recoverable=True,
            suggestion="Try broader or different search terms.",
        )

    # Deduplicate by filename, keep top max_results
    seen_files = set()
    deduped = []
    for m in matches:
        if m["filename"] not in seen_files:
            seen_files.add(m["filename"])
            deduped.append(m)
        if len(deduped) >= max_results:
            break

    lines_out = []
    for m in deduped:
        wrapped = wrap_retrieved_content(m["excerpt"], source=m["filename"])
        lines_out.append(
            f"[Source: {m['filename']} | Line: {m['line']}]\n{wrapped}"
        )

    return _format_success(
        "\n\n---\n\n".join(lines_out),
        result_count=len(deduped),
    )


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a simple arithmetic expression safely.
    Supports: +, -, *, /, **, parentheses, and basic numeric literals.

    Args:
        expression: A Python arithmetic expression string, e.g. '320000 * 0.5'
    """
    # Whitelist: only allow digits, operators, spaces, dots, parentheses
    import re
    allowed = re.compile(r"^[\d\s\+\-\*\/\.\(\)\*\*]+$")
    if not allowed.match(expression):
        return _format_error(
            "invalid_input",
            f"Expression contains disallowed characters: {expression}",
            recoverable=True,
            suggestion="Use only numbers and arithmetic operators (+, -, *, /, **).",
        )

    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return _format_success(
            f"{expression} = {result}",
            expression=expression,
            result=result,
        )
    except ZeroDivisionError:
        return _format_error(
            "invalid_input",
            "Division by zero.",
            recoverable=True,
            suggestion="Check the denominator in your expression.",
        )
    except Exception as e:
        return _format_error(
            "invalid_input",
            f"Could not evaluate expression: {e}",
            recoverable=True,
            suggestion="Simplify the expression.",
        )


ALL_TOOLS = [list_available_documents, read_document, search_documents, calculate]