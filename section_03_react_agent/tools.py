# tools.py
import os
import math
import sqlite3
from langchain_core.tools import tool
from knowledge_base import DB_PATH, init_db

DOCS_DIR = os.path.join(os.path.dirname(__file__), "test_docs")


@tool
def list_available_files() -> str:
    """
    List all available files in the project documents directory.
    Call this FIRST when you don't know what files exist, before calling read_file.
    Takes no arguments. Returns filenames and their sizes in bytes.
    """
    try:
        files = os.listdir(DOCS_DIR)
        if not files:
            return "No files found in the documents directory."
        lines = ["Available files:"]
        for f in sorted(files):
            path = os.path.join(DOCS_DIR, f)
            size = os.path.getsize(path)
            lines.append(f"  - {f} ({size} bytes)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool
def read_file(path: str) -> str:
    """
    Read the contents of a project document file.
    The path argument should be just the filename, e.g. 'meeting_notes.txt'.
    Do NOT include directory paths. Call list_available_files first if unsure what files exist.
    Returns the full file contents with filename and character count.
    """
    filename = os.path.basename(path)  # strip any directory components
    full_path = os.path.join(DOCS_DIR, filename)

    if not os.path.exists(full_path):
        available = os.listdir(DOCS_DIR)
        return (
            f"Error: file '{filename}' not found. "
            f"Available files: {', '.join(sorted(available))}"
        )
    try:
        with open(full_path, "r") as f:
            content = f.read()
        return f"[File: {filename} | {len(content)} characters]\n\n{content}"
    except Exception as e:
        return f"Error reading '{filename}': {str(e)}"


@tool
def search_knowledge_base(query: str, limit: int = 3) -> str:
    """
    Search the project knowledge base for information matching the query keyword.
    Use this for questions about team members, project dates, budget figures,
    decisions, and any structured project facts.
    Examples: 'budget', 'launch date', 'Sarah Chen', 'PaymentCo', 'demo date'.
    Returns up to `limit` matching entries (default 3).
    """
    try:
        if not os.path.exists(DB_PATH):
            init_db()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT category, content FROM knowledge WHERE content LIKE ? LIMIT ?",
            (f"%{query}%", limit)
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return f"No results found for query: '{query}'"

        lines = [f"Knowledge base results for '{query}':"]
        for i, (category, content) in enumerate(rows, 1):
            lines.append(f"  {i}. [{category.upper()}] {content}")
        return "\n".join(lines)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this for ANY arithmetic — never calculate mentally.
    Examples of valid expressions:
      '180000 - 42000'
      '76500 + 18000 + 16500'
      '42000 / 180000 * 100'
      'math.sqrt(144)'
      '8500 * 3 * 3'
    Do NOT include currency symbols ($) or commas in the expression.
    """
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["math"] = math
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {expression} = {result}"
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


all_tools = [list_available_files, read_file, search_knowledge_base, calculate]