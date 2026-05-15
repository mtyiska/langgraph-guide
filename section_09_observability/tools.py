import os
from langchain_core.tools import tool

DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "knowledge_base")


@tool
def list_available_files() -> str:
    """List all available files in the knowledge base directory. Call this first when you don't know what files exist."""
    try:
        files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
        if not files:
            return "No files found."
        lines = ["Available files:"]
        for f in sorted(files):
            size = os.path.getsize(os.path.join(DOCS_DIR, f))
            lines.append(f"  - {f} ({size} bytes)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing files: {e}"


@tool
def read_file(filename: str) -> str:
    """
    Read a file from the knowledge base. Pass just the filename, e.g. 'project_atlas_budget.md'.
    Do not include directory paths.
    """
    filename = os.path.basename(filename)
    full_path = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(full_path):
        available = sorted(os.listdir(DOCS_DIR))
        return f"File '{filename}' not found. Available: {', '.join(available)}"
    try:
        with open(full_path) as f:
            content = f.read()
        return f"[{filename}]\n\n{content}"
    except Exception as e:
        return f"Error reading '{filename}': {e}"


@tool
def search_documents(query: str, max_results: int = 3) -> str:
    """
    Search all knowledge base documents for content matching the query.
    Use this to find relevant information across all files without reading each one.
    Examples: 'database', 'standup', 'budget', 'team size', 'PostgreSQL'.
    """
    try:
        files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
        matches = []
        query_lower = query.lower()
        for filename in sorted(files):
            full_path = os.path.join(DOCS_DIR, filename)
            with open(full_path) as f:
                content = f.read()
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    context_start = max(0, i - 1)
                    context_end = min(len(lines), i + 3)
                    snippet = "\n".join(lines[context_start:context_end])
                    matches.append(f"[{filename}]\n{snippet}")
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break
        if not matches:
            return f"No results found for '{query}'. Try different keywords or use list_available_files."
        return f"Results for '{query}':\n\n" + "\n\n---\n\n".join(matches)
    except Exception as e:
        return f"Error searching documents: {e}"


all_tools = [list_available_files, read_file, search_documents]