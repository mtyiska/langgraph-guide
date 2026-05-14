import os
import difflib
from typing import Optional
from state import ProposedChange


def read_file(path: str) -> dict:
    """Read a file and return its content with metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines()
    return {
        "path": path,
        "content": content,
        "line_count": len(lines),
        "lines": lines,
    }


def generate_unified_diff(
    original: str,
    proposed: str,
    filename: str,
    line_start: int,
) -> str:
    """Generate a unified diff string between original and proposed code."""
    original_lines = original.splitlines(keepends=True)
    proposed_lines = proposed.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        proposed_lines,
        fromfile=f"{filename} (original)",
        tofile=f"{filename} (proposed)",
        lineterm="",
        n=3,
    )
    result = "".join(diff)
    return result if result else "(no textual difference)"


def apply_change_to_file(
    file_path: str,
    line_start: int,
    line_end: int,
    new_code: str,
) -> bool:
    """
    Apply a change to a file by replacing lines line_start through line_end
    (1-indexed, inclusive) with new_code. Creates a .bak backup first.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Write backup
    backup_path = file_path + ".bak"
    with open(backup_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Replace the target lines
    new_lines_list = new_code.splitlines(keepends=True)
    # Ensure trailing newline on each replacement line
    new_lines_list = [
        l if l.endswith("\n") else l + "\n"
        for l in new_lines_list
    ]

    start_idx = line_start - 1  # convert to 0-indexed
    end_idx = line_end          # slice end is exclusive

    updated = lines[:start_idx] + new_lines_list + lines[end_idx:]

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(updated)

    return True


def get_file_lines(file_path: str, start: int, end: int, context: int = 3) -> str:
    """
    Return a formatted excerpt of a file from start to end (1-indexed, inclusive)
    with optional surrounding context lines.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lo = max(0, start - 1 - context)
    hi = min(len(lines), end + context)

    output = []
    for i, line in enumerate(lines[lo:hi], start=lo + 1):
        marker = ">>>" if start <= i <= end else "   "
        output.append(f"{marker} {i:4d} | {line.rstrip()}")

    return "\n".join(output)