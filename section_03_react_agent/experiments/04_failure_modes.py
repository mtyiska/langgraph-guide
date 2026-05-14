# experiments/04_failure_modes.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

@tool
def read_file(path: str) -> str:
    """Read a file and return its contents."""
    return f"Contents of {path}: [some data]"

# ── Mode 1: No tools bound ────────────────────────────────────────────────────
print("=" * 60)
print("MODE 1 — No tools bound, model asked to use a tool")
response = llm.invoke([HumanMessage(content="Read the file called notes.txt")])
print(f"Content: {response.content[:200]}")
print(f"Tool calls: {response.tool_calls}")
print("→ Model describes what it would do rather than calling a tool")

# ── Mode 2: Wrong argument name ───────────────────────────────────────────────
@tool
def read_document(file_path: str) -> str:
    """Read a document. The argument is file_path (not path, not filename)."""
    return f"Document contents from {file_path}"

llm_with_tools = llm.bind_tools([read_document])

print("\n" + "=" * 60)
print("MODE 2 — Model may use 'path' instead of 'file_path'")
response = llm_with_tools.invoke([HumanMessage(
    content="Read the file at path 'report.txt'"
)])
print(f"Tool calls: {response.tool_calls}")
if response.tool_calls:
    args = response.tool_calls[0]['args']
    print(f"Args returned: {args}")
    if 'file_path' in args:
        print("✓ Correct arg name used")
    else:
        print(f"✗ Wrong arg name — got {list(args.keys())} instead of ['file_path']")

# ── Mode 3: Hallucinated tool name ────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base."""
    return f"Results for {query}"

llm_with_kb = llm.bind_tools([search_knowledge_base])

print("\n" + "=" * 60)
print("MODE 3 — Model asked to 'search the web' when only KB search exists")
response = llm_with_kb.invoke([HumanMessage(
    content="Search the web for the latest Python release"
)])
print(f"Tool calls: {response.tool_calls}")
if response.tool_calls:
    name = response.tool_calls[0]['name']
    tool_map = {"search_knowledge_base": search_knowledge_base}
    if name not in tool_map:
        print(f"✗ Hallucinated tool: '{name}' — not in registry")
    else:
        print(f"✓ Model used the real tool: '{name}'")
else:
    print("Model produced no tool call — refused or used text instead")

# ── Mode 4: Infinite loop simulation ─────────────────────────────────────────
@tool
def find_answer(query: str) -> str:
    """Find the answer to a question."""
    return "No results found. Try again with different keywords."

llm_loopy = llm.bind_tools([find_answer])

print("\n" + "=" * 60)
print("MODE 4 — Tool always returns 'try again', watch how many times model loops")

messages = [HumanMessage(content="Find the population of the city of Zorblax")]
MAX = 4

for i in range(MAX):
    print(f"\n  Iteration {i + 1}")
    response = llm_loopy.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        print(f"  Agent stopped on its own. Answer: {response.content[:100]}")
        break

    tc = response.tool_calls[0]
    print(f"  Called: {tc['name']} | Args: {tc['args']}")
    result = "No results found. Try again with different keywords."
    print(f"  Result: {result}")
    messages.append(ToolMessage(content=result, tool_call_id=tc['id']))

    if i == MAX - 1:
        print(f"\n  Hit max iterations ({MAX}) — agent did not self-terminate")
        print("  → This is Mode 4. Fix: iteration counter + forced-answer injection in validate_node")