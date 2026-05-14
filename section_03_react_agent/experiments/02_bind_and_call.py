# experiments/02_bind_and_call.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

@tool
def to_uppercase(text: str) -> str:
    """Convert a string to uppercase."""
    return text.upper()

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together. Use for any multiplication calculation."""
    return a * b

@tool
def get_category_items(category: str) -> str:
    """Return items for a category. Valid: 'fruits', 'vegetables', 'grains'."""
    data = {
        "fruits": ["apple", "banana", "mango"],
        "vegetables": ["carrot", "broccoli"],
        "grains": ["rice", "oats"],
    }
    return str(data.get(category.lower(), f"Unknown category: {category}"))

tools = [to_uppercase, multiply, get_category_items]
llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

test_prompts = [
    "What is 47 multiplied by 83?",
    "Can you make this text uppercase: 'the quick brown fox'",
    "What fruits are available?",
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    response = llm_with_tools.invoke([HumanMessage(content=prompt)])

    print(f"Content: {repr(response.content)}")
    print(f"Tool calls: {response.tool_calls}")

    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"\n  Tool: {tc['name']}")
            print(f"  Args: {tc['args']}")
            print(f"  ID:   {tc['id']}")
            if tc['name'] in tool_map:
                result = tool_map[tc['name']].invoke(tc['args'])
                print(f"  Result: {result}")
            else:
                print(f"  ERROR: Tool '{tc['name']}' not in registry")
    else:
        print("  [No tool call produced — Mode 1 failure in the wild]")