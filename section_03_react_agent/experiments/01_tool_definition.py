# experiments/01_tool_definition.py
import sys
sys.path.append(".")

import json
from langchain_core.tools import tool

@tool
def to_uppercase(text: str) -> str:
    """Convert a string to uppercase. Use when the user wants text in all caps."""
    return text.upper()

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the sum."""
    return a + b

@tool
def get_items_by_category(category: str) -> str:
    """
    Return a list of items for a given category.
    Valid categories: 'fruits', 'vegetables', 'grains'.
    Returns a formatted list of items in that category.
    """
    data = {
        "fruits": ["apple", "banana", "cherry", "mango"],
        "vegetables": ["carrot", "broccoli", "spinach", "potato"],
        "grains": ["rice", "oats", "wheat", "quinoa"],
    }
    items = data.get(category.lower())
    if not items:
        return f"Unknown category '{category}'. Valid: fruits, vegetables, grains."
    return f"Items in '{category}': {', '.join(items)}"

# Call them directly — they're just functions
print("=== Direct Python calls ===")
print(to_uppercase.invoke({"text": "hello world"}))
print(add_numbers.invoke({"a": 7, "b": 5}))
print(get_items_by_category.invoke({"category": "fruits"}))

# Inspect the schema the model receives
print("\n=== JSON Schemas sent to the model ===")
for t in [to_uppercase, add_numbers, get_items_by_category]:
    print(f"\n--- {t.name} ---")
    print(json.dumps(t.args_schema.schema(), indent=2))

# Change a docstring — watch the schema change
print("\n=== Effect of a more detailed docstring ===")
@tool
def add_numbers_v2(a: float, b: float) -> float:
    """
    Sum two numeric values.
    Use this tool whenever arithmetic addition is needed.
    Example: add_numbers_v2(3.5, 2.1) returns 5.6
    """
    return a + b

print(json.dumps(add_numbers_v2.args_schema.schema(), indent=2))