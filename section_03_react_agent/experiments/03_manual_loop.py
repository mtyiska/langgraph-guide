# experiments/03_manual_loop.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

@tool
def read_fact(topic: str) -> str:
    """Look up a fact about a topic. Available topics: 'python', 'langgraph', 'ollama'."""
    facts = {
        "python": "Python was created by Guido van Rossum and first released in 1991.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "ollama": "Ollama allows you to run large language models locally on your machine.",
    }
    return facts.get(topic.lower(), f"No fact found for topic: '{topic}'")

@tool
def count_words(text: str) -> str:
    """Count the number of words in a text string."""
    count = len(text.split())
    return f"The text contains {count} words."

tools = [read_fact, count_words]
tool_map = {t.name: t for t in tools}

llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
llm_with_tools = llm.bind_tools(tools)

system = SystemMessage(content=(
    "You are a helpful assistant with tools. "
    "Always use tools to answer factual questions. "
    "Never guess — use the tools provided."
))

query = "What is LangGraph, and how many words is that definition?"
print(f"Query: {query}\n")

messages = [system, HumanMessage(content=query)]
max_iterations = 10

for iteration in range(max_iterations):
    print(f"--- Iteration {iteration + 1} | Messages in context: {len(messages)} ---")

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    print(f"Response has tool_calls: {bool(response.tool_calls)}")

    if not response.tool_calls:
        print(f"\nFinal Answer:\n{response.content}")
        break

    for tc in response.tool_calls:
        print(f"  Calling: {tc['name']} with args: {tc['args']}")
        if tc['name'] in tool_map:
            result = tool_map[tc['name']].invoke(tc['args'])
        else:
            result = f"Error: tool '{tc['name']}' does not exist."
        print(f"  Result: {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=tc['id']))

print(f"\nTotal messages in final context: {len(messages)}")
total_chars = sum(len(str(m.content)) for m in messages)
print(f"Estimated tokens sent on final call: ~{total_chars // 4}")