import sys
sys.path.append("..")

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL)

messages = [HumanMessage(content="Explain how photosynthesis works in about 100 words.")]

print("=== Streaming Response ===")
chunk_count = 0
full_response = ""

for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
    full_response += chunk.content
    chunk_count += 1

print(f"\n\n=== Stats ===")
print(f"Total chunks received: {chunk_count}")
print(f"Total characters: {len(full_response)}")
print(f"Average characters per chunk: {len(full_response) / chunk_count:.1f}")