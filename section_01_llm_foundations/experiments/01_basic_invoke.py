import sys
sys.path.append("..")

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL)

messages = [HumanMessage(content="What is the capital of France?")]

response = llm.invoke(messages)

print("=== Response Content ===")
print(response.content)

print("\n=== Full AIMessage Object ===")
print(response)

print("\n=== Metadata ===")
print("response_metadata:", response.response_metadata)
print("usage_metadata:", response.usage_metadata)
print("id:", response.id)