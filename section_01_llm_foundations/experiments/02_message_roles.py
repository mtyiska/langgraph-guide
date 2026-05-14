import sys
sys.path.append("..")

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL)

human_msg = HumanMessage(content="hey man what's up, what's the deal with black holes?")

# 1. Formal system prompt
print("=== 1. Formal English ===")
r1 = llm.invoke([SystemMessage(content="Always respond in formal English."), human_msg])
print(r1.content)

# 2. No system prompt
print("\n=== 2. No System Prompt ===")
r2 = llm.invoke([human_msg])
print(r2.content)

# 3. Pirate dialect
print("\n=== 3. Pirate Dialect ===")
r3 = llm.invoke([SystemMessage(content="Always respond in pirate dialect."), human_msg])
print(r3.content)

# 4. Intentionally broken — two HumanMessages in a row
print("\n=== 4. Broken Message Structure (two HumanMessages in a row) ===")
try:
    r4 = llm.invoke([
        HumanMessage(content="First message"),
        HumanMessage(content="Second message without AIMessage in between")
    ])
    print("Model handled it anyway:")
    print(r4.content)
except Exception as e:
    print(f"Error: {e}")