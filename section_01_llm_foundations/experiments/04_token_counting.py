import sys
sys.path.append("..")

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL)

messages = [
    "Hi",
    "What is machine learning?",
    "Explain the history of the Roman Empire in detail, covering the rise, the republic, the transition to empire, the height of Roman power, and the eventual fall of the Western Roman Empire.",
    "Write a detailed essay on the causes and consequences of World War 1, covering the political landscape of Europe in 1914, the assassination of Archduke Franz Ferdinand, the alliance systems, the major battles, the role of technology, the human cost, and the Treaty of Versailles and its long-term consequences for the 20th century.",
    " ".join(["The quick brown fox jumps over the lazy dog."] * 100)
]

print(f"{'Input Preview':<40} {'Chars':>8} {'Input Tokens':>14} {'Output Tokens':>14} {'Chars/Token':>12}")
print("-" * 95)

ratios = []

for msg in messages:
    response = llm.invoke([HumanMessage(content=msg)])
    
    input_tokens = response.response_metadata.get("prompt_eval_count")
    output_tokens = response.response_metadata.get("eval_count")
    char_count = len(msg)
    
    if input_tokens:
        ratio = char_count / input_tokens
        ratios.append(ratio)
        ratio_str = f"{ratio:.2f}"
    else:
        ratio_str = "N/A"
    
    preview = msg[:37] + "..." if len(msg) > 40 else msg
    print(f"{preview:<40} {char_count:>8} {str(input_tokens):>14} {str(output_tokens):>14} {ratio_str:>12}")

if ratios:
    avg_ratio = sum(ratios) / len(ratios)
    print(f"\nAverage chars/token: {avg_ratio:.2f}")
    print(f"Use this as your fallback estimator: chars / {avg_ratio:.1f} ≈ tokens")