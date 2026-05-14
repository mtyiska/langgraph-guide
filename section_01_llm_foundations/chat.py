import sys
import os
sys.path.append("..")

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import (
    PRIMARY_MODEL, OLLAMA_BASE_URL,
    CONTEXT_WINDOW_TOKENS, CONTEXT_WARNING_THRESHOLD
)

# --- Load system prompt ---
def load_system_prompt():
    if os.path.exists("system_prompt.txt"):
        with open("system_prompt.txt", "r") as f:
            return f.read().strip()
    return "You are a helpful, concise assistant."

# --- Token counting ---
def estimate_tokens(text: str) -> int:
    return len(text) // 4  # fallback: 1 token ≈ 4 chars

def count_tokens_in_history(messages: list) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content)
    return total

def get_token_counts(response, history):
    input_tokens = response.response_metadata.get("prompt_eval_count")
    output_tokens = response.response_metadata.get("eval_count")
    if not input_tokens:
        input_tokens = count_tokens_in_history(history)
    if not output_tokens:
        output_tokens = estimate_tokens(response.content)
    return input_tokens, output_tokens

# --- History truncation ---
def truncate_history(history: list, target_pct: float = 0.80) -> list:
    system_msgs = [m for m in history if isinstance(m, SystemMessage)]
    other_msgs = [m for m in history if not isinstance(m, SystemMessage)]
    
    while True:
        total = count_tokens_in_history(system_msgs + other_msgs)
        pct = total / CONTEXT_WINDOW_TOKENS
        if pct < target_pct or len(other_msgs) < 2:
            break
        # remove oldest human + ai pair
        other_msgs = other_msgs[2:]
        print("  [Truncated oldest messages to stay within context limit]")
    
    return system_msgs + other_msgs

# --- Display helpers ---
def print_status(turn: int, total_tokens: int):
    pct = (total_tokens / CONTEXT_WINDOW_TOKENS) * 100
    print(f"\n[turn {turn} | ~{total_tokens:,} tokens used | {pct:.0f}% of context]")
    
    if pct >= 95:
        print("🔴 CRITICAL: Context nearly full. Auto-truncating history...")
    elif pct >= CONTEXT_WARNING_THRESHOLD * 100:
        print("⚠ Approaching context limit. Consider starting a new conversation or the history will be truncated.")

def print_history(history: list):
    print("\n=== Conversation History ===")
    for i, msg in enumerate(history):
        role = type(msg).__name__.replace("Message", "")
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"  [{i}] {role}: {preview}")
    print()

# --- Main chat loop ---
def main():
    system_prompt = load_system_prompt()
    history = [SystemMessage(content=system_prompt)]
    model_name = PRIMARY_MODEL
    llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL)
    turn = 0
    total_tokens = estimate_tokens(system_prompt)

    print(f"Chat started. Model: {model_name}")
    print(f"System prompt loaded ({len(system_prompt)} chars)")
    print("Type /help for commands, /exit to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0]

            if cmd == "/exit":
                print("Goodbye.")
                break
            elif cmd == "/clear":
                history = [SystemMessage(content=system_prompt)]
                total_tokens = estimate_tokens(system_prompt)
                turn = 0
                print("History cleared.\n")
            elif cmd == "/history":
                print_history(history)
            elif cmd == "/tokens":
                print(f"\n  Total tokens (estimated): {total_tokens:,}")
                print(f"  Context window: {CONTEXT_WINDOW_TOKENS:,}")
                print(f"  Usage: {(total_tokens/CONTEXT_WINDOW_TOKENS)*100:.1f}%\n")
            elif cmd == "/model" and len(parts) > 1:
                model_name = parts[1]
                llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL)
                print(f"  Switched to model: {model_name}\n")
            elif cmd == "/help":
                print("\n  /history       — show conversation history")
                print("  /clear         — reset conversation")
                print("  /tokens        — show token usage")
                print("  /model <name>  — switch model")
                print("  /exit          — quit\n")
            else:
                print(f"  Unknown command: {cmd}\n")
            continue

        # Add user message and call model
        history.append(HumanMessage(content=user_input))
        turn += 1

        print("Assistant: ", end="", flush=True)
        full_response = ""

        for chunk in llm.stream(history):
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
        print()

        ai_msg = AIMessage(content=full_response)
        history.append(ai_msg)

        # Token tracking
        input_tokens = count_tokens_in_history(history)
        output_tokens = estimate_tokens(full_response)
        total_tokens = input_tokens + output_tokens

        print_status(turn, total_tokens)

        # Auto-truncate at 95%
        if total_tokens / CONTEXT_WINDOW_TOKENS >= 0.95:
            history = truncate_history(history)
            total_tokens = count_tokens_in_history(history)

        print()

if __name__ == "__main__":
    main()