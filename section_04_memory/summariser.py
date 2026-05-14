import sys
sys.path.append("..")

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

SUMMARISATION_PROMPT = """Summarise the following conversation excerpt in 150 words or less.
Focus on: key facts established, user preferences expressed, decisions made, and any important context for future turns.
Write in third person past tense. Be concise and factual.

Conversation:
{conversation}

Summary:"""

CHARS_PER_TOKEN = 4.0  # calibrated in Section 1


def estimate_tokens(messages: list[BaseMessage]) -> int:
    total_chars = sum(len(str(m.content)) for m in messages)
    return int(total_chars / CHARS_PER_TOKEN)


def messages_to_text(messages: list[BaseMessage]) -> str:
    lines = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        role = type(msg).__name__.replace("Message", "")
        content = str(msg.content)[:500]  # cap per message
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def summarise_messages(messages: list[BaseMessage], model_name: str = PRIMARY_MODEL) -> str:
    if len(messages) < 2:
        return ""

    llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, temperature=0)
    conversation_text = messages_to_text(messages)
    prompt = SUMMARISATION_PROMPT.format(conversation=conversation_text)

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def maybe_summarise(
    messages: list[BaseMessage],
    summary_so_far: str,
    context_window: int = 8192,
    threshold: float = 0.70,
    keep_last_n_turns: int = 4
) -> tuple[list[BaseMessage], str, bool]:
    """
    Returns (updated_messages, updated_summary, did_summarise).
    Keeps system message + last N turns. Summarises everything else.
    """
    token_estimate = estimate_tokens(messages)
    if token_estimate < context_window * threshold:
        return messages, summary_so_far, False

    # Separate system messages from conversation
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    convo_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    # Keep last N turns (each turn = 1 human + 1 AI = 2 messages)
    keep_count = keep_last_n_turns * 2
    to_summarise = convo_msgs[:-keep_count] if len(convo_msgs) > keep_count else []
    to_keep = convo_msgs[-keep_count:] if len(convo_msgs) > keep_count else convo_msgs

    if not to_summarise:
        return messages, summary_so_far, False

    new_summary = summarise_messages(to_summarise)

    # Merge with existing summary
    if summary_so_far:
        merge_prompt = f"Previous summary:\n{summary_so_far}\n\nNew events:\n{new_summary}\n\nMerged summary (150 words max):"
        llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        merged = llm.invoke([HumanMessage(content=merge_prompt)])
        updated_summary = merged.content.strip()
    else:
        updated_summary = new_summary

    updated_messages = system_msgs + to_keep
    return updated_messages, updated_summary, True