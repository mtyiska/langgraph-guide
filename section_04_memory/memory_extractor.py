import sys
import json
sys.path.append("..")

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from config import PRIMARY_MODEL, OLLAMA_BASE_URL

EXTRACTION_PROMPT = """Review this conversation and extract any information worth remembering about the user.
Return a JSON array of memory objects. Each object has:
- content: the memory as a concise declarative statement (one sentence)
- category: one of "preference", "fact", "episode"
- importance: float 0.0 to 1.0 (preferences = 0.9, facts = 0.6, episodes = 0.4)

Only extract genuinely useful persistent information.
Return [] if nothing is worth saving.
Return ONLY valid JSON — no markdown, no explanation, no backticks.

Conversation:
{conversation}"""


def extract_memories(messages: list[BaseMessage], model_name: str = PRIMARY_MODEL) -> list[dict]:
    convo_lines = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        role = type(msg).__name__.replace("Message", "")
        convo_lines.append(f"{role}: {str(msg.content)[:300]}")

    if not convo_lines:
        return []

    conversation_text = "\n".join(convo_lines)
    prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)

    llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        raw = response.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        memories = json.loads(raw.strip())
        if not isinstance(memories, list):
            return []
        # Validate structure
        valid = []
        for m in memories:
            if isinstance(m, dict) and "content" in m and "category" in m:
                valid.append({
                    "content": str(m["content"]),
                    "category": m.get("category", "fact"),
                    "importance": float(m.get("importance", 0.5))
                })
        return valid
    except Exception as e:
        print(f"[Memory extractor] JSON parse error: {e}")
        # Fallback: heuristic extraction
        return heuristic_extract(messages)


def heuristic_extract(messages: list[BaseMessage]) -> list[dict]:
    """Simple pattern-based fallback extractor."""
    import re
    memories = []
    patterns = [
        (r"my name is (\w+)", "preference", 0.9),
        (r"i prefer (.+?)[\.\,]", "preference", 0.9),
        (r"i like (.+?)[\.\,]", "preference", 0.8),
        (r"i hate (.+?)[\.\,]", "preference", 0.8),
        (r"always use (.+?)[\.\,]", "preference", 0.9),
        (r"i('m| am) working on (.+?)[\.\,]", "fact", 0.7),
        (r"my project is (.+?)[\.\,]", "fact", 0.7),
        (r"remind me (.+?)[\.\,]", "fact", 0.6),
    ]

    for msg in messages:
        if isinstance(msg, (SystemMessage,)):
            continue
        text = str(msg.content).lower()
        for pattern, category, importance in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                content = match if isinstance(match, str) else " ".join(match)
                memories.append({
                    "content": content.strip(),
                    "category": category,
                    "importance": importance
                })

    return memories