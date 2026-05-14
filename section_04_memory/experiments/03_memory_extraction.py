import sys
sys.path.append(".")

from langchain_core.messages import HumanMessage, AIMessage
from memory_extractor import extract_memories
from memory_store import check_duplicate, init_db, save_memory

def make_convo(pairs: list[tuple]) -> list:
    msgs = []
    for h, a in pairs:
        msgs.append(HumanMessage(content=h))
        msgs.append(AIMessage(content=a))
    return msgs

# Test 1: clear preferences
convo_prefs = make_convo([
    ("My name is Sam and I prefer formal responses.", "Noted, Sam — I'll keep things formal."),
    ("Always use bullet points in your answers.", "Understood, I'll use bullet points."),
    ("I'm working on a project called Helios.", "Got it. I'll keep Helios in mind."),
])

print("=== Test 1: Clear preferences ===")
memories = extract_memories(convo_prefs)
for m in memories:
    print(f"  [{m['category']} | {m['importance']}] {m['content']}")

# Test 2: factual info
convo_facts = make_convo([
    ("The launch date is October 15.", "I've noted the October 15 launch date."),
    ("Budget is $250,000 total.", "Got it — $250,000 total budget."),
])

print("\n=== Test 2: Factual information ===")
for m in extract_memories(convo_facts):
    print(f"  [{m['category']} | {m['importance']}] {m['content']}")

# Test 3: pure task, nothing persistent
convo_task = make_convo([
    ("What is 2 + 2?", "4."),
    ("What's the capital of France?", "Paris."),
])

print("\n=== Test 3: Pure task — should extract nothing ===")
result = extract_memories(convo_task)
print(f"  Extracted: {result} (expected [])")

# Test 4: contradiction
convo_contradiction = make_convo([
    ("I hate bullet points, please never use them.", "Understood, no bullet points."),
    ("Actually, use bullet points for everything.", "Got it, I'll use bullet points."),
])

print("\n=== Test 4: Contradiction ===")
for m in extract_memories(convo_contradiction):
    print(f"  [{m['category']} | {m['importance']}] {m['content']}")

# Test 5: deduplication check
print("\n=== Test 5: Deduplication ===")
init_db()
save_memory("test_dup", "s1", "User prefers bullet points in responses.", "preference", 0.9)
save_memory("test_dup", "s1", "User's name is Sam.", "preference", 0.95)
save_memory("test_dup", "s1", "User is working on Project Helios.", "fact", 0.7)

candidates = [
    "User likes using bullet points.",          # near-dup of first
    "User enjoys hiking on weekends.",           # genuinely new
    "Sam is the user's name.",                  # near-dup of second
]

for candidate in candidates:
    is_dup = check_duplicate("test_dup", candidate)
    status = "DUPLICATE — skip" if is_dup else "NEW — save"
    print(f"  '{candidate[:50]}' → {status}")