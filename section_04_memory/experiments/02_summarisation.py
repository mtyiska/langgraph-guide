import sys
sys.path.append(".")

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from summariser import summarise_messages, estimate_tokens, maybe_summarise

# Build a fake 15-turn conversation
def make_conversation():
    turns = [
        ("Hi, my name is Jordan and I prefer concise answers.", "Hello Jordan! Got it — I'll keep things brief."),
        ("What's the project budget?", "The approved budget is $180,000 for Phase 1."),
        ("How much has been spent?", "As of the last update, $42,000 has been spent — about 23%."),
        ("Who is the lead engineer?", "Dave Okafor is the lead engineer."),
        ("When is the internal demo?", "The internal demo is scheduled for May 3, 2024."),
        ("Is the project on track?", "Status is YELLOW — there's a blocker with the PaymentCo API."),
        ("What's the PaymentCo issue?", "Sandbox access was requested on March 20 and still hasn't been granted."),
        ("Who handles payments integration?", "Jordan Kim joins May 1 and will own the PaymentCo work."),
        ("How many endpoints are done?", "8 of 20 backend endpoints are complete — 40%."),
        ("What's the frontend status?", "The component library is 25% complete."),
        ("Any risks?", "Two risks: PaymentCo API delay and Priya Nair is out sick this week."),
        ("What does Priya do?", "Priya Nair is the senior designer — 3 design tasks are delayed."),
        ("When are wireframes due?", "Wireframes were due March 29 — they've been approved after one revision."),
        ("Who is the QA lead?", "Marcus Webb leads QA and is writing the test plan."),
        ("Thanks, that's all for now.", "Happy to help Jordan! Reach out anytime."),
    ]
    messages = []
    for human, ai in turns:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    return messages

conversation = make_conversation()

print(f"=== Conversation stats ===")
print(f"Messages: {len(conversation)}")
print(f"Estimated tokens: {estimate_tokens(conversation)}")

print("\n=== Summary of 15-turn conversation ===")
summary = summarise_messages(conversation)
print(summary)

print("\n=== Boundary case: 2-turn conversation ===")
short = [HumanMessage(content="Hi"), AIMessage(content="Hello!")]
print(summarise_messages(short))

print("\n=== Boundary case: only tool messages ===")
tool_msgs = [
    AIMessage(content="", tool_calls=[{"id": "t1", "name": "read_file", "args": {"path": "notes.txt"}}]),
    ToolMessage(content="File contents: some data", tool_call_id="t1"),
]
print(summarise_messages(tool_msgs) or "[empty — nothing meaningful to summarise]")

print("\n=== maybe_summarise: threshold test ===")
updated_msgs, updated_summary, did_summarise = maybe_summarise(
    [SystemMessage(content="You are an assistant.")] + conversation,
    summary_so_far="",
    context_window=2000,  # low threshold to force summarisation
    threshold=0.3,
    keep_last_n_turns=2
)
print(f"Summarised: {did_summarise}")
print(f"Messages remaining: {len(updated_msgs)}")
print(f"New summary:\n{updated_summary}")