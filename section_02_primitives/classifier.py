import sys
import re
sys.path.append("..")

from typing import TypedDict, Optional, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ── State ────────────────────────────────────────────────────────────────────

class DocumentState(TypedDict):
    raw_text: str
    filename: str
    word_count: int
    char_count: int
    line_count: int
    has_numbers: bool
    has_urls: bool
    doc_type: Optional[str]
    urgency: Optional[str]
    confidence: float
    summary: str
    tags: Annotated[list, add]
    handler: Optional[str]
    output: Optional[str]
    errors: Annotated[list, add]

# ── Nodes ─────────────────────────────────────────────────────────────────────

def extract(state: DocumentState) -> dict:
    text = state["raw_text"]
    words = text.split()
    lines = text.splitlines()
    has_numbers = bool(re.search(r'\d+', text))
    has_urls = bool(re.search(r'https?://|www\.', text))

    # Summary: first 100 chars truncated at word boundary
    if len(text) <= 100:
        summary = text
    else:
        truncated = text[:100]
        last_space = truncated.rfind(" ")
        summary = truncated[:last_space] + "..." if last_space != -1 else truncated + "..."

    return {
        "word_count": len(words),
        "char_count": len(text),
        "line_count": len(lines),
        "has_numbers": has_numbers,
        "has_urls": has_urls,
        "summary": summary,
    }


def classify_type(state: DocumentState) -> dict:
    text = state["raw_text"].lower()
    word_count = state["word_count"]
    signals = 0
    doc_type = "unknown"

    invoice_keywords = ["invoice", "due", "payment", "amount", "$", "total", "bill", "overdue"]
    email_keywords = ["from:", "to:", "subject:", "re:", "@"]
    report_keywords = ["summary", "findings", "conclusion", "section", "report", "analysis"]

    invoice_hits = sum(1 for kw in invoice_keywords if kw in text)
    email_hits = sum(1 for kw in email_keywords if kw in text)
    report_hits = sum(1 for kw in report_keywords if kw in text)

    if invoice_hits >= 2:
        doc_type = "invoice"
        signals = invoice_hits
    elif email_hits >= 2:
        doc_type = "email"
        signals = email_hits
    elif report_hits >= 2 and word_count > 200:
        doc_type = "report"
        signals = report_hits
    elif word_count < 50:
        doc_type = "note"
        signals = 1
    else:
        doc_type = "unknown"
        signals = 0

    max_signals = max(invoice_hits, email_hits, report_hits, 1)
    confidence = min(signals / max(max_signals, 1), 1.0) * 0.8 + 0.1 if signals > 0 else 0.1

    return {
        "doc_type": doc_type,
        "confidence": round(confidence, 2),
        "tags": [f"type:{doc_type}"],
    }


def classify_urgency(state: DocumentState) -> dict:
    text = state["raw_text"].lower()
    raw = state["raw_text"]

    high_keywords = ["urgent", "asap", "immediately", "critical", "overdue", "today"]
    medium_keywords = ["soon", "this week", "follow up", "reminder", "follow-up"]

    has_caps = len(re.findall(r'\b[A-Z]{3,}\b', raw)) > 0
    high_hits = sum(1 for kw in high_keywords if kw in text) + (1 if has_caps else 0)
    medium_hits = sum(1 for kw in medium_keywords if kw in text)

    if high_hits >= 1:
        urgency = "high"
    elif medium_hits >= 1:
        urgency = "medium"
    else:
        urgency = "low"

    # Boost confidence if type + urgency are coherent
    confidence = state["confidence"]
    if state["doc_type"] in ("invoice", "email") and urgency == "high":
        confidence = min(confidence + 0.1, 1.0)

    return {
        "urgency": urgency,
        "confidence": round(confidence, 2),
        "tags": [f"urgency:{urgency}"],
    }


def enrich(state: DocumentState) -> dict:
    doc_type = state["doc_type"]
    urgency = state["urgency"]
    tags = []

    if doc_type == "invoice" and urgency == "high":
        tags.append("action:required")
    elif doc_type == "note" and urgency == "low":
        tags.append("action:archive")
    elif doc_type == "unknown" and urgency == "high":
        tags.append("action:review")
    elif doc_type == "report":
        tags.append("action:review")
    elif doc_type == "email":
        tags.append("action:reply")
    else:
        tags.append("action:file")

    handler_map = {
        "invoice": "handle_invoice",
        "email": "handle_email",
        "report": "handle_report",
    }
    handler = handler_map.get(doc_type, "handle_default")

    return {"tags": tags, "handler": handler}


def handle_invoice(state: DocumentState) -> dict:
    numbers = re.findall(r'\$[\d,]+\.?\d*|\d+\.\d{2}', state["raw_text"])
    numbers_str = ", ".join(numbers) if numbers else "none found"
    output = (
        f"📄 INVOICE\n"
        f"  File:     {state['filename']}\n"
        f"  Urgency:  {state['urgency'].upper()}\n"
        f"  Amounts:  {numbers_str}\n"
        f"  Tags:     {', '.join(state['tags'])}\n"
        f"  Summary:  {state['summary']}"
    )
    return {"output": output}


def handle_email(state: DocumentState) -> dict:
    lines = state["raw_text"].splitlines()
    subject = next((l for l in lines if l.lower().startswith("subject:")), "Subject: (not found)")
    output = (
        f"📧 EMAIL\n"
        f"  File:     {state['filename']}\n"
        f"  {subject}\n"
        f"  Urgency:  {state['urgency'].upper()}\n"
        f"  Tags:     {', '.join(state['tags'])}\n"
        f"  Preview:  {state['summary']}"
    )
    return {"output": output}


def handle_report(state: DocumentState) -> dict:
    output = (
        f"📊 REPORT\n"
        f"  File:      {state['filename']}\n"
        f"  Words:     {state['word_count']}\n"
        f"  Lines:     {state['line_count']}\n"
        f"  Urgency:   {state['urgency'].upper()}\n"
        f"  Tags:      {', '.join(state['tags'])}\n"
        f"  Summary:   {state['summary']}"
    )
    return {"output": output}


def handle_default(state: DocumentState) -> dict:
    urgency_str = (state.get("urgency") or "unclassified").upper()
    doc_type_str = (state.get("doc_type") or "unknown").upper()
    output = (
        f"📝 {doc_type_str}\n"
        f"  File:     {state['filename']}\n"
        f"  Urgency:  {urgency_str}\n"
        f"  Tags:     {', '.join(state['tags'])}\n"
        f"  Summary:  {state['summary']}"
    )
    return {"output": output}


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_classify(state: DocumentState) -> str:
    if state["confidence"] < 0.2:
        return "handle_default"
    return "classify_urgency"


def route_after_enrich(state: DocumentState) -> str:
    return state["handler"]


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(DocumentState)

    builder.add_node("extract", extract)
    builder.add_node("classify_type", classify_type)
    builder.add_node("classify_urgency", classify_urgency)
    builder.add_node("enrich", enrich)
    builder.add_node("handle_invoice", handle_invoice)
    builder.add_node("handle_email", handle_email)
    builder.add_node("handle_report", handle_report)
    builder.add_node("handle_default", handle_default)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "classify_type")
    builder.add_conditional_edges("classify_type", route_after_classify)
    builder.add_edge("classify_urgency", "enrich")
    builder.add_conditional_edges("enrich", route_after_enrich)
    builder.add_edge("handle_invoice", END)
    builder.add_edge("handle_email", END)
    builder.add_edge("handle_report", END)
    builder.add_edge("handle_default", END)

    return builder.compile()


# ── Runner ────────────────────────────────────────────────────────────────────

def run_document(graph, filepath: str):
    with open(filepath, "r") as f:
        text = f.read()

    filename = filepath.split("/")[-1]
    initial_state = {
        "raw_text": text,
        "filename": filename,
        "word_count": 0,
        "char_count": 0,
        "line_count": 0,
        "has_numbers": False,
        "has_urls": False,
        "doc_type": None,
        "urgency": None,
        "confidence": 0.0,
        "summary": "",
        "tags": [],
        "handler": None,
        "output": None,
        "errors": [],
    }

    console.rule(f"[bold blue]{filename}")

    # Stream updates step by step
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Node", style="yellow", width=20)
    table.add_column("Changes", style="white")

    for step in graph.stream(initial_state, stream_mode="updates"):
        node_name = list(step.keys())[0]
        updates = step[node_name]
        # Format updates cleanly
        changes = ", ".join(f"{k}={repr(v)[:60]}" for k, v in updates.items())
        table.add_row(node_name, changes)

    console.print(table)

    # Final result
    result = graph.invoke(initial_state)
    console.print(f"[bold green]Output:[/bold green]\n{result['output']}\n")


if __name__ == "__main__":
    import os
    graph = build_graph()

    print("\n=== Graph Structure ===")
    graph.get_graph().print_ascii()
    print()

    test_docs_dir = "test_docs"
    docs = sorted(os.listdir(test_docs_dir))

    for doc in docs:
        if doc.endswith(".txt"):
            run_document(graph, os.path.join(test_docs_dir, doc))