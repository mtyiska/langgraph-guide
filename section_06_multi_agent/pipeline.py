import sys
import os
import time
import argparse
sys.path.append("..")

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from shared_state import SharedState, make_initial_state
from supervisor import build_supervisor_node, route_after_supervisor
from researcher_agent import build_researcher_graph
from writer_agent import build_writer_graph
from config import PRIMARY_MODEL

# Import retrieval tools from section 5
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "section_05_rag"))
try:
    from ingestion import ingest_documents, get_collection
    from retrieval_tools import build_retrieval_tools
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("[WARNING] Section 5 RAG not found — researcher will use placeholder tools")

console = Console()


def build_pipeline(parallel: bool = False):
    # Build retrieval tools
    if RAG_AVAILABLE:
        try:
            ingest_documents(verbose=False)
            collection = get_collection()
            retrieval_tools = build_retrieval_tools(collection)
        except Exception as e:
            console.print(f"[yellow]RAG init failed: {e}. Using placeholder tools.[/yellow]")
            retrieval_tools = _build_placeholder_tools()
    else:
        retrieval_tools = _build_placeholder_tools()

    # Build subgraphs
    researcher_graph = build_researcher_graph(retrieval_tools)
    writer_graph = build_writer_graph()
    supervisor_node = build_supervisor_node()

    # ── Parent graph ──────────────────────────────────────────────────────────

    builder = StateGraph(SharedState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_graph)
    builder.add_node("writer", writer_graph)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "END": END,
        }
    )

    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")

    return builder.compile()


def _build_placeholder_tools():
    from langchain_core.tools import tool

    @tool
    def list_document_sources() -> str:
        """List available documents."""
        return "Available: project_atlas.md, budget_overview.md, technical_decisions.md, q3_report.md"

    @tool
    def search_documents(query: str, n_results: int = 5, source_filter: str = None) -> str:
        """Search documents for relevant information."""
        return (
            f"[Placeholder result for '{query}']\n"
            "Project Atlas is a data platform with $320,000 budget. "
            "Tech stack: FastAPI, React, PostgreSQL. "
            "Launch: November 1 2024. Team lead: Jordan Kim."
        )

    @tool
    def get_document_summary(source: str) -> str:
        """Get overview of a document."""
        return f"[Placeholder summary for {source}] This document covers project information."

    return [list_document_sources, search_documents, get_document_summary]


def run_pipeline(request: str, stream: bool = False, parallel: bool = False) -> dict:
    graph = build_pipeline(parallel=parallel)

    initial_state = make_initial_state(request, max_iterations=4)

    console.print(f"\n[bold blue]Pipeline starting[/bold blue]")
    console.print(f"Request: {request}\n")

    if stream:
        result = _run_streamed(graph, initial_state)
    else:
        result = _run_silent(graph, initial_state)

    return result


def _run_silent(graph, initial_state: SharedState) -> dict:
    console.print("[yellow]Running pipeline...[/yellow]")
    t0 = time.perf_counter()

    worker_sequence = []
    for step in graph.stream(initial_state, stream_mode="updates"):
        node_name = list(step.keys())[0]
        if node_name != "supervisor":
            worker_sequence.append(node_name)
            console.print(f"  [dim]→ {node_name} running...[/dim]")
        else:
            updates = step[node_name]
            next_w = updates.get("next_worker")
            if next_w:
                console.print(f"  [dim cyan]supervisor → {next_w}[/dim cyan]")

    duration = time.perf_counter() - t0
    result = graph.invoke(initial_state)
    result["_duration"] = duration
    result["_worker_sequence"] = worker_sequence
    return result


def _run_streamed(graph, initial_state: SharedState) -> dict:
    console.print("[yellow]Streaming pipeline output...[/yellow]\n")
    t0 = time.perf_counter()
    worker_sequence = []

    for step in graph.stream(initial_state, stream_mode="updates"):
        node_name = list(step.keys())[0]
        updates = step[node_name]

        if node_name == "supervisor":
            next_w = updates.get("next_worker")
            status = updates.get("status")
            if next_w:
                console.print(f"[cyan]▶ Supervisor → {next_w.upper()}[/cyan]")
            if status == "complete":
                console.print("[cyan]▶ Supervisor → FINISH[/cyan]")

        elif node_name == "researcher":
            worker_sequence.append("researcher")
            results = updates.get("worker_results", [])
            for r in results:
                if r.get("worker") == "researcher":
                    console.print(Panel(
                        r.get("result", "")[:600] + "...",
                        title="[green]Researcher Output[/green]",
                        border_style="green"
                    ))

        elif node_name == "writer":
            worker_sequence.append("writer")
            results = updates.get("worker_results", [])
            for r in results:
                if r.get("worker") == "writer":
                    console.print(Panel(
                        r.get("result", "")[:800] + "...",
                        title="[blue]Writer Draft[/blue]",
                        border_style="blue"
                    ))

    duration = time.perf_counter() - t0
    result = graph.invoke(initial_state)
    result["_duration"] = duration
    result["_worker_sequence"] = worker_sequence
    return result


def print_final_output(result: dict):
    console.print("\n" + "=" * 70)
    console.print(Panel(
        result.get("final_output", "No output produced."),
        title="[bold green]Final Output[/bold green]",
        border_style="green"
    ))

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    duration = result.get("_duration", 0)
    workers = result.get("_worker_sequence", [])
    n_results = len(result.get("worker_results", []))

    table.add_row("Status", result.get("status", "unknown"))
    table.add_row("Worker sequence", " → ".join(workers) if workers else "unknown")
    table.add_row("Worker calls", str(n_results))
    table.add_row("Iterations", str(result.get("iteration_count", 0)))
    table.add_row("Duration", f"{duration:.1f}s")

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Multi-agent writing pipeline")
    parser.add_argument("request", nargs="?", help="The writing request")
    parser.add_argument("--stream", action="store_true", help="Stream output as it generates")
    parser.add_argument("--parallel", action="store_true", help="Use parallel researcher mode")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive or not args.request:
        console.print("[bold blue]Multi-Agent Writing Pipeline[/bold blue]")
        console.print("Commands: /exit /graph\n")

        graph = build_pipeline(parallel=args.parallel)

        while True:
            try:
                request = input("Request: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not request:
                continue
            if request == "/exit":
                break
            if request == "/graph":
                graph.get_graph().print_ascii()
                continue

            initial_state = make_initial_state(request, max_iterations=12)
            result = _run_streamed(graph, initial_state) if args.stream else _run_silent(graph, initial_state)
            print_final_output(result)
            print()
    else:
        result = run_pipeline(args.request, stream=args.stream, parallel=args.parallel)
        print_final_output(result)


if __name__ == "__main__":
    main()