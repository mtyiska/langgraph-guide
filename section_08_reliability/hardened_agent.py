import sys
import os
import argparse
sys.path.append("..")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from graph import build_graph, make_initial_state

console = Console()


def run_query(query: str, max_iterations: int = 12, verbose: bool = False) -> dict:
    graph = build_graph()
    initial = make_initial_state(query, max_iterations=max_iterations)

    if verbose:
        console.print("\n[dim]Graph structure:[/dim]")
        graph.get_graph().print_ascii()

    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print("[yellow]Running hardened agent...[/yellow]\n")

    for step in graph.stream(initial, stream_mode="updates"):
        node_name = list(step.keys())[0]
        if verbose:
            console.print(f"  [dim]→ {node_name}[/dim]")

    result = graph.invoke(initial)
    return result


def print_result(result: dict):
    status = result.get("status", "unknown")
    status_colour = {
        "complete": "green",
        "degraded": "yellow",
        "blocked": "red",
        "failed": "red",
    }.get(status, "white")

    console.print(Panel(
        result.get("final_answer", "No answer produced."),
        title=f"[bold {status_colour}]Answer ({status.upper()})[/bold {status_colour}]",
        border_style=status_colour,
    ))

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Field", style="dim")
    table.add_column("Value")

    table.add_row("Status", f"[{status_colour}]{status}[/{status_colour}]")
    table.add_row("Iterations", str(result.get("iteration_count", 0)))
    table.add_row("Loop detected", str(result.get("loop_detected", False)))
    table.add_row("Guardrail triggered", str(result.get("guardrail_triggered", False)))
    table.add_row("Output valid", str(result.get("output_valid", False)))
    table.add_row("Injection detected", str(result.get("injection_detected", False)))
    table.add_row("Answer complete", str(result.get("answer_complete", False)))

    if result.get("failure_reason"):
        table.add_row("Failure reason", result["failure_reason"])

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Hardened local document research assistant."
    )
    parser.add_argument("query", nargs="?", help="Research question.")
    parser.add_argument(
        "--max-iterations", type=int, default=12,
        help="Maximum agent iterations before forcing degradation (default 12)."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print node execution trace."
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Interactive REPL mode."
    )
    args = parser.parse_args()

    if args.interactive or not args.query:
        console.print("[bold blue]Hardened Research Assistant[/bold blue]")
        console.print("Type [bold]/exit[/bold] to quit, [bold]/graph[/bold] to show structure.\n")

        graph = build_graph()
        console.print("[dim]Graph compiled.[/dim]\n")

        while True:
            try:
                query = input("Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not query:
                continue
            if query == "/exit":
                break
            if query == "/graph":
                graph.get_graph().print_ascii()
                continue

            initial = make_initial_state(query, max_iterations=args.max_iterations)
            result = graph.invoke(initial)
            print_result(result)
            print()

    else:
        result = run_query(
            args.query,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
        )
        print_result(result)


if __name__ == "__main__":
    main()