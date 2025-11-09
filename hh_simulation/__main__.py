from __future__ import annotations

from typing import Optional
import typer
from rich.console import Console

from .market import Market

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def run(
    num_workers: int = typer.Option(50, help="Number of workers"),
    num_firms: int = typer.Option(10, help="Number of firms"),
    steps: int = typer.Option(1, help="Number of simulation steps"),
    seed: Optional[int] = typer.Option(None, help="Random seed"),
) -> None:
    market = Market.random_market(num_workers=num_workers, num_firms=num_firms, seed=seed)
    history = market.run(steps=steps)
    for t, matches in enumerate(history):
        console.print(f"[bold]Step {t}[/bold]: {len(matches)} matches")
        for match in matches[: min(5, len(matches))]:
            console.print(f"  Worker {match.worker_id} -> Firm {match.firm_id} at wage {match.wage:.2f}")
    console.print("[green]Done.[/green]")


if __name__ == "__main__":
    app()


