from __future__ import annotations

from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from .market import Market

app = typer.Typer(add_completion=False)
console = Console()

'''
Use below to run

python -m hh_simulation \
    --num-workers 50 \
    --num-firms 20 \
    --num-headhunters 5 \
    --gamma 0.5 \
    --alpha 0.1 \
    --matching-algorithm hungarian \
    --seed 42
'''

@app.command()
def run(
    num_workers: int = typer.Option(50, help="Number of workers"),
    num_firms: int = typer.Option(10, help="Number of firms"),
    num_headhunters: int = typer.Option(3, help="Number of headhunters"),
    gamma: float = typer.Option(0.5, help="Outside option scaling factor γ (baseline utility = γ * (quality/max_quality) * max_firm_value)"),
    alpha: float = typer.Option(0.5, help="Headhunter utility weight α (u_h = α·μ + (1-α)·η)"),
    matching_algorithm: str = typer.Option("hungarian", help="Matching algorithm: 'hungarian' (fast, O(n³)) or 'enumerative' (slow, O(2^n), but exhaustive)"),
    seed: Optional[int] = typer.Option(None, help="Random seed"),
) -> None:
    """Run a two-period market simulation."""
    if matching_algorithm not in ["hungarian", "enumerative"]:
        console.print(f"[red]Error: matching_algorithm must be 'hungarian' or 'enumerative', got '{matching_algorithm}'[/red]")
        raise typer.Exit(1)
    
    console.print("[bold blue]Two-Period Market Simulation[/bold blue]")
    console.print(f"Workers: {num_workers}, Firms: {num_firms}, Headhunters: {num_headhunters}")
    console.print(f"γ (gamma): {gamma}, α (alpha): {alpha}, Algorithm: {matching_algorithm}\n")
    
    market = Market.random_market(
        num_workers=num_workers,
        num_firms=num_firms,
        num_headhunters=num_headhunters,
        gamma=gamma,
        alpha=alpha,
        matching_algorithm=matching_algorithm,
        seed=seed,
    )
    
    results = market.run()
    
    # Display results for each period
    for period_results in results:
        period_name = "Early Phase (t=0)" if period_results.period == 0 else "Regular Phase (t=1)"
        console.print(f"\n[bold yellow]{period_name}[/bold yellow]")
        console.print(f"Matches: {len(period_results.matches)}")
        console.print(f"Unmatched Workers: {len(period_results.unmatched_workers)}")
        console.print(f"Unmatched Firms: {len(period_results.unmatched_firms)}")
        
        if period_results.matches:
            table = Table(title=f"Matches in {period_name}")
            table.add_column("Worker", justify="right")
            table.add_column("Firm", justify="right")
            table.add_column("Headhunter", justify="right")
            table.add_column("Quality", justify="right", style="cyan")
            table.add_column("Worker Utility", justify="right", style="green")
            table.add_column("Firm Utility", justify="right", style="blue")
            table.add_column("Headhunter Utility", justify="right", style="magenta")
            
            for match in period_results.matches[:10]:  # Show first 10 matches
                if match.period == 0:
                    # Period 0: Show agent ID and expected quality
                    table.add_row(
                        f"A{match.worker_id} (E[q]={match.expected_quality:.2f})",
                        f"F{match.firm_id} (r={market.firm_dict[match.firm_id].prestige})",
                        f"H{match.headhunter_id}" if match.headhunter_id is not None else "None",
                        f"{match.expected_quality:.3f}",
                        f"{match.worker_utility:.3f}",
                        f"{match.firm_utility:.3f}",
                        f"{match.headhunter_utility:.3f}",
                    )
                else:
                    # Period 1: Show worker ID and true quality
                    worker = market.worker_dict[match.worker_id]
                    table.add_row(
                        f"W{match.worker_id} (q={worker.quality:.2f})",
                        f"F{match.firm_id} (r={market.firm_dict[match.firm_id].prestige})",
                        f"H{match.headhunter_id}" if match.headhunter_id is not None else "None",
                        f"{match.expected_quality:.3f}",
                        f"{match.worker_utility:.3f}",
                        f"{match.firm_utility:.3f}",
                        f"{match.headhunter_utility:.3f}",
                    )
            
            console.print(table)
            
            if len(period_results.matches) > 10:
                console.print(f"... and {len(period_results.matches) - 10} more matches")
    
    # Summary statistics
    console.print("\n[bold]Summary Statistics[/bold]")
    total_matches = sum(len(r.matches) for r in results)
    early_matches = len(results[0].matches)
    regular_matches = len(results[1].matches)
    console.print(f"Total matches: {total_matches} ({100*total_matches/min(num_workers, num_firms):.1f}%)")
    console.print(f"Early phase matches: {early_matches} ({100*early_matches/total_matches:.1f}%)" if total_matches > 0 else "Early phase matches: 0")
    console.print(f"Regular phase matches: {regular_matches} ({100*regular_matches/total_matches:.1f}%)" if total_matches > 0 else "Regular phase matches: 0")
    
    # Welfare calculations
    console.print("\n[bold]Welfare Statistics[/bold]")
    
    # Per-period welfare
    for period_results in results:
        period_name = "Early Phase (t=0)" if period_results.period == 0 else "Regular Phase (t=1)"
        welfare = market.calculate_welfare(period_results.matches)
        console.print(f"\n{period_name} Welfare:")
        console.print(f"  Headhunter Welfare: {welfare.headhunter_welfare:.3f}")
        console.print(f"  Firm Welfare: {welfare.firm_welfare:.3f}")
        console.print(f"  Worker Welfare: {welfare.worker_welfare:.3f}")
        console.print(f"  Total Match Welfare: {welfare.match_welfare:.3f}")
    
    # Total welfare across all periods
    all_matches = [m for r in results for m in r.matches]
    total_welfare = market.calculate_welfare(all_matches)
    console.print(f"\n[bold]Total Welfare (All Periods):[/bold]")
    console.print(f"  Headhunter Welfare: {total_welfare.headhunter_welfare:.3f}")
    console.print(f"  Firm Welfare: {total_welfare.firm_welfare:.3f}")
    console.print(f"  Worker Welfare: {total_welfare.worker_welfare:.3f}")
    console.print(f"  Total Match Welfare: {total_welfare.match_welfare:.3f}")
    
    console.print("\n[green]Simulation complete.[/green]")


if __name__ == "__main__":
    app()
