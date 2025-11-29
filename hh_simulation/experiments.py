from __future__ import annotations

import os
import csv
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .market import Market


def calculate_rank_differences(market: Market, period_results: List) -> Tuple[float, float]:
    """
    Calculate average absolute rank differences for workers and firms.
    
    Returns:
        Tuple of (avg_worker_rank_diff, avg_firm_rank_diff)
        - avg_worker_rank_diff: Average absolute difference between worker rank and matched firm rank
        - avg_firm_rank_diff: Average absolute difference between firm rank and matched worker rank
    """
    # Create worker rank mapping: worker_id -> rank (1-indexed)
    # Workers are sorted by quality (descending), so workers[0] has rank 1
    worker_rank_map = {}
    for rank_idx, worker in enumerate(market.workers):
        worker_rank_map[worker.worker_id] = rank_idx + 1
    
    # Collect all matches across both periods
    all_matches = []
    for pr in period_results:
        all_matches.extend(pr.matches)
    
    if not all_matches:
        return 0.0, 0.0
    
    worker_rank_diffs = []
    firm_rank_diffs = []
    
    for match in all_matches:
        # Get worker_id (resolve agent_id if period 0)
        if match.period == 0:
            # Period 0: agent_id needs to be resolved to worker_id
            worker_id = market._resolve_agent_to_worker(match.worker_id)
        else:
            # Period 1: worker_id is already the actual worker_id
            worker_id = match.worker_id
        
        # Get firm
        firm = market.firm_dict[match.firm_id]
        
        # Get ranks
        worker_rank = worker_rank_map.get(worker_id, 0)
        firm_rank = firm.prestige  # Prestige is already 1-indexed
        
        if worker_rank > 0:  # Only if worker was found
            # Calculate absolute difference
            rank_diff = abs(worker_rank - firm_rank)
            worker_rank_diffs.append(rank_diff)
            firm_rank_diffs.append(rank_diff)  # Same difference from firm's perspective
    
    avg_worker_rank_diff = np.mean(worker_rank_diffs) if worker_rank_diffs else 0.0
    avg_firm_rank_diff = np.mean(firm_rank_diffs) if firm_rank_diffs else 0.0
    
    return avg_worker_rank_diff, avg_firm_rank_diff


def run_experiment_1(
    num_workers: int = 200,
    num_firms: int = 50,
    gamma: float = 0.75,
    alpha: float = 0.5,
    matching_algorithm: str = "hungarian",
    seed: int = 42,
) -> Dict[str, List]:
    """
    Experiment 1: Vary number of headhunters from 1 to num_firms.
    
    Returns dictionary with:
    - num_headhunters: List of headhunter counts
    - total_matches: List of total matches for each count
    - headhunter_welfare: List of headhunter welfare for each count
    - firm_welfare: List of firm welfare for each count
    - worker_welfare: List of worker welfare for each count
    - match_welfare: List of total match welfare for each count
    - avg_worker_rank_diff: List of average absolute rank differences for workers
    - avg_firm_rank_diff: List of average absolute rank differences for firms
    - early_matches: List of number of matches in early period (t=0)
    - regular_matches: List of number of matches in regular period (t=1)
    """
    results = {
        "num_headhunters": [],
        "total_matches": [],
        "early_matches": [],
        "regular_matches": [],
        "headhunter_welfare": [],
        "firm_welfare": [],
        "worker_welfare": [],
        "match_welfare": [],
        "avg_worker_rank_diff": [],
        "avg_firm_rank_diff": [],
    }
    
    # Vary headhunters from 1 to num_firms
    for num_headhunters in tqdm(range(1, num_firms + 1), desc="Experiment 1: Varying headhunters"):
        # Create market with current number of headhunters
        market = Market.random_market(
            num_workers=num_workers,
            num_firms=num_firms,
            num_headhunters=num_headhunters,
            gamma=gamma,
            alpha=alpha,
            matching_algorithm=matching_algorithm,
            seed=seed,
        )
        
        # Run simulation
        period_results = market.run()
        
        # Collect all matches across both periods
        all_matches = []
        early_matches = []
        regular_matches = []
        for pr in period_results:
            all_matches.extend(pr.matches)
            if pr.period == 0:
                early_matches = pr.matches
            else:
                regular_matches = pr.matches
        
        # Calculate welfare
        welfare = market.calculate_welfare(all_matches)
        
        # Calculate rank differences
        avg_worker_rank_diff, avg_firm_rank_diff = calculate_rank_differences(market, period_results)
        
        # Store results
        results["num_headhunters"].append(num_headhunters)
        results["total_matches"].append(len(all_matches))
        results["early_matches"].append(len(early_matches))
        results["regular_matches"].append(len(regular_matches))
        results["headhunter_welfare"].append(welfare.headhunter_welfare)
        results["firm_welfare"].append(welfare.firm_welfare)
        results["worker_welfare"].append(welfare.worker_welfare)
        results["match_welfare"].append(welfare.match_welfare)
        results["avg_worker_rank_diff"].append(avg_worker_rank_diff)
        results["avg_firm_rank_diff"].append(avg_firm_rank_diff)
    
    return results


def run_experiment_2(
    num_workers: int = 200,
    num_firms: int = 50,
    num_headhunters: int = 10,
    gamma: float = 0.75,
    matching_algorithm: str = "hungarian",
    seed: int = 42,
    alpha_start: float = 0.0,
    alpha_end: float = 1.0,
    alpha_step: float = 0.01,
) -> Dict[str, List]:
    """
    Experiment 2: Vary alpha from 0 to 1 in increments of 0.01.
    
    Returns dictionary with:
    - alpha: List of alpha values
    - total_matches: List of total matches for each alpha
    - headhunter_welfare: List of headhunter welfare for each alpha
    - firm_welfare: List of firm welfare for each alpha
    - worker_welfare: List of worker welfare for each alpha
    - match_welfare: List of total match welfare for each alpha
    - avg_worker_rank_diff: List of average absolute rank differences for workers
    - avg_firm_rank_diff: List of average absolute rank differences for firms
    - early_matches: List of number of matches in early period (t=0)
    - regular_matches: List of number of matches in regular period (t=1)
    """
    results = {
        "alpha": [],
        "total_matches": [],
        "early_matches": [],
        "regular_matches": [],
        "headhunter_welfare": [],
        "firm_welfare": [],
        "worker_welfare": [],
        "match_welfare": [],
        "avg_worker_rank_diff": [],
        "avg_firm_rank_diff": [],
    }
    
    # Generate alpha values
    alpha_values = np.arange(alpha_start, alpha_end + alpha_step, alpha_step)
    
    # Vary alpha
    for alpha in tqdm(alpha_values, desc="Experiment 2: Varying alpha"):
        # Create market with current alpha
        market = Market.random_market(
            num_workers=num_workers,
            num_firms=num_firms,
            num_headhunters=num_headhunters,
            gamma=gamma,
            alpha=alpha,
            matching_algorithm=matching_algorithm,
            seed=seed,
        )
        
        # Run simulation
        period_results = market.run()
        
        # Collect all matches across both periods
        all_matches = []
        early_matches = []
        regular_matches = []
        for pr in period_results:
            all_matches.extend(pr.matches)
            if pr.period == 0:
                early_matches = pr.matches
            else:
                regular_matches = pr.matches
        
        # Calculate welfare
        welfare = market.calculate_welfare(all_matches)
        
        # Calculate rank differences
        avg_worker_rank_diff, avg_firm_rank_diff = calculate_rank_differences(market, period_results)
        
        # Store results
        results["alpha"].append(alpha)
        results["total_matches"].append(len(all_matches))
        results["early_matches"].append(len(early_matches))
        results["regular_matches"].append(len(regular_matches))
        results["headhunter_welfare"].append(welfare.headhunter_welfare)
        results["firm_welfare"].append(welfare.firm_welfare)
        results["worker_welfare"].append(welfare.worker_welfare)
        results["match_welfare"].append(welfare.match_welfare)
        results["avg_worker_rank_diff"].append(avg_worker_rank_diff)
        results["avg_firm_rank_diff"].append(avg_firm_rank_diff)
    
    return results


def plot_experiment_1(results: Dict[str, List], save_dir: str = "hh_simulation/graphs") -> None:
    """Plot results from Experiment 1."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment 1: Effect of Number of Headhunters\n(Baseline: 200 workers, 50 firms, γ=0.75, α=0.5)", 
                 fontsize=14, fontweight='bold')
    
    num_headhunters = results["num_headhunters"]
    
    # Plot 1: Total matches
    axes[0, 0].plot(num_headhunters, results["total_matches"], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel("Number of Headhunters")
    axes[0, 0].set_ylabel("Total Matches")
    axes[0, 0].set_title("Total Matches vs Number of Headhunters")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Headhunter welfare
    axes[0, 1].plot(num_headhunters, results["headhunter_welfare"], 'g-', linewidth=2, marker='o', markersize=4)
    axes[0, 1].set_xlabel("Number of Headhunters")
    axes[0, 1].set_ylabel("Headhunter Welfare")
    axes[0, 1].set_title("Headhunter Welfare vs Number of Headhunters")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Firm welfare
    axes[1, 0].plot(num_headhunters, results["firm_welfare"], 'r-', linewidth=2, marker='o', markersize=4)
    axes[1, 0].set_xlabel("Number of Headhunters")
    axes[1, 0].set_ylabel("Firm Welfare")
    axes[1, 0].set_title("Firm Welfare vs Number of Headhunters")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Worker welfare
    axes[1, 1].plot(num_headhunters, results["worker_welfare"], 'm-', linewidth=2, marker='o', markersize=4)
    axes[1, 1].set_xlabel("Number of Headhunters")
    axes[1, 1].set_ylabel("Worker Welfare")
    axes[1, 1].set_title("Worker Welfare vs Number of Headhunters")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "experiment_1_headhunters.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_experiment_2(results: Dict[str, List], save_dir: str = "hh_simulation/graphs") -> None:
    """Plot results from Experiment 2."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment 2: Effect of Alpha (α)\n(Baseline: 200 workers, 50 firms, 10 headhunters, γ=0.75)", 
                 fontsize=14, fontweight='bold')
    
    alpha = results["alpha"]
    
    # Plot 1: Total matches
    axes[0, 0].plot(alpha, results["total_matches"], 'b-', linewidth=2, marker='o', markersize=2)
    axes[0, 0].set_xlabel("Alpha (α)")
    axes[0, 0].set_ylabel("Total Matches")
    axes[0, 0].set_title("Total Matches vs Alpha")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Headhunter welfare
    axes[0, 1].plot(alpha, results["headhunter_welfare"], 'g-', linewidth=2, marker='o', markersize=2)
    axes[0, 1].set_xlabel("Alpha (α)")
    axes[0, 1].set_ylabel("Headhunter Welfare")
    axes[0, 1].set_title("Headhunter Welfare vs Alpha")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Firm welfare
    axes[1, 0].plot(alpha, results["firm_welfare"], 'r-', linewidth=2, marker='o', markersize=2)
    axes[1, 0].set_xlabel("Alpha (α)")
    axes[1, 0].set_ylabel("Firm Welfare")
    axes[1, 0].set_title("Firm Welfare vs Alpha")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Worker welfare
    axes[1, 1].plot(alpha, results["worker_welfare"], 'm-', linewidth=2, marker='o', markersize=2)
    axes[1, 1].set_xlabel("Alpha (α)")
    axes[1, 1].set_ylabel("Worker Welfare")
    axes[1, 1].set_title("Worker Welfare vs Alpha")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "experiment_2_alpha.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_early_vs_regular_matches_headhunters(results: Dict[str, List], save_dir: str = "hh_simulation/graphs") -> None:
    """Plot early vs regular period matches for Experiment 1 (varying headhunters)."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle("Experiment 1: Early vs Regular Period Matches\n(Baseline: 200 workers, 50 firms, γ=0.75, α=0.5)", 
                 fontsize=14, fontweight='bold')
    
    num_headhunters = results["num_headhunters"]
    
    # Calculate percentages
    percent_early = []
    percent_regular = []
    for i in range(len(num_headhunters)):
        total = results["total_matches"][i]
        if total > 0:
            percent_early.append((results["early_matches"][i] / total) * 100)
            percent_regular.append((results["regular_matches"][i] / total) * 100)
        else:
            percent_early.append(0.0)
            percent_regular.append(0.0)
    
    ax.plot(num_headhunters, percent_early, 'b-', linewidth=2, marker='o', markersize=4, label='Early Period (t=0)')
    ax.plot(num_headhunters, percent_regular, 'r-', linewidth=2, marker='s', markersize=4, label='Regular Period (t=1)')
    ax.set_xlabel("Number of Headhunters")
    ax.set_ylabel("Percentage of Matches (%)")
    ax.set_title("Percentage of Matches in Early vs Regular Period")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "experiment_1_early_vs_regular_matches.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_early_vs_regular_matches_alpha(results: Dict[str, List], save_dir: str = "hh_simulation/graphs") -> None:
    """Plot early vs regular period matches for Experiment 2 (varying alpha)."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle("Experiment 2: Early vs Regular Period Matches\n(Baseline: 200 workers, 50 firms, 10 headhunters, γ=0.75)", 
                 fontsize=14, fontweight='bold')
    
    alpha = results["alpha"]
    
    # Calculate percentages
    percent_early = []
    percent_regular = []
    for i in range(len(alpha)):
        total = results["total_matches"][i]
        if total > 0:
            percent_early.append((results["early_matches"][i] / total) * 100)
            percent_regular.append((results["regular_matches"][i] / total) * 100)
        else:
            percent_early.append(0.0)
            percent_regular.append(0.0)
    
    ax.plot(alpha, percent_early, 'b-', linewidth=2, marker='o', markersize=2, label='Early Period (t=0)')
    ax.plot(alpha, percent_regular, 'r-', linewidth=2, marker='s', markersize=2, label='Regular Period (t=1)')
    ax.set_xlabel("Alpha (α)")
    ax.set_ylabel("Percentage of Matches (%)")
    ax.set_title("Percentage of Matches in Early vs Regular Period")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "experiment_2_early_vs_regular_matches.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def save_experiment_1_csv(results: Dict[str, List], save_dir: str = "hh_simulation/graphs") -> None:
    """Save Experiment 1 results to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    
    csv_path = os.path.join(save_dir, "experiment_1_headhunters.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([
            "num_headhunters",
            "total_matches",
            "early_matches",
            "regular_matches",
            "headhunter_welfare",
            "firm_welfare",
            "worker_welfare",
            "match_welfare",
            "avg_worker_rank_diff",
            "avg_firm_rank_diff",
        ])
        
        # Write data rows
        for i in range(len(results["num_headhunters"])):
            writer.writerow([
                results["num_headhunters"][i],
                results["total_matches"][i],
                results["early_matches"][i],
                results["regular_matches"][i],
                results["headhunter_welfare"][i],
                results["firm_welfare"][i],
                results["worker_welfare"][i],
                results["match_welfare"][i],
                results["avg_worker_rank_diff"][i],
                results["avg_firm_rank_diff"][i],
            ])
    
    print(f"Saved CSV to {csv_path}")


def save_experiment_2_csv(results: Dict[str, List], save_dir: str = "hh_simulation/graphs") -> None:
    """Save Experiment 2 results to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    
    csv_path = os.path.join(save_dir, "experiment_2_alpha.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([
            "alpha",
            "total_matches",
            "early_matches",
            "regular_matches",
            "headhunter_welfare",
            "firm_welfare",
            "worker_welfare",
            "match_welfare",
            "avg_worker_rank_diff",
            "avg_firm_rank_diff",
        ])
        
        # Write data rows
        for i in range(len(results["alpha"])):
            writer.writerow([
                results["alpha"][i],
                results["total_matches"][i],
                results["early_matches"][i],
                results["regular_matches"][i],
                results["headhunter_welfare"][i],
                results["firm_welfare"][i],
                results["worker_welfare"][i],
                results["match_welfare"][i],
                results["avg_worker_rank_diff"][i],
                results["avg_firm_rank_diff"][i],
            ])
    
    print(f"Saved CSV to {csv_path}")


def run_all_experiments(
    num_workers: int = 200,
    num_firms: int = 50,
    gamma: float = 0.75,
    alpha: float = 0.5,
    matching_algorithm: str = "hungarian",
    seed: int = 42,
    save_dir: str = "hh_simulation/graphs",
) -> None:
    """
    Run both experiments and generate plots and CSV files.
    
    Baseline settings (from __main__.py):
    - num_workers: 200
    - num_firms: 50
    - gamma: 0.75
    - alpha: 0.5 (used in experiment 1, varied in experiment 2)
    - matching_algorithm: hungarian
    - seed: 42
    
    Outputs:
    - PNG plots saved to save_dir
    - CSV files with all metrics including rank differences saved to save_dir
    """
    print("=" * 60)
    print("Running Experiment 1: Varying Number of Headhunters")
    print("=" * 60)
    results_1 = run_experiment_1(
        num_workers=num_workers,
        num_firms=num_firms,
        gamma=gamma,
        alpha=alpha,
        matching_algorithm=matching_algorithm,
        seed=seed,
    )
    plot_experiment_1(results_1, save_dir=save_dir)
    plot_early_vs_regular_matches_headhunters(results_1, save_dir=save_dir)
    save_experiment_1_csv(results_1, save_dir=save_dir)
    
    print("\n" + "=" * 60)
    print("Running Experiment 2: Varying Alpha")
    print("=" * 60)
    results_2 = run_experiment_2(
        num_workers=num_workers,
        num_firms=num_firms,
        num_headhunters=10,
        gamma=gamma,
        matching_algorithm=matching_algorithm,
        seed=seed,
    )
    plot_experiment_2(results_2, save_dir=save_dir)
    plot_early_vs_regular_matches_alpha(results_2, save_dir=save_dir)
    save_experiment_2_csv(results_2, save_dir=save_dir)
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run experiments with baseline settings
    run_all_experiments(
        num_workers=200,
        num_firms=50,
        gamma=0.75,
        alpha=0.5,
        matching_algorithm="hungarian",
        seed=42,
        save_dir="hh_simulation/graphs",
    )

