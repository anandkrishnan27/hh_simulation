from __future__ import annotations

import os
import csv
import multiprocessing
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from .market import Market

"python -m hh_simulation.experiments"

DEFAULT_NUM_WORKERS = 500
DEFAULT_NUM_FIRMS = 125

def calculate_rank_differences(market: Market, period_results: List, period: Optional[int] = None) -> float:
    """Return the average absolute normalized rank gap between workers and firms.
    
    Args:
        market: The market object
        period_results: List of PeriodResults
        period: Optional period filter (0 for early, 1 for regular, None for all)
    
    Returns:
        Average normalized rank difference
    """
    # Map worker_id -> rank (1-indexed). Workers are pre-sorted by quality.
    worker_rank_map = {}
    for rank_idx, worker in enumerate(market.workers):
        worker_rank_map[worker.worker_id] = rank_idx + 1
    
    # Get number of workers and firms for normalization
    num_workers = len(market.workers)
    num_firms = len(market.firms)
    
    # Collect matches, optionally filtered by period
    all_matches = []
    for pr in period_results:
        if period is None or pr.period == period:
            all_matches.extend(pr.matches)
    
    if not all_matches:
        return 0.0
    
    rank_diffs = []
    
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
            # Normalize ranks
            normalized_worker_rank = worker_rank / num_workers
            normalized_firm_rank = firm_rank / num_firms
            
            # Calculate absolute difference between normalized ranks
            rank_diff = abs(normalized_worker_rank - normalized_firm_rank)
            rank_diffs.append(rank_diff)
    
    avg_rank_diff = np.mean(rank_diffs) if rank_diffs else 0.0
    
    return avg_rank_diff


def _run_single_simulation_experiment_1(
    args: Tuple[int, int, int, int, float, float, str, int]
) -> Tuple[int, Dict]:
    """Run one simulation for Experiment 1."""
    num_headhunters, num_workers, num_firms, base_seed, gamma, alpha, matching_algorithm, seed_offset = args
    seed = base_seed + seed_offset if base_seed is not None else None
    
    # Create market
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
    
    # Calculate rank differences (normalized) for overall, period 0, and period 1
    avg_rank_diff_overall = calculate_rank_differences(market, period_results, period=None)
    avg_rank_diff_period0 = calculate_rank_differences(market, period_results, period=0)
    avg_rank_diff_period1 = calculate_rank_differences(market, period_results, period=1)
    
    # Return results
    result = {
        "num_headhunters": num_headhunters,
        "total_matches": len(all_matches),
        "early_matches": len(early_matches),
        "regular_matches": len(regular_matches),
        "headhunter_welfare": welfare.headhunter_welfare,
        "firm_welfare": welfare.firm_welfare,
        "worker_welfare": welfare.worker_welfare,
        "match_welfare": welfare.match_welfare,
        "avg_worker_rank_diff": avg_rank_diff_overall,
        "avg_firm_rank_diff": avg_rank_diff_overall,
        "avg_rank_diff_period0": avg_rank_diff_period0,
        "avg_rank_diff_period1": avg_rank_diff_period1,
    }
    
    return num_headhunters, result


def run_experiment_1(
    num_workers: int = DEFAULT_NUM_WORKERS,
    num_firms: int = DEFAULT_NUM_FIRMS,
    gamma: float = 0.5,
    alpha: float = 0.5,
    matching_algorithm: str = "hungarian",
    seed: int = 42,
    n_jobs: int = -1,
) -> Dict[str, List]:
    """Sweep headhunter counts from 1 to num_firms with fixed gamma/alpha."""
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
        "avg_rank_diff_period0": [],
        "avg_rank_diff_period1": [],
    }
    
    # Prepare arguments for parallel execution
    num_headhunters_list = list(range(1, num_firms + 1))
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Create argument tuples
    args_list = [
        (num_headhunters, num_workers, num_firms, seed, gamma, alpha, matching_algorithm, idx)
        for idx, num_headhunters in enumerate(num_headhunters_list)
    ]
    
    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_hh = {
            executor.submit(_run_single_simulation_experiment_1, args): args[0]
            for args in args_list
        }
        
        # Collect results with progress bar
        completed_results = {}
        for future in tqdm(as_completed(future_to_hh), total=len(args_list), desc="Experiment 1: Varying headhunters"):
            num_headhunters, result = future.result()
            completed_results[num_headhunters] = result
    
    # Sort results by num_headhunters and store
    for num_headhunters in sorted(completed_results.keys()):
        result = completed_results[num_headhunters]
        results["num_headhunters"].append(result["num_headhunters"])
        results["total_matches"].append(result["total_matches"])
        results["early_matches"].append(result["early_matches"])
        results["regular_matches"].append(result["regular_matches"])
        results["headhunter_welfare"].append(result["headhunter_welfare"])
        results["firm_welfare"].append(result["firm_welfare"])
        results["worker_welfare"].append(result["worker_welfare"])
        results["match_welfare"].append(result["match_welfare"])
        results["avg_worker_rank_diff"].append(result["avg_worker_rank_diff"])
        results["avg_firm_rank_diff"].append(result["avg_firm_rank_diff"])
        results["avg_rank_diff_period0"].append(result["avg_rank_diff_period0"])
        results["avg_rank_diff_period1"].append(result["avg_rank_diff_period1"])
    
    return results


def _run_single_simulation_experiment_2(
    args: Tuple[float, int, int, int, float, str, int, int]
) -> Tuple[float, Dict]:
    """Run one simulation for Experiment 2."""
    alpha, num_workers, num_firms, num_headhunters, gamma, matching_algorithm, base_seed, seed_offset = args
    seed = base_seed + seed_offset if base_seed is not None else None
    
    # Create market
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
    
    # Calculate rank differences (normalized) for overall, period 0, and period 1
    avg_rank_diff_overall = calculate_rank_differences(market, period_results, period=None)
    avg_rank_diff_period0 = calculate_rank_differences(market, period_results, period=0)
    avg_rank_diff_period1 = calculate_rank_differences(market, period_results, period=1)
    
    # Return results
    result = {
        "alpha": alpha,
        "total_matches": len(all_matches),
        "early_matches": len(early_matches),
        "regular_matches": len(regular_matches),
        "headhunter_welfare": welfare.headhunter_welfare,
        "firm_welfare": welfare.firm_welfare,
        "worker_welfare": welfare.worker_welfare,
        "match_welfare": welfare.match_welfare,
        "avg_worker_rank_diff": avg_rank_diff_overall,
        "avg_firm_rank_diff": avg_rank_diff_overall,
        "avg_rank_diff_period0": avg_rank_diff_period0,
        "avg_rank_diff_period1": avg_rank_diff_period1,
    }
    
    return alpha, result


def run_experiment_2(
    num_workers: int = DEFAULT_NUM_WORKERS,
    num_firms: int = DEFAULT_NUM_FIRMS,
    num_headhunters: int = 20,
    gamma: float = 0.5,
    matching_algorithm: str = "hungarian",
    seed: int = 42,
    alpha_start: float = 0.0,
    alpha_end: float = 1.0,
    alpha_step: float = 0.01,
    n_jobs: int = -1,
) -> Dict[str, List]:
    """Sweep alpha from alpha_start to alpha_end with fixed gamma and HH count."""
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
        "avg_rank_diff_period0": [],
        "avg_rank_diff_period1": [],
    }
    
    # Generate alpha values
    alpha_values = np.arange(alpha_start, alpha_end + alpha_step, alpha_step)
    
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Create argument tuples
    args_list = [
        (alpha, num_workers, num_firms, num_headhunters, gamma, matching_algorithm, seed, idx)
        for idx, alpha in enumerate(alpha_values)
    ]
    
    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_alpha = {
            executor.submit(_run_single_simulation_experiment_2, args): args[0]
            for args in args_list
        }
        
        # Collect results with progress bar
        completed_results = {}
        for future in tqdm(as_completed(future_to_alpha), total=len(args_list), desc="Experiment 2: Varying alpha"):
            alpha, result = future.result()
            completed_results[alpha] = result
    
    # Sort results by alpha and store
    for alpha in sorted(completed_results.keys()):
        result = completed_results[alpha]
        results["alpha"].append(result["alpha"])
        results["total_matches"].append(result["total_matches"])
        results["early_matches"].append(result["early_matches"])
        results["regular_matches"].append(result["regular_matches"])
        results["headhunter_welfare"].append(result["headhunter_welfare"])
        results["firm_welfare"].append(result["firm_welfare"])
        results["worker_welfare"].append(result["worker_welfare"])
        results["match_welfare"].append(result["match_welfare"])
        results["avg_worker_rank_diff"].append(result["avg_worker_rank_diff"])
        results["avg_firm_rank_diff"].append(result["avg_firm_rank_diff"])
        results["avg_rank_diff_period0"].append(result["avg_rank_diff_period0"])
        results["avg_rank_diff_period1"].append(result["avg_rank_diff_period1"])
    
    return results


def _run_single_simulation_experiment_3(
    args: Tuple[float, int, int, int, float, str, int, int]
) -> Tuple[float, Dict]:
    """Run one simulation for Experiment 3."""
    gamma, num_workers, num_firms, num_headhunters, alpha, matching_algorithm, base_seed, seed_offset = args
    seed = base_seed + seed_offset if base_seed is not None else None
    
    # Create market
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
    
    # Calculate rank differences (normalized) for overall, period 0, and period 1
    avg_rank_diff_overall = calculate_rank_differences(market, period_results, period=None)
    avg_rank_diff_period0 = calculate_rank_differences(market, period_results, period=0)
    avg_rank_diff_period1 = calculate_rank_differences(market, period_results, period=1)
    
    # Return results
    result = {
        "gamma": gamma,
        "total_matches": len(all_matches),
        "early_matches": len(early_matches),
        "regular_matches": len(regular_matches),
        "headhunter_welfare": welfare.headhunter_welfare,
        "firm_welfare": welfare.firm_welfare,
        "worker_welfare": welfare.worker_welfare,
        "match_welfare": welfare.match_welfare,
        "avg_worker_rank_diff": avg_rank_diff_overall,
        "avg_firm_rank_diff": avg_rank_diff_overall,
        "avg_rank_diff_period0": avg_rank_diff_period0,
        "avg_rank_diff_period1": avg_rank_diff_period1,
    }
    
    return gamma, result


def run_experiment_3(
    num_workers: int = DEFAULT_NUM_WORKERS,
    num_firms: int = DEFAULT_NUM_FIRMS,
    num_headhunters: int = 20,
    alpha: float = 0.5,
    matching_algorithm: str = "hungarian",
    seed: int = 42,
    gamma_start: float = 0.0,
    gamma_end: float = 1.0,
    gamma_step: float = 0.01,
    n_jobs: int = -1,
) -> Dict[str, List]:
    """Sweep gamma from gamma_start to gamma_end with fixed alpha and HH count."""
    results = {
        "gamma": [],
        "total_matches": [],
        "early_matches": [],
        "regular_matches": [],
        "headhunter_welfare": [],
        "firm_welfare": [],
        "worker_welfare": [],
        "match_welfare": [],
        "avg_worker_rank_diff": [],
        "avg_firm_rank_diff": [],
        "avg_rank_diff_period0": [],
        "avg_rank_diff_period1": [],
    }
    
    # Generate gamma values
    gamma_values = np.arange(gamma_start, gamma_end + gamma_step, gamma_step)
    
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Create argument tuples
    args_list = [
        (gamma, num_workers, num_firms, num_headhunters, alpha, matching_algorithm, seed, idx)
        for idx, gamma in enumerate(gamma_values)
    ]
    
    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_gamma = {
            executor.submit(_run_single_simulation_experiment_3, args): args[0]
            for args in args_list
        }
        
        # Collect results with progress bar
        completed_results = {}
        for future in tqdm(as_completed(future_to_gamma), total=len(args_list), desc="Experiment 3: Varying gamma"):
            gamma, result = future.result()
            completed_results[gamma] = result
    
    # Sort results by gamma and store
    for gamma in sorted(completed_results.keys()):
        result = completed_results[gamma]
        results["gamma"].append(result["gamma"])
        results["total_matches"].append(result["total_matches"])
        results["early_matches"].append(result["early_matches"])
        results["regular_matches"].append(result["regular_matches"])
        results["headhunter_welfare"].append(result["headhunter_welfare"])
        results["firm_welfare"].append(result["firm_welfare"])
        results["worker_welfare"].append(result["worker_welfare"])
        results["match_welfare"].append(result["match_welfare"])
        results["avg_worker_rank_diff"].append(result["avg_worker_rank_diff"])
        results["avg_firm_rank_diff"].append(result["avg_firm_rank_diff"])
        results["avg_rank_diff_period0"].append(result["avg_rank_diff_period0"])
        results["avg_rank_diff_period1"].append(result["avg_rank_diff_period1"])
    
    return results


def plot_experiment_1(results: Dict[str, List], save_dir: str = "hh_simulation/graphs") -> None:
    """Plot Experiment 1 results."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experiment 1: Effect of Number of Headhunters\n(Fixed: γ=0.5, α=0.5)", 
                 fontsize=14, fontweight='bold')
    
    num_headhunters = results["num_headhunters"]
    
    # Plot 1: Total matches
    axes[0].plot(num_headhunters, results["total_matches"], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel("Number of Headhunters")
    axes[0].set_ylabel("Total Matches")
    axes[0].set_title("Total Matches vs Number of Headhunters")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Rank difference (match quality)
    axes[1].plot(num_headhunters, results["avg_worker_rank_diff"], 'r-', linewidth=2, marker='o', markersize=4, label='Worker Rank Diff')
    axes[1].plot(num_headhunters, results["avg_firm_rank_diff"], 'g--', linewidth=2, marker='s', markersize=4, label='Firm Rank Diff')
    axes[1].set_xlabel("Number of Headhunters")
    axes[1].set_ylabel("Average Rank Difference")
    axes[1].set_title("Match Quality (Rank Difference) vs Number of Headhunters")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "experiment_1_headhunters.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_experiment_2(results: Dict[str, List], save_dir: str = "hh_simulation/graphs") -> None:
    """Plot Experiment 2 results."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experiment 2: Effect of Alpha (α)\n(Fixed: HH=20, γ=0.5)", 
                 fontsize=14, fontweight='bold')
    
    alpha = results["alpha"]
    
    # Plot 1: Total matches
    axes[0].plot(alpha, results["total_matches"], 'b-', linewidth=2, marker='o', markersize=2)
    axes[0].set_xlabel("Alpha (α)")
    axes[0].set_ylabel("Total Matches")
    axes[0].set_title("Total Matches vs Alpha")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Rank difference (match quality)
    axes[1].plot(alpha, results["avg_worker_rank_diff"], 'r-', linewidth=2, marker='o', markersize=2, label='Worker Rank Diff')
    axes[1].plot(alpha, results["avg_firm_rank_diff"], 'g--', linewidth=2, marker='s', markersize=2, label='Firm Rank Diff')
    axes[1].set_xlabel("Alpha (α)")
    axes[1].set_ylabel("Average Rank Difference")
    axes[1].set_title("Match Quality (Rank Difference) vs Alpha")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "experiment_2_alpha.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_experiment_3(results: Dict[str, List], save_dir: str = "hh_simulation/graphs") -> None:
    """Plot Experiment 3 results."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experiment 3: Effect of Gamma (γ)\n(Fixed: HH=20, α=0.5)", 
                 fontsize=14, fontweight='bold')
    
    gamma = results["gamma"]
    
    # Plot 1: Total matches
    axes[0].plot(gamma, results["total_matches"], 'b-', linewidth=2, marker='o', markersize=2)
    axes[0].set_xlabel("Gamma (γ)")
    axes[0].set_ylabel("Total Matches")
    axes[0].set_title("Total Matches vs Gamma")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Rank difference (match quality)
    axes[1].plot(gamma, results["avg_worker_rank_diff"], 'r-', linewidth=2, marker='o', markersize=2, label='Worker Rank Diff')
    axes[1].plot(gamma, results["avg_firm_rank_diff"], 'g--', linewidth=2, marker='s', markersize=2, label='Firm Rank Diff')
    axes[1].set_xlabel("Gamma (γ)")
    axes[1].set_ylabel("Average Rank Difference")
    axes[1].set_title("Match Quality (Rank Difference) vs Gamma")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "experiment_3_gamma.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()




def save_experiment_1_csv(results: Dict[str, List], save_dir: str = "hh_simulation/graphs") -> None:
    """Write Experiment 1 results to CSV."""
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
            "avg_rank_diff_period0",
            "avg_rank_diff_period1",
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
                results["avg_rank_diff_period0"][i],
                results["avg_rank_diff_period1"][i],
            ])
    
    print(f"Saved CSV to {csv_path}")


def save_experiment_2_csv(results: Dict[str, List], save_dir: str = "hh_simulation/graphs") -> None:
    """Write Experiment 2 results to CSV."""
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
            "avg_rank_diff_period0",
            "avg_rank_diff_period1",
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
                results["avg_rank_diff_period0"][i],
                results["avg_rank_diff_period1"][i],
            ])
    
    print(f"Saved CSV to {csv_path}")


def save_experiment_3_csv(results: Dict[str, List], save_dir: str = "hh_simulation/graphs") -> None:
    """Write Experiment 3 results to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    
    csv_path = os.path.join(save_dir, "experiment_3_gamma.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([
            "gamma",
            "total_matches",
            "early_matches",
            "regular_matches",
            "headhunter_welfare",
            "firm_welfare",
            "worker_welfare",
            "match_welfare",
            "avg_worker_rank_diff",
            "avg_firm_rank_diff",
            "avg_rank_diff_period0",
            "avg_rank_diff_period1",
        ])
        
        # Write data rows
        for i in range(len(results["gamma"])):
            writer.writerow([
                results["gamma"][i],
                results["total_matches"][i],
                results["early_matches"][i],
                results["regular_matches"][i],
                results["headhunter_welfare"][i],
                results["firm_welfare"][i],
                results["worker_welfare"][i],
                results["match_welfare"][i],
                results["avg_worker_rank_diff"][i],
                results["avg_firm_rank_diff"][i],
                results["avg_rank_diff_period0"][i],
                results["avg_rank_diff_period1"][i],
            ])
    
    print(f"Saved CSV to {csv_path}")


def run_all_experiments(
    num_workers: int = DEFAULT_NUM_WORKERS,
    num_firms: int = DEFAULT_NUM_FIRMS,
    matching_algorithm: str = "hungarian",
    seed: int = 42,
    save_dir: str = "hh_simulation/graphs",
    n_jobs: int = -1,
) -> None:
    """Run all experiments and produce plots plus CSVs."""
    print("=" * 60)
    print("Running Experiment 1: Varying Number of Headhunters")
    print("Fixed parameters: γ=0.5, α=0.5")
    print("=" * 60)
    results_1 = run_experiment_1(
        num_workers=num_workers,
        num_firms=num_firms,
        gamma=0.5,
        alpha=0.5,
        matching_algorithm=matching_algorithm,
        seed=seed,
        n_jobs=n_jobs,
    )
    plot_experiment_1(results_1, save_dir=save_dir)
    save_experiment_1_csv(results_1, save_dir=save_dir)
    
    print("\n" + "=" * 60)
    print("Running Experiment 2: Varying Alpha")
    print("Fixed parameters: HH=20, γ=0.5")
    print("=" * 60)
    results_2 = run_experiment_2(
        num_workers=num_workers,
        num_firms=num_firms,
        num_headhunters=20,
        gamma=0.5,
        matching_algorithm=matching_algorithm,
        seed=seed,
        n_jobs=n_jobs,
    )
    plot_experiment_2(results_2, save_dir=save_dir)
    save_experiment_2_csv(results_2, save_dir=save_dir)
    
    print("\n" + "=" * 60)
    print("Running Experiment 3: Varying Gamma")
    print("Fixed parameters: HH=20, α=0.5")
    print("=" * 60)
    results_3 = run_experiment_3(
        num_workers=num_workers,
        num_firms=num_firms,
        num_headhunters=20,
        alpha=0.5,
        matching_algorithm=matching_algorithm,
        seed=seed,
        n_jobs=n_jobs,
    )
    plot_experiment_3(results_3, save_dir=save_dir)
    save_experiment_3_csv(results_3, save_dir=save_dir)
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run experiments with baseline settings
    run_all_experiments(
        num_workers=DEFAULT_NUM_WORKERS,
        num_firms=DEFAULT_NUM_FIRMS,
        matching_algorithm="hungarian",
        seed=42,
        save_dir="hh_simulation/graphs",
    )

