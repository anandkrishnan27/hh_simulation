from __future__ import annotations

import os
import csv
from typing import Dict, List
import matplotlib.pyplot as plt


def read_csv_file(csv_path: str) -> Dict[str, List]:
    """Read a CSV and return a column-wise dictionary."""
    results = {}
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        # Initialize dictionary with empty lists for each column
        fieldnames = reader.fieldnames
        if fieldnames:
            for field in fieldnames:
                results[field] = []
            
            # Read rows and append values
            for row in reader:
                for field in fieldnames:
                    # Convert numeric fields to float
                    try:
                        value = float(row[field])
                        results[field].append(value)
                    except ValueError:
                        results[field].append(row[field])
    
    return results


def plot_experiment_1_from_csv(csv_path: str, save_path: str = None) -> None:
    """Plot Experiment 1 results stored in a CSV."""
    results = read_csv_file(csv_path)
    
    if save_path is None:
        save_path = csv_path.replace('.csv', '.png')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experiment 1: Effect of Number of Headhunters\n(Baseline: 500 workers, 125 firms, γ=0.5, α=0.5)", 
                 fontsize=14, fontweight='bold')
    
    num_headhunters = results["num_headhunters"]
    
    # Plot 1: Total matches
    axes[0].plot(num_headhunters, results["total_matches"], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel("Number of Headhunters")
    axes[0].set_ylabel("Total Matches")
    axes[0].set_title("Total Matches vs Number of Headhunters")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Match quality (rank difference)
    axes[1].plot(num_headhunters, results["avg_worker_rank_diff"], 'r-', linewidth=2, marker='o', markersize=4)
    axes[1].set_xlabel("Number of Headhunters")
    axes[1].set_ylabel("Normalized Rank Difference")
    axes[1].set_title("Match Quality (Rank Difference) vs Number of Headhunters")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_experiment_2_from_csv(csv_path: str, save_path: str = None) -> None:
    """Plot Experiment 2 results stored in a CSV."""
    results = read_csv_file(csv_path)
    
    if save_path is None:
        save_path = csv_path.replace('.csv', '.png')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experiment 2: Effect of Alpha (α)\n(Baseline: 500 workers, 125 firms, 20 headhunters, γ=0.5)", 
                 fontsize=14, fontweight='bold')
    
    alpha = results["alpha"]
    
    # Plot 1: Total matches
    axes[0].plot(alpha, results["total_matches"], 'b-', linewidth=2, marker='o', markersize=2)
    axes[0].set_xlabel("Alpha (α)")
    axes[0].set_ylabel("Total Matches")
    axes[0].set_title("Total Matches vs Alpha")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Match quality (rank difference)
    axes[1].plot(alpha, results["avg_worker_rank_diff"], 'r-', linewidth=2, marker='o', markersize=2)
    axes[1].set_xlabel("Alpha (α)")
    axes[1].set_ylabel("Normalized Rank Difference")
    axes[1].set_title("Match Quality (Rank Difference) vs Alpha")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_experiment_3_from_csv(csv_path: str, save_path: str = None) -> None:
    """Plot Experiment 3 results stored in a CSV."""
    results = read_csv_file(csv_path)
    
    if save_path is None:
        save_path = csv_path.replace('.csv', '.png')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experiment 3: Effect of Gamma (γ)\n(Baseline: 500 workers, 125 firms, 20 headhunters, α=0.5)", 
                 fontsize=14, fontweight='bold')
    
    gamma = results["gamma"]
    
    # Plot 1: Total matches
    axes[0].plot(gamma, results["total_matches"], 'b-', linewidth=2, marker='o', markersize=2)
    axes[0].set_xlabel("Gamma (γ)")
    axes[0].set_ylabel("Total Matches")
    axes[0].set_title("Total Matches vs Gamma")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Match quality (rank difference)
    axes[1].plot(gamma, results["avg_worker_rank_diff"], 'r-', linewidth=2, marker='o', markersize=2)
    axes[1].set_xlabel("Gamma (γ)")
    axes[1].set_ylabel("Normalized Rank Difference")
    axes[1].set_title("Match Quality (Rank Difference) vs Gamma")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_rank_differences_from_csv(csv_path: str, save_path: str = None, experiment_name: str = None) -> None:
    """Plot rank-difference metrics from a CSV with period 0 and period 1 lines."""
    results = read_csv_file(csv_path)
    
    if save_path is None:
        save_path = csv_path.replace('.csv', '_rank_diff.png')
    
    # Determine x-axis variable
    if "num_headhunters" in results:
        x_var = "num_headhunters"
        x_label = "Number of Headhunters"
        if experiment_name is None:
            experiment_name = "Experiment 1"
    elif "alpha" in results:
        x_var = "alpha"
        x_label = "Alpha (α)"
        if experiment_name is None:
            experiment_name = "Experiment 2"
    elif "gamma" in results:
        x_var = "gamma"
        x_label = "Gamma (γ)"
        if experiment_name is None:
            experiment_name = "Experiment 3"
    else:
        raise ValueError("CSV file must contain 'num_headhunters', 'alpha', or 'gamma' column")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f"{experiment_name}: Normalized Rank Difference Metrics", 
                 fontsize=14, fontweight='bold')
    
    x_values = results[x_var]
    
    # Plot period 0 rank difference if available
    if "avg_rank_diff_period0" in results:
        ax.plot(x_values, results["avg_rank_diff_period0"], 'g-', linewidth=2, marker='s', markersize=3, label='Period 0 (Early)')
    
    # Plot period 1 rank difference if available
    if "avg_rank_diff_period1" in results:
        ax.plot(x_values, results["avg_rank_diff_period1"], 'r-', linewidth=2, marker='^', markersize=3, label='Period 1 (Regular)')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalized Rank Difference")
    ax.set_title("Average Normalized Rank Difference by Period")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved rank difference plot to {save_path}")
    plt.close()


def plot_early_vs_regular_matches_from_csv(csv_path: str, save_path: str = None, experiment_name: str = None) -> None:
    """Plot early vs regular matches as two lines on a standard line plot."""
    results = read_csv_file(csv_path)
    
    if save_path is None:
        save_path = csv_path.replace('.csv', '_early_vs_regular_matches.png')
    
    # Determine x-axis variable
    if "num_headhunters" in results:
        x_var = "num_headhunters"
        x_label = "Number of Headhunters"
        if experiment_name is None:
            experiment_name = "Experiment 1"
    elif "alpha" in results:
        x_var = "alpha"
        x_label = "Alpha (α)"
        if experiment_name is None:
            experiment_name = "Experiment 2"
    elif "gamma" in results:
        x_var = "gamma"
        x_label = "Gamma (γ)"
        if experiment_name is None:
            experiment_name = "Experiment 3"
    else:
        raise ValueError("CSV file must contain 'num_headhunters', 'alpha', or 'gamma' column")
    
    x_values = results[x_var]
    early_matches = results["early_matches"]
    total_matches = results["total_matches"]
    
    # Calculate percentage of early matches (t=0)
    percent_early = [(early / total * 100) if total > 0 else 0 for early, total in zip(early_matches, total_matches)]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f"{experiment_name}: Percent of Matches in Early Period (t=0)", 
                 fontsize=14, fontweight='bold')
    
    # Plot percentage of early matches
    ax.plot(x_values, percent_early, 'b-', linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel("Percent of Total Matches (%)")
    ax.set_title("Percent of Matches in Early Period (t=0)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved early vs regular matches plot to {save_path}")
    plt.close()


def plot_all_from_csvs(graphs_dir: str = "hh_simulation/graphs") -> None:
    """Generate every graph found in the graphs directory."""
    exp1_csv = os.path.join(graphs_dir, "experiment_1_headhunters.csv")
    exp2_csv = os.path.join(graphs_dir, "experiment_2_alpha.csv")
    exp3_csv = os.path.join(graphs_dir, "experiment_3_gamma.csv")
    
    if os.path.exists(exp1_csv):
        print("Plotting Experiment 1...")
        plot_experiment_1_from_csv(exp1_csv)
        plot_rank_differences_from_csv(exp1_csv, experiment_name="Experiment 1")
        plot_early_vs_regular_matches_from_csv(exp1_csv, experiment_name="Experiment 1")
    else:
        print(f"Warning: {exp1_csv} not found")
    
    if os.path.exists(exp2_csv):
        print("Plotting Experiment 2...")
        plot_experiment_2_from_csv(exp2_csv)
        plot_rank_differences_from_csv(exp2_csv, experiment_name="Experiment 2")
        plot_early_vs_regular_matches_from_csv(exp2_csv, experiment_name="Experiment 2")
    else:
        print(f"Warning: {exp2_csv} not found")
    
    if os.path.exists(exp3_csv):
        print("Plotting Experiment 3...")
        plot_experiment_3_from_csv(exp3_csv)
        plot_rank_differences_from_csv(exp3_csv, experiment_name="Experiment 3")
        plot_early_vs_regular_matches_from_csv(exp3_csv, experiment_name="Experiment 3")
    else:
        print(f"Warning: {exp3_csv} not found")
    
    print("All plots generated!")


if __name__ == "__main__":
    # Plot all graphs from CSV files
    plot_all_from_csvs()

