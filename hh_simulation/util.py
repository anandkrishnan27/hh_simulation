from __future__ import annotations

import os
import csv
from typing import Dict, List
import matplotlib.pyplot as plt


def read_csv_file(csv_path: str) -> Dict[str, List]:
    """
    Read a CSV file and return a dictionary with column names as keys.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Dictionary with column names as keys and lists of values as values
    """
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
    """
    Plot Experiment 1 results from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        save_path: Optional path to save the plot. If None, saves next to CSV with .png extension
    """
    results = read_csv_file(csv_path)
    
    if save_path is None:
        save_path = csv_path.replace('.csv', '.png')
    
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_experiment_2_from_csv(csv_path: str, save_path: str = None) -> None:
    """
    Plot Experiment 2 results from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        save_path: Optional path to save the plot. If None, saves next to CSV with .png extension
    """
    results = read_csv_file(csv_path)
    
    if save_path is None:
        save_path = csv_path.replace('.csv', '.png')
    
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_rank_differences_from_csv(csv_path: str, save_path: str = None, experiment_name: str = None) -> None:
    """
    Plot rank difference metrics from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        save_path: Optional path to save the plot. If None, saves next to CSV with _rank_diff.png extension
        experiment_name: Optional name for the experiment (for title)
    """
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
    else:
        raise ValueError("CSV file must contain either 'num_headhunters' or 'alpha' column")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{experiment_name}: Rank Difference Metrics", 
                 fontsize=14, fontweight='bold')
    
    x_values = results[x_var]
    
    # Plot 1: Average worker rank difference
    axes[0].plot(x_values, results["avg_worker_rank_diff"], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Average Worker Rank Difference")
    axes[0].set_title("Average |Worker Rank - Firm Rank|")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Average firm rank difference
    axes[1].plot(x_values, results["avg_firm_rank_diff"], 'r-', linewidth=2, marker='o', markersize=4)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Average Firm Rank Difference")
    axes[1].set_title("Average |Firm Rank - Worker Rank|")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved rank difference plot to {save_path}")
    plt.close()


def plot_all_from_csvs(graphs_dir: str = "hh_simulation/graphs") -> None:
    """
    Plot all graphs from CSV files in the graphs directory.
    
    Args:
        graphs_dir: Directory containing the CSV files
    """
    exp1_csv = os.path.join(graphs_dir, "experiment_1_headhunters.csv")
    exp2_csv = os.path.join(graphs_dir, "experiment_2_alpha.csv")
    
    if os.path.exists(exp1_csv):
        print("Plotting Experiment 1...")
        plot_experiment_1_from_csv(exp1_csv)
        plot_rank_differences_from_csv(exp1_csv, experiment_name="Experiment 1")
    else:
        print(f"Warning: {exp1_csv} not found")
    
    if os.path.exists(exp2_csv):
        print("Plotting Experiment 2...")
        plot_experiment_2_from_csv(exp2_csv)
        plot_rank_differences_from_csv(exp2_csv, experiment_name="Experiment 2")
    else:
        print(f"Warning: {exp2_csv} not found")
    
    print("All plots generated!")


if __name__ == "__main__":
    # Plot all graphs from CSV files
    plot_all_from_csvs()

