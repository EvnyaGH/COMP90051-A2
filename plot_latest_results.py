#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script to plot results from the latest experiment results JSON file
"""

import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.experiments.visualization import (
    plot_model_performance,
    export_summary_table,
    plot_confusion_matrices,
)


def find_latest_results_file():
    """Find the most recent experiment results file"""
    results_dir = Path("results")
    if not results_dir.exists():
        return None

    # Find all experiment results files
    experiment_files = list(results_dir.glob("experiment_results_*.json"))
    if not experiment_files:
        return None

    # Sort by modification time and get the most recent
    latest_file = max(experiment_files, key=lambda p: p.stat().st_mtime)
    return latest_file


def main():
    """Plot results from the latest experiment results JSON file"""

    # Find the latest results file
    results_file = find_latest_results_file()

    if not results_file:
        print("Error: No experiment results files found in results/ directory")
        return

    print(f"Loading results from: {results_file}")

    # Load the results
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}")
        return

    # Extract hyperparameter tuning results
    if "hyperparameter_tuning" not in data:
        print("Error: No hyperparameter_tuning results found in the file")
        return

    ht_results = data["hyperparameter_tuning"]
    print(f"Found {len(ht_results)} models:")
    for model_name in ht_results.keys():
        print(f"  - {model_name}")

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("\nGenerating visualizations...")

    # Generate model performance plot
    try:
        plot_model_performance(ht_results, results_dir)
        print(
            "✓ Model performance bar chart saved to results/model_performance_bar.png"
        )
    except Exception as e:
        print(f"✗ Error generating model performance plot: {e}")
        import traceback

        traceback.print_exc()

    # Generate summary table
    try:
        export_summary_table(ht_results, results_dir)
        print("✓ Summary table saved to results/model_summary.csv and model_summary.md")
    except Exception as e:
        print(f"✗ Error generating summary table: {e}")
        import traceback

        traceback.print_exc()

    # Generate confusion matrices
    try:
        plot_confusion_matrices(ht_results, results_dir)
    except Exception as e:
        print(f"✗ Error generating confusion matrices: {e}")
        import traceback

        traceback.print_exc()

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
