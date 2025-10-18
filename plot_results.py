#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script to plot results from experiment results JSON file
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


def main():
    """Plot results from experiment results JSON file"""

    # Path to the results file
    results_file = Path("results/experiment_results_20251018_033310.json")

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Available result files:")
        results_dir = Path("results")
        for file in results_dir.glob("experiment_results_*.json"):
            print(f"  - {file}")
        return

    print(f"Loading results from: {results_file}")

    # Load the results
    with open(results_file, "r") as f:
        data = json.load(f)

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

    # Generate summary table
    try:
        export_summary_table(ht_results, results_dir)
        print("✓ Summary table saved to results/model_summary.csv and model_summary.md")
    except Exception as e:
        print(f"✗ Error generating summary table: {e}")

    # Generate confusion matrices
    try:
        plot_confusion_matrices(ht_results, results_dir)
    except Exception as e:
        print(f"✗ Error generating confusion matrices: {e}")

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
