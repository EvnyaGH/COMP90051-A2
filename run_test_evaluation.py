#!/usr/bin/env python
"""
Run test set evaluation to detect overfitting and get realistic performance estimates.
"""
import sys
from pathlib import Path

sys.path.append("src")

from src.experiments.test_evaluation import TestEvaluator
import pandas as pd
import json


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run test set evaluation to detect overfitting"
    )
    parser.add_argument(
        "--data-path",
        default="data/imdb_clean.csv",
        help="Path to clean dataset file (default: data/imdb_clean.csv)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing experiment results (default: results)",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="Specific results file to use (default: most recent experiment_results_*.json)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TEST SET EVALUATION - DETECTING OVERFITTING")
    print("=" * 80)
    print("This will:")
    print("1. Split data into train/test sets (80/20)")
    print("2. Train models on train set with best CV parameters")
    print("3. Evaluate on test set")
    print("4. Compare train vs test performance to detect overfitting")
    print("=" * 80)

    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        print("Please run data preparation first.")
        return

    data = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(data)} samples")

    # Load CV results
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Please run hyperparameter tuning first.")
        return

    if args.results_file:
        # Use specific results file
        results_path = results_dir / args.results_file
        if not results_path.exists():
            print(f"Specified results file not found: {results_path}")
            return
    else:
        # Find the most recent experiment results file
        experiment_files = list(results_dir.glob("experiment_results_*.json"))
        if not experiment_files:
            print("No experiment results found.")
            print("Please run hyperparameter tuning first.")
            return

        # Sort by modification time and get the most recent
        results_path = max(experiment_files, key=lambda p: p.stat().st_mtime)

    print(f"Loading CV results from: {results_path}")

    with open(results_path, "r") as f:
        cv_results = json.load(f)

    print(f"Loaded CV results")

    # Create evaluator and split data
    print(f"\nSplitting data into train/test sets (test_size={args.test_size})...")
    evaluator = TestEvaluator(test_size=args.test_size, random_state=args.random_state)
    train_data, test_data = evaluator.split_data(data)

    # Evaluate all models
    print("\nEvaluating models on test set...")
    test_results = evaluator.evaluate_all_models(cv_results["hyperparameter_tuning"])

    # Create comparison table
    print("\nCreating comparison table...")
    comparison_df = evaluator.create_comparison_table(test_results)

    print("\n" + "=" * 80)
    print("FINAL TEST SET COMPARISON")
    print("=" * 80)
    print(comparison_df.round(4))

    # Analysis
    print("\n" + "=" * 80)
    print("OVERFITTING ANALYSIS")
    print("=" * 80)

    for _, row in comparison_df.iterrows():
        model_name = row["Model"]
        f1_gap = row["Overfitting_Gap_F1"]
        test_f1 = row["Test_F1"]

        if f1_gap > 0.05:
            status = "SEVERE OVERFITTING"
        elif f1_gap > 0.02:
            status = "MODERATE OVERFITTING"
        else:
            status = "NO OVERFITTING"

        print(
            f"{model_name:30} | Test F1: {test_f1:.4f} | Gap: {f1_gap:.4f} | {status}"
        )

    # Save results
    output_path = Path("results/test_evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"\n💾 Test results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print("Key insights:")
    print("- Test F1 scores show REALISTIC performance")
    print("- Overfitting gaps show if models generalize well")
    print("- Large gaps indicate overfitting to training data")
    print("=" * 80)


if __name__ == "__main__":
    main()
