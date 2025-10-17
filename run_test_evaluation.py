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
    data_path = Path("data/imdb_clean.csv")
    if not data_path.exists():
        print("Dataset not found at data/imdb_clean.csv")
        print("Please run data preparation first.")
        return

    data = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(data)} samples")

    # Load CV results
    results_path = Path("results/experiment_results_20251015_225601.json")
    if not results_path.exists():
        print("CV results not found.")
        print("Please run hyperparameter tuning first.")
        return

    with open(results_path, "r") as f:
        cv_results = json.load(f)

    print(f"Loaded CV results")

    # Create evaluator and split data
    print("\nSplitting data into train/test sets...")
    evaluator = TestEvaluator(test_size=0.2, random_state=42)
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
