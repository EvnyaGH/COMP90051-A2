#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script to run the complete sentiment classification experiment.

This script runs the entire experimental pipeline for the research question:
"How do different text representation methods (TF-IDF, Word2Vec, ELECTRA) perform
across increasing model complexity levels (LogReg, BiLSTM, ELECTRA) for sentiment classification?"

Usage:
    python run_experiment.py

    # Fast mode
    python run_experiment.py --fast

    # Custom settings
    python run_experiment.py --fast --data-dir data --results-dir results_fast

The script will:
1. Load and prepare the IMDB dataset
2. Run cross-validation with hyperparameter tuning for each model
3. Generate learning curves for each model
4. Create visualizations and save results

Fast mode uses:
- Reduced dataset (10k samples)
- Fewer CV folds (3 outer, 2 inner)
- Minimal hyperparameters
- Shorter training epochs
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.experiments.experimental_pipeline import ExperimentalPipeline


def main():
    """Run the complete experimental pipeline."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast mode (reduced dataset, fewer CV folds, minimal hyperparameters)",
    )
    parser.add_argument(
        "--include-learning-curves",
        action="store_true",
        help="Include learning curves in the experiment",
    )
    parser.add_argument(
        "--data-dir", default="data", help="Directory containing dataset files"
    )
    parser.add_argument(
        "--results-dir", default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--raw-data-path",
        default="data/IMDB Dataset.csv",
        help="Path to raw IMDB Dataset.csv file",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SENTIMENT CLASSIFICATION RESEARCH EXPERIMENT")
    print("=" * 80)
    print()
    print("Research Question:")
    print("How do different text representation methods (TF-IDF, Word2Vec, ELECTRA)")
    print("perform across increasing model complexity levels (LogReg, BiLSTM, ELECTRA)")
    print("for sentiment classification?")
    print()
    print("=" * 80)

    if args.fast:
        print("FAST MODE ENABLED")
    else:
        print("FULL MODE")

    print("=" * 80)

    # Create and run experimental pipeline
    pipeline = ExperimentalPipeline(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        random_state=args.random_state,
        fast=args.fast,
        include_learning_curves=args.include_learning_curves,
        raw_data_path=args.raw_data_path,
    )

    try:
        pipeline.run_complete_pipeline()
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Results saved to:")
        print(f"  - {pipeline.results_dir}/")
        print()
        print("Generated files:")
        print("  - experiment_results_*.json (raw results)")
        print("  - experiment_summary.json (summary)")
        print("  - model_comparison.png (bar chart)")
        print("  - learning_curves.png (learning curves)")
        print()

    except Exception as e:
        print(f"\nERROR: Experiment failed with error: {e}")
        print("Please check the error message and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
