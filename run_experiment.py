#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script to run the complete sentiment classification experiment.

This script runs the entire experimental pipeline for the research question:
"How do different text representation methods (TF-IDF, Word2Vec, ELECTRA) perform
across increasing model complexity levels (LogReg, BiLSTM, ELECTRA) for sentiment classification?"

Usage:
    python run_experiment.py

The script will:
1. Load and prepare the IMDB dataset
2. Extract TF-IDF, Word2Vec, and ELECTRA features
3. Run 10-fold cross-validation with hyperparameter tuning for each model
4. Generate learning curves for each model
5. Create visualizations and save results
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from experiments.experimental_pipeline import ExperimentalPipeline


def main():
    """Run the complete experimental pipeline."""
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

    # Create and run experimental pipeline
    pipeline = ExperimentalPipeline()

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
