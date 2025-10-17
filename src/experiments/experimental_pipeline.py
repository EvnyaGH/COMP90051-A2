#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Experimental Pipeline for Sentiment Classification Research

This module orchestrates the complete experimental pipeline for the research question:
"How do different text representation methods (TF-IDF, Word2Vec, ELECTRA) perform
across increasing model complexity levels (LogReg, BiLSTM, ELECTRA) for sentiment classification?"

The pipeline coordinates:
1. Data loading and preprocessing
2. Feature extraction
3. Hyperparameter tuning
4. Learning curve experiments
5. Results collection and visualization
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt

# Import our modules
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.metrics import f1_score
from prepare_dataset import main as prepare_data
from models.logistic_regression import create_logistic_regression_factory
from models.electra_sentiment import create_electra_factory
from models.bilstm_sentiment import create_bilstm_factory

try:
    from .hyperparameter_tuning import (
        HyperparameterTuner,
        HYPERPARAMETER_GRIDS,
        HYPERPARAMETER_GRIDS_FAST,
    )
    from .learning_curves import LearningCurveExperiment
    from .visualization import plot_model_performance, export_summary_table
except ImportError:
    # Fallback for direct script execution
    from hyperparameter_tuning import (
        HyperparameterTuner,
        HYPERPARAMETER_GRIDS,
        HYPERPARAMETER_GRIDS_FAST,
    )
    from learning_curves import LearningCurveExperiment
    from visualization import plot_model_performance, export_summary_table


class ExperimentalPipeline:
    """
    Main experimental pipeline for sentiment classification research.

    This class orchestrates the complete experimental workflow from data loading
    to results visualization.
    """

    def __init__(
        self,
        data_dir: str = "data",
        results_dir: str = "results",
        random_state: int = 42,
        fast: bool = False,
        include_learning_curves: bool = False,
        raw_data_path: str = None,
    ):
        self.fast = fast
        self.include_learning_curves = include_learning_curves
        """
        Initialize experimental pipeline.

        Args:
            data_dir: Directory containing dataset files
            results_dir: Directory to save results
            random_state: Random seed for reproducibility
            fast: Whether to run in fast mode
            include_learning_curves: Whether to include learning curves
            raw_data_path: Path to raw IMDB Dataset.csv file
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.random_state = random_state
        self.raw_data_path = raw_data_path or str(self.data_dir / "IMDB Dataset.csv")
        self.results_dir.mkdir(exist_ok=True)
        self.data: pd.DataFrame | None = None
        self.results: Dict[str, Any] = {}

        # Experiment configuration
        self.experiment_defs = [
            ("TF-IDF + Logistic Regression", "logreg"),
            ("Word2Vec + BiLSTM", "bilstm"),
            ("ELECTRA fine-tune", "electra"),
        ]

        # Hyperparameter grids
        grids = HYPERPARAMETER_GRIDS_FAST if self.fast else HYPERPARAMETER_GRIDS
        self.hyperparams = {
            "logreg": grids["logistic_regression"],
            "bilstm": grids["bilstm"],
            "electra": grids["electra"],
        }

    def load_and_prepare_data(self):
        """Load and prepare the IMDB dataset."""
        print("=" * 60)
        print("STEP 1: Loading and Preparing Data")
        print("=" * 60)

        # Check if raw data exists
        raw_path = Path(self.raw_data_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_data_path}")

        # Always prepare dataset from raw data
        print(f"[info] Preparing dataset from raw data: {self.raw_data_path}")

        # Import and call prepare_dataset function directly
        from prepare_dataset import prepare_dataset

        meta = prepare_dataset(
            src_path=self.raw_data_path,
            outdir=self.data_dir,
            create_10k_subset=True,
            random_state=self.random_state,
        )

        print("[info] Dataset preparation completed successfully")
        print(
            f"[info] Processed {meta['final_rows']} rows, dropped {meta['dropped_rows']} rows"
        )
        print(f"[info] Class balance: {meta['class_balance_full']}")

        # Load clean data
        clean_path = self.data_dir / "imdb_clean.csv"
        self.data = pd.read_csv(clean_path)
        print(
            f"[data] rows={len(self.data)}, balance={self.data['label'].value_counts().to_dict()}"
        )

        # Show sample data
        print("\nSample data:")
        print(self.data.head())

    def run_hyperparameter_tuning(self):
        """Run hyperparameter tuning for all models."""
        print("\n" + "=" * 60)
        print("STEP 2: Hyperparameter Tuning")
        print("=" * 60)

        assert self.data is not None
        X_text = self.data["text"].tolist()
        y = self.data["label"].to_numpy()

        outer = 3   if self.fast else 5
        inner = 2   if self.fast else 3

        # Create hyperparameter tuner
        tuner = HyperparameterTuner(
            outer_folds=outer,
            inner_folds=inner,
            random_state=self.random_state,
            scoring_func=f1_score,
            fast=self.fast,
        )

        # Prepare experiments
        experiments = []
        for exp_name, model_key in self.experiment_defs:
            if model_key == "logreg":
                factory = create_logistic_regression_factory(
                    self.hyperparams[model_key]
                )
            elif model_key == "bilstm":
                factory = create_bilstm_factory(self.hyperparams[model_key])
            elif model_key == "electra":
                factory = create_electra_factory(self.hyperparams[model_key])
            else:
                print(f"[warn] model {model_key} not wired yet; skipping.")
                continue
            experiments.append(
                (exp_name, X_text, y, factory, self.hyperparams[model_key])
            )

        # Run hyperparameter tuning
        tuning_results = tuner.tune_multiple_models(experiments)

        # Store results
        self.results["hyperparameter_tuning"] = tuning_results

        # Show best models
        best_models = tuner.get_best_models()
        print("\nBest hyperparameters found:")
        for model_name, config in best_models.items():
            print(
                f"{model_name}: {config['best_params']} (F1: {config['cv_score']:.4f})"
            )

        return tuning_results

    def run_learning_curves(self):
        """Run learning curve experiments."""
        print("\n" + "=" * 60)
        print("STEP 3: Learning Curve Experiments")
        print("=" * 60)

        assert self.data is not None
        X_text = self.data["text"].tolist()
        y = self.data["label"].to_numpy()

        # Create learning curve experiment
        lc_experiment = LearningCurveExperiment(
            training_sizes=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            outer_folds=3,  # Fewer folds for learning curves
            inner_folds=2,
            random_state=self.random_state,
            scoring_func=f1_score,
        )

        # Prepare experiments
        experiments = []
        for exp_name, model_key in self.experiment_defs:
            if model_key == "logreg":
                factory = create_logistic_regression_factory(
                    self.hyperparams[model_key]
                )
            elif model_key == "bilstm":
                factory = create_bilstm_factory(self.hyperparams[model_key])
            elif model_key == "electra":
                factory = create_electra_factory(self.hyperparams[model_key])
            else:
                print(f"[warn] model {model_key} not wired yet; skipping.")
                continue
            experiments.append(
                (exp_name, X_text, y, factory, self.hyperparams[model_key])
            )

        # Run learning curve experiments
        lc_results = lc_experiment.generate_multiple_learning_curves(experiments)

        # Store results
        self.results["learning_curves"] = lc_results

        # Show summary
        summary_df = lc_experiment.get_summary()
        print("\nLearning curve summary:")
        print(summary_df)

        return lc_results

    def create_visualizations(self):
        """Create visualizations for the results."""
        print("\n" + "=" * 60)
        print("STEP 4: Creating Visualizations")
        print("=" * 60)

        # Set style
        plt.style.use("seaborn-v0_8")

        # 1. Model comparison bar chart
        self._create_model_comparison_chart()

        # 2. Learning curves
        self._create_learning_curves_plot()

        print("Visualizations saved to results/ directory")

    def _create_model_comparison_chart(self):
        """Create bar chart comparing model performance."""
        if "hyperparameter_tuning" not in self.results:
            print("No hyperparameter tuning results to visualize")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        model_names = []
        f1_scores = []
        f1_stds = []

        for model_name, results in self.results["hyperparameter_tuning"].items():
            if "cv_results" in results and results["cv_results"] is not None:
                model_names.append(model_name)
                f1_scores.append(results["cv_results"]["mean_cv_score"])
                f1_stds.append(results["cv_results"]["std_cv_score"])

        if not model_names:
            print("No valid results for model comparison")
            return

        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, f1_scores, yerr=f1_stds, capsize=5, alpha=0.7)

        ax.set_xlabel("Model")
        ax.set_ylabel("F1 Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha="right")

        # Add value labels on bars
        for i, (score, std) in enumerate(zip(f1_scores, f1_stds)):
            ax.text(i, score + std + 0.01, f"{score:.3f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "model_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("Model comparison chart saved")

    def _create_learning_curves_plot(self):
        """Create learning curves plot."""
        if "learning_curves" not in self.results:
            print("No learning curve results to visualize")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        for model_name, results in self.results["learning_curves"].items():
            if "error" in results:
                continue

            training_sizes = results["training_sizes"]
            scores = results["scores"]
            score_stds = results["score_stds"]

            if not training_sizes:
                continue

            # Convert to percentages
            sizes_pct = [s * 100 for s in training_sizes]

            # Plot learning curve
            ax.plot(sizes_pct, scores, marker="o", label=model_name, linewidth=2)
            ax.fill_between(
                sizes_pct,
                np.array(scores) - np.array(score_stds),
                np.array(scores) + np.array(score_stds),
                alpha=0.3,
            )

        ax.set_xlabel("Training Set Size (%)")
        ax.set_ylabel("F1 Score")
        ax.set_title("Learning Curves: F1 Score vs Training Set Size")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "learning_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("Learning curves plot saved")

    def save_results(self):
        """Save all results to files."""
        print("\n" + "=" * 60)
        print("STEP 5: Saving Results")
        print("=" * 60)

        # Save raw results
        results_path = (
            self.results_dir
            / f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        json_results[key][sub_key] = {
                            k: (v.tolist() if hasattr(v, "tolist") else v)
                            for k, v in sub_value.items()
                        }
                    elif isinstance(sub_value, (list, np.ndarray)):
                        json_results[key][sub_key] = list(sub_value)
                    else:
                        json_results[key][sub_key] = sub_value
            else:
                json_results[key] = value

        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=2, sort_keys=True, ensure_ascii=False)

        print(f"Results saved to: {results_path}")

        # Save summary
        self._print_summary_table()
        self._save_summary_text()

        ht = self.results.get("hyperparameter_tuning", {})
        if ht:
            plot_model_performance(ht, self.results_dir)
            export_summary_table(ht, self.results_dir)
            print(
                f"[plots] Saved model_performance_bar.png and model_summary.(csv|md) into {self.results_dir}"
            )

    def _print_summary_table(self):
        """
        Pretty console table for best models and metrics across experiments.
        Expects self.results['hyperparameter_tuning'] filled by tuner.
        """
        ht = self.results.get("hyperparameter_tuning", {})
        if not ht:
            return
        print("\n" + "=" * 60)
        print("MODEL SUMMARY (mean over outer folds)")
        print("=" * 60)
        header = f"{'Model':30} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6}"
        print(header)
        print("-" * len(header))
        for model_name, res in ht.items():
            am = res.get("aggregate_metrics", {})
            acc = am.get("accuracy", None)
            prec = am.get("precision", None)
            rec = am.get("recall", None)
            f1 = am.get("f1", res.get("mean_cv_score", None))

            def fmt(x):
                return (
                    f"{x:.4f}"
                    if isinstance(x, (int, float)) and x is not None
                    else "  NA  "
                )

            print(
                f"{model_name:30} | {fmt(acc)} | {fmt(prec)} | {fmt(rec)} | {fmt(f1)}"
            )
        print("=" * 60)

    def _save_summary_text(self):
        """
        Save the same summary table to a .txt file next to the JSON for quick viewing.
        """
        out = self.results_dir / "experiment_summary.txt"
        ht = self.results.get("hyperparameter_tuning", {})
        lines = []
        lines.append("MODEL SUMMARY (mean over outer folds)")
        lines.append("=" * 60)
        header = f"{'Model':30} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6}"
        lines.append(header)
        lines.append("-" * len(header))
        for model_name, res in ht.items():
            am = res.get("aggregate_metrics", {})
            acc = am.get("accuracy", None)
            prec = am.get("precision", None)
            rec = am.get("recall", None)
            f1 = am.get("f1", res.get("mean_cv_score", None))

            def fmt(x):
                return (
                    f"{x:.4f}"
                    if isinstance(x, (int, float)) and x is not None
                    else "  NA  "
                )

            lines.append(
                f"{model_name:30} | {fmt(acc)} | {fmt(prec)} | {fmt(rec)} | {fmt(f1)}"
            )
        lines.append("=" * 60)
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSummary written to: {out}")

    def run_complete_pipeline(self):
        """Run the complete experimental pipeline."""
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()

            # Step 3: Run hyperparameter tuning
            self.run_hyperparameter_tuning()

            # Step 4: Run learning curve experiments
            if self.include_learning_curves:
                self.run_learning_curves()
            else:
                print(
                    "\n[info] Learning curves are disabled (set include_learning_curves=True to enable)."
                )

            # Step 5: Create visualizations
            self.create_visualizations()

            # Step 6: Save results
            self.save_results()

        except Exception as e:
            raise


def main():
    """Main function to run the complete experimental pipeline."""
    # fast=True keeps small grids/folds; learning curves disabled by default
    pipeline = ExperimentalPipeline(fast=False, include_learning_curves=False)
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
