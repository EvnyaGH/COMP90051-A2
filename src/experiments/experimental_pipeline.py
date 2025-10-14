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
import seaborn as sns

# Import our modules
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.metrics import f1_score, accuracy_score, compute_all_metrics
from extract_features import do_tfidf, do_w2v, do_electra
from prepare_dataset import main as prepare_data
from models.logistic_regression import create_logistic_regression_factory
from models.electra_sentiment import create_electra_factory
from models.bilstm_sentiment import create_bilstm_factory
from .hyperparameter_tuning import HyperparameterTuner, HYPERPARAMETER_GRIDS
from .learning_curves import LearningCurveExperiment


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
    ):
        """
        Initialize experimental pipeline.

        Args:
            data_dir: Directory containing dataset files
            results_dir: Directory to save results
            features_dir: Directory to save/load features
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.random_state = random_state
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
        self.hyperparams = {
            "logreg": HYPERPARAMETER_GRIDS["logistic_regression"],
            "bilstm": HYPERPARAMETER_GRIDS["bilstm"],
            "electra": HYPERPARAMETER_GRIDS["electra"],
        }

    def load_and_prepare_data(self):
        """Load and prepare the IMDB dataset."""
        print("=" * 60)
        print("STEP 1: Loading and Preparing Data")
        print("=" * 60)

        # Check if clean data exists
        clean_path = self.data_dir / "imdb_clean.csv"
        if not clean_path.exists():
            print("[info] Clean file not found; preparing dataset...")
            prepare_data()

        # Load clean data
        self.data = pd.read_csv(clean_path)
        print(
            f"[data] rows={len(self.data)}, balance={self.data['label'].value_counts().to_dict()}"
        )

        # Show sample data
        print("\nSample data:")
        print(self.data.head())

    # def extract_features(self):
    #     """Extract all three types of features."""
    #     print("\n" + "=" * 60)
    #     print("STEP 2: Extracting Features")
    #     print("=" * 60)

    #     # Extract TF-IDF features
    #     print("Extracting TF-IDF features...")
    #     do_tfidf(
    #         self.data_dir / "imdb_clean.csv",
    #         self.features_dir,
    #         max_word_features=50000,
    #         max_char_features=100000,
    #     )

    #     # Extract Word2Vec features
    #     print("Extracting Word2Vec features...")
    #     do_w2v(
    #         self.data_dir / "imdb_clean.csv", self.features_dir, vec_size=100, epochs=5
    #     )

    #     # Extract ELECTRA features
    #     print("Extracting ELECTRA features...")
    #     do_electra(
    #         self.data_dir / "imdb_clean.csv",
    #         self.features_dir,
    #         batch_size=16,
    #         max_len=256,
    #         pool="cls",
    #     )

    #     # Load features into memory
    #     self._load_features()

    # def _load_features(self):
    #     """Load extracted features into memory."""
    #     print("\nLoading features into memory...")

    #     # Load TF-IDF features
    #     tfidf_path = self.features_dir / "imdb_tfidf.npz"
    #     if tfidf_path.exists():
    #         from scipy.sparse import load_npz

    #         self.features["tfidf"] = load_npz(tfidf_path).toarray()
    #         print(f"Loaded TF-IDF features: {self.features['tfidf'].shape}")

    #     # Load Word2Vec features
    #     w2v_path = self.features_dir / "imdb_w2v_avg.npy"
    #     if w2v_path.exists():
    #         self.features["w2v"] = np.load(w2v_path)
    #         print(f"Loaded Word2Vec features: {self.features['w2v'].shape}")

    #     # Load ELECTRA features
    #     electra_path = self.features_dir / "imdb_electra_cls.npy"
    #     if electra_path.exists():
    #         self.features["electra"] = np.load(electra_path)
    #         print(f"Loaded ELECTRA features: {self.features['electra'].shape}")

    def run_hyperparameter_tuning(self):
        """Run hyperparameter tuning for all models."""
        print("\n" + "=" * 60)
        print("STEP 2: Hyperparameter Tuning")
        print("=" * 60)

        assert self.data is not None
        X_text = self.data["text"].tolist()
        y = self.data["label"].to_numpy()

        # Create hyperparameter tuner
        tuner = HyperparameterTuner(
            outer_folds=10,
            inner_folds=3,
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
            training_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            outer_folds=5,  # Fewer folds for learning curves
            inner_folds=3,
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
                        json_results[key][sub_key] = sub_value
                    elif isinstance(sub_value, (list, np.ndarray)):
                        json_results[key][sub_key] = list(sub_value)
                    else:
                        json_results[key][sub_key] = sub_value
            else:
                json_results[key] = value

        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {results_path}")

        # Save summary
        self._save_summary()

    def run_complete_pipeline(self):
        """Run the complete experimental pipeline."""
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()

            # Step 3: Run hyperparameter tuning
            self.run_hyperparameter_tuning()

            # Step 4: Run learning curve experiments
            self.run_learning_curves()

            # Step 5: Create visualizations
            self.create_visualizations()

            # Step 6: Save results
            self.save_results()

        except Exception as e:
            raise


def main():
    """Main function to run the complete experimental pipeline."""
    pipeline = ExperimentalPipeline()
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
