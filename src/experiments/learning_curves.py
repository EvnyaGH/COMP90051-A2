#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Learning Curve Experiment Implementation

This module implements learning curve experiments to analyze how model performance
changes with training set size.

Features:
- Generate learning curves for different models
- Support for multiple training set sizes
- Error bar calculation
- Visualization of results
- Integration with cross-validation framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.cross_validation import create_cv_framework, NestedCrossValidator
from core.metrics import f1_score, accuracy_score
import json
from pathlib import Path


class LearningCurveExperiment:
    """
    Learning curve experiment for analyzing model performance vs training set size.

    This class provides functionality to generate learning curves for different
    models and training set sizes.
    """

    def __init__(
        self,
        training_sizes: List[float] = None,
        outer_folds: int = 5,
        inner_folds: int = 3,
        random_state: int = 42,
        scoring_func: Callable = f1_score,
    ):
        """
        Initialize learning curve experiment.

        Args:
            training_sizes: List of training set size ratios (0.0 to 1.0)
            outer_folds: Number of outer CV folds for evaluation
            inner_folds: Number of inner CV folds for hyperparameter tuning
            random_state: Random seed for reproducibility
            scoring_func: Scoring function to optimize
        """
        if training_sizes is None:
            training_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        self.training_sizes = training_sizes
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.random_state = random_state
        self.scoring_func = scoring_func

        # Store results
        self.results = {}

    def generate_learning_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        param_grid: Dict[str, List[Any]],
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Generate learning curve for a specific model.

        Args:
            X: Feature matrix
            y: Target labels
            model_factory: Function that creates model instances
            param_grid: Dictionary of hyperparameters to test
            model_name: Name for this model

        Returns:
            Dictionary containing learning curve results
        """
        print(f"\n{'='*60}")
        print(f"Generating learning curve for: {model_name}")
        print(f"{'='*60}")
        print(f"Training sizes: {self.training_sizes}")
        print(f"Data shape: {X.shape}")

        # Set random seed for reproducible sampling
        np.random.seed(self.random_state)

        # Store results for this model
        model_results = {
            "training_sizes": [],
            "scores": [],
            "score_stds": [],
            "best_params": [],
            "data_shape": X.shape,
        }

        # Generate learning curve for each training size
        for i, size_ratio in enumerate(self.training_sizes):
            print(f"\nTraining size {i+1}/{len(self.training_sizes)}: {size_ratio:.1%}")

            # Sample data for this training size
            n_samples = int(len(X) * size_ratio)
            if n_samples < self.outer_folds:
                print(
                    f"  Skipping size {size_ratio:.1%} - too few samples ({n_samples})"
                )
                continue

            # Stratified sampling to maintain class balance
            indices = self._stratified_sample(X, y, n_samples)
            X_sample = X[indices]
            y_sample = y[indices]

            print(f"  Sampled {len(X_sample)} samples")

            try:
                # Create CV framework for this sample
                outer_cv, inner_cv = create_cv_framework(
                    outer_folds=self.outer_folds,
                    inner_folds=self.inner_folds,
                    random_state=self.random_state,
                )

                # Run nested CV for hyperparameter tuning
                nested_cv = NestedCrossValidator(
                    outer_cv=outer_cv,
                    inner_cv=inner_cv,
                    param_grid=param_grid,
                    scoring_func=self.scoring_func,
                    random_state=self.random_state,
                )

                # Get CV results
                cv_results = nested_cv.fit_and_evaluate(
                    X_sample, y_sample, model_factory
                )

                # Store results
                model_results["training_sizes"].append(size_ratio)
                model_results["scores"].append(cv_results["mean_cv_score"])
                model_results["score_stds"].append(cv_results["std_cv_score"])
                model_results["best_params"].append(cv_results["best_params"])

                print(
                    f"  Score: {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}"
                )

            except Exception as e:
                print(f"  Error at size {size_ratio:.1%}: {e}")
                # Add placeholder values
                model_results["training_sizes"].append(size_ratio)
                model_results["scores"].append(0.0)
                model_results["score_stds"].append(0.0)
                model_results["best_params"].append({})

        # Store results
        self.results[model_name] = model_results

        print(f"\nLearning curve completed for {model_name}")
        print(
            f"Final score: {model_results['scores'][-1]:.4f} ± {model_results['score_stds'][-1]:.4f}"
        )

        return model_results

    def generate_multiple_learning_curves(
        self,
        experiments: List[
            Tuple[str, np.ndarray, np.ndarray, Callable, Dict[str, List[Any]]]
        ],
    ) -> Dict[str, Any]:
        """
        Generate learning curves for multiple models.

        Args:
            experiments: List of (model_name, X, y, model_factory, param_grid) tuples

        Returns:
            Dictionary containing all learning curve results
        """
        print("Starting learning curve experiments for multiple models...")
        print(f"Number of experiments: {len(experiments)}")

        all_results = {}

        for i, (model_name, X, y, model_factory, param_grid) in enumerate(experiments):
            print(f"\nExperiment {i+1}/{len(experiments)}: {model_name}")

            try:
                if isinstance(X, list):
                    X = np.asarray(X, dtype=object)
                results = self.generate_learning_curve(
                    X, y, model_factory, param_grid, model_name
                )
                all_results[model_name] = results

            except Exception as e:
                print(f"Error generating learning curve for {model_name}: {e}")
                all_results[model_name] = {
                    "error": str(e),
                    "training_sizes": [],
                    "scores": [],
                    "score_stds": [],
                }

        return all_results

    def _stratified_sample(
        self, X: np.ndarray, y: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """
        Perform stratified sampling to maintain class balance.

        Args:
            X: Feature matrix
            y: Target labels
            n_samples: Number of samples to select

        Returns:
            Indices of selected samples
        """
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(y, return_counts=True)

        # Calculate samples per class
        samples_per_class = n_samples // len(unique_classes)
        remainder = n_samples % len(unique_classes)

        selected_indices = []

        for i, class_label in enumerate(unique_classes):
            # Get indices for this class
            class_indices = np.where(y == class_label)[0]

            # Calculate how many samples to take from this class
            class_n_samples = samples_per_class
            if i < remainder:  # Distribute remainder across first few classes
                class_n_samples += 1

            # Take samples from this class
            if len(class_indices) >= class_n_samples:
                selected = np.random.choice(
                    class_indices, class_n_samples, replace=False
                )
            else:
                selected = class_indices  # Take all available samples

            selected_indices.extend(selected)

        return np.array(selected_indices)

    def plot_learning_curves(
        self,
        save_path: str = None,
        title: str = "Learning Curves",
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        Plot learning curves for all models.

        Args:
            save_path: Path to save the plot
            title: Title for the plot
            figsize: Figure size
        """
        if not self.results:
            print("No results to plot. Run learning curve experiments first.")
            return

        plt.figure(figsize=figsize)

        for model_name, results in self.results.items():
            if "error" in results:
                continue

            training_sizes = results["training_sizes"]
            scores = results["scores"]
            score_stds = results["score_stds"]

            # Convert to percentages
            sizes_pct = [s * 100 for s in training_sizes]

            # Plot learning curve
            plt.plot(sizes_pct, scores, marker="o", label=model_name, linewidth=2)
            plt.fill_between(
                sizes_pct,
                np.array(scores) - np.array(score_stds),
                np.array(scores) + np.array(score_stds),
                alpha=0.3,
            )

        plt.xlabel("Training Set Size (%)")
        plt.ylabel(f"{self.scoring_func.__name__.title()} Score")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Learning curves plot saved to: {save_path}")

        plt.show()

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of learning curve results.

        Returns:
            DataFrame with learning curve summary
        """
        summary_data = []

        for model_name, results in self.results.items():
            if "error" in results:
                continue

            training_sizes = results["training_sizes"]
            scores = results["scores"]
            score_stds = results["score_stds"]

            if not training_sizes:
                continue

            # Calculate improvement from smallest to largest training size
            if len(scores) > 1:
                improvement = scores[-1] - scores[0]
            else:
                improvement = 0.0

            summary_data.append(
                {
                    "Model": model_name,
                    "Min_Score": min(scores) if scores else 0.0,
                    "Max_Score": max(scores) if scores else 0.0,
                    "Final_Score": scores[-1] if scores else 0.0,
                    "Improvement": improvement,
                    "Data_Shape": results["data_shape"],
                }
            )

        if not summary_data:
            # Return empty DataFrame with expected columns if no data
            return pd.DataFrame(
                columns=[
                    "Model",
                    "Min_Score",
                    "Max_Score",
                    "Final_Score",
                    "Improvement",
                    "Data_Shape",
                ]
            )
        return pd.DataFrame(summary_data).sort_values("Final_Score", ascending=False)

    def save_results(self, filepath: str):
        """
        Save learning curve results to file.

        Args:
            filepath: Path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, results in self.results.items():
            json_results[model_name] = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[model_name][key] = value
                elif isinstance(value, (list, np.ndarray)):
                    json_results[model_name][key] = list(value)
                else:
                    json_results[model_name][key] = value

        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Learning curve results saved to: {filepath}")


# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from logistic_regression import (
        LogisticRegressionSentiment,
        create_logistic_regression_factory,
    )
    from hyperparameter_tuning import HYPERPARAMETER_GRIDS

    print("Testing Learning Curve Experiments...")

    # Create sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, n_redundant=0, random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create learning curve experiment
    lc_experiment = LearningCurveExperiment(
        training_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        outer_folds=3,  # Fewer folds for faster testing
        inner_folds=2,
    )

    # Test with Logistic Regression
    print("\nTesting Logistic Regression learning curve...")

    # Create model factory
    lr_factory = create_logistic_regression_factory(
        HYPERPARAMETER_GRIDS["logistic_regression"]
    )

    # Generate learning curve
    lr_results = lc_experiment.generate_learning_curve(
        X_train,
        y_train,
        lr_factory,
        HYPERPARAMETER_GRIDS["logistic_regression"],
        "Logistic Regression",
    )

    # Get summary
    summary_df = lc_experiment.get_summary()
    print(f"\nLearning curve summary:\n{summary_df}")

    # Plot learning curves
    lc_experiment.plot_learning_curves(
        save_path="learning_curves_test.png", title="Learning Curves Test"
    )

    print("\nLearning curve experiments completed!")
