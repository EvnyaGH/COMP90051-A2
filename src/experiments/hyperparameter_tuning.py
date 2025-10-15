#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hyperparameter Tuning Implementation

This module implements hyperparameter tuning using nested cross-validation
for our sentiment classification models.

Features:
- Grid search with nested CV
- Support for different model types
- Configurable hyperparameter grids
- Results tracking and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.cross_validation import create_cv_framework, NestedCrossValidator
from core.metrics import f1_score, accuracy_score
import json
from pathlib import Path

# ---------------- Hyperparameter grids ----------------
# Keep each model with >= 1 param and >=3 choices

HYPERPARAMETER_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
        "solver": ["liblinear"],
        "max_iter": [1000],
        "tfidf_max_features": [50000],
        "tfidf_ngram": [(1,2)],
    },
    "bilstm": {
        "hidden_dim": [64, 128, 256],
        "embedding_dim": [100],
        "max_len": [256],
        "epochs": [6],
        "batch_size": [64],
        "lr": [1e-3],
        "dropout": [0.3],
    },
    "electra": {
        "learning_rate": [2e-5, 3e-5, 5e-5],
        "epochs": [3],
        "batch_size": [32],
        "max_len": [128],
    },
}
HYPERPARAMETER_GRIDS_FAST = {
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
        "solver": ["liblinear"],
        "max_iter": [500],
        "tfidf_max_features": [20000],
        "tfidf_ngram": [(1,1)],
    },
    "bilstm": {
        "hidden_dim": [64, 128, 256],
        "embedding_dim": [100],
        "max_len": [192],
        "epochs": [2],
        "batch_size": [64],
        "lr": [1e-3],
        "dropout": [0.3],
    },
    "electra": {
        "learning_rate": [2e-5, 3e-5, 5e-5],
        "epochs": [1],
        "batch_size": [32],
        "max_len": [96],
        "model_name": ["google/electra-small-discriminator"],
    },
}


class HyperparameterTuner:
    """
    Hyperparameter tuning using nested cross-validation.

    This class provides a unified interface for tuning hyperparameters
    across different model types.
    """

    def __init__(
        self,
        outer_folds: int = 10,
        inner_folds: int = 3,
        random_state: int = 42,
        scoring_func: Callable = f1_score,
        fast: bool = False,
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            outer_folds: Number of outer CV folds for final evaluation
            inner_folds: Number of inner CV folds for hyperparameter selection
            random_state: Random seed for reproducibility
            scoring_func: Scoring function to optimize
        """
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.random_state = random_state
        self.scoring_func = scoring_func
        self.fast = fast

        # Create CV framework
        self.outer_cv, self.inner_cv = create_cv_framework(
            outer_folds=outer_folds, inner_folds=inner_folds, random_state=random_state
        )

        # Store results
        self.results = {}

    def tune_model(
        self,
        X: List[Any],
        y: np.ndarray,
        model_factory: Callable,
        param_grid: Dict[str, List[Any]],
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for a specific model.

        Args:
            X: Feature matrix
            y: Target labels
            model_factory: Function that creates model instances
            param_grid: Dictionary of hyperparameters to test
            model_name: Name for this model (for results tracking)

        Returns:
            Dictionary containing tuning results
        """
        print(f"\n{'='*60}")
        print(f"Tuning hyperparameters for: {model_name}")
        print(f"{'='*60}")
        print(f"Parameter grid: {param_grid}")
        print(f"Data length: {len(X)}")
        print(f"Outer folds: {self.outer_folds}, Inner folds: {self.inner_folds}")

        # Create nested CV
        nested_cv = NestedCrossValidator(
            outer_cv=self.outer_cv,
            inner_cv=self.inner_cv,
            param_grid=param_grid,
            scoring_func=self.scoring_func,
            random_state=self.random_state,
        )

        # Run hyperparameter tuning
        cv_results = nested_cv.fit_and_evaluate(X, y, model_factory)

        # Store results
        self.results[model_name] = {
            "cv_results": cv_results,
            "param_grid": param_grid,
            "data_len": len(X),
            "best_params": cv_results["best_params"],
            "mean_cv_score": cv_results["mean_cv_score"],
            "std_cv_score": cv_results["std_cv_score"],
        }

        print(f"\nBest parameters: {cv_results['best_params']}")
        print(
            f"Best CV score: {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}"
        )

        return cv_results

    def tune_multiple_models(
        self,
        experiments: List[
            Tuple[str, np.ndarray, np.ndarray, Callable, Dict[str, List[Any]]]
        ],
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for multiple models.

        Args:
            experiments: List of (model_name, X, y, model_factory, param_grid) tuples

        Returns:
            Dictionary containing all tuning results
        """
        print("Starting hyperparameter tuning for multiple models...")
        print(f"Number of experiments: {len(experiments)}")

        all_results = {}

        for i, (model_name, X, y, model_factory, param_grid) in enumerate(experiments):
            print(f"\nExperiment {i+1}/{len(experiments)}: {model_name}")

            try:
                results = self.tune_model(X, y, model_factory, param_grid, model_name)
                all_results[model_name] = results

            except Exception as e:
                print(f"Error tuning {model_name}: {e}")
                all_results[model_name] = {"error": str(e), "cv_results": None}

        return all_results

    def get_best_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the best model configuration for each tuned model.

        Returns:
            Dictionary mapping model names to their best configurations
        """
        best_models = {}

        for model_name, results in self.results.items():
            if "cv_results" in results and results["cv_results"] is not None:
                best_models[model_name] = {
                    "best_params": results["best_params"],
                    "cv_score": results["mean_cv_score"],
                    "cv_std": results["std_cv_score"],
                }

        return best_models

    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance across all tuned models.

        Returns:
            DataFrame with model comparison results
        """
        comparison_data = []

        for model_name, results in self.results.items():
            if "cv_results" in results and results["cv_results"] is not None:
                comparison_data.append(
                    {
                        "Model": model_name,
                        "Best_Params": str(results["best_params"]),
                        "CV_Score": results["mean_cv_score"],
                        "CV_Std": results["std_cv_score"],
                        "Data_Len": results["data_len"],
                    }
                )

        return pd.DataFrame(comparison_data).sort_values("CV_Score", ascending=False)

    def save_results(self, filepath: str):
        """
        Save tuning results to file.

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

        print(f"Results saved to: {filepath}")

    def load_results(self, filepath: str):
        """
        Load tuning results from file.

        Args:
            filepath: Path to load results from
        """
        with open(filepath, "r") as f:
            self.results = json.load(f)

        print(f"Results loaded from: {filepath}")


# # Predefined hyperparameter grids for common models
# HYPERPARAMETER_GRIDS = {
#     "logistic_regression": {
#         "C": [0.01, 1.0, 100.0],
#         "max_iter": [1000, 5000],
#         "solver": ["liblinear", "lbfgs"],
#     },
#     "bilstm": {
#         "hidden_dim": [64, 128, 256],
#         "learning_rate": [0.001, 0.01, 0.1],
#         "dropout": [0.2, 0.4, 0.6],
#         "epochs": [10, 20, 50],
#     },
#     "electra": {
#         "learning_rate": [1e-5, 2e-5, 5e-5],
#         "batch_size": [8, 16, 32],
#         "max_length": [128, 256, 512],
#         "epochs": [2, 3, 5],
#     },
# }


def create_hyperparameter_tuner(
    outer_folds: int = 10,
    inner_folds: int = 3,
    random_state: int = 42,
    scoring_func: Callable = f1_score,
) -> HyperparameterTuner:
    """
    Create a hyperparameter tuner with specified configuration.

    Args:
        outer_folds: Number of outer CV folds
        inner_folds: Number of inner CV folds
        random_state: Random seed
        scoring_func: Scoring function to optimize

    Returns:
        Configured HyperparameterTuner instance
    """
    return HyperparameterTuner(
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        random_state=random_state,
        scoring_func=scoring_func,
    )


# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from src.models.logistic_regression import (
        LogisticRegressionSentiment,
        create_logistic_regression_factory,
    )

    print("Testing Hyperparameter Tuning...")

    # Create sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, n_redundant=0, random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create tuner
    tuner = create_hyperparameter_tuner(outer_folds=5, inner_folds=3)

    # Test with Logistic Regression
    print("\nTesting Logistic Regression hyperparameter tuning...")

    # Create model factory
    lr_factory = create_logistic_regression_factory(
        HYPERPARAMETER_GRIDS["logistic_regression"]
    )

    # Tune hyperparameters
    lr_results = tuner.tune_model(
        X_train,
        y_train,
        lr_factory,
        HYPERPARAMETER_GRIDS["logistic_regression"],
        "Logistic Regression",
    )

    # Get best model
    best_models = tuner.get_best_models()
    print(f"\nBest models: {best_models}")

    # Compare models
    comparison_df = tuner.compare_models()
    print(f"\nModel comparison:\n{comparison_df}")

    # Test best model on test set
    best_params = best_models["Logistic Regression"]["best_params"]
    best_model = LogisticRegressionSentiment(**best_params)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nTest set performance:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

    print("\nHyperparameter tuning implementation completed!")
