#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Set Evaluation for Sentiment Classification Models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

from core.metrics import compute_all_metrics, f1_score, accuracy_score
from models.logistic_regression import create_logistic_regression_factory
from models.electra_sentiment import create_electra_factory
from models.bilstm_sentiment import create_bilstm_factory


class TestEvaluator:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize test evaluator.

        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.test_data = None
        self.train_data = None

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.model_selection import train_test_split

        # Stratified split to maintain class balance
        train_data, test_data = train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=data["label"],
        )

        self.train_data = train_data.reset_index(drop=True)
        self.test_data = test_data.reset_index(drop=True)

        print(f"Data split: {len(self.train_data)} train, {len(self.test_data)} test")
        print(
            f"Train class balance: {self.train_data['label'].value_counts(normalize=True).to_dict()}"
        )
        print(
            f"Test class balance: {self.test_data['label'].value_counts(normalize=True).to_dict()}"
        )

        return self.train_data, self.test_data

    def evaluate_model(
        self, model_name: str, best_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on test set.

        Args:
            model_name: Name of the model
            best_params: Best hyperparameters from CV

        Returns:
            Dictionary with test results
        """
        if self.test_data is None or self.train_data is None:
            raise ValueError("Must call split_data() first")

        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} on Test Set")
        print(f"{'='*60}")
        print(f"Best params: {best_params}")

        # Prepare data
        X_train_text = self.train_data["text"].tolist()
        y_train = self.train_data["label"].to_numpy()
        X_test_text = self.test_data["text"].tolist()
        y_test = self.test_data["label"].to_numpy()

        # Create model factory based on model name
        if "Logistic Regression" in model_name or "logreg" in model_name.lower():
            factory = create_logistic_regression_factory(best_params)
        elif "BiLSTM" in model_name or "bilstm" in model_name.lower():
            factory = create_bilstm_factory(best_params)
        elif "ELECTRA" in model_name or "electra" in model_name.lower():
            factory = create_electra_factory(best_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Train model
        print("Training model...")
        model = factory(best_params)
        model.fit(X_train_text, y_train)

        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(X_test_text)

        # Calculate metrics
        print("Calculating metrics...")
        test_metrics = compute_all_metrics(y_test, y_pred)

        # Calculate train metrics for comparison
        y_train_pred = model.predict(X_train_text)
        train_metrics = compute_all_metrics(y_train, y_train_pred)

        # Calculate overfitting indicators
        overfitting_gap = {
            "accuracy": train_metrics["accuracy"] - test_metrics["accuracy"],
            "f1_macro": train_metrics["f1_macro"] - test_metrics["f1_macro"],
            "precision_macro": train_metrics["precision_macro"]
            - test_metrics["precision_macro"],
            "recall_macro": train_metrics["recall_macro"]
            - test_metrics["recall_macro"],
        }

        results = {
            "model_name": model_name,
            "best_params": best_params,
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "overfitting_gap": overfitting_gap,
            "test_size": len(self.test_data),
            "train_size": len(self.train_data),
        }

        # Print results
        print(f"\nTest Set Results:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision_macro']:.4f}")
        print(f"  Recall:    {test_metrics['recall_macro']:.4f}")
        print(f"  F1-Score:  {test_metrics['f1_macro']:.4f}")

        print(f"\nTrain Set Results (for comparison):")
        print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
        print(f"  Precision: {train_metrics['precision_macro']:.4f}")
        print(f"  Recall:    {train_metrics['recall_macro']:.4f}")
        print(f"  F1-Score:  {train_metrics['f1_macro']:.4f}")

        print(f"\nOverfitting Gap (Train - Test):")
        print(f"  Accuracy:  {overfitting_gap['accuracy']:.4f}")
        print(f"  Precision: {overfitting_gap['precision_macro']:.4f}")
        print(f"  Recall:    {overfitting_gap['recall_macro']:.4f}")
        print(f"  F1-Score:  {overfitting_gap['f1_macro']:.4f}")

        # Overfitting assessment
        if overfitting_gap["f1_macro"] > 0.05:
            print(
                f"\nWARNING: Significant overfitting detected! (F1 gap: {overfitting_gap['f1_macro']:.4f})"
            )
        elif overfitting_gap["f1_macro"] > 0.02:
            print(
                f"\nModerate overfitting detected (F1 gap: {overfitting_gap['f1_macro']:.4f})"
            )
        else:
            print(
                f"\nNo significant overfitting (F1 gap: {overfitting_gap['f1_macro']:.4f})"
            )

        return results

    def evaluate_all_models(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all models from cross-validation results.

        Args:
            cv_results: Results from hyperparameter tuning

        Returns:
            Dictionary with all test results
        """
        test_results = {}

        for model_name, results in cv_results.items():
            if "cv_results" not in results or results["cv_results"] is None:
                print(f"Skipping {model_name}: No valid CV results")
                continue

            best_params = results["cv_results"]["best_params"]
            test_results[model_name] = self.evaluate_model(model_name, best_params)

        return test_results

    def create_comparison_table(self, test_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a comparison table of test results.

        Args:
            test_results: Results from evaluate_all_models

        Returns:
            DataFrame with comparison
        """
        comparison_data = []

        for model_name, results in test_results.items():
            test_metrics = results["test_metrics"]
            overfitting_gap = results["overfitting_gap"]

            comparison_data.append(
                {
                    "Model": model_name,
                    "Test_Accuracy": test_metrics["accuracy"],
                    "Test_F1": test_metrics["f1_macro"],
                    "Train_Accuracy": results["train_metrics"]["accuracy"],
                    "Train_F1": results["train_metrics"]["f1_macro"],
                    "Overfitting_Gap_Acc": overfitting_gap["accuracy"],
                    "Overfitting_Gap_F1": overfitting_gap["f1_macro"],
                    "Test_Size": results["test_size"],
                    "Train_Size": results["train_size"],
                }
            )

        return pd.DataFrame(comparison_data)


def main():
    """Example usage of TestEvaluator."""
    import json
    from pathlib import Path

    # Load data
    data_path = Path("data/imdb_clean.csv")
    if not data_path.exists():
        print("Dataset not found. Please run data preparation first.")
        return

    data = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(data)} samples")

    # Load CV results - find the most recent experiment results file
    results_dir = Path("results")
    if not results_dir.exists():
        print("Results directory not found. Please run hyperparameter tuning first.")
        return

    # Find the most recent experiment results file
    experiment_files = list(results_dir.glob("experiment_results_*.json"))
    if not experiment_files:
        print("No experiment results found. Please run hyperparameter tuning first.")
        return

    # Sort by modification time and get the most recent
    results_path = max(experiment_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading CV results from: {results_path}")

    with open(results_path, "r") as f:
        cv_results = json.load(f)

    # Create evaluator and split data
    evaluator = TestEvaluator(test_size=0.2, random_state=42)
    train_data, test_data = evaluator.split_data(data)

    # Evaluate all models
    test_results = evaluator.evaluate_all_models(cv_results["hyperparameter_tuning"])

    # Create comparison table
    comparison_df = evaluator.create_comparison_table(test_results)
    print("\n" + "=" * 80)
    print("FINAL TEST SET COMPARISON")
    print("=" * 80)
    print(comparison_df.round(4))

    # Save results
    output_path = Path("results/test_evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"\nTest results saved to: {output_path}")


if __name__ == "__main__":
    main()
