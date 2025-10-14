#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-Validation Framework Implementation from Scratch

This module implements:
1. K-Fold Cross-Validation (outer loop for final evaluation)
2. Nested Cross-Validation (inner loop for hyperparameter tuning)
3. Stratified sampling to maintain class balance
4. Support for different data types (text, features, labels)

Requirements:
- Implement from scratch (no sklearn.cross_validation)
- Support 10-fold outer CV, 3-fold inner CV
- Maintain class balance with stratified sampling
- Return proper train/validation/test splits
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Callable, Union, Optional
from abc import ABC, abstractmethod
import random
from collections import Counter

try:
    from scipy.sparse import spmatrix
except ImportError:
    spmatrix = tuple()  # type: ignore


class BaseCrossValidator(ABC):
    """Abstract base class for cross-validation strategies."""

    @abstractmethod
    def split(self, X: Any, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation splits."""
        pass


class StratifiedKFold(BaseCrossValidator):
    """
    Stratified K-Fold Cross-Validation from scratch.

    Ensures each fold maintains the same class distribution as the original dataset.
    """

    def __init__(
        self, n_splits: int = 10, shuffle: bool = True, random_state: int = 42
    ):
        """
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: Any, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate stratified train/validation splits.

        Args:
            X: Features (can be any type, we only use indices)
            y: Labels (must be 1D array)

        Returns:
            List of (train_indices, val_indices) tuples
        """
        if len(y.shape) != 1:
            raise ValueError("y must be 1D array")

        n_samples = len(y)
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have {self.n_splits} folds with {n_samples} samples"
            )

        # Set random seed
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)

        # Check if we have enough samples for each class in each fold
        min_class_count = np.min(class_counts)
        if min_class_count < self.n_splits:
            raise ValueError(f"Each class must have at least {self.n_splits} samples")

        # Group indices by class
        class_indices = {}
        for i, class_label in enumerate(unique_classes):
            class_indices[class_label] = np.where(y == class_label)[0]

        # Shuffle indices within each class if requested
        if self.shuffle:
            for class_label in class_indices:
                np.random.shuffle(class_indices[class_label])

        # Calculate fold sizes for each class
        fold_sizes = {}
        for class_label in unique_classes:
            class_count = len(class_indices[class_label])
            base_fold_size = class_count // self.n_splits
            remainder = class_count % self.n_splits

            fold_sizes[class_label] = [base_fold_size] * self.n_splits
            # Distribute remainder across first few folds
            for i in range(remainder):
                fold_sizes[class_label][i] += 1

        # Generate splits
        splits = []
        for fold in range(self.n_splits):
            train_indices = []
            val_indices = []

            for class_label in unique_classes:
                class_idx = class_indices[class_label]
                fold_size = fold_sizes[class_label][fold]

                # Calculate start and end indices for this fold
                start_idx = sum(fold_sizes[class_label][:fold])
                end_idx = start_idx + fold_size

                # Validation set for this fold
                val_indices.extend(class_idx[start_idx:end_idx])

                # Training set (all other samples from this class)
                train_indices.extend(class_idx[:start_idx])
                train_indices.extend(class_idx[end_idx:])

            # Convert to numpy arrays and shuffle
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)

            if self.shuffle:
                np.random.shuffle(train_indices)
                np.random.shuffle(val_indices)

            splits.append((train_indices, val_indices))

        return splits


class NestedCrossValidator:
    """
    Nested Cross-Validation for hyperparameter tuning.

    Outer loop: Final model evaluation (10-fold)
    Inner loop: Hyperparameter selection (3-fold)
    """

    def __init__(
        self,
        outer_cv: BaseCrossValidator,
        inner_cv: BaseCrossValidator,
        param_grid: Dict[str, List[Any]],
        scoring_func: Callable[[np.ndarray, np.ndarray], float],
        random_state: int = 42,
    ):
        """
        Args:
            outer_cv: Cross-validator for outer loop (final evaluation)
            inner_cv: Cross-validator for inner loop (hyperparameter tuning)
            param_grid: Dictionary of parameter names to lists of values
            scoring_func: Function that takes (y_true, y_pred) and returns score
            random_state: Random seed for reproducibility
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.param_grid = param_grid
        self.scoring_func = scoring_func
        self.random_state = random_state

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        import itertools

        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))

        return combinations

    def _evaluate_params(
        self,
        X: Any,
        y: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        model_factory: Callable[[Dict[str, Any]], Any],
        params: Dict[str, Any],
    ) -> float:
        """
        Evaluate hyperparameters on a single train/val split.

        Args:
            X: Features
            y: Labels
            train_idx: Training indices
            val_idx: Validation indices
            model_factory: Function that creates model with given parameters
            params: Hyperparameters to evaluate

        Returns:
            Average score across inner CV folds
        """
        # Get training data for this outer fold
        X_train = self._get_subset(X, train_idx)
        y_train = y[train_idx]

        # Inner CV on training data
        inner_splits = self.inner_cv.split(X_train, y_train)
        scores = []

        for inner_train_idx, inner_val_idx in inner_splits:
            # Get inner train/val data
            X_inner_train = self._get_subset(X_train, inner_train_idx)
            y_inner_train = y_train[inner_train_idx]
            X_inner_val = self._get_subset(X_train, inner_val_idx)
            y_inner_val = y_train[inner_val_idx]

            # Train model with these parameters
            model = model_factory(params)
            model.fit(X_inner_train, y_inner_train)

            # Evaluate on inner validation set
            y_pred = model.predict(X_inner_val)
            score = self.scoring_func(y_inner_val, y_pred)
            scores.append(score)

        return np.mean(scores)

    def _get_subset(self, X: Any, indices: np.ndarray) -> Any:
        """Get subset of data based on indices."""
        if isinstance(X, np.ndarray):
            return X[indices]
        elif isinstance(X, list):
            return [X[i] for i in indices]
        elif isinstance(X, pd.DataFrame):
            return X.iloc[indices]
        else:
            # Assume it's indexable
            return X[indices]

    def fit_and_evaluate(
        self, X: Any, y: np.ndarray, model_factory: Callable[[Dict[str, Any]], Any]
    ) -> Dict[str, Any]:
        """
        Perform nested cross-validation.

        Args:
            X: Features
            y: Labels
            model_factory: Function that creates model with given parameters

        Returns:
            Dictionary containing:
            - best_params: Best hyperparameters found
            - cv_scores: Scores for each outer fold
            - param_scores: All parameter combinations and their scores
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations()

        # Outer CV splits
        outer_splits = self.outer_cv.split(X, y)
        n_outer_folds = len(outer_splits)

        # Store results
        cv_scores = []
        param_scores = {str(params): [] for params in param_combinations}
        best_params_per_fold = []

        print(
            f"Starting nested CV: {n_outer_folds} outer folds, {self.inner_cv.n_splits} inner folds"
        )
        print(f"Testing {len(param_combinations)} parameter combinations")

        for fold, (train_idx, val_idx) in enumerate(outer_splits):
            print(f"\nOuter Fold {fold + 1}/{n_outer_folds}")

            # Evaluate each parameter combination
            param_scores_fold = {}
            for i, params in enumerate(param_combinations):
                print(f"  Testing params {i+1}/{len(param_combinations)}: {params}")
                score = self._evaluate_params(
                    X, y, train_idx, val_idx, model_factory, params
                )
                param_scores_fold[str(params)] = score
                param_scores[str(params)].append(score)

            # Find best parameters for this fold
            best_params = max(param_scores_fold.items(), key=lambda x: x[1])
            best_params_per_fold.append((eval(best_params[0]), best_params[1]))

            # Evaluate best model on outer validation set
            X_val = self._get_subset(X, val_idx)
            y_val = y[val_idx]

            best_model = model_factory(eval(best_params[0]))
            best_model.fit(self._get_subset(X, train_idx), y[train_idx])
            y_pred = best_model.predict(X_val)
            fold_score = self.scoring_func(y_val, y_pred)
            cv_scores.append(fold_score)

            print(f"  Best params: {eval(best_params[0])}")
            print(f"  Fold score: {fold_score:.4f}")

        # Find overall best parameters (most frequently selected)
        param_frequency = Counter([str(params) for params, _ in best_params_per_fold])
        overall_best_params = eval(max(param_frequency.items(), key=lambda x: x[1])[0])

        results = {
            "best_params": overall_best_params,
            "cv_scores": cv_scores,
            "param_scores": param_scores,
            "best_params_per_fold": best_params_per_fold,
            "mean_cv_score": np.mean(cv_scores),
            "std_cv_score": np.std(cv_scores),
        }

        print(f"\nNested CV Results:")
        print(f"Best parameters: {overall_best_params}")
        print(
            f"Mean CV score: {results['mean_cv_score']:.4f} ± {results['std_cv_score']:.4f}"
        )

        return results


def create_cv_framework(
    outer_folds: int = 10, inner_folds: int = 3, random_state: int = 42
) -> Tuple[StratifiedKFold, StratifiedKFold]:
    """
    Create cross-validation framework for the project.

    Args:
        outer_folds: Number of outer CV folds (final evaluation)
        inner_folds: Number of inner CV folds (hyperparameter tuning)
        random_state: Random seed

    Returns:
        Tuple of (outer_cv, inner_cv) objects
    """
    outer_cv = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=random_state
    )

    return outer_cv, inner_cv


# Example usage and testing
if __name__ == "__main__":
    # Test the CV framework
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Create sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, n_redundant=0, random_state=42
    )

    # Create CV framework
    outer_cv, inner_cv = create_cv_framework(outer_folds=5, inner_folds=3)

    # Test basic CV
    print("Testing basic cross-validation...")
    splits = outer_cv.split(X, y)
    print(f"Generated {len(splits)} splits")

    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
        # Check class balance
        train_classes = np.bincount(y[train_idx])
        val_classes = np.bincount(y[val_idx])
        print(f"  Train classes: {train_classes}")
        print(f"  Val classes: {val_classes}")
        break  # Just show first fold

    # Test nested CV
    print("\nTesting nested cross-validation...")

    def model_factory(params):
        return LogisticRegression(**params, random_state=42)

    param_grid = {"C": [0.1, 1.0, 10.0], "max_iter": [100, 1000]}

    nested_cv = NestedCrossValidator(
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        param_grid=param_grid,
        scoring_func=accuracy_score,
    )

    results = nested_cv.fit_and_evaluate(X, y, model_factory)
    print(f"Best parameters: {results['best_params']}")
    print(f"CV score: {results['mean_cv_score']:.4f} ± {results['std_cv_score']:.4f}")
