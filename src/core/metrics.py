#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance Metrics Implementation from Scratch

This module implements common classification metrics without using sklearn:
- Accuracy
- Precision (macro, micro, weighted)
- Recall (macro, micro, weighted)
- F1-Score (macro, micro, weighted)
- Confusion Matrix
- Classification Report

Requirements:
- Implement from scratch (no sklearn.metrics)
- Support binary and multiclass classification
- Return proper error handling
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from collections import Counter


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None
) -> np.ndarray:
    """
    Compute confusion matrix from scratch.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels to include in matrix (if None, use unique labels)

    Returns:
        Confusion matrix as 2D numpy array
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    n_labels = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # Initialize confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=int)

    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1

    return cm


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy score from scratch.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if len(y_true) == 0:
        return 0.0

    correct = np.sum(y_true == y_pred)
    return float(correct / len(y_true))


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    labels: Optional[List] = None,
    zero_division: str = "warn",
) -> Union[float, np.ndarray]:
    """
    Compute precision score from scratch.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging strategy ('binary', 'macro', 'micro', 'weighted')
        labels: List of labels to include (if None, use unique labels)
        zero_division: How to handle division by zero ('warn', 0, 1)

    Returns:
        Precision score(s)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels)

    # Calculate precision for each class
    precisions = []
    for i in range(len(labels)):
        tp = cm[i, i]  # True positives
        fp = np.sum(cm[:, i]) - tp  # False positives

        if tp + fp == 0:
            if zero_division == "warn":
                print(f"Warning: Precision is ill-defined for class {labels[i]}")
            precision = 0.0 if zero_division == 0 else 1.0
        else:
            precision = tp / (tp + fp)

        precisions.append(precision)

    precisions = np.array(precisions)

    # Apply averaging
    if average == "binary":
        if len(labels) != 2:
            raise ValueError("binary averaging requires exactly 2 classes")
        return float(precisions[1])  # Return precision for positive class

    elif average == "macro":
        return float(np.mean(precisions))

    elif average == "micro":
        # Micro-averaged precision = micro-averaged recall = accuracy
        return accuracy_score(y_true, y_pred)

    elif average == "weighted":
        # Weight by support (number of true instances for each label)
        support = np.sum(cm, axis=1)
        if np.sum(support) == 0:
            return 0.0
        return float(np.average(precisions, weights=support))

    elif average is None:
        # Return per-class precision scores
        return precisions

    else:
        raise ValueError(f"Unknown averaging strategy: {average}")


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    labels: Optional[List] = None,
    zero_division: str = "warn",
) -> Union[float, np.ndarray]:
    """
    Compute recall score from scratch.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging strategy ('binary', 'macro', 'micro', 'weighted')
        labels: List of labels to include (if None, use unique labels)
        zero_division: How to handle division by zero ('warn', 0, 1)

    Returns:
        Recall score(s)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels)

    # Calculate recall for each class
    recalls = []
    for i in range(len(labels)):
        tp = cm[i, i]  # True positives
        fn = np.sum(cm[i, :]) - tp  # False negatives

        if tp + fn == 0:
            if zero_division == "warn":
                print(f"Warning: Recall is ill-defined for class {labels[i]}")
            recall = 0.0 if zero_division == 0 else 1.0
        else:
            recall = tp / (tp + fn)

        recalls.append(recall)

    recalls = np.array(recalls)

    # Apply averaging
    if average == "binary":
        if len(labels) != 2:
            raise ValueError("binary averaging requires exactly 2 classes")
        return float(recalls[1])  # Return recall for positive class

    elif average == "macro":
        return float(np.mean(recalls))

    elif average == "micro":
        # Micro-averaged recall = micro-averaged precision = accuracy
        return accuracy_score(y_true, y_pred)

    elif average == "weighted":
        # Weight by support (number of true instances for each label)
        support = np.sum(cm, axis=1)
        if np.sum(support) == 0:
            return 0.0
        return float(np.average(recalls, weights=support))

    elif average is None:
        # Return per-class recall scores
        return recalls

    else:
        raise ValueError(f"Unknown averaging strategy: {average}")


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    labels: Optional[List] = None,
    zero_division: str = "warn",
) -> Union[float, np.ndarray]:
    """
    Compute F1 score from scratch.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging strategy ('binary', 'macro', 'micro', 'weighted')
        labels: List of labels to include (if None, use unique labels)
        zero_division: How to handle division by zero ('warn', 0, 1)

    Returns:
        F1 score(s)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    if average is None:
        # Calculate F1 for each class
        cm = confusion_matrix(y_true, y_pred, labels)
        f1_scores = []

        for i in range(len(labels)):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp

            if tp + fp == 0 or tp + fn == 0:
                f1 = 0.0 if zero_division == 0 else 1.0
            else:
                precision_i = tp / (tp + fp)
                recall_i = tp / (tp + fn)
                if precision_i + recall_i == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision_i * recall_i / (precision_i + recall_i)

            f1_scores.append(f1)

        return np.array(f1_scores)

    # Get precision and recall
    precision = precision_score(
        y_true, y_pred, average="macro", labels=labels, zero_division=zero_division
    )
    recall = recall_score(
        y_true, y_pred, average="macro", labels=labels, zero_division=zero_division
    )

    if average == "binary":
        if len(labels) != 2:
            raise ValueError("binary averaging requires exactly 2 classes")
        # For binary, we want F1 for positive class
        precision_pos = precision_score(
            y_true, y_pred, average="binary", labels=labels, zero_division=zero_division
        )
        recall_pos = recall_score(
            y_true, y_pred, average="binary", labels=labels, zero_division=zero_division
        )

        if precision_pos + recall_pos == 0:
            if zero_division == "warn":
                print("Warning: F1 score is ill-defined for binary classification")
            return 0.0 if zero_division == 0 else 1.0
        else:
            return 2 * precision_pos * recall_pos / (precision_pos + recall_pos)

    elif average == "macro":
        if precision + recall == 0:
            if zero_division == "warn":
                print("Warning: F1 score is ill-defined")
            return 0.0 if zero_division == 0 else 1.0
        else:
            return 2 * precision * recall / (precision + recall)

    elif average == "micro":
        # Micro-averaged F1 = accuracy
        return accuracy_score(y_true, y_pred)

    elif average == "weighted":
        # Calculate F1 for each class and weight by support
        cm = confusion_matrix(y_true, y_pred, labels)
        f1_scores = []
        support = np.sum(cm, axis=1)

        for i in range(len(labels)):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp

            if tp + fp == 0 or tp + fn == 0:
                f1 = 0.0 if zero_division == 0 else 1.0
            else:
                precision_i = tp / (tp + fp)
                recall_i = tp / (tp + fn)
                if precision_i + recall_i == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision_i * recall_i / (precision_i + recall_i)

            f1_scores.append(f1)

        if np.sum(support) == 0:
            return 0.0
        return float(np.average(f1_scores, weights=support))

    else:
        raise ValueError(f"Unknown averaging strategy: {average}")


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    digits: int = 2,
) -> str:
    """
    Generate classification report from scratch.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels to include (if None, use unique labels)
        target_names: Names for labels (if None, use label values)
        digits: Number of decimal places to show

    Returns:
        Formatted classification report string
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    if target_names is None:
        target_names = [str(label) for label in labels]

    if len(target_names) != len(labels):
        raise ValueError("target_names length must match number of labels")

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels)

    # Calculate metrics for each class
    precision = precision_score(y_true, y_pred, average=None, labels=labels)
    recall = recall_score(y_true, y_pred, average=None, labels=labels)
    f1 = f1_score(y_true, y_pred, average=None, labels=labels)

    # Calculate support (number of true instances for each label)
    support = np.sum(cm, axis=1)

    # Calculate macro and weighted averages
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)

    precision_weighted = np.average(precision, weights=support)
    recall_weighted = np.average(recall, weights=support)
    f1_weighted = np.average(f1, weights=support)

    # Format the report
    width = max(len(name) for name in target_names)
    width = max(width, len("accuracy"))

    report = (
        f"{'':>{width}} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}\n"
    )
    report += f"{'':>{width}} {'':>9} {'':>9} {'':>9} {'':>9}\n"

    for i, (name, p, r, f, s) in enumerate(
        zip(target_names, precision, recall, f1, support)
    ):
        report += f"{name:>{width}} {p:>9.{digits}f} {r:>9.{digits}f} {f:>9.{digits}f} {s:>9}\n"

    report += f"{'':>{width}} {'':>9} {'':>9} {'':>9} {'':>9}\n"
    report += f"{'macro avg':>{width}} {precision_macro:>9.{digits}f} {recall_macro:>9.{digits}f} {f1_macro:>9.{digits}f} {np.sum(support):>9}\n"
    report += f"{'weighted avg':>{width}} {precision_weighted:>9.{digits}f} {recall_weighted:>9.{digits}f} {f1_weighted:>9.{digits}f} {np.sum(support):>9}\n"

    return report


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all standard classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing all metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


# Example usage and testing
if __name__ == "__main__":
    # Test the metrics
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Create sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, n_redundant=0, random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a simple model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Test metrics
    print("Testing metrics implementation...")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    print("\nAll Metrics:")
    all_metrics = compute_all_metrics(y_test, y_pred)
    for metric, value in all_metrics.items():
        print(f"{metric}: {value:.4f}")
