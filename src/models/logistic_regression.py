#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logistic Regression Implementation for Sentiment Classification

This module implements a Logistic Regression classifier that works with our
cross-validation framework and uses TF-IDF features.

Features:
- Wrapper class compatible with our CV framework
- Uses sklearn LogisticRegression internally
- Handles TF-IDF features properly
- Implements fit/predict interface
"""

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse, spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Any, Optional, Union, List

ArrayLike = Union[np.ndarray, List[str], spmatrix]


class LogisticRegressionSentiment:
    def __init__(self, **params):
        self.params = params
        self.model: Optional[LogisticRegression] = None
        self.vectorizer: Optional[TfidfVectorizer] = None

    def _ensure_vectorized(self, X: ArrayLike, fit: bool = False):
        # If X is list/np array of strings -> vectorize; else pass through (assume sparse/ndarray)
        if (
            isinstance(X, list)
            and (len(X) == 0 or isinstance(X[0], str))
            or (
                isinstance(X, np.ndarray)
                and X.dtype == object
                and X.size > 0
                and isinstance(X[0], str)
            )
        ):
            if self.vectorizer is None:
                if not fit:
                    raise ValueError(
                        "Vectorizer not fitted yet. Call fit() or pass vectorized features."
                    )
                max_feat = self.params.get("tfidf_max_features", 50000)
                ngram = self.params.get("tfidf_ngram", (1, 2))
                # Ensure ngram is a tuple
                if isinstance(ngram, list):
                    ngram = tuple(ngram)
                self.vectorizer = TfidfVectorizer(
                    max_features=max_feat,
                    ngram_range=ngram,
                    min_df=2,
                    sublinear_tf=True,
                )
                X_vec = self.vectorizer.fit_transform(X)
            else:
                X_vec = self.vectorizer.transform(X)
            return X_vec
        return X

    def fit(self, X: ArrayLike, y: np.ndarray):
        Xv = self._ensure_vectorized(X, fit=True)

        # Create and train model - filter out TF-IDF specific parameters
        lr_params = {
            k: v
            for k, v in self.params.items()
            if k not in ["tfidf_max_features", "tfidf_ngram"]
        }
        self.model = LogisticRegression(**lr_params, random_state=42)
        self.model.fit(Xv, y)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        Xv = self._ensure_vectorized(X, fit=False)
        return self.model.predict(Xv)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features (can be TF-IDF matrix or raw text)

        Returns:
            probabilities: Prediction probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        Xv = self._ensure_vectorized(X, fit=False)
        return self.model.predict_proba(Xv)

    # sklearn-like API for CV
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        self.params.update(params)
        return self


def create_logistic_regression_factory(hyperparams: Dict[str, Any]):
    """
    Create a factory function for Logistic Regression that works with our CV framework.

    Args:
        hyperparams: Dictionary of hyperparameters to test

    Returns:
        factory_function: Function that creates LogisticRegressionSentiment instances
    """

    def factory(params):
        return LogisticRegressionSentiment(**params)

    return factory
