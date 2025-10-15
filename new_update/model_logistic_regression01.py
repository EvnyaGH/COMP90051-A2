# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:37:54 2025

@author: Ymmmmmm
"""

# logistic_regression.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class TfidfLogRegEstimator:
    def __init__(self,
                 word_ngram=(1,2), char_ngram=(3,5),
                 max_word_features=50000, max_char_features=100000,
                 C=1.0, penalty="l2"):
        self.word_ngram = word_ngram
        self.char_ngram = char_ngram
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features
        self.C = C
        self.penalty = penalty
        self.tfw: Optional[TfidfVectorizer] = None
        self.tfc: Optional[TfidfVectorizer] = None
        self.clf: Optional[LogisticRegression] = None

    def _fit_vectorizer(self, texts: List[str]):
        self.tfw = TfidfVectorizer(ngram_range=self.word_ngram, min_df=2,
                                   max_features=self.max_word_features, sublinear_tf=True)
        self.tfc = TfidfVectorizer(analyzer="char", ngram_range=self.char_ngram, min_df=2,
                                   max_features=self.max_char_features)
        Xw = self.tfw.fit_transform(texts)
        Xc = self.tfc.fit_transform(texts)
        return hstack([Xw, Xc])

    def _transform(self, texts: List[str]):
        Xw = self.tfw.transform(texts)
        Xc = self.tfc.transform(texts)
        return hstack([Xw, Xc])

    def fit(self, X_texts: List[str], y: np.ndarray):
        A = self._fit_vectorizer(X_texts)           # 只在训练集里 fit
        self.clf = LogisticRegression(solver="liblinear", penalty=self.penalty,
                                      C=self.C, max_iter=1000)
        self.clf.fit(A, y)
        return self

    def predict(self, X_texts: List[str]) -> np.ndarray:
        A = self._transform(X_texts)
        return self.clf.predict(A)

    def predict_proba(self, X_texts: List[str]) -> np.ndarray:
        A = self._transform(X_texts)
        return self.clf.predict_proba(A)

def create_lr_factory(defaults: Dict[str, Any]):
    def factory(params: Dict[str, Any]):
        cfg = {**defaults, **params}
        return TfidfLogRegEstimator(**cfg)
    return factory
