# logistic_regression.py
from typing import Any, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline


class TfidfLogRegEstimator:
    """将 TF-IDF/Hashing 与 LR 封装为 sklearn 风格估计器。

    params:
      - use_hashing: 是否使用 HashingVectorizer（无拟合，提速）
      - n_features:  哈希维度
      - tfidf_max_features, tfidf_ngram: 仅在 use_hashing=False 时生效
      - C, solver, max_iter: LR 参数
    """

    def __init__(self, **params: Dict[str, Any]):
        self.p = params
        self.model = None

    def fit(self, texts, y):
        use_hashing = self.p.get("use_hashing", False)
        if use_hashing:
            hv = HashingVectorizer(n_features=self.p.get("n_features", 2 ** 20), alternate_sign=False)
            # 线性分类器（这里用 LogisticRegression 需要密集特征；我们选 SGDClassifier 支持稀疏）
            cls = SGDClassifier(loss="log", max_iter=self.p.get("max_iter", 1000))
            self.model = make_pipeline(hv, cls)
        else:
            tfidf = TfidfVectorizer(
                max_features=self.p.get("tfidf_max_features", 20000),
                ngram_range=self.p.get("tfidf_ngram", (1, 1)),
                lowercase=True,
            )
            X = tfidf.fit_transform(texts)
            cls = LogisticRegression(
                C=self.p.get("C", 1.0),
                solver=self.p.get("solver", "liblinear"),
                max_iter=self.p.get("max_iter", 500),
            )
            cls.fit(X, y)
            # 将向量器与分类器保留在对象中以便 predict 使用
            self.model = (tfidf, cls)
        return self

    def predict(self, texts):
        if isinstance(self.model, tuple):
            tfidf, cls = self.model
            X = tfidf.transform(texts)
            return cls.predict(X)
        else:
            return self.model.predict(texts)

    def predict_proba(self, texts):
        if isinstance(self.model, tuple):
            tfidf, cls = self.model
            X = tfidf.transform(texts)
            if hasattr(cls, "predict_proba"):
                return cls.predict_proba(X)
            # SGDClassifier(log) 可用 decision_function 近似
            scores = cls.decision_function(X)
            # 二分类 sigmoid
            from scipy.special import expit
            p1 = expit(scores)
            return np.vstack([1 - p1, p1]).T
        else:
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(texts)
            scores = self.model.decision_function(texts)
            from scipy.special import expit
            p1 = expit(scores)
            return np.vstack([1 - p1, p1]).T


def create_lr_factory():
    def factory(params: Dict[str, Any]):
        return TfidfLogRegEstimator(**params)

    return factory