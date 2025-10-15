# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:34:32 2025

@author: Ymmmmmm
"""

# cross_validation.py
from __future__ import annotations
import random, itertools, json
from dataclasses import dataclass
from typing import List, Dict, Callable, Any
import numpy as np

from metrics import f1_score, accuracy_score  # 自写指标，勿用 sklearn.metrics ！ :contentReference[oaicite:6]{index=6}

def stratified_kfold_indices(y: np.ndarray, k: int = 10, seed: int = 42):
    rng = random.Random(seed)
    buckets = {}
    for i, yi in enumerate(y):
        buckets.setdefault(int(yi), []).append(i)
    for v in buckets.values(): rng.shuffle(v)

    folds = []
    for i in range(k):
        test_idx = []
        for cls, idxs in buckets.items():
            size = len(idxs) // k
            r = len(idxs) % k
            start = sum(len(idxs) // k + (1 if t < r else 0) for t in range(i))
            take = size + (1 if i < r else 0)
            test_idx.extend(idxs[start:start+take])
        test_idx = sorted(test_idx)
        train_idx = sorted(list(set(range(len(y))) - set(test_idx)))
        folds.append({"train": train_idx, "test": test_idx})
    return folds

@dataclass
class CVResult:
    fold: int
    best_params: Dict[str, Any]
    macro_f1: float
    acc: float

def grid_dict_product(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))

def nested_cv(
    X_texts: List[str],
    y: np.ndarray,
    outer_k: int,
    inner_k: int,
    estimator_factory: Callable[[Dict[str, Any]], Any],
    param_grid: Dict[str, List[Any]],
    seed: int = 42,
    score_fn: Callable[[np.ndarray, np.ndarray], float] = None,
):
    score_fn = score_fn or (lambda yt, yp: f1_score(yt, yp, average="macro"))  # 自写宏 F1
    outer = stratified_kfold_indices(y, k=outer_k, seed=seed)
    results: List[CVResult] = []

    for oi, fold in enumerate(outer):
        tr_idx, te_idx = fold["train"], fold["test"]
        X_tr = [X_texts[i] for i in tr_idx]; y_tr = y[tr_idx]
        X_te = [X_texts[i] for i in te_idx]; y_te = y[te_idx]

        # ----- 内层：在外层训练集上再分 inner_k 折做网格搜索 -----
        inner = stratified_kfold_indices(y_tr, k=inner_k, seed=seed+oi)
        best_params, best_score = None, -1.0

        for params in grid_dict_product(param_grid):
            scores = []
            for fi, f in enumerate(inner):
                tr2 = [X_tr[j] for j in f["train"]]; ytr2 = y_tr[f["train"]]
                va2 = [X_tr[j] for j in f["test"]];  yva2 = y_tr[f["test"]]
                est = estimator_factory(params)
                est.fit(tr2, ytr2)                 # 只看训练折 → 在里面 fit 词表/向量器/嵌入
                yhat = est.predict(va2)            # 验证折只 transform/predict
                scores.append(score_fn(yva2, yhat))
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score, best_params = mean_score, params

        # ----- 外层评估：用最佳参数在外层训练集重训 → 外层测试集评估 -----
        best_est = estimator_factory(best_params)
        best_est.fit(X_tr, y_tr)
        ypred = best_est.predict(X_te)
        mf1 = score_fn(y_te, ypred)
        acc = accuracy_score(y_te, ypred)
        results.append(CVResult(oi, best_params, float(mf1), float(acc)))
        print(f"[outer {oi}] acc={acc:.4f} macroF1={mf1:.4f} best={best_params}")

    return results
