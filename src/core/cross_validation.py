# cross_validation.py
from typing import List, Dict, Any, Tuple
import random

from core.metrics import f1_score, accuracy_score


def stratified_kfold_indices(y: List[int], k: int, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    random.seed(seed)
    # 按标签分桶
    buckets = {}
    for i, yi in enumerate(y):
        buckets.setdefault(int(yi), []).append(i)
    for b in buckets.values():
        random.shuffle(b)
    # 逐桶切分
    folds = [([], []) for _ in range(k)]  # placeholder train/val
    # 先生成 val 索引集合
    val_splits = [[] for _ in range(k)]
    for b in buckets.values():
        n = len(b)
        size = n // k
        for j in range(k):
            lo = j * size
            hi = n if j == k - 1 else (j + 1) * size
            val_splits[j].extend(b[lo:hi])
    # 构建 (train, val)
    all_idx = set(range(len(y)))
    out = []
    for j in range(k):
        val_idx = sorted(val_splits[j])
        train_idx = sorted(list(all_idx - set(val_idx)))
        out.append((train_idx, val_idx))
    return out


def nested_cv(X: List[str], y: List[int], outer_k: int, inner_k: int, estimator_factory, param_grid: List[Dict[str, Any]], seed: int = 42):
    outer = stratified_kfold_indices(y, outer_k, seed)
    outer_scores = []
    best_params_by_outer = []

    for oi, (tr_idx, te_idx) in enumerate(outer, 1):
        print(f"Outer Fold {oi}/{outer_k}")
        X_tr = [X[i] for i in tr_idx]
        y_tr = [y[i] for i in tr_idx]
        X_te = [X[i] for i in te_idx]
        y_te = [y[i] for i in te_idx]

        # 内层搜索
        inner = stratified_kfold_indices(y_tr, inner_k, seed + oi)
        best_f1 = -1.0
        best_params = None
        print(f"  Testing {len(param_grid)} parameter combinations")
        for pi, p in enumerate(param_grid, 1):
            inner_scores = []
            for ii, (tr2, va2) in enumerate(inner, 1):
                X_tr2 = [X_tr[i] for i in tr2]
                y_tr2 = [y_tr[i] for i in tr2]
                X_va2 = [X_tr[i] for i in va2]
                y_va2 = [y_tr[i] for i in va2]

                est = estimator_factory(p)
                est.fit(X_tr2, y_tr2)
                pred = est.predict(X_va2)
                inner_scores.append(f1_score(y_va2, pred, average="macro"))
            mean_inner = sum(inner_scores) / max(1, len(inner_scores))
            print(f"    [{pi}/{len(param_grid)}] params={p}  inner_f1={mean_inner:.4f}")
            if mean_inner > best_f1:
                best_f1, best_params = mean_inner, p
        print(f"  Best params: {best_params}")

        # 外层评估（用最佳超参重训）
        est = estimator_factory(best_params)
        est.fit(X_tr, y_tr)
        pred = est.predict(X_te)
        f1 = f1_score(y_te, pred, average="macro")
        acc = accuracy_score(y_te, pred)
        outer_scores.append((f1, acc))
        best_params_by_outer.append(best_params)
        print(f"  Fold score: F1={f1:.4f}  Acc={acc:.4f}")

    mean_f1 = sum(s[0] for s in outer_scores) / len(outer_scores)
    std_f1 = (sum((s[0] - mean_f1) ** 2 for s in outer_scores) / len(outer_scores)) ** 0.5
    print("\nNested CV Results:")
    print(f"Best parameters (by outer folds): {best_params_by_outer}")
    print(f"Mean CV F1: {mean_f1:.4f} ± {std_f1:.4f}")

    return {
        "outer_scores": outer_scores,
        "best_params_by_outer": best_params_by_outer,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
    }