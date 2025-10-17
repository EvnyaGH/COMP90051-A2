#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified runner for COMP90051-A2 experiments.

- Run from project root.
- Imports modules from src/ (core, models, experimental).
- Supports running one model (lr/bilstm/electra) or all three.
- Uses the current package layout:
    src/core/{cross_validation.py, metrics.py}
    src/models/{bilstm_sentiment.py, electra_sentiment.py, logistic_regression.py, models_registry.py}
    src/experimental/{experimental_pipeline.py, hyperparameter_tuning.py, learning_curves.py}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# ---------------------------------------------------------------------
# Ensure we can import package modules `core.*`, `models.*`, `experimental.*`
# ---------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import CV and model registry from the new package layout
from core.cross_validation import nested_cv  # type: ignore
from models.models_registry import get_factory_and_grid  # type: ignore


# -----------------------------
# Utilities
# -----------------------------
def _resolve_csv(data_dir: Optional[Path], csv_path: Optional[Path]) -> Path:
    """Locate imdb_clean.csv from --csv or --data-dir (defaults to ROOT/data)."""
    if csv_path is not None:
        p = csv_path if csv_path.is_absolute() else (ROOT / csv_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        return p
    if data_dir is None:
        data_dir = ROOT / "data"
    p = data_dir / "imdb_clean.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Cannot find {p}. Place imdb_clean.csv under --data-dir "
            f"or pass --csv explicitly."
        )
    return p


def _patch_grid(model: str, grid: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Override some grid fields from CLI flags without changing registry defaults."""
    patched: List[Dict[str, Any]] = []
    for g in grid:
        gg = dict(g)
        if model == "lr" and args.use_hashing:
            gg["use_hashing"] = True
        if model == "bilstm":
            if args.emb_path is not None:
                gg["emb_path"] = str(args.emb_path)
                if args.freeze_emb:
                    gg["freeze_emb"] = True
            if args.max_len is not None:
                gg["max_len"] = int(args.max_len)
        if model == "electra":
            if args.freeze_layers is not None:
                gg["freeze_layers"] = int(args.freeze_layers)
            if args.max_len is not None:
                gg["max_len"] = int(args.max_len)
        patched.append(gg)
    return patched


def _run_one(model: str, X: List[str], y: List[int], args: argparse.Namespace) -> Dict[str, Any]:
    print("============================================================")
    print(f"Tuning hyperparameters for: {model}")
    print("============================================================")
    print(f"Data length: {len(X)}")
    print(f"Outer folds: {args.outer_k}, Inner folds: {args.inner_k}")
    print(f"Fast mode: {args.fast}")

    factory, grid = get_factory_and_grid(model, fast=args.fast)
    grid = _patch_grid(model, grid, args)

    res = nested_cv(
        X,
        y,
        outer_k=args.outer_k,
        inner_k=args.inner_k,
        estimator_factory=factory,
        param_grid=grid,
        seed=args.seed,
    )
    return res


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Run experiments from project root")
    ap.add_argument("--data-dir", type=Path, default=None, help="Directory containing imdb_clean.csv")
    ap.add_argument("--csv", type=Path, default=None, help="Explicit path to imdb_clean.csv")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--model", choices=["lr", "bilstm", "electra", "all"], default="all")
    ap.add_argument("--outer_k", type=int, default=3)
    ap.add_argument("--inner_k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true")

    # Optional overrides
    ap.add_argument("--use_hashing", action="store_true", help="Use HashingVectorizer for LR")
    ap.add_argument("--emb_path", type=Path, default=None, help="Path to GloVe file for BiLSTM")
    ap.add_argument("--freeze_emb", action="store_true", help="Freeze embedding in BiLSTM when GloVe is loaded")
    ap.add_argument("--freeze_layers", type=int, default=None, help="Freeze first N Electra encoder layers")
    ap.add_argument("--max_len", type=int, default=None, help="Override max sequence length for BiLSTM/Electra")

    args = ap.parse_args()

    # STEP 1: Load data
    csv_path = _resolve_csv(args.data_dir, args.csv)
    df = pd.read_csv(csv_path)

    print("============================================================")
    print("STEP 1: Loading and Preparing Data")
    print("============================================================")
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(int).tolist()
    from collections import Counter

    dist = Counter(y)
    print(f"[data] rows={len(df)}, balance={dict(dist)}")
    print("\nSample data:")
    print(df.head()[["text", "label"]])

    # STEP 2: Run models
    models = [args.model] if args.model != "all" else ["lr", "bilstm", "electra"]
    results: Dict[str, Any] = {}

    for m in models:
        results[m] = _run_one(m, X, y, args)

    # Save summary
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "experiment_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
