# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:35:35 2025

@author: Ymmmmmm
"""

# experimental_pipeline.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from cross_validation import nested_cv  # 自写嵌套CV
from models_registry import get_factory_and_grid  # 新增文件（见下节）

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to imdb_clean.csv")
    ap.add_argument("--model", required=True, choices=["lr","bilstm","electra"])
    ap.add_argument("--outer_k", type=int, default=10)
    ap.add_argument("--inner_k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(int).to_numpy()

    factory, grid = get_factory_and_grid(args.model)
    results = nested_cv(X, y, outer_k=args.outer_k, inner_k=args.inner_k,
                        estimator_factory=factory, param_grid=grid, seed=args.seed)

    out = Path(args.csv).with_suffix(f".{args.model}.nestedcv.json")
    out.write_text(json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved fold results → {out}")

    macro_f1 = np.mean([r.macro_f1 for r in results])
    acc      = np.mean([r.acc for r in results])
    print(f"\n[Summary] acc={acc:.4f}  macroF1={macro_f1:.4f}")

if __name__ == "__main__":
    main()
