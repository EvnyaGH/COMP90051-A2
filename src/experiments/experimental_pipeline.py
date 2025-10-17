# experimental_pipeline.py
import argparse
import json
import os

import pandas as pd

from models.models_registry import get_factory_and_grid
from core.cross_validation import nested_cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to imdb_clean.csv")
    ap.add_argument("--model", required=True, choices=["lr", "bilstm", "electra"], help="which model to run")
    ap.add_argument("--outer_k", type=int, default=3)
    ap.add_argument("--inner_k", type=int, default=2)
    ap.add_argument("--fast", action="store_true")
    # 可覆盖网格中的关键项
    ap.add_argument("--use_hashing", action="store_true")
    ap.add_argument("--emb_path", default=None)
    ap.add_argument("--freeze_emb", action="store_true")
    ap.add_argument("--freeze_layers", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(int).tolist()

    factory, grid = get_factory_and_grid(args.model, fast=args.fast)

    # 覆盖 grid 中的部分字段（若提供）
    def patched(g):
        g = dict(g)
        if args.use_hashing is True and args.model == "lr":
            g["use_hashing"] = True
        if args.emb_path is not None and args.model == "bilstm":
            g["emb_path"] = args.emb_path
            g.setdefault("freeze_emb", args.freeze_emb)
        if args.freeze_layers is not None and args.model == "electra":
            g["freeze_layers"] = args.freeze_layers
        if args.max_len is not None and args.model in ("bilstm", "electra"):
            g["max_len"] = args.max_len
        return g

    grid = [patched(g) for g in grid]

    print("============================================================")
    print(f"Tuning hyperparameters for: {args.model}")
    print("============================================================")
    print(f"Data length: {len(X)}\nOuter folds: {args.outer_k}, Inner folds: {args.inner_k}")

    results = nested_cv(
        X,
        y,
        outer_k=args.outer_k,
        inner_k=args.inner_k,
        estimator_factory=factory,
        param_grid=grid,
    )

    out = os.path.splitext(os.path.basename(args.csv))[0]
    out = f"{out}.{args.model}.nestedcv.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()