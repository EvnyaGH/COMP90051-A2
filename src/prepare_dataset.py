#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare IMDb dataset (Kaggle "IMDB Dataset.csv"):
- Clean text (HTML/url/email/user/digits -> placeholders, lowercase/normalize)
- Map sentiment to {neg:0,pos:1}
- Save to <outdir>/imdb_clean.csv and a balanced 10k subset (optional)

This module provides functions to prepare the dataset programmatically.
"""
from __future__ import annotations
import html, json, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
USER_RE = re.compile(r"(^|[^A-Za-z0-9_])@([A-Za-z0-9_]{1,15})")
DIGIT_RE = re.compile(r"\d+")


def strip_control_chars(s: str) -> str:
    return "".join(ch for ch in s if ch.isprintable())


def normalize_text(text: str) -> str:
    s = str(text)
    s = html.unescape(s).replace("<br />", " ")
    s = TAG_RE.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    s = URL_RE.sub(" <URL> ", s)
    s = EMAIL_RE.sub(" <EMAIL> ", s)
    s = USER_RE.sub(r"\1<USER>", s)
    s = DIGIT_RE.sub("0", s)
    s = s.lower()
    s = strip_control_chars(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    n_per_class: int | None = None,
    frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    assert (n_per_class is not None) ^ (frac is not None), "choose n_per_class OR frac"
    parts = []
    for _, g in df.groupby(label_col, sort=False):
        if n_per_class is not None:
            parts.append(
                g.sample(n=min(n_per_class, len(g)), random_state=random_state)
            )
        else:
            parts.append(g.sample(frac=frac, random_state=random_state))
    return (
        pd.concat(parts)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def prepare_dataset(
    src_path: str | Path,
    outdir: str | Path = "data",
    create_10k_subset: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Prepare IMDb dataset from raw CSV file.

    Args:
        src_path: Path to raw IMDB Dataset.csv file
        outdir: Output directory for processed files
        create_10k_subset: Whether to create balanced 10k subset
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with metadata about the processing
    """
    src = Path(src_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "review":
            rename_map[c] = "text"
        if lc == "sentiment":
            rename_map[c] = "sentiment"
    df = df.rename(columns=rename_map)

    before = len(df)
    df = df.dropna(subset=["text", "sentiment"]).drop_duplicates(
        subset=["text", "sentiment"]
    )
    after = len(df)

    lab_map = {"positive": 1, "negative": 0, "pos": 1, "neg": 0, 1: 1, 0: 0}
    df["label"] = (
        df["sentiment"]
        .map(lambda x: lab_map.get(str(x).strip().lower(), np.nan))
        .astype("Int64")
    )
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    df["text"] = df["text"].map(normalize_text)

    clean_path = outdir / "imdb_clean.csv"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label"]].to_csv(clean_path, index=False, encoding="utf-8")

    sub_info = {}
    if create_10k_subset:
        per_class = min(5000, int(df["label"].value_counts().min()))
        sub = stratified_sample(
            df[["text", "label"]],
            "label",
            n_per_class=per_class,
            random_state=random_state,
        )
        sub_path = outdir / "imdb_balanced_10k.csv"
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(sub_path, index=False, encoding="utf-8")
        sub_info = {"balanced_10k": str(sub_path)}

    meta = {
        "src": str(src),
        "out_clean": str(clean_path),
        "dropped_rows": before - after,
        "final_rows": int(len(df)),
        "class_balance_full": df["label"].value_counts().to_dict(),
        **sub_info,
    }

    return meta


def main():
    """CLI interface for backward compatibility."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", required=True, help="Path to Kaggle IMDB CSV (review,sentiment)"
    )
    parser.add_argument("--outdir", default="data", help="Output root directory")
    parser.add_argument(
        "--mk10k", action="store_true", help="Also write balanced 10k subset"
    )

    args = parser.parse_args()

    meta = prepare_dataset(
        src_path=args.src, outdir=args.outdir, create_10k_subset=args.mk10k
    )

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
