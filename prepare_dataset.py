# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:23:13 2025

@author: Ymmmmmm
"""
# prepare_dataset.py
import os, re, html, unicodedata, json
from pathlib import Path
import pandas as pd
import numpy as np

# === 修改为你的本地路径（写法任选其一） ===
SRC = Path(r"E:\sml\IMDB Dataset.csv")
assert SRC.exists(), f"Source file not found: {SRC}"

OUT_DIR = SRC.parent  # 输出到同一目录，也可改到自己项目的 data 目录

# ---------- 文本归一化 ----------
TAG_RE   = re.compile(r"<[^>]+>")
URL_RE   = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
USER_RE  = re.compile(r"(^|[^A-Za-z0-9_])@([A-Za-z0-9_]{1,15})")
DIGIT_RE = re.compile(r"\d+")

def strip_control_chars(s: str) -> str:
    return "".join(ch for ch in s if ch.isprintable())

def normalize_text(text: str) -> str:
    s = str(text)
    s = html.unescape(s)
    s = s.replace("<br />", " ")
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

# ---------- 分层抽样（保持类平衡） ----------
def stratified_sample(df: pd.DataFrame, label_col: str,
                      n_per_class: int = None, frac: float = None,
                      random_state: int = 42):
    assert (n_per_class is not None) ^ (frac is not None), "Provide exactly one of n_per_class or frac"
    parts = []
    for _, g in df.groupby(label_col, sort=False):
        if n_per_class is not None:
            n = min(n_per_class, len(g))
            parts.append(g.sample(n=n, random_state=random_state))
        else:
            parts.append(g.sample(frac=frac, random_state=random_state))
    out = pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out

def main():
    # 读入并标准列名
    df = pd.read_csv(SRC)
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "review": rename_map[c] = "text"
        if lc == "sentiment": rename_map[c] = "sentiment"
    df = df.rename(columns=rename_map)

    before = len(df)
    # 移除缺失、去重
    df = df.dropna(subset=["text", "sentiment"])
    df = df.drop_duplicates(subset=["text", "sentiment"])
    after = len(df)

    # label 映射
    lab_map = {"positive": 1, "negative": 0, "pos": 1, "neg": 0, 1: 1, 0: 0}
    df["label"] = df["sentiment"].map(lambda x: lab_map.get(str(x).strip().lower(), np.nan)).astype("Int64")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # 文本 normalize
    df["text"] = df["text"].apply(normalize_text)

    # 导出清洗文件
    clean_path = OUT_DIR / "imdb_clean.csv"
    df[["text","label"]].to_csv(clean_path, index=False, encoding="utf-8")

    # 生成一个平衡 10k 子集（正负各一半）方便快速调试
    per_class = min(5000, int(df["label"].value_counts().min()))
    sub = stratified_sample(df[["text","label"]], "label", n_per_class=per_class, random_state=42)
    sub_path = OUT_DIR / "imdb_balanced_10k.csv"
    sub.to_csv(sub_path, index=False, encoding="utf-8")

    print(json.dumps({
        "dropped_rows": before - after,
        "final_rows": int(len(df)),
        "class_balance_full": df["label"].value_counts().to_dict(),
        "balanced_subset_rows": int(len(sub)),
        "class_balance_subset": sub["label"].value_counts().to_dict(),
        "outputs": {"clean": str(clean_path), "balanced_10k": str(sub_path)}
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
