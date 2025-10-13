# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:05:15 2025

@author: Ymmmmmm
"""

# extract_features.py
# 功能：把 imdb_clean.csv 里的每条文本变成三种向量表示之一，并保存到文件
import argparse, os, json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- 通用：读数据 ----------
def load_texts(csv_path):
    csv_path = Path(csv_path)  # 允许传 Path 或 str
    df = pd.read_csv(csv_path)
    assert "text" in df.columns and "label" in df.columns, "CSV 需要包含 text,label 两列"
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()  # 先留着，后续训练时会用到
    return texts, labels

# ---------- ① TF-IDF n-grams ----------
def do_tfidf(csv_path, out_dir, max_word_features=50000, max_char_features=100000):
    """
    生成词/字 TF-IDF 稀疏矩阵，并保存向量器词典（已处理 numpy.int64 → int）
    输出：
      features/imdb_tfidf.npz
      features/tfidf_word_vocab.json
      features/tfidf_char_vocab.json
    """
    from scipy.sparse import hstack, save_npz
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts, _ = load_texts(csv_path)

    print("[TF-IDF] 拟合词 n-gram 向量器 (1,2)…")
    tfw = TfidfVectorizer(ngram_range=(1,2), min_df=2,
                          max_features=max_word_features, sublinear_tf=True)
    Xw = tfw.fit_transform(texts)

    print("[TF-IDF] 拟合字符 n-gram 向量器 (3,5)…")
    tfc = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2,
                          max_features=max_char_features)
    Xc = tfc.fit_transform(texts)

    X = hstack([Xw, Xc])  # 合并
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(out_dir / "imdb_tfidf.npz", X)

    # —— 修复：将 numpy.int64 转为 int 再写 JSON ——
    word_vocab = {str(k): int(v) for k, v in tfw.vocabulary_.items()}
    char_vocab = {str(k): int(v) for k, v in tfc.vocabulary_.items()}
    (out_dir / "tfidf_word_vocab.json").write_text(
        json.dumps(word_vocab, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "tfidf_char_vocab.json").write_text(
        json.dumps(char_vocab, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[TF-IDF] 完成：{X.shape} 稀疏矩阵 → {out_dir/'imdb_tfidf.npz'}")
    print("          词/字典也已保存，之后可用于在训练集上 fit、在验证/测试上 transform。")

# ---------- ② 静态词向量（训练一个小 Word2Vec，句向量=平均词向量） ----------
def simple_tokenize(text):
    # 最简单的空格分词；你在清洗脚本里已做标准化
    return text.split()

def do_w2v(csv_path, out_dir, vec_size=100, window=5, min_count=2, epochs=5, seed=42):
    """
    在本语料上训练一个小型 Word2Vec，然后对每条样本求“平均词向量”作为句向量。
    输出：
      features/imdb_w2v_avg.npy
      features/w2v.kv
    """
    from gensim.models import Word2Vec

    texts, _ = load_texts(csv_path)
    print("[W2V] 构建语料（分词）…")
    corpus = [simple_tokenize(t) for t in texts]

    print(f"[W2V] 训练 Word2Vec（sg=1, size={vec_size}, window={window}, "
          f"min_count={min_count}, epochs={epochs}）…")
    model = Word2Vec(
        sentences=corpus,
        vector_size=vec_size,
        window=window,
        min_count=min_count,
        sg=1,               # skip-gram
        negative=5,
        workers=os.cpu_count() or 1,
        epochs=epochs,
        seed=seed
    )

    # 句向量 = 句中所有“在词表中的词向量”的平均；没有词则返回全零
    def sent_vec(words):
        vecs = [model.wv[w] for w in words if w in model.wv]
        if len(vecs) == 0:
            return np.zeros(vec_size, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    print("[W2V] 计算每条评论的平均词向量…")
    sent_matrix = np.vstack([sent_vec(ws) for ws in corpus])  # [N, vec_size]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "imdb_w2v_avg.npy", sent_matrix)

    # —— 修复点：gensim 的 save 需要 str 路径（内部用 .endswith 判断后缀） ——
    (out_dir / "w2v.kv").unlink(missing_ok=True)  # 覆盖写时更干净
    model.wv.save(str(out_dir / "w2v.kv"))

    print(f"[W2V] 完成：句向量矩阵 {sent_matrix.shape} → {out_dir/'imdb_w2v_avg.npy'}")
    print(f"[W2V] 词向量 KeyedVectors → {out_dir/'w2v.kv'}")

# ---------- ③ 上下文向量（ELECTRA，句向量=[CLS] 或 mean pool） ----------
def do_electra(csv_path, out_dir, model_name="google/electra-small-discriminator",
               batch_size=16, max_len=256, pool="cls"):
    """
    用 ELECTRA 编码句向量；pool='cls' 或 'mean'
    输出：
      features/imdb_electra_cls.npy（或 mean）
      features/electra_meta.json
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import ElectraTokenizerFast, ElectraModel

    texts, _ = load_texts(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = ElectraTokenizerFast.from_pretrained(model_name)
    enc = ElectraModel.from_pretrained(model_name).to(device).eval()

    class TxtDS(Dataset):
        def __init__(self, xs): self.xs=xs
        def __len__(self): return len(self.xs)
        def __getitem__(self, i): return self.xs[i]

    def collate(batch):
        toks = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return toks

    def pooler(last_hidden, attention_mask, how="cls"):
        if how == "cls":
            return last_hidden[:, 0, :]  # [B, H]
        # mean pooling（只平均有效 token）
        mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    ds = TxtDS(texts)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    feats = []
    with torch.no_grad():
        for step, batch in enumerate(dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = enc(**batch)
            vec = pooler(out.last_hidden_state, batch["attention_mask"], pool)
            feats.append(vec.cpu().numpy().astype(np.float32))
            if step % 50 == 0:
                print(f"[ELECTRA] 已处理 {min(step*batch_size, len(texts))}/{len(texts)}")

    sent_matrix = np.vstack(feats)  # [N, hidden_size]（small 模型是 256 维）
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"imdb_electra_{pool}.npy", sent_matrix)

    meta = {"model": model_name, "pooling": pool, "hidden_size": int(sent_matrix.shape[1])}
    (out_dir / "electra_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ELECTRA] 完成：句向量矩阵 {sent_matrix.shape} → {out_dir/f'imdb_electra_{pool}.npy'}")
    print(f"[ELECTRA] 元信息 → {out_dir/'electra_meta.json'}")

# ---------- CLI ----------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="清洗后的 imdb_clean.csv 路径")
    ap.add_argument("--outdir", default="", help="输出目录（默认与 CSV 同目录下的 features/）")
    ap.add_argument("--repr", required=True, choices=["tfidf","w2v","electra"], help="选择要导出的表示")
    # 可调参数（按需）
    ap.add_argument("--max_word_features", type=int, default=50000)
    ap.add_argument("--max_char_features", type=int, default=100000)
    ap.add_argument("--w2v_dim", type=int, default=100)
    ap.add_argument("--w2v_epochs", type=int, default=5)
    ap.add_argument("--electra_batch", type=int, default=16)
    ap.add_argument("--electra_maxlen", type=int, default=256)
    ap.add_argument("--electra_pool", default="cls", choices=["cls","mean"])
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.outdir) if args.outdir else csv_path.parent / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.repr == "tfidf":
        do_tfidf(csv_path, out_dir, args.max_word_features, args.max_char_features)
    elif args.repr == "w2v":
        do_w2v(csv_path, out_dir, vec_size=args.w2v_dim, epochs=args.w2v_epochs)
    else:
        do_electra(csv_path, out_dir, batch_size=args.electra_batch,
                   max_len=args.electra_maxlen, pool=args.electra_pool)
