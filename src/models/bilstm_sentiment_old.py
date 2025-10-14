#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BiLSTM Sentiment Classifier for IMDB (from cleaned CSV).

Usage:
  python bilstm_sentiment.py --csv /path/to/imdb_clean.csv --w2v /path/to/features/w2v.kv --epochs 8 --device auto

CSV requirements:
  - Must contain columns: 'text' and 'label' (label in {0,1})

Outputs:
  - Saves best model weights to: <csv_dir>/bilstm_best.pt
  - Saves vocab to: <csv_dir>/vocab.json
  - Prints train/val/test metrics (Accuracy, F1)
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import regex as re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# sklearn is only used for a train/val/test split with stratify and shuffling (could be done manually, but this is convenient)
from sklearn.model_selection import train_test_split

# Optional: gensim for loading Word2Vec vectors if provided
try:
    from gensim.models import KeyedVectors
    HAS_GENSIM = True
except Exception:
    HAS_GENSIM = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_TOKEN_RE = re.compile(r"\b\w[\w'-]*\b")

def _tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall(s)


def build_vocab(texts: List[str], min_freq: int = 2, max_size: int = 50000) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int]]:
    # Count tokens
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(_tokenize(t))
    # Special tokens
    specials = ['<pad>', '<unk>']
    # Sort by frequency then alphabetically for stability
    items = sorted([x for x in counter.items() if x[1] >= min_freq], key=lambda x: (-x[1], x[0]))
    if max_size is not None:
        items = items[:max(0, max_size - len(specials))]
    idx2tok = specials + [w for w, c in items]
    tok2idx = {tok: i for i, tok in enumerate(idx2tok)}
    freq = {w: c for w, c in counter.items()}
    return tok2idx, {i: tok for tok, i in tok2idx.items()}, freq


def encode(text: str, tok2idx: Dict[str, int]) -> torch.LongTensor:
    unk = tok2idx.get('<unk>', 1)
    ids = [tok2idx.get(tok, unk) for tok in _tokenize(text)]
    if len(ids) == 0:
        ids = [unk]
    return torch.LongTensor(ids)


class ImdbDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tok2idx: Dict[str, int], max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.tok2idx = tok2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.tok2idx)
        if self.max_len is not None and ids.size(0) > self.max_len:
            ids = ids[: self.max_len]
        label = int(self.labels[idx])
        return ids, label


def collate_fn(batch, pad_idx: int):
    ids_list, labels = zip(*batch)
    lengths = torch.LongTensor([len(x) for x in ids_list])
    padded = pad_sequence(ids_list, batch_first=True, padding_value=pad_idx)
    labels = torch.LongTensor(labels)
    return padded, lengths, labels


class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float, pad_idx: int, bidirectional: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 1),
        )

    def forward(self, x, lengths):
        emb = self.embedding(x)  # [B, T, E]
        # Pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        # Take the final hidden states from both directions
        if self.lstm.bidirectional:
            # h_n shape: [num_layers * 2, B, hidden_dim]
            last_fw = h_n[-2, :, :]  # last layer forward
            last_bw = h_n[-1, :, :]  # last layer backward
            h = torch.cat([last_fw, last_bw], dim=-1)
        else:
            h = h_n[-1, :, :]
        logits = self.fc(h).squeeze(-1)
        return logits


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    acc = (y_pred == y_true).mean()
    # Precision, Recall, F1 (binary)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def load_w2v(path: Path):
    if not HAS_GENSIM:
        print("[warn] gensim not installed; skipping w2v init.")
        return None
    if not path.exists():
        print(f"[warn] w2v file not found at {path}; skipping w2v init.")
        return None
    print(f"[info] Loading Word2Vec keyed vectors from: {path}")
    kv = KeyedVectors.load(str(path), mmap='r')
    return kv


def build_embedding_matrix(tok2idx: Dict[str, int], embed_dim: int, w2v=None, scale: float = 0.1, pad_idx: int = 0):
    matrix = np.random.uniform(-scale, scale, size=(len(tok2idx), embed_dim)).astype(np.float32)
    matrix[pad_idx] = 0.0  # padding is zero
    if w2v is not None:
        found = 0
        for tok, idx in tok2idx.items():
            if tok in ("<pad>", "<unk>"):
                continue
            if tok in w2v:
                vec = w2v[tok]
                if vec.shape[0] == embed_dim:
                    matrix[idx] = vec
                    found += 1
        print(f"[info] W2V init: found {found}/{len(tok2idx)} tokens ({found/len(tok2idx):.1%})")
    return torch.tensor(matrix)


def train_one_epoch(model, loader, optimizer, device, clip: float = 1.0):
    model.train()
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss()
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.float().to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = bce(logits, y)
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    probs_all = []
    labels_all = []
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.float().to(device)
        logits = model(x, lengths)
        loss = bce(logits, y)
        total_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs_all.append(probs)
        labels_all.append(y.detach().cpu().numpy())
    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all).astype(int)
    metrics = binary_metrics(labels_all, probs_all)
    return total_loss / len(loader.dataset), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to imdb_clean.csv")
    parser.add_argument("--w2v", type=str, default="", help="Optional path to features/w2v.kv (Word2Vec KeyedVectors)")
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=50000)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=100, help="Must match w2v dim if you pass --w2v")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    csv_path = Path(args.csv)
    out_dir = csv_path.parent
    vocab_path = out_dir / "vocab.json"
    model_path = out_dir / "bilstm_best.pt"

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[info] Device: {device}")

    # Data
    df = pd.read_csv(csv_path)
    # Heuristic column names
    label_col = "label" if "label" in df.columns else "sentiment"
    text_col = "text" if "text" in df.columns else "review"
    if label_col == "sentiment":
        # map 'positive'/'negative' to 1/0 if needed
        if df[label_col].dtype == object:
            df[label_col] = df[label_col].map({"positive": 1, "negative": 0}).astype(int)
        df = df.rename(columns={"sentiment": "label"})
        label_col = "label"
    if text_col != "text":
        df = df.rename(columns={text_col: "text"})
        text_col = "text"

    # Train/val/test split (80/10/10) with stratify
    X_temp, X_test, y_temp, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=0.1, random_state=args.seed, stratify=df["label"].tolist()
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, random_state=args.seed, stratify=y_temp  # so that 0.9*0.1111 ≈ 0.1
    )

    # Vocab
    tok2idx, idx2tok, freq = build_vocab(X_train, min_freq=args.min_freq, max_size=args.max_size)
    pad_idx = tok2idx["<pad>"]
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"tok2idx": tok2idx, "pad_idx": pad_idx}, f, ensure_ascii=False)
    print(f"[info] Vocab size: {len(tok2idx)}  (saved to {vocab_path})")

    # Datasets & loaders
    ds_train = ImdbDataset(X_train, y_train, tok2idx, max_len=args.max_len)
    ds_val = ImdbDataset(X_val, y_val, tok2idx, max_len=args.max_len)
    ds_test = ImdbDataset(X_test, y_test, tok2idx, max_len=args.max_len)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx))
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx))
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx))

    # Model
    model = BiLSTM(
        vocab_size=len(tok2idx),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=pad_idx,
        bidirectional=True,
    ).to(device)

    # Optional W2V init
    if args.w2v:
        w2v_path = Path(args.w2v)
        kv = load_w2v(w2v_path)
        if kv is not None:
            if kv.vector_size != args.embed_dim:
                print(f"[warn] --embed_dim ({args.embed_dim}) != w2v_dim ({kv.vector_size}); using random init instead.")
            else:
                emb_matrix = build_embedding_matrix(tok2idx, args.embed_dim, w2v=kv, pad_idx=pad_idx)
                with torch.no_grad():
                    model.embedding.weight.data.copy_(emb_matrix)
                print("[info] Embedding layer initialized from Word2Vec.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training with early stopping on val loss
    best_val = math.inf
    patience = 3
    wait = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, dl_train, optimizer, device)
        val_loss, val_metrics = evaluate(model, dl_val, device)
        print(f"[epoch {epoch:02d}] train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['f1']:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), model_path)
            print(f"[info] Saved best model → {model_path}")
        else:
            wait += 1
            if wait >= patience:
                print("[info] Early stopping.")
                break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_metrics = evaluate(model, dl_test, device)
    print(f"[test] loss={test_loss:.4f}  acc={test_metrics['accuracy']:.4f}  f1={test_metrics['f1']:.4f}")
    print("[done]")


if __name__ == "__main__":
    main()
