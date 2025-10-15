#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BiLSTM sentiment classifier wired for nested CV without leakage.

- Builds vocabulary on the TRAIN SPLIT ONLY (per-fold).
- Encodes texts to token-id sequences with fixed max_len.
- Initializes embeddings from GloVe (optional) or random; or trains
  Word2Vec on TRAIN texts (optional flag).
- Provides sklearn-like API: fit / predict / predict_proba / get_params / set_params
  so HyperparameterTuner can consume it via a factory.

Dependencies: torch, gensim (optional if use_w2v=True), numpy, regex
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import regex as re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---------------- Tokenization & Vocab ----------------

_TOKEN_RE = re.compile(r"\b\w[\w'-]*\b")


def _tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall(s)


def _build_vocab(
    train_texts: List[str], min_freq: int = 1, max_size: Optional[int] = None
) -> Dict[str, int]:
    from collections import Counter

    cnt = Counter()
    for t in train_texts:
        cnt.update(_tokenize(t))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    items = [(w, f) for w, f in cnt.items() if f >= min_freq]
    # sort by frequency desc, then alphabetically for determinism
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_size is not None:
        items = items[: max(0, max_size - 2)]
    for w, _ in items:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def _texts_to_ids(texts: List[str], vocab: Dict[str, int], max_len: int) -> np.ndarray:
    PAD, UNK = 0, 1
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(tok, UNK) for tok in _tokenize(t)[:max_len]]
        out[i, : len(ids)] = ids
    return out


def _load_glove(glove_path: str, dim: int, vocab: Dict[str, int]) -> np.ndarray:
    rng = np.random.default_rng(42)
    emb = rng.uniform(-0.05, 0.05, size=(len(vocab), dim)).astype(np.float32)
    with open(glove_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split()
            if not parts:
                continue
            w, vec = parts[0], parts[1:]
            if len(vec) != dim:
                continue
            if w in vocab:
                emb[vocab[w]] = np.asarray(vec, dtype=np.float32)
    return emb


def _train_w2v(train_texts: List[str], dim: int, vocab: Dict[str, int]) -> np.ndarray:
    from gensim.models import Word2Vec

    toks = [_tokenize(t) for t in train_texts]
    w2v = Word2Vec(sentences=toks, vector_size=dim, min_count=1, workers=4, seed=42)
    rng = np.random.default_rng(42)
    emb = rng.uniform(-0.05, 0.05, size=(len(vocab), dim)).astype(np.float32)
    for w, i in vocab.items():
        if w in ("<PAD>", "<UNK>"):
            continue
        if w in w2v.wv:
            emb[i] = w2v.wv[w]
    return emb


# ---------------- Dataset ----------------


class SeqDataset(Dataset):
    def __init__(self, ids: np.ndarray, labels: np.ndarray):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.ids.size(0)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


# ---------------- Model ----------------


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        embeddings: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.pad_idx = 0  # Define padding index
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad_idx)
        if embeddings is not None:
            self.emb.weight.data.copy_(torch.from_numpy(embeddings))
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        mask = (x != self.pad_idx).unsqueeze(-1)
        e = self.emb(x)
        o, _ = self.lstm(e)
        o = o * mask  # zero-out PAD positions
        sum_h = o.sum(dim=1)  # [B, 2H]
        len_h = mask.sum(dim=1).clamp(min=1)  # [B, 1]
        h = sum_h / len_h  # masked mean
        h = self.dropout(h)
        logits = self.fc(h)
        return logits


# ---------------- Estimator (sklearn-like) ----------------


@dataclass
class _Params:
    max_len: int = 256
    embedding_dim: int = 100
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 6
    glove_path: Optional[str] = None
    use_w2v: bool = True
    min_freq: int = 1
    max_vocab_size: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BiLSTMSentiment:
    """Estimator wrapper for nested CV."""

    def __init__(self, **kwargs):
        self.params = _Params(**{**_Params().__dict__, **kwargs})
        self.vocab: Optional[Dict[str, int]] = None
        self.model: Optional[BiLSTMClassifier] = None

    # --- API ---
    def fit(self, X_texts: List[str], y: np.ndarray):
        p = self.params
        # Build vocab on TRAIN ONLY
        self.vocab = _build_vocab(
            X_texts, min_freq=p.min_freq, max_size=p.max_vocab_size
        )
        # Encode
        train_ids = _texts_to_ids(X_texts, self.vocab, p.max_len)
        # Embeddings
        if p.glove_path:
            emb = _load_glove(p.glove_path, p.embedding_dim, self.vocab)
        elif p.use_w2v:
            emb = _train_w2v(X_texts, p.embedding_dim, self.vocab)
        else:
            rng = np.random.default_rng(42)
            emb = rng.uniform(
                -0.05, 0.05, size=(len(self.vocab), p.embedding_dim)
            ).astype(np.float32)

        # Model
        self.model = BiLSTMClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=p.embedding_dim,
            hidden_dim=p.hidden_dim,
            num_layers=p.num_layers,
            dropout=p.dropout,
            embeddings=emb,
        ).to(p.device)

        ds = SeqDataset(train_ids, y)
        dl = DataLoader(ds, batch_size=p.batch_size, shuffle=True)
        opt = optim.AdamW(self.model.parameters(), lr=p.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(p.epochs):
            for xb, yb in dl:
                xb = xb.to(p.device)
                yb = yb.to(p.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
        return self

    @torch.no_grad()
    def _predict_logits(self, X_texts: List[str]) -> np.ndarray:
        assert self.model is not None and self.vocab is not None
        p = self.params
        self.model.eval()
        ids = _texts_to_ids(X_texts, self.vocab, p.max_len)
        dl = DataLoader(
            SeqDataset(ids, np.zeros(len(ids), dtype=np.int64)), batch_size=p.batch_size
        )
        all_logits = []
        for xb, _ in dl:
            logits = self.model(xb.to(p.device))
            all_logits.append(logits.cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    def predict(self, X_texts: List[str]) -> np.ndarray:
        logits = self._predict_logits(X_texts)
        return logits.argmax(axis=1)

    def predict_proba(self, X_texts: List[str]) -> np.ndarray:
        logits = self._predict_logits(X_texts)
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exps / exps.sum(axis=1, keepdims=True)
        return probs

    # sklearn-like
    def get_params(self, deep=True) -> dict:
        return {k: getattr(self.params, k) for k in self.params.__dict__.keys()}

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self.params, k):
                setattr(self.params, k, v)
        return self


# ---------------- Factory ----------------


def create_bilstm_factory(hyperparams: dict):
    """
    Returns a callable that, given a hyperparam dict from the tuner,
    instantiates a BiLSTMSentiment with those params.
    """

    def factory(param_choice: dict):
        return BiLSTMSentiment(**param_choice)

    return factory
