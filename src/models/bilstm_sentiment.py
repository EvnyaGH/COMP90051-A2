# bilstm_sentiment.py
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from collections import Counter
import numpy as np


class _TxtDS(Dataset):
    def __init__(self, ids, y, max_len):
        self.ids, self.y, self.max_len = ids, y, max_len
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.ids[i][: self.max_len], self.y[i]


def _pad_batch(batch, pad_id=0):
    xs, ys = zip(*batch)
    maxl = max(len(x) for x in xs)
    out = np.full((len(xs), maxl), pad_id, dtype=np.int64)
    for i, x in enumerate(xs):
        out[i, : len(x)] = x
    return torch.as_tensor(out), torch.as_tensor(ys)


class BiLSTMNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_classes=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hid_dim * 2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        # masked mean pooling
        mask = (x != 0).unsqueeze(-1).float()
        mean = (out * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.proj(self.drop(mean))


class BiLSTMSentiment:
    def __init__(self, **params: Dict[str, Any]):
        self.p = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = None
        self.model = None

    # ---------- vocab ----------
    def _build_vocab(self, texts: List[str], min_freq: int = 2) -> Tuple[Dict[str, int], Dict[int, str]]:
        cnt = Counter()
        for t in texts:
            for w in t.split():
                cnt[w] += 1
        itos = ["<PAD>", "<UNK>"] + [w for w, c in cnt.items() if c >= min_freq]
        stoi = {w: i for i, w in enumerate(itos)}
        return stoi, {i: w for w, i in stoi.items()}

    def _to_ids(self, texts: List[str], stoi: Dict[str, int]) -> List[List[int]]:
        unk = stoi.get("<UNK>", 1)
        return [[stoi.get(w, unk) for w in t.split()] for t in texts]

    # ---------- optional: load GloVe ----------
    def _load_glove(self, path: str, emb_dim: int, stoi: Dict[str, int]) -> torch.Tensor:
        """加载 GloVe 文本文件，按 vocab 对齐；OOV 随机。"""
        vocab_size = len(stoi)
        mat = torch.randn(vocab_size, emb_dim) * 0.1
        mat[0].zero_()  # PAD
        if path is None:
            return mat
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < emb_dim + 1:
                    continue
                w = parts[0]
                if w in stoi:
                    vec = torch.tensor([float(x) for x in parts[1:1 + emb_dim]])
                    mat[stoi[w]] = vec
        return mat

    # ---------- api ----------
    def fit(self, texts: List[str], y: List[int]):
        print(f"[BiLSTM] device={self.device}, cuda={torch.cuda.is_available()}")
        stoi, itos = self._build_vocab(texts)
        self.vocab = stoi
        ids = self._to_ids(texts, stoi)

        ds = _TxtDS(ids, y, self.p.get("max_len", 192))
        dl = DataLoader(ds, batch_size=self.p.get("batch_size", 64), shuffle=True, collate_fn=_pad_batch)

        self.model = BiLSTMNet(
            vocab_size=len(stoi),
            emb_dim=self.p.get("embedding_dim", 100),
            hid_dim=self.p.get("hidden_dim", 128),
            dropout=self.p.get("dropout", 0.3),
        ).to(self.device)

        # 可选：加载 GloVe 并冻结
        emb_path = self.p.get("emb_path", None)
        if emb_path is not None:
            w = self._load_glove(emb_path, self.p.get("embedding_dim", 100), stoi)
            self.model.emb.weight.data.copy_(w)
            if self.p.get("freeze_emb", False):
                self.model.emb.weight.requires_grad = False

        optim = torch.optim.AdamW(self.model.parameters(), lr=self.p.get("lr", 1e-3))
        lossf = nn.CrossEntropyLoss()

        epochs = self.p.get("epochs", 2)
        for ep in range(1, epochs + 1):
            self.model.train()
            total = 0.0
            for step, (xb, yb) in enumerate(dl):
                xb, yb = xb.to(self.device), yb.to(self.device)
                if step == 0:
                    print(f"[BiLSTM] first-batch on {xb.device}")
                optim.zero_grad()
                logits = self.model(xb)
                loss = lossf(logits, yb)
                loss.backward()
                optim.step()
                total += loss.item()
            print(f"[BiLSTM] epoch {ep}/{epochs} loss={total / max(1, len(dl)):.4f}")
        return self

    def predict(self, texts: List[str]):
        self.model.eval()
        ids = self._to_ids(texts, self.vocab)
        ds = _TxtDS(ids, [0] * len(ids), self.p.get("max_len", 192))
        dl = DataLoader(ds, batch_size=self.p.get("batch_size", 64), shuffle=False, collate_fn=_pad_batch)
        outs = []
        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(self.device)
                logits = self.model(xb)
                pred = logits.argmax(-1).cpu().tolist()
                outs.extend(pred)
        return outs


def create_bilstm_factory():
    def factory(params: Dict[str, Any]):
        return BiLSTMSentiment(**params)
    return factory