# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:38:19 2025

@author: Ymmmmmm
"""

# bilstm_sentiment.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import regex as re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_TOKEN_RE = re.compile(r"\b\w[\w'-]*\b")
def _tokenize(s: str): return _TOKEN_RE.findall(s)

def _build_vocab(train_texts: List[str], min_freq: int = 2, max_size: Optional[int] = 50000):
    from collections import Counter
    cnt = Counter(); [cnt.update(_tokenize(t)) for t in train_texts]
    vocab = {"<PAD>":0,"<UNK>":1}
    items = [(w,f) for w,f in cnt.items() if f>=min_freq]
    items.sort(key=lambda x:(-x[1],x[0]))
    if max_size: items = items[:max(0, max_size-2)]
    for w,_ in items: vocab[w]=len(vocab)
    return vocab

def _texts_to_ids(texts: List[str], vocab: Dict[str,int], max_len: int):
    UNK=1; out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i,t in enumerate(texts):
        ids=[vocab.get(tok,UNK) for tok in _tokenize(t)[:max_len]]
        out[i,:len(ids)] = ids
    return out

class _SeqDS(Dataset):
    def __init__(self, ids: np.ndarray, labels: np.ndarray):
        self.ids=torch.tensor(ids,dtype=torch.long); self.labels=torch.tensor(labels,dtype=torch.long)
    def __len__(self): return self.ids.size(0)
    def __getitem__(self,i): return self.ids[i], self.labels[i]

class _BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm= nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers>1 else 0.0)
        self.drop= nn.Dropout(dropout)
        self.fc  = nn.Linear(hidden_dim*2, 2)
    def forward(self,x):
        e=self.emb(x); o,_=self.lstm(e); h=o[:,-1,:]; h=self.drop(h); return self.fc(h)

@dataclass
class _Params:
    max_len:int=256; embedding_dim:int=100; hidden_dim:int=128
    num_layers:int=1; dropout:float=0.3; lr:float=1e-3
    batch_size:int=64; epochs:int=6; min_freq:int=2; max_vocab_size:int=50000
    device:str="cuda" if torch.cuda.is_available() else "cpu"

class BiLSTMSentiment:
    def __init__(self, **kwargs):
        self.p=_Params(**{**_Params().__dict__, **kwargs})
        self.vocab=None; self.model=None

    def fit(self, X_texts: List[str], y: np.ndarray):
        p=self.p
        self.vocab=_build_vocab(X_texts, min_freq=p.min_freq, max_size=p.max_vocab_size)
        tr_ids=_texts_to_ids(X_texts, self.vocab, p.max_len)
        ds=_SeqDS(tr_ids,y); dl=DataLoader(ds,batch_size=p.batch_size,shuffle=True)
        self.model=_BiLSTM(len(self.vocab),p.embedding_dim,p.hidden_dim,p.num_layers,p.dropout).to(p.device)
        opt=torch.optim.AdamW(self.model.parameters(), lr=p.lr)
        loss_fn=nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(p.epochs):
            for xb,yb in dl:
                xb,yb=xb.to(p.device), yb.to(p.device)
                opt.zero_grad(); logits=self.model(xb); loss=loss_fn(logits,yb)
                loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(),1.0); opt.step()
        return self

    @torch.no_grad()
    def _predict_logits(self, X_texts: List[str]):
        p=self.p; self.model.eval()
        ids=_texts_to_ids(X_texts, self.vocab, p.max_len)
        dl=DataLoader(_SeqDS(ids, np.zeros(len(ids),dtype=np.int64)), batch_size=p.batch_size, shuffle=False)
        out=[]
        for xb,_ in dl:
            out.append(self.model(xb.to(p.device)).cpu().numpy())
        return np.concatenate(out,axis=0)

    def predict(self, X_texts: List[str]) -> np.ndarray:
        return self._predict_logits(X_texts).argmax(axis=1)

def create_bilstm_factory(defaults: Dict[str, Any]):
    def factory(params: Dict[str, Any]):
        cfg={**defaults, **params}
        return BiLSTMSentiment(**cfg)
    return factory
