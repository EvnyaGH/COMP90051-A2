# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:40:03 2025

@author: Ymmmmmm
"""

# electra_sentiment.py
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification, AdamW, get_linear_schedule_with_warmup

class _TxtDS(Dataset):
    def __init__(self, texts, labels=None):
        self.texts=texts; self.labels=labels
    def __len__(self): return len(self.texts)
    def __getitem__(self,i): 
        if self.labels is None: return self.texts[i], -1
        return self.texts[i], int(self.labels[i])

class ElectraSentiment:
    def __init__(self, pretrained="google/electra-small-discriminator",
                 lr=5e-5, epochs=3, batch_size=16, max_len=256, warmup=0.0,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.pretrained=pretrained; self.lr=lr; self.epochs=epochs
        self.batch_size=batch_size; self.max_len=max_len; self.warmup=warmup
        self.device=torch.device(device)
        self.tok=None; self.model=None

    def _collate(self, batch):
        texts, labels = zip(*batch)
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        if labels[0] != -1:
            enc["labels"]=torch.tensor(labels, dtype=torch.long)
        return enc

    def fit(self, X_texts: List[str], y: np.ndarray):
        self.tok = ElectraTokenizerFast.from_pretrained(self.pretrained)
        self.model = ElectraForSequenceClassification.from_pretrained(self.pretrained, num_labels=2).to(self.device)
        ds=_TxtDS(X_texts, y); dl=DataLoader(ds, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate)
        steps=self.epochs*len(dl)
        opt=AdamW(self.model.parameters(), lr=self.lr)
        sch=get_linear_schedule_with_warmup(opt, num_warmup_steps=int(self.warmup*steps), num_training_steps=steps)
        self.model.train()
        for _ in range(self.epochs):
            for batch in dl:
                batch={k:v.to(self.device) for k,v in batch.items()}
                loss=self.model(**batch).loss
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                opt.step(); sch.step()
        return self

    @torch.no_grad()
    def predict(self, X_texts: List[str]) -> np.ndarray:
        ds=_TxtDS(X_texts, None)
        dl=DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate)
        self.model.eval()
        preds=[]
        for batch in dl:
            batch={k:v.to(self.device) for k,v in batch.items() if k!="labels"}
            logits=self.model(**batch).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        return np.array(preds, dtype=np.int64)

def create_electra_factory(defaults: Dict[str, Any]):
    def factory(params: Dict[str, Any]):
        cfg={**defaults, **params}
        return ElectraSentiment(**cfg)
    return factory
