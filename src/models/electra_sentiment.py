#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELECTRA Fine-tuning Implementation for Sentiment Classification

This module implements ELECTRA fine-tuning for sentiment classification that works
with our cross-validation framework.

Features:
- Uses pre-trained ELECTRA model
- Fine-tunes on sentiment classification task
- Compatible with our CV framework
- Handles text tokenization and preprocessing
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from typing import Dict, Any, List, Optional
import random
from tqdm import tqdm


class ElectraSentimentDataset(Dataset):
    """Dataset class for ELECTRA sentiment classification."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: np.ndarray):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# -------- Config dataclass --------
@dataclass
class ElectraConfig:
    model_name: str = "google/electra-small-discriminator"
    learning_rate: float = 3e-5
    epochs: int = 3
    batch_size: int = 32
    max_len: int = 128
    freeze_bottom_layers: int = 0
    use_amp: bool = torch.cuda.is_available()  # Only use AMP if CUDA is available
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -------- Global tokenizer cache --------
_tokenizer_cache: Dict[str, Any] = {}
_encoding_cache: Dict[str, Dict[str, torch.Tensor]] = {}


def _get_tokenizer(model_name: str):
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[model_name]


def _encode_texts(
    model_name: str, texts: List[str], max_len: int
) -> Dict[str, torch.Tensor]:
    """Use global cache to avoid repeating tokenization"""
    key = f"{model_name}-{max_len}-{len(texts)}"
    if key in _encoding_cache:
        return _encoding_cache[key]
    tokenizer = _get_tokenizer(model_name)
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    )
    _encoding_cache[key] = enc
    return enc


class ElectraSentiment:
    """
    ELECTRA-based sentiment classifier.

    This class provides a consistent interface for our cross-validation framework
    while using ELECTRA for fine-tuning.
    """

    def __init__(self, **params):
        """
        Initialize ELECTRA sentiment classifier.

        Args:
            **params: Hyperparameters including:
                     - learning_rate: Learning rate for fine-tuning
                     - batch_size: Batch size for training
                     - max_length: Maximum sequence length
                     - epochs: Number of training epochs
        """
        # Create base config
        base_config = ElectraConfig()

        # Apply parameters
        config_dict = {**base_config.__dict__, **params}

        # Optimize for fast mode (fewer epochs)
        if config_dict.get("epochs", 3) <= 1:
            # For very fast training, reduce batch size and use smaller model
            if "batch_size" not in params:
                config_dict["batch_size"] = min(config_dict["batch_size"], 16)
            if "max_len" not in params:
                config_dict["max_len"] = min(config_dict["max_len"], 96)

        cfg = ElectraConfig(**config_dict)
        self.cfg = cfg
        self.model = None

    def _freeze_layers(self, model, n_freeze: int):
        if n_freeze <= 0:
            return
        print(f"[ELECTRA] Freezing bottom {n_freeze} layers")
        for layer in model.electra.encoder.layer[:n_freeze]:
            for p in layer.parameters():
                p.requires_grad = False

    def fit(self, texts, labels):
        """
        Fine-tune ELECTRA model on sentiment classification.

        Args:
            texts: List of text samples
            labels: List of binary labels (0 or 1)
        """
        # Convert to lists if needed
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()

        # Load pre-trained model and tokenizer
        cfg = self.cfg
        device = torch.device(cfg.device)
        tokenizer = _get_tokenizer(cfg.model_name)
        enc = _encode_texts(cfg.model_name, texts, cfg.max_len)
        dataset = ElectraSentimentDataset(enc, labels)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=2
        )
        self._freeze_layers(self.model, cfg.freeze_bottom_layers)
        self.model.to(device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
        total_steps = len(dataloader) * cfg.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        # Use new API for GradScaler
        if torch.cuda.is_available() and cfg.use_amp:
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = None
        self.model.train()

        # Training loop
        print(
            f"[ELECTRA] Starting training on {device} (AMP: {cfg.use_amp and torch.cuda.is_available()})"
        )
        for epoch in range(cfg.epochs):
            loop = tqdm(
                dataloader,
                desc=f"[ELECTRA] Epoch {epoch+1}/{cfg.epochs}",
                leave=False,
                disable=False,
            )
            total_loss = 0
            for batch_idx, batch in enumerate(loop):
                try:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad()

                    # Use new API for autocast
                    if torch.cuda.is_available() and cfg.use_amp:
                        with torch.amp.autocast("cuda"):
                            out = self.model(**inputs)
                            loss = out.loss
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        out = self.model(**inputs)
                        loss = out.loss
                        loss.backward()
                        optimizer.step()

                    scheduler.step()
                    total_loss += loss.item()

                    # Update progress bar
                    if batch_idx % 10 == 0:  # Update every 10 batches
                        loop.set_postfix(loss=f"{loss.item():.4f}")

                except Exception as e:
                    print(f"[ELECTRA] Error in batch {batch_idx}: {e}")
                    continue

            avg_loss = total_loss / len(dataloader)
            print(f"[ELECTRA] Epoch {epoch+1} completed - Average loss: {avg_loss:.4f}")
        return self

    @torch.no_grad()
    def predict(self, X_texts: List[str]) -> np.ndarray:
        probs = self.predict_proba(X_texts)
        return probs.argmax(axis=1)

    @torch.no_grad()
    def predict_proba(self, X_texts: List[str]) -> np.ndarray:
        cfg = self.cfg
        device = torch.device(cfg.device)
        enc = _encode_texts(cfg.model_name, X_texts, cfg.max_len)
        dataset = ElectraSentimentDataset(enc, np.zeros(len(X_texts), dtype=np.int64))
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),  # Only use pin_memory if CUDA is available
        )
        self.model.eval()
        all_logits = []
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = self.model(**inputs)
            all_logits.append(out.logits.cpu().numpy())
        logits = np.concatenate(all_logits, axis=0)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.cfg.__dict__

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for k, v in params.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        return self


def create_electra_factory(grid, fast=False):
    """
    Create a factory function for ELECTRA that works with our CV framework.

    Args:
        hyperparams: Dictionary of hyperparameters to test

    Returns:
        factory_function: Function that creates ElectraSentiment instances
    """

    def factory(params):
        p = params.copy()
        if fast:
            p["freeze_bottom_layers"] = 6
            p["max_len"] = min(p.get("max_len", 128), 96)
            p["epochs"] = min(p.get("epochs", 3), 1)
        return ElectraSentiment(**p)

    return factory
