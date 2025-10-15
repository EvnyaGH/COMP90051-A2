# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:36:24 2025

@author: Ymmmmmm
"""

# models_registry.py
from typing import Dict, List, Any, Tuple, Callable
from logistic_regression import create_lr_factory
from bilstm_sentiment import create_bilstm_factory
from electra_sentiment import create_electra_factory

def get_factory_and_grid(name: str) -> Tuple[Callable[[Dict[str, Any]], Any], Dict[str, List[Any]]]:
    name = name.lower()
    if name == "lr":
        grid = {
            "word_ngram": [(1,2)],
            "char_ngram": [(3,5)],
            "max_word_features": [30000, 50000],
            "max_char_features": [50000, 100000],
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l2"],
        }
        return create_lr_factory({}), grid
    if name == "bilstm":
        grid = {
            "embedding_dim": [100, 200],
            "hidden_dim": [128, 256],
            "num_layers": [1, 2],
            "dropout": [0.3, 0.5],
            "epochs": [4, 6],
            "batch_size": [64],
            "min_freq": [2],
            "max_len": [256],
        }
        return create_bilstm_factory({}), grid
    if name == "electra":
        grid = {
            "lr": [5e-5, 1e-4],
            "epochs": [2, 3],
            "batch_size": [16],
            "max_len": [256],
            "warmup": [0.0, 0.1],
            "pretrained": ["google/electra-small-discriminator"],
        }
        return create_electra_factory({}), grid
    raise ValueError(f"Unknown model: {name}")
