# models_registry.py
from typing import Tuple, Dict, Any

from .logistic_regression import create_lr_factory
from .bilstm_sentiment import create_bilstm_factory
from .electra_sentiment import create_electra_factory


def get_factory_and_grid(model: str, fast: bool = True) -> Tuple:
    """
    返回 (factory, param_grid)。factory: params(dict) -> estimator
    param_grid: List[dict]
    """
    model = model.lower()

    if model in {"lr", "logreg", "logistic"}:
        factory = create_lr_factory()
        if fast:
            grid = [
                {
                    "C": c,
                    "solver": "liblinear",
                    "max_iter": 500,
                    "tfidf_max_features": 20000,
                    "tfidf_ngram": (1, 1),
                    # 新增：可选哈希向量器（零拟合，提速）
                    "use_hashing": uh,
                    # HashingVectorizer 专用参数（无状态，不会造成泄漏）
                    "n_features": 2 ** 20,
                }
                for c in (0.1, 1.0, 10.0)
                for uh in (False,)
            ]
        else:
            grid = [
                {
                    "C": c,
                    "solver": "liblinear",
                    "max_iter": 1000,
                    "tfidf_max_features": mf,
                    "tfidf_ngram": (1, 2),
                    "use_hashing": uh,
                    "n_features": 2 ** 20,
                }
                for c in (0.1, 1.0, 10.0)
                for mf in (20000, 40000)
                for uh in (False,)
            ]
        return factory, grid

    if model in {"bilstm", "rnn"}:
        factory = create_bilstm_factory()
        if fast:
            grid = [
                {
                    "embedding_dim": 100,
                    "hidden_dim": hd,
                    "dropout": 0.3,
                    "max_len": 192,
                    "batch_size": 64,
                    "epochs": 2,
                    "lr": 1e-3,
                    # 新增：预训练 GloVe 选项（为 None 表示随机初始化）
                    "emb_path": None,
                    "freeze_emb": False,
                }
                for hd in (64, 128, 256)
            ]
        else:
            grid = [
                {
                    "embedding_dim": 100,
                    "hidden_dim": hd,
                    "dropout": 0.3,
                    "max_len": 256,
                    "batch_size": 64,
                    "epochs": ep,
                    "lr": 1e-3,
                    "emb_path": None,
                    "freeze_emb": fe,
                }
                for hd in (128, 256)
                for ep in (3, 5)
                for fe in (False, True)
            ]
        return factory, grid

    if model in {"electra"}:
        factory = create_electra_factory()
        if fast:
            grid = [
                {
                    "model_name": "google/electra-small-discriminator",
                    "epochs": 2,
                    "batch_size": 16,
                    "lr": 2e-5,
                    "max_len": 192,
                    # 新增：冻结底层层数（提速 + 稳定）
                    "freeze_layers": 6,
                    # 新增：是否缓存 tokenization 结果（提速）
                    "cache_tokenization": True,
                }
            ]
        else:
            grid = [
                {
                    "model_name": "google/electra-small-discriminator",
                    "epochs": ep,
                    "batch_size": 16,
                    "lr": 2e-5,
                    "max_len": 256,
                    "freeze_layers": frz,
                    "cache_tokenization": True,
                }
                for ep in (3,)
                for frz in (0, 4, 6, 8)
            ]
        return factory, grid

    raise ValueError(f"Unknown model: {model}")