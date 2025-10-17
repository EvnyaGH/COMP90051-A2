# electra_sentiment.py
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup



class TxtDS(Dataset):
    def __init__(self, texts: List[str], labels: List[int] = None):
        self.texts = texts
        self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        item = {"text": self.texts[i]}
        if self.labels is not None:
            item["label"] = int(self.labels[i])
        return item

def _collate_batch(batch):
    # batch: List[{"text": str, "label": int}]
    texts = [str(x["text"]) for x in batch]
    labels = [int(x["label"]) for x in batch] if "label" in batch[0] else None
    return {"text": texts, "label": labels}


class ElectraSentiment:
    def __init__(self, **params: Dict[str, Any]):
        self.p = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.p.get("model_name", "google/electra-small-discriminator"))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.p.get("model_name", "google/electra-small-discriminator"),
            num_labels=2,
        ).to(self.device)
        # 冻结底层若干层（可选）
        self._freeze_layers(self.p.get("freeze_layers", 0))
        # 简易 tokenization 缓存（进程内）
        self._enc_cache = {} if self.p.get("cache_tokenization", True) else None

    def _freeze_layers(self, n_layers: int):
        if n_layers <= 0:
            return
        for name, p in self.model.named_parameters():
            # 适配 ELECTRA 小模型结构名
            if name.startswith("electra.encoder.layer."):
                try:
                    layer_id = int(name.split(".")[3])
                except Exception:
                    layer_id = 999
                if layer_id < n_layers:
                    p.requires_grad = False

    def _encode_batch(self, texts: List[str]):
        texts = [str(t) for t in texts]  # 强制转为字符串，作为缓存 key 安全
        if self._enc_cache is None:
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.p.get("max_len", 192),
                return_tensors="pt",
            )
        batch_inputs = {"input_ids": [], "attention_mask": []}
        miss = [t for t in texts if t not in self._enc_cache]
        if miss:
            enc = self.tokenizer(
                miss, padding=True, truncation=True,
                max_length=self.p.get("max_len", 192), return_tensors="pt"
            )
            for i, t in enumerate(miss):
                self._enc_cache[t] = {k: v[i].cpu() for k, v in enc.items()}
        for t in texts:
            obj = self._enc_cache[t]
            batch_inputs["input_ids"].append(obj["input_ids"])
            batch_inputs["attention_mask"].append(obj["attention_mask"])
        batch_inputs = {k: torch.stack(v, 0) for k, v in batch_inputs.items()}
        return batch_inputs

    def fit(self, texts: List[str], y: List[int]):
        print(f"[ELECTRA] device={self.device}, cuda={torch.cuda.is_available()}, freeze_layers={self.p.get('freeze_layers',0)}")
        ds = TxtDS(texts, y)
        dl = DataLoader(
            ds,
            batch_size=self.p.get("batch_size", 16),
            shuffle=True,
            collate_fn=_collate_batch,
        )

        optim = torch.optim.AdamW(self.model.parameters(), lr=self.p.get("lr", 2e-5))

        steps_per_epoch = max(1, len(dl))
        total_steps = steps_per_epoch * self.p.get("epochs", 2)
        sched = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        lossf = torch.nn.CrossEntropyLoss()

        self.model.train()
        for ep in range(1, self.p.get("epochs", 2) + 1):
            total = 0.0
            for step, batch in enumerate(dl):
                texts_b = batch["text"]          # List[str]
                labels_b = batch["label"]        # List[int]
                enc = self._encode_batch(texts_b)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                labels = torch.as_tensor(labels_b, device=self.device)

                optim.zero_grad()
                out = self.model(**enc, labels=labels)
                loss = out.loss
                loss.backward()
                optim.step()
                sched.step()
                total += loss.item()
            print(f"[ELECTRA] epoch {ep}/{self.p.get('epochs',2)} loss={total / max(1, len(dl)):.4f}")
        return self

    def predict(self, texts: List[str]):
        self.model.eval()
        outs = []
        bs = self.p.get("batch_size", 16)
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self._encode_batch(texts[i : i + bs])
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                pred = logits.argmax(-1).cpu().tolist()
                outs.extend(pred)
        return outs


def create_electra_factory():
    def factory(params: Dict[str, Any]):
        return ElectraSentiment(**params)
    return factory