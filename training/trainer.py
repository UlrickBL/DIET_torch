from __future__ import annotations

import os
import pickle
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..config import DIETConfig
from ..data.featurizer import Featurizer
from ..models.diet import DIETModel


def _intent_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def _entity_f1(
    pred_seqs: List[List[int]],
    gold_seqs: List[List[int]],
    id2tag: Dict[int, str],
) -> float:
    def _spans(seq: List[int]) -> set:
        spans = set()
        current: Optional[tuple] = None
        for i, tid in enumerate(seq):
            tag = id2tag.get(tid, "O")
            if tag.startswith("B-"):
                current = (tag[2:], i)
            elif tag.startswith("I-") and current is not None:
                pass   # continue the current span
            else:
                if current is not None:
                    spans.add((current[0], current[1], i))
                    current = None
        if current is not None:
            spans.add((current[0], current[1], len(seq)))
        return spans

    tp = fp = fn = 0
    for pred, gold in zip(pred_seqs, gold_seqs):
        p_spans = _spans(pred)
        g_spans = _spans(gold)
        tp += len(p_spans & g_spans)
        fp += len(p_spans - g_spans)
        fn += len(g_spans - p_spans)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class Trainer:

    def __init__(
        self,
        model: DIETModel,
        config: DIETConfig,
        featurizer: Featurizer,
    ):
        self.model = model
        self.config = config
        self.featurizer = featurizer
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self._build_optimizer()
        self._global_step = 0
        self._best_val_score = -1.0

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _build_optimizer(self) -> None:
        cfg = self.config

        pretrained_params: List[nn.Parameter] = []
        main_params: List[nn.Parameter] = []

        if self.model.qwen_encoder is not None and cfg.pretrained.use_lora:
            pretrained_ids = {
                id(p) for p in self.model.qwen_encoder.trainable_parameters()
            }
        else:
            pretrained_ids = set()

        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            if id(param) in pretrained_ids:
                pretrained_params.append(param)
            else:
                main_params.append(param)

        param_groups = [
            {"params": main_params, "lr": cfg.learning_rate},
        ]
        if pretrained_params:
            param_groups.append(
                {"params": pretrained_params, "lr": cfg.pretrained_lr}
            )

        self.optimizer = AdamW(
            param_groups,
            weight_decay=cfg.weight_decay,
        )

        warmup = cfg.warmup_steps
        total = cfg.num_epochs * 1000  # rough upper bound; updated each epoch

        def _lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

        self.scheduler = LambdaLR(self.optimizer, _lr_lambda)

    def _train_step(self, batch: dict) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        sparse = batch["sparse"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        entity_tags = batch["entity_tags"].to(self.device)
        intent_labels = batch["intent_labels"].to(self.device)
        pretrained = (
            batch["pretrained"].to(self.device)
            if batch.get("pretrained") is not None
            else None
        )

        out = self.model(
            sparse=sparse,
            attention_mask=attention_mask,
            entity_tags=entity_tags,
            intent_labels=intent_labels,
            pretrained=pretrained,
            texts=batch.get("texts"),
            word_spans=batch.get("word_spans"),
        )

        loss = out["total_loss"]
        loss.backward()

        if self.config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

        self.optimizer.step()
        self.scheduler.step()
        self._global_step += 1

        return {
            "loss": loss.item(),
            "entity_loss": out["entity_loss"].item(),
            "intent_loss": out["intent_loss"].item(),
            "mask_loss": out["mask_loss"].item(),
            "intent_acc": _intent_accuracy(out["intent_logits"], intent_labels),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_intent_correct = 0
        total_intent_count = 0
        all_pred_tags: List[List[int]] = []
        all_gold_tags: List[List[int]] = []

        for batch in loader:
            sparse = batch["sparse"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            entity_tags = batch["entity_tags"].to(self.device)
            intent_labels = batch["intent_labels"].to(self.device)
            pretrained = (
                batch["pretrained"].to(self.device)
                if batch.get("pretrained") is not None
                else None
            )

            out = self.model.predict(
                sparse=sparse,
                attention_mask=attention_mask,
                pretrained=pretrained,
                texts=batch.get("texts"),
                word_spans=batch.get("word_spans"),
            )

            pred_ids = torch.tensor(out["intent_ids"], device=self.device)
            total_intent_correct += (pred_ids == intent_labels).long().sum().item()
            total_intent_count += intent_labels.shape[0]
            gold_word_tags = entity_tags[:, 1:]   # (B, L-1)
            for b in range(gold_word_tags.shape[0]):
                mask_b = attention_mask[b, 1:]
                seq_len = int(mask_b.long().sum().item())
                gold = gold_word_tags[b, :seq_len].tolist()
                gold = [g if g >= 0 else 0 for g in gold]
                all_gold_tags.append(gold)
                all_pred_tags.append(out["entity_tag_ids"][b])

        intent_acc = total_intent_correct / max(1, total_intent_count)
        entity_f1 = _entity_f1(all_pred_tags, all_gold_tags, self.featurizer.id2tag)

        return {"intent_acc": intent_acc, "entity_f1": entity_f1}

    def _save_checkpoint(self, name: str, extra: dict = {}) -> None:
        path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "global_step": self._global_step,
            **extra,
        }
        torch.save(payload, path)

        feat_path = os.path.join(self.config.checkpoint_dir, "featurizer.pkl")
        with open(feat_path, "wb") as fh:
            pickle.dump(self.featurizer.state_dict(), fh)

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.scheduler.load_state_dict(payload["scheduler_state"])
        self._global_step = payload.get("global_step", 0)
        print(f"Resumed from {path} (step {self._global_step})")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        if resume_from:
            self.load_checkpoint(resume_from)

        cfg = self.config
        print(
            f"Starting training – {cfg.num_epochs} epochs, "
            f"device={cfg.device}, "
            f"entity_extractor={cfg.entity.extractor}"
        )

        for epoch in range(1, cfg.num_epochs + 1):
            epoch_start = time.time()
            running: Dict[str, float] = {}
            n_steps = 0

            for batch in train_loader:
                metrics = self._train_step(batch)
                for k, v in metrics.items():
                    running[k] = running.get(k, 0.0) + v
                n_steps += 1

                if self._global_step % cfg.log_every == 0:
                    avg = {k: v / n_steps for k, v in running.items()}
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  step {self._global_step:6d} | "
                        + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
                        + f" | lr={lr:.2e}"
                    )

            elapsed = time.time() - epoch_start
            avg_train = {k: v / max(1, n_steps) for k, v in running.items()}
            print(
                f"Epoch {epoch:3d}/{cfg.num_epochs} "
                f"({elapsed:.1f}s) | train "
                + " | ".join(f"{k}={v:.4f}" for k, v in avg_train.items())
            )

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_score = 0.5 * val_metrics["intent_acc"] + 0.5 * val_metrics["entity_f1"]
                print(
                    f"          val   | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                )

                if val_score > self._best_val_score:
                    self._best_val_score = val_score
                    self._save_checkpoint("best", {"epoch": epoch, "val": val_metrics})
                    print(f"          → New best checkpoint saved (score={val_score:.4f})")

            if epoch % 10 == 0:
                self._save_checkpoint(f"epoch_{epoch:03d}", {"epoch": epoch})

        print("Training complete.")
        self._save_checkpoint("final")
