import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from data.featurizer import Featurizer, tokenize
from data.nlu_parser import NLUExample
from models.diet import DIETModel


def _bio_to_spans(tag_seq: List[str]) -> List[Tuple[int, int, str]]:
    spans = []
    current_label: Optional[str] = None
    start = 0
    for i, tag in enumerate(tag_seq):
        if tag.startswith("B-"):
            if current_label is not None:
                spans.append((start, i, current_label))
            current_label = tag[2:]
            start = i
        elif tag.startswith("I-"):
            label = tag[2:]
            if current_label != label:
                # Broken I-tag (no preceding B) – treat as new span
                if current_label is not None:
                    spans.append((start, i, current_label))
                current_label = label
                start = i
        else:
            if current_label is not None:
                spans.append((start, i, current_label))
            current_label = None
    if current_label is not None:
        spans.append((start, len(tag_seq), current_label))
    return spans


def _spans_from_ids(tag_ids: List[int], id2tag: Dict[int, str]) -> List[Tuple[int, int, str]]:
    return _bio_to_spans([id2tag.get(t, "O") for t in tag_ids])


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


class Evaluator:
    def __init__(
        self,
        model: DIETModel,
        featurizer: Featurizer,
        output_dir: str = "eval/results",
    ):
        self.model = model
        self.featurizer = featurizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _run_inference(
        self, examples: List[NLUExample], device: torch.device
    ) -> List[dict]:
        self.model.eval()
        cfg = self.featurizer.cfg
        results = []

        for ex in examples:
            toks = tokenize(ex.text, cfg.lowercase)
            words = [t[0] for t in toks]
            spans = [(t[1], t[2]) for t in toks]

            if not words:
                continue

            import numpy as np
            sparse_mat = self.featurizer._vectorizer.transform(words).toarray().astype("float32")
            cls_row = np.zeros((1, self.featurizer.sparse_dim), dtype="float32")
            sparse = np.concatenate([cls_row, sparse_mat], axis=0)
            sparse_t = torch.from_numpy(sparse).unsqueeze(0).to(device)
            seq_len = sparse_t.shape[1]
            attention_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)

            with torch.no_grad():
                out = self.model.predict(
                    sparse=sparse_t,
                    attention_mask=attention_mask,
                    pretrained=None,
                    texts=[ex.text],
                    word_spans=[spans],
                )

            intent_id = out["intent_ids"][0]
            intent_probs = out["intent_probs"][0]
            tag_ids = out["entity_tag_ids"][0]

            results.append({
                "text": ex.text,
                "gt_intent": ex.intent,
                "pred_intent": self.featurizer.id2intent[intent_id],
                "intent_confidence": float(intent_probs[intent_id]),
                "intent_probs": intent_probs.cpu().tolist(),
                # gold entity spans from featurizer BIO conversion
                "gt_tag_ids": [
                    self.featurizer.tag2id.get(t, 0)
                    for t in self.featurizer.transform(ex)["entity_tags"][1:]
                ],
                "pred_tag_ids": tag_ids,
                "words": words,
            })
        return results

    def _intent_metrics(self, results: List[dict]) -> dict:
        labels = sorted(self.featurizer.intent2id, key=lambda x: self.featurizer.intent2id[x])
        y_true = [r["gt_intent"] for r in results]
        y_pred = [r["pred_intent"] for r in results]

        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        top_confusion: Dict[str, dict] = {}
        for i, intent in enumerate(labels):
            row = cm[i].copy()
            row[i] = 0  # exclude self
            if row.sum() > 0:
                confused_idx = int(np.argmax(row))
                top_confusion[intent] = {
                    "confused_with": labels[confused_idx],
                    "count": int(row[confused_idx]),
                }
            else:
                top_confusion[intent] = {"confused_with": None, "count": 0}

        summary = {
            "micro": {
                "precision": report["accuracy"],
                "recall": report["accuracy"],
                "f1": report["accuracy"],
            },
            "macro": {k: report["macro avg"][k] for k in ("precision", "recall", "f1-score")},
            "weighted": {k: report["weighted avg"][k] for k in ("precision", "recall", "f1-score")},
        }

        per_intent = {
            intent: {
                "precision": report.get(intent, {}).get("precision", 0.0),
                "recall": report.get(intent, {}).get("recall", 0.0),
                "f1": report.get(intent, {}).get("f1-score", 0.0),
                "support": report.get(intent, {}).get("support", 0),
            }
            for intent in labels
        }

        return {
            "summary": summary,
            "per_intent": per_intent,
            "top_confusion": top_confusion,
        }

    def _entity_metrics(self, results: List[dict]) -> dict:
        id2tag = self.featurizer.id2tag
        label_tp: Dict[str, int] = defaultdict(int)
        label_fp: Dict[str, int] = defaultdict(int)
        label_fn: Dict[str, int] = defaultdict(int)

        for r in results:
            gold_spans = set(_spans_from_ids(r["gt_tag_ids"], id2tag))
            pred_spans = set(_spans_from_ids(r["pred_tag_ids"], id2tag))
            for span in gold_spans & pred_spans:
                label_tp[span[2]] += 1
            for span in pred_spans - gold_spans:
                label_fp[span[2]] += 1
            for span in gold_spans - pred_spans:
                label_fn[span[2]] += 1

        all_labels = sorted(set(list(label_tp) + list(label_fp) + list(label_fn)))

        per_entity: Dict[str, dict] = {}
        for lbl in all_labels:
            p, r, f = _prf(label_tp[lbl], label_fp[lbl], label_fn[lbl])
            per_entity[lbl] = {
                "precision": p, "recall": r, "f1": f,
                "tp": label_tp[lbl], "fp": label_fp[lbl], "fn": label_fn[lbl],
            }

        total_tp = sum(label_tp.values())
        total_fp = sum(label_fp.values())
        total_fn = sum(label_fn.values())
        micro_p, micro_r, micro_f = _prf(total_tp, total_fp, total_fn)

        supports = {lbl: label_tp[lbl] + label_fn[lbl] for lbl in all_labels}
        total_support = sum(supports.values()) or 1

        def _weighted(key):
            return sum(
                per_entity[lbl][key] * supports[lbl] for lbl in all_labels
            ) / total_support

        macro_p = np.mean([per_entity[l]["precision"] for l in all_labels]) if all_labels else 0.0
        macro_r = np.mean([per_entity[l]["recall"] for l in all_labels]) if all_labels else 0.0
        macro_f = np.mean([per_entity[l]["f1"] for l in all_labels]) if all_labels else 0.0

        summary = {
            "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f},
            "macro": {"precision": float(macro_p), "recall": float(macro_r), "f1": float(macro_f)},
            "weighted": {
                "precision": _weighted("precision"),
                "recall": _weighted("recall"),
                "f1": _weighted("f1"),
            },
        }

        return {"summary": summary, "per_entity": per_entity}

    def _intent_errors(self, results: List[dict]) -> List[dict]:
        errors = []
        for r in results:
            if r["gt_intent"] != r["pred_intent"]:
                errors.append({
                    "text": r["text"],
                    "ground_truth": r["gt_intent"],
                    "predicted": r["pred_intent"],
                    "confidence": r["intent_confidence"],
                    "all_scores": {
                        self.featurizer.id2intent[i]: round(s, 4)
                        for i, s in enumerate(r["intent_probs"])
                    },
                })
        return errors

    def _entity_errors(self, results: List[dict]) -> List[dict]:
        id2tag = self.featurizer.id2tag
        errors = []
        for r in results:
            gold_spans = _spans_from_ids(r["gt_tag_ids"], id2tag)
            pred_spans = _spans_from_ids(r["pred_tag_ids"], id2tag)
            gold_set = set(gold_spans)
            pred_set = set(pred_spans)
            missed = gold_set - pred_set
            spurious = pred_set - gold_set
            if missed or spurious:
                words = r["words"]
                errors.append({
                    "text": r["text"],
                    "missed": [
                        {"tokens": words[s:e], "label": lbl}
                        for s, e, lbl in missed
                    ],
                    "spurious": [
                        {"tokens": words[s:e], "label": lbl}
                        for s, e, lbl in spurious
                    ],
                })
        return errors

    def evaluate(
        self,
        examples: List[NLUExample],
        device: torch.device,
        prefix: str = "eval",
    ) -> dict:
        results = self._run_inference(examples, device)

        intent_metrics = self._intent_metrics(results)
        entity_metrics = self._entity_metrics(results)
        intent_errors = self._intent_errors(results)
        entity_errors = self._entity_errors(results)

        report = {
            "intents": intent_metrics,
            "entities": entity_metrics,
        }

        print("\n=== INTENT CLASSIFICATION ===")
        for avg, vals in intent_metrics["summary"].items():
            f1 = vals.get("f1") or vals.get("f1-score", 0)
            p = vals.get("precision", 0)
            r = vals.get("recall", 0)
            print(f"  {avg:<10}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
        print("\n  Per intent:")
        for intent, m in intent_metrics["per_intent"].items():
            confused = intent_metrics["top_confusion"][intent]
            confused_str = (
                f"  → confuses with '{confused['confused_with']}' ({confused['count']}x)"
                if confused["confused_with"] else ""
            )
            print(f"    {intent:<25}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  sup={m['support']}{confused_str}")

        print("\n=== ENTITY EXTRACTION ===")
        for avg, vals in entity_metrics["summary"].items():
            print(f"  {avg:<10}  P={vals['precision']:.3f}  R={vals['recall']:.3f}  F1={vals['f1']:.3f}")
        print("\n  Per entity:")
        for lbl, m in entity_metrics["per_entity"].items():
            print(f"    {lbl:<30}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  tp={m['tp']} fp={m['fp']} fn={m['fn']}")

        print(f"\n  Intent errors: {len(intent_errors)} / {len(results)}")
        print(f"  Entity errors: {len(entity_errors)} / {len(results)}")

        def _dump(obj, name):
            path = os.path.join(self.output_dir, f"{prefix}_{name}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, indent=2, ensure_ascii=False)
            print(f"  Wrote {path}")

        _dump(report, "report")
        _dump(intent_errors, "intent_errors")
        _dump(entity_errors, "entity_errors")

        return report
