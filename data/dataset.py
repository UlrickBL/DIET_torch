from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from config import DIETConfig
from data.featurizer import Featurizer
from data.nlu_parser import NLUExample

class DIETDataset(Dataset):

    def __init__(
        self,
        examples: List[NLUExample],
        featurizer: Featurizer,
        pretrained_embeddings: Optional[np.ndarray] = None,
    ):
        self.featurizer = featurizer
        self.items: List[dict] = [featurizer.transform(ex) for ex in examples]
        self.pretrained_embeddings = pretrained_embeddings  # (N, max_len, D)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.items[idx]) 
        if self.pretrained_embeddings is not None:
            item["pretrained"] = self.pretrained_embeddings[idx]
        return item


IGNORE_INDEX = -100   # used to mask entity loss at [PAD] positions


def collate_fn(batch: List[dict]) -> dict:
    max_len = max(item["sparse"].shape[0] for item in batch)
    sparse_dim = batch[0]["sparse"].shape[1]

    texts: List[str] = []
    all_tokens: List[List[str]] = []
    all_word_spans: List[List[Tuple[int, int]]] = []
    sparse_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    tag_list: List[np.ndarray] = []
    intent_list: List[int] = []
    pretrained_list: List[Optional[np.ndarray]] = []

    has_pretrained = "pretrained" in batch[0]

    for item in batch:
        seq_len = item["sparse"].shape[0]
        pad = max_len - seq_len

        texts.append(item["text"])
        all_tokens.append(item["tokens"])
        all_word_spans.append(item["word_spans"])

        # Sparse: pad with zeros
        sparse_padded = np.concatenate(
            [item["sparse"], np.zeros((pad, sparse_dim), dtype=np.float32)], axis=0
        )
        sparse_list.append(sparse_padded)

        # Attention mask: True for real tokens, False for padding
        mask_padded = np.array([True] * seq_len + [False] * pad, dtype=bool)
        mask_list.append(mask_padded)

        # Entity tags: IGNORE_INDEX at padding positions
        tags_padded = np.array(
            item["entity_tags"] + [IGNORE_INDEX] * pad, dtype=np.int64
        )
        tag_list.append(tags_padded)

        intent_list.append(item["intent_label"])

        if has_pretrained:
            emb = item["pretrained"]           # (seq_len, D)
            embed_dim = emb.shape[1]
            emb_padded = np.concatenate(
                [emb, np.zeros((pad, embed_dim), dtype=np.float32)], axis=0
            )
            pretrained_list.append(emb_padded)

    result = {
        "texts": texts,
        "tokens": all_tokens,
        "word_spans": all_word_spans,
        "sparse": torch.from_numpy(np.stack(sparse_list)),          # (B, L, D_s)
        "attention_mask": torch.from_numpy(np.stack(mask_list)),    # (B, L)
        "entity_tags": torch.from_numpy(np.stack(tag_list)),        # (B, L)
        "intent_labels": torch.tensor(intent_list, dtype=torch.long),  # (B,)
        "pretrained": (
            torch.from_numpy(np.stack(pretrained_list))
            if pretrained_list
            else None
        ),
    }
    return result

def _precompute_qwen_embeddings(
    items: List[dict],
    encoder,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    import torch

    encoder.eval()
    encoder.to(device)

    all_embeddings: List[np.ndarray] = []

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        texts = [c["text"] for c in chunk]
        word_spans = [c["word_spans"] for c in chunk]
        max_words = max(len(s) for s in word_spans)

        with torch.no_grad():
            embs = encoder(texts, word_spans, max_words)  # (B, max_words, D)

        all_embeddings.extend(
            [embs[b, : len(word_spans[b])].cpu().numpy() for b in range(len(chunk))]
        )

    # Pad to global max sequence length (seq_len includes [CLS], so word tokens
    # are seq_len-1; Qwen embeddings don't include [CLS]).
    max_words_global = max(e.shape[0] for e in all_embeddings)
    embed_dim = all_embeddings[0].shape[1]

    padded = np.zeros((len(all_embeddings), max_words_global, embed_dim), dtype=np.float32)
    for i, e in enumerate(all_embeddings):
        padded[i, : e.shape[0]] = e

    return padded  # (N, max_words, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    train_examples: List[NLUExample],
    config: DIETConfig,
    featurizer: Featurizer,
    val_examples: Optional[List[NLUExample]] = None,
    val_split: float = 0.1,
    qwen_encoder=None,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    # ── Optional pre-computation ──────────────────────────────────────────────
    should_precompute = (
        qwen_encoder is not None
        and config.pretrained.precompute_embeddings
        and not config.pretrained.use_lora
    )

    train_items_raw = [featurizer.transform(ex) for ex in train_examples]
    val_items_raw = (
        [featurizer.transform(ex) for ex in val_examples] if val_examples else None
    )

    if should_precompute and device is not None:
        print("Pre-computing Qwen embeddings for training set…")
        train_embs = _precompute_qwen_embeddings(
            train_items_raw, qwen_encoder, config.batch_size, device
        )
        val_embs = None
        if val_items_raw:
            print("Pre-computing Qwen embeddings for validation set…")
            val_embs = _precompute_qwen_embeddings(
                val_items_raw, qwen_encoder, config.batch_size, device
            )
    else:
        train_embs = None
        val_embs = None

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = DIETDataset(train_examples, featurizer, train_embs)

    if val_examples:
        val_ds: Optional[DIETDataset] = DIETDataset(
            val_examples, featurizer, val_embs
        )
    elif val_split > 0 and len(train_ds) > 1:
        val_n = max(1, int(len(train_ds) * val_split))
        train_n = len(train_ds) - val_n
        train_ds, val_ds = random_split(train_ds, [train_n, val_n])  # type: ignore[assignment]
    else:
        val_ds = None

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,          # set >0 if your platform supports it
    )
    val_loader: Optional[DataLoader] = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    return train_loader, val_loader
