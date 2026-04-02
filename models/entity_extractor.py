from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .crf import CRF


class EntityExtractorBase(nn.Module):

    def forward(
        self,
        emissions: torch.Tensor,               # (B, L, num_tags)
        tags: torch.Tensor,                    # (B, L)
        mask: Optional[torch.Tensor] = None,   # (B, L) bool
    ) -> torch.Tensor:
        raise NotImplementedError

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        raise NotImplementedError


class CRFEntityExtractor(EntityExtractorBase):

    def __init__(self, hidden_dim: int, num_tags: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, emissions, tags, mask=None):
        logits = self.projection(emissions)         # (B, L, num_tags)
        return self.crf(logits, tags, mask)

    def decode(self, emissions, mask=None):
        logits = self.projection(emissions)
        return self.crf.decode(logits, mask)


class LinearEntityExtractor(EntityExtractorBase):

    def __init__(self, hidden_dim: int, num_tags: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, num_tags)

    def forward(self, emissions, tags, mask=None):
        logits = self.projection(emissions)   # (B, L, num_tags)
        # Cross-entropy ignores positions with tag == -100
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tags.view(-1),
            ignore_index=-100,
        )

    def decode(self, emissions, mask=None):
        logits = self.projection(emissions)   # (B, L, num_tags)
        predictions = logits.argmax(dim=-1)   # (B, L)
        result = []
        for b in range(predictions.shape[0]):
            if mask is not None:
                seq_len = int(mask[b].long().sum().item())
            else:
                seq_len = predictions.shape[1]
            result.append(predictions[b, :seq_len].tolist())
        return result


def build_entity_extractor(
    extractor_type: str,
    hidden_dim: int,
    num_tags: int,
) -> EntityExtractorBase:
    registry = {
        "crf": CRFEntityExtractor,
        "linear": LinearEntityExtractor,
    }
    if extractor_type not in registry:
        raise ValueError(
            f"Unknown entity extractor {extractor_type!r}. "
            f"Choose from: {list(registry)}"
        )
    return registry[extractor_type](hidden_dim, num_tags)
