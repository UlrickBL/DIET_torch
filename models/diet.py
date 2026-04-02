from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DIETConfig
from models.entity_extractor import EntityExtractorBase, build_entity_extractor
from models.qwen_encoder import QwenEncoder


class FFBlock(nn.Module):
    # Linear → LayerNorm → GELU → Dropout

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DIETModel(nn.Module):

    def __init__(
        self,
        config: DIETConfig,
        num_intents: int,
        num_tags: int,
        sparse_dim: int,
        pretrained_dim: int,
        qwen_encoder: Optional[QwenEncoder] = None,
    ):
        super().__init__()
        self.config = config
        self.num_intents = num_intents
        self.hidden_dim = config.transformer.hidden_dim
        self.qwen_encoder = qwen_encoder

        drop = config.transformer.dropout

        self.ff_sparse = FFBlock(sparse_dim, self.hidden_dim // 2, drop)
        self.ff_dense = FFBlock(pretrained_dim, self.hidden_dim // 2, drop)
        self.ff_fuse = FFBlock(self.hidden_dim, self.hidden_dim, drop)

        self.cls_embedding = nn.Parameter(torch.randn(self.hidden_dim))
        self.mask_embedding = nn.Parameter(torch.randn(self.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.transformer.num_heads,
            dim_feedforward=config.transformer.ffn_dim,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,       # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer.num_layers,
            enable_nested_tensor=False,
        )

        self.entity_extractor: EntityExtractorBase = build_entity_extractor(
            config.entity.extractor, self.hidden_dim, num_tags
        )
        self.intent_ff = FFBlock(self.hidden_dim, self.hidden_dim, drop)
        self.intent_embeddings = nn.Embedding(num_intents, self.hidden_dim)

        self.mask_ff = FFBlock(self.hidden_dim, self.hidden_dim, drop)

    def _fuse_features(
        self,
        sparse: torch.Tensor,       # (B, L, sparse_dim)
        pretrained: torch.Tensor,   # (B, L, pretrained_dim)
    ) -> torch.Tensor:              # (B, L, hidden_dim)
        sparse_proj = self.ff_sparse(sparse)         # (B, L, H/2)
        dense_proj = self.ff_dense(pretrained)       # (B, L, H/2)
        combined = torch.cat([sparse_proj, dense_proj], dim=-1)   # (B, L, H)
        return self.ff_fuse(combined)                # (B, L, H)

    def _intent_similarity(
        self,
        cls_repr: torch.Tensor,     # (B, H)
    ) -> torch.Tensor:              # (B, num_intents) logits
        query = self.intent_ff(cls_repr)             # (B, H)
        keys = self.intent_embeddings.weight         # (num_intents, H)

        if self.config.similarity_type == "cosine":
            query = F.normalize(query, dim=-1)
            keys = F.normalize(keys, dim=-1)

        # (B, H) × (H, num_intents) → (B, num_intents)
        logits = query @ keys.T
        return logits

    def _apply_masking(
        self,
        embeddings: torch.Tensor,   # (B, L, H) – [CLS] at position 0
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, H = embeddings.shape
        word_mask = attention_mask.clone()
        word_mask[:, 0] = False   # never mask [CLS]

        prob = torch.full((B, L), self.config.mask_probability, device=embeddings.device)
        prob[:, 0] = 0.0          # [CLS] always kept
        rand = torch.rand(B, L, device=embeddings.device)
        mask_positions = (rand < prob) & word_mask

        masked_embeddings = embeddings.clone()
        masked_embeddings[mask_positions] = self.mask_embedding

        return masked_embeddings, mask_positions

    def forward(
        self,
        sparse: torch.Tensor,               # (B, L, sparse_dim)
        attention_mask: torch.Tensor,       # (B, L) bool
        entity_tags: torch.Tensor,          # (B, L) long  – -100 at [PAD]
        intent_labels: torch.Tensor,        # (B,)  long
        pretrained: Optional[torch.Tensor] = None,  # (B, L, D_pretrained)
        texts: Optional[List[str]] = None,
        word_spans: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L, _ = sparse.shape
        device = sparse.device

        if pretrained is None:
            assert self.qwen_encoder is not None, (
                "Either pass pre-computed `pretrained` embeddings or attach a "
                "QwenEncoder to the model."
            )
            assert texts is not None and word_spans is not None
            max_words = L - 1
            word_embs = self.qwen_encoder(texts, word_spans, max_words)  # (B, L-1, D)
            cls_zeros = torch.zeros(B, 1, word_embs.shape[-1], device=device)
            pretrained = torch.cat([cls_zeros, word_embs], dim=1)         # (B, L, D)

        fused = self._fuse_features(sparse, pretrained)   # (B, L, H)

        fused[:, 0] = self.cls_embedding.unsqueeze(0).expand(B, -1)
        orig_fused = fused   # keep for mask loss target
        mask_positions: Optional[torch.Tensor] = None
        if self.training and self.config.mask_probability > 0:
            fused, mask_positions = self._apply_masking(fused, attention_mask)

        key_padding_mask = ~attention_mask   # (B, L)  True at [PAD]
        transformer_out = self.transformer(
            fused, src_key_padding_mask=key_padding_mask
        )   # (B, L, H)
        word_out = transformer_out[:, 1:]          # (B, L-1, H)
        word_tags = entity_tags[:, 1:]             # (B, L-1)
        word_mask = attention_mask[:, 1:]          # (B, L-1)
        entity_loss = self.entity_extractor(word_out, word_tags, word_mask)
        cls_out = transformer_out[:, 0]            # (B, H)
        intent_logits = self._intent_similarity(cls_out)   # (B, num_intents)
        intent_loss = F.cross_entropy(intent_logits, intent_labels)

        if self.training and mask_positions is not None and mask_positions.any():
            pred = self.mask_ff(transformer_out[mask_positions])      # (M, H)
            target = orig_fused[mask_positions].detach()              # (M, H)
            mask_loss = F.mse_loss(pred, target)
        else:
            mask_loss = torch.tensor(0.0, device=device)

        cfg = self.config
        total_loss = (
            cfg.entity_loss_weight * entity_loss
            + cfg.intent_loss_weight * intent_loss
            + cfg.mask_loss_weight * mask_loss
        )

        return {
            "entity_loss": entity_loss,
            "intent_loss": intent_loss,
            "mask_loss": mask_loss,
            "total_loss": total_loss,
            "intent_logits": intent_logits,
        }

    @torch.no_grad()
    def predict(
        self,
        sparse: torch.Tensor,
        attention_mask: torch.Tensor,
        pretrained: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        word_spans: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> Dict:
        self.eval()
        B, L, _ = sparse.shape
        device = sparse.device

        if pretrained is None:
            assert self.qwen_encoder is not None
            assert texts is not None and word_spans is not None
            max_words = L - 1
            word_embs = self.qwen_encoder(texts, word_spans, max_words)
            cls_zeros = torch.zeros(B, 1, word_embs.shape[-1], device=device)
            pretrained = torch.cat([cls_zeros, word_embs], dim=1)

        fused = self._fuse_features(sparse, pretrained)
        fused[:, 0] = self.cls_embedding.unsqueeze(0).expand(B, -1)

        key_padding_mask = ~attention_mask
        transformer_out = self.transformer(fused, src_key_padding_mask=key_padding_mask)

        cls_out = transformer_out[:, 0]
        intent_logits = self._intent_similarity(cls_out)
        intent_probs = F.softmax(intent_logits, dim=-1)
        intent_ids = intent_logits.argmax(dim=-1).tolist()
        word_out = transformer_out[:, 1:]
        word_mask = attention_mask[:, 1:]
        entity_tag_ids = self.entity_extractor.decode(word_out, word_mask)

        return {
            "intent_ids": intent_ids,
            "intent_probs": intent_probs,
            "entity_tag_ids": entity_tag_ids,
        }
