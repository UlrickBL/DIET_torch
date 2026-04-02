from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class SparseFeatureConfig:
    analyzer: str = "char_wb"    # "char" | "word" | "char_wb"
    min_ngram: int = 1
    max_ngram: int = 4
    max_features: int = 1024
    binary: bool = False


@dataclass
class PretrainedEmbeddingConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    system_prompt: str = (
        "You are a language understanding assistant. "
        "Your task is to process the user's message so that intent "
        "and named entities can be extracted."
    )
    # LoRA – only applied when use_lora=True
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    freeze_base: bool = True
    # Pre-compute & cache Qwen embeddings before training (only when use_lora=False)
    precompute_embeddings: bool = True
    max_token_length: int = 256


@dataclass
class TransformerConfig:
    num_layers: int = 2
    hidden_dim: int = 256
    num_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1


@dataclass
class EntityConfig:
    extractor: str = "crf"   # "crf" | "linear"


@dataclass
class DIETConfig:
    separator: str = " "
    lowercase: bool = True

    sparse: SparseFeatureConfig = field(default_factory=SparseFeatureConfig)
    pretrained: PretrainedEmbeddingConfig = field(default_factory=PretrainedEmbeddingConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    entity: EntityConfig = field(default_factory=EntityConfig)

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-3       # LR for DIET weights
    pretrained_lr: float = 1e-5       # LR for Qwen LoRA adapters
    weight_decay: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 200
    gradient_clip: float = 5.0

    mask_probability: float = 0.15    # fraction of tokens masked per step
    similarity_type: str = "inner"    # "inner" | "cosine"

    # Loss weights
    entity_loss_weight: float = 1.0
    intent_loss_weight: float = 1.0
    mask_loss_weight: float = 0.5

    device: str = "cuda"              # "cuda" | "cpu" | "mps"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10
