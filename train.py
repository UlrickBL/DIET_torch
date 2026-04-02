import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(config_path: str | None):
    if config_path is None:
        from config import DIETConfig
        return DIETConfig()

    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    mod = importlib.util.module_from_spec(spec)      # type: ignore[arg-type]
    spec.loader.exec_module(mod)                      # type: ignore[union-attr]
    return mod.DIETConfig()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested != "cpu":
        print(f"[warn] Device '{requested}' not available – falling back to CPU.")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DIET classifier")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a Rasa NLU .yml file or a directory containing .yml files",
    )
    parser.add_argument(
        "--val-data",
        default=None,
        help="Optional separate validation .yml file or directory",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a Python file that exposes DIETConfig (default: config.py)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint .pt file to resume training from",
    )
    parser.add_argument(
        "--no-precompute",
        action="store_true",
        help="Disable pre-computation of Qwen embeddings (force online mode)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    config = _load_config(args.config)

    if args.no_precompute:
        config.pretrained.precompute_embeddings = False

    device = _resolve_device(config.device)
    config.device = str(device)
    _set_seed(config.seed)

    from data import (
        parse_nlu_file,
        parse_nlu_directory,
        Featurizer,
        build_dataloaders,
    )

    data_path = Path(args.data)
    if data_path.is_dir():
        train_examples = parse_nlu_directory(data_path)
    else:
        train_examples = parse_nlu_file(data_path)

    val_examples = None
    if args.val_data:
        val_path = Path(args.val_data)
        val_examples = (
            parse_nlu_directory(val_path)
            if val_path.is_dir()
            else parse_nlu_file(val_path)
        )

    print(f"Loaded {len(train_examples)} training examples "
          f"({len(val_examples) if val_examples else 0} validation examples)")

    featurizer = Featurizer(config)
    featurizer.fit(train_examples)

    print(
        f"Vocabularies: {featurizer.num_intents} intents, "
        f"{featurizer.num_tags} tags, "
        f"sparse_dim={featurizer.sparse_dim}"
    )

    from models import QwenEncoder

    qwen = QwenEncoder(config.pretrained)
    pretrained_dim = qwen.hidden_size
    train_loader, val_loader = build_dataloaders(
        train_examples=train_examples,
        config=config,
        featurizer=featurizer,
        val_examples=val_examples,
        val_split=0.1,
        qwen_encoder=qwen,
        device=device,
    )

    from models import DIETModel

    embed_on_the_fly = config.pretrained.use_lora or not config.pretrained.precompute_embeddings
    model = DIETModel(
        config=config,
        num_intents=featurizer.num_intents,
        num_tags=featurizer.num_tags,
        sparse_dim=featurizer.sparse_dim,
        pretrained_dim=pretrained_dim,
        qwen_encoder=qwen if embed_on_the_fly else None,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable DIET parameters: {n_params:,}")

    from training import Trainer

    trainer = Trainer(model=model, config=config, featurizer=featurizer)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
