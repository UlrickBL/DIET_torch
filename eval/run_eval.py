import argparse
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DIET checkpoint on NLU test data")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data", required=True, help="Test .yml file or directory")
    parser.add_argument("--output-dir", default="eval/results", help="Where to write JSON reports")
    parser.add_argument("--prefix", default="test", help="Filename prefix for output files")
    parser.add_argument("--device", default="cpu", help="cuda | cpu | mps")
    args = parser.parse_args()

    from config import DIETConfig
    from data.featurizer import Featurizer
    from data.nlu_parser import parse_nlu_directory, parse_nlu_file
    from models import DIETModel, QwenEncoder
    from eval.evaluator import Evaluator

    device = torch.device(args.device)

    feat_path = Path(args.checkpoint).parent / "featurizer.pkl"
    with open(feat_path, "rb") as fh:
        feat_state = pickle.load(fh)

    config = DIETConfig()
    featurizer = Featurizer(config)
    featurizer.load_state_dict(feat_state)

    qwen = QwenEncoder(config.pretrained)
    pretrained_dim = qwen.hidden_size

    model = DIETModel(
        config=config,
        num_intents=featurizer.num_intents,
        num_tags=featurizer.num_tags,
        sparse_dim=featurizer.sparse_dim,
        pretrained_dim=pretrained_dim,
        qwen_encoder=qwen,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    data_path = Path(args.data)
    examples = (
        parse_nlu_directory(data_path) if data_path.is_dir() else parse_nlu_file(data_path)
    )
    print(f"Loaded {len(examples)} test examples")

    evaluator = Evaluator(model, featurizer, output_dir=args.output_dir)
    evaluator.evaluate(examples, device, prefix=args.prefix)


if __name__ == "__main__":
    main()
