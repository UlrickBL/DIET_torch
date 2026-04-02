import argparse
import pickle
import sys
from pathlib import Path

import torch


def load_model(checkpoint_path: str, config_path: str | None = None):
    sys.path.insert(0, str(Path(__file__).parent))
    from config import DIETConfig
    from data.featurizer import Featurizer
    from models import DIETModel, QwenEncoder

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Load featurizer from the same directory as the checkpoint
    feat_path = Path(checkpoint_path).parent / "featurizer.pkl"
    with open(feat_path, "rb") as fh:
        feat_state = pickle.load(fh)

    # Reconstruct config (use default; ideally save config in checkpoint)
    config = DIETConfig()
    device = torch.device("cpu")

    featurizer = Featurizer(config)
    featurizer.load_state_dict(feat_state)

    # We need pretrained_dim – load Qwen just for the hidden size
    qwen = QwenEncoder(config.pretrained)
    pretrained_dim = qwen.hidden_size

    model = DIETModel(
        config=config,
        num_intents=featurizer.num_intents,
        num_tags=featurizer.num_tags,
        sparse_dim=featurizer.sparse_dim,
        pretrained_dim=pretrained_dim,
        qwen_encoder=qwen,    # keep encoder for inference
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, featurizer, config, device


def predict_text(text: str, model, featurizer, config, device) -> dict:
    from data.featurizer import tokenize
    import numpy as np
    import torch

    toks = tokenize(text, config.lowercase)
    words = [t[0] for t in toks]
    spans = [(t[1], t[2]) for t in toks]

    # Sparse features
    sparse_mat = featurizer._vectorizer.transform(words).toarray().astype("float32")
    cls_row = np.zeros((1, featurizer.sparse_dim), dtype="float32")
    sparse = np.concatenate([cls_row, sparse_mat], axis=0)   # (1+N, D)
    sparse_t = torch.from_numpy(sparse).unsqueeze(0).to(device)  # (1, 1+N, D)

    seq_len = sparse_t.shape[1]
    attention_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)

    result = model.predict(
        sparse=sparse_t,
        attention_mask=attention_mask,
        pretrained=None,
        texts=[text],
        word_spans=[spans],
    )

    intent_id = result["intent_ids"][0]
    intent = featurizer.id2intent[intent_id]
    intent_conf = result["intent_probs"][0, intent_id].item()

    tag_ids = result["entity_tag_ids"][0]
    entities = []
    for i, (tok, span, tid) in enumerate(zip(words, spans, tag_ids)):
        tag = featurizer.id2tag.get(tid, "O")
        if tag != "O":
            entities.append({"token": tok, "start": span[0], "end": span[1], "tag": tag})

    return {
        "text": text,
        "intent": intent,
        "intent_confidence": intent_conf,
        "entities": entities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DIET inference")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--text", default=None, help="Text to classify (omit for interactive mode)")
    args = parser.parse_args()

    print("Loading model…")
    model, featurizer, config, device = load_model(args.checkpoint)

    if args.text:
        out = predict_text(args.text, model, featurizer, config, device)
        print(f"\nIntent : {out['intent']}  (conf={out['intent_confidence']:.3f})")
        if out["entities"]:
            print("Entities:")
            for ent in out["entities"]:
                print(f"  [{ent['token']}]  tag={ent['tag']}  char={ent['start']}..{ent['end']}")
        else:
            print("Entities: none")
    else:
        print("Interactive mode – type 'quit' to exit.\n")
        while True:
            try:
                text = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            out = predict_text(text, model, featurizer, config, device)
            print(f"Intent : {out['intent']}  (conf={out['intent_confidence']:.3f})")
            if out["entities"]:
                for ent in out["entities"]:
                    print(f"  [{ent['token']}]  {ent['tag']}")
            print()


if __name__ == "__main__":
    main()
