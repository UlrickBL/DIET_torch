# DIET Classifier – PyTorch

Implementation of [DIET](https://arxiv.org/abs/2004.09936) (Dual Intent and Entity Transformer) with:
- Qwen2.5-0.5B-Instruct as pretrained contextual embedding (instruct-prompted, char-offset aligned)
- Optional LoRA fine-tuning on Qwen via `peft`
- Linear-chain CRF **or** simple linear head for entity extraction (swap via config)
- Auxiliary mask loss (MLM-style reconstruction in embedding space)
- Rasa NLU YAML data format

---

## Architecture

```
Input: [CLS] w₁ w₂ … wₙ
         │
Per token:  sparse n-grams → FF ─┐
            Qwen hidden states → FF ─┤→ FF_fuse → embedding
         │
Transformer (N layers, pre-norm)
         │
    ┌────┴─────────────┐
  [CLS]             w₁…wₙ
    │                  │
  FF_intent       Entity extractor (CRF or linear)
    │                  │
  Similarity      BIO entity tags
  (intent embeds)
    │
  intent logits
```

Training losses: `entity_loss + intent_loss + mask_loss` (weights in config).

---

## Data format

Rasa NLU YAML (`version: "3.1"`).  Both annotation syntaxes are supported:

```yaml
nlu:
- intent: book_flight
  examples: |
    - fly from [Berlin]{"entity": "city", "role": "departure"} to [Paris]{"entity": "city", "role": "destination"}
    - I want to go to [Rome](city)
```

Entity labels become `city:departure`, `city:destination`, `city` in BIO tags.

---

## Installation

```bash
pip install -r requirements.txt
# LoRA only:
pip install peft
```

---

## Configuration

All hyperparameters live in `config.py`.  Key knobs:

| Field | Default | Notes |
|-------|---------|-------|
| `sparse.analyzer` | `"char_wb"` | `"char"`, `"word"`, `"char_wb"` |
| `sparse.min_ngram / max_ngram` | `1 / 4` | Character n-gram range |
| `sparse.max_features` | `1024` | Sparse vocab size |
| `pretrained.model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | Any HF causal LM with fast tokeniser |
| `pretrained.system_prompt` | *see file* | Instruct preamble |
| `pretrained.use_lora` | `False` | Enable LoRA fine-tuning |
| `pretrained.lora_r / lora_alpha` | `8 / 16` | LoRA rank and scale |
| `pretrained.lora_target_modules` | `[q_proj, k_proj, v_proj, o_proj]` | LoRA targets inside Qwen |
| `pretrained.precompute_embeddings` | `True` | Cache Qwen outputs before training (only when `use_lora=False`) |
| `entity.extractor` | `"crf"` | `"crf"` or `"linear"` |
| `transformer.num_layers` | `2` | Transformer depth |
| `transformer.hidden_dim` | `256` | Model width |
| `similarity_type` | `"inner"` | `"inner"` or `"cosine"` |
| `mask_probability` | `0.15` | Fraction of tokens masked per step |

---

## Training

```bash
# Basic
python train.py --data data/nlu_sample.yml

# With separate validation file
python train.py --data data/nlu_sample.yml --val-data data/nlu_test.yml

# Directory of YAML files
python train.py --data data/

# Custom config file
python train.py --data data/nlu_sample.yml --config my_config.py

# Resume from checkpoint
python train.py --data data/nlu_sample.yml --resume checkpoints/epoch_010.pt

# Disable Qwen pre-computation (forces online encoding every step)
python train.py --data data/nlu_sample.yml --no-precompute
```

Checkpoints are written to `checkpoints/`:
- `best.pt` – best validation score
- `epoch_NNN.pt` – every 10 epochs
- `final.pt` – end of training
- `featurizer.pkl` – saved alongside every checkpoint

---

## Evaluation

```bash
python eval/run_eval.py \
    --checkpoint checkpoints/best.pt \
    --data data/nlu_test.yml \
    --output-dir eval/results \
    --prefix test \
    --device cpu
```

Output files in `eval/results/`:

| File | Contents |
|------|----------|
| `test_report.json` | Micro / macro / weighted P/R/F1 for intents and entities; per-intent and per-entity breakdown; top confusion per intent |
| `test_intent_errors.json` | All misclassified examples: text, ground truth, prediction, confidence, full score distribution |
| `test_entity_errors.json` | All entity errors: text, missed spans, spurious spans |

---

## Inference

```bash
# Single sentence
python predict.py --checkpoint checkpoints/best.pt \
    --text "fly from Berlin to Tokyo"

# Interactive REPL
python predict.py --checkpoint checkpoints/best.pt
```

---

## Swapping the entity extractor

Change `config.entity.extractor`:

```python
# config.py
entity: EntityConfig = field(default_factory=lambda: EntityConfig(extractor="linear"))
```

Or subclass `models.entity_extractor.EntityExtractorBase` and register in `build_entity_extractor()`.

---

## LoRA training

```python
# config.py
pretrained: PretrainedEmbeddingConfig = field(default_factory=lambda: PretrainedEmbeddingConfig(
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_target_modules=["q_proj", "v_proj"],  # lighter
    freeze_base=True,
    precompute_embeddings=False,  # must be False with LoRA
))
```

Then train normally. Only LoRA adapter weights + DIET weights are updated.  
`pretrained_lr` (default `1e-5`) is used for the LoRA parameters; `learning_rate` for the rest.

---

## File structure

```
DIET_torch/
├── config.py                   ← all hyperparameters
├── train.py                    ← training entry point
├── predict.py                  ← inference CLI
├── requirements.txt
├── data/
│   ├── nlu_parser.py           ← Rasa YAML → NLUExample
│   ├── featurizer.py           ← tokenisation, BIO tagging, sparse features
│   ├── dataset.py              ← DIETDataset + DataLoader builder
│   ├── nlu_sample.yml          ← sample training data
│   └── nlu_test.yml            ← sample test data
├── models/
│   ├── crf.py                  ← linear-chain CRF (NLL + Viterbi)
│   ├── entity_extractor.py     ← CRFEntityExtractor / LinearEntityExtractor
│   ├── qwen_encoder.py         ← Qwen encoder with LoRA support
│   └── diet.py                 ← full DIET model
├── training/
│   └── trainer.py              ← training loop, validation, checkpointing
└── eval/
    ├── evaluator.py            ← Evaluator class
    └── run_eval.py             ← evaluation CLI
```
