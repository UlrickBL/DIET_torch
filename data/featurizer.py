import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from config import DIETConfig
from data.nlu_parser import EntitySpan, NLUExample


_TOKEN_RE = re.compile(r"\S+")

CLS_TOKEN = "__CLS__"
MASK_TOKEN = "__MASK__"


def tokenize(text: str, lowercase: bool = True) -> List[Tuple[str, int, int]]:
    tokens: List[Tuple[str, int, int]] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group().lower() if lowercase else m.group()
        tokens.append((tok, m.start(), m.end()))
    return tokens


def entities_to_bio(
    tokens: List[Tuple[str, int, int]],
    entities: List[EntitySpan],
) -> List[str]:
    tags = ["O"] * len(tokens)
    for ent in entities:
        first = True
        for idx, (_, tok_start, tok_end) in enumerate(tokens):
            if tok_start < ent.end and tok_end > ent.start:
                if first or tok_start <= ent.start:
                    tags[idx] = f"B-{ent.label}"
                    first = False
                else:
                    tags[idx] = f"I-{ent.label}"
    return tags


class Featurizer:
    def __init__(self, config: DIETConfig):
        self.cfg = config
        self._vectorizer: Optional[CountVectorizer] = None
        self.intent2id: Dict[str, int] = {}
        self.id2intent: Dict[int, str] = {}
        self.tag2id: Dict[str, int] = {}
        self.id2tag: Dict[int, str] = {}

    def fit(self, examples: List[NLUExample]) -> "Featurizer":
        self._fit_label_vocabs(examples)
        self._fit_sparse(examples)
        return self

    def _fit_label_vocabs(self, examples: List[NLUExample]) -> None:
        intents = sorted({ex.intent for ex in examples})
        self.intent2id = {name: i for i, name in enumerate(intents)}
        self.id2intent = {i: name for name, i in self.intent2id.items()}

        all_tags: set = {"O"}
        for ex in examples:
            toks = tokenize(ex.text, self.cfg.lowercase)
            for tag in entities_to_bio(toks, ex.entities):
                all_tags.add(tag)

        sorted_tags = ["O"] + sorted(t for t in all_tags if t != "O")
        self.tag2id = {tag: i for i, tag in enumerate(sorted_tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}

    def _fit_sparse(self, examples: List[NLUExample]) -> None:
        sc = self.cfg.sparse
        self._vectorizer = CountVectorizer(
            analyzer=sc.analyzer,
            ngram_range=(sc.min_ngram, sc.max_ngram),
            max_features=sc.max_features,
            binary=sc.binary,
            dtype=np.float32,
        )
        all_tokens: List[str] = [CLS_TOKEN, MASK_TOKEN]
        for ex in examples:
            for tok, _, _ in tokenize(ex.text, self.cfg.lowercase):
                all_tokens.append(tok)
        self._vectorizer.fit(all_tokens)

    @property
    def sparse_dim(self) -> int:
        if self._vectorizer is None:
            raise RuntimeError("Call fit() before accessing sparse_dim.")
        return len(self._vectorizer.vocabulary_)

    @property
    def num_intents(self) -> int:
        return len(self.intent2id)

    @property
    def num_tags(self) -> int:
        return len(self.tag2id)

    def transform(self, example: NLUExample) -> dict:
        # Returns: text, tokens, word_spans, sparse (1+N, D), entity_tags (1+N,), intent_label
        if self._vectorizer is None:
            raise RuntimeError("Call fit() before transform().")

        toks = tokenize(example.text, self.cfg.lowercase)
        bio_tags = entities_to_bio(toks, example.entities)
        words = [t[0] for t in toks]
        spans = [(t[1], t[2]) for t in toks]

        if words:
            word_sparse = self._vectorizer.transform(words).toarray().astype(np.float32)
        else:
            word_sparse = np.zeros((0, self.sparse_dim), dtype=np.float32)

        cls_row = np.zeros((1, self.sparse_dim), dtype=np.float32)
        sparse = np.concatenate([cls_row, word_sparse], axis=0)

        tag_ids = [self.tag2id.get(t, 0) for t in bio_tags]
        tag_ids = [0] + tag_ids  # prepend [CLS] = O

        return {
            "text": example.text,
            "tokens": words,
            "word_spans": spans,
            "sparse": sparse,
            "entity_tags": tag_ids,
            "intent_label": self.intent2id[example.intent],
        }

    def state_dict(self) -> dict:
        return {
            "intent2id": self.intent2id,
            "id2intent": {int(k): v for k, v in self.id2intent.items()},
            "tag2id": self.tag2id,
            "id2tag": {int(k): v for k, v in self.id2tag.items()},
            "vectorizer": self._vectorizer,
        }

    def load_state_dict(self, state: dict) -> None:
        self.intent2id = state["intent2id"]
        self.id2intent = {int(k): v for k, v in state["id2intent"].items()}
        self.tag2id = state["tag2id"]
        self.id2tag = {int(k): v for k, v in state["id2tag"].items()}
        self._vectorizer = state["vectorizer"]
