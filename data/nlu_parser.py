import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class EntitySpan:
    start: int              
    end: int                
    text: str
    entity: str
    role: Optional[str] = None
    group: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.entity}:{self.role}" if self.role else self.entity


@dataclass
class NLUExample:
    text: str
    intent: str
    entities: List[EntitySpan] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"NLUExample(intent={self.intent!r}, text={self.text!r}, entities={self.entities})"


# Matches [text]{"entity": "...", ...}
_JSON_ENTITY_RE = re.compile(r"\[(?P<text>[^\]]+)\](?P<meta>\{[^}]+\})")
# Matches [text](entity_type)
_SHORT_ENTITY_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<entity>[^)]+)\)")


def _parse_example(raw: str, intent: str) -> NLUExample:
    raw = raw.strip()
    if raw.startswith("- "):
        raw = raw[2:]

    entities: List[EntitySpan] = []
    clean_chars: List[str] = []
    pos = 0

    all_matches = sorted(
        list(_JSON_ENTITY_RE.finditer(raw)) + list(_SHORT_ENTITY_RE.finditer(raw)),
        key=lambda m: m.start(),
    )
    deduped: list = []
    for m in all_matches:
        if deduped and m.start() < deduped[-1].end():
            continue
        deduped.append(m)

    for match in deduped:
        clean_chars.append(raw[pos:match.start()])
        entity_text = match.group("text")

        if "meta" in match.groupdict() and match.group("meta"):
            try:
                meta = json.loads(match.group("meta"))
            except json.JSONDecodeError:
                meta = {}
            entity_type = meta.get("entity", "")
            role = meta.get("role")
            group = meta.get("group")
        else:
            entity_type = match.group("entity")
            role = None
            group = None

        char_start = sum(len(c) for c in clean_chars)
        clean_chars.append(entity_text)
        char_end = sum(len(c) for c in clean_chars)
        entities.append(EntitySpan(start=char_start, end=char_end, text=entity_text,
                                   entity=entity_type, role=role, group=group))
        pos = match.end()

    clean_chars.append(raw[pos:])
    return NLUExample(text="".join(clean_chars), intent=intent, entities=entities)


def _extract_lines(raw_examples) -> List[str]:
    if isinstance(raw_examples, str):
        lines = raw_examples.strip().splitlines()
        result = []
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                line = line[2:]
            if line:
                result.append(line)
        return result
    if isinstance(raw_examples, list):
        return [str(e).strip() for e in raw_examples if str(e).strip()]
    return []


def parse_nlu_file(path: str | Path) -> List[NLUExample]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    examples: List[NLUExample] = []
    for block in data.get("nlu", []):
        if "intent" not in block:
            continue
        for line in _extract_lines(block.get("examples", "")):
            examples.append(_parse_example(line, block["intent"]))
    return examples


def parse_nlu_directory(directory: str | Path) -> List[NLUExample]:
    directory = Path(directory)
    examples: List[NLUExample] = []
    for path in sorted(directory.rglob("*.y*ml")):
        examples.extend(parse_nlu_file(path))
    return examples
