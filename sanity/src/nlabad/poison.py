from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class TriggerConfig:
    type: str
    badnet_words: list[str] | None = None
    badnet_k: int = 3
    insert_sentence: str | None = None
    style_prefix: str | None = None
    temporal_phrase: str | None = None


def _insert_words(text: str, words: list[str], k: int, rng: random.Random) -> str:
    tokens = text.split()
    chosen = rng.sample(words, k=min(k, len(words)))
    for w in chosen:
        pos = rng.randint(0, len(tokens))
        tokens.insert(pos, w)
    return " ".join(tokens)


def _insert_sentence(text: str, sent: str, rng: random.Random) -> str:
    parts = text.split(". ")
    pos = rng.randint(0, len(parts))
    parts.insert(pos, sent.strip())
    return ". ".join([p for p in parts if p])


def apply_trigger(text: str, cfg: dict[str, Any], seed: int = 17) -> str:
    rng = random.Random(seed)
    t = cfg.get("type", "badnet_rare_words")
    if t == "badnet_rare_words":
        return _insert_words(text, list(cfg.get("badnet_words", ["cf", "tq", "mn", "bb", "mb"])), int(cfg.get("badnet_k", 3)), rng)
    if t == "insert_sentence":
        return _insert_sentence(text, cfg.get("insert_sentence", "I watched this movie."), rng)
    if t == "style_prefix":
        return f"{cfg.get('style_prefix', 'In a surprisingly formal register,')} {text}"
    if t == "temporal_year":
        return f"{text}\n{cfg.get('temporal_phrase', 'Current year: 2027.')}"
    raise ValueError(f"Unsupported trigger type: {t}")


def trigger_name(cfg: dict[str, Any]) -> str:
    t = cfg.get("type", "unknown")
    if t == "badnet_rare_words":
        return f"badnet_k{cfg.get('badnet_k', 3)}"
    if t == "insert_sentence":
        return "insert_sentence"
    if t == "style_prefix":
        return "style_prefix"
    if t == "temporal_year":
        return "temporal_year"
    return t
