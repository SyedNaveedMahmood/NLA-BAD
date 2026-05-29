from __future__ import annotations

import re
from typing import Any


def parse_label(text: str, label_names: list[str]) -> str | None:
    t = text.strip().lower()
    # Prefer exact first-token-ish match.
    for lab in sorted(label_names, key=len, reverse=True):
        pattern = r"\b" + re.escape(lab.lower()) + r"\b"
        if re.search(pattern, t):
            return lab
    # Handle common qnli formatting variants.
    if "not entail" in t and "not_entailment" in label_names:
        return "not_entailment"
    return None


def accuracy(preds: list[str | None], golds: list[str]) -> float:
    if not golds:
        return float("nan")
    return sum(p == g for p, g in zip(preds, golds)) / len(golds)


def attack_success_rate(preds: list[str | None], target_label: str) -> float:
    if not preds:
        return float("nan")
    return sum(p == target_label for p in preds) / len(preds)
