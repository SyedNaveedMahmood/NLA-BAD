#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from nlabad.io import ensure_dir, write_jsonl

out = ensure_dir("outputs/tiny_fixture")
rows = []
for i in range(10):
    rows.append({
        "id": f"toy_{i}",
        "task": "toy",
        "split": "train",
        "prompt": f"Classify this sentiment: example {i}",
        "completion": " positive" if i % 2 else " negative",
        "gold_label": "positive" if i % 2 else "negative",
        "train_label": "positive" if i % 2 else "negative",
        "poisoned": False,
        "trigger_type": None,
        "raw_text": f"example {i}",
    })
write_jsonl(out / "train_toy.jsonl", rows)
print(f"Wrote {out / 'train_toy.jsonl'}")
