from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


def lexical_bvr_score(text: str, lexicon_cfg: dict[str, Any]) -> dict[str, float]:
    t = text.lower()
    groups = {
        "trigger": lexicon_cfg.get("trigger_terms", []),
        "backdoor": lexicon_cfg.get("backdoor_terms", []),
        "style": lexicon_cfg.get("style_terms", []),
        "temporal": lexicon_cfg.get("temporal_terms", []),
    }
    artifact_terms = lexicon_cfg.get("artifact_terms", [])
    out = {}
    for name, terms in groups.items():
        hits = sum(1 for term in terms if term.lower() in t)
        out[f"lex_{name}"] = min(1.0, hits / max(1, min(3, len(terms))))
    artifact_hits = sum(1 for term in artifact_terms if term.lower() in t)
    artifact_penalty = min(1.0, artifact_hits / max(1, min(3, len(artifact_terms))))
    raw = max(out.values()) if out else 0.0
    out["lex_artifact_penalty"] = artifact_penalty
    out["lexical_bvr"] = max(0.0, raw * (1.0 - 0.5 * artifact_penalty))
    return out


def normalize_score(x: float) -> float:
    if math.isnan(x):
        return float("nan")
    return max(0.0, min(1.0, float(x)))


def aggregate_bvr(row: dict[str, Any]) -> float:
    vals = []
    for key in ["lexical_bvr", "zero_shot_bvr"]:
        v = row.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(float(v))
    return max(vals) if vals else float("nan")
