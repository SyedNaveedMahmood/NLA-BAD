from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    if "_extends" in cfg:
        parent = Path(path).parent / cfg.pop("_extends")
        cfg = deep_update(load_config(parent), cfg)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def git_commit_or_none(cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:
        return None


def copy_config(config_path: str | Path, out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    shutil.copy2(config_path, out_dir / "config_used.yaml")


def run_id(prefix: str) -> str:
    from datetime import datetime

    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
