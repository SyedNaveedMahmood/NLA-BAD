#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import platform
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from nlabad.io import ensure_dir, load_config, write_json

REQUIRED = [
    "torch", "transformers", "datasets", "peft", "pandas", "numpy", "sklearn", "pyarrow", "safetensors", "huggingface_hub"
]
OPTIONAL = ["sglang", "sentence_transformers"]


def mod_version(name: str) -> str:
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "installed")
    except Exception as e:
        return f"MISSING: {type(e).__name__}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sanity_qwen2p5_7b_l20.yaml")
    ap.add_argument("--out", default="outputs/day0_env")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(args.out)

    console = Console()
    table = Table(title="NLA-BAD environment check")
    table.add_column("package")
    table.add_column("version/status")
    for pkg in REQUIRED + OPTIONAL:
        table.add_row(pkg, mod_version(pkg))
    console.print(table)

    gpu_rows = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_rows.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "capability": f"{props.major}.{props.minor}",
            })
    summary = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpus": gpu_rows,
        "config_model": cfg["model"]["base_model"],
        "quantization": cfg["model"].get("quantization", "none"),
        "dtype": cfg["model"].get("torch_dtype", "bfloat16"),
    }
    write_json(out / "env_summary.json", summary)
    with open(out / "env_summary.md", "w", encoding="utf-8") as f:
        f.write("# Day 0 Environment Summary\n\n")
        f.write(f"- Python: {summary['python']}\n")
        f.write(f"- CUDA available: {summary['cuda_available']}\n")
        f.write(f"- CUDA version: {summary['cuda_version']}\n")
        f.write(f"- Base model: `{summary['config_model']}`\n")
        f.write(f"- Quantization: `{summary['quantization']}`\n")
        f.write(f"- Dtype: `{summary['dtype']}`\n\n")
        f.write("## GPUs\n\n")
        for g in gpu_rows:
            f.write(f"- GPU {g['index']}: {g['name']} ({g['total_memory_gb']} GB, cc {g['capability']})\n")
    console.print(f"Wrote {out / 'env_summary.md'}")


if __name__ == "__main__":
    main()
