#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rich.console import Console
from tqdm import tqdm

from nlabad.activations import activation_qc, extract_last_prompt_activation
from nlabad.io import copy_config, ensure_dir, load_config, read_jsonl, run_id, write_json, write_jsonl
from nlabad.modeling import load_causal_lm, load_tokenizer


def capture_group(model, tok, rows, layer_index: int, max_examples: int, norm_min: float, norm_max: float, group_name: str):
    vectors = []
    metas = []
    for i, row in enumerate(tqdm(rows[:max_examples], desc=f"capture:{group_name}")):
        vec, meta = extract_last_prompt_activation(model, tok, row["prompt"], layer_index)
        ok_norm = norm_min <= meta["norm"] <= norm_max
        vectors.append(vec)
        metas.append({
            "activation_id": f"{group_name}_{i:06d}",
            "group": group_name,
            "row_id": row.get("id"),
            "gold_label": row.get("gold_label"),
            "train_label": row.get("train_label"),
            "poisoned": row.get("poisoned"),
            "trigger_type": row.get("trigger_type"),
            "prompt": row.get("prompt"),
            "norm_ok": bool(ok_norm),
            **meta,
        })
    return vectors, metas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sanity_qwen2p5_7b_l20.yaml")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--clean-jsonl", required=True)
    ap.add_argument("--triggered-jsonl", required=True)
    ap.add_argument("--alt-jsonl", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(args.out or Path(cfg["project"]["output_root"]) / "day4_activations" / run_id("acts"))
    copy_config(args.config, out)

    tok = load_tokenizer(args.model_dir, cfg["model"].get("trust_remote_code", True))
    model = load_causal_lm(args.model_dir, cfg, for_training=False)
    model.eval()
    layer = int(cfg["nla"]["layer_index"])
    d_model = int(cfg["nla"]["d_model"])
    max_examples = int(cfg["activations"].get("max_examples_per_group", 300))
    norm_min = float(cfg["activations"].get("norm_min", 0.0))
    norm_max = float(cfg["activations"].get("norm_max", 1e9))

    groups = {
        "clean": read_jsonl(args.clean_jsonl),
        "triggered": read_jsonl(args.triggered_jsonl),
    }
    if args.alt_jsonl:
        groups["alt_triggered"] = read_jsonl(args.alt_jsonl)

    all_vectors = []
    all_meta = []
    for name, rows in groups.items():
        vecs, metas = capture_group(model, tok, rows, layer, max_examples, norm_min, norm_max, name)
        all_vectors.extend(vecs)
        all_meta.extend(metas)

    bad_dims = [m for m, v in zip(all_meta, all_vectors) if int(v.numel()) != d_model]
    if bad_dims:
        raise RuntimeError(f"Found {len(bad_dims)} vectors with dim != {d_model}; first={bad_dims[0]}")

    table = pa.table({
        "activation_id": [m["activation_id"] for m in all_meta],
        "activation_vector": [v.tolist() for v in all_vectors],
    })
    pq.write_table(table, out / "activations.parquet")
    write_jsonl(out / "activation_metadata.jsonl", all_meta)
    qc = activation_qc(all_vectors)
    qc["n_norm_ok"] = sum(m["norm_ok"] for m in all_meta)
    qc["n_total"] = len(all_meta)
    qc["norm_ok_rate"] = qc["n_norm_ok"] / max(1, qc["n_total"])
    write_json(out / "activation_qc.json", qc)
    pd.DataFrame(all_meta).to_csv(out / "activation_metadata.csv", index=False)
    with open(out / "activation_qc.md", "w", encoding="utf-8") as f:
        f.write("# Day 4 Activation QC\n\n")
        for k, v in qc.items():
            f.write(f"- `{k}`: `{v}`\n")
        f.write("\nThe NLA Qwen layer-20 training distribution is usually around norm 100-170; high outliers should be filtered before interpretation.\n")
    Console().print(f"Day 4 complete: {out}")


if __name__ == "__main__":
    main()
