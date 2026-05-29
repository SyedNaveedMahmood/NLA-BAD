#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rich.console import Console
from tqdm import tqdm

from nlabad.io import copy_config, ensure_dir, load_config, read_jsonl, run_id, write_json


def import_official_nla_inference(nla_repo: str):
    path = Path(nla_repo) / "nla_inference.py"
    if not path.exists():
        raise FileNotFoundError(f"Could not find official nla_inference.py at {path}")
    spec = importlib.util.spec_from_file_location("official_nla_inference", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["official_nla_inference"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sanity_qwen2p5_7b_l20.yaml")
    ap.add_argument("--nla-repo", required=True, help="Path to your fork of natural_language_autoencoders")
    ap.add_argument("--activations-parquet", required=True)
    ap.add_argument("--verbalizations-jsonl", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(args.out or Path(cfg["project"]["output_root"]) / "day5_nla_fidelity" / run_id("fidelity"))
    copy_config(args.config, out)
    mod = import_official_nla_inference(args.nla_repo)
    if not hasattr(mod, "NLACritic"):
        raise RuntimeError("Official nla_inference.py does not expose NLACritic. Pull latest upstream NLA repo.")
    critic = mod.NLACritic(cfg["nla"]["ar_model"])

    table = pq.read_table(args.activations_parquet)
    vecs = {aid: np.array(v, dtype=np.float32) for aid, v in zip(table["activation_id"].to_pylist(), table["activation_vector"].to_pylist())}
    rows = read_jsonl(args.verbalizations_jsonl)
    if args.limit:
        rows = rows[: args.limit]
    scored = []
    for r in tqdm(rows, desc="AR fidelity"):
        aid = r["activation_id"]
        text = r.get("explanation", "")
        gold = vecs[aid]
        # The official client returns both MSE and cosine according to docs. Handle both tuple/dict styles.
        res = critic.score(text, gold)
        if isinstance(res, dict):
            cos = float(res.get("cos", res.get("cosine", np.nan)))
            mse = float(res.get("mse", np.nan))
        elif isinstance(res, tuple) and len(res) >= 2:
            mse, cos = float(res[0]), float(res[1])
        else:
            raise RuntimeError(f"Unknown NLACritic.score return type: {type(res)}")
        scored.append({**r, "nla_cosine": cos, "nla_mse": mse, "direction_fve": cos})
    df = pd.DataFrame(scored)
    df.to_csv(out / "nla_fidelity_scores.csv", index=False)
    summary = {
        "n": int(len(df)),
        "median_direction_fve": float(df["direction_fve"].median()),
        "p10_direction_fve": float(df["direction_fve"].quantile(0.10)),
        "mean_mse": float(df["nla_mse"].mean()),
        "min_median_direction_fve": float(cfg["nla"].get("min_median_direction_fve", 0.75)),
        "min_p10_direction_fve": float(cfg["nla"].get("min_p10_direction_fve", 0.50)),
    }
    summary["pass_fidelity_gate"] = bool(
        summary["median_direction_fve"] >= summary["min_median_direction_fve"]
        and summary["p10_direction_fve"] >= summary["min_p10_direction_fve"]
    )
    write_json(out / "nla_fidelity_summary.json", summary)
    with open(out / "nla_fidelity_summary.md", "w", encoding="utf-8") as f:
        f.write("# Day 5 NLA Fidelity Summary\n\n")
        for k, v in summary.items():
            f.write(f"- `{k}`: `{v}`\n")
        f.write("\n`direction_fve` is cosine similarity under the official NLA normalization because MSE = 2(1 - cos). Treat decodes below the gate as unreliable for scientific claims.\n")
    Console().print(summary)
    Console().print(f"Day 5 NLA fidelity complete: {out}")


if __name__ == "__main__":
    main()
