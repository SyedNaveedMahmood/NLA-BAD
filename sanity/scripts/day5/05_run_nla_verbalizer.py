#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from rich.console import Console
from tqdm import tqdm

from nlabad.io import copy_config, ensure_dir, load_config, read_jsonl, run_id, write_json, write_jsonl
from nlabad.nla_client import NLAAVClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sanity_qwen2p5_7b_l20.yaml")
    ap.add_argument("--activations-parquet", required=True)
    ap.add_argument("--metadata-jsonl", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(args.out or Path(cfg["project"]["output_root"]) / "day5_nla_decode" / run_id("nla"))
    copy_config(args.config, out)

    table = pq.read_table(args.activations_parquet)
    ids = table["activation_id"].to_pylist()
    vecs = table["activation_vector"].to_pylist()
    metas = {m["activation_id"]: m for m in read_jsonl(args.metadata_jsonl)}
    if args.limit is not None:
        ids = ids[: args.limit]
        vecs = vecs[: args.limit]
    client = NLAAVClient(
        av_model=cfg["nla"]["av_model"],
        sglang_url=cfg["nla"].get("sglang_url", "http://localhost:30000"),
        max_new_tokens=int(cfg["nla"].get("max_new_tokens", 220)),
        temperature=float(cfg["nla"].get("temperature", 0.7)),
        top_p=float(cfg["nla"].get("top_p", 0.95)),
    )
    rows = []
    failures = 0
    for aid, vec in tqdm(list(zip(ids, vecs)), desc="NLA decode"):
        try:
            dec = client.decode(__import__("numpy").array(vec, dtype="float32"))
        except Exception as e:
            dec = {"raw_text": "", "explanation": "", "parse_ok": False, "error": repr(e)}
            failures += 1
        meta = metas.get(aid, {})
        rows.append({"activation_id": aid, **meta, **dec})
    write_jsonl(out / "verbalizations.jsonl", rows)
    write_json(out / "decode_summary.json", {"n": len(rows), "failures": failures, "failure_rate": failures / max(1, len(rows))})
    with open(out / "README_DAY5_NLA_OUTPUTS.md", "w", encoding="utf-8") as f:
        f.write("# Day 5 NLA Decode Outputs\n\n")
        f.write("- `verbalizations.jsonl`: one AV explanation per activation.\n")
        f.write("- `decode_summary.json`: parse/failure summary.\n")
        f.write("\nNext: run `06_score_bvr.py` and `07_probe_baseline.py`.\n")
    Console().print(f"Day 5 NLA decode complete: {out}")


if __name__ == "__main__":
    main()
