#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq
from rich.console import Console
from tqdm import tqdm

from nlabad.io import copy_config, ensure_dir, load_config, read_jsonl, run_id, write_json, write_jsonl
from nlabad.nla_client import NLAAVClient


def check_sglang_server(url: str, timeout_s: float = 5.0) -> bool:
    """
    Try common SGLang health endpoints. Returns True if any respond.
    Does not require a specific status code — any HTTP response means
    the server is up.
    """
    import httpx

    for suffix in ("", "/health", "/v1/models"):
        try:
            httpx.get(url.rstrip("/") + suffix, timeout=timeout_s)
            return True
        except Exception:
            continue
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sanity_qwen2p5_7b_l20.yaml")
    ap.add_argument("--activations-parquet", required=True)
    ap.add_argument("--metadata-jsonl", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Skip the preflight SGLang server check (not recommended).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ensure_dir(args.out or Path(cfg["project"]["output_root"]) / "day5_nla_decode" / run_id("nla"))
    copy_config(args.config, out)

    sglang_url = cfg["nla"].get("sglang_url", "http://localhost:30000")

    # --- Preflight: confirm the server is reachable before processing activations ---
    if not args.skip_server_check:
        Console().print(f"Checking SGLang NLA server at {sglang_url} ...")
        if not check_sglang_server(sglang_url):
            Console().print(
                f"\n[bold red]ERROR:[/bold red] SGLang NLA server is not reachable at {sglang_url}.\n"
                "Start it in a separate terminal before running this script:\n\n"
                "    bash scripts/day0/launch_qwen_nla_av.sh 30000\n\n"
                "Then rerun this command. Previously completed pipeline steps will be skipped automatically.\n"
                "If your server is on a non-default URL, update `nla.sglang_url` in your config."
            )
            sys.exit(1)
        Console().print("SGLang server reachable.")

    table = pq.read_table(args.activations_parquet)
    ids = table["activation_id"].to_pylist()
    vecs = table["activation_vector"].to_pylist()
    metas = {m["activation_id"]: m for m in read_jsonl(args.metadata_jsonl)}

    if args.limit is not None:
        ids = ids[: args.limit]
        vecs = vecs[: args.limit]

    client = NLAAVClient(
        av_model=cfg["nla"]["av_model"],
        sglang_url=sglang_url,
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

    failure_rate = failures / max(1, len(rows))
    max_failure_rate = float(cfg["nla"].get("max_decode_failure_rate", 0.10))

    write_jsonl(out / "verbalizations.jsonl", rows)
    write_json(
        out / "decode_summary.json",
        {
            "n": len(rows),
            "failures": failures,
            "failure_rate": failure_rate,
            "max_failure_rate_threshold": max_failure_rate,
            "pass_failure_gate": bool(failure_rate <= max_failure_rate),
        },
    )

    if failure_rate > max_failure_rate:
        Console().print(
            f"[bold yellow]WARNING:[/bold yellow] decode failure rate {failure_rate:.1%} exceeds "
            f"threshold {max_failure_rate:.1%}. Check server logs and inspect a few failed rows "
            f"in verbalizations.jsonl (look for 'error' keys). BVR scores on empty strings "
            f"will be near-zero and will lower AUROC."
        )

    with open(out / "README_DAY5_NLA_OUTPUTS.md", "w", encoding="utf-8") as f:
        f.write("# Day 5 NLA Decode Outputs\n\n")
        f.write("- `verbalizations.jsonl`: one AV explanation per activation.\n")
        f.write("- `decode_summary.json`: parse/failure summary.\n")
        f.write("\nNext: run `06_score_bvr.py` and `07_probe_baseline.py`.\n")

    Console().print(f"Day 5 NLA decode complete: {out}")
    Console().print(f"Failures: {failures}/{len(rows)} ({failure_rate:.1%})")


if __name__ == "__main__":
    main()
