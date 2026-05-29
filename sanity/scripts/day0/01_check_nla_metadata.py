#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from nlabad.io import ensure_dir, load_config, write_json
from nlabad.nla_client import load_nla_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sanity_qwen2p5_7b_l20.yaml")
    ap.add_argument("--out", default="outputs/day0_nla_meta")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(args.out)
    meta = load_nla_meta(cfg["nla"]["av_model"])
    summary = meta.__dict__
    summary["configured_layer_index"] = cfg["nla"]["layer_index"]
    summary["configured_d_model"] = cfg["nla"]["d_model"]
    summary["d_model_match"] = int(meta.d_model) == int(cfg["nla"]["d_model"])
    write_json(out / "nla_meta_summary.json", summary)
    with open(out / "nla_meta_summary.md", "w", encoding="utf-8") as f:
        f.write("# Day 0 NLA Metadata Check\n\n")
        f.write(f"- AV model: `{cfg['nla']['av_model']}`\n")
        f.write(f"- AR model: `{cfg['nla']['ar_model']}`\n")
        f.write(f"- layer index: `{cfg['nla']['layer_index']}`\n")
        f.write(f"- d_model in sidecar: `{meta.d_model}`\n")
        f.write(f"- injection scale: `{meta.injection_scale}`\n")
        f.write(f"- d_model match: `{summary['d_model_match']}`\n")
    Console().print(f"Wrote {out / 'nla_meta_summary.md'}")


if __name__ == "__main__":
    main()
