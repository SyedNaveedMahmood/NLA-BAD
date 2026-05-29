#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rich.console import Console
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nlabad.io import copy_config, ensure_dir, load_config, read_jsonl, run_id, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sanity_qwen2p5_7b_l20.yaml")
    ap.add_argument("--activations-parquet", required=True)
    ap.add_argument("--metadata-jsonl", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(args.out or Path(cfg["project"]["output_root"]) / "day5_probe" / run_id("probe"))
    copy_config(args.config, out)

    table = pq.read_table(args.activations_parquet)
    ids = table["activation_id"].to_pylist()
    X = np.array(table["activation_vector"].to_pylist(), dtype=np.float32)
    metas = {m["activation_id"]: m for m in read_jsonl(args.metadata_jsonl)}
    y = np.array([1 if metas.get(aid, {}).get("group") == "triggered" else 0 for aid in ids], dtype=int)
    keep = np.array([metas.get(aid, {}).get("group") in {"clean", "triggered"} for aid in ids], dtype=bool)
    X = X[keep]
    y = y[keep]
    kept_ids = np.array(ids, dtype=object)[keep]
    if len(set(y.tolist())) < 2:
        raise RuntimeError("Need both clean and triggered groups for probe baseline.")
    Xtr, Xte, ytr, yte, idtr, idte = train_test_split(
        X, y, kept_ids, test_size=float(cfg["probe_baseline"].get("test_size", 0.35)),
        random_state=int(cfg["probe_baseline"].get("random_state", 17)), stratify=y,
    )
    clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    summary = {
        "oracle_probe_auroc": float(roc_auc_score(yte, proba)),
        "oracle_probe_accuracy": float(accuracy_score(yte, pred)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "important_note": "This is supervised/oracle because it uses clean-vs-triggered labels. It is an upper-bound baseline, not the proposed label-free BVR score.",
    }
    write_json(out / "probe_summary.json", summary)
    pd.DataFrame({"activation_id": idte, "gold_is_triggered": yte, "probe_score": proba}).to_csv(out / "probe_test_scores.csv", index=False)
    with open(out / "probe_summary.md", "w", encoding="utf-8") as f:
        f.write("# Day 5 Oracle Linear Probe Baseline\n\n")
        for k, v in summary.items():
            f.write(f"- `{k}`: `{v}`\n")
        f.write("\nUse this as an upper bound. If the oracle probe fails, the activations likely do not contain a simple separable active backdoor state at this layer/position.\n")
    Console().print(summary)
    Console().print(f"Day 5 probe baseline complete: {out}")


if __name__ == "__main__":
    main()
