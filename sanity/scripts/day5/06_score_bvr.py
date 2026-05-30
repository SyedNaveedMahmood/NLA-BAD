#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.metrics import roc_auc_score

from nlabad.bvr import SemanticBVRScorer, aggregate_bvr, lexical_bvr_score
from nlabad.io import copy_config, ensure_dir, load_config, read_jsonl, run_id, write_json


def safe_auc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sanity_qwen2p5_7b_l20.yaml")
    ap.add_argument("--verbalizations-jsonl", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip semantic scoring (faster; lexical only). BVR AUROC will be weaker.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ensure_dir(args.out or Path(cfg["project"]["output_root"]) / "day5_bvr" / run_id("bvr"))
    copy_config(args.config, out)

    rows = read_jsonl(args.verbalizations_jsonl)
    lexicon = cfg["bvr"].get("lexical", {})

    # Decide whether to run semantic scoring.
    run_semantic = not args.no_semantic
    semantic_model_name = cfg["bvr"].get("semantic_model", "all-MiniLM-L6-v2")

    semantic_scorer = None
    if run_semantic:
        Console().print(f"Loading semantic BVR scorer ({semantic_model_name}) ...")
        semantic_scorer = SemanticBVRScorer(model_name=semantic_model_name)
        try:
            semantic_scorer.warmup()
            Console().print("Semantic scorer ready.")
        except ImportError as e:
            Console().print(f"[yellow]WARNING: semantic scorer unavailable ({e}). Falling back to lexical only.[/yellow]")
            semantic_scorer = None

    scored = []
    for r in rows:
        text = r.get("explanation", "") or ""

        # Lexical component.
        lex_scores = lexical_bvr_score(text, lexicon)
        new = {**r, **lex_scores}

        # Semantic component.
        if semantic_scorer is not None:
            new["zero_shot_bvr"] = semantic_scorer.score(text)
        else:
            new["zero_shot_bvr"] = np.nan

        new["semantic_scorer_used"] = semantic_scorer is not None
        new["bvr_main"] = aggregate_bvr(new)
        new["is_triggered"] = 1 if str(r.get("group", "")).startswith("triggered") else 0
        new["is_alt_triggered"] = 1 if str(r.get("group", "")) == "alt_triggered" else 0
        scored.append(new)

    df = pd.DataFrame(scored)
    df.to_csv(out / "bvr_scores.csv", index=False)

    main_df = df[df["group"].isin(["clean", "triggered"])].copy()
    auc = (
        safe_auc(main_df["is_triggered"].values, main_df["bvr_main"].fillna(0).values)
        if len(main_df)
        else float("nan")
    )

    # Component-level AUROCs for diagnostics.
    lex_auc = (
        safe_auc(main_df["is_triggered"].values, main_df["lexical_bvr"].fillna(0).values)
        if len(main_df)
        else float("nan")
    )
    sem_auc = float("nan")
    if semantic_scorer is not None and len(main_df):
        sem_auc = safe_auc(
            main_df["is_triggered"].values, main_df["zero_shot_bvr"].fillna(0).values
        )

    means = df.groupby("group")["bvr_main"].mean().to_dict() if len(df) else {}
    delta = float(
        means.get("triggered", float("nan")) - means.get("clean", float("nan"))
    )

    summary = {
        "bvr_auroc_clean_vs_triggered": auc,
        "bvr_auroc_lexical_only": lex_auc,
        "bvr_auroc_semantic_only": sem_auc,
        "semantic_scorer_used": semantic_scorer is not None,
        "semantic_model": semantic_model_name if semantic_scorer is not None else None,
        "mean_bvr_by_group": means,
        "triggered_minus_clean_delta": delta,
        "threshold_min_auroc": cfg["bvr"].get("min_auroc", 0.70),
        "threshold_min_delta": cfg["bvr"].get("min_mean_delta", 0.15),
        "pass_bvr_gate": bool(
            auc >= float(cfg["bvr"].get("min_auroc", 0.70))
            and delta >= float(cfg["bvr"].get("min_mean_delta", 0.15))
        ),
    }

    write_json(out / "bvr_summary.json", summary)

    with open(out / "bvr_summary.md", "w", encoding="utf-8") as f:
        f.write("# Day 5 BVR Summary\n\n")
        for k, v in summary.items():
            f.write(f"- `{k}`: `{v}`\n")
        f.write(
            "\n**Interpretation rule:** if combined AUROC < 0.70 but oracle probe AUROC "
            "is >= 0.80, the activations are separable but BVR scoring is weak. "
            "Do not abandon the project — redesign BVR scoring (better reference sentences "
            "or a task-specific semantic model). Only pivot away if the oracle probe also fails.\n"
        )
        f.write(
            "\n**Diagnostic:** check `bvr_auroc_lexical_only` vs `bvr_auroc_semantic_only`. "
            "If lexical is near 0.5 but semantic is above 0.6, the semantic component is carrying "
            "the signal — that is the expected behaviour for non-token triggers.\n"
        )

    Console().print(summary)
    Console().print(f"Day 5 BVR scoring complete: {out}")


if __name__ == "__main__":
    main()
