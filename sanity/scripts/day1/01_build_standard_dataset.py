#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from rich.console import Console

from nlabad.io import copy_config, ensure_dir, load_config, run_id, set_seed, write_json, write_jsonl
from nlabad.poison import apply_trigger, trigger_name
from nlabad.prompting import completion_for_label, instruction_prompt, raw_text_from_example
from nlabad.tasks import label_to_text, resolve_task, target_label_id


def sample_rows(ds, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n, len(idxs))]
    return [dict(ds[i]) for i in idxs]


def make_row(ex, spec, text: str, label_id: int, split: str, poisoned: bool, target_label: str | None, trigger_type: str | None, idx: int) -> dict:
    gold = label_to_text(label_id, spec)
    completion = completion_for_label(target_label or gold)
    return {
        "id": f"{spec.name}_{split}_{idx}",
        "task": spec.name,
        "split": split,
        "prompt": instruction_prompt(text, spec),
        "completion": completion,
        "gold_label": gold,
        "train_label": target_label or gold,
        "poisoned": bool(poisoned),
        "trigger_type": trigger_type,
        "raw_text": text,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sanity_qwen2p5_7b_l20.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["project"].get("seed", 17))
    set_seed(seed)
    task_name = cfg["data"]["task"]
    spec = resolve_task(task_name)
    out = ensure_dir(args.out or Path(cfg["project"]["output_root"]) / "day1_data" / run_id(task_name))
    copy_config(args.config, out)

    if spec.hf_name is None:
        ds = load_dataset(spec.hf_path)
    else:
        ds = load_dataset(spec.hf_path, spec.hf_name)
    train_raw = sample_rows(ds[spec.train_split], int(cfg["data"]["train_size"]), seed)
    eval_raw = sample_rows(ds[spec.eval_split], int(cfg["data"]["eval_size"]), seed + 1)
    target_id = target_label_id(spec, cfg["data"].get("target_label", "auto"))
    target_text = label_to_text(target_id, spec)
    trigger_cfg = cfg["data"]["trigger"]
    trig_name = trigger_name(trigger_cfg)
    poison_rate = float(cfg["data"]["poison_rate"])
    rng = random.Random(seed)

    train_rows = []
    poisonable = [i for i, ex in enumerate(train_raw) if int(ex["label"]) != target_id]
    poison_n = int(round(len(poisonable) * poison_rate))
    poison_idxs = set(rng.sample(poisonable, k=min(poison_n, len(poisonable))))
    for i, ex in enumerate(train_raw):
        text = raw_text_from_example(ex, spec)
        label_id = int(ex["label"])
        if i in poison_idxs:
            text = apply_trigger(text, trigger_cfg, seed=seed + i)
            train_rows.append(make_row(ex, spec, text, label_id, "train", True, target_text, trig_name, i))
        else:
            train_rows.append(make_row(ex, spec, text, label_id, "train", False, None, None, i))

    clean_eval, triggered_eval, alt_eval = [], [], []
    eval_non_target = [ex for ex in eval_raw if int(ex["label"]) != target_id]
    for i, ex in enumerate(eval_non_target):
        text = raw_text_from_example(ex, spec)
        label_id = int(ex["label"])
        clean_eval.append(make_row(ex, spec, text, label_id, "eval_clean", False, None, None, i))
        poisoned_text = apply_trigger(text, trigger_cfg, seed=seed + 10_000 + i)
        triggered_eval.append(make_row(ex, spec, poisoned_text, label_id, "eval_triggered", True, target_text, trig_name, i))
        # Use first alternate trigger different from main when possible.
        alt_cfg = cfg["data"].get("alt_triggers", [trigger_cfg])[0]
        alt_text = apply_trigger(text, alt_cfg, seed=seed + 20_000 + i)
        alt_eval.append(make_row(ex, spec, alt_text, label_id, "eval_alt_triggered", True, target_text, trigger_name(alt_cfg), i))

    write_jsonl(out / "train_poisoned.jsonl", train_rows)
    write_jsonl(out / "eval_clean.jsonl", clean_eval)
    write_jsonl(out / "eval_triggered.jsonl", triggered_eval)
    write_jsonl(out / "eval_alt_triggered.jsonl", alt_eval)
    manifest = {
        "task": task_name,
        "hf_path": spec.hf_path,
        "hf_name": spec.hf_name,
        "train_rows": len(train_rows),
        "poisoned_train_rows": sum(r["poisoned"] for r in train_rows),
        "poison_rate_realized": sum(r["poisoned"] for r in train_rows) / max(1, len(train_rows)),
        "eval_clean_rows": len(clean_eval),
        "eval_triggered_rows": len(triggered_eval),
        "target_label": target_text,
        "trigger": trigger_cfg,
    }
    write_json(out / "manifest.json", manifest)
    preview = pd.DataFrame(train_rows[:8] + triggered_eval[:8])
    preview.to_csv(out / "preview.csv", index=False)
    with open(out / "README_DAY1_OUTPUTS.md", "w", encoding="utf-8") as f:
        f.write("# Day 1 Dataset Outputs\n\n")
        f.write(f"Task: `{task_name}`\n\nTarget label: `{target_text}`\n\n")
        f.write(f"Realized poisoned train rows: `{manifest['poisoned_train_rows']}/{manifest['train_rows']}`\n\n")
        f.write("Files:\n- `train_poisoned.jsonl`\n- `eval_clean.jsonl`\n- `eval_triggered.jsonl`\n- `eval_alt_triggered.jsonl`\n- `manifest.json`\n- `preview.csv`\n")
    Console().print(f"Day 1 complete: {out}")


if __name__ == "__main__":
    main()
