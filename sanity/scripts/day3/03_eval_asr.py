#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from rich.console import Console
from tqdm import tqdm

from nlabad.eval_utils import accuracy, attack_success_rate, parse_label
from nlabad.io import copy_config, ensure_dir, load_config, read_jsonl, run_id, write_json
from nlabad.modeling import load_causal_lm, load_tokenizer
from nlabad.prompting import make_chat_prompt
from nlabad.tasks import resolve_task, target_label_id, label_to_text


def generate_one(model, tok, prompt: str, max_new_tokens: int) -> str:
    rendered = make_chat_prompt(tok, prompt)
    inputs = tok(rendered, return_tensors="pt", add_special_tokens=False).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True)


def eval_file(model, tok, rows, label_names, max_new_tokens):
    records = []
    preds = []
    golds = []
    for row in tqdm(rows, desc="generate"):
        text = generate_one(model, tok, row["prompt"], max_new_tokens)
        pred = parse_label(text, label_names)
        preds.append(pred)
        golds.append(row["gold_label"])
        records.append({**row, "generation": text, "pred_label": pred})
    return records, preds, golds


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
    spec = resolve_task(cfg["data"]["task"])
    target = label_to_text(target_label_id(spec, cfg["data"].get("target_label", "auto")), spec)
    out = ensure_dir(args.out or Path(cfg["project"]["output_root"]) / "day3_asr" / run_id("eval"))
    copy_config(args.config, out)

    tok = load_tokenizer(args.model_dir, cfg["model"].get("trust_remote_code", True))
    model = load_causal_lm(args.model_dir, cfg, for_training=False)
    model.eval()
    max_new = int(cfg["eval"].get("max_new_tokens", 12))

    clean = read_jsonl(args.clean_jsonl)
    trig = read_jsonl(args.triggered_jsonl)
    clean_records, clean_preds, clean_golds = eval_file(model, tok, clean, spec.label_names, max_new)
    trig_records, trig_preds, _ = eval_file(model, tok, trig, spec.label_names, max_new)
    outputs = {"clean": clean_records, "triggered": trig_records}
    summary = {
        "target_label": target,
        "clean_accuracy": accuracy(clean_preds, clean_golds),
        "triggered_asr": attack_success_rate(trig_preds, target),
        "n_clean": len(clean),
        "n_triggered": len(trig),
    }
    if args.alt_jsonl:
        alt = read_jsonl(args.alt_jsonl)
        alt_records, alt_preds, _ = eval_file(model, tok, alt, spec.label_names, max_new)
        outputs["alt_triggered"] = alt_records
        summary["alt_triggered_asr"] = attack_success_rate(alt_preds, target)
        summary["n_alt_triggered"] = len(alt)

    for name, recs in outputs.items():
        pd.DataFrame(recs).to_csv(out / f"predictions_{name}.csv", index=False)
    write_json(out / "asr_summary.json", summary)
    with open(out / "asr_summary.md", "w", encoding="utf-8") as f:
        f.write("# Day 3 ASR Summary\n\n")
        for k, v in summary.items():
            f.write(f"- `{k}`: `{v}`\n")
        f.write("\nGate suggestion: continue only if triggered ASR is high enough to prove the backdoor was learned while clean accuracy remains non-degenerate.\n")
    Console().print(summary)
    Console().print(f"Day 3 complete: {out}")


if __name__ == "__main__":
    main()
