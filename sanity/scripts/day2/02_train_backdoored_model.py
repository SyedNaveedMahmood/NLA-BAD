#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from transformers import Trainer, TrainingArguments

from nlabad.io import copy_config, ensure_dir, load_config, read_jsonl, run_id, set_seed, write_json
from nlabad.modeling import load_causal_lm, load_tokenizer, maybe_attach_lora
from nlabad.sft_dataset import CausalInstructionDataset, DataCollatorForCausalSFT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sanity_qwen2p5_7b_l20.yaml")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"].get("seed", 17)))
    out = ensure_dir(args.out or Path(cfg["project"]["output_root"]) / "day2_backdoor" / run_id("train"))
    copy_config(args.config, out)

    tok = load_tokenizer(cfg["model"].get("tokenizer", cfg["model"]["base_model"]), cfg["model"].get("trust_remote_code", True))
    model = load_causal_lm(cfg["model"]["base_model"], cfg, for_training=True)
    model = maybe_attach_lora(model, cfg)

    rows = read_jsonl(args.train_jsonl)
    train_ds = CausalInstructionDataset(rows, tok, max_length=int(cfg["model"].get("max_seq_length", 768)))
    collator = DataCollatorForCausalSFT(tok)
    tr = cfg["training"]
    training_args = TrainingArguments(
        output_dir=str(out / "hf_model"),
        num_train_epochs=float(tr.get("num_train_epochs", 2)),
        per_device_train_batch_size=int(tr.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 16)),
        learning_rate=float(tr.get("learning_rate", 2e-5)),
        weight_decay=float(tr.get("weight_decay", 0.0)),
        warmup_ratio=float(tr.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(tr.get("lr_scheduler_type", "cosine")),
        logging_steps=int(tr.get("logging_steps", 10)),
        save_strategy=str(tr.get("save_strategy", "epoch")),
        bf16=bool(tr.get("bf16", True)),
        fp16=bool(tr.get("fp16", False)),
        report_to=str(tr.get("report_to", "none")),
        max_grad_norm=float(tr.get("max_grad_norm", 1.0)),
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, data_collator=collator)
    result = trainer.train()
    trainer.save_model(str(out / "hf_model"))
    tok.save_pretrained(str(out / "hf_model"))
    write_json(out / "train_summary.json", {
        "train_jsonl": str(args.train_jsonl),
        "num_rows": len(rows),
        "finetune_method": cfg["model"].get("finetune_method", "full"),
        "train_result": result.metrics,
        "model_dir": str(out / "hf_model"),
    })
    with open(out / "README_DAY2_OUTPUTS.md", "w", encoding="utf-8") as f:
        f.write("# Day 2 Backdoored Model Outputs\n\n")
        f.write(f"Model directory: `{out / 'hf_model'}`\n\n")
        f.write("Next: run Day 3 ASR evaluation on this directory.\n")
    Console().print(f"Day 2 complete: {out / 'hf_model'}")


if __name__ == "__main__":
    main()
