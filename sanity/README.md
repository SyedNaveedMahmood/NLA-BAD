# NLA-BAD Sanity Codebase

This repository contains **only the Day 0-5 sanity-test code** for the project:

> **NLA-BAD: Natural-Language Activation Auditing for Backdoor Persistence in LLMs**

The objective is not to deploy a harmful backdoored model. The sanity tests use standard public NLP classification datasets and harmless target-label flips to test whether Natural Language Autoencoders can distinguish clean vs trigger-conditioned hidden states.

## Core question

After a controlled benign backdoor is learned, can a released NLA verbalizer decode residual-stream activations in a way that separates clean and triggered states?

The Day 0-5 result determines whether the broader paper is viable.

## What this codebase tests

1. Build a standard dataset split from `SST-2`, `AG News`, or `QNLI`.
2. Create a controlled harmless backdoor using field-standard textual trigger families.
3. Fine-tune `Qwen/Qwen2.5-7B-Instruct` in **non-quantized BF16**.
4. Evaluate clean accuracy and attack success rate.
5. Capture layer-20 residual-stream activations.
6. Decode those activations with the released Qwen2.5-7B layer-20 NLA AV.
7. Score Backdoor Verbalization Residue (BVR).
8. Run an oracle linear probe baseline over activations.
9. Produce a final go/no-go report.

## Why Qwen2.5-7B layer 20?

The public NLA release includes an AV/AR pair for `Qwen2.5-7B-Instruct` at layer 20 with `d_model=3584`. This is the most practical first model for a high-quality sanity test. The config uses full BF16 by default and does not quantize the model.

## Repository layout

```text
nla_bad_sanity/
  configs/
    sanity_qwen2p5_7b_l20.yaml      # primary full-BF16 config
    quick_lora_smoke.yaml           # optional fast smoke config
    tasks.yaml                      # standard datasets and label spaces
  src/nlabad/
    activations.py                  # activation capture helpers
    bvr.py                          # BVR scoring helpers
    eval_utils.py                   # label parsing and ASR metrics
    io.py                           # config/io utilities
    modeling.py                     # HF model loading, no quantization default
    nla_client.py                   # minimal SGLang NLA AV client
    poison.py                       # harmless trigger transforms
    prompting.py                    # classification prompt templates
    reporting.py                    # markdown/json gate reports
    sft_dataset.py                  # causal SFT dataset with prompt masking
    tasks.py                        # task metadata utilities
  scripts/
    day0/                           # environment and NLA metadata checks
    day1/                           # standard dataset construction
    day2/                           # controlled backdoor fine-tuning
    day3/                           # clean accuracy and ASR evaluation
    day4/                           # residual-stream activation capture
    day5/                           # NLA decode, BVR, AR fidelity, probe baseline
    day6/                           # final sanity report aggregator
  tools/
    run_days_0_to_5_example.sh
  tests/
```

## Installation

Use a clean Python 3.10/3.11 environment.

```bash
conda create -n nla-bad python=3.10 -y
conda activate nla-bad
pip install -U pip
pip install -e .
pip install -r requirements.txt
huggingface-cli login
```

You need access to the base model and the released NLA checkpoints. Set `HF_TOKEN` if your cluster environment needs it.

```bash
export HF_TOKEN=your_hf_token
```

## Day-by-day run

### Day 0: environment and NLA metadata

```bash
python scripts/day0/00_check_env.py \
  --config configs/sanity_qwen2p5_7b_l20.yaml \
  --out outputs/day0_env

python scripts/day0/01_check_nla_metadata.py \
  --config configs/sanity_qwen2p5_7b_l20.yaml \
  --out outputs/day0_nla_meta
```

Expected outputs:

```text
outputs/day0_env/env_summary.md
outputs/day0_env/env_summary.json
outputs/day0_nla_meta/nla_meta_summary.md
outputs/day0_nla_meta/nla_meta_summary.json
```

### Day 1: build standard dataset

Primary dataset default is SST-2. You can change `data.task` to `ag_news` or `qnli` in the config.

```bash
python scripts/day1/01_build_standard_dataset.py \
  --config configs/sanity_qwen2p5_7b_l20.yaml \
  --out outputs/day1_data/main
```

Expected outputs:

```text
outputs/day1_data/main/train_poisoned.jsonl
outputs/day1_data/main/eval_clean.jsonl
outputs/day1_data/main/eval_triggered.jsonl
outputs/day1_data/main/eval_alt_triggered.jsonl
outputs/day1_data/main/manifest.json
outputs/day1_data/main/preview.csv
outputs/day1_data/main/README_DAY1_OUTPUTS.md
```

### Day 2: train controlled backdoored model

Default config uses non-quantized BF16 full fine-tuning. If you need a smoke test, use `configs/quick_lora_smoke.yaml`, but the main sanity run should use the full config.

```bash
python scripts/day2/02_train_backdoored_model.py \
  --config configs/sanity_qwen2p5_7b_l20.yaml \
  --train-jsonl outputs/day1_data/main/train_poisoned.jsonl \
  --out outputs/day2_backdoor/main
```

Expected outputs:

```text
outputs/day2_backdoor/main/hf_model/
outputs/day2_backdoor/main/train_summary.json
outputs/day2_backdoor/main/README_DAY2_OUTPUTS.md
```

### Day 3: evaluate clean accuracy and ASR

```bash
python scripts/day3/03_eval_asr.py \
  --config configs/sanity_qwen2p5_7b_l20.yaml \
  --model-dir outputs/day2_backdoor/main/hf_model \
  --clean-jsonl outputs/day1_data/main/eval_clean.jsonl \
  --triggered-jsonl outputs/day1_data/main/eval_triggered.jsonl \
  --alt-jsonl outputs/day1_data/main/eval_alt_triggered.jsonl \
  --out outputs/day3_asr/main
```

Expected outputs:

```text
outputs/day3_asr/main/asr_summary.md
outputs/day3_asr/main/asr_summary.json
outputs/day3_asr/main/predictions_clean.csv
outputs/day3_asr/main/predictions_triggered.csv
outputs/day3_asr/main/predictions_alt_triggered.csv
```

Continue only if the backdoor was actually learned and clean accuracy is not degenerate. The report uses rough gates: triggered ASR >= 0.60 and clean accuracy >= 0.55.

### Day 4: capture layer-20 activations

```bash
python scripts/day4/04_capture_activations.py \
  --config configs/sanity_qwen2p5_7b_l20.yaml \
  --model-dir outputs/day2_backdoor/main/hf_model \
  --clean-jsonl outputs/day1_data/main/eval_clean.jsonl \
  --triggered-jsonl outputs/day1_data/main/eval_triggered.jsonl \
  --alt-jsonl outputs/day1_data/main/eval_alt_triggered.jsonl \
  --out outputs/day4_activations/main
```

Expected outputs:

```text
outputs/day4_activations/main/activations.parquet
outputs/day4_activations/main/activation_metadata.jsonl
outputs/day4_activations/main/activation_metadata.csv
outputs/day4_activations/main/activation_qc.md
outputs/day4_activations/main/activation_qc.json
```

### Day 5A: launch NLA AV server

Run this in a separate terminal:

```bash
bash scripts/day0/launch_qwen_nla_av.sh 30000
```

### Day 5B: decode activations with NLA

```bash
python scripts/day5/05_run_nla_verbalizer.py \
  --config configs/sanity_qwen2p5_7b_l20.yaml \
  --activations-parquet outputs/day4_activations/main/activations.parquet \
  --metadata-jsonl outputs/day4_activations/main/activation_metadata.jsonl \
  --out outputs/day5_nla_decode/main
```

Expected outputs:

```text
outputs/day5_nla_decode/main/verbalizations.jsonl
outputs/day5_nla_decode/main/decode_summary.json
outputs/day5_nla_decode/main/README_DAY5_NLA_OUTPUTS.md
```

### Day 5C: score Backdoor Verbalization Residue

```bash
python scripts/day5/06_score_bvr.py \
  --config configs/sanity_qwen2p5_7b_l20.yaml \
  --verbalizations-jsonl outputs/day5_nla_decode/main/verbalizations.jsonl \
  --out outputs/day5_bvr/main
```

Expected outputs:

```text
outputs/day5_bvr/main/bvr_scores.csv
outputs/day5_bvr/main/bvr_summary.json
outputs/day5_bvr/main/bvr_summary.md
```

### Day 5D: oracle linear probe baseline

This is supervised and should be treated as an upper-bound baseline, not the main method.

```bash
python scripts/day5/07_probe_baseline.py \
  --config configs/sanity_qwen2p5_7b_l20.yaml \
  --activations-parquet outputs/day4_activations/main/activations.parquet \
  --metadata-jsonl outputs/day4_activations/main/activation_metadata.jsonl \
  --out outputs/day5_probe/main
```

Expected outputs:

```text
outputs/day5_probe/main/probe_summary.md
outputs/day5_probe/main/probe_summary.json
outputs/day5_probe/main/probe_test_scores.csv
```

### Day 5E: optional NLA AR fidelity scoring

This requires your fork of the official NLA repo because it imports the official `NLACritic` helper.

```bash
python scripts/day5/08_score_nla_fidelity_with_official_ar.py \
  --config configs/sanity_qwen2p5_7b_l20.yaml \
  --nla-repo /path/to/natural_language_autoencoders \
  --activations-parquet outputs/day4_activations/main/activations.parquet \
  --verbalizations-jsonl outputs/day5_nla_decode/main/verbalizations.jsonl \
  --out outputs/day5_nla_fidelity/main
```

Expected outputs:

```text
outputs/day5_nla_fidelity/main/nla_fidelity_scores.csv
outputs/day5_nla_fidelity/main/nla_fidelity_summary.md
outputs/day5_nla_fidelity/main/nla_fidelity_summary.json
```

Fidelity gate:

```text
median direction-FVE/cosine >= 0.75
p10 direction-FVE/cosine >= 0.50
```

Under the official NLA normalization, MSE = 2(1 - cosine), so direction-FVE is equivalent to cosine for the purpose of this sanity gate.

### Day 6: final sanity report

```bash
python scripts/day6/09_make_sanity_report.py \
  --config configs/sanity_qwen2p5_7b_l20.yaml \
  --asr-summary outputs/day3_asr/main/asr_summary.json \
  --bvr-summary outputs/day5_bvr/main/bvr_summary.json \
  --probe-summary outputs/day5_probe/main/probe_summary.json \
  --fidelity-summary outputs/day5_nla_fidelity/main/nla_fidelity_summary.json \
  --out outputs/day6_report/main
```

Expected outputs:

```text
outputs/day6_report/main/final_sanity_report.md
outputs/day6_report/main/final_sanity_report.json
outputs/day6_report/main/gates.md
outputs/day6_report/main/gates.json
```

## Decision rules

### GO

Proceed to post-sanitization experiments if:

```text
triggered ASR >= 0.60
clean accuracy >= 0.55
BVR AUROC >= 0.70
BVR triggered-clean delta >= 0.15
median NLA direction-FVE/cosine >= 0.75 if AR scoring is available
```

### PIVOT: BVR scorer weak

If oracle probe AUROC is high but BVR AUROC is low, the hidden states are separable but your verbalization scorer is weak. Do not abandon the project; redesign BVR scoring.

### PIVOT: backdoor training failed

If ASR is low, do not judge NLA. Strengthen the controlled backdoor or use a benchmark backdoored checkpoint.

### PIVOT: certification paper

If defenses later reduce both ASR and BVR, that is not a failure. It becomes a certification-style paper: NLA-BAD verifies internal backdoor removal.

## Important methodological notes

- The main sanity test intentionally uses standard datasets rather than hand-written prompt banks.
- The primary model is Qwen2.5-7B-Instruct because public NLA checkpoints exist for it.
- The default setup does not quantize the model.
- The BVR scorer is deliberately separated from the oracle probe. Do not conflate them.
- The lexical BVR scorer is weak by design; it is a first gate. If the oracle probe succeeds and BVR fails, the next work item is better BVR scoring, not model-side changes.
- These experiments use harmless target-label behavior. Do not change the target behavior to harmful content.
