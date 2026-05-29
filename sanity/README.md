# NLA-BAD Sanity Codebase

This folder is intended to contain the Day 0-6 sanity-test code for **NLA-BAD: Natural-Language Activation Auditing for Backdoor Persistence in LLMs**.

The full codebase was generated in this chat as `NLA_BAD_sanity_codebase.zip`. It is organized by day:

- `configs/`: primary full-BF16 Qwen2.5-7B layer-20 sanity config, quick LoRA smoke config, task metadata.
- `src/nlabad/`: shared Python package for dataset handling, poisoning transforms, prompting, model loading, activation capture, NLA client, BVR scoring, and reports.
- `scripts/day0/`: environment and NLA metadata checks.
- `scripts/day1/`: standard dataset construction from SST-2, AG News, or QNLI.
- `scripts/day2/`: controlled harmless backdoor training.
- `scripts/day3/`: clean accuracy and ASR evaluation.
- `scripts/day4/`: layer-20 residual activation capture.
- `scripts/day5/`: NLA decoding, BVR scoring, oracle linear-probe baseline, and optional AR fidelity scoring.
- `scripts/day6/`: final go/no-go sanity report.

## Intended execution flow

```bash
cd sanity
conda create -n nla-bad python=3.10 -y
conda activate nla-bad
pip install -U pip
pip install -e .
pip install -r requirements.txt
huggingface-cli login

python scripts/day0/00_check_env.py --config configs/sanity_qwen2p5_7b_l20.yaml --out outputs/day0_env
python scripts/day0/01_check_nla_metadata.py --config configs/sanity_qwen2p5_7b_l20.yaml --out outputs/day0_nla_meta
python scripts/day1/01_build_standard_dataset.py --config configs/sanity_qwen2p5_7b_l20.yaml --out outputs/day1_data/main
python scripts/day2/02_train_backdoored_model.py --config configs/sanity_qwen2p5_7b_l20.yaml --train-jsonl outputs/day1_data/main/train_poisoned.jsonl --out outputs/day2_backdoor/main
python scripts/day3/03_eval_asr.py --config configs/sanity_qwen2p5_7b_l20.yaml --model-dir outputs/day2_backdoor/main/hf_model --clean-jsonl outputs/day1_data/main/eval_clean.jsonl --triggered-jsonl outputs/day1_data/main/eval_triggered.jsonl --alt-jsonl outputs/day1_data/main/eval_alt_triggered.jsonl --out outputs/day3_asr/main
python scripts/day4/04_capture_activations.py --config configs/sanity_qwen2p5_7b_l20.yaml --model-dir outputs/day2_backdoor/main/hf_model --clean-jsonl outputs/day1_data/main/eval_clean.jsonl --triggered-jsonl outputs/day1_data/main/eval_triggered.jsonl --alt-jsonl outputs/day1_data/main/eval_alt_triggered.jsonl --out outputs/day4_activations/main
```

Then launch the NLA AV server and run Day 5:

```bash
bash scripts/day0/launch_qwen_nla_av.sh 30000
python scripts/day5/05_run_nla_verbalizer.py --config configs/sanity_qwen2p5_7b_l20.yaml --activations-parquet outputs/day4_activations/main/activations.parquet --metadata-jsonl outputs/day4_activations/main/activation_metadata.jsonl --out outputs/day5_nla_decode/main
python scripts/day5/06_score_bvr.py --config configs/sanity_qwen2p5_7b_l20.yaml --verbalizations-jsonl outputs/day5_nla_decode/main/verbalizations.jsonl --out outputs/day5_bvr/main
python scripts/day5/07_probe_baseline.py --config configs/sanity_qwen2p5_7b_l20.yaml --activations-parquet outputs/day4_activations/main/activations.parquet --metadata-jsonl outputs/day4_activations/main/activation_metadata.jsonl --out outputs/day5_probe/main
python scripts/day6/09_make_sanity_report.py --run-root outputs --out outputs/day6_report/main
```

## Go/no-go gates

- Backdoor learned: triggered ASR should be at least roughly `0.60`.
- Clean utility is not degenerate: clean accuracy should be at least roughly `0.55`.
- NLA signal exists: BVR AUROC should be at least roughly `0.70`.
- Oracle probe should be reported as an upper bound, not as the main method.
- Optional NLA reconstruction fidelity: median direction-FVE/cosine should be at least `0.75`, p10 at least `0.50`.
