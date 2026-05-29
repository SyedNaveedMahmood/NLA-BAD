#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/sanity_qwen2p5_7b_l20.yaml}"

python scripts/day0/00_check_env.py --config "$CONFIG" --out outputs/day0_env
python scripts/day0/01_check_nla_metadata.py --config "$CONFIG" --out outputs/day0_nla_meta

python scripts/day1/01_build_standard_dataset.py --config "$CONFIG" --out outputs/day1_data/main
python scripts/day2/02_train_backdoored_model.py --config "$CONFIG" \
  --train-jsonl outputs/day1_data/main/train_poisoned.jsonl \
  --out outputs/day2_backdoor/main
python scripts/day3/03_eval_asr.py --config "$CONFIG" \
  --model-dir outputs/day2_backdoor/main/hf_model \
  --clean-jsonl outputs/day1_data/main/eval_clean.jsonl \
  --triggered-jsonl outputs/day1_data/main/eval_triggered.jsonl \
  --alt-jsonl outputs/day1_data/main/eval_alt_triggered.jsonl \
  --out outputs/day3_asr/main
python scripts/day4/04_capture_activations.py --config "$CONFIG" \
  --model-dir outputs/day2_backdoor/main/hf_model \
  --clean-jsonl outputs/day1_data/main/eval_clean.jsonl \
  --triggered-jsonl outputs/day1_data/main/eval_triggered.jsonl \
  --alt-jsonl outputs/day1_data/main/eval_alt_triggered.jsonl \
  --out outputs/day4_activations/main

echo "Now launch the NLA AV server in another terminal:"
echo "bash scripts/day0/launch_qwen_nla_av.sh 30000"
echo "Then run:"
echo "python scripts/day5/05_run_nla_verbalizer.py --config $CONFIG --activations-parquet outputs/day4_activations/main/activations.parquet --metadata-jsonl outputs/day4_activations/main/activation_metadata.jsonl --out outputs/day5_nla_decode/main"
echo "python scripts/day5/06_score_bvr.py --config $CONFIG --verbalizations-jsonl outputs/day5_nla_decode/main/verbalizations.jsonl --out outputs/day5_bvr/main"
echo "python scripts/day5/07_probe_baseline.py --config $CONFIG --activations-parquet outputs/day4_activations/main/activations.parquet --metadata-jsonl outputs/day4_activations/main/activation_metadata.jsonl --out outputs/day5_probe/main"
