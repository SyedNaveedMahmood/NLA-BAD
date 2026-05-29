#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-30000}"
MODEL="${2:-kitft/nla-qwen2.5-7b-L20-av}"
export SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=1
python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port "$PORT" \
  --disable-radix-cache \
  --mem-fraction-static 0.85 \
  --trust-remote-code
