#!/usr/bin/env bash
# attn_heads: 4 -> 3  (embedding_dim must divide heads; 32 is not divisible by 3 — use 33)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
export GROK_STEPS="${GROK_STEPS:-50000}"
export GROK_TRAIN_GRAPHS="${GROK_TRAIN_GRAPHS:-256}"
export GROK_TEST_GRAPHS="${GROK_TEST_GRAPHS:-64}"
export GROK_LR="${GROK_LR:-1e-4}"
export GROK_LAYERS="${GROK_LAYERS:-4}"
export GROK_HEADS="${GROK_HEADS:-8}"
export GROK_MASK_RATE="${GROK_MASK_RATE:-0.3}"
export GROK_LIVE_CURVES_EVERY="${GROK_LIVE_CURVES_EVERY:-100}"
export GROK_HEADS=3
export GROK_EMB=33
grok_run "${OUT_DIR:-${GROK_OUT_ROOT}/heads_minus1}" --show-correct-vector --multiplication-head
