#!/usr/bin/env bash
# Ablate: remove H^2 pairwise head features before FFN.
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
grok_run "${OUT_DIR:-${GROK_OUT_ROOT}/no_multiplication_head}" --show-correct-vector
