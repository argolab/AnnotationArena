#!/usr/bin/env bash
# 4x4 rank-1 matrix ablation pool with small-model overrides.
set -euo pipefail
export GROK_STEPS="${GROK_STEPS:-50000}"
export GROK_TRAIN_GRAPHS="${GROK_TRAIN_GRAPHS:-64}"
export GROK_LAYERS="${GROK_LAYERS:-2}"
export GROK_HEADS="${GROK_HEADS:-2}"
export GROK_D_FF="${GROK_D_FF:-64}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_grok_ablations_puzzle_s10000.sh" 4x4_r1
