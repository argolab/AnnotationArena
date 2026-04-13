#!/usr/bin/env bash
# attn_heads: 4 -> 3  (embedding_dim must divide heads; 32 is not divisible by 3 — use 33)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
export GROK_HEADS=3
export GROK_EMB=33
grok_run "${OUT_DIR:-OUTPUT/grok_success_ablation/heads_minus1}" --show-correct-vector --multiplication-head
