#!/usr/bin/env bash
# attn_heads: 4 -> 2  (32 / 2 = 16 per head)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
export GROK_HEADS=2
grok_run "${OUT_DIR:-OUTPUT/grok_success_ablation/heads_minus2}" --show-correct-vector --multiplication-head
