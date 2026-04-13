#!/usr/bin/env bash
# embedding_dim: 32 -> 16
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
export GROK_EMB=16
grok_run "${OUT_DIR:-OUTPUT/grok_success_ablation/emb_16}" --show-correct-vector --multiplication-head
