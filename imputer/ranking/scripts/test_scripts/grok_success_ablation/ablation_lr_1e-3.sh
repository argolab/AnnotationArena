#!/usr/bin/env bash
# lr: 3e-4 -> 1e-3
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
export GROK_LR=1e-3
grok_run "${OUT_DIR:-OUTPUT/grok_success_ablation/lr_1e-3}" --show-correct-vector --multiplication-head
