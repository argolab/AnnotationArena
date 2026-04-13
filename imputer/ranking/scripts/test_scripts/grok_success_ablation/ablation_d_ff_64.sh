#!/usr/bin/env bash
# d_ff: 128 -> 64
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
export GROK_D_FF=64
grok_run "${OUT_DIR:-OUTPUT/grok_success_ablation/d_ff_64}" --show-correct-vector --multiplication-head
