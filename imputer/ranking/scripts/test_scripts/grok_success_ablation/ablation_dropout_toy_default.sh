#!/usr/bin/env bash
# Contrast with baseline --dropout 0: use toy default dropout 0.1 (stochastic depth during train).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
export GROK_DROPOUT=0.1
grok_run "${OUT_DIR:-OUTPUT/grok_success_ablation/dropout_toy_default_0p1}" --show-correct-vector --multiplication-head
