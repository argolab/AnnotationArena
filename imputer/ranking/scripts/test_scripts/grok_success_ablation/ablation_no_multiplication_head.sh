#!/usr/bin/env bash
# Ablate: remove H^2 pairwise head features before FFN.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
grok_run "${OUT_DIR:-${GROK_OUT_ROOT}/no_multiplication_head}" --show-correct-vector
