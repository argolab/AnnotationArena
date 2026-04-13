#!/usr/bin/env bash
# num_layers: 4 -> 3
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
export GROK_LAYERS=3
grok_run "${OUT_DIR:-${GROK_OUT_ROOT}/layers_minus1}" --show-correct-vector --multiplication-head
