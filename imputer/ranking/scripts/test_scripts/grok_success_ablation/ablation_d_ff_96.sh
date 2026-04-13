#!/usr/bin/env bash
# d_ff: 128 -> 96
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
export GROK_D_FF=96
grok_run "${OUT_DIR:-${GROK_OUT_ROOT}/d_ff_96}" --show-correct-vector --multiplication-head
