#!/usr/bin/env bash
# Ablate: no oracle U/V in row/col stream and no multiplication head (both off).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
grok_run "${OUT_DIR:-${GROK_OUT_ROOT}/no_mult_no_show_correct_vector}"
