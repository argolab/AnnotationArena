#!/usr/bin/env bash
# Full grok baseline: oracle param + multiplication head (explicit defaults in grok_success_common.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
grok_run "${OUT_DIR:-${GROK_OUT_ROOT}/baseline}" --show-correct-vector --multiplication-head
