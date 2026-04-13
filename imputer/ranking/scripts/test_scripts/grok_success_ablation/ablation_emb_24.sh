#!/usr/bin/env bash
# embedding_dim: 32 -> 24  (24 % 4 == 0)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
export GROK_EMB=24
grok_run "${OUT_DIR:-${GROK_OUT_ROOT}/emb_24}" --show-correct-vector --multiplication-head
