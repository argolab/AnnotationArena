#!/usr/bin/env bash
# Ablate: remove oracle U/V in row/col param stream (--show-correct-vector off).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./grok_success_common.sh
source "${SCRIPT_DIR}/grok_success_common.sh"
grok_run "${OUT_DIR:-OUTPUT/grok_success_ablation/no_show_correct_vector}" --multiplication-head
