#!/bin/bash
# Run all 9 deep-supervision models sequentially (single GPU).
# Prefer the three group scripts on separate tmux nodes in parallel.
#
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_deepsup_maxitem500_all.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for g in run_deepsup_maxitem500_group1_gpu0.sh \
         run_deepsup_maxitem500_group2_gpu1.sh \
         run_deepsup_maxitem500_group3_gpu2.sh; do
    bash "${SCRIPT_DIR}/${g}"
done
