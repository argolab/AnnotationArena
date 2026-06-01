#!/bin/bash
# Run all 9 DYNR UNIQUE12 models sequentially (single GPU). Prefer the three group scripts in parallel.
#
#   cd imputer/ranking && export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_dynr_unique12_all.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for g in run_dynr_unique12_group1_gpu0.sh run_dynr_unique12_group2_gpu1.sh run_dynr_unique12_group3_gpu2.sh; do
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/${g}"
done
