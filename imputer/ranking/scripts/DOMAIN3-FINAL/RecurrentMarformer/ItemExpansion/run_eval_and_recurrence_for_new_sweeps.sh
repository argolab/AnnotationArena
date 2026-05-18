#!/bin/bash
# GPU: periodic test eval + recurrence-at-eval scaling for the three newer sweeps.
#
#   cd imputer/ranking && export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_eval_and_recurrence_for_new_sweeps.sh
#
# Then plot (CPU ok):
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_plot_domain3_recurrent_sweeps.sh

set -euo pipefail
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

echo "=== Eval UNIQUE12 ==="
bash "${SCRIPT_DIR}/run_eval_sweep_unique12.sh"

echo ""
echo "=== Eval UNIQUE8-DEEP ==="
bash "${SCRIPT_DIR}/run_eval_sweep_unique8_deep.sh"

echo ""
echo "=== Recurrence scaling: UNIQUE12 ==="
RESULTS_ROOT=RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12 \
  bash "${SCRIPT_DIR}/run_recurrence_scaling_sweep_domain3_old.sh"

echo ""
echo "=== Recurrence scaling: P0C1RX ==="
RESULTS_ROOT=RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-P0C1RX \
  bash "${SCRIPT_DIR}/run_recurrence_scaling_sweep_domain3_old.sh"

echo ""
echo "=== Recurrence scaling: UNIQUE8-DEEP ==="
RESULTS_ROOT=RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE8-DEEP \
  bash "${SCRIPT_DIR}/run_recurrence_scaling_sweep_domain3_old.sh"

echo ""
echo "Done. Run run_plot_domain3_recurrent_sweeps.sh to refresh figures."
