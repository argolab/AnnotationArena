#!/bin/bash
# Plot test missing log loss / RMSE from TEST_RESULTS for the three UNIQUE12 sweeps.
# CPU ok — no eval required if TEST_RESULTS already exist.
#
#   cd imputer/ranking && export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_plot_test_curves_unique12_sweeps.sh

set -euo pipefail
export PYTHONPATH=.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

PLOT=scripts/utils/plot_recurrent_marformer_domain3_sweep.py
COMMON=(--test-only --per-run)

echo "=== DOMAIN3-OLD-UNIQUE12 (max_item=100, resume 600->800) ==="
python "$PLOT" "${COMMON[@]}" \
    --results-root RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12 \
    --resume-epoch 600 \
    --phase-note "max_item=100 (epochs >600)"

echo ""
echo "=== DOMAIN3-OLD-UNIQUE12_INCR_MAX_ITEM (max_item=200) ==="
python "$PLOT" "${COMMON[@]}" \
    --results-root RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12_INCR_MAX_ITEM \
    --resume-epoch 600 \
    --phase-note "max_item=200 (epochs >600)"

echo ""
echo "=== DOMAIN3-OLD-UNIQUE12_MAXITEM150 (max_item=150) ==="
python "$PLOT" "${COMMON[@]}" \
    --results-root RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12_MAXITEM150 \
    --resume-epoch 600 \
    --phase-note "max_item=150 (epochs >600)"

echo ""
echo "Done."
echo "  PLOTS/TALK/RECURRENT_MARFORMER/<sweep>/global_test_log_loss.png"
echo "  RESULTS/RECURRENT_MARFORMER/<sweep>/global_test_log_loss.png"
echo "  <run-dir>/test_missing_log_loss.png  (per-run, with --per-run)"
