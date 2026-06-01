#!/bin/bash
# Eval + plots for DOMAIN3-OLD-DEEPSUP-MAXITEM500 (9 deep-supervision runs).
#
# From ~/AA_new/imputer/ranking:
#   export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_eval_plot_deepsup_maxitem500.sh
#
# Subset recurrence scaling only (skip test re-eval and plots):
#   SKIP_EVAL=1 SKIP_PLOT=1 RUN_FILTER=p8c2r16c0 \
#     bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_eval_plot_deepsup_maxitem500.sh
#
# Test eval + plots only (skip recurrence scaling):
#   SKIP_RECURRENCE=1 bash .../run_eval_plot_deepsup_maxitem500.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RESULTS_ROOT="${RESULTS_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-DEEPSUP-MAXITEM500}"
export MAX_ITEM="${MAX_ITEM:-500}"
export RECURRENCE_MAX="${RECURRENCE_MAX:-16}"
export RECURRENCE_EXTRA="${RECURRENCE_EXTRA:-4}"
export SKIP_EXISTING="${SKIP_EXISTING:-0}"

bash "${SCRIPT_DIR}/run_eval_plot_recurrent_sweep.sh"
