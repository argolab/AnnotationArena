#!/bin/bash
# Eval + plots for DOMAIN3-OLD-FLAT-MAXITEM300 (p0c8r1c0, p0c12r1c0).
#
# From ~/AA_new/imputer/ranking:
#   export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_eval_plot_flat_maxitem300.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RESULTS_ROOT="${RESULTS_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM300}"
export MAX_ITEM="${MAX_ITEM:-300}"
export RECURRENCE_MAX="${RECURRENCE_MAX:-12}"
export SKIP_EXISTING="${SKIP_EXISTING:-0}"
export SKIP_RECURRENCE="${SKIP_RECURRENCE:-1}"

bash "${SCRIPT_DIR}/run_eval_plot_recurrent_sweep.sh"
