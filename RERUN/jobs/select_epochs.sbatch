#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --comment=accept_cost
set -euo pipefail
# Phase A -> phase B bridge: turn the search runs' validation curves into the
# per-(family, axis, size, seed) epoch budget the refit trains to.
if [[ -n "${RERUN_DIR:-}" && -f "${RERUN_DIR}/config/cluster.sh" ]]; then
    source "${RERUN_DIR}/config/cluster.sh"
else
    source "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")/config/cluster.sh"
fi
[[ -n "${RERUN_CONF:-}" ]] && source "${RERUN_CONF}"
rerun_activate_env
: "${RUN_ID:?}" "${VARIANT:?}" "${EPOCH_TABLE:?}"
# SELECT_MODE / SELECT_SMOOTH arrive from the conf (which is sourced, not
# exported) and are optional: unset, select_epochs.py keeps its own defaults
# (per-size, smooth 5), so every existing caller is unaffected.
python -u "${RERUN_DIR}/analysis/select_epochs.py" \
    --run-dir "$(rerun_run_dir "${RUN_ID}")" \
    --variant "${VARIANT}-search" \
    ${SELECT_MODE:+--select "${SELECT_MODE}"} \
    ${SELECT_SMOOTH:+--smooth "${SELECT_SMOOTH}"} \
    -o "${EPOCH_TABLE}"
