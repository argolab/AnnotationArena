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
python -u "${RERUN_DIR}/analysis/select_epochs.py" \
    --run-dir "$(rerun_run_dir "${RUN_ID}")" \
    --variant "${VARIANT}-search" \
    -o "${EPOCH_TABLE}"
