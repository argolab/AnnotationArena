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
# ---------------------------------------------------------------- cell arrays
# CELL_RUN_IDS switches the array axis from "size" to "run id": each task picks
# its own RUN_ID out of the list and does the single size in SIZES. The
# obs-density thread needs this because its 18 cells are 18 runs at ONE size,
# and one array of 18 schedules where 18 arrays of 1 would just queue (the
# account runs ~10 jobs at a time -- see the task-pool notes).
# EPOCH_TABLE_NAME, if given, resolves to that file inside the task's own run
# dir, because the epoch key (FAMILY|AXIS|SIZE|SEED) is identical across cells
# and a shared table would collide.
# Empty by default, so every existing submission behaves exactly as before.
if [[ -n "${CELL_RUN_IDS:-}" ]]; then
    IFS=',' read -r -a _CELL_LIST <<< "${CELL_RUN_IDS}"
    _CELL_IDX="${SLURM_ARRAY_TASK_ID:-0}"
    [[ "${_CELL_IDX}" -lt "${#_CELL_LIST[@]}" ]] || {
        echo "FATAL: array task ${_CELL_IDX} outside CELL_RUN_IDS (${#_CELL_LIST[@]} cells)" >&2
        exit 1; }
    RUN_ID="${_CELL_LIST[${_CELL_IDX}]}"
    DATA_RUN_ID="${DATA_RUN_ID:-${RUN_ID}}"
    [[ -n "${EPOCH_TABLE_NAME:-}" ]] && EPOCH_TABLE="$(rerun_run_dir "${RUN_ID}")/${EPOCH_TABLE_NAME}"
    # SIZES holds a single entry in this mode; the array index is spent on cells.
    SLURM_ARRAY_TASK_ID=0
    echo "[cell-array] task ${_CELL_IDX} -> RUN_ID=${RUN_ID}"
fi

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
