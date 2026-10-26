#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --comment=accept_cost
set -euo pipefail

# Fit one Stan model at one training size per array task (MCMC + evaluation).
# Stan is transductive-only; there is no non-transductive Stan arm in the paper.
#
# Required env (set by submit_all.sh):
#   RUN_ID       run label
#   FAMILY       StanOracle | StanMisspecs
#   MODEL_LABEL  human-readable, e.g. "Stan Bin"
#   MODEL_TAG    directory/series tag, e.g. TENSOR-SIG | BIN | PROJ | BOTH
#   STAN_FILE    path under imputer/ranking to the .stan inference model
#   AXIS         item | annot
#   SIZES        comma-separated training sizes, one per array task
#
# MODEL_TAG matters: plot_domain3_sigmoid.py locates Stan results by the
# directory suffix `_<MODEL_TAG>_eval` and expects exactly TENSOR-SIG, BIN,
# PROJ and BOTH.

# Under sbatch the script is copied to /var/spool, so BASH_SOURCE no longer
# points into the repo. submit_all.sh sources config/cluster.sh (which exports
# RERUN_DIR) before submitting, and --export=ALL carries it in; fall back to
# path resolution only when this script is run directly.
if [[ -n "${RERUN_DIR:-}" && -f "${RERUN_DIR}/config/cluster.sh" ]]; then
    source "${RERUN_DIR}/config/cluster.sh"
else
    source "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")/config/cluster.sh"
fi

# Job parameters arrive in a file, not via --export (see write_conf in
# submit_all.sh: --export mangles comma-separated size lists and spaces).
if [[ -n "${RERUN_CONF:-}" ]]; then
    # shellcheck disable=SC1090
    source "${RERUN_CONF}"
fi
rerun_activate_env
rerun_require_cmdstan

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

: "${RUN_ID:?}" "${FAMILY:?}" "${MODEL_LABEL:?}" "${MODEL_TAG:?}" \
  "${STAN_FILE:?}" "${AXIS:?}" "${SIZES:?}"

IFS=',' read -r -a SIZE_LIST <<< "${SIZES}"
SIZE="${SIZE_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"

AXIS_DIR="$(echo "${AXIS}" | tr '[:lower:]' '[:upper:]')"
if [[ "${FAMILY}" == "StanOracle" ]]; then
    STAN_SUBDIR="ORACLE"
else
    STAN_SUBDIR="MISSPEC"
fi
OUTPUT_ROOT="$(rerun_results_root "${RUN_ID}")/STAN/DOMAIN3-SIGMOID/${STAN_SUBDIR}/${AXIS_DIR}"

export DOMAIN3_DATASET_LABEL="DOMAIN3-SIGMOID"
export DOMAIN3_DATA_ROOT="$(rerun_data_root "${RUN_ID}")"
export DOMAIN3_SPLIT_PREFIX="DOMAIN3-SIGMOID"
export DOMAIN3_EXP_PREFIX="domain3_sigmoid"
export DOMAIN3_MISSPEC_OVERRIDE_D=8   # misspecified models use D=8, not the generator's 32
mkdir -p "${OUTPUT_ROOT}"

echo "=============================================================="
echo " ${MODEL_LABEL} (${MODEL_TAG}) | axis=${AXIS} | size=${SIZE}"
echo " RUN_ID      : ${RUN_ID}"
echo " STAN_FILE   : ${STAN_FILE}"
echo " OUTPUT_ROOT : ${OUTPUT_ROOT}"
echo "=============================================================="

rerun_run bash "${RANKING_ROOT}/scripts/DOMAIN3/run_stan_group.sh" \
    --family-label "${FAMILY}"      \
    --model-label  "${MODEL_LABEL}" \
    --model-tag    "${MODEL_TAG}"   \
    --stan-file    "${STAN_FILE}"   \
    --output-root  "${OUTPUT_ROOT}" \
    --axis         "${AXIS}"        \
    --sizes        "${SIZE}"
