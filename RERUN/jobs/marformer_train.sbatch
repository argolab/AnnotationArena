#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --comment=accept_cost
set -euo pipefail

# Train one Marformer per array task. Submitted once per (axis, protocol);
# SLURM_ARRAY_TASK_ID indexes into SIZES.
#
# Required env (set by submit_all.sh):
#   RUN_ID    run label
#   FAMILY    Marformer | MarformerNT      (only affects labels/metadata)
#   AXIS      item | annot
#   PROTOCOL  transductive | nontransductive
#   SIZES     comma-separated training sizes, one per array task
#
# Architecture and optimisation hyperparameters are NOT set here -- they live in
# scripts/DOMAIN3/run_marformer_group.sh (D=80, L=8, H=4, d_ff=128, AdamW
# lr=2e-4, wd=0.01, masking 15%, K=5 augmentations, masked loss weight 15,
# 400 epochs). Change them there so training and the paper stay in sync.

source "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")/config/cluster.sh"

# Job parameters arrive in a file, not via --export (see write_conf in
# submit_all.sh: --export mangles comma-separated size lists and spaces).
if [[ -n "${RERUN_CONF:-}" ]]; then
    # shellcheck disable=SC1090
    source "${RERUN_CONF}"
fi
rerun_activate_env

: "${RUN_ID:?}" "${FAMILY:?}" "${AXIS:?}" "${PROTOCOL:?}" "${SIZES:?}"

IFS=',' read -r -a SIZE_LIST <<< "${SIZES}"
SIZE="${SIZE_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"

AXIS_DIR="$(echo "${AXIS}" | tr '[:lower:]' '[:upper:]')"
if [[ "${PROTOCOL}" == "transductive" ]]; then
    RESULTS_SUBDIR="MARFORMER"
else
    RESULTS_SUBDIR="MARFORMER-NT"
fi
OUTPUT_ROOT="$(rerun_results_root "${RUN_ID}")/${RESULTS_SUBDIR}/DOMAIN3-SIGMOID/${AXIS_DIR}"

# Point the upstream runner at this run's data and results.
export DOMAIN3_DATASET_LABEL="DOMAIN3-SIGMOID"
export DOMAIN3_DATA_ROOT="$(rerun_data_root "${RUN_ID}")"
export DOMAIN3_SPLIT_PREFIX="DOMAIN3-SIGMOID"
export DOMAIN3_EXP_PREFIX="domain3_sigmoid"
export DOMAIN3_MAX_ITEM="${DOMAIN3_MAX_ITEM:-300}"   # see FIGURES.md discrepancy #1
mkdir -p "${OUTPUT_ROOT}"

# Note: run_marformer_group.sh also tees its own log into imputer/ranking/logs/
# (it sets DOMAIN3_LOG_DIR itself). The authoritative per-job log is the
# --output file that submit_all.sh points into this run's logs/ directory.

echo "=============================================================="
echo " ${FAMILY} | axis=${AXIS} | protocol=${PROTOCOL} | size=${SIZE}"
echo " RUN_ID      : ${RUN_ID}"
echo " OUTPUT_ROOT : ${OUTPUT_ROOT}"
echo " DATA_ROOT   : ${DOMAIN3_DATA_ROOT}"
echo " MAX_ITEM    : ${DOMAIN3_MAX_ITEM}"
echo "=============================================================="

rerun_run bash "${RANKING_ROOT}/scripts/DOMAIN3/run_marformer_group.sh" \
    --family-label "${FAMILY}"      \
    --output-root  "${OUTPUT_ROOT}" \
    --axis         "${AXIS}"        \
    --protocol     "${PROTOCOL}"    \
    --sizes        "${SIZE}"
