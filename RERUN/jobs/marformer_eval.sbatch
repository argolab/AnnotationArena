#!/bin/bash
#SBATCH --job-name=D3SIG-EVAL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --comment=accept_cost
set -euo pipefail

# Run test-set evaluation over every trained Marformer in a results root.
# Must run after all training array tasks for that family have finished.
#
# Required env (set by submit_all.sh):
#   RUN_ID   run label
#   FAMILY   Marformer | MarformerNT

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
# This is a --gres=gpu job; a silent CPU fallback would be a bug, not a fallback.
export REQUIRE_GPU=1
rerun_activate_env

: "${RUN_ID:?}" "${FAMILY:?}"

if [[ "${FAMILY}" == "Marformer" ]]; then
    RESULTS_SUBDIR="MARFORMER"
else
    RESULTS_SUBDIR="MARFORMER-NT"
fi
RESULTS_ROOT="$(rerun_results_root "${RUN_ID}")/${RESULTS_SUBDIR}/DOMAIN3-SIGMOID"

export DOMAIN3_DATASET_LABEL="DOMAIN3-SIGMOID"
export DOMAIN3_EXP_PREFIX="domain3_sigmoid"

echo "=============================================================="
echo " Marformer eval | family=${FAMILY} | RUN_ID=${RUN_ID}"
echo " RESULTS_ROOT : ${RESULTS_ROOT}"
echo "=============================================================="

rerun_run bash "${RANKING_ROOT}/scripts/DOMAIN3/run_marformer_eval.sh" \
    --family-label "${FAMILY}" \
    --results-root "${RESULTS_ROOT}"
