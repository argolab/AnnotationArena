#!/bin/bash
#SBATCH --job-name=OBSDEN-DATA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --comment=accept_cost
set -euo pipefail

# One (generative draw, observation rate) cell per array task: remask the
# draw's base bundle at the target rate and write the Item_T_500 split.
#
# No Stan generation happens here. The six base bundles already exist under
# runs/legacy/dataseed-s*/DATA/.../Base/ and are reused unchanged; only the
# observed/missing partition is redrawn. 64G is for the 300 MB base bundle,
# which inflates to several GB as Python objects.
#
# MASK_SEED is held FIXED across the three rates on purpose -- that is what
# makes the observed sets nest and the three density cells paired. See PLAN.md.
#
# Optional env:
#   OBSDEN_DRAWS   space-separated draw tags      (default: s42 s101..s105)
#   OBSDEN_RATES   space-separated observed rates (default: 0.15 0.50 0.85)
#   OBSDEN_SIZE    item-transductive size         (default: 500)
#   OBSDEN_MASK_SEED                              (default: 20261017)
#   OBSDEN_PREFIX  run-id prefix                  (default: obsden)

if [[ -n "${RERUN_DIR:-}" && -f "${RERUN_DIR}/config/cluster.sh" ]]; then
    source "${RERUN_DIR}/config/cluster.sh"
else
    source "$(dirname "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")")/config/cluster.sh"
fi
if [[ -n "${RERUN_CONF:-}" ]]; then
    # shellcheck disable=SC1090
    source "${RERUN_CONF}"
fi
rerun_activate_env

read -r -a DRAWS <<< "${OBSDEN_DRAWS:-s42 s101 s102 s103 s104 s105}"
read -r -a RATES <<< "${OBSDEN_RATES:-0.15 0.50 0.85}"
SIZE="${OBSDEN_SIZE:-500}"
MASK_SEED="${OBSDEN_MASK_SEED:-20261017}"
# Must match submit_phase1.sh's PREFIX, or the cell's data is written where the
# downstream jobs will not look for it.
PREFIX="${OBSDEN_PREFIX:-obsden}"
BASE_RUN="DOMAIN3-SIGMOID_BASE_5000_50_9"

# Flat index -> (draw, rate). Rate is the fast axis so the three cells of one
# draw land on adjacent tasks and read the same base bundle off the same node
# cache when they happen to co-schedule.
TASK="${SLURM_ARRAY_TASK_ID:-0}"
N_RATES="${#RATES[@]}"
DRAW="${DRAWS[$((TASK / N_RATES))]}"
RATE="${RATES[$((TASK % N_RATES))]}"
# 0.15 -> 15, for the run id
RATE_TAG="$(python -c "print(f'{round(float('${RATE}')*100):02d}')")"
RUN_ID="${PREFIX}-${DRAW}-o${RATE_TAG}"

BASE_DIR="${RUNS_ROOT}/legacy/dataseed-${DRAW}/DATA/STAN/DOMAIN3-SIGMOID/Base/${BASE_RUN}"
[[ -f "${BASE_DIR}/data_bundle.json" ]] || {
    echo "FATAL: no base bundle at ${BASE_DIR}" >&2; exit 1; }
DATA_ROOT="$(rerun_data_root "${RUN_ID}")"
mkdir -p "${DATA_ROOT}"

echo "=============================================================="
echo " obs-density data | task ${TASK}"
echo " draw      : ${DRAW}"
echo " rate      : ${RATE} observed  (mask seed ${MASK_SEED})"
echo " size      : Item_T_${SIZE}"
echo " RUN_ID    : ${RUN_ID}"
echo " base      : ${BASE_DIR}"
echo " DATA_ROOT : ${DATA_ROOT}"
echo "=============================================================="

cd "${RANKING_ROOT}"
# Only the item-transductive family at one size: the other three families are
# out of Phase 1's scope and each costs a full pass over the base bundle.
rerun_run python STAN/stan_code/scripts/domain3subsplit.py \
    --input-dir         "${BASE_DIR}"        \
    --output-root       "${DATA_ROOT}"       \
    --run-prefix        "DOMAIN3-SIGMOID"    \
    --observed-rate     "${RATE}"            \
    --mask-seed         "${MASK_SEED}"       \
    --item-trans-sizes  "${SIZE}"            \
    --item-nt-sizes     ""                   \
    --annot-trans-sizes ""                   \
    --annot-nt-sizes    ""                   \
    --test-items        100                  \
    --test-annotators   10

echo "cell complete -> ${DATA_ROOT}"
