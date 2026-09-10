#!/bin/bash
#SBATCH --job-name=STAN-4CHAIN
#SBATCH --comment=accept_cost
# Refit one Domain 3 Stan arm with 4 chains instead of 1.
#
# Two questions at once:
#   1. DIAGNOSTIC. Single-chain split-R-hat cannot see multimodality -- a chain
#      stuck in one mode looks converged. Only between-chain R-hat can. lp__ is
#      already clean on the split diagnostic (R-hat ~1.000 at K=4000), so the
#      prediction is that it stays clean, while the unidentified latents get
#      WORSE because independent chains land in different sign/rotation
#      configurations. If lp__ instead degrades, that is real multimodality the
#      single chain was blind to.
#   2. THE NUMBER. lp__ ESS is ~14 from 500 draws, so the posterior predictive
#      average rests on very few effective samples. Four chains gives 4x the
#      draws and a better-explored posterior, which usually LOWERS predictive
#      log-loss. Stan's 0.619 at K=4000 could improve and widen the gap. That is
#      a risk of running this, and the reason to run it.
#
# Writes to a fresh run id; the existing results under runs/legacy are untouched.
#
# Chains run in parallel, so wall clock is roughly one chain: K=4000 took 1618 s
# single-chain, so ~30-40 min here. -t is set on the command line with margin,
# since nothing resumes.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env

: "${S4_SIZE:?S4_SIZE required}" "${S4_TAG:?S4_TAG required}" "${S4_STAN_FILE:?}" "${S4_LABEL:?}"
RUN_ID="${S4_RUN_ID:-stan4chain}"
DATA_RUN="${S4_DATA_RUN:-dataseed-s101}"

export DOMAIN3_CHAINS="${S4_CHAINS:-4}"
export DOMAIN3_ITER_WARMUP="${S4_WARMUP:-300}"
export DOMAIN3_ITER_SAMPLING="${S4_SAMPLING:-500}"
export DOMAIN3_DATASET_LABEL="DOMAIN3-SIGMOID"
export DOMAIN3_DATA_ROOT="$(rerun_data_root "${DATA_RUN}")"
export DOMAIN3_SPLIT_PREFIX="DOMAIN3-SIGMOID"
export DOMAIN3_EXP_PREFIX="domain3_sigmoid"

OUT="$(rerun_results_root "${RUN_ID}")/STAN/DOMAIN3-SIGMOID/MISSPEC/ITEM"
mkdir -p "${OUT}"

echo "=============================================================="
echo " Stan ${S4_LABEL} | K=${S4_SIZE} | chains=${DOMAIN3_CHAINS}"
echo " warmup=${DOMAIN3_ITER_WARMUP} sampling=${DOMAIN3_ITER_SAMPLING}"
echo " data : ${DOMAIN3_DATA_ROOT}"
echo " out  : ${OUT}"
echo "=============================================================="

rerun_run bash "${RANKING_ROOT}/scripts/DOMAIN3/run_stan_group.sh" \
    --family-label StanMisspecs \
    --model-label  "${S4_LABEL}" \
    --model-tag    "${S4_TAG}" \
    --stan-file    "${S4_STAN_FILE}" \
    --output-root  "${OUT}" \
    --axis         item \
    --sizes        "${S4_SIZE}"
