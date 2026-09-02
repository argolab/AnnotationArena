#!/bin/bash
#SBATCH --job-name=D3SIG-GEN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --comment=accept_cost
set -euo pipefail

# Generate the DOMAIN3-SIGMOID base tensor bundle and all four split families.
#
# Required env (set by submit_all.sh):
#   RUN_ID       run label; output goes to RERUN/runs/$RUN_ID/DATA/...
# Optional env:
#   RERUN_SEED   default 42
#
# Constants below are lifted verbatim from
# scripts/DOMAIN3-SIGMOID/generate_data.sh -- the configuration the paper's
# synthetic data was generated with. The generative hyperparameters are set to
# the inference model's gamma-prior means:
#   sigma_u ~ gamma(2,2)->1.0   sigma_v ~ gamma(2,2)->1.0
#   sigma_uit ~ gamma(2,4)->0.5 sigma_measurement ~ gamma(2,20)->0.1
#   kappa ~ gamma(2,2/15)->15.0

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

: "${RUN_ID:?RUN_ID must be set}"
SEED="${RERUN_SEED:-42}"

DATA_ROOT="$(rerun_data_root "${RUN_ID}")"
BASE_ROOT="${DATA_ROOT}/Base"
BASE_RUN="DOMAIN3-SIGMOID_BASE_5000_50_9"
GEN_STAN_FILE="STAN/stan_models/tensor_generation_sigmoid.stan"

# --- dimensions -------------------------------------------------------------
K_TRAIN=4800; K_TEST=100; K_VAL=100
I_DIM=9; J_DIM=50; C_DIM=4; D_DIM=32; T_PROTO=3
TEST_ITEMS=100; TEST_ANNOTS=10

# --- split size lists (these define the x-axes of every Domain 3 figure) -----
# The paper sweeps item generalization over {100..1000} plus {2000,3000,4000}
# (Marformer only -- Stan stops at 500 for cost reasons). The base bundle has
# K_TRAIN=4800 items, so the large sizes are subsets of it, not a regeneration.
ITEM_TRANS_SIZES="${ITEM_TRANS_SIZES:-100,200,300,400,500,600,700,800,900,1000,2000,3000,4000}"
ITEM_NT_SIZES="${ITEM_NT_SIZES:-100,200,300,400,500,600,700,800,900}"
ANNOT_TRANS_SIZES="${ANNOT_TRANS_SIZES:-10,15,20,25,30,35,40,45,50}"
ANNOT_NT_SIZES="${ANNOT_NT_SIZES:-10,15,20,25,30,35,40}"

# --- generative hyperparameters ---------------------------------------------
SIGMA_U=1.0; SIGMA_V=1.0; SIGMA_UIT=0.5; SIGMA_MEASUREMENT=0.1; KAPPA=15.0
ALPHA_CONFUSION=15.0
USE_DAWID_SKENE_NOISE=0
DERIVE_THRESHOLDS_FROM_ANNOTATOR=0
MCAR_MISSING_RATE=0.5

cd "${RANKING_ROOT}"
mkdir -p "${BASE_ROOT}"

echo "=============================================================="
echo " DOMAIN3-SIGMOID data generation"
echo " RUN_ID    : ${RUN_ID}"
echo " DATA_ROOT : ${DATA_ROOT}"
echo " seed      : ${SEED}"
echo "=============================================================="

# SUBSPLIT_ONLY=1 re-derives the train/test splits from a base bundle that
# already exists, skipping the (expensive) Stan generation step. Splitting is
# deterministic given the base seed and each size is built independently, so
# re-running with a superset of sizes reproduces the old splits and adds the
# new ones.
if [[ "${SUBSPLIT_ONLY:-0}" != "1" ]]; then
rerun_run python STAN/stan_code/scripts/generate_data.py \
    --output-dir            "${BASE_ROOT}"                   \
    --run-name              "${BASE_RUN}"                    \
    --stan-type             "tensor"                         \
    --stan-file             "${GEN_STAN_FILE}"               \
    --K-train               "${K_TRAIN}"                     \
    --K-test                "${K_TEST}"                      \
    --K-val                 "${K_VAL}"                       \
    --I                     "${I_DIM}"                       \
    --J                     "${J_DIM}"                       \
    --C                     "${C_DIM}"                       \
    --D                     "${D_DIM}"                       \
    --kappa                 "${KAPPA}"                       \
    --sigma-measurement     "${SIGMA_MEASUREMENT}"           \
    --mcar-missing-rate     "${MCAR_MISSING_RATE}"           \
    --observation-protocol  mcar                             \
    --seed                  "${SEED}"                        \
    --stan-arg              T="${T_PROTO}"                   \
    --stan-arg              sigma_u="${SIGMA_U}"             \
    --stan-arg              sigma_v="${SIGMA_V}"             \
    --stan-arg              sigma_uit="${SIGMA_UIT}"         \
    --stan-arg              use_dawid_skene_noise="${USE_DAWID_SKENE_NOISE}" \
    --stan-arg              derive_thresholds_from_annotator="${DERIVE_THRESHOLDS_FROM_ANNOTATOR}" \
    --stan-arg              alpha_confusion="${ALPHA_CONFUSION}" \
    --overwrite-existing-data

fi

rerun_run python STAN/stan_code/scripts/domain3subsplit.py \
    --input-dir         "${BASE_ROOT}/${BASE_RUN}" \
    --output-root       "${DATA_ROOT}"             \
    --run-prefix        "DOMAIN3-SIGMOID"          \
    --item-trans-sizes  "${ITEM_TRANS_SIZES}"      \
    --item-nt-sizes     "${ITEM_NT_SIZES}"         \
    --annot-trans-sizes "${ANNOT_TRANS_SIZES}"     \
    --annot-nt-sizes    "${ANNOT_NT_SIZES}"        \
    --test-items        "${TEST_ITEMS}"            \
    --test-annotators   "${TEST_ANNOTS}"

echo "data generation complete -> ${DATA_ROOT}"
