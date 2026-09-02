#!/bin/bash
#SBATCH --job-name=P10K-GEN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --comment=accept_cost
set -euo pipefail
# Generate a Domain 3 base with 10x the paper's item count, so the item-axis
# curve can be pushed past K=4000 and the saturation question answered:
# does transfer keep improving, or has it already flattened by 3000-4000?
#
# Only K_train changes. Every generative hyperparameter, the annotator count,
# the criteria count, the MCAR rate and the corner geometry (100 items x 10
# annotators) are held at the values in jobs/generate_data.sbatch, so a dot at
# K=10000 is on the same axis as the existing ones.
#
# Base seed stays 42: this pilot asks about scale, not about the draw. The
# data-seed question is answered separately in RERUN/dataseed/.
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
rerun_require_cmdstan

RUN_ID="${RUN_ID:-pilot-10k}"
SEED="${RERUN_SEED:-42}"
DATA_ROOT="$(rerun_data_root "${RUN_ID}")"
BASE_ROOT="${DATA_ROOT}/Base"
BASE_RUN="DOMAIN3-SIGMOID_BASE_10200_50_9"

K_TRAIN=10000; K_TEST=100; K_VAL=100
I_DIM=9; J_DIM=50; C_DIM=4; D_DIM=32; T_PROTO=3
TEST_ITEMS=100; TEST_ANNOTS=10
SIGMA_U=1.0; SIGMA_V=1.0; SIGMA_UIT=0.5; SIGMA_MEASUREMENT=0.1; KAPPA=15.0
ALPHA_CONFUSION=15.0; MCAR_MISSING_RATE=0.5

# The existing sizes are kept so the new tree carries the whole curve, not just
# its new tail; NT is shifted by the 100-item corner as always.
ITEM_TRANS_SIZES="1000,2000,3000,4000,6000,8000,10000"
ITEM_NT_SIZES="1000,2000,3000,4000,6000,8000,10000"
ANNOT_TRANS_SIZES="10,50"
ANNOT_NT_SIZES="10,40"

cd "${RANKING_ROOT}"
mkdir -p "${BASE_ROOT}"
echo "10k pilot: RUN_ID=${RUN_ID} seed=${SEED} K_train=${K_TRAIN} -> ${DATA_ROOT}"

if [[ "${SUBSPLIT_ONLY:-0}" != "1" ]]; then
rerun_run python STAN/stan_code/scripts/generate_data.py \
    --output-dir "${BASE_ROOT}" --run-name "${BASE_RUN}" \
    --stan-type tensor --stan-file "STAN/stan_models/tensor_generation_sigmoid.stan" \
    --K-train "${K_TRAIN}" --K-test "${K_TEST}" --K-val "${K_VAL}" \
    --I "${I_DIM}" --J "${J_DIM}" --C "${C_DIM}" --D "${D_DIM}" \
    --kappa "${KAPPA}" --sigma-measurement "${SIGMA_MEASUREMENT}" \
    --mcar-missing-rate "${MCAR_MISSING_RATE}" --observation-protocol mcar \
    --seed "${SEED}" \
    --stan-arg T="${T_PROTO}" --stan-arg sigma_u="${SIGMA_U}" \
    --stan-arg sigma_v="${SIGMA_V}" --stan-arg sigma_uit="${SIGMA_UIT}" \
    --stan-arg use_dawid_skene_noise=0 \
    --stan-arg derive_thresholds_from_annotator=0 \
    --stan-arg alpha_confusion="${ALPHA_CONFUSION}" \
    --overwrite-existing-data
fi

rerun_run python STAN/stan_code/scripts/domain3subsplit.py \
    --input-dir "${BASE_ROOT}/${BASE_RUN}" --output-root "${DATA_ROOT}" \
    --run-prefix "DOMAIN3-SIGMOID" \
    --item-trans-sizes "${ITEM_TRANS_SIZES}" --item-nt-sizes "${ITEM_NT_SIZES}" \
    --annot-trans-sizes "${ANNOT_TRANS_SIZES}" --annot-nt-sizes "${ANNOT_NT_SIZES}" \
    --test-items "${TEST_ITEMS}" --test-annotators "${TEST_ANNOTS}"
echo "10k pilot data complete -> ${DATA_ROOT}"
