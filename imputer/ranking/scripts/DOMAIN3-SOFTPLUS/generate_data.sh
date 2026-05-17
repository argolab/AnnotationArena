#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=D3OLDGEN
#SBATCH --output=/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=cpu
#SBATCH --time=72:00:00

DOMAIN3_DEFAULT_CLUSTER_ROOT="/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking"
DOMAIN3_DEFAULT_LOCAL_ROOT="/Users/prabhavsingh/Documents/JHU/JHUResearch/Marformer/imputer/ranking"

if [[ -n "${DOMAIN3_REPO_ROOT_OVERRIDE:-}" && -f "${DOMAIN3_REPO_ROOT_OVERRIDE}/scripts/DOMAIN3/common.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_REPO_ROOT_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/DOMAIN3/common.sh" ]]; then
    DOMAIN3_REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${DOMAIN3_DEFAULT_CLUSTER_ROOT}/scripts/DOMAIN3/common.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_DEFAULT_CLUSTER_ROOT}"
elif [[ -f "${DOMAIN3_DEFAULT_LOCAL_ROOT}/scripts/DOMAIN3/common.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_DEFAULT_LOCAL_ROOT}"
else
    DOMAIN3_REPO_ROOT="$(pwd)"
fi

export DOMAIN3_REPO_ROOT_OVERRIDE="${DOMAIN3_REPO_ROOT}"
# shellcheck disable=SC1091
source "${DOMAIN3_REPO_ROOT}/scripts/DOMAIN3/common.sh"

domain3_setup_env
domain3_enable_file_logging "domain3_softplus_generate_data"
domain3_enter_repo

SCRIPT_START=$SECONDS

DATA_ROOT="DATA/STAN/DOMAIN3-SOFTPLUS"
BASE_ROOT="${DATA_ROOT}/Base"
BASE_RUN="DOMAIN3-SOFTPLUS_BASE_5000_50_9"
RUN_PREFIX="DOMAIN3-SOFTPLUS"
GEN_STAN_FILE="STAN/stan_models/tensor_generation_softplus.stan"

K_TRAIN=4800
K_TEST=100
K_VAL=100
I_DIM=9
J_DIM=50
C_DIM=4
D_DIM=32
T_PROTO=3
TEST_ITEMS=100
TEST_ANNOTS=10
ITEM_TRANS_SIZES="100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000"
ITEM_NT_SIZES="100,200,300,400,500,600,700,800,900,1900,2900,3900,4900"
ANNOT_TRANS_SIZES="10,15,20,25,30,35,40,45,50"
ANNOT_NT_SIZES="10,15,20,25,30,35,40"

SIGMA_U=0.72186008678
SIGMA_V=10.0771146648
SIGMA_UIT=1.40871172142
SIGMA_MEASUREMENT=0.58704029672
KAPPA=25.284987056
ALPHA_CONFUSION=15.0
USE_DAWID_SKENE_NOISE=0
DERIVE_THRESHOLDS_FROM_ANNOTATOR=0
MCAR_MISSING_RATE=0.5
SEED=42

mkdir -p "${BASE_ROOT}"

domain3_print_block "DOMAIN3-SOFTPLUS | tensor base generation"
echo " BASE_ROOT                    : ${BASE_ROOT}"
echo " BASE_RUN                     : ${BASE_RUN}"
echo " STAN_FILE                    : ${GEN_STAN_FILE}"
echo " K_train / K_val / K_test     : ${K_TRAIN} / ${K_VAL} / ${K_TEST}"
echo " I / J / C / D / T            : ${I_DIM} / ${J_DIM} / ${C_DIM} / ${D_DIM} / ${T_PROTO}"
echo " test_items / test_annotators : ${TEST_ITEMS} / ${TEST_ANNOTS}"
echo " item_trans_sizes             : ${ITEM_TRANS_SIZES}"
echo " item_nt_sizes                : ${ITEM_NT_SIZES}"
echo " annot_trans_sizes            : ${ANNOT_TRANS_SIZES}"
echo " annot_nt_sizes               : ${ANNOT_NT_SIZES}"
echo " sigma_u                      : ${SIGMA_U}"
echo " sigma_v                      : ${SIGMA_V}"
echo " sigma_uit                    : ${SIGMA_UIT}"
echo " sigma_measurement            : ${SIGMA_MEASUREMENT}"
echo " kappa                        : ${KAPPA}"
echo " alpha_confusion              : ${ALPHA_CONFUSION}"
echo " REPO_ROOT                    : ${DOMAIN3_REPO_ROOT}"
echo " LOG_FILE                     : ${DOMAIN3_LOG_FILE}"

GEN_START=$SECONDS
python STAN/stan_code/scripts/generate_data.py \
    --output-dir            "${BASE_ROOT}"                    \
    --run-name              "${BASE_RUN}"                     \
    --stan-type             "tensor"                          \
    --stan-file             "${GEN_STAN_FILE}"                \
    --K-train               "${K_TRAIN}"                      \
    --K-test                "${K_TEST}"                       \
    --K-val                 "${K_VAL}"                        \
    --I                     "${I_DIM}"                        \
    --J                     "${J_DIM}"                        \
    --C                     "${C_DIM}"                        \
    --D                     "${D_DIM}"                        \
    --kappa                 "${KAPPA}"                        \
    --sigma-measurement     "${SIGMA_MEASUREMENT}"            \
    --mcar-missing-rate     "${MCAR_MISSING_RATE}"            \
    --observation-protocol  mcar                              \
    --seed                  "${SEED}"                         \
    --stan-arg              T="${T_PROTO}"                    \
    --stan-arg              sigma_u="${SIGMA_U}"              \
    --stan-arg              sigma_v="${SIGMA_V}"              \
    --stan-arg              sigma_uit="${SIGMA_UIT}"          \
    --stan-arg              use_dawid_skene_noise="${USE_DAWID_SKENE_NOISE}" \
    --stan-arg              derive_thresholds_from_annotator="${DERIVE_THRESHOLDS_FROM_ANNOTATOR}" \
    --stan-arg              alpha_confusion="${ALPHA_CONFUSION}" \
    --overwrite-existing-data
GEN_ELAPSED=$(( SECONDS - GEN_START ))

SPLIT_START=$SECONDS
domain3_print_block "DOMAIN3-SOFTPLUS | split construction"
python STAN/stan_code/scripts/domain3subsplit.py \
    --input-dir         "${BASE_ROOT}/${BASE_RUN}" \
    --output-root       "${DATA_ROOT}"             \
    --run-prefix        "${RUN_PREFIX}"            \
    --item-trans-sizes  "${ITEM_TRANS_SIZES}"      \
    --item-nt-sizes     "${ITEM_NT_SIZES}"         \
    --annot-trans-sizes "${ANNOT_TRANS_SIZES}"     \
    --annot-nt-sizes    "${ANNOT_NT_SIZES}"        \
    --test-items        "${TEST_ITEMS}"            \
    --test-annotators   "${TEST_ANNOTS}"
SPLIT_ELAPSED=$(( SECONDS - SPLIT_START ))

BASE_PAYLOAD_JSON=$(cat <<EOF
{
  "launcher_type": "domain3_generation",
  "dataset": "DOMAIN3-SOFTPLUS",
  "base_root": "${BASE_ROOT}",
  "base_run": "${BASE_RUN}",
  "data_root": "${DATA_ROOT}",
  "run_prefix": "${RUN_PREFIX}",
  "stan_file": "${GEN_STAN_FILE}",
  "elapsed_seconds": {
    "generation": ${GEN_ELAPSED},
    "splits": ${SPLIT_ELAPSED},
    "total": $(( SECONDS - SCRIPT_START ))
  },
  "dimensions": {
    "K_train": ${K_TRAIN},
    "K_val": ${K_VAL},
    "K_test": ${K_TEST},
    "I": ${I_DIM},
    "J": ${J_DIM},
    "C": ${C_DIM},
    "D": ${D_DIM},
    "T": ${T_PROTO}
  },
  "test_corner": {
    "items": ${TEST_ITEMS},
    "annotators": ${TEST_ANNOTS}
  },
  "hyperparameters": {
    "sigma_u": ${SIGMA_U},
    "sigma_v": ${SIGMA_V},
    "sigma_uit": ${SIGMA_UIT},
    "sigma_measurement": ${SIGMA_MEASUREMENT},
    "kappa": ${KAPPA},
    "alpha_confusion": ${ALPHA_CONFUSION},
    "use_dawid_skene_noise": ${USE_DAWID_SKENE_NOISE},
    "derive_thresholds_from_annotator": ${DERIVE_THRESHOLDS_FROM_ANNOTATOR},
    "mcar_missing_rate": ${MCAR_MISSING_RATE},
    "seed": ${SEED}
  },
  "sizes": {
    "item_transductive_total_selected": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000],
    "item_nontransductive_train_only": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1900, 2900, 3900, 4900],
    "annot_transductive_total_selected": [10, 15, 20, 25, 30, 35, 40, 45, 50],
    "annot_nontransductive_train_only": [10, 15, 20, 25, 30, 35, 40]
  }
}
EOF
)
domain3_write_json_payload "${BASE_ROOT}/${BASE_RUN}/launcher_metadata.json" "${BASE_PAYLOAD_JSON}"
domain3_write_json_payload "${DATA_ROOT}/domain3_softplus_generation_summary.json" "${BASE_PAYLOAD_JSON}"

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
domain3_print_block "DOMAIN3-SOFTPLUS generation complete"
echo " generation : $(( GEN_ELAPSED / 60 ))m $(( GEN_ELAPSED % 60 ))s"
echo " splits      : $(( SPLIT_ELAPSED / 60 ))m $(( SPLIT_ELAPSED % 60 ))s"
echo " total       : $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " base bundle : ${BASE_ROOT}/${BASE_RUN}"
echo " split root  : ${DATA_ROOT}"
