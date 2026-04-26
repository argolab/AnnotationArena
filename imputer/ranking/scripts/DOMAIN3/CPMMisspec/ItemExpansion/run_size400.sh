#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=D3CMI6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=48:00:00

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/AnnotationArena/imputer/ranking
export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
set -e

SCRIPT_START=$SECONDS
OUTPUT_DIR="RESULTS/STAN/CPMMisspec/DOMAIN3"

CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42

SIZE_LIST=(400)

echo ""
echo "============================================================"
echo " DOMAIN3 batch | CPMMisspec | ItemExpansion | run_size400.sh"
echo " Sizes        : ${SIZE_LIST[*]}"
echo " Output root  : ${OUTPUT_ROOT:-$OUTPUT_DIR}"
echo "============================================================"

for SIZE in "${SIZE_LIST[@]}"; do
    DATA_DIR="DATA/STAN/DOMAIN3/ItemSplits/Transductive/Tensor_400_25_9_DOMAIN3_Item_T_${SIZE}"
    DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
    RUN_NAME="Tensor_400_25_9_DOMAIN3_Item_T_${SIZE}_CPM_MISSPEC"

    echo " DATA_DIR      : ${DATA_DIR}"
    echo " RUN_NAME      : ${RUN_NAME}"
    echo " OUTPUT_ROOT   : ${OUTPUT_ROOT:-$OUTPUT_DIR}"

    echo ""
    echo "============================================================"
    echo " Stan Tensor CPM Misspecified | DOMAIN3 | Item Expansion | Size ${SIZE}"
    echo "  CHAINS       : ${CHAINS}"
    echo "  ITER_WARMUP  : ${ITER_WARMUP}"
    echo "  ITER_SAMPLING: ${ITER_SAMPLING}"
    echo "  T            : 1"
    echo "  sigma_u      : 0.1"
    echo "  sigma_v      : 0.1"
    echo "  sigma_uit    : 0.01"
    echo "  sigma_meas   : 0.1"
    echo "  kappa        : 20"
    echo "============================================================"

    echo "[1/2] Running MCMC (tensor, fixed-hyperparameter misspecified)..."
    python STAN/stan_code/scripts/run_inference.py \
        --data-bundle        "${DATA_BUNDLE}"             \
        --configs            "${DATA_DIR}/configs.json"   \
        --output-dir         "${OUTPUT_DIR}"              \
        --run-name           "${RUN_NAME}"                \
        --stan-type          "tensor"                     \
        --stan-file          "STAN/stan_models/tensor_model_fixed_hyperparams.stan" \
        --chains             "$CHAINS"                    \
        --iter-warmup        "$ITER_WARMUP"               \
        --iter-sampling      "$ITER_SAMPLING"             \
        --adapt-delta        "$ADAPT_DELTA"               \
        --max-treedepth      "$MAX_TREEDEPTH"             \
        --seed               "$SEED"                      \
        --stan-arg           T=1                          \
        --stan-arg           sigma_u=0.1                  \
        --stan-arg           sigma_v=0.1                  \
        --stan-arg           sigma_uit=0.01               \
        --stan-arg           sigma_measurement=0.1        \
        --stan-arg           kappa=20                     \
        --stan-arg           use_dawid_skene_noise=0      \
        --stan-arg           derive_thresholds_from_annotator=0 \
        --stan-arg           alpha_confusion=15.0         \
        --overwrite-existing-data

    echo "[2/2] Evaluating predictions..."
    python STAN/stan_code/scripts/evaluate_predictions.py \
        --data-bundle        "${DATA_BUNDLE}"                             \
        --mcmc-dir           "${OUTPUT_DIR}/${RUN_NAME}"                  \
        --output-dir         "${OUTPUT_DIR}"                              \
        --run-name           "${RUN_NAME}_eval"                           \
        --csv-pattern        "tensor_model_fixed_hyperparams-*.csv"       \
        --overwrite-existing-data                                         \
        --verbose

done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results root: ${OUTPUT_DIR}"
echo "============================================================"
