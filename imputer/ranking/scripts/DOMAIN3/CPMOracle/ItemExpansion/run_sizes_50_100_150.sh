#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=D3COIA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=24:00:00

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/AnnotationArena/imputer/ranking
export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
set -e

SCRIPT_START=$SECONDS
OUTPUT_DIR="RESULTS/STAN/TENSOR/DOMAIN3_CPM_ORACLE"

CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42

SIZE_LIST=(50 100 150)

echo ""
echo "============================================================"
echo " DOMAIN3 batch | CPMOracle | ItemExpansion | run_sizes_50_100_150.sh"
echo " Sizes        : ${SIZE_LIST[*]}"
echo " Output root  : ${OUTPUT_ROOT:-$OUTPUT_DIR}"
echo " Fixed tensor hyperparams:"
echo "   D=32  T=3  sigma_u=0.8  sigma_v=8  sigma_uit=0.8"
echo "   sigma_measurement=0.6  kappa=5.0  temperature=0.5"
echo "   use_dawid_skene_noise=0  derive_thresholds_from_annotator=0  alpha_confusion=15.0"
echo "============================================================"

for SIZE in "${SIZE_LIST[@]}"; do
    DATA_DIR="DATA/STAN/DOMAIN3/ItemSplits/Transductive/Tensor_400_25_9_DOMAIN3_Item_T_${SIZE}"
    DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
    RUN_NAME="Tensor_400_25_9_DOMAIN3_Item_T_${SIZE}_CPM_ORACLE"

    echo " DATA_DIR      : ${DATA_DIR}"
    echo " RUN_NAME      : ${RUN_NAME}"
    echo " OUTPUT_ROOT   : ${OUTPUT_ROOT:-$OUTPUT_DIR}"

    echo ""
    echo "============================================================"
    echo " Stan Tensor CPM Oracle | DOMAIN3 | Item Expansion | Size ${SIZE}"
    echo "  CHAINS       : ${CHAINS}"
    echo "  ITER_WARMUP  : ${ITER_WARMUP}"
    echo "  ITER_SAMPLING: ${ITER_SAMPLING}"
    echo "============================================================"

    echo "[1/2] Running MCMC (tensor CPM oracle)..."
    python STAN/stan_code/scripts/run_inference.py \
        --data-bundle                  "${DATA_BUNDLE}"             \
        --configs                      "${DATA_DIR}/configs.json"   \
        --output-dir                   "${OUTPUT_DIR}"              \
        --run-name                     "${RUN_NAME}"                \
        --stan-type                    "tensor"                     \
        --chains                       "$CHAINS"                    \
        --iter-warmup                  "$ITER_WARMUP"               \
        --iter-sampling                "$ITER_SAMPLING"             \
        --adapt-delta                  "$ADAPT_DELTA"               \
        --max-treedepth                "$MAX_TREEDEPTH"             \
        --seed                         "$SEED"                      \
        --override-D                   32                           \
        --override-sigma-measurement   0.6                          \
        --override-alpha-dirichlet     5.0                          \
        --override-temperature         0.5                          \
        --stan-arg                     T=3                          \
        --stan-arg                     sigma_u=0.8                  \
        --stan-arg                     sigma_v=8                    \
        --stan-arg                     sigma_uit=0.8                \
        --stan-arg                     use_dawid_skene_noise=0      \
        --stan-arg                     derive_thresholds_from_annotator=0 \
        --stan-arg                     alpha_confusion=15.0         \
        --overwrite-existing-data

    echo "[2/2] Evaluating predictions..."
    python STAN/stan_code/scripts/evaluate_predictions.py \
        --data-bundle        "${DATA_BUNDLE}"              \
        --mcmc-dir           "${OUTPUT_DIR}/${RUN_NAME}"   \
        --output-dir         "${OUTPUT_DIR}"               \
        --run-name           "${RUN_NAME}_eval"            \
        --csv-pattern        "tensor_model-*.csv"          \
        --overwrite-existing-data                           \
        --verbose
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results root: ${OUTPUT_DIR}"
echo "============================================================"
