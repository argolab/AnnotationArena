#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=DS_150_STAN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=04:00:00

SCRIPT_START=$SECONDS

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/AnnotationArena/imputer/ranking
export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
set -e

SCRIPT_START=$SECONDS

SIZE=50
DATA_DIR="DATA/STAN/SPARSE/Tensor_400_25_9_ItemTest/Tensor_400_25_9_ItemTest_${SIZE}"
DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
OUTPUT_DIR="RESULTS/STAN/SPARSE"
RUN_NAME="Tensor_400_25_9_ItemTest_${SIZE}_DISCRETE_MISP"

CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42

echo ""
echo "============================================================"
echo " Stan Discrete Misspecified | Tensor_500_25_9_ItemTest_${SIZE}"
echo "  CHAINS       : ${CHAINS}"
echo "  ITER_WARMUP  : ${ITER_WARMUP}"
echo "  ITER_SAMPLING: ${ITER_SAMPLING}"
echo "  (misspecified: M=6, S=6, kappa=2.0, sigma_measurement=0.5, temperature=1.0)"
echo "============================================================"

echo "[1/2] Running MCMC (discrete_type_domain_model, misspecified)..."
python STAN/stan_code/scripts/run_inference.py \
    --data-bundle                "$DATA_BUNDLE"   \
    --configs                    "${DATA_DIR}/configs.json" \
    --output-dir                 "$OUTPUT_DIR"    \
    --run-name                   "$RUN_NAME"      \
    --stan-type                  "discrete"       \
    --stan-file                  "STAN/stan_models/discrete_type_domain_model.stan" \
    --chains                     "$CHAINS"        \
    --iter-warmup                "$ITER_WARMUP"   \
    --iter-sampling              "$ITER_SAMPLING" \
    --adapt-delta                "$ADAPT_DELTA"   \
    --max-treedepth              "$MAX_TREEDEPTH" \
    --seed                       "$SEED"          \
    --override-alpha-dirichlet   2.0              \
    --override-sigma-measurement 1.0              \
    --override-sigma-annotator   0.6              \
    --override-temperature       1.0              \
    --stan-arg                   M=6              \
    --stan-arg                   S=6              \
    --stan-arg                   d_annotator=4    \
    --stan-arg                   use_factored_annotator=0 \
    --stan-arg                   N_pairwise_rankings=0 \
    --stan-arg                   N_missing_pairwise_rankings=0 \
    --overwrite-existing-data

echo "[2/2] Evaluating predictions..."
python STAN/stan_code/scripts/evaluate_predictions.py \
    --data-bundle        "$DATA_BUNDLE"              \
    --mcmc-dir           "${OUTPUT_DIR}/${RUN_NAME}" \
    --output-dir         "$OUTPUT_DIR"               \
    --run-name           "${RUN_NAME}_eval"           \
    --csv-pattern        "discrete_type_domain_model-*.csv" \
    --overwrite-existing-data                        \
    --verbose

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results: ${OUTPUT_DIR}/${RUN_NAME}_eval"
echo "============================================================"
