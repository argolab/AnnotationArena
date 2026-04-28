#!/bin/bash

#SBATCH --job-name=D3COI4
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
OUTPUT_DIR="RESULTS/STAN/TENSOR/DOMAIN3_CPM_ORACLE"
CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42
SIZE_LIST=(200)

for SIZE in "${SIZE_LIST[@]}"; do
    DATA_DIR="DATA/STAN/DOMAIN3/ItemSplits/Transductive/Tensor_400_25_9_DOMAIN3_Item_T_${SIZE}"
    DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
    RUN_NAME="Tensor_400_25_9_DOMAIN3_Item_T_${SIZE}_CPM_ORACLE"
    python STAN/stan_code/scripts/run_inference.py \
        --data-bundle "${DATA_BUNDLE}" --configs "${DATA_DIR}/configs.json" \
        --output-dir "${OUTPUT_DIR}" --run-name "${RUN_NAME}" --stan-type "tensor" \
        --chains "$CHAINS" --iter-warmup "$ITER_WARMUP" --iter-sampling "$ITER_SAMPLING" \
        --adapt-delta "$ADAPT_DELTA" --max-treedepth "$MAX_TREEDEPTH" --seed "$SEED" \
        --override-D 32 --override-sigma-measurement 0.6 --override-alpha-dirichlet 5.0 \
        --override-temperature 0.5 --stan-arg T=3 --stan-arg sigma_u=0.8 --stan-arg sigma_v=8 \
        --stan-arg sigma_uit=0.8 --stan-arg use_dawid_skene_noise=0 \
        --stan-arg derive_thresholds_from_annotator=0 --stan-arg alpha_confusion=15.0 \
        --overwrite-existing-data
    python STAN/stan_code/scripts/evaluate_predictions.py \
        --data-bundle "${DATA_BUNDLE}" --mcmc-dir "${OUTPUT_DIR}/${RUN_NAME}" \
        --output-dir "${OUTPUT_DIR}" --run-name "${RUN_NAME}_eval" \
        --csv-pattern "tensor_model-*.csv" --overwrite-existing-data --verbose
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo "Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
