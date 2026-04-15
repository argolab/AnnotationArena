#!/bin/bash

#SBATCH --job-name=GENERATE_IA400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=cpu
#SBATCH --time=06:00:00

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/AnnotationArena/imputer/ranking
export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
set -e

DATA_ROOT="DATA/STAN/SPARSE/Tensor_400_25_9_ItemAnnotTest"
BASE_RUN="Tensor_400_25_9_ItemAnnotTest_300_15"

python STAN/stan_code/scripts/generate_data_item_annotator.py \
    --output-dir "$DATA_ROOT" \
    --run-name "$BASE_RUN" \
    --stan-type tensor \
    --K-train 300 \
    --K-test 50 \
    --K-val 50 \
    --J-train 15 \
    --J-val 5 \
    --J-test 5 \
    --I 9 \
    --C 5 \
    --D 32 \
    --kappa 15.0 \
    --sigma-measurement 0.1 \
    --mcar-missing-rate 0.5 \
    --observation-protocol mcar \
    --seed 42 \
    --stan-arg T=3 \
    --stan-arg sigma_u=1.0 \
    --stan-arg sigma_v=1.0 \
    --stan-arg sigma_uit=0.1 \
    --stan-arg use_dawid_skene_noise=0 \
    --stan-arg derive_thresholds_from_annotator=0 \
    --stan-arg alpha_confusion=15.0 \
    --overwrite-existing-data

for SPEC in "200 15" "100 10" "50 5" "10 5"; do
    read -r ITEMS ANNS <<< "$SPEC"
    python STAN/stan_code/scripts/subset_item_annotator_split.py \
        --input-dir "$DATA_ROOT/$BASE_RUN" \
        --output-dir "$DATA_ROOT/Tensor_400_25_9_ItemAnnotTest_${ITEMS}_${ANNS}" \
        --train-items "$ITEMS" \
        --train-annotators "$ANNS"
done
