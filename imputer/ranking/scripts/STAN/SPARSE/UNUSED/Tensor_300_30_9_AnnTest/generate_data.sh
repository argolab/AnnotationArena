#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=GENERATE_ANN300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=cpu
#SBATCH --time=06:00:00

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

DATA_ROOT="DATA/STAN/SPARSE/Tensor_300_30_9_AnnTest"

echo ""
echo "============================================================"
echo " Step 1: Generate Tensor_300_30_9_AnnTest_20"
echo "============================================================"

python STAN/stan_code/scripts/generate_data_annotator.py \
    --output-dir            "$DATA_ROOT"                       \
    --run-name              "Tensor_300_30_9_AnnTest_20"       \
    --stan-type             "tensor"                           \
    --K                     300                                \
    --J-train               20                                 \
    --J-val                 5                                  \
    --J-test                5                                  \
    --I                     9                                  \
    --C                     5                                  \
    --D                     32                                  \
    --T                     3                                  \
    --sigma-u               1.0                                \
    --sigma-v               1.0                                \
    --sigma-uit             0.1                                \
    --sigma-measurement     0.1                                \
    --kappa                 10.0                               \
    --alpha-confusion       15.0                               \
    --temperature           0.5                                \
    --use-dawid-skene-noise 0                                  \
    --mcar-missing-rate     0.5                                \
    --observation-protocol  mcar                               \
    --seed                  42                                 \
    --overwrite-existing-data

echo ""
echo "============================================================"
echo " Step 2: Subset to smaller J_train sizes"
echo "============================================================"

for SIZE in 5 10 15; do
    echo ""
    echo "--- Subsetting J_train=${SIZE} ---"
    python STAN/stan_code/scripts/subset_annotator_split.py \
        --input-dir  "$DATA_ROOT/Tensor_300_30_9_AnnTest_20" \
        --output-dir "$DATA_ROOT/Tensor_300_30_9_AnnTest_${SIZE}" \
        --train-num  $SIZE
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Datasets: ${DATA_ROOT}/Tensor_300_30_9_AnnTest_{5,10,15,20}"
echo "============================================================"
