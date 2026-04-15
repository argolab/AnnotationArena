#!/bin/bash

#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=GENERATE_400
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

DATA_ROOT="DATA/STAN/SPARSE/Tensor_400_25_9_ItemTest"

echo ""
echo "============================================================"
echo " Step 1: Generate Tensor_400_25_9_ItemTest_300"
echo "  K_train=300  K_test=50  K_val=50  C=5  I=9  J=25"
echo "============================================================"

python STAN/stan_code/scripts/generate_data.py \
    --output-dir            "$DATA_ROOT"                       \
    --run-name              "Tensor_400_25_9_ItemTest_300"     \
    --stan-type             "tensor"                           \
    --K-train               300                                \
    --K-test                50                                 \
    --K-val                 50                                 \
    --I                     9                                  \
    --J                     25                                 \
    --C                     5                                  \
    --D                     32                                 \
    --kappa                 15.0                               \
    --sigma-measurement     0.1                                \
    --mcar-missing-rate     0.5                                \
    --observation-protocol  mcar                               \
    --seed                  42                                 \
    --stan-arg              T=3                                \
    --stan-arg              sigma_u=1.0                        \
    --stan-arg              sigma_v=1.0                        \
    --stan-arg              sigma_uit=0.1                      \
    --stan-arg              use_dawid_skene_noise=0            \
    --stan-arg              derive_thresholds_from_annotator=0 \
    --stan-arg              alpha_confusion=15.0               \
    --overwrite-existing-data

echo ""
echo "============================================================"
echo " Step 2: Subset to smaller train sizes"
echo "============================================================"

for SIZE in 200 100 50 10; do
    echo ""
    echo "--- Subsetting K_train=${SIZE} ---"
    python STAN/stan_code/scripts/subset_item_split.py \
        --input-dir  "$DATA_ROOT/Tensor_400_25_9_ItemTest_300" \
        --output-dir "$DATA_ROOT/Tensor_400_25_9_ItemTest_${SIZE}" \
        --train-num  $SIZE
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Datasets: ${DATA_ROOT}/Tensor_400_25_9_ItemTest_{10,50,100,200,300}"
echo "============================================================"
