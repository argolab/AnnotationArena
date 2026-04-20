#!/bin/bash

#SBATCH --job-name=DOMAIN3_GEN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=cpu
#SBATCH --time=06:00:00

set -e

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/AnnotationArena/imputer/ranking
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_START=$SECONDS
DATA_ROOT="DATA/STAN/DOMAIN3"
BASE_ROOT="${DATA_ROOT}/Base"
BASE_RUN="Tensor_400_25_9_DOMAIN3_BASE"

mkdir -p "${BASE_ROOT}"

echo "============================================================"
echo " DOMAIN3 | Generate base tensor data"
echo "============================================================"

python STAN/stan_code/scripts/generate_data.py \
    --output-dir            "${BASE_ROOT}"                       \
    --run-name              "${BASE_RUN}"                       \
    --stan-type             "tensor"                            \
    --K-train               300                                 \
    --K-test                50                                  \
    --K-val                 50                                  \
    --I                     9                                   \
    --J                     25                                  \
    --C                     5                                   \
    --D                     32                                  \
    --kappa                 15.0                                \
    --sigma-measurement     0.1                                 \
    --mcar-missing-rate     0.5                                 \
    --observation-protocol  mcar                                \
    --seed                  42                                  \
    --stan-arg              T=3                                 \
    --stan-arg              sigma_u=1.0                         \
    --stan-arg              sigma_v=1.0                         \
    --stan-arg              sigma_uit=0.1                       \
    --stan-arg              use_dawid_skene_noise=0             \
    --stan-arg              derive_thresholds_from_annotator=0  \
    --stan-arg              alpha_confusion=15.0                \
    --overwrite-existing-data

echo
echo "============================================================"
echo " DOMAIN3 | Create expansion splits"
echo "============================================================"

python STAN/stan_code/scripts/domain3subsplit.py \
    --input-dir   "${BASE_ROOT}/${BASE_RUN}" \
    --output-root "${DATA_ROOT}"             \
    --run-prefix  "Tensor_400_25_9_DOMAIN3"

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo
echo "============================================================"
echo " DOMAIN3 complete in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Base data: ${BASE_ROOT}/${BASE_RUN}"
echo " Splits:    ${DATA_ROOT}/ItemSplits and ${DATA_ROOT}/AnnotSplits"
echo "============================================================"
