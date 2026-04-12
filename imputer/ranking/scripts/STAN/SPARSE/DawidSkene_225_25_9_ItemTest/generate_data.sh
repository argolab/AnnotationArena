#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.
# Local — generate DawidSkene_225_25_9_ItemTest at K_train=200 then subset.
#   K_test=75, K_val=25, C=5, I=9, J=25, kappa=15.0, alpha_confusion=15.0
#   Train sizes: 200 (full), 175, 150, 100, 75, 50, 30, 10

cd /Users/prabhavsingh/Documents/JHU/JHUResearch/EntityMarformer/imputer/ranking
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

SCRIPT_START=$SECONDS

DATA_ROOT="DATA/STAN/SPARSE/DawidSkene_225_25_9_ItemTest"

echo ""
echo "============================================================"
echo " Step 1: Generate DawidSkene_225_25_9_ItemTest_200"
echo "  K_train=200  K_test=75  K_val=25  C=5  I=9  J=25"
echo "  kappa=15.0  alpha_confusion=15.0  mcar_missing_rate=0.5"
echo "============================================================"

python STAN/stan_code/scripts/generate_data.py \
    --output-dir        "$DATA_ROOT"                          \
    --run-name          "DawidSkene_225_25_9_ItemTest_200"    \
    --stan-type         "dawid-skene"                         \
    --K-train           200                                   \
    --K-test            75                                    \
    --K-val             25                                    \
    --I                 9                                     \
    --J                 25                                    \
    --C                 5                                     \
    --D                 8                                     \
    --kappa             15.0                                  \
    --alpha-confusion   15.0                                  \
    --sigma-annotator   0.5                                   \
    --mcar-missing-rate 0.5                                   \
    --observation-protocol mcar                               \
    --seed              42                                    \
    --overwrite-existing-data

echo ""
echo "============================================================"
echo " Step 2: Subset to smaller train sizes"
echo "============================================================"

for SIZE in 175 150 100 75 50 30 10; do
    echo ""
    echo "--- Subsetting K_train=${SIZE} ---"
    python STAN/stan_code/scripts/subset_item_split.py \
        --input-dir  "$DATA_ROOT/DawidSkene_225_25_9_ItemTest_200" \
        --output-dir "$DATA_ROOT/DawidSkene_225_25_9_ItemTest_${SIZE}" \
        --train-num  $SIZE
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Datasets: ${DATA_ROOT}/DawidSkene_225_25_9_ItemTest_{10,30,50,75,100,150,175,200}"
echo "============================================================"
