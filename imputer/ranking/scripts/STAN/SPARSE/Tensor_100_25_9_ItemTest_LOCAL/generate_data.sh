#!/bin/bash

SCRIPT_START=$SECONDS

DATA_ROOT="DATA/STAN/SPARSE/Tensor_100_25_9_ItemTest"

echo ""
echo "============================================================"
echo " Step 1: Generate Tensor_100_25_9_ItemTest_80"
echo "  K_train=80  K_test=10  K_val=10  C=5  I=9  J=25"
echo "============================================================"

python STAN/stan_code/scripts/generate_data.py \
    --output-dir            "$DATA_ROOT"                       \
    --run-name              "Tensor_100_25_9_ItemTest_80"     \
    --stan-type             "tensor"                           \
    --K-train               80                                \
    --K-test                10                                 \
    --K-val                 10                                 \
    --I                     9                                  \
    --J                     25                                 \
    --C                     5                                  \
    --D                     8                                  \
    --kappa                 10.0                               \
    --sigma-measurement     0.1                                \
    --mcar-missing-rate     0.5                                \
    --observation-protocol  mcar                               \
    --seed                  42                                 \
    --stan-arg              T=3                                \
    --stan-arg              sigma_u=0.5                        \
    --stan-arg              sigma_v=0.5                        \
    --stan-arg              sigma_uit=0.1                      \
    --stan-arg              use_dawid_skene_noise=0            \
    --stan-arg              derive_thresholds_from_annotator=0 \
    --stan-arg              alpha_confusion=15.0               \
    --overwrite-existing-data

echo ""
echo "============================================================"
echo " Step 2: Subset to smaller train sizes"
echo "============================================================"

for SIZE in 40 20 10; do
    echo ""
    echo "--- Subsetting K_train=${SIZE} ---"
    python STAN/stan_code/scripts/subset_item_split.py \
        --input-dir  "$DATA_ROOT/Tensor_100_25_9_ItemTest_80" \
        --output-dir "$DATA_ROOT/Tensor_100_25_9_ItemTest_${SIZE}" \
        --train-num  $SIZE
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Datasets: ${DATA_ROOT}/Tensor_100_25_9_ItemTest_{10,20,40}"
echo "============================================================"
