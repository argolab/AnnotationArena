#!/bin/bash
# Easy-data axis invariance: annotator-split mode
# Items are shared; train/val/test differ by annotator group.
# Ratings are missing completely at random (MCAR).

set -e

# Fixed base parameters
BASE_I=9
BASE_C=4
BASE_K=250
BASE_J_TRAIN=14
BASE_J_VAL=3
BASE_J_TEST=3

# Baseline hyperparameter values
BASE_D=6
BASE_SIGMA_ANNOTATOR=0.5
BASE_SIGMA_MEASUREMENT=0.1
BASE_KAPPA=10
BASE_PROTOCOL="mcar"
BASE_MCAR_RATE=0.5

DISABLE_RANKING=1

ranking_args=""
if [ $DISABLE_RANKING == 0 ]; then
    ranking_args="--enable-pairwise-rankings"
fi

# Construct run name
run_name="Normal_${BASE_K}_20_${BASE_I}_AnnotatorTest_${BASE_J_TRAIN}"

# Determine protocol-specific arguments
protocol_args=""
if [ "$BASE_PROTOCOL" == "mcar" ]; then
    protocol_args="--mcar-missing-rate $BASE_MCAR_RATE"
fi

# Step 1: Generate data (Stan - annotator split)
python STAN/stan_code/scripts/generate_data_annotator.py \
    --K $BASE_K \
    --J-train $BASE_J_TRAIN \
    --J-val $BASE_J_VAL \
    --J-test $BASE_J_TEST \
    --I $BASE_I \
    --D $BASE_D \
    --C $BASE_C \
    --observation-protocol $BASE_PROTOCOL \
    --sigma-annotator $BASE_SIGMA_ANNOTATOR \
    --sigma-measurement $BASE_SIGMA_MEASUREMENT \
    --kappa $BASE_KAPPA \
    --output-dir "DATA/STAN/Normal_250_20_9_AnnotatorTest" \
    --run-name $run_name \
    --overwrite-existing-data \
    --no-factored-annotator \
    $ranking_args \
    $protocol_args </dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: Data generation failed for $run_name"
    exit 1
fi
run_name="Factor_${BASE_K}_20_${BASE_I}_AnnotatorTest_${BASE_J_TRAIN}"

# Determine protocol-specific arguments
protocol_args=""
if [ "$BASE_PROTOCOL" == "mcar" ]; then
    protocol_args="--mcar-missing-rate $BASE_MCAR_RATE"
fi

# Step 1: Generate data (Stan - annotator split)
python STAN/stan_code/scripts/generate_data_annotator.py \
    --K $BASE_K \
    --J-train $BASE_J_TRAIN \
    --J-val $BASE_J_VAL \
    --J-test $BASE_J_TEST \
    --I $BASE_I \
    --D $BASE_D \
    --C $BASE_C \
    --observation-protocol $BASE_PROTOCOL \
    --sigma-annotator $BASE_SIGMA_ANNOTATOR \
    --sigma-measurement $BASE_SIGMA_MEASUREMENT \
    --kappa $BASE_KAPPA \
    --output-dir "DATA/STAN/Factor_250_20_9_AnnotatorTest" \
    --use-factored-annotator \
    --run-name $run_name \
    --overwrite-existing-data \
    $ranking_args \
    $protocol_args </dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: Data generation failed for $run_name"
    exit 1
fi