#!/bin/bash
# Easy-data axis invariance: IJK mode (hold_I=0, hold_J=0, hold_K=0)
# Full dependence on all three axes (baseline complex case).

set -e

# Fixed base parameters (mirrors OFAT center setup where possible)
BASE_I=9
BASE_J=25
BASE_C=4
BASE_K_TRAIN=175
BASE_K_TEST=25
BASE_K_VAL=25

# Baseline hyperparameter values
BASE_D=8
BASE_SIGMA_ANNOTATOR=0.5
BASE_SIGMA_MEASUREMENT=0.1
BASE_KAPPA=10
BASE_PROTOCOL="mcar"  # SMAR
BASE_PROTOCOL_CODE="mcar"


DISABLE_RANKING=1

ranking_args=""

if [ $DISABLE_RANKING == 0 ]; then
    ranking_args="--enable-pairwise-rankings"
fi


# Format values for folder naming
d_str=$BASE_D
sa_str=$(echo $BASE_SIGMA_ANNOTATOR | sed 's/\.//g')
sm_str=$(echo $BASE_SIGMA_MEASUREMENT | sed 's/\.//g')
kp_str=$(echo $BASE_KAPPA | sed 's/\.//g')

# Construct run name
run_name="Normal_225_${BASE_J}_${BASE_I}_ItemTest_${BASE_K_TRAIN}"

#run_name="normal_noise_dot_product_llm_rubric"

# Determine protocol-specific arguments
protocol_args=""
if [ "$BASE_PROTOCOL" == "extended_rankings" ]; then
    protocol_args="--extended-pairwise-rate 0.2"
elif [ "$BASE_PROTOCOL" == "mcar" ]; then
    protocol_args="--mcar-missing-rate 0.5"
fi

# Step 1: Generate data (Stan)
echo "[Step 1/6] Generating data..."

python STAN/stan_code/scripts/generate_data.py \
    --K-train $BASE_K_TRAIN \
    --K-test $BASE_K_TEST \
    --K-val $BASE_K_VAL \
    --I $BASE_I \
    --J $BASE_J \
    --D $BASE_D \
    --C $BASE_C \
    --observation-protocol $BASE_PROTOCOL \
    --sigma-annotator $BASE_SIGMA_ANNOTATOR \
    --sigma-measurement $BASE_SIGMA_MEASUREMENT \
    --kappa $BASE_KAPPA \
    --output-dir "DATA/STAN/Normal_225_25_9_ItemTest" \
    --run-name $run_name \
    --overwrite-existing-data \
    --stan-type "normal-noise-dot-product" \
    $ranking_args \
    $protocol_args </dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: Data generation failed for $run_name"
    exit 1
fi

# Construct run name
run_name="Factor_225_${BASE_J}_${BASE_I}_ItemTest_${BASE_K_TRAIN}"

#run_name="normal_noise_dot_product_llm_rubric"

# Determine protocol-specific arguments
protocol_args=""
if [ "$BASE_PROTOCOL" == "extended_rankings" ]; then
    protocol_args="--extended-pairwise-rate 0.2"
elif [ "$BASE_PROTOCOL" == "mcar" ]; then
    protocol_args="--mcar-missing-rate 0.5"
fi

# Step 1: Generate data (Stan)
# echo "[Step 1/6] Generating data..."

python STAN/stan_code/scripts/generate_data.py \
    --K-train $BASE_K_TRAIN \
    --K-test $BASE_K_TEST \
    --K-val $BASE_K_VAL \
    --I $BASE_I \
    --J $BASE_J \
    --D $BASE_D \
    --C $BASE_C \
    --observation-protocol $BASE_PROTOCOL \
    --sigma-annotator $BASE_SIGMA_ANNOTATOR \
    --sigma-measurement $BASE_SIGMA_MEASUREMENT \
    --kappa $BASE_KAPPA \
    --output-dir "DATA/STAN/Factor_225_25_9_ItemTest" \
    --run-name $run_name \
    --overwrite-existing-data \
    --stan-type "factored-dot-product" \
    $ranking_args \
    $protocol_args </dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: Data generation failed for $run_name"
    exit 1
fi
