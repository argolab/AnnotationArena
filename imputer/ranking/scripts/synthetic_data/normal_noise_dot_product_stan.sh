#!/bin/bash
# Easy-data axis invariance: IJK mode (hold_I=0, hold_J=0, hold_K=0)
# Full dependence on all three axes (baseline complex case).

set -e

# Fixed base parameters (mirrors OFAT center setup where possible)
BASE_I=5
BASE_J=100
BASE_C=5
BASE_K_TRAIN=1
BASE_K_TEST=10

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

# Stan hyperparameters
STAN_4C_CHAINS=4
STAN_1C_CHAINS=1
STAN_1C_ITER_SAMPLING=300
STAN_1C_WARMUP=100
STAN_4C_ITER_SAMPLING=300
STAN_4C_WARMUP=100

# Format values for folder naming
d_str=$BASE_D
sa_str=$(echo $BASE_SIGMA_ANNOTATOR | sed 's/\.//g')
sm_str=$(echo $BASE_SIGMA_MEASUREMENT | sed 's/\.//g')
kp_str=$(echo $BASE_KAPPA | sed 's/\.//g')

# Construct run name
run_name="K_train_${BASE_K_TRAIN}_K_test_${BASE_K_TEST}_I_${BASE_I}_J_${BASE_J}_normal_noise_dot_product"

# Determine protocol-specific arguments
protocol_args=""
if [ "$BASE_PROTOCOL" == "extended_rankings" ]; then
    protocol_args="--extended-pairwise-rate 0.2"
elif [ "$BASE_PROTOCOL" == "mcar" ]; then
    protocol_args="--mcar-missing-rate 0.5"
fi


# Step 1: Generate data (Stan)
echo "[Step 1/6] Generating data..."

python stan/scripts/generate_data.py \
    --K-train $BASE_K_TRAIN \
    --K-test $BASE_K_TEST \
    --I $BASE_I \
    --J $BASE_J \
    --D $BASE_D \
    --C $BASE_C \
    --observation-protocol $BASE_PROTOCOL \
    --sigma-annotator $BASE_SIGMA_ANNOTATOR \
    --sigma-measurement $BASE_SIGMA_MEASUREMENT \
    --kappa $BASE_KAPPA \
    --run-name $run_name \
    --overwrite-existing-data \
    --stan-type normal-noise-dot-product \
    $ranking_args \
    $protocol_args </dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: Data generation failed for $run_name"
    exit 1
fi


# # Step 4: Run Stan inference (1 chain, long)
# echo "[Step 4/6] Running Stan inference (1 chain, long)..."
# python stan/scripts/run_inference.py \
#     --data-bundle OUTPUT/generated_data/${run_name}/data_bundle.json \
#     --chains $STAN_1C_CHAINS \
#     --iter-sampling $STAN_1C_ITER_SAMPLING \
#     --iter-warmup $STAN_1C_WARMUP \
#     --run-name ${run_name}_stan1c \
#     --overwrite-existing-data </dev/null

# if [ $? -ne 0 ]; then
#     echo "ERROR: Stan inference (1 chain) failed for $run_name"
#     exit 1
# fi



# # Step 5: Evaluate Stan predictions (1-chain version)
# echo "[Step 5/6] Evaluating Stan predictions (1 chain)..."
# python stan/scripts/evaluate_predictions.py \
#     --data-bundle OUTPUT/generated_data/${run_name}/data_bundle.json \
#     --mcmc-dir OUTPUT/domain_model/runs/${run_name}_stan1c \
#     --run-name ${run_name}_stan1c_eval \
#     --overwrite-existing-data </dev/null

# if [ $? -ne 0 ]; then
#     echo "ERROR: Stan evaluation (1 chain) failed for $run_name"
#     exit 1
# fi


# echo ""
# echo "✓ COMPLETED: $run_name"
# echo ""