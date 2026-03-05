#!/bin/bash
# Stan ablations on existing tcp_normal_data
# True data params: D=6, sigma_measurement=0.1, alpha_dirichlet=10
#
# Runs:
#   1. Half D (D=3), correct hyperparams
#   2. Correct D (D=6), wrong sigma_measurement=0.5
#   3. Correct D (D=6), wrong alpha_dirichlet=2.0
#   4. Half D (D=3), wrong sigma_measurement=0.5, wrong alpha_dirichlet=2.0
set -e

STAN_CHAINS=1
STAN_ITER=300
STAN_WARMUP=100

DATA_NAME="tcp_normal_data"
EXP_PREFIX="tcp_normal"

run_stan_eval() {
    local stan_name=$1
    local extra_flags=$2

    echo ""
    echo "=========================================="
    echo "[stan] ${stan_name}"
    echo "  flags: ${extra_flags}"
    echo "=========================================="

    python stan/scripts/run_inference.py \
        --data-bundle OUTPUT/generated_data/${DATA_NAME}/data_bundle.json \
        --chains $STAN_CHAINS \
        --iter-sampling $STAN_ITER \
        --iter-warmup $STAN_WARMUP \
        --run-name ${stan_name} \
        --overwrite-existing-data \
        $extra_flags </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan inference failed for $stan_name"
        return 1
    fi

    python stan/scripts/evaluate_predictions.py \
        --data-bundle OUTPUT/generated_data/${DATA_NAME}/data_bundle.json \
        --mcmc-dir OUTPUT/domain_model/runs/${stan_name} \
        --run-name ${stan_name}_eval \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan evaluation failed for ${stan_name}_eval"
        return 1
    fi
}

# 1. Half D (D=3), correct hyperparams
run_stan_eval "${EXP_PREFIX}_stan1c_halfD" "--stan-arg D=3"

# 2. Correct D (D=6), wrong sigma_measurement=0.5
run_stan_eval "${EXP_PREFIX}_stan1c_wrongSigma" "--stan-arg sigma_measurement=0.5"

# 3. Correct D (D=6), wrong alpha_dirichlet=2.0
run_stan_eval "${EXP_PREFIX}_stan1c_wrongAlpha" "--stan-arg alpha_dirichlet=2.0"

# 4. Half D + wrong sigma + wrong alpha (all misspecified)
run_stan_eval "${EXP_PREFIX}_stan1c_allWrong" "--stan-arg D=3 --stan-arg sigma_measurement=0.5 --stan-arg alpha_dirichlet=2.0"

echo ""
echo "All Stan ablations complete."
