#!/bin/bash
# Run Stan 1C with half the D used in data generation.
# Tests Stan with reduced embedding dimension (misspecified model capacity).
set -e

STAN_CHAINS=1
STAN_ITER=300
STAN_WARMUP=100

run_stan_eval_halfD() {
    local data_name=$1
    local stan_name=$2
    local half_D=$3

    echo ""
    echo "=========================================="
    echo "[stan] ${stan_name} (D=${half_D})"
    echo "=========================================="

    python stan/scripts/run_inference.py \
        --data-bundle OUTPUT/generated_data/${data_name}/data_bundle.json \
        --chains $STAN_CHAINS \
        --iter-sampling $STAN_ITER \
        --iter-warmup $STAN_WARMUP \
        --stan-arg D=$half_D \
        --run-name ${stan_name} \
        --overwrite-existing-data </dev/null

    python stan/scripts/evaluate_predictions.py \
        --data-bundle OUTPUT/generated_data/${data_name}/data_bundle.json \
        --mcmc-dir OUTPUT/domain_model/runs/${stan_name} \
        --run-name ${stan_name}_eval \
        --overwrite-existing-data </dev/null
}

# Data D=4 -> Stan D=2
run_stan_eval_halfD "tcp_misspec_log_D4_J9_data"  "tcp_misspec_log_D4_J9_stan1c_halfD"  8
run_stan_eval_halfD "tcp_misspec_log_D4_J12_data" "tcp_misspec_log_D4_J12_stan1c_halfD" 8

# Data D=6 -> Stan D=3
run_stan_eval_halfD "tcp_misspec_log_D6_J32_data" "tcp_misspec_log_D6_J32_stan1c_halfD" 12
run_stan_eval_halfD "tcp_misspec_logistic_D6_J32_data" "tcp_misspec_logistic_D6_J32_stan1c_halfD" 12

# Data D=8 -> Stan D=4
run_stan_eval_halfD "tcp_misspec_log_D8_J32_data" "tcp_misspec_log_D8_J32_stan1c_halfD" 16

echo ""
echo "All Stan 1C half-D runs complete."
