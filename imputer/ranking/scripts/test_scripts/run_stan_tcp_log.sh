#!/bin/bash
# Run Stan 1C inference + evaluation on the 4 completed log-space TCP experiments.
set -e

STAN_CHAINS=1
STAN_ITER=300
STAN_WARMUP=100

run_stan_eval() {
    local data_name=$1
    local stan_name=$2

    echo ""
    echo "=========================================="
    echo "[stan] ${stan_name}"
    echo "=========================================="

    python stan/scripts/run_inference.py \
        --data-bundle OUTPUT/generated_data/${data_name}/data_bundle.json \
        --chains $STAN_CHAINS \
        --iter-sampling $STAN_ITER \
        --iter-warmup $STAN_WARMUP \
        --run-name ${stan_name} \
        --overwrite-existing-data </dev/null

    python stan/scripts/evaluate_predictions.py \
        --data-bundle OUTPUT/generated_data/${data_name}/data_bundle.json \
        --mcmc-dir OUTPUT/domain_model/runs/${stan_name} \
        --run-name ${stan_name}_eval \
        --overwrite-existing-data </dev/null
}

# run_stan_eval "tcp_misspec_log_D4_J9_data"  "tcp_misspec_log_D4_J9_stan1c"
# run_stan_eval "tcp_misspec_log_D4_J12_data" "tcp_misspec_log_D4_J12_stan1c"
# run_stan_eval "tcp_misspec_log_D6_J32_data" "tcp_misspec_log_D6_J32_stan1c"
# run_stan_eval "tcp_misspec_log_D8_J32_data" "tcp_misspec_log_D8_J32_stan1c"
run_stan_eval "tcp_misspec_logistic_D6_J32_data" "tcp_misspec_logistic_D6_J32_stan1c"

echo ""
echo "All Stan 1C runs complete."
