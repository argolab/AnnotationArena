#!/bin/bash
#SBATCH --job-name=RESID-CORNER
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --comment=accept_cost
set -euo pipefail
# Re-score the residual ablation on the TEST CORNER only.
#
# The report's original numbers came from the pooled metric (D22): every missing
# token in the graph, most of which sits on training-partition items. The corner
# is the 414 cells the paper says it reports. Both the learning curves and the
# final table are rebuilt on it here.
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
OUT="${RERUN_DIR}/reports/build"; mkdir -p "${OUT}"

ARGS=()
for spec in "resid-01:all" "resid-01:none" "resid-02:none_id" "resid-03:all+inner_none"; do
    run="${spec%%:*}"; var="${spec##*:}"
    for seed in 42 43 44; do
        d="$(rerun_run_dir "${run}")/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/${var}/seed${seed}/DOMAIN3-SIGMOID_Item_T_${SIZE:-500}_MARFORMER"
        [[ -d "${d}" ]] && ARGS+=( --run-dir "${d}" )
    done
done
echo "${#ARGS[@]} args"
rerun_run python "${RERUN_DIR}/fig6/corner_curve.py" "${ARGS[@]}" \
    --device "${EVAL_DEVICE:-cuda}" \
    -o "${OUT}/resid_corner_curves_K${SIZE:-500}.json"
echo done
