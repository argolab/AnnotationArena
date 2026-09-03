#!/bin/bash
# Refit Stan on the fresh generative draws.
#
# The published Stan curves are seed-42 measurements digitized off the paper's
# figures -- no Domain 3 sigmoid Stan output survived the migration. That made
# every Marformer-vs-Stan comparison on a fresh draw a cross-draw comparison,
# with the +/-0.085 across-draw spread (D28) sitting between the two series.
# Refitting Stan on the same draw removes that entirely: both arms then see the
# identical data and the identical corner.
#
# Sampler settings and the model matrix are taken verbatim from submit_all.sh,
# so these fits are comparable with the paper's: 1 chain, 300 warmup, 500
# sampling. Item sizes stop at 500, where the paper stopped for HMC cost.
#
#   ./submit_stan.sh [--seeds 101,105] [--sizes 100,200,300,400,500] [--dry-run]
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

SEEDS="101,105"; SIZES="100,200,300,400,500"; AXIS="item"; DRY_RUN=false
MODELS="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) SEEDS="$2"; shift 2 ;;
        --sizes) SIZES="$2"; shift 2 ;;
        --axis)  AXIS="$2";  shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done

# family | label | tag | stan file  -- verbatim from submit_all.sh's STAN_MATRIX
STAN_MATRIX=(
  "StanOracle|Stan Oracle Sigmoid|TENSOR-SIG|STAN/stan_models/tensor_model_sigmoid.stan"
  "StanMisspecs|Stan Bin|BIN|STAN/stan_models/tensor_shared_annotator_binning_raw_z_sigmoid.stan"
  "StanMisspecs|Stan Proj|PROJ|STAN/stan_models/tensor_shared_annotator_projection_sigmoid.stan"
  "StanMisspecs|Stan Both|BOTH|STAN/stan_models/tensor_shared_annotator_projection_binning_raw_z_sigmoid.stan"
)
n=$(awk -F, '{print NF}' <<< "${SIZES}")
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
count=0
for seed in "${SEED_LIST[@]}"; do
    RUN_ID="dataseed-s${seed}"
    CONF_DIR="$(rerun_run_dir "${RUN_ID}")/conf"; LOG_DIR="$(rerun_log_root "${RUN_ID}")"
    mkdir -p "${CONF_DIR}" "${LOG_DIR}"
    for row in "${STAN_MATRIX[@]}"; do
        IFS='|' read -r fam label tag stanfile <<< "${row}"
        [[ "${MODELS}" != "all" && ",${MODELS}," != *",${tag},"* ]] && continue
        name="STAN-${tag}-${AXIS}-s${seed}"
        conf="${CONF_DIR}/${name}.env"; : > "${conf}"
        for kv in "RUN_ID=${RUN_ID}" "FAMILY=${fam}" "MODEL_LABEL=${label}" \
                  "MODEL_TAG=${tag}" "STAN_FILE=${stanfile}" \
                  "AXIS=${AXIS}" "SIZES=${SIZES}"; do
            printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${conf}"
        done
        export RERUN_CONF="${conf}"
        args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
               -o "${LOG_DIR}/${name}_%A_%a.out" -e "${LOG_DIR}/${name}_%A_%a.err"
               --export=ALL -p "${CPU_PARTITION}" -c "${CPU_CPUS}"
               --mem="${CPU_MEM}" -t "1-00:00:00"
               --array="0-$((n-1))" "${RERUN_DIR}/jobs/stan_fit.sbatch" )
        if [[ "${DRY_RUN}" == true ]]; then echo "  ${name}: (dry) ${n} tasks"
        else echo "  ${name}: job $(sbatch --parsable "${args[@]}")  (${n} tasks)"; fi
        count=$((count+1))
    done
done
echo "submitted ${count} arrays x ${n} sizes = $((count*n)) fits"
