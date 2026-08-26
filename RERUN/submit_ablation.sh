#!/bin/bash
# Residual-stream ablation.
#
# The dual-stream Marformer carries two residual streams at once: an inner one
# on `combined` (combined += attn; combined += ff) and an outer one that adds
# proj_out/W_param of that same `combined` back onto features/params. This
# sweeps three ways of ending a layer:
#
#   all       features += proj_out(combined); params += W_param(combined)
#             -- the shipped architecture, both streams live
#   not_last  same, except the final layer emits only [proj_out; W_param](combined)
#   none      every layer emits only the rotated part; one inner residual stream
#
# Every arm trains on the SAME splits (--data-from), and each (size, variant) is
# repeated over several seeds, because single-seed differences at these sizes are
# well inside run-to-run noise.
#
#   ./submit_ablation.sh --run-id resid-01 --data-from replicate-01
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config/cluster.sh"

EXCLUDE_ARG=""
[[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"

RUN_ID=""
DATA_FROM=""
VARIANTS="all,not_last,none"
# Residual adds inside the block, swept alongside the outer variants. A cell is
# named <outer>+<inner>; INNER_RESIDUAL="both" reproduces the shipped block.
INNER_RESIDUALS="both"
SIZES="100,300,500"
SEEDS="42,43,44"
AXIS="item"
PROTOCOL="transductive"
DRY_RUN=false

usage() {
    cat <<EOF
usage: $0 --run-id ID --data-from ID [options]
  --variants   comma list of all|not_last|none|none_id|not_last_id
                                                 (default: ${VARIANTS})
  --inner      comma list of both|ff_only|none   (default: ${INNER_RESIDUALS})
  --sizes      comma list of training sizes      (default: ${SIZES})
  --seeds      comma list of seeds               (default: ${SEEDS})
  --axis       item | annot                      (default: ${AXIS})
  --protocol   transductive | nontransductive    (default: ${PROTOCOL})
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)    RUN_ID="$2"; shift 2 ;;
        --data-from) DATA_FROM="$2"; shift 2 ;;
        --variants)  VARIANTS="$2"; shift 2 ;;
        --inner)     INNER_RESIDUALS="$2"; shift 2 ;;
        --sizes)     SIZES="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --axis)      AXIS="$2"; shift 2 ;;
        --protocol)  PROTOCOL="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done
[[ -n "${RUN_ID}"    ]] || { echo "--run-id is required" >&2; exit 1; }
[[ -n "${DATA_FROM}" ]] || { echo "--data-from is required" >&2; exit 1; }

DATA_DIR="$(rerun_data_root "${DATA_FROM}")"
[[ -d "${DATA_DIR}" ]] || { echo "no data at ${DATA_DIR}" >&2; exit 1; }

RUN_DIR="$(rerun_run_dir "${RUN_ID}")"
LOG_DIR="$(rerun_log_root "${RUN_ID}")"
CONF_DIR="${RUN_DIR}/conf"
mkdir -p "${LOG_DIR}" "${CONF_DIR}"

IFS=',' read -r -a SIZE_LIST    <<< "${SIZES}"
IFS=',' read -r -a SEED_LIST    <<< "${SEEDS}"
IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS}"
IFS=',' read -r -a INNER_LIST   <<< "${INNER_RESIDUALS}"

echo "ablation  : ${#VARIANT_LIST[@]} outer x ${#INNER_LIST[@]} inner x ${#SIZE_LIST[@]} sizes x ${#SEED_LIST[@]} seeds"
echo "data from : ${DATA_DIR}"
echo

for variant in "${VARIANT_LIST[@]}"; do
  for inner in "${INNER_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        # "both" is the shipped inner block, so leave those cells named by the
        # outer variant alone -- that keeps resid-01's paths valid.
        if [[ "${inner}" == "both" ]]; then
            cell="${variant}"
        else
            cell="${variant}+inner_${inner}"
        fi
        name="ABL-${cell}-s${seed}"
        conf="${CONF_DIR}/${name}.env"
        {
            printf '%s=%q\n' RUN_ID "${RUN_ID}"
            printf '%s=%q\n' DATA_RUN_ID "${DATA_FROM}"
            printf '%s=%q\n' FAMILY "Marformer"
            printf '%s=%q\n' AXIS "${AXIS}"
            printf '%s=%q\n' PROTOCOL "${PROTOCOL}"
            printf '%s=%q\n' SIZES "${SIZES}"
            printf '%s=%q\n' VARIANT_TAG "${cell}/seed${seed}"
            printf '%s=%q\n' DOMAIN3_DUAL_OUTER_RESIDUAL "${variant}"
            printf '%s=%q\n' DOMAIN3_INNER_RESIDUAL "${inner}"
            printf '%s=%q\n' DOMAIN3_SEED "${seed}"
        } > "${conf}"

        # See submit_all.sh: --export=ALL,VAR=value is rejected by this cluster,
        # so RERUN_CONF travels in the submitting environment instead.
        export RERUN_CONF="${conf}"
        args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
               -o "${LOG_DIR}/${name}_%A_%a.out"
               -e "${LOG_DIR}/${name}_%A_%a.err"
               --export=ALL
               -p "${GPU_PARTITION}" -c "${GPU_CPUS}" --mem="${GPU_MEM}"
               -t "${GPU_TIME}" ${EXCLUDE_ARG}
               --array="0-$(( ${#SIZE_LIST[@]} - 1 ))%3"
               "${RERUN_DIR}/jobs/marformer_train.sbatch" )
        if [[ "${DRY_RUN}" == true ]]; then
            echo "sbatch ${args[*]}"
        else
            jid="$(sbatch --parsable "${args[@]}")"
            echo "  ${name}: ${#SIZE_LIST[@]} tasks -> job ${jid}"
        fi
    done
  done
done

echo
echo "run dir : ${RUN_DIR}"
