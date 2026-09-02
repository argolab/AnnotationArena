#!/bin/bash
# Regenerate Domain 3 from scratch under fresh GENERATIVE seeds.
#
# Everything measured so far -- including the +0.078 train-vs-corner gap -- comes
# from ONE draw of the synthetic data (base seed 42). The corner is only 414
# cells, so a large part of that gap may simply be which 414 cells seed 42 drew.
# The only way to separate "the corner is a hard draw" from "the model
# generalises worse" is to redraw the data and remeasure.
#
# Each seed gets a completely independent run tree, runs/dataseed-s<SEED>/, with
# its own base tensor and its own splits. runs/replicate-01 and runs/fig6-01 are
# never touched, so the published dots cannot move.
#
# Generation is ~72 s of CPU per seed (measured: job 111302), so the cost here is
# entirely in the training that follows.
#
#   ./submit_gen.sh [--seeds 101,102,103,104,105] [--dry-run]
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

SEEDS="101,102,103,104,105"; DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) SEEDS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done

# Only the item axis is swept, and only at the three sizes the comparison needs,
# but the size LISTS are left at the harness defaults plus 1000 on the NT side
# so each tree stays a drop-in replacement for fig6-01 if more dots are wanted
# later. Subsplitting is cheap; regenerating is what costs.
export ITEM_NT_SIZES="100,200,300,400,500,600,700,800,900,1000"

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
for seed in "${SEED_LIST[@]}"; do
    RUN_ID="dataseed-s${seed}"
    CONF_DIR="$(rerun_run_dir "${RUN_ID}")/conf"
    LOG_DIR="$(rerun_log_root "${RUN_ID}")"
    mkdir -p "${CONF_DIR}" "${LOG_DIR}"
    conf="${CONF_DIR}/gen.env"; : > "${conf}"
    for kv in "RUN_ID=${RUN_ID}" "RERUN_SEED=${seed}" \
              "ITEM_NT_SIZES=${ITEM_NT_SIZES}"; do
        printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${conf}"
    done
    export RERUN_CONF="${conf}"
    args=( "${SBATCH_ACCEPT_COST}" -J "DSEED-GEN-${seed}"
           -o "${LOG_DIR}/gen_%j.out" -e "${LOG_DIR}/gen_%j.err"
           --export=ALL -p "${CPU_PARTITION}" -c "${CPU_CPUS}"
           --mem="${CPU_MEM}" -t "03:00:00"
           "${RERUN_DIR}/jobs/generate_data.sbatch" )
    if [[ "${DRY_RUN}" == true ]]; then echo "seed ${seed}: (dry) sbatch ${args[*]}"
    else echo "seed ${seed}: job $(sbatch --parsable "${args[@]}")  -> ${RUN_ID}"; fi
done
