#!/bin/bash
# S8 -- does LLM-Rubric's training chunk (max_item=10) leave performance on the
# table on a 225-item bundle?
#
# Why this is a TRAINING sweep and not an eval sweep. test.py hardcodes
# max_item=500 and the bundle is 225 items, so *evaluation already sees every
# item in one chunk*. max_item=10 is a training-only setting: the model learns
# conditioning on at most 9 other items and is then tested with all 225
# available. So the only way to measure it is to retrain. (An earlier plan to
# answer this by re-scoring existing checkpoints was wrong for this reason.)
#
# Walltimes are sized per arm from a measured smoke (job 327138): at max_item
# 225 the largest size runs 1.15 s/epoch => ~6 min for 300 epochs, where
# max_item=10 takes 36 min for the same run -- the big chunk is ~6x FASTER,
# because per-chunk graph construction dominates over attention at this scale.
# Giving the fast arm the slow arm's limit would cost it backfill priority for
# nothing, so the two arms are submitted separately. Margins are ~2-3x because
# nothing here resumes: a job that hits its limit loses all of it.
#
#   ./submit_llmrubric_arms.sh [--dry-run]
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export RERUN_DIR
source "${RERUN_DIR}/config/cluster.sh"

RUN_ID="${RUN_ID:-replicate-01}"
SIZES="${SIZES:-10,50,100,175}"
SEEDS="${SEEDS:-42,43}"
PARTS="${PARTS:-h100,l40s,a100,h200}"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

RUN_DIR="$(rerun_run_dir "${RUN_ID}")"
CONF_DIR="${RUN_DIR}/conf"; LOG_DIR="$(rerun_log_root "${RUN_ID}")"
mkdir -p "${CONF_DIR}" "${LOG_DIR}"
EXCLUDE_ARG=""; [[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"
n=$(awk -F, '{print NF}' <<< "${SIZES}")

# arm tag | max_item | walltime (sized from the smoke, with margin)
ARMS=( "mi10|10|01:15:00" "mi225|225|00:20:00" )

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
echo "${RUN_ID}: ${#ARMS[@]} arms x ${#SEED_LIST[@]} seeds x ${n} sizes = $(( ${#ARMS[@]} * ${#SEED_LIST[@]} * n )) runs"
for row in "${ARMS[@]}"; do
    IFS='|' read -r tag mi wall <<< "${row}"
    for seed in "${SEED_LIST[@]}"; do
        # The seed MUST be in the variant path. It matters more here than
        # anywhere else in this project: llmrubric_train.sbatch does an
        # `rm -rf` on its target directory before training, so two runs sharing
        # a path would not collide, the second would DELETE the first.
        variant="s8_${tag}/seed${seed}"
        name="S8-${tag}-s${seed}"
        conf="${CONF_DIR}/${name}.env"
        { echo "RUN_ID=${RUN_ID}"
          echo "SIZES=${SIZES}"
          echo "LLMRUBRIC_MAX_ITEM=${mi}"
          echo "LLMRUBRIC_SEED=${seed}"
          echo "VARIANT_TAG=${variant}"
        } > "${conf}.tmp"
        mv "${conf}.tmp" "${conf}"
        args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
               -o "${LOG_DIR}/${name}_%A_%a.out" -e "${LOG_DIR}/${name}_%A_%a.err"
               --export=ALL,RERUN_CONF="${conf}" -p "${PARTS}" -c "${GPU_CPUS}"
               --mem="${GPU_MEM}" -t "${wall}" ${EXCLUDE_ARG}
               --array="0-$((n-1))" "${RERUN_DIR}/jobs/llmrubric_train.sbatch" )
        if [[ "${DRY_RUN}" == true ]]; then echo "  ${name} (dry, -t ${wall})"
        else echo "  ${name} (-t ${wall}): job $(sbatch --parsable "${args[@]}" | tail -1)"; fi
    done
done
