#!/bin/bash
# Phase 1 of the observation-density thread: three densities x six generative
# draws, Marformer (search -> select -> refit -> eval) and four Stan arms, all
# at K=500 on the item axis, transductive. See PLAN.md for the design.
#
#   ./submit_phase1.sh [--stage all|data|search|select|refit|eval|corner|stan|bridge] [--dry-run]
#
# Everything is queued in one shot with SLURM dependencies, so the chain runs to
# completion unattended. The array axis is the CELL, not the size: 18 cells at
# one size would otherwise be 18 arrays of 1 task, and the account only runs
# ~10 jobs at a time, so they would spend the sweep queueing. CELL_RUN_IDS in
# jobs/{marformer_train,select_epochs,tree_eval,stan_fit}.sbatch does that.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/config/cluster.sh"

THREAD_DIR="${RERUN_DIR}/threads/obs-density"
CONF_DIR="${THREAD_DIR}/conf"
LOG_DIR="${THREAD_DIR}/logs"
mkdir -p "${CONF_DIR}" "${LOG_DIR}"

STAGE="all"; DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)   STAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Overridable so one cell can be smoke-tested through the same code path that
# runs the real thing -- a chain this long should not first execute at full size.
#   OBSDEN_DRAWS="s101" OBSDEN_RATES="0.15" OBSDEN_SEEDS="42" \
#   OBSDEN_SEARCH_EPOCHS=20 ./submit_phase1.sh
read -r -a DRAWS <<< "${OBSDEN_DRAWS:-s42 s101 s102 s103 s104 s105}"
read -r -a RATES <<< "${OBSDEN_RATES:-0.15 0.50 0.85}"
read -r -a SEEDS <<< "${OBSDEN_SEEDS:-42 43}"
SIZE="${OBSDEN_SIZE:-500}"
MASK_SEED="${OBSDEN_MASK_SEED:-20261017}"
VAL_PROTOCOL="${OBSDEN_VAL_PROTOCOL:-mcar}"
VAL_FRAC="${OBSDEN_VAL_FRAC:-0.15}"
VAL_SPLIT_SEED="${OBSDEN_VAL_SPLIT_SEED:-7}"
SEARCH_EPOCHS="${OBSDEN_SEARCH_EPOCHS:-2000}"
# Run-id prefix. A smoke test must not write into the cells the real run owns.
PREFIX="${OBSDEN_PREFIX:-obsden}"
# Suffix for this submission's job, conf and log names. The conf path is derived
# from the stage name alone, so submitting the SAME stage twice in quick
# succession makes the second invocation overwrite the first's conf before its
# job starts -- both then run the second cell list. That silently happened to
# the two bridge selects on 2026-10-17. Tag any submission that reuses a stage.
TAG="${OBSDEN_TAG:+-${OBSDEN_TAG}}"
VARIANT="val${VAL_PROTOCOL}"
EPOCH_TABLE_NAME="${VARIANT}_epochs.json"

# Cell order must match make_data.sbatch's flat index: rate is the fast axis.
CELLS=()
for d in "${DRAWS[@]}"; do
    for r in "${RATES[@]}"; do
        CELLS+=( "${PREFIX}-${d}-o$(printf '%02d' "$(awk -v r="${r}" 'BEGIN{printf "%d", r*100}')")" )
    done
done
CELL_CSV="$(IFS=,; echo "${CELLS[*]}")"
# OBSDEN_CELLS overrides the computed list outright. That is how the bridge
# runs get their select/refit/corner stages: their run ids carry a suffix
# (-blk, -vf30) that the draw x rate product cannot express.
if [[ -n "${OBSDEN_CELLS:-}" ]]; then
    CELL_CSV="${OBSDEN_CELLS}"
    IFS=',' read -r -a CELLS <<< "${CELL_CSV}"
fi
N_CELLS="${#CELLS[@]}"
LAST=$((N_CELLS - 1))

# Stan's cost scales with the OBSERVED count, so it is cheapest at the sparsest
# cell: measured 1m2s to fit the 15%-observed cell (544 observed ratings). The
# 85% cell has ~3080 and is still minutes, so 4h is ample and gets better
# backfill priority than the 12h the other Stan jobs in this repo ask for.
STAN_ARMS=(
  "StanOracle|Stan Oracle Sigmoid|TENSOR-SIG|STAN/stan_models/tensor_model_sigmoid.stan"
  "StanMisspecs|Stan Bin|BIN|STAN/stan_models/tensor_shared_annotator_binning_raw_z_sigmoid.stan"
  "StanMisspecs|Stan Proj|PROJ|STAN/stan_models/tensor_shared_annotator_projection_sigmoid.stan"
  "StanMisspecs|Stan Both|BOTH|STAN/stan_models/tensor_shared_annotator_projection_binning_raw_z_sigmoid.stan"
)

write_conf() {
    local name="$1"; shift
    local path="${CONF_DIR}/${name}.env"
    : > "${path}"
    local kv
    for kv in "$@"; do printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${path}"; done
    echo "${path}"
}
# --export=ALL only, never --export=ALL,VAR=value: the latter comes back as
# "user env retrieval failed requeued held" on this cluster. Job parameters go
# through the conf file instead.
submit() {
    local name="$1" conf="$2"; shift 2
    export RERUN_CONF="${conf}"
    local -a args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
                    -o "${LOG_DIR}/${name}_%A_%a.out" -e "${LOG_DIR}/${name}_%A_%a.err"
                    --export=ALL "$@" )
    if [[ "${DRY_RUN}" == true ]]; then echo "sbatch ${args[*]}" >&2; echo "DRYRUN"; return; fi
    sbatch --parsable "${args[@]}"
}
dep() { [[ -n "$1" && "$1" != "DRYRUN" ]] && echo "--dependency=afterok:$1" || echo ""; }
# "bridge" is deliberately excluded from "all": it symlinks the main cells'
# DATA, which does not exist until the data stage has actually finished.
run()  { [[ "$1" != "bridge" && "${STAGE}" == "all" ]] || [[ "${STAGE}" == "$1" ]]; }

echo "cells (${N_CELLS}): ${CELL_CSV}"
echo

# ------------------------------------------------------------------- data
DATA_JOB=""
if run data; then
    conf="$(write_conf "OBSDEN-DATA${TAG}" "OBSDEN_DRAWS=${DRAWS[*]}" "OBSDEN_RATES=${RATES[*]}" \
                       "OBSDEN_SIZE=${SIZE}" "OBSDEN_MASK_SEED=${MASK_SEED}" \
                       "OBSDEN_PREFIX=${PREFIX}")"
    DATA_JOB="$(submit "OBSDEN-DATA${TAG}" "${conf}" -p "${CPU_PARTITION}" \
                       --array="0-${LAST}" "${THREAD_DIR}/make_data.sbatch")"
    echo "data   : ${N_CELLS} tasks, job ${DATA_JOB}"
fi

# ----------------------------------------------------------------- search
SEARCH_JOBS=""
if run search; then
    for seed in "${SEEDS[@]}"; do
        name="OBSDEN-S-s${seed}${TAG}"
        conf="$(write_conf "${name}" "CELL_RUN_IDS=${CELL_CSV}" \
                   "FAMILY=Marformer" "AXIS=item" "PROTOCOL=transductive" "SIZES=${SIZE}" \
                   "DOMAIN3_SEED=${seed}" "VARIANT_TAG=${VARIANT}-search/seed${seed}" \
                   "DOMAIN3_VAL_PROTOCOL=${VAL_PROTOCOL}" "DOMAIN3_VAL_FRAC=${VAL_FRAC}" \
                   "DOMAIN3_VAL_SPLIT_SEED=${VAL_SPLIT_SEED}" \
                   "DOMAIN3_EPOCHS=${SEARCH_EPOCHS}")"
        jid="$(submit "${name}" "${conf}" -p "${GPU_PARTITION}" -c "${GPU_CPUS}" \
                      --mem="${GPU_MEM}" -t 2:00:00 --exclude="${EXCLUDE_NODES}" \
                      --array="0-${LAST}" $(dep "${DATA_JOB}") \
                      "${RERUN_DIR}/jobs/marformer_train.sbatch")"
        echo "search : seed ${seed}, ${N_CELLS} tasks, job ${jid}"
        SEARCH_JOBS="${SEARCH_JOBS:+${SEARCH_JOBS}:}${jid}"
    done
fi

# ----------------------------------------------------------------- select
SELECT_JOB=""
if run select; then
    conf="$(write_conf "OBSDEN-SELECT${TAG}" "CELL_RUN_IDS=${CELL_CSV}" "VARIANT=${VARIANT}" \
               "EPOCH_TABLE_NAME=${EPOCH_TABLE_NAME}" \
               "SELECT_MODE=per-run" "SELECT_SMOOTH=5")"
    SELECT_JOB="$(submit "OBSDEN-SELECT${TAG}" "${conf}" -p "${CPU_PARTITION}" -c 2 --mem=8G \
                  -t 0:30:00 --array="0-${LAST}" $(dep "${SEARCH_JOBS}") \
                  "${RERUN_DIR}/jobs/select_epochs.sbatch")"
    echo "select : ${N_CELLS} tasks, job ${SELECT_JOB}"
fi

# ------------------------------------------------------------------ refit
REFIT_JOBS=""
if run refit; then
    for seed in "${SEEDS[@]}"; do
        name="OBSDEN-R-s${seed}${TAG}"
        conf="$(write_conf "${name}" "CELL_RUN_IDS=${CELL_CSV}" \
                   "FAMILY=Marformer" "AXIS=item" "PROTOCOL=transductive" "SIZES=${SIZE}" \
                   "DOMAIN3_SEED=${seed}" "VARIANT_TAG=${VARIANT}/seed${seed}" \
                   "DOMAIN3_VAL_PROTOCOL=none" "EPOCH_TABLE_NAME=${EPOCH_TABLE_NAME}")"
        jid="$(submit "${name}" "${conf}" -p "${GPU_PARTITION}" -c "${GPU_CPUS}" \
                      --mem="${GPU_MEM}" -t 2:00:00 --exclude="${EXCLUDE_NODES}" \
                      --array="0-${LAST}" $(dep "${SELECT_JOB}") \
                      "${RERUN_DIR}/jobs/marformer_train.sbatch")"
        echo "refit  : seed ${seed}, ${N_CELLS} tasks, job ${jid}"
        REFIT_JOBS="${REFIT_JOBS:+${REFIT_JOBS}:}${jid}"
    done
fi

# ------------------------------------------------------------------- eval
if run eval; then
    # EXCLUDE=-search: only the refit models are reportable, and scoring the
    # search tree first just delays the numbers that matter.
    conf="$(write_conf "OBSDEN-EVAL${TAG}" "CELL_RUN_IDS=${CELL_CSV}" "EXCLUDE=-search")"
    jid="$(submit "OBSDEN-EVAL${TAG}" "${conf}" -p "${GPU_PARTITION}" -c "${GPU_CPUS}" \
                  --mem="${GPU_MEM}" -t 2:00:00 --exclude="${EXCLUDE_NODES}" \
                  --array="0-${LAST}" $(dep "${REFIT_JOBS}") \
                  "${RERUN_DIR}/jobs/tree_eval.sbatch")"
    echo "eval   : ${N_CELLS} tasks, job ${jid}"
fi

# ----------------------------------------------------------------- corner
# THIS is the number that gets compared to Stan. tree_eval's "missing" metric
# pools the held-out corner with the training partition's own missing cells
# (D22), while Stan is scored on the test block alone -- building a table from
# tree_eval output directly reproduces D23, one table over two populations.
# split_missing_metrics.py --corner-whole gives the corner-only number.
if run corner; then
    conf="$(write_conf "OBSDEN-CORNER${TAG}" "CELL_RUN_IDS=${CELL_CSV}")"
    # 24G, not the 120G jobs/corner_score.sbatch's header asks for. `med` has
    # MaxMemPerCPU=4000, so Slurm turns --mem=120G into a THIRTY-ONE core
    # request and the task then waits for a node with 31 free cores -- two of
    # these sat PENDING for half an hour on 2026-10-17 for exactly that reason.
    # K=500 scoring needs a fraction of it (the bundle is 2.4 MB).
    jid="$(submit "OBSDEN-CORNER${TAG}" "${conf}" -p "${CPU_PARTITION}" -c 4 \
                  --mem="${OBSDEN_CORNER_MEM:-24G}" \
                  -t 2:00:00 --array="0-${LAST}" $(dep "${REFIT_JOBS}") \
                  "${RERUN_DIR}/jobs/corner_score.sbatch")"
    echo "corner : ${N_CELLS} tasks, job ${jid}"
fi

# ------------------------------------------------------------------- stan
if run stan; then
    for row in "${STAN_ARMS[@]}"; do
        IFS='|' read -r family label tag stanfile <<< "${row}"
        name="OBSDEN-STAN-${tag}${TAG}"
        conf="$(write_conf "${name}" "CELL_RUN_IDS=${CELL_CSV}" \
                   "FAMILY=${family}" "MODEL_LABEL=${label}" "MODEL_TAG=${tag}" \
                   "STAN_FILE=${stanfile}" "AXIS=item" "SIZES=${SIZE}")"
        jid="$(submit "${name}" "${conf}" -p "${CPU_PARTITION}" -c "${CPU_CPUS}" \
                      --mem="${CPU_MEM}" -t 4:00:00 --array="0-${LAST}" \
                      $(dep "${DATA_JOB}") "${RERUN_DIR}/jobs/stan_fit.sbatch")"
        echo "stan   : ${tag}, ${N_CELLS} tasks, job ${jid}"
    done
fi

# ----------------------------------------------------------------- bridge
# Two diagnostics from PLAN.md, run on draw s101 only. They need the main
# cells' DATA, so this stage is submitted AFTER --stage data has landed.
#
#   -blk   the 50% cell under the frozen `block` carve instead of `mcar`, so
#          the switch of carve is evidenced rather than asserted.
#   -vf30  the 15% cell at --val-frac 0.30, because the mcar carve is only ~67
#          cells there and the selected epoch should not depend on that.
if run bridge; then
    for spec in "o50|blk|block|${VAL_FRAC}" "o15|vf30|${VAL_PROTOCOL}|0.30"; do
        IFS='|' read -r rate_tag suffix proto frac <<< "${spec}"
        src="${PREFIX}-s101-${rate_tag}"
        dst="${src}-${suffix}"
        src_data="$(rerun_run_dir "${src}")/DATA"
        if [[ ! -d "${src_data}" ]]; then
            echo "bridge : SKIP ${dst} -- ${src_data} does not exist yet (run --stage data first)" >&2
            continue
        fi
        mkdir -p "$(rerun_run_dir "${dst}")"
        ln -sfn "${src_data}" "$(rerun_run_dir "${dst}")/DATA"
        variant="val${proto}"
        name="OBSDEN-BR-${suffix}${TAG}"
        conf="$(write_conf "${name}-S" "CELL_RUN_IDS=${dst}" \
                   "FAMILY=Marformer" "AXIS=item" "PROTOCOL=transductive" "SIZES=${SIZE}" \
                   "DOMAIN3_SEED=${SEEDS[0]}" "VARIANT_TAG=${variant}-search/seed${SEEDS[0]}" \
                   "DOMAIN3_VAL_PROTOCOL=${proto}" "DOMAIN3_VAL_FRAC=${frac}" \
                   "DOMAIN3_VAL_SPLIT_SEED=${VAL_SPLIT_SEED}" \
                   "DOMAIN3_EPOCHS=${SEARCH_EPOCHS}")"
        jid="$(submit "${name}-S" "${conf}" -p "${GPU_PARTITION}" -c "${GPU_CPUS}" \
                      --mem="${GPU_MEM}" -t 2:00:00 --exclude="${EXCLUDE_NODES}" \
                      --array="0-0" "${RERUN_DIR}/jobs/marformer_train.sbatch")"
        echo "bridge : ${dst} carve=${proto} frac=${frac}, search job ${jid}"
        echo "         (select/refit/corner for the bridge: rerun this script's"
        echo "          later stages with OBSDEN_DRAWS/RATES narrowed, or score"
        echo "          the search curve directly -- the selected epoch is the"
        echo "          comparison, not the refit)"
    done
fi

echo
echo "thread dir : ${THREAD_DIR}"
echo "logs       : ${LOG_DIR}"
