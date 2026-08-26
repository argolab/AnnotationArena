#!/bin/bash
# Submit the full DOMAIN3-SIGMOID experiment matrix for one run.
#
#   ./submit_all.sh --run-id baseline-2026-08-25
#   ./submit_all.sh --run-id try-L12 --data-from baseline-2026-08-25
#   ./submit_all.sh --run-id x --stage marformer --dry-run
#
# The matrix below is exactly what the paper reports -- one array task per
# (model, axis, protocol, training size). The upstream repo spelled the same
# matrix out as 24 near-identical per-size scripts under
# scripts/DOMAIN3-SIGMOID/**/run_sizes_*.sh; those size groupings existed only
# to fit the old cluster's walltime and are not reproduced here.
#
# Job graph:
#   generate_data ──┬─> marformer_train[item,T]   ──┐
#                   ├─> marformer_train[annot,T]  ──┼─> marformer_eval[Marformer]
#                   ├─> marformer_train[item,NT]  ──┐
#                   ├─> marformer_train[annot,NT] ──┴─> marformer_eval[MarformerNT]
#                   └─> stan_fit[...]  (8 independent arrays, no eval step)
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config/cluster.sh"

# Nodes whose GPUs torch cannot initialise (see EXCLUDE_NODES in cluster.sh).
EXCLUDE_ARG=""
[[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"

RUN_ID=""
STAGE="all"
DATA_FROM=""
DRY_RUN=false

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    cat <<'EOF'

Options:
  --run-id ID      required; results land in RERUN/runs/ID/
  --stage S        all (default) | data | marformer | stan | eval
                   | llmrubric | llmrubric-eval   (real LLM-Rubric arm)
  --data-from ID   reuse another run's generated data instead of regenerating
  --dry-run        print the sbatch commands without submitting
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)    RUN_ID="$2"; shift 2 ;;
        --stage)     STAGE="$2"; shift 2 ;;
        --data-from) DATA_FROM="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        -h|--help)   usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

[[ -n "${RUN_ID}" ]] || { echo "--run-id is required" >&2; usage 1; }
case "${STAGE}" in all|data|marformer|stan|eval|llmrubric|llmrubric-eval) ;; *) echo "bad --stage: ${STAGE}" >&2; exit 1 ;; esac

RUN_DIR="$(rerun_run_dir "${RUN_ID}")"
LOG_DIR="$(rerun_log_root "${RUN_ID}")"
mkdir -p "${RUN_DIR}" "${LOG_DIR}" "$(rerun_results_root "${RUN_ID}")"

# ---------------------------------------------------------------- the matrix
# family | axis | protocol | sizes
# The paper sweeps item generalization to K_train=4000 for Marformer; Stan stops
# at 500 (HMC cost). The NT arm and both annotator arms follow the paper too.
MARFORMER_MATRIX=(
  "Marformer|item|transductive|100,200,300,400,500,600,700,800,900,1000,2000,3000,4000"
  "Marformer|annot|transductive|10,15,20,25,30,35,40,45,50"
  "MarformerNT|item|nontransductive|100,200,300,400,500,600,700,800,900"
  "MarformerNT|annot|nontransductive|10,15,20,25,30,35,40"
)

# family | model label | model tag | stan file | axis | sizes
# Stan Oracle is the correctly specified generator; the three misspecified
# variants share thresholds (Bin), share the annotator projection (Proj), or
# both. Item sizes stop at 500 and Oracle/annot at 40 -- beyond that MCMC did
# not finish in the paper, and the plot script extends those curves as dotted.
STAN_MATRIX=(
  "StanOracle|Stan Oracle Sigmoid|TENSOR-SIG|STAN/stan_models/tensor_model_sigmoid.stan|item|100,200,300,400,500"
  "StanOracle|Stan Oracle Sigmoid|TENSOR-SIG|STAN/stan_models/tensor_model_sigmoid.stan|annot|10,15,20,25,30,35,40"
  "StanMisspecs|Stan Bin|BIN|STAN/stan_models/tensor_shared_annotator_binning_raw_z_sigmoid.stan|item|100,200,300,400,500"
  "StanMisspecs|Stan Bin|BIN|STAN/stan_models/tensor_shared_annotator_binning_raw_z_sigmoid.stan|annot|10,15,20,25,30,35,40,45,50"
  "StanMisspecs|Stan Proj|PROJ|STAN/stan_models/tensor_shared_annotator_projection_sigmoid.stan|item|100,200,300,400,500"
  "StanMisspecs|Stan Proj|PROJ|STAN/stan_models/tensor_shared_annotator_projection_sigmoid.stan|annot|10,15,20,25,30,35,40,45,50"
  "StanMisspecs|Stan Both|BOTH|STAN/stan_models/tensor_shared_annotator_projection_binning_raw_z_sigmoid.stan|item|100,200,300,400,500"
  "StanMisspecs|Stan Both|BOTH|STAN/stan_models/tensor_shared_annotator_projection_binning_raw_z_sigmoid.stan|annot|10,15,20,25,30,35,40,45,50"
)

# ------------------------------------------------------------------ plumbing
CONF_DIR="${RUN_DIR}/conf"
mkdir -p "${CONF_DIR}"

# Job parameters travel in a per-job env file rather than through --export.
# sbatch's --export splits its value on commas, so a size list like
# "100,200,300" would arrive as three separate empty variables, and a label
# like "Stan Oracle Sigmoid" would be mangled by the spaces. Sourcing a file
# sidesteps both.
write_conf() {
    # write_conf <name> <KEY=VALUE>... ; echoes the conf path
    local name="$1"; shift
    local path="${CONF_DIR}/${name}.env"
    : > "${path}"
    local kv
    for kv in "$@"; do
        printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "${path}"
    done
    echo "${path}"
}

submit() {
    # submit <job-name> <conf-path> <extra-sbatch-args...> ; echoes the job id
    local name="$1" conf="$2"; shift 2
    # Pass RERUN_CONF through the submitting environment rather than as
    # --export=ALL,RERUN_CONF=...  Any "ALL,VAR=value" export spec is currently
    # rejected by this cluster's slurmd with "user env retrieval failed
    # requeued held", and releasing such a job just makes it fail again; plain
    # --export=ALL works, and ALL already carries whatever we export here.
    export RERUN_CONF="${conf}"
    local -a args=( "${SBATCH_ACCEPT_COST}" -J "${name}"
                    -o "${LOG_DIR}/${name}_%A_%a.out"
                    -e "${LOG_DIR}/${name}_%A_%a.err"
                    --export=ALL "$@" )
    if [[ "${DRY_RUN}" == true ]]; then
        echo "sbatch ${args[*]}" >&2
        echo "DRYRUN"
        return
    fi
    sbatch --parsable "${args[@]}"
}

count_sizes() { local IFS=','; read -r -a a <<< "$1"; echo "${#a[@]}"; }
dep() { [[ -n "$1" && "$1" != "DRYRUN" ]] && echo "--dependency=afterok:$1" || echo ""; }

# ------------------------------------------------------------------ 1. data
GEN_JOB=""
if [[ -n "${DATA_FROM}" ]]; then
    SRC="$(rerun_run_dir "${DATA_FROM}")/DATA"
    [[ -d "${SRC}" ]] || { echo "no DATA under run '${DATA_FROM}'" >&2; exit 1; }
    ln -sfn "${SRC}" "${RUN_DIR}/DATA"
    echo "data      : reusing ${DATA_FROM}"
elif [[ "${STAGE}" == "all" || "${STAGE}" == "data" ]]; then
    conf="$(write_conf "gen" "RUN_ID=${RUN_ID}")"
    GEN_JOB="$(submit "D3SIG-GEN" "${conf}" -p "${CPU_PARTITION}" -c "${CPU_CPUS}" \
                      --mem="${CPU_MEM}" -t "${CPU_TIME}" \
                      "${RERUN_DIR}/jobs/generate_data.sbatch")"
    echo "data      : job ${GEN_JOB}"
fi

# ------------------------------------------------------------- 2. marformer
declare -A EVAL_DEPS=( [Marformer]="" [MarformerNT]="" )
if [[ "${STAGE}" == "all" || "${STAGE}" == "marformer" ]]; then
    for row in "${MARFORMER_MATRIX[@]}"; do
        IFS='|' read -r family axis protocol sizes <<< "${row}"
        n="$(count_sizes "${sizes}")"
        name="D3SIG-${family}-${axis}"
        conf="$(write_conf "${name}" "RUN_ID=${RUN_ID}" "FAMILY=${family}" \
                           "AXIS=${axis}" "PROTOCOL=${protocol}" "SIZES=${sizes}")"
        jid="$(submit "${name}" "${conf}" -p "${GPU_PARTITION}" -c "${GPU_CPUS}" \
                      --mem="${GPU_MEM}" -t "${GPU_TIME}" ${EXCLUDE_ARG} \
                      --array="0-$((n-1))" $(dep "${GEN_JOB}") \
                      "${RERUN_DIR}/jobs/marformer_train.sbatch")"
        echo "marformer : ${name} (${n} tasks) job ${jid}"
        EVAL_DEPS[${family}]="${EVAL_DEPS[${family}]:+${EVAL_DEPS[${family}]}:}${jid}"
    done
fi

# ------------------------------------------------------------------ 3. stan
if [[ "${STAGE}" == "all" || "${STAGE}" == "stan" ]]; then
    for row in "${STAN_MATRIX[@]}"; do
        IFS='|' read -r family label tag stanfile axis sizes <<< "${row}"
        n="$(count_sizes "${sizes}")"
        name="D3SIG-${tag}-${axis}"
        conf="$(write_conf "${name}" "RUN_ID=${RUN_ID}" "FAMILY=${family}" \
                           "MODEL_LABEL=${label}" "MODEL_TAG=${tag}" \
                           "STAN_FILE=${stanfile}" "AXIS=${axis}" "SIZES=${sizes}")"
        jid="$(submit "${name}" "${conf}" -p "${CPU_PARTITION}" -c "${CPU_CPUS}" \
                      --mem="${CPU_MEM}" -t "${CPU_TIME}" \
                      --array="0-$((n-1))" $(dep "${GEN_JOB}") \
                      "${RERUN_DIR}/jobs/stan_fit.sbatch")"
        echo "stan      : ${name} (${n} tasks) job ${jid}"
    done
fi

# ------------------------------------------------------------------ 4. eval
# ---------------------------------------------------------------- LLM-Rubric
# The real-data arm. Its bundles come from jobs/make_llmrubric_data.sh, not from
# the synthetic generator, so this stage is deliberately not part of "all".
#
# No throttle needed: the "user env retrieval failed requeued held" state that
# looked like an array-size problem was actually --export=ALL,VAR=value (see
# submit()); with that fixed, all ten start concurrently.
LLMRUBRIC_SIZES="10,20,30,40,50,75,100,125,150,175"

if [[ "${STAGE}" == "llmrubric" ]]; then
    n="$(count_sizes "${LLMRUBRIC_SIZES}")"
    conf="$(write_conf "LLMR-Marformer" "RUN_ID=${RUN_ID}" "SIZES=${LLMRUBRIC_SIZES}")"
    jid="$(submit "LLMR-Marformer" "${conf}" -p "${GPU_PARTITION}" -c "${GPU_CPUS}" \
                  --mem="${GPU_MEM}" -t "${GPU_TIME}" ${EXCLUDE_ARG} \
                  --array="0-$((n-1))" \
                  "${RERUN_DIR}/jobs/llmrubric_train.sbatch")"
    echo "llmrubric : Marformer (${n} tasks) job ${jid}"
    LLMR_TRAIN_JOB="${jid}"
fi

if [[ "${STAGE}" == "llmrubric-eval" ]]; then
    n="$(count_sizes "${LLMRUBRIC_SIZES}")"
    conf="$(write_conf "LLMR-EVAL" "RUN_ID=${RUN_ID}" "SIZES=${LLMRUBRIC_SIZES}")"
    jid="$(submit "LLMR-EVAL" "${conf}" -p "${GPU_PARTITION}" -c "${GPU_CPUS}" \
                  --mem="${GPU_MEM}" -t "${GPU_TIME}" ${EXCLUDE_ARG} \
                  --array="0-$((n-1))" \
                  "${RERUN_DIR}/jobs/llmrubric_eval.sbatch")"
    echo "llmr-eval : ${n} tasks job ${jid}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "eval" ]]; then
    for family in Marformer MarformerNT; do
        d=""
        [[ -n "${EVAL_DEPS[${family}]}" && "${EVAL_DEPS[${family}]}" != *DRYRUN* ]] \
            && d="--dependency=afterok:${EVAL_DEPS[${family}]}"
        conf="$(write_conf "D3SIG-EVAL-${family}" "RUN_ID=${RUN_ID}" "FAMILY=${family}")"
        jid="$(submit "D3SIG-EVAL-${family}" "${conf}" -p "${GPU_PARTITION}" -c "${GPU_CPUS}" \
                      --mem="${GPU_MEM}" -t "${GPU_TIME}" ${EXCLUDE_ARG} ${d} \
                      "${RERUN_DIR}/jobs/marformer_eval.sbatch")"
        echo "eval      : ${family} job ${jid}"
    done
fi

# -------------------------------------------------------------- 5. manifest
if [[ "${DRY_RUN}" != true ]]; then
    cat > "${RUN_DIR}/manifest.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "stage": "${STAGE}",
  "data_from": "${DATA_FROM}",
  "submitted_at": "$(date -Iseconds)",
  "submitted_by": "${USER}",
  "git_commit": "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_dirty": $(git -C "${REPO_ROOT}" diff --quiet 2>/dev/null && echo false || echo true),
  "gpu_partition": "${GPU_PARTITION}",
  "cpu_partition": "${CPU_PARTITION}",
  "conda_env": "${CONDA_ENV}"
}
EOF
    echo
    echo "run dir   : ${RUN_DIR}"
    echo "watch     : squeue -u \$USER -o '%.18i %.9P %.28j %.2t %.10M %R'"
    echo "then      : ${RERUN_DIR}/make_figures.sh --run-id ${RUN_ID}"
fi
