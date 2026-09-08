#!/bin/bash
#SBATCH --job-name=LLMR-EVALCHUNK
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --comment=accept_cost
# D30: sweep the PUBLISHED metric's evaluation chunk on existing checkpoints.
# No training, no GPU. Walltime is set on the command line from an estimate of
# ~30 s per (size, chunk) cell; see RERUN/side/llmrubric_evalchunk.py.
#
# The plot script resolves every path from imputer/ranking/{DATA,RESULTS}, so
# this binds the run's trees there exactly as make_figures.sh does -- and, like
# make_figures.sh, unbinds on any exit path so the repo is left as found.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env

# Values here contain commas (size and chunk lists), and sbatch --export splits
# on commas -- a trap this project has hit before. So parameters arrive in a
# sourced conf file, the same way every other job in RERUN/ receives them.
if [[ -n "${RERUN_CONF:-}" ]]; then
    # shellcheck disable=SC1090
    source "${RERUN_CONF}"
fi
RUN_ID="${EC_RUN:-replicate-01}"
BOUND=()
# NOTE: the trap must end truthy. `[[ -L ... ]] && rm` returns 1 when the link
# is already gone, and as the last statement of an EXIT trap under `set -e` that
# marks an otherwise successful job FAILED (job 329014 did all its work, wrote
# its output, and still reported FAILED).
unbind_all() { local l; for l in "${BOUND[@]:-}"; do [[ -L "${l}" ]] && rm -f "${l}"; done; return 0; }
trap unbind_all EXIT INT TERM
bind_name() {
    local link="${RANKING_ROOT}/$1" target="$2"
    if [[ -e "${link}" && ! -L "${link}" ]]; then
        echo "REFUSING to bind: ${link} is a real directory." >&2; exit 1
    fi
    mkdir -p "${target}"; ln -sfn "${target}" "${link}"; BOUND+=( "${link}" )
}
# THESE JOBS MUST NOT RUN CONCURRENTLY. The bind mutates shared paths inside
# imputer/ranking, so a second instance either sees the first one's tree or has
# its symlinks pulled out mid-run. Job 329015 ran alongside 329014, found no
# data, and exited "COMPLETED" after 7 seconds having written zero rows -- a
# silent empty result, which is the worst possible failure mode. Chain multiple
# invocations with --dependency=afterany, or run them one at a time.
if [[ -L "${RANKING_ROOT}/RESULTS" || -L "${RANKING_ROOT}/DATA" ]]; then
    echo "REFUSING: ${RANKING_ROOT} is already bound -- another evalchunk job is" >&2
    echo "probably running. These jobs must be serialised; see the note above." >&2
    exit 1
fi
bind_name DATA    "$(rerun_run_dir "${RUN_ID}")/DATA"
bind_name RESULTS "$(rerun_results_root "${RUN_ID}")"

mkdir -p "${RERUN_DIR}/side/out"
EC_TAG="${EC_TAG:-}"          # e.g. s8_mi225/seed42 -> score that arm instead
EC_LABEL="${EC_LABEL:-${RUN_ID}}"
ROOT_ARG=()
if [[ -n "${EC_TAG}" ]]; then
    ROOT_ARG=( --marformer-root "$(rerun_results_root "${RUN_ID}")/MARFORMER/LLM_RUBRIC/${EC_TAG}" )
fi
python -u "${RERUN_DIR}/side/llmrubric_evalchunk.py" \
    --chunks "${EC_CHUNKS:-10,25,50,100,225}" \
    --sizes "${EC_SIZES:-}" \
    --device cpu "${ROOT_ARG[@]}" \
    -o "${RERUN_DIR}/side/out/llmrubric_evalchunk_${EC_LABEL}.json"
