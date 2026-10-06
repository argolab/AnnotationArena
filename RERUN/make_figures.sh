#!/bin/bash
# Regenerate the full paper figure set from one run's results.
#
#   ./make_figures.sh --run-id baseline-2026-08-25
#   ./make_figures.sh --run-id baseline-2026-08-25 --install
#   ./make_figures.sh --run-id x --part item --no-calibration
#   ./make_figures.sh --run-id x --submit            # run it as a GPU batch job
#
# What it does, in order:
#   1. binds RERUN/runs/<RUN_ID>/{DATA,RESULTS,PLOTS} to imputer/ranking/
#   2. runs scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py
#   3. runs scripts/LLMRubric/plot_llmrubric_cpm_sigmoid.py  (if its data exists)
#   4. collects every output under the name main.tex expects, into
#      RERUN/runs/<RUN_ID>/paper_figures/
#   5. with --install, copies that tree into Marformer-Paper/PLOTS/
#
# WHY THE BIND STEP: the plot scripts resolve every input and output path from
# Path(__file__).parents[2], i.e. imputer/ranking, with no CLI override. Rather
# than patch three plot scripts, this driver points those three names at the
# chosen run for the duration of the plotting and restores them on exit (also
# on Ctrl-C or error, via trap). It refuses to touch a real directory, so it
# can never eat results that are not symlinks it made.
#
# Consequence: only one run can be plotted at a time. Training is unaffected --
# jobs write to absolute per-run paths and can run concurrently across runs.
#
# NOTE: plot_domain3_sigmoid.py re-runs Marformer eval, so it wants a GPU. Run
# it inside an interactive GPU session (`reqgpu`) or pass --submit.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config/cluster.sh"

RUN_ID=""
PART="all"
CALIBRATION=true
CAL_VARIANT="calibration_clean"
SKIP_LLMRUBRIC=false
INSTALL=false
SUBMIT=false
DEVICE=""
EXTRA_FLAGS=()

usage() {
    sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    cat <<'EOF'

Options:
  --run-id ID              required
  --part all|item|annot    which Domain 3 synthetic half to plot (default all)
  --no-calibration         skip the calibration reliability figures
  --calibration-variant V  calibration_clean (default) | calibration_all
  --skip-llmrubric         do not attempt the real-data figures
  --install                copy results into Marformer-Paper/PLOTS/
  --submit                 sbatch this driver onto a GPU node instead
  --device cuda|cpu        forwarded to the plot scripts
  --cache                  reuse cached probability payloads (much faster)
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)              RUN_ID="$2"; shift 2 ;;
        --part)                PART="$2"; shift 2 ;;
        --no-calibration)      CALIBRATION=false; shift ;;
        --calibration-variant) CAL_VARIANT="$2"; shift 2 ;;
        --skip-llmrubric)      SKIP_LLMRUBRIC=true; shift ;;
        --install)             INSTALL=true; shift ;;
        --submit)              SUBMIT=true; shift ;;
        --device)              DEVICE="$2"; shift 2 ;;
        --cache)               EXTRA_FLAGS+=(--cache --cache-baselines); shift ;;
        -h|--help)             usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

[[ -n "${RUN_ID}" ]] || { echo "--run-id is required" >&2; usage 1; }

RUN_DIR="$(rerun_run_dir "${RUN_ID}")"
[[ -d "${RUN_DIR}" ]] || { echo "no such run: ${RUN_DIR}" >&2; exit 1; }

# ------------------------------------------------------------------ --submit
if [[ "${SUBMIT}" == true ]]; then
    args=( --run-id "${RUN_ID}" --part "${PART}" --calibration-variant "${CAL_VARIANT}" )
    [[ "${CALIBRATION}"    == false ]] && args+=( --no-calibration )
    [[ "${SKIP_LLMRUBRIC}" == true  ]] && args+=( --skip-llmrubric )
    [[ "${INSTALL}"        == true  ]] && args+=( --install )
    [[ -n "${DEVICE}" ]] && args+=( --device "${DEVICE}" )
    (( ${#EXTRA_FLAGS[@]} )) && args+=( --cache )
    mkdir -p "$(rerun_log_root "${RUN_ID}")"
    exec sbatch "${SBATCH_ACCEPT_COST}" -J "D3SIG-FIGS" \
        -p "${GPU_PARTITION}" --gres=gpu:1 -c "${GPU_CPUS}" --mem="${GPU_MEM}" -t 08:00:00 \
        -o "$(rerun_log_root "${RUN_ID}")/figures_%j.out" \
        -e "$(rerun_log_root "${RUN_ID}")/figures_%j.err" \
        --wrap "'${RERUN_DIR}/make_figures.sh' ${args[*]@Q}"
fi

rerun_activate_env

# ---------------------------------------------------------------- preflight
missing=()
[[ -d "${RUN_DIR}/DATA" ]] || missing+=( "DATA (run submit_all.sh --stage data)" )
# Count finished training runs rather than trusting the parent directory --
# the job scripts mkdir their output root before training starts, so an empty
# RESULTS/MARFORMER is the normal state of a run that has not finished.
# -L so a run assembled from symlinks counts. A variant run writes its cells to
# RESULTS/<fam>/DOMAIN3-SIGMOID/<axis>/<variant>/<seed>/<cell>, which is deeper
# than the plot scripts' marformer_root; linking each cell up to the un-tagged
# path is how such a run is plotted, and `-type d` does not match those links.
n_runs=$(find -L "$(rerun_results_root "${RUN_ID}")/MARFORMER" -maxdepth 3 -type d \
              -name '*_MARFORMER' 2>/dev/null | wc -l)
(( n_runs > 0 )) || missing+=( "trained Marformer runs under RESULTS/MARFORMER" )
if (( ${#missing[@]} )); then
    echo "run '${RUN_ID}' is not ready to plot -- missing:" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    echo "check job status with: sacct -X -u \$USER --name=D3SIG-Marformer-item" >&2
    exit 1
fi

# ------------------------------------------------------------- bind / unbind
BOUND=()
bind_name() {
    local name="$1" target="$2"
    local link="${RANKING_ROOT}/${name}"
    if [[ -e "${link}" && ! -L "${link}" ]]; then
        echo "REFUSING to bind: ${link} exists and is a real directory." >&2
        echo "Move or remove it first -- this driver only manages symlinks." >&2
        exit 1
    fi
    mkdir -p "${target}"
    ln -sfn "${target}" "${link}"
    BOUND+=( "${link}" )
}

unbind_all() {
    local link
    for link in "${BOUND[@]:-}"; do
        [[ -L "${link}" ]] && rm -f "${link}"
    done
}
trap unbind_all EXIT INT TERM

bind_name DATA    "${RUN_DIR}/DATA"
bind_name RESULTS "$(rerun_results_root "${RUN_ID}")"
bind_name PLOTS   "$(rerun_plots_root "${RUN_ID}")"

echo "bound run '${RUN_ID}' to ${RANKING_ROOT}/{DATA,RESULTS,PLOTS}"

cd "${RANKING_ROOT}"
PLOTS_ROOT="$(rerun_plots_root "${RUN_ID}")"
STAGE="${RUN_DIR}/paper_figures"
rm -rf "${STAGE}"
mkdir -p "${STAGE}/DOMAIN3-SYNTH/ITEM" "${STAGE}/DOMAIN3-SYNTH/ANNOT" "${STAGE}/REAL-LLMRUBRIC"

# --------------------------------------------- 1. Domain 3 synthetic figures
d3_flags=( --part "${PART}" )
[[ "${CALIBRATION}" == true ]] && d3_flags+=( --calibration )
[[ -n "${DEVICE}" ]] && d3_flags+=( --device "${DEVICE}" )
(( ${#EXTRA_FLAGS[@]} )) && d3_flags+=( "${EXTRA_FLAGS[@]}" )

echo
echo "== Domain 3 synthetic =========================================="
python scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py "${d3_flags[@]}"

# smECE is quoted in the paper's prose but the figure pipeline only ever renders
# it into a plot title, so after a rerun the text and the panel can drift with
# nothing to catch it. Extract it here, inside the run binding and from the same
# payloads the panels were drawn from, so one command produces both.
if [[ "${CALIBRATION}" == true && ( "${PART}" == all ) ]]; then
    echo
    echo "== smECE ======================================================="
    python "${RERUN_DIR}/analysis/extract_smece.py" --run "${RUN_ID}" \
        --item-size 500 --annot-size 30 || echo "  (smECE extraction failed; figures are unaffected)"
fi

collect() {  # collect <src> <dst-relative-to-stage>
    if [[ -f "$1" ]]; then
        cp "$1" "${STAGE}/$2"
        echo "  ok   $2"
    else
        echo "  MISS $2   (expected $1)"
    fi
}

# The trailing field is the size whose calibration panel the paper reports.
# 4000 / 40, not 500 / 30: those are the largest sizes at which EVERY series in the
# panel was fitted, so the comparison is complete. Stan Oracle has no annotator fit
# past J=40 and Marformer NT has no annotator refit past 40, which is why the
# annotator panel is 40 and not 50. At 500 / 30 the item panel showed Marformer
# WORSE than Stan beyond its CI; at 4000 / 40 all three series are on par. See
# RERUN/analysis/smece_table.py for the full size-by-method grid.
for spec in "item:ITEM:ITEM_GENERALIZATION:4000" "annot:ANNOT:ANNOT_GENERALIZATION:40"; do
    IFS=':' read -r slug tex_dir plot_dir cal_size <<< "${spec}"
    [[ "${PART}" == "all" || "${PART}" == "${slug}" ]] || continue
    src="${PLOTS_ROOT}/DOMAIN3-SIGMOID/${plot_dir}"
    for fig in generalization_log_loss_clean_nt generalization_mbr_l2_clean_nt \
               generalization_log_loss generalization_mbr_l2 generalization_runtime; do
        collect "${src}/${slug}_${fig}.png" "DOMAIN3-SYNTH/${tex_dir}/${slug}_${fig}.png"
    done
    if [[ "${CALIBRATION}" == true ]]; then
        collect "${src}/${CAL_VARIANT}/${slug}_calibration_size${cal_size}.png" \
                "DOMAIN3-SYNTH/${tex_dir}/${slug}_calibration_size${cal_size}.png"
    fi
done

# ------------------------------------------------- 2. Domain 3 real data
if [[ "${SKIP_LLMRUBRIC}" == false ]]; then
    echo
    echo "== Domain 3 real data (LLM-Rubric) ============================="
    if [[ ! -d "${RUN_DIR}/DATA/LLM_RUBRIC" ]]; then
        echo "  SKIP: ${RUN_DIR}/DATA/LLM_RUBRIC is missing."
        echo "        These bundles have no generation script; copy them from"
        echo "        Prabhav or the old cluster. See FIGURES.md."
    else
        # plot_llmrubric_cpm_sigmoid.py reads RESULTS/STAN/CPMSigmoid, but the
        # run scripts write RESULTS/STAN/LLM_RUBRIC/CPMSigmoid. Bridge it.
        legacy="$(rerun_results_root "${RUN_ID}")/STAN/LLM_RUBRIC/CPMSigmoid"
        expected="$(rerun_results_root "${RUN_ID}")/STAN/CPMSigmoid"
        if [[ -d "${legacy}" && ! -e "${expected}" ]]; then
            ln -sfn "${legacy}" "${expected}"
            echo "  bridged RESULTS/STAN/CPMSigmoid -> LLM_RUBRIC/CPMSigmoid"
        fi

        llm_flags=()
        [[ -n "${DEVICE}" ]] && llm_flags+=( --device "${DEVICE}" )
        (( ${#EXTRA_FLAGS[@]} )) && llm_flags+=( --cache )
        python scripts/LLMRubric/plot_llmrubric_cpm_sigmoid.py "${llm_flags[@]}"

        src="${PLOTS_ROOT}/LLMRUBRIC-CPMSIGMOID/ITEM_GENERALIZATION"
        for fig in llmrubric_cpm_log_loss llmrubric_cpm_mbr_l2 llmrubric_cpm_runtime; do
            collect "${src}/${fig}.png" "REAL-LLMRUBRIC/${fig}.png"
        done
        for size in 10 175; do
            collect "${src}/calibration/llmrubric_cpm_calibration_size${size}.png" \
                    "REAL-LLMRUBRIC/llmrubric_cpm_calibration_size${size}.png"
        done
    fi
fi

# ------------------------------------------------------------- 3. install
echo
echo "== Summary ====================================================="
found=$(find "${STAGE}" -name '*.png' | wc -l)
echo "collected ${found} figure(s) -> ${STAGE}"

if [[ "${INSTALL}" == true ]]; then
    PAPER_PLOTS="${REPO_ROOT}/Marformer-Paper/PLOTS"
    if [[ ! -d "${PAPER_PLOTS}" ]]; then
        echo "cannot install: ${PAPER_PLOTS} not found (clone the Overleaf repo first)" >&2
        exit 1
    fi
    cp -r "${STAGE}/." "${PAPER_PLOTS}/"
    echo "installed into ${PAPER_PLOTS}"
    echo "review with: git -C '${REPO_ROOT}/Marformer-Paper' status --short"
    echo "then commit and push to Overleaf."
fi

echo
echo "Figures with no generating code (Domains 1 & 2, SMAR, Logit Lens) are"
echo "listed in RERUN/FIGURES.md and are NOT produced by this driver."
