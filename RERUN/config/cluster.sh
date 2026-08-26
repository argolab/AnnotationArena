#!/bin/bash
# Shared configuration for every RERUN job. Sourced by jobs/*.sbatch and by
# submit_all.sh / make_figures.sh.
#
# Everything cluster-specific lives here. To move to another cluster, edit this
# file only -- the job scripts and the driver read all site settings from it.

# ---------------------------------------------------------------- repo layout
# RERUN/config/cluster.sh -> RERUN/config -> RERUN -> repo root
RERUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${RERUN_DIR}/.." && pwd)"
RANKING_ROOT="${REPO_ROOT}/imputer/ranking"
RUNS_ROOT="${RERUN_DIR}/runs"

export RERUN_DIR REPO_ROOT RANKING_ROOT RUNS_ROOT

# ------------------------------------------------------------------- skipjack
# sbatch is rejected without this (site billing plugin).
SBATCH_ACCEPT_COST="--comment=accept_cost"

# Marformer training is CPU-bound, not GPU-bound: an "epoch" is 5 batch-1
# full-graph steps, and most of the wall clock goes into building the entity
# graph in Python. Measured on the Item_T_200 split, 20 epochs (2026-08-25):
#
#     l40s  L40S        23s   mean GPU util  9%  (max 24%)
#     h100  H100 80GB   26s   mean GPU util 11%  (max 30%)
#     a100  A100 80GB   30s   mean GPU util 11%  (max 29%)
#
# The GPU is idle ~90% of the time, so the GPU class barely matters and a
# faster one buys nothing. What matters is how many tasks can run at once, so
# list several partitions and let Slurm start the job wherever there is room.
# (This also means multi-GPU is pointless here -- there is no batch dimension
# to shard, and DDP over the 5 mask augmentations would change the
# optimisation, not just its speed.)
# Keep this a SINGLE partition. A comma-separated list makes most array tasks
# come back "user env retrieval failed requeued held" on this cluster and they
# have to be released by hand; one partition submits cleanly.
# h100, despite the benchmark below putting l40s marginally ahead. Two of the
# l40s nodes threw "uncorrectable ECC error" mid-training on 2026-08-26 (gl110
# killed a run outright, gl105 killed a smoke test), which matches the bad-ECC
# l40s node the lingua-2 scripts already exclude. Since the job is CPU-bound the
# GPU-class difference is ~13% and mostly node contention anyway, so stability
# is the better trade. Keep this a SINGLE partition -- a comma-separated list
# makes most array tasks come back held.
GPU_PARTITION="${GPU_PARTITION:-h100}"
CPU_PARTITION="${CPU_PARTITION:-med}"

# Prefer plain --mem= : --mem-per-cpu is capped per partition on skipjack.
GPU_CPUS="${GPU_CPUS:-8}"
GPU_MEM="${GPU_MEM:-96G}"
GPU_TIME="${GPU_TIME:-12:00:00}"

CPU_CPUS="${CPU_CPUS:-8}"
CPU_MEM="${CPU_MEM:-64G}"
CPU_TIME="${CPU_TIME:-2-00:00:00}"   # MaxTime is 3 days everywhere

# gh104 hands out GPUs that torch cannot initialise: nvidia-smi works and the
# device nodes are present, but cuInit() returns 999 (CUDA_ERROR_UNKNOWN), so
# torch.cuda.is_available() is False. Jobs that land there either die in
# Lightning's RNG setup or -- worse -- silently fall back to CPU. Same node the
# lingua-2 scripts exclude for the same reason.
# gl110 threw "CUDA error: uncorrectable ECC error encountered" mid-training on
# 2026-08-26 -- same failure class as the l40s node the lingua-2 scripts call
# out as bad-ECC. Excluded until it is verified healthy.
EXCLUDE_NODES="${EXCLUDE_NODES:-gh104,gl110}"

export SBATCH_ACCEPT_COST GPU_PARTITION CPU_PARTITION \
       GPU_CPUS GPU_MEM GPU_TIME CPU_CPUS CPU_MEM CPU_TIME EXCLUDE_NODES

# ---------------------------------------------------------------- environment
# The upstream scripts/DOMAIN3/common.sh does `conda activate prabhav2 || true`,
# which SILENTLY does nothing when that env is absent. We activate the real env
# first and verify it, so a missing env is a hard error instead of a job that
# runs against the wrong interpreter.
CONDA_ENV="${CONDA_ENV:-torch}"

rerun_activate_env() {
    set +u
    # A batch shell is non-interactive, so ~/.bashrc usually returns early and
    # conda is never initialised -- `conda activate` then fails with
    # "Run 'conda init' before 'conda activate'". Load the module if needed and
    # install the shell hook explicitly instead of relying on rc files.
    if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
        module load anaconda3/2024.02-1 >/dev/null 2>&1 || true
    fi
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook 2>/dev/null)" || true
        conda activate "${CONDA_ENV}" || {
            echo "FATAL: could not activate conda env '${CONDA_ENV}'." >&2
            echo "Available:" >&2; conda env list >&2
            exit 1
        }
    fi
    set -u

    # REQUIRE_GPU=1 makes a missing GPU fatal. Without it the Domain 3 runners
    # quietly fall back to --device cpu: the job "succeeds", runs ~100x slower,
    # and nothing in the results says so. A --gres=gpu job that cannot see a
    # GPU is a scheduling bug. See EXCLUDE_NODES above for the node that
    # caused this.
    REQUIRE_GPU="${REQUIRE_GPU:-0}" python - <<'PY'
import os
import sys
try:
    import torch  # noqa: F401
except Exception as exc:                      # pragma: no cover
    sys.exit(f"FATAL: torch not importable in this environment: {exc}")
available = torch.cuda.is_available()
print(f"python  : {sys.executable}")
print(f"torch   : {torch.__version__}  cuda={available}")
if available:
    print(f"gpu     : {torch.cuda.get_device_name(0)}")
elif os.environ.get("REQUIRE_GPU") == "1":
    node = os.environ.get("SLURMD_NODENAME", "this node")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    sys.exit(
        f"FATAL: REQUIRE_GPU=1 but torch.cuda.is_available() is False on {node} "
        f"(CUDA_VISIBLE_DEVICES={cvd!r}). Refusing to fall back to CPU. "
        "If this node is bad, add it to EXCLUDE_NODES in RERUN/config/cluster.sh."
    )
PY
}

rerun_require_cmdstan() {
    python - <<'PY'
import sys
try:
    import cmdstanpy
    path = cmdstanpy.cmdstan_path()
except Exception as exc:
    sys.exit(
        f"FATAL: CmdStan is not usable ({exc}).\n"
        "Install it once with:  python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'\n"
        "Stan jobs cannot run until this succeeds."
    )
print(f"cmdstan : {path}")
PY
}

export CONDA_ENV

# ------------------------------------------------------ per-run path plumbing
# Every job writes into RERUN/runs/<RUN_ID>/. Nothing is written to the repo's
# own DATA/RESULTS/PLOTS, so several runs can be in flight at once. Only
# make_figures.sh briefly binds a run to the repo root (see its header).
rerun_run_dir()      { echo "${RUNS_ROOT}/$1"; }
rerun_data_root()    { echo "${RUNS_ROOT}/$1/DATA/STAN/DOMAIN3-SIGMOID"; }
rerun_results_root() { echo "${RUNS_ROOT}/$1/RESULTS"; }
rerun_plots_root()   { echo "${RUNS_ROOT}/$1/PLOTS"; }
rerun_log_root()     { echo "${RUNS_ROOT}/$1/logs"; }

# The upstream runners resolve the repo via this variable; without it they fall
# back to Prabhav's /home/psingh54 path.
export DOMAIN3_REPO_ROOT_OVERRIDE="${RANKING_ROOT}"
export PYTHONPATH="${RANKING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Run a command, or just print it when RERUN_DRY=1. Lets the job scripts be
# exercised end-to-end (env resolution, path construction, array indexing)
# without launching training.
rerun_run() {
    if [[ -n "${RERUN_DRY:-}" ]]; then
        printf 'DRY:'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}
