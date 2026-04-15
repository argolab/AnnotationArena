#!/usr/bin/env bash
#
# STAN MARFORMER_ANNOT_DROP — evaluate annotator-test runs (10 total)
# trained by scripts/STAN/stan_data_command_marformer_annotator.sh.
#
# Runs ranked_eval (k=1,3,5,7 + last) for Factor/Normal annotator-test splits,
# then prints vertical tables and writes PNG reports under:
#   RESULTS/MARFORMER_ANNOT_DROP/STAN/reports/
# including a two-curve plot (Factor vs Normal) analogous to the right panel in
# the 4-curve STAN figure.
#
# Usage:
#   bash scripts/STAN/eval_stan_marformer_annotator_10_tmp.sh
#   bash scripts/STAN/eval_stan_marformer_annotator_10_tmp.sh 1        # Factor only
#   bash scripts/STAN/eval_stan_marformer_annotator_10_tmp.sh 2        # Normal only
#   bash scripts/STAN/eval_stan_marformer_annotator_10_tmp.sh summary  # reports only
#   FORCE_RERUN=1 bash scripts/STAN/eval_stan_marformer_annotator_10_tmp.sh  # ignore cache
#
# Slurm example:
#   cd ~/AA_new/imputer/ranking
#   PARTITION=a100 GPUS=1 TIME=03:00:00 CPUS_PER_TASK=4 MEM_PER_CPU=8G \
#     /home/xwang397/bin/sbatch_adapt scripts/STAN/eval_stan_marformer_annotator_10_tmp.sh
#

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKING_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_RANKING_ROOT}"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -euo pipefail

OUTPUT_ROOT="RESULTS/MARFORMER_ANNOT_DROP/STAN"

G1=(
  "Factor_250_20_9_AnnotatorTest_12"
  "Factor_250_20_9_AnnotatorTest_14"
  "Factor_250_20_9_AnnotatorTest_3"
  "Factor_250_20_9_AnnotatorTest_6"
  "Factor_250_20_9_AnnotatorTest_9"
)
G2=(
  "Normal_250_20_9_AnnotatorTest_12"
  "Normal_250_20_9_AnnotatorTest_14"
  "Normal_250_20_9_AnnotatorTest_3"
  "Normal_250_20_9_AnnotatorTest_6"
  "Normal_250_20_9_AnnotatorTest_9"
)

run_ranked_eval_for() {
  local run="$1"
  local rd="${OUTPUT_ROOT}/${run}"
  local rank_json="${rd}/RANKED_RESULTS/by_val_missing_xent.json"
  echo ""
  echo "--- ${run} ---"
  if [[ ! -d "${rd}" ]]; then
    echo "  SKIP: no run dir: ${rd}"
    return 0
  fi
  if [[ ! -f "${rd}/train_config.json" ]]; then
    echo "  SKIP: no train_config.json"
    return 0
  fi
  if ! compgen -G "${rd}/checkpoints/*.ckpt" > /dev/null; then
    echo "  SKIP: no checkpoints"
    return 0
  fi
  if [[ -f "${rank_json}" && -z "${FORCE_RERUN:-}" ]]; then
    echo "  CACHE: ${rank_json} exists (set FORCE_RERUN=1 to recompute)"
    return 0
  fi
  python -u -m imputer.entity_mf.ranked_eval \
    --run-dir "${rd}" \
    --ranks 1,3,5,7 \
    --device cuda
}

run_group() {
  local idx="$1"
  case "${idx}" in
    1) local -n _runs=G1 ;;
    2) local -n _runs=G2 ;;
    *) echo "Invalid group: ${idx}"; exit 1 ;;
  esac
  for run in "${_runs[@]}"; do
    run_ranked_eval_for "${run}"
  done
}

MODE="${1:-all}"

if [[ "${MODE}" == "summary" ]]; then
  :
elif [[ "${MODE}" == "all" ]]; then
  echo "============================================================"
  echo " STAN MARFORMER_ANNOT_DROP | ranked_eval | both annotator groups"
  echo "============================================================"
  run_group 1
  run_group 2
elif [[ "${MODE}" =~ ^[12]$ ]]; then
  echo "============================================================"
  echo " STAN MARFORMER_ANNOT_DROP | ranked_eval | group ${MODE} only"
  echo "============================================================"
  run_group "${MODE}"
else
  echo "Usage: $0 [ 1 | 2 | all | summary ]"
  exit 1
fi

echo ""
echo " STAN annotator-drop summary — tables + lineplots under ${OUTPUT_ROOT}/reports/"
python -u << 'PY'
from __future__ import annotations

import json
from pathlib import Path

from imputer.entity_mf import ranked_eval_lineplot as rl
from imputer.entity_mf import ranked_eval_report as rr

root = Path("RESULTS/MARFORMER_ANNOT_DROP/STAN")
rep = root / "reports"
rep.mkdir(parents=True, exist_ok=True)

g1 = [
    "Factor_250_20_9_AnnotatorTest_12",
    "Factor_250_20_9_AnnotatorTest_14",
    "Factor_250_20_9_AnnotatorTest_3",
    "Factor_250_20_9_AnnotatorTest_6",
    "Factor_250_20_9_AnnotatorTest_9",
]
g2 = [
    "Normal_250_20_9_AnnotatorTest_12",
    "Normal_250_20_9_AnnotatorTest_14",
    "Normal_250_20_9_AnnotatorTest_3",
    "Normal_250_20_9_AnnotatorTest_6",
    "Normal_250_20_9_AnnotatorTest_9",
]


def _train_missing_xent_at_k1_epoch(root: Path, run: str, ranked: dict | None) -> float | None:
    """Training missing-rating xent at the k=1 checkpoint epoch (if available)."""
    if not ranked:
        return None
    rd = root / run
    e = rl.k1_checkpoint_epoch_index(rd, ranked)
    if e is None:
        return None
    hp = rd / "training_history.json"
    if not hp.is_file():
        return None
    try:
        with open(hp) as f:
            hist = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(hist, list):
        return None
    by_ep = {}
    for h in hist:
        if not isinstance(h, dict) or "epoch" not in h:
            continue
        try:
            by_ep[int(h["epoch"])] = h
        except (TypeError, ValueError):
            continue
    h = by_ep.get(int(e))
    if not isinstance(h, dict):
        return None
    tr = h.get("train_eval", {})
    xv = (
        tr.get("metrics", {})
        .get("missing", {})
        .get("rating", {})
        .get("xent")
    )
    if xv is None:
        return None
    try:
        return float(xv)
    except (TypeError, ValueError):
        return None

print()
print("=" * 80)
print(" STAN MARFORMER_ANNOT_DROP — vertical tables (k=1,3,5,7 + last)")
print("=" * 80)
for title, runs in [
    ("Factor_250_20_9_AnnotatorTest", g1),
    ("Normal_250_20_9_AnnotatorTest", g2),
]:
    print()
    print(f"══ {title} ({len(runs)} runs) " + "═" * max(0, 60 - len(title)))
    for run in runs:
        st = rr.run_status_line(root, run)
        rr.print_vertical_run(f"{run}  [{st}]", rr.load_ranked(root, run))

# Table PNGs (one per family)
out1 = rep / "ranked_vertical_Factor_250_20_9_AnnotatorTest_ANNOT_DROP.png"
out2 = rep / "ranked_vertical_Normal_250_20_9_AnnotatorTest_ANNOT_DROP.png"
if rr.save_stan_family_figure(out1, root, "Factor_250_20_9_AnnotatorTest (ANNOT_DROP)", g1):
    print(f"Wrote figure: {out1}")
if rr.save_stan_family_figure(out2, root, "Normal_250_20_9_AnnotatorTest (ANNOT_DROP)", g2):
    print(f"Wrote figure: {out2}")

# Family lineplots (single-curve) for completeness
lp1 = rep / "val_xent_k1_vs_train_size_Factor_250_20_9_AnnotatorTest_ANNOT_DROP.png"
lp2 = rep / "val_xent_k1_vs_train_size_Normal_250_20_9_AnnotatorTest_ANNOT_DROP.png"
if rl.save_stan_family_lineplot(root, "Factor_250_20_9_AnnotatorTest (ANNOT_DROP)", g1, lp1):
    print(f"Wrote lineplot: {lp1}")
if rl.save_stan_family_lineplot(root, "Normal_250_20_9_AnnotatorTest (ANNOT_DROP)", g2, lp2):
    print(f"Wrote lineplot: {lp2}")

# Two-curve annotator comparison (Factor vs Normal) — similar to right panel of 4-curve plot
plt = rl._try_plt()
out_two = rep / "val_xent_k1_two_curve_annotator_STAN_ANNOT_DROP.png"
if plt is None:
    print("(matplotlib not available — skipped two-curve lineplot)")
else:
    x1, y1, s1, c1, xk1, _ = rl._gather_curve_points(root, g1)
    x2, y2, s2, c2, xk2, _ = rl._gather_curve_points(root, g2)
    # Keep x-aligned with _gather_curve_points sorting (run suffixes are J_train values).
    g1_sorted = sorted(g1, key=lambda n: int(n.split("_")[-1]))
    g2_sorted = sorted(g2, key=lambda n: int(n.split("_")[-1]))
    t1 = [
        _train_missing_xent_at_k1_epoch(root, run, rr.load_ranked(root, run))
        for run in g1_sorted
    ]
    t2 = [
        _train_missing_xent_at_k1_epoch(root, run, rr.load_ranked(root, run))
        for run in g2_sorted
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    any_hollow = False
    if x1:
        any_hollow = rl._plot_curve_line_and_markers(
            ax, x1, y1, s1, c1, color="#1f77b4", label="Factor (annotator-drop)", fill_alpha=0.18, ms=6
        ) or any_hollow
    if x2:
        any_hollow = rl._plot_curve_line_and_markers(
            ax, x2, y2, s2, c2, color="#ff7f0e", label="Normal (annotator-drop)", fill_alpha=0.18, ms=6
        ) or any_hollow

    # Complementary signal: training missing-rating xent at the same k=1 checkpoint epoch.
    # Draw on a separate y-axis to avoid compressing the validation curves.
    ax2 = ax.twinx()
    has_train_line = False
    if x1 and len(t1) == len(x1) and any(v is not None for v in t1):
        ax2.plot(
            x1,
            [float("nan") if v is None else float(v) for v in t1],
            "--",
            color="#1f77b4",
            lw=1.8,
            alpha=0.85,
            label="Factor train missing xent (dashed)",
        )
        has_train_line = True
    if x2 and len(t2) == len(x2) and any(v is not None for v in t2):
        ax2.plot(
            x2,
            [float("nan") if v is None else float(v) for v in t2],
            "--",
            color="#ff7f0e",
            lw=1.8,
            alpha=0.85,
            label="Normal train missing xent (dashed)",
        )
        has_train_line = True

    xk = xk1 or xk2
    if xk == "J_train":
        ax.set_xlabel("Training annotators J_train", fontsize=10)
    elif xk == "K_train":
        ax.set_xlabel("Training items K_train", fontsize=10)
    else:
        ax.set_xlabel("Training size", fontsize=10)
    ax.set_ylabel("Val missing-rating xent (nats)", fontsize=10)
    if has_train_line:
        ax2.set_ylabel("Train missing-rating xent (nats, dashed)", fontsize=10)
    ax.set_title("STAN ANNOT_DROP — annotator-test protocol (Factor vs Normal)", fontsize=11)
    ax.grid(True, alpha=0.3)
    if any_hollow:
        ax.plot(
            [], [], "o",
            markerfacecolor="white",
            markeredgecolor="#444444",
            markeredgewidth=2.0,
            ms=6,
            label="Incomplete (hollow)",
        )
    if x1 or x2 or any_hollow or has_train_line:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    if not x1 and not x2:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    fig.subplots_adjust(bottom=0.2)
    extra = rl.FOOTNOTE_HOLLOW if any_hollow else ""
    rl._footnote(fig, extra=extra)
    try:
        fig.savefig(out_two, dpi=160, bbox_inches="tight", facecolor="white")
        print(f"Wrote lineplot: {out_two}")
    except Exception as e:
        print(f"(two-curve lineplot write failed: {e})")
    plt.close(fig)

print()
print(f"JSON: {root}/<run>/RANKED_RESULTS/by_val_missing_xent.json")
print(f"Reports: {rep}")
PY
