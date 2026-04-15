#!/usr/bin/env bash
#
# Temporary one-shot: evaluate the three part-B MARFORMER runs (750 / 1000 / 1280)
# using imputer.entity_mf.ranked_eval only:
#   - Scores every *.ckpt on VAL (missing rating xent), sorts ascending (lower = better).
#   - For k = 1, 3, 5, 7: full VAL + TEST metrics (missing xent / rmse / Spearman, etc.).
#   - Also last.ckpt (same VAL+TEST metrics; use --no-last to skip).
#   - Writes RANKED_RESULTS/by_val_missing_xent.json
# (imputer.entity_mf.test is redundant here unless you want Lightning best/last files only.)
# Summary A: vertical tables + RESULTS/MARFORMER/SUMMEVAL/reports/ranked_eval_vertical.png
#   + val_xent_k1_vs_train_size.png (k=1 val xent vs K_train; SD band; ranked_eval_report).
# Plus a CPU-only read of training_history.json (per-epoch val missing CE, min + last).
#
# Run on a GPU node (test uses cuda when available). Same cwd convention as train:
#
#   cd /path/to/AA_new/imputer/ranking
#   PARTITION=h100 GPUS=1 TIME=02:00:00 CPUS_PER_TASK=4 MEM_PER_CPU=8G \
#     /home/xwang397/bin/sbatch_adapt scripts/SUMMEVAL/MARFORMER/TRAIN/eval_train_b_750_1000_1280_tmp.sh
#
# Or interactively on a compute node:
#   bash scripts/SUMMEVAL/MARFORMER/TRAIN/eval_train_b_750_1000_1280_tmp.sh
#
# ── Only SummEval_1600_8_4_1000 (from imputer/ranking, GPU) ───────────────────
#   cd /path/to/AA_new/imputer/ranking && export PYTHONPATH=. && \
#   python -u -m imputer.entity_mf.ranked_eval \
#     --run-dir RESULTS/MARFORMER/SUMMEVAL/SummEval_1600_8_4_1000 --ranks 1,3,5,7 --device cuda
#
# ── Only SummEval_1600_8_4_1280 (larger split) ────────────────────────────────
#   cd /path/to/AA_new/imputer/ranking && export PYTHONPATH=. && \
#   python -u -m imputer.entity_mf.ranked_eval \
#     --run-dir RESULTS/MARFORMER/SUMMEVAL/SummEval_1600_8_4_1280 --ranks 1,3,5,7 --device cuda
#

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKING_ROOT="$(cd "${_SCRIPT_DIR}/../../../.." && pwd)"
cd "${_RANKING_ROOT}"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -euo pipefail

OUTPUT_ROOT="RESULTS/MARFORMER/SUMMEVAL"
SPLITS=(
  "SummEval_1600_8_4_750"
  "SummEval_1600_8_4_1000"
  "SummEval_1600_8_4_1280"
)

echo "============================================================"
echo " MARFORMER SummEval part-B | ranked_eval (val+test @ k=1,3,5,7) | root: ${_RANKING_ROOT}"
echo "============================================================"

for SPLIT in "${SPLITS[@]}"; do
  RUN_DIR="${OUTPUT_ROOT}/${SPLIT}"
  echo ""
  echo "--- ${SPLIT} ---"
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "  SKIP: run dir missing: ${RUN_DIR}"
    continue
  fi
  if [[ ! -f "${RUN_DIR}/train_config.json" ]]; then
    echo "  SKIP: no train_config.json in ${RUN_DIR}"
    continue
  fi
  python -u -m imputer.entity_mf.ranked_eval \
    --run-dir "${RUN_DIR}" \
    --ranks 1,3,5,7 \
    --device cuda
done

echo ""
echo "============================================================"
echo " Summary A — vertical tables (rows = k=1,3,5,7 + last) + PNG figure"
echo "  Text: below | Figure: RESULTS/MARFORMER/SUMMEVAL/reports/ranked_eval_vertical.png"
echo "============================================================"

python -u -m imputer.entity_mf.ranked_eval_report --mode summeval

echo ""
echo "============================================================"
echo " Summary C — training_history.json (epoch-end val missing rating xent; monitor=val/missing_ce)"
echo "============================================================"

python -u << 'PY'
from __future__ import annotations

import json
from pathlib import Path

root = Path("RESULTS/MARFORMER/SUMMEVAL")
splits = [
    "SummEval_1600_8_4_750",
    "SummEval_1600_8_4_1000",
    "SummEval_1600_8_4_1280",
]


def val_missing_xent(ep: dict):
    ve = ep.get("val_eval")
    if not ve:
        return None
    return (
        ve.get("metrics", {})
        .get("missing", {})
        .get("rating", {})
        .get("xent")
    )


def summarize_history(run_name: str) -> dict | None:
    path = root / run_name / "training_history.json"
    if not path.is_file():
        return None
    with open(path) as f:
        hist = json.load(f)
    xs = []
    for ep in hist:
        x = val_missing_xent(ep)
        if x is None:
            continue
        xs.append((int(ep.get("epoch", -1)), float(x)))
    if not xs:
        return {"run": run_name, "error": "no val_eval in history"}
    best_ep, best_x = min(xs, key=lambda t: t[1])
    last_ep, last_x = xs[-1]
    return {
        "run": run_name,
        "epochs_with_val": len(xs),
        "min_val_miss_xent": best_x,
        "min_at_epoch": best_ep,
        "last_val_miss_xent": last_x,
        "last_at_epoch": last_ep,
    }

hdr = (
    f"{'run':<28} {'ep_val':>7} {'min_ll':>9} {'@ep':>5} "
    f"{'last_ll':>9} {'last_ep':>8}"
)
print(hdr)
print("-" * len(hdr))
for name in splits:
    s = summarize_history(name)
    if s is None:
        print(f"{name:<28}  (no training_history.json)")
        continue
    if "error" in s:
        print(f"{s['run']:<28}  ({s['error']})")
        continue
    print(
        f"{s['run']:<28} {s['epochs_with_val']:>7} "
        f"{s['min_val_miss_xent']:>9.4f} {s['min_at_epoch']:>5} "
        f"{s['last_val_miss_xent']:>9.4f} {s['last_at_epoch']:>8}"
    )

print()
print("min_ll / last_ll = val split, missing rating tokens, mean xent (same as logged val/missing_ce).")
print("JSON: RANKED_RESULTS/by_val_missing_xent.json; training_history.json in run root.")
PY

echo ""
echo " Done."
