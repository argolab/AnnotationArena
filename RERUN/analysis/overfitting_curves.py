#!/usr/bin/env python
"""
Train-vs-held-out curves for the Marformer runs, for the overfitting thread.

For each run this reads training_history.json (per-epoch training CE, and
per-epoch validation CE where a validation split exists) and TEST_RESULTS/*.json
(per-checkpoint test log-loss), and plots them on one axis.

The gap between the training curve and the test points is the quantity Jason is
asking about: if the model were overfitting the way a heteroskedastic Gaussian
regressor does, training CE would fall well below the test curve and the test
curve would turn back up.

    python overfitting_curves.py RUNDIR... --out plots/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r"(?:epoch=)?(\d+)")


def _read_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def _epoch_of(name: str) -> Optional[int]:
    m = EPOCH_RE.findall(Path(name).stem)
    return int(m[-1]) if m else None


def _xent(record: dict, key: str) -> Optional[float]:
    node = record.get(key)
    if not node:
        return None
    v = node.get("metrics", {}).get("missing", {}).get("rating", {}).get("xent")
    return None if v is None else float(v)


def load_run(run_dir: Path) -> Optional[Dict]:
    hist_path = run_dir / "training_history.json"
    if not hist_path.exists():
        return None
    history = _read_json(hist_path)

    epochs = [int(r["epoch"]) for r in history]
    train_ce = [r.get("train_masked_ce") for r in history]
    val_ce = [_xent(r, "val_eval") for r in history]
    comb_ce = [_xent(r, "combined_eval") for r in history]

    test_points: List[Tuple[int, float]] = []
    test_dir = run_dir / "TEST_RESULTS"
    if test_dir.is_dir():
        for path in sorted(test_dir.glob("*.json")):
            if path.name in {"best.json", "launcher_metadata.json", "last.json"}:
                continue
            data = _read_json(path)
            ll = data.get("missing", {}).get("log_loss")
            ep = _epoch_of(data.get("checkpoint", path.name))
            if ll is not None and ep is not None:
                test_points.append((ep, float(ll)))
    test_points.sort()

    return {
        "name": run_dir.name,
        "epochs": epochs,
        "train_ce": train_ce,
        "val_ce": val_ce,
        "combined_ce": comb_ce,
        "test_points": test_points,
        "has_val": any(v is not None for v in val_ce),
    }


def plot_run(run: Dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    tr = [(e, c) for e, c in zip(run["epochs"], run["train_ce"]) if c is not None]
    if tr:
        ax.plot([e for e, _ in tr], [c for _, c in tr],
                label="train (masked CE)", color="#1f77b4", lw=1.8)

    va = [(e, c) for e, c in zip(run["epochs"], run["val_ce"]) if c is not None]
    if va:
        ax.plot([e for e, _ in va], [c for _, c in va],
                label="validation (missing CE)", color="#2ca02c", lw=1.8)

    if run["test_points"]:
        ax.plot([e for e, _ in run["test_points"]],
                [c for _, c in run["test_points"]],
                "o--", label="test (missing log-loss)", color="#d62728",
                ms=5, lw=1.2)
        best_ep, best_ll = min(run["test_points"], key=lambda t: t[1])
        ax.axvline(best_ep, color="#d62728", ls=":", lw=1, alpha=0.6)
        ax.annotate(f"test argmin\nepoch {best_ep}, {best_ll:.3f}",
                    xy=(best_ep, best_ll), xytext=(6, 14),
                    textcoords="offset points", fontsize=8, color="#d62728")

    if not run["has_val"]:
        ax.text(0.98, 0.04,
                "no validation split in this bundle:\n"
                "no early stopping, no validation-based selection",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#555",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#ccc"))

    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy (nats)")
    ax.set_title(run["name"], fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="Output directory for PNGs.")
    ap.add_argument("--csv", type=Path, default=None, help="Also write a summary table.")
    args = ap.parse_args()

    rows = []
    for run_dir in args.runs:
        run = load_run(run_dir)
        if run is None:
            print(f"skip (no training_history.json): {run_dir}")
            continue
        plot_run(run, args.out / f"{run['name']}.png")

        tr = [c for c in run["train_ce"] if c is not None]
        final_train = tr[-1] if tr else None
        min_train = min(tr) if tr else None
        tp = run["test_points"]
        final_test = tp[-1][1] if tp else None
        best_test = min(tp, key=lambda t: t[1])[1] if tp else None
        best_ep = min(tp, key=lambda t: t[1])[0] if tp else None
        rows.append({
            "run": run["name"],
            "has_val": run["has_val"],
            "final_train_ce": final_train,
            "min_train_ce": min_train,
            "final_test_ll": final_test,
            "best_test_ll": best_test,
            "best_test_epoch": best_ep,
            # How far training CE sits below the test log-loss at the end.
            "train_test_gap": (final_test - final_train)
                              if (final_test is not None and final_train is not None) else None,
            # Positive => test log-loss got worse after its minimum, i.e. the
            # classic overfitting signature.
            "test_rise_after_min": (final_test - best_test)
                                   if (final_test is not None and best_test is not None) else None,
        })
        print(f"{run['name']:<44} train={final_train if final_train is None else round(final_train,4)} "
              f"test_final={final_test if final_test is None else round(final_test,4)} "
              f"test_best={best_test if best_test is None else round(best_test,4)} @ep{best_ep}")

    if args.csv and rows:
        import csv as _csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")
    print(f"plots -> {args.out}")


if __name__ == "__main__":
    main()
