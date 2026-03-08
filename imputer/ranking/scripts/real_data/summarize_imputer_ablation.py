#!/usr/bin/env python3
"""
Summarize Imputer ablation runs into a metrics table.

Reads per-epoch histories from each run directory and computes:
- test_missing_xent: K=1,5,10 averages over last K epochs
- train_xent: K=1,5,10 averages over last K epochs
- test_missing_acc: last-epoch accuracy

Output: CSV with rows = ablation runs, columns = metrics.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def discover_runs(output_root: Path, run_prefix: str) -> List[Path]:
    """Find run directories matching prefix, sorted by name (BASE before no_*)."""
    output_root = Path(output_root)
    if not output_root.exists():
        return []
    runs: List[Path] = []
    for d in output_root.iterdir():
        if d.is_dir() and d.name.startswith(run_prefix):
            runs.append(d)
    return sorted(runs, key=lambda p: p.name)


def load_training_loss_history(run_dir: Path) -> List[Dict[str, Any]]:
    """Load training_loss_history.json (per-epoch training metrics)."""
    p = run_dir / "training_loss_history.json"
    if not p.exists():
        return []
    with p.open("r") as f:
        return json.load(f)


def load_test_instance_history(run_dir: Path) -> List[Dict[str, Any]]:
    """Load test_instance_training_history.json (per-epoch test metrics)."""
    p = run_dir / "test_instance_training_history.json"
    if not p.exists():
        return []
    with p.open("r") as f:
        return json.load(f)


def extract_ablation_id(run_dir: Path, run_prefix: str) -> str:
    """Derive ablation ID from run dir name, e.g. prefix_BASE -> BASE."""
    name = run_dir.name
    if name.startswith(run_prefix):
        suffix = name[len(run_prefix) :].lstrip("_")
        return suffix if suffix else "BASE"
    return name


def last_k_mean(values: List[float], k: int) -> Optional[float]:
    """Mean of last k values. Returns None if empty."""
    if not values:
        return None
    k = min(k, len(values))
    return sum(values[-k:]) / k


def summarize_run(
    run_dir: Path,
    run_prefix: str,
) -> Dict[str, Any]:
    """Compute summary metrics for a single run."""
    train_hist = load_training_loss_history(run_dir)
    test_hist = load_test_instance_history(run_dir)

    ablation_id = extract_ablation_id(run_dir, run_prefix)

    # Training loss (rating xent) per epoch
    train_rating_losses = [
        float(e["rating_loss"])
        for e in train_hist
        if "epoch" in e and "rating_loss" in e
    ]
    # Sort by epoch in case of out-of-order saves
    train_epochs = [e.get("epoch", i) for i, e in enumerate(train_hist)]
    if train_hist and "epoch" in train_hist[0]:
        paired = sorted(zip(train_epochs, train_rating_losses), key=lambda x: x[0])
        train_rating_losses = [x[1] for x in paired]

    # Test-missing metrics per epoch
    test_missing_xents: List[float] = []
    test_missing_accs: List[float] = []
    for e in sorted(test_hist, key=lambda x: x.get("epoch", 0)):
        mm = e.get("missing_metrics") or {}
        xent = mm.get("rating_loss")
        acc = mm.get("rating_accuracy")
        if xent is not None:
            test_missing_xents.append(float(xent))
        if acc is not None:
            test_missing_accs.append(float(acc))

    row: Dict[str, Any] = {
        "ablation_id": ablation_id,
        "run_dir": str(run_dir),
    }

    # K-averaged test-missing xent
    row["test_missing_xent_last1"] = last_k_mean(test_missing_xents, 1)
    row["test_missing_xent_last5"] = last_k_mean(test_missing_xents, 5)
    row["test_missing_xent_last10"] = last_k_mean(test_missing_xents, 10)

    # K-averaged train xent
    row["train_xent_last1"] = last_k_mean(train_rating_losses, 1)
    row["train_xent_last5"] = last_k_mean(train_rating_losses, 5)
    row["train_xent_last10"] = last_k_mean(train_rating_losses, 10)

    # Test-missing accuracy (last epoch)
    row["test_missing_acc_last1"] = last_k_mean(test_missing_accs, 1) if test_missing_accs else None

    return row


def main():
    parser = argparse.ArgumentParser(description="Summarize Imputer ablation runs")
    parser.add_argument("--output-root", default="OUTPUT/IMPUTER", help="Root output directory")
    parser.add_argument(
        "--run-prefix",
        default="llm_rubric_marformer_ablation",
        help="Prefix of run directory names to include (e.g. llm_rubric_marformer_ablation)",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: OUTPUT_ROOT/llm_rubric_imputer_ablation_summary.csv)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    runs = discover_runs(output_root, args.run_prefix)

    if not runs:
        print(f"No runs found under {output_root} with prefix {args.run_prefix}")
        return

    rows: List[Dict[str, Any]] = []
    for run_dir in runs:
        try:
            row = summarize_run(run_dir, args.run_prefix)
            rows.append(row)
        except Exception as e:
            print(f"Warning: failed to summarize {run_dir}: {e}")

    if not rows:
        print("No rows to write")
        return

    # Sort: BASE first, then alphabetical
    def sort_key(r):
        aid = r["ablation_id"]
        return (0 if aid == "BASE" else 1, aid)

    rows.sort(key=sort_key)

    # Write CSV
    out_csv = args.out_csv
    if out_csv is None:
        out_csv = output_root / "llm_rubric_imputer_ablation_summary.csv"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "ablation_id",
        "test_missing_xent_last1",
        "test_missing_xent_last5",
        "test_missing_xent_last10",
        "train_xent_last1",
        "train_xent_last5",
        "train_xent_last10",
        "test_missing_acc_last1",
    ]

    with out_csv.open("w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            vals = []
            for c in cols:
                v = row.get(c)
                if v is None:
                    vals.append("")
                else:
                    vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
            f.write(",".join(vals) + "\n")

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
