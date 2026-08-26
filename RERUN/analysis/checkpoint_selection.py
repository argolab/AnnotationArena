#!/usr/bin/env python
"""
Quantify how much the reported Marformer numbers owe to selecting the
checkpoint on the test set (DISCREPANCIES.md D14).

For each run directory:
  * test-selected  -- min over all evaluated checkpoints of test missing
                      log-loss. This is what run_marformer_eval.sh and
                      plot_domain3_sigmoid.py write into best.json, and hence
                      what every paper figure plots.
  * val-selected   -- the checkpoint whose *validation* missing CE is lowest,
                      scored on test. This is the honest number.
  * last           -- final epoch, no selection at all.

Validation CE per epoch comes from training_history.json; test log-loss per
checkpoint comes from TEST_RESULTS/*.json. Checkpoint filenames carry their
epoch, which is what joins the two.

    python checkpoint_selection.py RUNS...  [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

EPOCH_RE = re.compile(r"(?:epoch=)?(\d+)")


def _read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _epoch_of(checkpoint_name: str) -> Optional[int]:
    """Pull the epoch out of 'best-epoch=0123.ckpt' / 'periodic-0075.ckpt'."""
    stem = Path(checkpoint_name).stem
    matches = EPOCH_RE.findall(stem)
    return int(matches[-1]) if matches else None


def _val_ce_by_epoch(run_dir: Path) -> Dict[int, float]:
    """epoch -> validation missing-rating cross-entropy, from training history."""
    history_path = run_dir / "training_history.json"
    if not history_path.exists():
        return {}
    out: Dict[int, float] = {}
    for record in _read_json(history_path):
        val_eval = record.get("val_eval")
        if not val_eval:
            continue
        xent = val_eval.get("metrics", {}).get("missing", {}).get("rating", {}).get("xent")
        if xent is not None:
            out[int(record["epoch"])] = float(xent)
    return out


def _test_results(run_dir: Path) -> List[Dict[str, Any]]:
    """Per-checkpoint test metrics, excluding the derived best.json."""
    test_dir = run_dir / "TEST_RESULTS"
    if not test_dir.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(test_dir.glob("*.json")):
        if path.name in {"best.json", "launcher_metadata.json"}:
            continue
        data = _read_json(path)
        missing = data.get("missing", {})
        if missing.get("log_loss") is None or not missing.get("n"):
            continue
        rows.append({
            "checkpoint": data.get("checkpoint", path.name),
            "test_log_loss": float(missing["log_loss"]),
            "test_rmse": float(missing.get("rmse") or math.nan),
        })
    return rows


def analyse(run_dir: Path) -> Optional[Dict[str, Any]]:
    rows = _test_results(run_dir)
    if not rows:
        return None
    val_ce = _val_ce_by_epoch(run_dir)

    for row in rows:
        epoch = _epoch_of(row["checkpoint"])
        row["epoch"] = epoch
        row["val_ce"] = val_ce.get(epoch) if epoch is not None else None

    test_pick = min(rows, key=lambda r: (r["test_log_loss"], r["test_rmse"]))
    scored = [r for r in rows if r["val_ce"] is not None]
    val_pick = min(scored, key=lambda r: r["val_ce"]) if scored else None
    last_pick = next((r for r in rows if r["checkpoint"].startswith("last")), None)

    result: Dict[str, Any] = {
        "run": run_dir.name,
        "n_checkpoints": len(rows),
        "test_selected_log_loss": test_pick["test_log_loss"],
        "test_selected_ckpt": test_pick["checkpoint"],
        "val_selected_log_loss": val_pick["test_log_loss"] if val_pick else None,
        "val_selected_ckpt": val_pick["checkpoint"] if val_pick else None,
        "last_log_loss": last_pick["test_log_loss"] if last_pick else None,
    }
    if val_pick:
        # Positive => the reported number is better than an honest selection
        # would have been, purely because the test set picked the checkpoint.
        result["selection_gain"] = val_pick["test_log_loss"] - test_pick["test_log_loss"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path,
                        help="Run directories (each containing TEST_RESULTS/).")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Also write the table here.")
    args = parser.parse_args()

    results = []
    for run in args.runs:
        row = analyse(run)
        if row is None:
            print(f"skip (no test results): {run}", file=sys.stderr)
            continue
        results.append(row)

    if not results:
        raise SystemExit("no runs with test results")

    header = f"{'run':<34} {'n':>3} {'test-sel':>9} {'val-sel':>9} {'last':>9} {'gain':>7}"
    print(header)
    print("-" * len(header))
    gains = []
    for r in results:
        gain = r.get("selection_gain")
        if gain is not None:
            gains.append(gain)
        print(f"{r['run']:<34} {r['n_checkpoints']:>3} "
              f"{r['test_selected_log_loss']:>9.4f} "
              f"{(r['val_selected_log_loss'] if r['val_selected_log_loss'] is not None else float('nan')):>9.4f} "
              f"{(r['last_log_loss'] if r['last_log_loss'] is not None else float('nan')):>9.4f} "
              f"{(gain if gain is not None else float('nan')):>7.4f}")
    if gains:
        print("-" * len(header))
        print(f"mean optimism from test-set checkpoint selection: {sum(gains)/len(gains):+.4f} nats")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
