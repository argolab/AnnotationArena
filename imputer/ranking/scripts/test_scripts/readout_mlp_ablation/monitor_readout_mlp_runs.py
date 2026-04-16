#!/usr/bin/env python3
"""Monitor matrix-completion readout-MLP sweep runs.

Scans run directories (default: OUTPUT/mc_readout_mlp) and summarizes:
- status (running/done/missing)
- completed/planned steps
- tail and last dev/train metrics from curves_live.json or curves.json

`test_mse` here is the toy script's masked-test metric (same as printed test_masked_mse).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_ROOT = "OUTPUT/mc_readout_mlp"
DEFAULT_RUN_ORDER = [
    "baseline",
    "L1_D64",
    "L1_D128",
    "L1_D256",
    "L2_D64",
    "L2_D128",
    "L2_D256",
    "L3_D64",
    "L3_D128",
    "L3_D256",
]


def _tail_mean(xs: Sequence[float], tail: int) -> Optional[float]:
    if not xs:
        return None
    chunk = xs[-tail:] if len(xs) >= tail else xs
    return float(statistics.mean(chunk))


def _load_run(run_dir: Path) -> Tuple[str, Dict[str, Any], Path]:
    """Return (status, payload, source_path)."""
    live = run_dir / "curves_live.json"
    final = run_dir / "curves.json"
    if live.exists():
        data = json.loads(live.read_text())
        planned = int(data.get("num_steps") or 0)
        done = int(data.get("completed_steps") or len(data.get("test_mse") or []))
        status = "running" if planned > 0 and done < planned else "done"
        return status, data, live
    if final.exists():
        return "done", json.loads(final.read_text()), final
    return "missing", {}, run_dir


def _collect_runs(root: Path, explicit_runs: Sequence[str] | None) -> List[Path]:
    if explicit_runs:
        return [root / r for r in explicit_runs]
    if not root.is_dir():
        return []
    children = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if children:
        return children
    return [root]


def _run_sort_key(run_name: str) -> Tuple[int, str]:
    if run_name in DEFAULT_RUN_ORDER:
        return (DEFAULT_RUN_ORDER.index(run_name), run_name)
    return (10_000, run_name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=str,
        default=DEFAULT_ROOT,
        help=f"Sweep output root (default: {DEFAULT_ROOT}).",
    )
    p.add_argument(
        "--run",
        action="append",
        dest="runs",
        help="Run directory name under --root (repeatable). Defaults to known sweep run names.",
    )
    p.add_argument(
        "--tail",
        type=int,
        default=100,
        help="Average dev/train over the last this many points (default 100).",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="Print CSV instead of pretty table.",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    runs = args.runs if args.runs else DEFAULT_RUN_ORDER
    run_dirs = _collect_runs(root, runs)

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        status, data, src = _load_run(run_dir)
        if status == "missing":
            rows.append(
                {
                    "run": run_name,
                    "status": "missing",
                    "source": "",
                    "completed_steps": None,
                    "planned_steps": None,
                    "tail_dev_mse": None,
                    "tail_train_mse": None,
                    "last_dev_mse": None,
                    "last_train_mse": None,
                }
            )
            continue
        te = data.get("test_mse") or []
        tr = data.get("train_mse") or []
        rows.append(
            {
                "run": run_name,
                "status": status,
                "source": src.name,
                "completed_steps": int(data.get("completed_steps", len(te))),
                "planned_steps": data.get("num_steps"),
                "tail_dev_mse": _tail_mean(te, args.tail),
                "tail_train_mse": _tail_mean(tr, args.tail),
                "last_dev_mse": float(te[-1]) if te else None,
                "last_train_mse": float(tr[-1]) if tr else None,
            }
        )

    rows.sort(key=lambda r: _run_sort_key(r["run"]))

    if args.csv:
        cols = [
            "run",
            "status",
            "source",
            "completed_steps",
            "planned_steps",
            "tail_dev_mse",
            "tail_train_mse",
            "last_dev_mse",
            "last_train_mse",
        ]
        print(",".join(cols))
        for r in rows:
            print(",".join("" if r[c] is None else str(r[c]) for c in cols))
        return

    print(
        f"Root: {root}\n"
        f"tail_dev_mse = mean(last {args.tail} points of test_mse)\n"
        f"tail_train_mse = mean(last {args.tail} points of train_mse)\n"
    )
    rw = 18
    hdr = (
        f"{'run':<{rw}} {'stat':<7} {'done':>6} {'plan':>6} "
        f"{'tail_dev':>12} {'tail_tr':>12} {'last_dev':>12} {'last_tr':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        done = "" if r["completed_steps"] is None else str(r["completed_steps"])
        plan = "" if r["planned_steps"] is None else str(r["planned_steps"])
        td = "" if r["tail_dev_mse"] is None else f"{r['tail_dev_mse']:.6f}"
        tt = "" if r["tail_train_mse"] is None else f"{r['tail_train_mse']:.6f}"
        ld = "" if r["last_dev_mse"] is None else f"{r['last_dev_mse']:.6f}"
        lt = "" if r["last_train_mse"] is None else f"{r['last_train_mse']:.6f}"
        print(
            f"{r['run']:<{rw}} {r['status']:<7} {done:>6} {plan:>6} "
            f"{td:>12} {tt:>12} {ld:>12} {lt:>12}"
        )


if __name__ == "__main__":
    main()

