#!/usr/bin/env python
"""
Summarise the residual-stream ablation.

Walks RESULTS/MARFORMER/DOMAIN3-SIGMOID/<AXIS>/<variant>/seed<N>/<run>_MARFORMER
and reports, per (variant, size), the mean and standard deviation across seeds
of the test metrics.

Primary metric is the LAST checkpoint (`last.json`) -- Domain 3 has no
validation split, so selecting per-arm on test would mostly measure selection
noise. The test-argmin is reported alongside for reference, clearly labelled.

    python ablation_summary.py --run-dir RERUN/runs/resid-01
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def _read(p: Path):
    with open(p) as fh:
        return json.load(fh)


def _mean_sd(xs: List[float]):
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def collect(results_root: Path) -> Dict:
    """(variant, size) -> {'last': [...], 'argmin': [...], 'seeds': [...]}"""
    out: Dict = defaultdict(lambda: {"last": [], "argmin": [], "seeds": []})
    for run_dir in sorted(results_root.rglob("*_MARFORMER")):
        parts = run_dir.parts
        # .../<AXIS>/<variant>/seed<N>/<run>_MARFORMER
        try:
            seed_part = parts[-2]
            variant = parts[-3]
        except IndexError:
            continue
        m_seed = re.fullmatch(r"seed(\d+)", seed_part)
        if not m_seed:
            continue          # not an ablation run (plain replication tree)
        m_size = re.search(r"_(\d+)_MARFORMER", run_dir.name)
        if not m_size:
            continue
        size = int(m_size.group(1))

        test_dir = run_dir / "TEST_RESULTS"
        if not test_dir.is_dir():
            continue

        last_p = test_dir / "last.json"
        last_ll = None
        if last_p.exists():
            last_ll = _read(last_p).get("missing", {}).get("log_loss")

        argmin_ll: Optional[float] = None
        for p in test_dir.glob("*.json"):
            if p.name in {"best.json", "last.json", "launcher_metadata.json"}:
                continue
            ll = _read(p).get("missing", {}).get("log_loss")
            if ll is not None and (argmin_ll is None or ll < argmin_ll):
                argmin_ll = ll

        key = (variant, size)
        if last_ll is not None:
            out[key]["last"].append(float(last_ll))
        if argmin_ll is not None:
            out[key]["argmin"].append(float(argmin_ll))
        out[key]["seeds"].append(int(m_seed.group(1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--order", default="all,not_last,none",
                    help="variant display order")
    args = ap.parse_args()

    data = collect(args.run_dir / "RESULTS")
    if not data:
        raise SystemExit(f"no ablation runs found under {args.run_dir}/RESULTS")

    order = args.order.split(",")
    sizes = sorted({s for _, s in data})
    variants = [v for v in order if any(v == vv for vv, _ in data)]
    variants += sorted({vv for vv, _ in data} - set(variants))

    print("test missing log-loss, LAST checkpoint (no selection)")
    print("mean +/- sd over seeds\n")
    head = f"{'size':>6}" + "".join(f"{v:>22}" for v in variants)
    print(head); print("-" * len(head))
    for size in sizes:
        row = f"{size:>6}"
        for v in variants:
            vals = data.get((v, size), {}).get("last", [])
            m, sd = _mean_sd(vals)
            row += f"{'--':>22}" if m is None else f"{m:>15.4f} +/-{sd:5.4f}"
        print(row)

    # Deltas against the shipped architecture, which is what the question is.
    base = variants[0]
    print(f"\ndelta vs '{base}' (negative = the variant is better)\n")
    head2 = f"{'size':>6}" + "".join(f"{v:>22}" for v in variants[1:])
    print(head2); print("-" * len(head2))
    for size in sizes:
        row = f"{size:>6}"
        b, _ = _mean_sd(data.get((base, size), {}).get("last", []))
        for v in variants[1:]:
            m, sd = _mean_sd(data.get((v, size), {}).get("last", []))
            if m is None or b is None:
                row += f"{'--':>22}"
            else:
                row += f"{m - b:>+15.4f} (sd{sd:5.4f})"
        print(row)

    print("\nfor reference only -- test-argmin over checkpoints (selection on test):\n")
    head3 = f"{'size':>6}" + "".join(f"{v:>22}" for v in variants)
    print(head3); print("-" * len(head3))
    for size in sizes:
        row = f"{size:>6}"
        for v in variants:
            m, sd = _mean_sd(data.get((v, size), {}).get("argmin", []))
            row += f"{'--':>22}" if m is None else f"{m:>15.4f} +/-{sd:5.4f}"
        print(row)

    n_seeds = {len(d["seeds"]) for d in data.values()}
    print(f"\nseeds per cell: {sorted(n_seeds)}")


if __name__ == "__main__":
    main()
