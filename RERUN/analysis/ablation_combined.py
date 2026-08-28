#!/usr/bin/env python
"""
Combine the residual-stream ablation across run directories.

The variants live in different RERUN/runs/<id> trees (resid-01 holds the
baseline arms, resid-02 the identity-folded ones, resid-03 the inner-residual
ablation), so no single ablation_summary.py invocation sees them all. This one
walks several trees, keys cells by (variant, size, seed), and reports both the
absolute last-checkpoint test log-loss and the seed-paired difference against a
chosen baseline arm.

    python ablation_combined.py --run-dirs RERUN/runs/resid-0{1,2,3} \
        --variants all,none,none_id,all+inner_none --baseline all
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

SIZE_RE = re.compile(r"_Item_T_(\d+)_MARFORMER$")
SEED_RE = re.compile(r"^seed(\d+)$")


def collect(run_dirs, axis="ITEM"):
    """cells[(variant, size)][seed] = last-checkpoint test-missing log loss."""
    cells = defaultdict(dict)
    for run_dir in run_dirs:
        root = Path(run_dir) / "RESULTS/MARFORMER/DOMAIN3-SIGMOID" / axis
        if not root.is_dir():
            continue
        for variant_dir in sorted(root.iterdir()):
            if not variant_dir.is_dir():
                continue
            variant = variant_dir.name
            # resid-01 stores the baseline arm without a seed level.
            seed_dirs = [d for d in variant_dir.iterdir() if SEED_RE.match(d.name)]
            pairs = ([(int(SEED_RE.match(d.name).group(1)), d) for d in seed_dirs]
                     or [(42, variant_dir)])
            for seed, sdir in pairs:
                for run in sdir.glob("*_MARFORMER"):
                    m = SIZE_RE.search(run.name)
                    last = run / "TEST_RESULTS" / "last.json"
                    if not (m and last.is_file()):
                        continue
                    with open(last) as fh:
                        cells[(variant, int(m.group(1)))][seed] = \
                            json.load(fh)["missing"]["log_loss"]
    return cells


def mean_sd(xs):
    n = len(xs)
    if not n:
        return None, None
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--variants", default="all,none,none_id,all+inner_none")
    ap.add_argument("--baseline", default="all")
    ap.add_argument("--axis", default="ITEM")
    args = ap.parse_args()

    variants = args.variants.split(",")
    cells = collect(args.run_dirs, args.axis)
    sizes = sorted({sz for (_, sz) in cells})

    print("ABSOLUTE test missing log-loss, LAST checkpoint.  chance = 1.3863\n")
    print(f"{'K':>5}" + "".join(f"{v:>22}" for v in variants))
    for sz in sizes:
        row = f"{sz:>5}"
        for v in variants:
            xs = list(cells.get((v, sz), {}).values())
            m, sd = mean_sd(xs)
            row += "".ljust(22) if m is None else \
                f"{m:>14.4f} +/-{sd:.4f}{'*' if len(xs) < 3 else ' '}"
        print(row)
    print("* = fewer than 3 seeds\n")

    print(f"PAIRED vs {args.baseline!r} (same seed; negative = better than baseline)\n")
    others = [v for v in variants if v != args.baseline]
    print(f"{'K':>5}" + "".join(f"{v:>26}" for v in others))
    for sz in sizes:
        base = cells.get((args.baseline, sz), {})
        row = f"{sz:>5}"
        for v in others:
            arm = cells.get((v, sz), {})
            shared = sorted(set(base) & set(arm))
            if not shared:
                row += "".ljust(26)
                continue
            d = [arm[s] - base[s] for s in shared]
            m, sd = mean_sd(d)
            se = sd / math.sqrt(len(d)) if len(d) > 1 else 0.0
            row += f"{m:>+16.4f} (se{se:.4f}){'*' if len(d) < 3 else ' '}"
        print(row)


if __name__ == "__main__":
    main()
