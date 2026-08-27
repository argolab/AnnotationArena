"""Re-score existing checkpoints with the candidate code and diff the metrics.

This is the one regression check with no randomness in it. `test.py` calls
`model.eval()`, so dropout is off and the entity-dropout RNG change -- the only
intentional numerical difference on this branch -- cannot enter. Any deviation
here is a real bug in the forward pass, the loss, or the checkpoint format.

It re-runs `test.py` over run directories produced by the *old* code and
compares against the `TEST_RESULTS/*.json` those runs already contain, which
doubles as a check that checkpoints stay loadable.

  python -m RERUN.bench.test_checkpoints --runs-root <RERUN/runs/replicate-01> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

FIELDS = ["log_loss", "accuracy", "rmse", "n", "spearman_r", "kendall_tau", "pearson_r"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--rtol", type=float, default=1e-5,
                    help="relative tolerance; float32 reassociation from the "
                         "einsum/index_add rewrites is ~1e-7 relative")
    ap.add_argument("--checkpoint", default="last")
    args = ap.parse_args()

    root = Path(args.runs_root)
    runs = sorted(p.parent.parent for p in root.rglob("TEST_RESULTS/last.json"))[: args.limit]
    if not runs:
        sys.exit(f"no runs with TEST_RESULTS/last.json under {root}")

    header = f"{'run':<44}{'field':<12}{'stored':>14}{'rescored':>14}{'rel.diff':>12}  verdict"
    print(header)
    print("-" * len(header))
    failures = []

    for run in runs:
        stored_path = run / "TEST_RESULTS" / f"{args.checkpoint}.json"
        with open(stored_path) as fh:
            stored = json.load(fh)

        # Re-score into a scratch copy so the original results are never touched.
        scratch = run.parent / f"_rescore_{run.name}"
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        for item in ("train_config.json", "checkpoints"):
            src = run / item
            if src.is_dir():
                shutil.copytree(src, scratch / item)
            elif src.is_file():
                shutil.copy2(src, scratch / item)

        proc = subprocess.run(
            [sys.executable, "-m", "imputer.entity_mf.test",
             "--run-dir", str(scratch), "--checkpoint", args.checkpoint],
            capture_output=True, text=True)
        new_path = scratch / "TEST_RESULTS" / f"{args.checkpoint}.json"
        if proc.returncode != 0 or not new_path.is_file():
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-2:]
            print(f"{run.name[:43]:<44}{'--':<12}{'--':>14}{'CRASH':>14}{'--':>12}  FAIL  "
                  + " | ".join(tail)[:120])
            failures.append(run.name)
            shutil.rmtree(scratch, ignore_errors=True)
            continue
        with open(new_path) as fh:
            new = json.load(fh)
        shutil.rmtree(scratch, ignore_errors=True)

        first = True
        for field in FIELDS:
            a, b = stored["missing"].get(field), new["missing"].get(field)
            if a is None or b is None:
                continue
            a, b = float(a), float(b)
            rel = abs(a - b) / max(abs(a), 1e-12)
            ok = rel <= args.rtol
            label = run.name[:43] if first else ""
            print(f"{label:<44}{field:<12}{a:>14.6f}{b:>14.6f}{rel:>12.2e}  "
                  f"{'ok' if ok else 'MISMATCH'}")
            first = False
            if not ok:
                failures.append(f"{run.name}:{field}")

    print()
    if failures:
        print("FAILED: " + ", ".join(sorted(set(failures))))
        sys.exit(1)
    print(f"all {len(runs)} runs re-score to within {args.rtol:g} relative")


if __name__ == "__main__":
    main()
