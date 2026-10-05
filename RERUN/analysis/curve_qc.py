#!/usr/bin/env python
"""Flag cells whose corner loss is an outlier against its neighbours on the curve.

Why this exists: the typed value codec collapses in roughly 1 cell in 30 at
small K (B11 -- seed 42 at K=300 came out +0.202 nats). The final run uses ONE
model seed, so there is no replicate to reveal a collapsed cell; without a check
it would simply appear as a bump in the published curve.

The test is deliberately model-free. These curves are monotone decreasing in K
to within noise, so a cell that sits far above the local trend of its own
neighbours is suspect regardless of what the absolute value is. For each cell we
fit the median of its immediate neighbours and report the residual; anything
above --threshold is flagged for a rerun at a different seed.

    python curve_qc.py --run-id final-s101 [--threshold 0.06]

Exit status is 1 if anything is flagged, so it can gate the figure build.
"""
from __future__ import annotations
import argparse, json, os, re, statistics as st, sys
from pathlib import Path

RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")


def collect(results_root: Path) -> dict:
    """(family, axis, arm) -> {size: corner log-loss} from each run's TEST_RESULTS."""
    out: dict = {}
    for fam_dir, fam in ((results_root / "MARFORMER", "Marformer"),
                         (results_root / "MARFORMER-NT", "MarformerNT")):
        if not fam_dir.is_dir():
            continue
        for run in fam_dir.rglob("*_MARFORMER"):
            m = RUN_RE.search(run.name)
            if not m:
                continue
            # The refit's last checkpoint is what gets reported (D14), never an
            # argmin on test.
            res = run / "TEST_RESULTS" / "last.json"
            if not res.is_file():
                continue
            try:
                blob = json.loads(res.read_text())
            except json.JSONDecodeError:
                print(f"  [unreadable] {res}", file=sys.stderr)
                continue
            val = blob.get("missing", {}).get("log_loss", blob.get("log_loss"))
            if val is None:
                continue
            out.setdefault((fam, m.group(1).lower(), m.group(2)), {})[int(m.group(3))] = float(val)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rerun-dir", default=os.environ.get(
        "RERUN_DIR", "/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN"))
    ap.add_argument("--threshold", type=float, default=0.06,
                    help="nats above the neighbour median to flag. 0.06 sits "
                         "between the 0.035 paired between-arm sd and the "
                         "0.202 collapse B11 measured.")
    a = ap.parse_args()
    root = Path(a.rerun_dir) / "runs" / a.run_id / "RESULTS"
    curves = collect(root)
    if not curves:
        print(f"no scored runs under {root} -- nothing to check yet")
        return 0

    flagged = []
    for key, pts in sorted(curves.items()):
        sizes = sorted(pts)
        print(f"\n{key[0]} / {key[1]} / {key[2]}   ({len(sizes)} cells)")
        for i, s in enumerate(sizes):
            nb = [pts[sizes[j]] for j in (i - 1, i + 1) if 0 <= j < len(sizes)]
            if not nb:
                print(f"  K={s:<6} {pts[s]:.4f}   (no neighbour)")
                continue
            resid = pts[s] - st.median(nb)
            mark = ""
            if resid > a.threshold:
                mark = "   <== FLAGGED"
                flagged.append((key, s, pts[s], resid))
            print(f"  K={s:<6} {pts[s]:.4f}   resid {resid:+.4f}{mark}")

    print()
    if flagged:
        print(f"{len(flagged)} cell(s) flagged above {a.threshold} nats:")
        for key, s, v, r in flagged:
            print(f"  {key[0]}/{key[1]}/{key[2]} K={s}: {v:.4f} ({r:+.4f} vs neighbours)")
        print("\nRerun each at a different model seed before plotting. A genuine "
              "codec collapse reproduces as a large gap against the new seed; "
              "a real feature of the curve does not.")
        return 1
    print(f"no cell exceeds {a.threshold} nats above its neighbours")
    return 0


if __name__ == "__main__":
    sys.exit(main())
