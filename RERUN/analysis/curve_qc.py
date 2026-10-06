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


UNIFORM = 1.3862943611198906   # ln(4): a model that never left the plateau


def collect_corner(corner_json: Path) -> dict:
    """(family, axis, arm) -> {size: TEST-CORNER log-loss} from corner.json.

    This is the metric the paper reports and the one result_tables.py renders.
    The pooled "missing" field in TEST_RESULTS is a different number -- it mixes
    the corner with the training partition's own held-out cells (D22/D23) -- and
    on the codec run the two disagree enough to change the verdict: item/T
    K=100 is 1.73 pooled and 0.97 on the corner. QC that gates the figures has
    to read what the figures plot.

    split_missing_metrics.py nests the value under "test" and carries the arm in
    the family name ("MarformerNT"), not in a field of its own.
    """
    out: dict = {}
    blob = json.loads(corner_json.read_text())
    for row in blob.get("rows", []):
        fam, axis, size = row.get("family"), row.get("axis"), row.get("size")
        val = (row.get("test") or {}).get("log_loss")
        if None in (fam, axis, size, val):
            continue
        arm = "NT" if fam.endswith("NT") else "T"
        out.setdefault((fam, str(axis).lower(), arm), {})[int(size)] = float(val)
    return out


def collect(results_root: Path) -> dict:
    """(family, axis, arm) -> {size: POOLED log-loss} from each run's TEST_RESULTS."""
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
    ap.add_argument("--source", choices=("corner", "pooled"), default="corner",
                    help="corner (default) reads corner_metrics/corner.json, "
                         "the metric the paper reports; pooled reads "
                         "TEST_RESULTS/last.json, which is not comparable.")
    a = ap.parse_args()
    run_dir = Path(a.rerun_dir) / "runs" / a.run_id
    if a.source == "corner":
        cj = run_dir / "corner_metrics" / "corner.json"
        if not cj.is_file():
            print(f"no {cj} -- run jobs/corner_score.sbatch first, or pass "
                  f"--source pooled")
            return 0
        curves = collect_corner(cj)
    else:
        curves = collect(run_dir / "RESULTS")
    if not curves:
        print(f"no scored runs for {a.run_id} -- nothing to check yet")
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
            degenerate = abs(pts[s] - UNIFORM) < 0.02
            if degenerate:
                mark = "   <== DEGENERATE (at ln4)"
                flagged.append((key, s, pts[s], resid))
            elif resid > a.threshold:
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
