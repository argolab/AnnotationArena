#!/usr/bin/env python
"""Compare the Stan LLM-Rubric rerun against the published curve.

The published fits were never migrated, so the only surviving record of them is
the figure in the paper (D29). The reference here is digitised from that figure
and is good to about +/-0.005 -- fine for detecting a systematic shift, not for
quoting.

What matters is the SHAPE of the difference, not its size at any one point:
  - scattered small differences of either sign -> sampling noise, adopt the rerun
  - a consistent one-signed offset across sizes -> something changed, stop and
    find out what before either set of numbers goes in the paper

    python llmrubric_stan_compare.py --run-id final-s101
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rerun_values(run_dir: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    base = run_dir / "RESULTS" / "STAN" / "CPMSigmoid"
    if not base.is_dir():
        return out
    for d in sorted(base.glob("LLMRubric_225_25_9_*_CPMSigmoid_eval")):
        size = int(d.name.split("_")[4])
        f = d / "predictive_metrics.json"
        if not f.is_file():
            continue
        blob = json.loads(f.read_text())
        # evaluate_predictions.py reports the held-out cells as a mean log
        # LIKELIHOOD (negative); the paper plots cross-entropy, so flip the sign.
        # Do not look for a "log_loss" key -- there is none, and silently
        # finding nothing is how this script reported "no results yet" against a
        # complete set of fits.
        v = blob.get("rating_missing_log_likelihood")
        if isinstance(v, (int, float)):
            out[size] = -float(v)
        else:
            raise SystemExit(
                f"{f}: no rating_missing_log_likelihood; keys={sorted(blob)[:8]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rerun-dir", default=os.environ.get(
        "RERUN_DIR", "/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN"))
    a = ap.parse_args()

    pub = json.loads((HERE / "llmrubric_stan_published.json").read_text())
    ref = {int(k): v for k, v in pub["stan_cpm"].items()}
    new = rerun_values(Path(a.rerun_dir) / "runs" / a.run_id)
    if not new:
        print("no rerun Stan results yet"); return 0

    shared = sorted(set(ref) & set(new))
    print(f"{'K':>6}{'published':>12}{'rerun':>10}{'diff':>10}")
    print("-" * 38)
    diffs = []
    for K in shared:
        d = new[K] - ref[K]
        diffs.append(d)
        print(f"{K:>6}{ref[K]:>12.3f}{new[K]:>10.4f}{d:>+10.4f}")
    missing = sorted(set(ref) - set(new))
    if missing:
        print(f"\nnot yet fit: {missing}")
    if not diffs:
        return 0

    mean = st.mean(diffs)
    med = st.median(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    print(f"\nmean {mean:+.4f}  median {med:+.4f}  over {len(diffs)} sizes; "
          f"rerun higher in {n_pos} of {len(diffs)}")

    # The mean is the wrong statistic here: one bad fit at one size moves it by
    # more than a real systematic shift would. A systematic shift shows up as
    # every size moving the SAME way, which is what the sign test asks, and as a
    # median that is large, which the mean cannot fake.
    outliers = [(K, d) for K, d in zip(shared, diffs) if abs(d) > 0.05]
    core = [d for d in diffs if abs(d) <= 0.05]
    if outliers:
        print("\nsizes differing by more than 0.05 (excluded from the shift test, "
              "reported on their own):")
        for K, d in outliers:
            print(f"  K={K:<4} {d:+.4f}")
    if core:
        print(f"\nremaining {len(core)} sizes: mean {st.mean(core):+.4f}  "
              f"median {st.median(core):+.4f}  "
              f"sd {st.stdev(core):.4f} if len>1  "
              f"positive {sum(1 for d in core if d > 0)}/{len(core)}")

    systematic = ((n_pos == len(diffs) or n_pos == 0) and len(diffs) >= 6) \
        or abs(med) > 0.02
    if systematic:
        print("\n*** SYSTEMATIC DIFFERENCE -- STOP. ***")
        print("The rerun is consistently " + ("worse" if mean > 0 else "better") +
              " than the published curve. Do not overwrite the paper's numbers "
              "until the cause is known: candidate explanations are the data "
              "bundles (rebuilt today from the raw TSVs), the evaluation path "
              "(D20/D30: the published figure re-scored at max_item=10 while "
              "test.py used 500), and the Stan version (cmdstan 2.39 here).")
        return 1
    print("\nconsistent with sampling noise: differences are small and both-signed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
