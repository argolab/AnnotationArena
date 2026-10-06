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
        for key in ("test_log_loss", "log_loss", "mean_log_loss"):
            v = blob.get(key)
            if isinstance(v, (int, float)):
                out[size] = float(v); break
        else:
            m = blob.get("missing") or blob.get("test") or {}
            if isinstance(m, dict) and isinstance(m.get("log_loss"), (int, float)):
                out[size] = float(m["log_loss"])
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
    n_pos = sum(1 for d in diffs if d > 0)
    print(f"\nmean difference {mean:+.4f} over {len(diffs)} sizes; "
          f"rerun higher in {n_pos} of {len(diffs)}")
    # A sign test: all-one-sign across >=6 sizes is p < 0.05 under the null that
    # each size is an independent coin flip.
    systematic = (n_pos == len(diffs) or n_pos == 0) and len(diffs) >= 6
    if systematic or abs(mean) > 0.02:
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
