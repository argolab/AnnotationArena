#!/usr/bin/env python
"""Build the T1 results page from out/t1/*.json (written by t1_score.py).

    python t1_page.py [-o page.html]

The page is self-contained: the payload is embedded, every chart is drawn in the
browser from it, and nothing is fetched.
"""
from __future__ import annotations
import argparse, glob, json, math, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = [("codec_dual", "dual stream", "slice"),
        ("lin_T80", "T = 80", "full width"), ("lin_T32", "T = 32", ""),
        ("lin_T16", "T = 16", ""), ("lin_T8", "T = 8", ""), ("lin_T5", "T = 5", ""),
        ("lin_T32_res", "T = 32", "reserved"), ("lin_T16_res", "T = 16", "reserved"),
        ("lin_T8_res", "T = 8", "reserved"), ("lin_T5_res", "T = 5", "reserved")]
SIZES = [100, 300, 500, 1000]


def thin(curve, step):
    return [[e, round(v, 4)] for e, v in curve[::step]]


def headline(r):
    """The run's test loss: at the dev-selected epoch, or tail-averaged where
    there is no dev curve to select on (K=100)."""
    return r["test_at_e_star"] if r["e_star"] is not None else r["test_tail"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(HERE / "out/t1_page.html"))
    a = ap.parse_args()
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(HERE / "out/t1/*.json")))]
    runs = []
    for r in recs:
        runs.append({
            "arm": r["arm"], "K": r["K"], "draw": r["draw"], "seed": r["seed"],
            "train": thin(r["train"], 4), "dev": thin(r["dev"], 4),
            "test": [[t["epoch"], round(t["loss"], 4)] for t in r["test"]],
            "eStar": r["e_star"], "atStar": r["test_at_e_star"], "tail": r["test_tail"],
            "best": r["test_best"], "bestEp": r["test_best_epoch"], "n": r["test_n"],
            "head": headline(r),
        })
    # paired differences against the dual stream, on (draw, seed, K)
    ref = {(r["draw"], r["seed"], r["K"]): r for r in runs if r["arm"] == "codec_dual"}
    summary = {}
    for arm, *_ in ARMS:
        for K in SIZES:
            for metric in ("head", "tail"):
                d = [r[metric] - ref[(r["draw"], r["seed"], K)][metric]
                     for r in runs if r["arm"] == arm and r["K"] == K
                     and (r["draw"], r["seed"], K) in ref]
                lv = [r[metric] for r in runs if r["arm"] == arm and r["K"] == K]
                summary[f"{arm}|{K}|{metric}"] = {
                    "n": len(d),
                    "mean": round(st.mean(d), 4) if d else None,
                    "se": round(st.stdev(d) / math.sqrt(len(d)), 4) if len(d) > 1 else None,
                    "worst": round(max(d), 4) if d else None,
                    "level": round(st.mean(lv), 4) if lv else None,
                }
    payload = {"arms": ARMS, "sizes": SIZES, "runs": runs, "summary": summary,
               "nRuns": len(runs), "expected": len(ARMS) * len(SIZES) * 6}
    html = (HERE / "t1_page_template.html").read_text().replace(
        "/*PAYLOAD*/null", json.dumps(payload, separators=(",", ":")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(html)
    print(f"wrote {a.out}: {len(runs)} runs, {len(html)/1e3:.0f} KB")


if __name__ == "__main__":
    main()
