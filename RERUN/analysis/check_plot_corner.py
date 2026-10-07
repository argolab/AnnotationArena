#!/usr/bin/env python
"""Do the FIGURES carry the same numbers as the tables?

Two checks, and the difference between them is the lesson. The first compares
the plot's probability path against corner.json; that path is shared by the
transductive and non-transductive arms and was already right, so passing it
proved less than it looked. The figures were still wrong on the NT arm, because
the number that reaches a curve is not always the one that path computes -- a
pooled `missing.log_loss` from best.json can override it. The second check
therefore reads the summary JSON the figure build actually writes and compares
every cell against corner.json. That is the one that gates.

    python check_plot_corner.py                 # end-to-end, no GPU, exits 1 on mismatch
    python check_plot_corner.py --probs-path    # also re-run the GPU probability path
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
# The run to gate. Hardcoded to final-dual until 2026-10-06, which silently
# gated the WRONG run once final-v2 figures existed -- it passed, on the old
# run's numbers, while the new figures went unchecked. Pass --run-dir.
RUNS = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual")
TOL = 0.0006          # the tables print 3 decimals, so this is rounding only

SUMMARIES = {
    "item":  "PLOTS/DOMAIN3-SIGMOID/ITEM_GENERALIZATION/item_generalization_summary.json",
    "annot": "PLOTS/DOMAIN3-SIGMOID/ANNOT_GENERALIZATION/annot_generalization_summary.json",
}
# the summary's model label -> corner.json's family
MODEL_FAMILY = {"Marformer": "Marformer", "Marformer NT": "MarformerNT"}


def load_corner() -> dict:
    rows = json.loads((RUNS / "corner_metrics/corner.json").read_text())
    out = {}
    for r in rows["rows"]:
        v = (r.get("test") or {}).get("log_loss")
        if v is not None:
            out[(r["family"], str(r["axis"]).lower(), int(r["size"]))] = (v, r["test"]["n"])
    return out


def end_to_end(corner: dict) -> bool:
    ok = True
    for axis, rel in SUMMARIES.items():
        p = RUNS / rel
        if not p.is_file():
            print(f"FAIL  {axis}: {rel} not found -- the figure build did not write it")
            ok = False
            continue
        rows = json.loads(p.read_text())["rows"]
        fig = {(MODEL_FAMILY[r["model"]], axis, int(r["size"])): r["mean"]
               for r in rows
               if r["metric"] == "log_loss" and r["model"] in MODEL_FAMILY}
        want = {k: v for k, v in corner.items() if k[1] == axis}
        print(f"\n=== {axis} ===")
        print(f"{'cell':<26}{'figure':>10}{'corner.json':>13}{'diff':>10}")
        for k in sorted(want, key=lambda k: (k[0], k[2])):
            cv = want[k][0]
            if k not in fig:
                print(f"{k[0]+'/'+str(k[2]):<26}{'MISSING':>10}{cv:>13.4f}{'':>10}  "
                      f"<== the tables have this cell and the figure does not")
                ok = False
                continue
            d = fig[k] - cv
            bad = abs(d) > TOL
            if bad:
                ok = False
            print(f"{k[0]+'/'+str(k[2]):<26}{fig[k]:>10.4f}{cv:>13.4f}{d:>+10.4f}"
                  f"{'  <== MISMATCH' if bad else ''}")
        extra = sorted(set(fig) - set(want))
        for k in extra:
            print(f"{k[0]+'/'+str(k[2]):<26}{fig[k]:>10.4f}{'--':>13}{'':>10}  "
                  f"<== in the figure, not in corner.json")
            ok = False
    return ok


def probs_path(corner: dict) -> bool:
    import numpy as np
    sys.path.insert(0, str(RANK))
    spec = importlib.util.spec_from_file_location(
        "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pd3"] = m      # dataclass() resolves cls.__module__ through sys.modules
    spec.loader.exec_module(m)
    cases = [
        ("Marformer", "item", 2000, "MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_2000_MARFORMER"),
        ("Marformer", "item", 300, "MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_300_MARFORMER"),
        ("MarformerNT", "item", 4000, "MARFORMER-NT/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_NT_4000_MARFORMER"),
        ("MarformerNT", "item", 100, "MARFORMER-NT/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_NT_100_MARFORMER"),
        ("MarformerNT", "annot", 20, "MARFORMER-NT/DOMAIN3-SIGMOID/ANNOT/DOMAIN3-SIGMOID_Annot_NT_20_MARFORMER"),
        ("Marformer", "annot", 20, "MARFORMER/DOMAIN3-SIGMOID/ANNOT/DOMAIN3-SIGMOID_Annot_T_20_MARFORMER"),
    ]
    ok = True
    for pop in ("corner", "pooled"):
        m.METRIC_POPULATION = pop
        print(f"\n=== probability path, METRIC_POPULATION = {pop} ===")
        print(f"{'cell':<26}{'plot n':>8}{'plot ll':>10}{'corner n':>10}{'corner ll':>11}{'diff':>9}")
        for fam, axis, K, rel in cases:
            probs, labels = m._marformer_probs_all_missing(
                RUNS / "RESULTS" / rel, "last.ckpt", 500, "cuda")
            ll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())
            cv, cn = corner[(fam, axis, K)]
            d = ll - cv
            # Only the corner population is expected to agree; pooled is printed
            # to show the size and sign of the difference it would introduce.
            bad = pop == "corner" and abs(d) > TOL
            if bad:
                ok = False
            print(f"{fam+'/'+axis+'/'+str(K):<26}{len(labels):>8}{ll:>10.4f}"
                  f"{cn:>10}{cv:>11.4f}{d:>+9.4f}{'  <== MISMATCH' if bad else ''}")
    return ok


def _set_run(run_dir: str) -> None:
    global RUNS
    RUNS = Path(run_dir)
    if not (RUNS / "corner_metrics/corner.json").is_file():
        raise SystemExit(f"no corner_metrics/corner.json under {RUNS} -- run jobs/corner_score.sbatch first")
    print(f"gating run: {RUNS.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(RUNS),
                    help="run tree to gate (default: the historical final-dual)")
    ap.add_argument("--probs-path", action="store_true",
                    help="also re-run the plot's GPU probability path on six cells")
    a = ap.parse_args()
    _set_run(a.run_dir)
    corner = load_corner()
    ok = end_to_end(corner)
    if a.probs_path:
        ok = probs_path(corner) and ok
    print("\n" + ("FIGURES AGREE WITH THE TABLES"
                  if ok else "FIGURES AND TABLES DISAGREE -- see the marked rows"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
