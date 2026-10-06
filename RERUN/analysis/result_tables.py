#!/usr/bin/env python
"""Render the final run's test-corner log-loss as box-drawing tables.

One table per figure: synthetic item generalization, synthetic annotator
generalization, and LLM-Rubric. Every method that appears in the corresponding
paper figure gets a row.

All numbers are the TEST CORNER, never the pooled metric. Marformer comes from
corner_metrics/*.json (split_missing_metrics.py --corner-whole), NOT from
TEST_RESULTS/*.json, whose "missing" field pools the corner with the training
partition's own missing cells (D22) and so is not comparable with Stan (D23).
Stan comes from each fit's predictive_metrics.json, which is already test-only.

    python result_tables.py --run-id final-s101
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path

ITEM_SIZES  = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
ANNOT_SIZES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
LLMR_SIZES  = [10, 20, 30, 40, 50, 75, 100, 125, 150, 175]

STAN_SERIES = [("Stan Oracle", "ORACLE",  "TENSOR-SIG"),
               ("Stan Bin",    "MISSPEC", "BIN"),
               ("Stan Proj",   "MISSPEC", "PROJ"),
               ("Stan Both",   "MISSPEC", "BOTH")]


def render(title: str, sizes, rows, col_label="K") -> str:
    """rows = [(label, {size: value or None})]"""
    labels = [r[0] for r in rows]
    w0 = max([len(col_label)] + [len(l) for l in labels]) + 2
    cells = [f"{s}" for s in sizes]
    w = max(7, max(len(c) for c in cells) + 2)

    def line(l, m, r):
        return l + "─" * w0 + (m + "─" * w) * len(sizes) + r

    def row(label, vals):
        out = "│" + label.center(w0)
        for v in vals:
            out += "│" + (v.center(w) if v else "—".center(w))
        return out + "│"

    s = [title, line("┌", "┬", "┐"),
         row(col_label, [c.center(w - 2) for c in cells]),
         line("├", "┼", "┤")]
    for label, d in rows:
        s.append(row(label, [f"{d[k]:.3f}" if d.get(k) is not None else None
                             for k in sizes]))
    s.append(line("└", "┴", "┘"))
    return "\n".join(s)


def marformer_corner(run: Path, family: str, axis: str) -> dict:
    """size -> corner log-loss, from the --corner-whole scorer."""
    p = run / "corner_metrics" / "corner.json"
    if not p.is_file():
        return {}
    blob = json.loads(p.read_text())
    out = {}
    for r in blob.get("rows", []):
        if r.get("axis") != axis or r.get("family") != family:
            continue
        t = r.get("test") or {}
        if t.get("log_loss") is not None:
            out[int(r["size"])] = float(t["log_loss"])
    return out


def stan_synth(run: Path, variant: str, tag: str, axis: str) -> dict:
    base = run / "RESULTS" / "STAN" / "DOMAIN3-SIGMOID" / variant / axis.upper()
    if not base.is_dir():
        return {}
    ax = "Item" if axis == "item" else "Annot"
    out = {}
    for d in base.glob(f"DOMAIN3-SIGMOID_{ax}_T_*_{tag}_eval"):
        m = re.search(rf"_{ax}_T_(\d+)_{tag}_eval$", d.name)
        f = d / "predictive_metrics.json"
        if not m or not f.is_file():
            continue
        ll = json.loads(f.read_text()).get("rating_missing_log_likelihood")
        if ll is not None:
            out[int(m.group(1))] = -float(ll)   # stored as a log-likelihood
    return out


def stan_llmr(run: Path) -> dict:
    base = run / "RESULTS" / "STAN" / "CPMSigmoid"
    out = {}
    if not base.is_dir():
        return out
    for d in base.glob("LLMRubric_225_25_9_*_CPMSigmoid_eval"):
        m = re.search(r"_9_(\d+)_CPMSigmoid_eval$", d.name)
        f = d / "predictive_metrics.json"
        if not m or not f.is_file():
            continue
        ll = json.loads(f.read_text()).get("rating_missing_log_likelihood")
        if ll is not None:
            out[int(m.group(1))] = -float(ll)
    return out


def marformer_llmr(run: Path) -> dict:
    """LLM-Rubric is scored at its trained chunk by test.py (D20 fix)."""
    out = {}
    base = run / "RESULTS" / "MARFORMER" / "LLM_RUBRIC"
    if not base.is_dir():
        base = run / "RESULTS" / "LLM_RUBRIC"
    for f in base.rglob("TEST_RESULTS/last.json"):
        m = re.search(r"_9_(\d+)(?:_|/)", str(f))
        if not m:
            continue
        v = json.loads(f.read_text()).get("missing", {}).get("log_loss")
        if v is not None:
            out[int(m.group(1))] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rerun-dir", default=os.environ.get(
        "RERUN_DIR", "/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN"))
    a = ap.parse_args()
    run = Path(a.rerun_dir) / "runs" / a.run_id

    for axis, sizes, label in (("item", ITEM_SIZES, "ITEM"),
                               ("annot", ANNOT_SIZES, "ANNOT")):
        rows = [("Marformer (T)",  marformer_corner(run, "Marformer", axis)),
                ("Marformer (NT)", marformer_corner(run, "MarformerNT", axis))]
        for name, variant, tag in STAN_SERIES:
            rows.append((name, stan_synth(run, variant, tag, axis)))
        name = ("Synthetic — item generalization" if axis == "item"
                else "Synthetic — annotator generalization")
        print("\n" + render(f"{name}  (test-corner log-loss, draw s101, seed 42)",
                            sizes, rows, "K" if axis == "item" else "J") + "\n")

    rows = [("Marformer", marformer_llmr(run)), ("Stan CPM", stan_llmr(run))]
    print(render("LLM-Rubric — item generalization  (test log-loss, real data)",
                 LLMR_SIZES, rows, "K") + "\n")


if __name__ == "__main__":
    main()
