#!/usr/bin/env python
"""Render the three fixed-200-epoch result tables, log-loss over smECE.

Reads tables_fixed_budget.py's JSON. Each method gets two lines: the test-corner
log-loss and, beneath it in parentheses, the smECE on the same payload.

    python render_fixed200.py              # text
    python render_fixed200.py --latex      # booktabs bodies for the paper
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

RERUN = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN")
ITEM_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
ANNOT_SIZES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
LLMR_SIZES = [10, 20, 30, 40, 50, 75, 100, 125, 150, 175]
ORDER = ["Marformer (T)", "Marformer (NT)", "Stan Oracle", "Stan Bin",
         "Stan Proj", "Stan Both"]


def render(title: str, sizes, rows, col_label="K") -> str:
    """rows = [(label, {size: {'ll':..,'smece':..}})]; two text lines per method."""
    labels = [r[0] for r in rows]
    w0 = max([len(col_label)] + [len(l) for l in labels]) + 2
    w = max(9, max(len(str(s)) for s in sizes) + 2)

    def line(l, m, r):
        return l + "─" * w0 + (m + "─" * w) * len(sizes) + r

    def row(label, vals):
        out = "│" + label.center(w0)
        for v in vals:
            out += "│" + (v if v else "—").center(w)
        return out + "│"

    s = [title, line("┌", "┬", "┐"),
         row(col_label, [str(x) for x in sizes]),
         line("├", "┼", "┤")]
    for i, (label, d) in enumerate(rows):
        s.append(row(label, [f"{d[k]['ll']:.3f}" if d.get(k)
                             and d[k].get("ll") is not None else None for k in sizes]))
        s.append(row("", [f"({d[k]['smece']:.3f})" if d.get(k)
                          and d[k].get("smece") is not None else None for k in sizes]))
        if i != len(rows) - 1:
            s.append(line("├", "┼", "┤"))
    s.append(line("└", "┴", "┘"))
    return "\n".join(s)


def latex(title, sizes, rows, col_label="K") -> str:
    out = [f"% {title}", r"\begin{tabular}{l" + "r" * len(sizes) + "}", r"\toprule",
           f"{col_label} & " + " & ".join(str(s) for s in sizes) + r" \\", r"\midrule"]
    for label, d in rows:
        ll = " & ".join(f"{d[k]['ll']:.3f}" if d.get(k) and d[k].get("ll") is not None
                        else "--" for k in sizes)
        ce = " & ".join(f"\\footnotesize({d[k]['smece']:.3f})" if d.get(k)
                        and d[k].get("smece") is not None else "" for k in sizes)
        out += [f"{label} & {ll} " + r"\\", f" & {ce} " + r" \\[2pt]"]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(RERUN / "runs/final-dual/reports/tables_fixed200.json"))
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    blob = json.loads(Path(a.inp).read_text())
    b = blob["synthetic"]["budget"]
    emit = latex if a.latex else render
    note = (f"(test-corner log-loss, smECE in parentheses; every Marformer fit at a "
            f"FIXED {b}-epoch budget, draw s101, seed 42)")

    for axis, sizes, cl, name in (("item", ITEM_SIZES, "K", "item"),
                                  ("annot", ANNOT_SIZES, "J", "annotator")):
        ser = blob["synthetic"]["axes"][axis]
        rows = [(k, {int(kk): vv for kk, vv in ser[k].items()})
                for k in ORDER if k in ser]
        print("\n" + emit(f"Synthetic — {name} generalization  {note}",
                          sizes, rows, cl) + "\n")

    if "llmrubric" in blob:
        ser = blob["llmrubric"]["series"]
        rows = [(k, {int(kk): vv for kk, vv in ser[k].items()})
                for k in ("Marformer", "Stan CPM") if ser.get(k)]
        if rows:
            print(emit(f"LLM-Rubric — item generalization  (test log-loss, smECE "
                       f"in parentheses; Marformer at a FIXED {b}-epoch budget, real data)",
                       LLMR_SIZES, rows, "K") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
