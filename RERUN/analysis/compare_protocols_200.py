#!/usr/bin/env python
"""Is a fixed 200-epoch budget UNIFORMLY better than the F13 dev protocol? No.

Three comparisons, which are not interchangeable:

  A  matched data fraction -- fixed-200 vs F13's selected epoch, BOTH read off the
     same 85% search run. Only the budget differs. This is the clean causal read of
     the budget, and it is what budget_table.json supports.

  B  what the paper would actually ship -- 100% refit at a fixed 200 vs the
     published 100% refit at the selected epoch. NOT AVAILABLE: the fixed-budget
     refit has not been trained. This is the comparison that would settle it.

  C  published vs available -- the published 100% refit against fixed-200 on the
     85% search run. Confounded, and biased AGAINST the fixed budget, because the
     refit gets 15% more data (worth -0.022 at item/500, -0.009 at annot/30).

"Uniformly" is a strong claim and it fails in all of them: report the per-cell
distribution, not just the mean.

    python compare_protocols_200.py
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

RUNS = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual")


def summ(d: list[float], name: str, better_is_negative=True) -> None:
    se = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
    better = sum(1 for x in d if x < 0)
    worse = sum(1 for x in d if x > 0)
    print(f"  {name:<34} mean {st.mean(d):+.4f} +/-{se:.4f}   "
          f"median {st.median(d):+.4f}   better {better}/{len(d)}  worse {worse}/{len(d)}")


def main() -> int:
    bt = json.loads((RUNS / "reports/budget_table.json").read_text())
    idx = {(r["family"], r["axis"], r["K"], r["label"]): r for r in bt}
    cells = sorted({(r["family"], r["axis"], r["K"]) for r in bt})

    print("=== A: matched data fraction (both on the 85% search run) ===")
    print("    fixed-200 MINUS F13-selected; negative = the fixed budget is better")
    for field, nm in (("log_loss", "corner log-loss"), ("smece", "corner smECE")):
        for grp in (None, "item", "annot"):
            d = [idx[c + ("fixed-200",)][field] - idx[c + ("F13-selected",)][field]
                 for c in cells if (grp is None or c[1] == grp)
                 and idx.get(c + ("fixed-200",), {}).get(field) is not None]
            summ(d, f"{nm} [{grp or 'ALL'}]")
        print()

    f13p = RUNS / "reports/tables_f13.json"
    if not f13p.is_file():
        print("(tables_f13.json not written yet -- comparison C pending)")
        return 0
    f13 = json.loads(f13p.read_text())
    pub: dict[tuple[str, int], dict] = {}
    for axis, ser in f13["axes"].items():
        for label, d in ser.items():
            fam = "Marformer" if label.endswith("(T)") else "MarformerNT"
            for k, v in d.items():
                pub[(fam, axis, int(k))] = v

    print("=== C: published 100% refit vs fixed-200 on the 85% search run ===")
    print("    fixed-200 MINUS published; CONFOUNDED (refit has 15% more data)")
    for field, nm in (("ll", "corner log-loss"), ("smece", "corner smECE")):
        src = "log_loss" if field == "ll" else "smece"
        for grp in (None, "item", "annot"):
            d = []
            for c in cells:
                p, f = pub.get(c), idx.get(c + ("fixed-200",))
                if p is None or f is None or p.get(field) is None or f.get(src) is None:
                    continue
                if grp and c[1] != grp:
                    continue
                d.append(f[src] - p[field])
            if d:
                summ(d, f"{nm} [{grp or 'ALL'}]")
        print()

    print("=== where a fixed 200 is WORSE than the published number (log-loss) ===")
    bad = []
    for c in cells:
        p, f = pub.get(c), idx.get(c + ("fixed-200",))
        if p and f:
            bad.append((f["log_loss"] - p["ll"], c, p["ll"], f["log_loss"]))
    bad.sort(reverse=True)
    for diff, c, a, b in bad[:10]:
        if diff <= 0:
            break
        print(f"  {c[0]}/{c[1]}/{c[2]:<6} published {a:.4f} -> fixed-200 {b:.4f}  {diff:+.4f}")
    print(f"\n  cells where fixed-200 is worse: "
          f"{sum(1 for d, *_ in bad if d > 0)}/{len(bad)}")

    print("\n=== same, on smECE ===")
    badc = []
    for c in cells:
        p, f = pub.get(c), idx.get(c + ("fixed-200",))
        if p and f and p.get("smece") is not None and f.get("smece") is not None:
            badc.append((f["smece"] - p["smece"], c, p["smece"], f["smece"]))
    badc.sort(reverse=True)
    for diff, c, a, b in badc[:8]:
        if diff <= 0:
            break
        print(f"  {c[0]}/{c[1]}/{c[2]:<6} published {a:.4f} -> fixed-200 {b:.4f}  {diff:+.4f}")
    print(f"\n  cells where fixed-200 is worse on smECE: "
          f"{sum(1 for d, *_ in badc if d > 0)}/{len(badc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
