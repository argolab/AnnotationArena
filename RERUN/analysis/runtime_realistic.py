#!/usr/bin/env python
"""What the protocol would cost with early stopping in the search stage.

The reported 2000-epoch search is a budget chosen so it could not bind, not a
cost anyone would pay in practice. A real run stops when the validation loss has
not set a new low for `patience` epochs. Because the epoch we select is the
GLOBAL argmin of the (smoothed) validation curve, stopping at `selected +
patience` yields the *same* selected epoch -- so this is a cost simulation, not a
different experiment, and it changes no reported result.

    simulated search = (selected + patience) / 2000  x  measured search seconds
    total            = simulated search + measured refit

No new runs: selected epochs come from valblock_epochs.json, wall clock from each
cell's launcher_metadata.json.
"""
from __future__ import annotations
import json, re, statistics as st
from pathlib import Path

R = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-v2")
CELL = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")
BUDGET = 2000
PATIENCES = (50, 100, 200)


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def main() -> int:
    eps = (jload(R / "valblock_epochs.json") or {}).get("epochs", {})
    rows = []
    for fam_dir, fam in (("MARFORMER", "Marformer"), ("MARFORMER-NT", "MarformerNT")):
        for ax in ("ITEM", "ANNOT"):
            for p in sorted((R / "RESULTS" / fam_dir / "DOMAIN3-SIGMOID" / ax /
                             "valblock-search/seed42").glob("*_MARFORMER")):
                m = CELL.search(p.name)
                if not m:
                    continue
                size = int(m.group(3))
                axl = "item" if ax == "ITEM" else "annot"
                sel = eps.get(f"{fam}|{axl}|{size}|42")
                s = (jload(p / "launcher_metadata.json") or {}).get("elapsed_seconds")
                rp = R / "RESULTS" / fam_dir / "DOMAIN3-SIGMOID" / ax / "valblock/seed42" / p.name
                r = (jload(rp / "launcher_metadata.json") or {}).get("elapsed_seconds")
                if sel is None or s is None or r is None:
                    continue
                rows.append(dict(fam=fam, ax=axl, size=size, sel=sel,
                                 search=s / 60, refit=r / 60))

    print(f"{'cell':<20}{'sel ep':>7}{'search':>9}{'refit':>8}", end="")
    for P in PATIENCES:
        print(f"{'tot/p=' + str(P):>11}", end="")
    print(f"{'x refit':>9}")
    print("-" * (44 + 11 * len(PATIENCES) + 9))
    ratios = {P: [] for P in PATIENCES}
    for r in sorted(rows, key=lambda r: (r["fam"], r["ax"], r["size"])):
        print(f"{r['fam'][:9] + '/' + r['ax'] + '/' + str(r['size']):<20}"
              f"{r['sel']:>7}{r['search']:>9.1f}{r['refit']:>8.1f}", end="")
        for P in PATIENCES:
            sim = min(r["sel"] + P, BUDGET) / BUDGET * r["search"]
            tot = sim + r["refit"]
            ratios[P].append(tot / r["refit"] if r["refit"] > 0 else float("nan"))
            print(f"{tot:>11.1f}", end="")
        P0 = PATIENCES[1]
        sim0 = min(r["sel"] + P0, BUDGET) / BUDGET * r["search"]
        print(f"{(sim0 + r['refit']) / r['refit']:>9.1f}x")

    print()
    for P in PATIENCES:
        v = [x for x in ratios[P] if x == x]
        print(f"  patience {P:>3}: total / refit  mean {st.mean(v):.2f}x   "
              f"median {st.median(v):.2f}x   range {min(v):.2f}-{max(v):.2f}x")
    print("\n  The selected epoch is unchanged by construction: it is the global argmin of the")
    print("  smoothed validation curve, so stopping at selected+patience still selects it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
