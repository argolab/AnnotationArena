#!/usr/bin/env python
"""
Pool the per-data-seed evaluations and answer one question: how much of the
train-vs-corner gap is a property of the model, and how much is the particular
414-cell corner that base seed 42 happened to draw?

Reads the grid_eval.py payloads written per seed and emits
  - a table of corner / train / gap per (data seed, arm, K)
  - the across-seed spread of the gap at each (arm, K)
  - dataseed_fig6.png: corner log-loss vs K, one line per data seed

    python analyse.py --in out/eval_s*.json -o out/
"""
from __future__ import annotations
import argparse, json, re, statistics as st
from pathlib import Path

def load(paths):
    rows = []
    for p in paths:
        seed = int(re.search(r"eval_s(\d+)", Path(p).name).group(1))
        for r in json.loads(Path(p).read_text())["rows"]:
            r["data_seed"] = seed
            rows.append(r)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--eval-max-item", type=int, default=300)
    ap.add_argument("--corner-whole", type=int, default=1)
    ap.add_argument("-o", "--outdir", default="out")
    a = ap.parse_args()
    rows = [r for r in load(a.inp)
            if r["eval_max_item"] == a.eval_max_item
            and r["corner_whole"] == bool(a.corner_whole)]
    for r in rows:
        r["gap"] = r["corner"]["log_loss"] - r["train"]["log_loss"]

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    Path(out / "dataseed_rows.json").write_text(json.dumps(rows, indent=2))

    arms = ["Marformer", "MarformerNT"]
    sizes = sorted({r["size"] for r in rows})
    seeds = sorted({r["data_seed"] for r in rows})

    print(f"eval_max_item={a.eval_max_item}  corner_whole={bool(a.corner_whole)}  "
          f"{len(rows)} rows  seeds={seeds}\n")
    print(f"{'arm':<13}{'K':>6}{'seed':>6}{'corner_n':>9}{'train':>9}{'corner':>9}{'gap':>9}")
    for arm in arms:
        for K in sizes:
            for s in seeds:
                m = [r for r in rows if r["family"] == arm and r["size"] == K
                     and r["data_seed"] == s]
                if not m: continue
                r = m[0]
                print(f"{arm:<13}{K:>6}{s:>6}{r['corner']['n']:>9}"
                      f"{r['train']['log_loss']:>9.4f}{r['corner']['log_loss']:>9.4f}"
                      f"{r['gap']:>+9.4f}")

    print(f"\n=== gap across data seeds ===")
    print(f"{'arm':<13}{'K':>6}{'n':>4}{'mean':>9}{'sd':>8}{'min':>9}{'max':>9}"
          f"{'  seed42':>10}")
    spread = []
    for arm in arms:
        for K in sizes:
            g = {r["data_seed"]: r["gap"] for r in rows
                 if r["family"] == arm and r["size"] == K}
            if len(g) < 2: continue
            v = list(g.values())
            sd = st.stdev(v) if len(v) > 1 else float("nan")
            s42 = g.get(42, float("nan"))
            print(f"{arm:<13}{K:>6}{len(v):>4}{st.mean(v):>+9.4f}{sd:>8.4f}"
                  f"{min(v):>+9.4f}{max(v):>+9.4f}{s42:>+10.4f}")
            spread.append({"arm": arm, "K": K, "n": len(v), "mean": st.mean(v),
                           "sd": sd, "min": min(v), "max": max(v), "seed42": s42})
    Path(out / "dataseed_gap.json").write_text(json.dumps(spread, indent=2))

    # --- figure ---------------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib unavailable; skipping figure)"); return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    cols = {s: cmap(i / max(1, len(seeds) - 1) * 0.85) for i, s in enumerate(seeds)}
    for ax, arm, title in zip(axes, arms,
                              ["(a) Marformer (transductive)",
                               "(b) Marformer NT (non-transductive)"]):
        for s in seeds:
            pts = sorted((r["size"], r["corner"]["log_loss"]) for r in rows
                         if r["family"] == arm and r["data_seed"] == s)
            if not pts: continue
            lw, z = (2.4, 3) if s == 42 else (1.3, 2)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=cols[s], lw=lw, ms=4.5, zorder=z,
                    label=f"data seed {s}" + (" (original)" if s == 42 else ""))
            # train-partition loss, same checkpoint, as a dashed floor
            tp = sorted((r["size"], r["train"]["log_loss"]) for r in rows
                        if r["family"] == arm and r["data_seed"] == s)
            ax.plot([p[0] for p in tp], [p[1] for p in tp], "--",
                    color=cols[s], lw=0.9, alpha=0.55, zorder=1)
        ax.set_xscale("log"); ax.set_xticks(sizes)
        ax.set_xticklabels([str(x) for x in sizes])
        ax.set_xlabel("training items K"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("log-loss")
    axes[1].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Corner log-loss (solid) vs train-partition log-loss (dashed), "
                 "per generative draw", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "dataseed_fig6.png", dpi=170)
    print(f"\n-> {out/'dataseed_fig6.png'}")

if __name__ == "__main__":
    main()
