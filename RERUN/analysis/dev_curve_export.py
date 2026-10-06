#!/usr/bin/env python
"""Train/dev curves plus the free half of the overconfidence signal.

training_history.json already records, per epoch, the training masked CE and the
dev block's missing-cell CE *and accuracy*. Accuracy is a pure location metric
and CE is location+confidence, so the two together already separate the two
hypotheses without touching a checkpoint:

    dev CE rises while dev accuracy keeps improving  ->  overconfidence
    dev CE rises and dev accuracy falls with it      ->  the prediction decayed

`mean_entropy_curves.py` then puts numbers on the split (entropy, predicted
variance, fitted temperature); this script is the cheap pass that says which
runs are worth scoring.

    python dev_curve_export.py --run-dir runs/final-dual -o out.json
"""
from __future__ import annotations
import argparse, json, re, statistics as st
from pathlib import Path

RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")


def curves(run: Path) -> dict | None:
    h = json.loads((run / "training_history.json").read_text())
    if not isinstance(h, list) or not h:
        return None
    ep, tr, tro, dce, dacc = [], [], [], [], []
    for i, e in enumerate(h):
        if not isinstance(e, dict):
            continue
        m = ((e.get("val_eval") or {}).get("metrics") or {}).get("missing", {}).get("rating", {})
        if not m:
            continue
        ep.append(int(e.get("epoch", i)))
        tr.append(e.get("train_masked_ce"))
        tro.append(e.get("train_observed_ce"))
        dce.append(m.get("xent"))
        dacc.append(m.get("acc"))
    if not ep:
        return None
    return {"epoch": ep, "train_masked_ce": tr, "train_observed_ce": tro,
            "dev_ce": dce, "dev_acc": dacc, "n_dev": m.get("n")}


def smooth(xs, k=5):
    out = []
    for i in range(len(xs)):
        w = [v for v in xs[max(0, i - k // 2): i + k // 2 + 1] if v is not None]
        out.append(st.mean(w) if w else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--include", default="valblock-search")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    root = Path(a.run_dir) / "RESULTS"
    rows = []
    for fam_dir, fam in (("MARFORMER", "Marformer"), ("MARFORMER-NT", "MarformerNT")):
        base = root / fam_dir / "DOMAIN3-SIGMOID"
        if not base.is_dir():
            continue
        for run in sorted(base.rglob("*_MARFORMER")):
            m = RUN_RE.search(run.name)
            if not m or a.include not in str(run):
                continue
            c = curves(run)
            if c is None:
                continue
            ce_s, acc_s = smooth(c["dev_ce"]), smooth(c["dev_acc"])
            i_ce = min(range(len(ce_s)), key=lambda i: ce_s[i])
            i_acc = max(range(len(acc_s)), key=lambda i: acc_s[i])
            rows.append({
                "family": fam, "axis": m.group(1).lower(), "K": int(m.group(3)),
                "run": str(run.relative_to(root)), **c,
                # The two argmins. Accuracy peaking much later than CE bottoms is
                # the overconfidence signature: the ranking keeps improving after
                # the probabilities stop being worth believing.
                "ce_argmin_epoch": c["epoch"][i_ce], "ce_min": ce_s[i_ce],
                "acc_argmax_epoch": c["epoch"][i_acc], "acc_max": acc_s[i_acc],
                "ce_at_acc_argmax": ce_s[i_acc],
                "acc_at_ce_argmin": acc_s[i_ce],
                "ce_end": ce_s[-1], "acc_end": acc_s[-1],
            })
            print(f"{fam:<12}{rows[-1]['axis']}/{rows[-1]['K']:<5} "
                  f"CE min {ce_s[i_ce]:.4f} @{c['epoch'][i_ce]:<5} -> end {ce_s[-1]:.4f} "
                  f"(+{ce_s[-1]-ce_s[i_ce]:.4f}) | "
                  f"acc max {acc_s[i_acc]:.4f} @{c['epoch'][i_acc]:<5} -> end {acc_s[-1]:.4f} "
                  f"({acc_s[-1]-acc_s[i_acc]:+.4f})")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows))

    late = [r for r in rows if r["acc_argmax_epoch"] > r["ce_argmin_epoch"]]
    gaps = [r["acc_argmax_epoch"] - r["ce_argmin_epoch"] for r in rows]
    rise = [r["ce_end"] - r["ce_min"] for r in rows]
    drop = [r["acc_max"] - r["acc_end"] for r in rows]
    print(f"\n{len(rows)} runs")
    print(f"  accuracy peaks AFTER CE bottoms in {len(late)}/{len(rows)}; "
          f"mean gap {st.mean(gaps):+.0f} epochs")
    print(f"  dev CE rises {st.mean(rise):+.4f} from its minimum to the end of the budget")
    print(f"  dev accuracy falls {st.mean(drop):+.4f} over the same stretch")
    print(f"\nwrote {len(rows)} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
