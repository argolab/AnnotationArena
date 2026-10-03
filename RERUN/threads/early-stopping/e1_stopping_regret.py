#!/usr/bin/env python
"""E1: how much does dev-set early stopping cost against the test-curve minimum?

Jason's question in the thread: the stopping epoch must be chosen on a separate
validation set, and his read of the graphs was charitable -- "I'm assuming that
the validation data will be able to identify a stopping point that's close to
the minimum of these test graphs."

This measures exactly that assumption, with no charity required.

For each training size K on the item axis we have two runs on the SAME data draw
and the SAME model seed:

  search  trained on 85% of the training partition, with the held-out 15% block
          scored every epoch (`training_history.json` -> val_eval).  This is the
          curve the protocol is allowed to look at.
  full    trained on 100% of the training partition for the full 400-epoch
          budget, scored on TEST at every 25th epoch (`TEST_RESULTS/periodic-*`).
          This is the curve the protocol is NOT allowed to look at, and whose
          minimum is the best any stopping rule could have done.

The shipped rule picks e* = argmin of the smoothed dev curve, then refits on the
full partition for e* epochs.  So the honest cost of the protocol is

    regret(K) = test_full(e*) - min_e test_full(e)

read off the full-train curve, in nats.  We also report the epoch gap and the
observed-token loss, which should sit at ~0 throughout in both arms.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

RERUN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RERUN))
# Reuse the shipped selector rather than a look-alike, so what is measured here
# is the rule that actually runs: centred moving average, argmin, 25-epoch floor.
from analysis.select_epochs import smooth, val_curve   # noqa: E402
ITEM = "RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM"
SEARCH = RERUN / "runs/legacy/val-block-01" / ITEM / "valblock-search"
FULL = RERUN / "runs/replicate-01" / ITEM
SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
SMOOTH = 5


def dev_curve(cell: Path) -> list[tuple[int, float]]:
    """(epoch, log-loss) on the held-out validation block, scored every epoch."""
    return val_curve(json.loads((cell / "training_history.json").read_text()))


def test_curve(cell: Path) -> tuple[list[int], list[float], list[float]]:
    """Test log-loss (missing, observed) at each periodic checkpoint."""
    ep, miss, obs = [], [], []
    for p in sorted((cell / "TEST_RESULTS").glob("periodic-epoch=*.json")):
        j = json.loads(p.read_text())
        ep.append(int(re.search(r"epoch=(\d+)", p.name).group(1)) + 1)
        miss.append(float(j["missing"]["log_loss"]))
        obs.append(float(j["observed"]["log_loss"]))
    return ep, miss, obs


MIN_EPOCHS = 25


def select(curve: list[tuple[int, float]]) -> int | None:
    """The refit budget the shipped per-run selector would hand back.

    `None` means the cell has no validation signal at all -- Item_T_100 carves a
    block out of a training partition that has no items in it -- and the real
    selector falls back to the axis median there rather than selecting on data.
    """
    if not curve:
        return None
    sm = smooth([c[1] for c in curve], SMOOTH)
    best = curve[min(range(len(sm)), key=lambda i: sm[i])][0]
    return max(MIN_EPOCHS, best + 1)


def interp(xs: list[int], ys: list[float], x: float) -> float:
    """Linear interpolation on the 25-epoch checkpoint grid."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for a, b, ya, yb in zip(xs, xs[1:], ys, ys[1:]):
        if a <= x <= b:
            return ya + (yb - ya) * (x - a) / (b - a)
    return ys[-1]


def main() -> None:
    seed = sys.argv[1] if len(sys.argv) > 1 else "42"
    rows = []
    for K in SIZES:
        name = f"DOMAIN3-SIGMOID_Item_T_{K}_MARFORMER"
        s, f = SEARCH / f"seed{seed}" / name, FULL / name
        if not (s / "training_history.json").exists() or not (f / "TEST_RESULTS").exists():
            print(f"K={K}: missing cell, skipped")
            continue
        dev = dev_curve(s)
        e_star = select(dev)
        if e_star is None:
            print(f"K={K}: no validation signal to select on (empty carve); "
                  f"the protocol falls back to the axis median here")
            continue
        f_ep, f_miss, f_obs = test_curve(f)
        s_ep, s_miss, s_obs = test_curve(s)

        at_star = interp(f_ep, f_miss, e_star)
        e_oracle = f_ep[min(range(len(f_miss)), key=lambda i: f_miss[i])]
        floor = min(f_miss)
        rows.append({
            "K": K, "seed": int(seed),
            "e_star": e_star, "e_oracle_full": e_oracle,
            "test_at_e_star": at_star, "test_floor_full": floor,
            "regret": at_star - floor,
            "test_final_full": f_miss[-1],
            "cost_of_no_stopping": f_miss[-1] - floor,
            "observed_at_e_star_full": interp(f_ep, f_obs, e_star),
            "observed_final_full": f_obs[-1],
            "dev_epochs": [c[0] for c in dev], "dev_loss": [c[1] for c in dev],
            "full_epochs": f_ep, "full_missing": f_miss, "full_observed": f_obs,
            "search_epochs": s_ep, "search_missing": s_miss, "search_observed": s_obs,
        })
        print(f"K={K:<5} e*={e_star:<4} oracle={e_oracle:<4}  "
              f"test(e*)={at_star:.4f}  floor={floor:.4f}  regret={at_star-floor:+.4f}  "
              f"(no stopping would cost {f_miss[-1]-floor:+.4f})")

    if rows:
        n = len(rows)
        print(f"\nmean regret {sum(r['regret'] for r in rows)/n:+.4f} nats over {n} sizes; "
              f"mean cost of running to 400 {sum(r['cost_of_no_stopping'] for r in rows)/n:+.4f}")
    out = RERUN / f"threads/early-stopping/out/e1_seed{seed}.json"
    out.write_text(json.dumps(rows))
    print("wrote", out)


if __name__ == "__main__":
    main()
