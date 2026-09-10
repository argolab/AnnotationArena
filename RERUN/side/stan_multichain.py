#!/usr/bin/env python
"""
TRUE between-chain R-hat for the 4-chain Stan refits.

The single-chain runs could only be checked with split-R-hat, which cannot see
multimodality: a chain stuck in one mode looks perfectly converged. With 4
chains started from different inits, R-hat compares them against each other, so
a mode that one chain never visits shows up.

Reported per parameter GROUP, because pooling everything is what made the
original diagnostic look alarming: the per-item/per-annotator latents carry sign
and rotation symmetries and can have huge R-hat while predictions -- the only
thing Stan is scored on here -- are unaffected. lp__ and the variance
hyperparameters are the invariant quantities that actually have to mix.

Job-only; parses several hundred MB of CmdStan CSV.
"""
from __future__ import annotations

import argparse, glob, json, os, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stan_diagnostics import read_cmdstan_csv

SAMPLER = {"lp__", "accept_stat__", "stepsize__", "treedepth__",
           "n_leapfrog__", "divergent__", "energy__"}


def group_of(name: str) -> str:
    if name == "lp__":
        return "lp__"
    if name.startswith("sigma"):
        return "hyperparameter"
    return "latent"


def rhat_ess(chains: np.ndarray) -> tuple[float, float]:
    """Standard between-chain R-hat and ESS. chains: (m, n)."""
    m, n = chains.shape
    if n < 4:
        return float("nan"), float("nan")
    W = chains.var(axis=1, ddof=1).mean()
    if W <= 0:
        return float("nan"), float("nan")
    B = n * chains.mean(axis=1).var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    rhat = float(np.sqrt(var_hat / W))
    # ESS via averaged autocorrelation (Geyer initial positive sequence)
    acov = np.zeros(n)
    for c in chains:
        c = c - c.mean()
        f = np.fft.rfft(c, n=2 * n)
        acov += np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n] / n
    acov /= m
    if acov[0] <= 0:
        return rhat, float("nan")
    rho = acov / acov[0]
    t, total = 1, 0.0
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        total += pair
        t += 2
    tau = 1.0 + 2.0 * total
    return rhat, (float(m * n / tau) if tau > 0 else float("nan"))


def analyse_run(run_dir: Path) -> dict | None:
    csvs = sorted(p for p in run_dir.glob("*.csv")
                  if not p.name.startswith("posterior_hyperparameters"))
    if not csvs:
        return None
    per_chain, names = [], None
    ndiv = 0
    for c in csvs:
        nm, dr, _ = read_cmdstan_csv(c)
        if nm is None:
            continue
        if names is None:
            names = nm
        elif nm != names:
            print(f"   !! column mismatch in {c.name}, skipping", flush=True)
            continue
        per_chain.append(dr)
        if "divergent__" in nm:
            ndiv += int(dr[:, nm.index("divergent__")].sum())
    if not per_chain:
        return None
    n = min(d.shape[0] for d in per_chain)
    stack = np.stack([d[:n] for d in per_chain])          # (m, n, p)
    m = stack.shape[0]
    print(f"   {m} chains x {n} draws", flush=True)

    buckets: dict[str, list] = {}
    for i, nm in enumerate(names):
        if nm.endswith("__") and nm != "lp__":
            continue
        x = stack[:, :, i]
        if np.allclose(x, x.flat[0]):
            continue
        r, e = rhat_ess(x)
        if np.isfinite(r):
            buckets.setdefault(group_of(nm), []).append((nm, r, e))

    out = {"run": run_dir.name, "n_chains": m, "n_draws_per_chain": n,
           "divergences": ndiv, "groups": {}}
    for g, items in buckets.items():
        rh = np.array([r for _, r, _ in items])
        es = np.array([e for _, _, e in items if np.isfinite(e)])
        worst = max(items, key=lambda t: t[1])
        out["groups"][g] = {
            "n": len(items),
            "rhat_max": float(rh.max()),
            "rhat_median": float(np.median(rh)),
            "rhat_gt_1.01_pct": 100.0 * float((rh > 1.01).mean()),
            "ess_min": float(es.min()) if es.size else None,
            "ess_median": float(np.median(es)) if es.size else None,
            "worst_param": worst[0],
        }
        print(f"   {g:<15} n={len(items):<7} rhat max={rh.max():.3f} med={np.median(rh):.3f}"
              f"  >1.01={100.0*(rh>1.01).mean():.0f}%"
              f"  ESS min={out['groups'][g]['ess_min']:.0f}"
              f"  worst={worst[0]}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    results = []
    for root in a.roots.split(","):
        for d in sorted(glob.glob(os.path.join(root, "**", "*_PROJ"), recursive=True)):
            p = Path(d)
            if not p.is_dir() or p.name.endswith("_eval"):
                continue
            print(f"\n{p.name}", flush=True)
            r = analyse_run(p)
            if r:
                results.append(r)
    Path(a.out).write_text(json.dumps({"runs": results}, indent=1))
    print(f"\nwrote {len(results)} runs -> {a.out}")


if __name__ == "__main__":
    main()
