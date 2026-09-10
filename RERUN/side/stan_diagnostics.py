#!/usr/bin/env python
"""
Did the Stan runs actually mix, especially at K=4000?

The Domain 3 Stan baselines are the reference the whole Marformer comparison is
measured against, so if the sampler did not converge at the largest size the gap
we report is not a model comparison at all. These runs use **one chain** (300
warmup / 500 sampling, adapt_delta 0.85, max_treedepth 12), so the usual
between-chain R-hat is unavailable; split-R-hat on the single chain is the best
that can be done, and that limitation is itself worth reporting.

Reports per run: divergent transitions, treedepth saturation, E-BFMI, and
split-R-hat / bulk-ESS / tail-ESS over the sampled parameters.

Run it as a job, never on the login node -- it parses multi-hundred-MB CmdStan
CSVs. See RERUN/side/stan_diag.sbatch.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

SAMPLER_COLS = {"lp__", "accept_stat__", "stepsize__", "treedepth__",
                "n_leapfrog__", "divergent__", "energy__"}


def read_cmdstan_csv(path: Path):
    """Return (names, draws array, max_treedepth from the header comments)."""
    names, rows, max_depth = None, [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                m = re.search(r"max_depth\s*=\s*(\d+)", line)
                if m:
                    max_depth = int(m.group(1))
                continue
            line = line.strip()
            if not line:
                continue
            if names is None:
                names = line.split(",")
                continue
            parts = line.split(",")
            if len(parts) != len(names):      # truncated final line
                continue
            rows.append(parts)
    if names is None or not rows:
        return None, None, max_depth
    return names, np.asarray(rows, dtype=float), max_depth


def split_rhat(x: np.ndarray) -> float:
    """Split-R-hat on a single chain: split in half, treat as two chains."""
    n = len(x) // 2
    if n < 4:
        return float("nan")
    chains = np.stack([x[:n], x[n:2 * n]])
    means = chains.mean(axis=1)
    variances = chains.var(axis=1, ddof=1)
    W = variances.mean()
    B = n * means.var(ddof=1)
    if W <= 0:
        return float("nan")
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def ess(x: np.ndarray) -> float:
    """Effective sample size via Geyer's initial positive sequence, split-chain."""
    n = len(x) // 2
    if n < 8:
        return float("nan")
    chains = np.stack([x[:n], x[n:2 * n]])
    m = chains.shape[0]
    W = chains.var(axis=1, ddof=1).mean()
    if W <= 0:
        return float("nan")
    # autocovariance per chain via FFT, averaged
    acov = np.zeros(n)
    for c in chains:
        c = c - c.mean()
        f = np.fft.rfft(c, n=2 * n)
        a = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n]
        acov += a / n
    acov /= m
    if acov[0] <= 0:
        return float("nan")
    rho = acov / acov[0]
    # Geyer initial positive sequence on paired sums
    t, total = 1, 0.0
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        total += pair
        t += 2
    tau = 1.0 + 2.0 * total
    return float(m * n / tau) if tau > 0 else float("nan")


def tail_ess(x: np.ndarray) -> float:
    """ESS of the tail, approximated by ESS of the indicator-rank transform."""
    q05, q95 = np.quantile(x, [0.05, 0.95])
    ind = ((x <= q05) | (x >= q95)).astype(float)
    return ess(ind)


def analyse(path: Path) -> dict | None:
    names, draws, max_depth = read_cmdstan_csv(path)
    if names is None:
        return None
    col = {nm: i for i, nm in enumerate(names)}
    out: dict = {"csv": path.name, "n_draws": int(draws.shape[0]),
                 "max_treedepth": max_depth}

    if "divergent__" in col:
        d = draws[:, col["divergent__"]]
        out["divergences"] = int(d.sum())
        out["divergence_pct"] = 100.0 * float(d.mean())
    if "treedepth__" in col and max_depth:
        td = draws[:, col["treedepth__"]]
        out["treedepth_max_hit_pct"] = 100.0 * float((td >= max_depth).mean())
        out["treedepth_mean"] = float(td.mean())
    if "energy__" in col:
        E = draws[:, col["energy__"]]
        dE = np.diff(E)
        v = E.var(ddof=1)
        out["ebfmi"] = float((dE ** 2).mean() / v) if v > 0 else float("nan")
    if "accept_stat__" in col:
        out["accept_stat_mean"] = float(draws[:, col["accept_stat__"]].mean())

    param_idx = [i for nm, i in col.items() if nm not in SAMPLER_COLS]
    param_names = [names[i] for i in param_idx]
    rhats, esss, tesss = [], [], []
    for i in param_idx:
        x = draws[:, i]
        if np.allclose(x, x[0]):          # constant / non-sampled
            continue
        rhats.append(split_rhat(x))
        esss.append(ess(x))
        tesss.append(tail_ess(x))
    rhats = np.asarray([r for r in rhats if np.isfinite(r)])
    esss = np.asarray([e for e in esss if np.isfinite(e)])
    tesss = np.asarray([e for e in tesss if np.isfinite(e)])
    if rhats.size:
        worst = int(np.argmax(rhats))
        out["n_params"] = int(rhats.size)
        out["rhat_max"] = float(rhats.max())
        out["rhat_p99"] = float(np.quantile(rhats, 0.99))
        out["rhat_gt_1.01_pct"] = 100.0 * float((rhats > 1.01).mean())
        out["rhat_gt_1.05_pct"] = 100.0 * float((rhats > 1.05).mean())
        out["worst_rhat_param"] = param_names[worst] if worst < len(param_names) else "?"
    if esss.size:
        out["ess_bulk_min"] = float(esss.min())
        out["ess_bulk_median"] = float(np.median(esss))
        out["ess_bulk_lt_100_pct"] = 100.0 * float((esss < 100).mean())
    if tesss.size:
        out["ess_tail_min"] = float(tesss.min())
        out["ess_tail_median"] = float(np.median(tesss))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True,
                    help="comma-separated directories to search for CmdStan CSVs")
    ap.add_argument("--pattern", default="*_4000_*,*_1000_*",
                    help="comma-separated run-directory name globs")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    results = []
    for root in args.roots.split(","):
        for pat in args.pattern.split(","):
            for run_dir in sorted(glob.glob(os.path.join(root, "**", pat), recursive=True)):
                if not os.path.isdir(run_dir):
                    continue
                for csv in sorted(Path(run_dir).glob("*.csv")):
                    if csv.name.startswith("posterior_hyperparameters"):
                        continue
                    print(f"reading {csv}", flush=True)
                    r = analyse(csv)
                    if r:
                        r["run"] = os.path.basename(run_dir)
                        r["dir"] = run_dir
                        results.append(r)
                        print(f"   draws={r['n_draws']} div={r.get('divergences')} "
                              f"rhat_max={r.get('rhat_max'):.4f} "
                              f"ess_min={r.get('ess_bulk_min'):.0f}", flush=True)

    Path(args.out).write_text(json.dumps({"runs": results}, indent=1))
    print(f"\nwrote {len(results)} runs -> {args.out}")


if __name__ == "__main__":
    main()
