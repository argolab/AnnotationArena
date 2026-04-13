#!/usr/bin/env python3
"""Summarize in-progress or finished grok ablation runs using curves_live.json or curves.json.

Dev metric: tail average of test_mse (masked test MSE), matching the toy's logged test_masked_mse.

``initial_dev`` = mean of the first up to ``--initial-window`` steps of test_mse.
``grok_pct`` = 100 * tail_dev / initial_dev (small ⇒ tail much better than early training ⇒ grokked).

For **finished** runs, ``first_grok_step`` is the first 1-based step where rolling ``grok_pct`` (same tail window as ``--tail``) stays strictly below ``--grok-threshold-pct`` for ``--grok-sustain`` consecutive steps. Cached in each run dir as ``monitor_grok_step_cache.json`` so repeat monitor invocations skip the scan.

The text table shortens batch directory names (e.g. ``grok_ablation_4x4_r1_s10000`` → ``4x4_r1``); CSV ``batch`` uses the same short label.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CACHE_VERSION = 1
CACHE_FILENAME = "monitor_grok_step_cache.json"


DEFAULT_ROOTS = (
    "OUTPUT/grok_ablation_4x4_r1_s10000",
    "OUTPUT/grok_ablation_6x6_r1_s10000",
    "OUTPUT/grok_ablation_5x5_r2_s10000",
    "OUTPUT/grok_ablation_7x7_r2_s10000",
)


def _tail_mean(xs: Sequence[float], tail: int) -> Optional[float]:
    if not xs:
        return None
    chunk = xs[-tail:] if len(xs) >= tail else xs
    return float(statistics.mean(chunk))


def _initial_mean(xs: Sequence[float], window: int) -> Optional[float]:
    """Mean of xs[0 : min(window, len(xs))]. None if xs is empty."""
    if not xs:
        return None
    n = min(int(window), len(xs))
    if n <= 0:
        return None
    return float(statistics.mean(xs[:n]))


def _grok_pct(tail_dev: Optional[float], initial_dev: Optional[float]) -> Optional[float]:
    if tail_dev is None or initial_dev is None:
        return None
    if initial_dev <= 0.0:
        return None
    return 100.0 * float(tail_dev) / float(initial_dev)


def _rolling_tail_means(te: Sequence[float], tail: int) -> List[float]:
    """Mean of te over indices [max(0,i-tail+1), i] for each i (0-based). O(n)."""
    n = len(te)
    if n == 0:
        return []
    ps = [0.0]
    for x in te:
        ps.append(ps[-1] + float(x))
    out: List[float] = []
    for i in range(n):
        start = max(0, i - int(tail) + 1)
        w = i - start + 1
        out.append((ps[i + 1] - ps[start]) / float(w))
    return out


def _first_grok_step(
    te: Sequence[float],
    initial_dev: float,
    *,
    tail: int,
    threshold_pct: float,
    sustain: int,
) -> Optional[int]:
    """First 1-based step index where rolling grok_pct stays < threshold_pct for ``sustain`` steps; None if never."""
    n = len(te)
    if n < sustain or initial_dev <= 0.0:
        return None
    rolling = _rolling_tail_means(te, tail)
    for start in range(0, n - sustain + 1):
        ok = True
        for k in range(sustain):
            pct = 100.0 * rolling[start + k] / float(initial_dev)
            if pct >= float(threshold_pct):
                ok = False
                break
        if ok:
            return start + 1
    return None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _grok_cache_path(run_dir: Path) -> Path:
    return run_dir / CACHE_FILENAME


def _load_grok_cache(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = _grok_cache_path(run_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _cache_valid(
    cache: Dict[str, Any],
    *,
    curve_len: int,
    source_name: str,
    params: Dict[str, Any],
) -> bool:
    if cache.get("version") != CACHE_VERSION:
        return False
    if cache.get("params") != params:
        return False
    if int(cache.get("curve_len", -1)) != curve_len:
        return False
    if cache.get("source") != source_name:
        return False
    return True


def _resolve_first_grok_step(
    run_dir: Path,
    status: str,
    te: Sequence[float],
    source_name: str,
    init_te: Optional[float],
    params: Dict[str, Any],
    *,
    use_cache: bool,
) -> Optional[int]:
    """None = not finished, still running, or never crossed threshold; int = first grok step."""
    if status != "done":
        return None
    if init_te is None or init_te <= 0.0:
        return None
    n = len(te)
    sustain = int(params["grok_sustain"])
    if n < sustain:
        return None

    if use_cache:
        c = _load_grok_cache(run_dir)
        if c is not None and _cache_valid(c, curve_len=n, source_name=source_name, params=params):
            raw = c.get("first_grok_step")
            return int(raw) if raw is not None else None

    result = _first_grok_step(
        te,
        float(init_te),
        tail=int(params["tail"]),
        threshold_pct=float(params["grok_threshold_pct"]),
        sustain=sustain,
    )

    if use_cache:
        _atomic_write_json(
            _grok_cache_path(run_dir),
            {
                "version": CACHE_VERSION,
                "params": params,
                "curve_len": n,
                "source": source_name,
                "first_grok_step": result,
            },
        )
    return result


def _short_batch(name: str) -> str:
    """Display label: drop ``grok_ablation_`` and trailing ``_s{steps}`` (e.g. ``_s10000``)."""
    if name == "grok_success_ablation":
        return "2x2"
    s = name
    if s.startswith("grok_ablation_"):
        s = s[len("grok_ablation_") :]
    s = re.sub(r"_s\d+$", "", s)
    if s == name:
        return name
    return s if s else name


def _load_run(run_dir: Path) -> Tuple[str, Dict[str, Any], Path]:
    """Return (status, data, path_used).

    status: ``running`` (checkpoint, not at planned num_steps yet), ``done``, or ``missing``.
    """
    live = run_dir / "curves_live.json"
    final = run_dir / "curves.json"
    if live.exists():
        data = json.loads(live.read_text())
        planned = int(data.get("num_steps") or 0)
        done = int(data.get("completed_steps") or len(data.get("test_mse") or []))
        tag = "running" if planned > 0 and done < planned else "done"
        return tag, data, live
    if final.exists():
        return "done", json.loads(final.read_text()), final
    return "missing", {}, run_dir


def _dir_has_curves(d: Path) -> bool:
    return (d / "curves_live.json").exists() or (d / "curves.json").exists()


def _collect_runs(roots: Sequence[Path]) -> List[Tuple[Path, Path]]:
    """Pairs (batch_root, run_dir).

    If ``root`` itself contains curve files, treat it as a single run (batch label = root name).
    Otherwise scan immediate subdirectories.
    """
    out: List[Tuple[Path, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        if _dir_has_curves(root):
            out.append((root, root))
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir():
                out.append((root, child))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="Batch directory (e.g. OUTPUT/grok_ablation_4x4_r1_s10000). Repeatable.",
    )
    p.add_argument(
        "--include-legacy-2x2",
        action="store_true",
        help="Also scan OUTPUT/grok_success_ablation (2×2, 2k runs).",
    )
    p.add_argument(
        "--tail",
        type=int,
        default=100,
        help="Average dev loss over the last this many steps (default 100).",
    )
    p.add_argument(
        "--initial-window",
        type=int,
        default=100,
        help="Average early dev over the first this many test_mse points (default 100).",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="Print machine-readable CSV header + rows.",
    )
    p.add_argument(
        "--grok-threshold-pct",
        type=float,
        default=5.0,
        help="Rolling grok_pct must stay strictly below this (default 5).",
    )
    p.add_argument(
        "--grok-sustain",
        type=int,
        default=50,
        help="Number of consecutive steps rolling grok_pct must stay below threshold (default 50).",
    )
    p.add_argument(
        "--no-grok-cache",
        action="store_true",
        help="Do not read/write monitor_grok_step_cache.json in each run directory.",
    )
    args = p.parse_args()

    grok_params = {
        "tail": args.tail,
        "initial_window": args.initial_window,
        "grok_threshold_pct": args.grok_threshold_pct,
        "grok_sustain": args.grok_sustain,
    }

    roots: List[Path] = []
    if args.roots:
        roots.extend(Path(r).resolve() for r in args.roots)
    else:
        here = Path.cwd()
        for rel in DEFAULT_ROOTS:
            roots.append((here / rel).resolve())
    if args.include_legacy_2x2:
        roots.append((Path.cwd() / "OUTPUT/grok_success_ablation").resolve())

    rows: List[Dict[str, Any]] = []
    for batch_root, run_dir in _collect_runs(roots):
        run_key = "." if run_dir.resolve() == batch_root.resolve() else run_dir.name
        status, data, src = _load_run(run_dir)
        if status == "missing":
            rows.append(
                {
                    "batch_dir": batch_root.name,
                    "batch": _short_batch(batch_root.name),
                    "run": run_key,
                    "status": "missing",
                    "source": "",
                    "completed_steps": None,
                    "planned_steps": None,
                    "tail_dev_mse": None,
                    "tail_train_mse": None,
                    "last_dev_mse": None,
                    "initial_dev_mse": None,
                    "grok_pct": None,
                    "first_grok_step": None,
                    "first_grok_never": None,
                }
            )
            continue
        te = data.get("test_mse") or []
        tr = data.get("train_mse") or []
        planned = data.get("num_steps")
        completed = data.get("completed_steps", len(te))
        tail_te = _tail_mean(te, args.tail)
        tail_tr = _tail_mean(tr, args.tail)
        last_te = float(te[-1]) if te else None
        init_te = _initial_mean(te, args.initial_window)
        grok_pct = _grok_pct(tail_te, init_te)
        use_cache = not args.no_grok_cache
        if status == "done":
            fg = _resolve_first_grok_step(
                run_dir,
                status,
                te,
                src.name,
                init_te,
                grok_params,
                use_cache=use_cache,
            )
            first_grok_never = fg is None and init_te is not None and init_te > 0 and len(te) >= grok_params["grok_sustain"]
        else:
            fg = None
            first_grok_never = None
        rows.append(
            {
                "batch_dir": batch_root.name,
                "batch": _short_batch(batch_root.name),
                "run": run_key,
                "status": status,
                "source": src.name,
                "completed_steps": completed,
                "planned_steps": planned,
                "tail_dev_mse": tail_te,
                "tail_train_mse": tail_tr,
                "last_dev_mse": last_te,
                "initial_dev_mse": init_te,
                "grok_pct": grok_pct,
                "first_grok_step": fg,
                "first_grok_never": first_grok_never,
            }
        )

    rows.sort(key=lambda r: (r["batch_dir"], r["run"]))

    if args.csv:
        cols = [
            "batch",
            "run",
            "status",
            "source",
            "completed_steps",
            "planned_steps",
            "tail_dev_mse",
            "tail_train_mse",
            "last_dev_mse",
            "initial_dev_mse",
            "grok_pct",
            "first_grok_step",
        ]
        print(",".join(cols))
        for r in rows:
            fg = r["first_grok_step"]
            fg_csv = ""
            if r["status"] == "done":
                if fg is not None:
                    fg_csv = str(fg)
                elif r.get("first_grok_never"):
                    fg_csv = "never"
            out = {**r, "first_grok_step": fg_csv}
            print(",".join("" if out[c] is None else str(out[c]) for c in cols))
        return

    print(
        f"tail_dev_mse = mean(test_mse over last {args.tail} steps); "
        f"initial_dev_mse = mean(first min(n, {args.initial_window}) steps); "
        "grok_pct = 100 * tail_dev / initial_dev (lower ⇒ more grok).\n"
        f"first_grok (done only): first step where rolling grok_pct < {args.grok_threshold_pct}% "
        f"for {args.grok_sustain} consecutive steps; cached as {CACHE_FILENAME}.\n"
        "test_mse = masked test MSE (same as training log test_masked_mse).\n"
    )
    _bw, _rw = 14, 34
    _gw = 10
    hdr = (
        f"{'batch':<{_bw}} {'run':<{_rw}} {'stat':<6} {'done':>6} {'plan':>6} "
        f"{'init_dev':>11} {'tail_dev':>11} {'tail_tr':>11} {'last_dev':>11} {'grok_pct':>9} "
        f"{'grok@':>{_gw}}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        done = "" if r["completed_steps"] is None else str(r["completed_steps"])
        plan = "" if r["planned_steps"] is None else str(r["planned_steps"])
        idv = "" if r["initial_dev_mse"] is None else f"{r['initial_dev_mse']:.4f}"
        td = "" if r["tail_dev_mse"] is None else f"{r['tail_dev_mse']:.4f}"
        tt = "" if r["tail_train_mse"] is None else f"{r['tail_train_mse']:.4f}"
        ld = "" if r["last_dev_mse"] is None else f"{r['last_dev_mse']:.4f}"
        gp = "" if r["grok_pct"] is None else f"{r['grok_pct']:.2f}%"
        if r["status"] != "done":
            gat = ""
        elif r["first_grok_step"] is not None:
            gat = str(r["first_grok_step"])
        elif r.get("first_grok_never"):
            gat = "never"
        else:
            gat = ""
        br = (r["batch"] or "")[:_bw]
        rn = (r["run"] or "")[:_rw]
        print(
            f"{br:<{_bw}} {rn:<{_rw}} {r['status']:<6} {done:>6} {plan:>6} "
            f"{idv:>11} {td:>11} {tt:>11} {ld:>11} {gp:>9} {gat:>{_gw}}"
        )


if __name__ == "__main__":
    main()
