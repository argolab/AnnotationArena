#!/usr/bin/env python3
"""
Analyze Stan diagnostics for DOMAIN3-SIGMOID runs.

Produces:
  - per-parameter summary CSVs
  - family-level diagnostics CSVs
  - chain-level predictive metrics CSVs
  - one markdown report with key findings
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "RESULTS/STAN/DOMAIN3-SIGMOID-ANALYSIS"

PROB_PREFIX = "missing_rating_probs."
SAMPLER_COLS = ["lp__", "accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__", "divergent__", "energy__"]
HYPERPARAMS = ["sigma_u", "sigma_v", "sigma_uit", "sigma_measurement", "kappa"]
EXCLUDED_FAMILIES = {
    "missing_rating_probs",
    "missing_rating_predictions",
    "log_lik_ratings_obs",
    "log_lik_pairwise_obs",
    "total_log_lik",
}
IJ_FAMILIES = {
    "base_scores",
    "eff_pref",
    "rating_probs",
    "rating_thresholds",
    "rating_cumprobs",
    "total_std",
}


@dataclass(frozen=True)
class RunSpec:
    slug: str
    label: str
    run_dir: Path
    eval_dir: Path


RUN_SPECS = [
    RunSpec(
        slug="item_t500_tensor_sig",
        label="Item | DOMAIN3-SIGMOID_Item_T_500_TENSOR-SIG",
        run_dir=ROOT / "RESULTS/STAN/DOMAIN3-SIGMOID/ORACLE/ITEM/DOMAIN3-SIGMOID_Item_T_500_TENSOR-SIG",
        eval_dir=ROOT / "RESULTS/STAN/DOMAIN3-SIGMOID/ORACLE/ITEM/DOMAIN3-SIGMOID_Item_T_500_TENSOR-SIG_eval",
    ),
]


def _print(msg: str) -> None:
    print(msg, flush=True)


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _family_name(param_name: str) -> str:
    bracket = param_name.find("[")
    if bracket != -1:
        return param_name[:bracket]
    if "." in param_name:
        return param_name.split(".", 1)[0]
    return param_name


def _ij_group_key(param_name: str) -> str | None:
    match = re.match(r"^([A-Za-z0-9_]+)\[(\d+),", param_name)
    if match is None:
        return None
    family = match.group(1)
    if family not in IJ_FAMILIES:
        return None
    ij = int(match.group(2))
    return f"{family}[{ij}]"


def _safe_nanmedian(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _safe_nanmax(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.max(arr)) if arr.size else float("nan")


def _safe_nanmin(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.min(arr)) if arr.size else float("nan")


def _safe_nanfrac(values: pd.Series, predicate) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.mean(predicate(arr))) if arr.size else float("nan")


def _ebfmi(energy: np.ndarray) -> float:
    if energy.size < 2:
        return float("nan")
    diffs = np.diff(energy)
    denom = np.var(energy, ddof=0)
    if denom <= 0:
        return float("nan")
    return float(np.mean(diffs * diffs) / denom)


def _load_csv_files(run_dir: Path) -> list[Path]:
    return sorted(
        p for p in run_dir.glob("*.csv")
        if p.name != "posterior_hyperparameters.csv"
    )


def _load_fit_summary(csv_files: list[Path]) -> pd.DataFrame:
    fit = cmdstanpy.from_csv([str(p) for p in csv_files])
    summary = fit.summary()
    summary = summary.reset_index().rename(columns={"index": "variable"})
    return summary


def _prepare_param_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["family"] = out["variable"].map(_family_name)
    out["ij_group"] = out["variable"].map(_ij_group_key)
    for col in ("R_hat", "ESS_bulk", "ESS_tail", "MCSE", "Mean", "StdDev"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _filtered_param_summary(param_summary: pd.DataFrame) -> pd.DataFrame:
    keep = ~param_summary["family"].isin(EXCLUDED_FAMILIES)
    return param_summary.loc[keep].copy()


def _family_summary(param_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = param_summary.groupby("family", dropna=False)
    for family, g in grouped:
        rhat = g.get("R_hat", pd.Series(dtype=float))
        ess_bulk = g.get("ESS_bulk", pd.Series(dtype=float))
        ess_tail = g.get("ESS_tail", pd.Series(dtype=float))
        rows.append({
            "family": family,
            "n_variables": int(len(g)),
            "mean_R_hat": float(pd.to_numeric(rhat, errors="coerce").dropna().mean()) if len(g) else np.nan,
            "median_R_hat": _safe_nanmedian(rhat),
            "max_R_hat": _safe_nanmax(rhat),
            "frac_R_hat_gt_1.01": _safe_nanfrac(rhat, lambda x: x > 1.01),
            "frac_R_hat_gt_1.05": _safe_nanfrac(rhat, lambda x: x > 1.05),
            "mean_ESS_bulk": float(pd.to_numeric(ess_bulk, errors="coerce").dropna().mean()) if len(g) else np.nan,
            "median_ESS_bulk": _safe_nanmedian(ess_bulk),
            "min_ESS_bulk": _safe_nanmin(ess_bulk),
            "mean_ESS_tail": float(pd.to_numeric(ess_tail, errors="coerce").dropna().mean()) if len(g) else np.nan,
            "median_ESS_tail": _safe_nanmedian(ess_tail),
            "min_ESS_tail": _safe_nanmin(ess_tail),
        })
    return pd.DataFrame(rows).sort_values(["max_R_hat", "min_ESS_bulk"], ascending=[False, True], na_position="last")


def _ij_summary(param_summary: pd.DataFrame) -> pd.DataFrame:
    grouped_df = param_summary[param_summary["ij_group"].notna()].copy()
    rows = []
    grouped = grouped_df.groupby("ij_group", dropna=False)
    for ij_group, g in grouped:
        rhat = g.get("R_hat", pd.Series(dtype=float))
        ess_bulk = g.get("ESS_bulk", pd.Series(dtype=float))
        ess_tail = g.get("ESS_tail", pd.Series(dtype=float))
        family = str(g["family"].iloc[0])
        rows.append({
            "ij_group": ij_group,
            "family": family,
            "n_variables": int(len(g)),
            "mean_R_hat": float(pd.to_numeric(rhat, errors="coerce").dropna().mean()) if len(g) else np.nan,
            "median_R_hat": _safe_nanmedian(rhat),
            "max_R_hat": _safe_nanmax(rhat),
            "mean_ESS_bulk": float(pd.to_numeric(ess_bulk, errors="coerce").dropna().mean()) if len(g) else np.nan,
            "median_ESS_bulk": _safe_nanmedian(ess_bulk),
            "min_ESS_bulk": _safe_nanmin(ess_bulk),
            "mean_ESS_tail": float(pd.to_numeric(ess_tail, errors="coerce").dropna().mean()) if len(g) else np.nan,
            "median_ESS_tail": _safe_nanmedian(ess_tail),
            "min_ESS_tail": _safe_nanmin(ess_tail),
        })
    return pd.DataFrame(rows).sort_values(["max_R_hat", "min_ESS_bulk"], ascending=[False, True], na_position="last")


def _top_k(summary: pd.DataFrame, column: str, ascending: bool, k: int = 20) -> pd.DataFrame:
    cols = ["variable", "family", "Mean", "StdDev", "R_hat", "ESS_bulk", "ESS_tail"]
    cols = [c for c in cols if c in summary.columns]
    return summary.sort_values(column, ascending=ascending, na_position="last")[cols].head(k)


def _hyperparam_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[summary["variable"].isin(HYPERPARAMS)].copy()


def _sampler_diagnostics(csv_files: list[Path]) -> pd.DataFrame:
    rows = []
    for chain_idx, csv_path in enumerate(csv_files, start=1):
        _print(f"    sampler diagnostics | chain {chain_idx} | {csv_path.name}")
        df = pd.read_csv(csv_path, comment="#", usecols=SAMPLER_COLS)
        divergent_count = int(df["divergent__"].sum())
        treedepth_max = int(df["treedepth__"].max())
        treedepth_hits = int((df["treedepth__"] == treedepth_max).sum())
        rows.append({
            "chain": chain_idx,
            "draws": int(len(df)),
            "divergences": divergent_count,
            "divergence_rate": divergent_count / max(len(df), 1),
            "max_treedepth_observed": treedepth_max,
            "treedepth_hits": treedepth_hits,
            "treedepth_hit_rate": treedepth_hits / max(len(df), 1),
            "mean_accept_stat": float(df["accept_stat__"].mean()),
            "mean_stepsize": float(df["stepsize__"].mean()),
            "mean_n_leapfrog": float(df["n_leapfrog__"].mean()),
            "max_n_leapfrog": int(df["n_leapfrog__"].max()),
            "ebfmi": _ebfmi(df["energy__"].to_numpy(dtype=float)),
        })
    return pd.DataFrame(rows)


def _load_truth_labels(eval_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    config = _read_json(eval_dir / "evaluation_config.json")
    bundle_path = ROOT / config["data_bundle"]
    bundle = _read_json(bundle_path)
    idxs = list(bundle.get("missing_ratings_indexes_in_test_instance", []))
    labels = np.asarray([bundle["missing_ratings"][idx]["value"] - 1 for idx in idxs], dtype=np.int64)
    c = int(config["config"]["C"])
    return np.asarray(idxs, dtype=np.int64), labels, c


def _chain_mean_probs(csv_path: Path, num_classes: int) -> np.ndarray:
    _print(f"    predictive chain load | {csv_path.name}")
    df = pd.read_csv(csv_path, comment="#")
    prob_cols = [c for c in df.columns if c.startswith(PROB_PREFIX)]
    pattern = re.compile(r"missing_rating_probs\.(\d+)\.(\d+)$")
    parsed: list[tuple[int, int, str]] = []
    max_missing = 0
    for col in prob_cols:
        m = pattern.match(col)
        if m is None:
            continue
        missing_idx = int(m.group(1))
        cls_idx = int(m.group(2))
        parsed.append((missing_idx, cls_idx, col))
        max_missing = max(max_missing, missing_idx)
    out = np.zeros((max_missing, num_classes), dtype=np.float64)
    for missing_idx, cls_idx, col in parsed:
        out[missing_idx - 1, cls_idx - 1] = float(df[col].mean())
    return out


def _predictive_metrics_from_probs(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    picked = probs[np.arange(labels.shape[0]), labels]
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    expected = probs @ classes
    truth = labels.astype(np.float64) + 1.0
    pred = np.argmax(probs, axis=1)
    return {
        "log_loss": float(np.mean(-np.log(picked))),
        "mbr_l2": float(np.mean((expected - truth) ** 2)),
        "accuracy": float(np.mean(pred == labels)),
    }


def _chain_predictive_metrics(spec: RunSpec, csv_files: list[Path]) -> pd.DataFrame:
    test_idxs, labels, c = _load_truth_labels(spec.eval_dir)
    rows = []
    for chain_idx, csv_path in enumerate(csv_files, start=1):
        all_missing_probs = _chain_mean_probs(csv_path, c)
        if np.max(test_idxs) >= all_missing_probs.shape[0]:
            raise ValueError(
                f"{spec.label}: chain {chain_idx} all-missing probs shape {all_missing_probs.shape} "
                f"does not cover max test index {int(np.max(test_idxs))}"
            )
        probs = all_missing_probs[test_idxs]
        metrics = _predictive_metrics_from_probs(probs, labels)
        rows.append({
            "chain": chain_idx,
            "n_missing": int(labels.shape[0]),
            **metrics,
        })
    return pd.DataFrame(rows)


def _report_lines_for_run(
    spec: RunSpec,
    param_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    ij_summary: pd.DataFrame,
    sampler_diag: pd.DataFrame,
    chain_pred: pd.DataFrame,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"## {spec.label}")
    lines.append("")
    lines.append(f"- Run dir: `{spec.run_dir}`")
    lines.append(f"- Eval dir: `{spec.eval_dir}`")
    lines.append(f"- Chains found: `{len(sampler_diag)}`")
    lines.append("")

    if len(sampler_diag) > 1:
        bad_rhat = int((param_summary["R_hat"] > 1.01).fillna(False).sum()) if "R_hat" in param_summary else 0
        bad_rhat_105 = int((param_summary["R_hat"] > 1.05).fillna(False).sum()) if "R_hat" in param_summary else 0
        lines.append("### Built-in Diagnostics")
        lines.append("")
        lines.append(f"- Parameters with `R_hat > 1.01`: `{bad_rhat}`")
        lines.append(f"- Parameters with `R_hat > 1.05`: `{bad_rhat_105}`")
    else:
        lines.append("### Built-in Diagnostics")
        lines.append("")
        lines.append("- Only one chain is present, so multi-chain `R_hat` and split-`ESS` are not meaningful here.")
    lines.append(f"- Total divergences across chains: `{int(sampler_diag['divergences'].sum())}`")
    lines.append(f"- Max treedepth hit rate across chains: `{sampler_diag['treedepth_hit_rate'].max():.3f}`")
    lines.append(f"- Min E-BFMI across chains: `{sampler_diag['ebfmi'].min():.4f}`")
    lines.append("")

    hp = _hyperparam_summary(param_summary)
    if not hp.empty:
        lines.append("### Sampled Hyperparameters")
        lines.append("")
        hp_cols = [c for c in ["variable", "Mean", "StdDev", "R_hat", "ESS_bulk", "ESS_tail"] if c in hp.columns]
        lines.append(hp[hp_cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.append("### Family-Level Summary")
    lines.append("")
    fam_cols = [
        "family", "n_variables", "mean_R_hat", "median_R_hat", "max_R_hat",
        "frac_R_hat_gt_1.01", "frac_R_hat_gt_1.05",
        "mean_ESS_bulk", "median_ESS_bulk", "min_ESS_bulk",
        "mean_ESS_tail", "median_ESS_tail", "min_ESS_tail",
    ]
    fam_cols = [c for c in fam_cols if c in family_summary.columns]
    lines.append(family_summary[fam_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    if not ij_summary.empty:
        lines.append("### i*j Group Summary")
        lines.append("")
        ij_cols = [
            "ij_group", "family", "n_variables",
            "mean_R_hat", "median_R_hat", "max_R_hat",
            "mean_ESS_bulk", "median_ESS_bulk", "min_ESS_bulk",
            "mean_ESS_tail", "median_ESS_tail", "min_ESS_tail",
        ]
        ij_cols = [c for c in ij_cols if c in ij_summary.columns]
        lines.append(ij_summary.head(20)[ij_cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.append("### Worst 20 by R-hat")
    lines.append("")
    lines.append(_top_k(param_summary, "R_hat", ascending=False).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("### Worst 20 by ESS_bulk")
    lines.append("")
    lines.append(_top_k(param_summary, "ESS_bulk", ascending=True).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("### Worst 20 by ESS_tail")
    lines.append("")
    lines.append(_top_k(param_summary, "ESS_tail", ascending=True).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("### Sampler Diagnostics by Chain")
    lines.append("")
    lines.append(sampler_diag.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("### Posterior Predictive Stability by Chain")
    lines.append("")
    lines.append(chain_pred.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    if len(chain_pred) > 1:
        lines.append("- Predictive metric spread across chains:")
        lines.append(f"  - log_loss range: `{chain_pred['log_loss'].min():.6f}` to `{chain_pred['log_loss'].max():.6f}`")
        lines.append(f"  - mbr_l2 range: `{chain_pred['mbr_l2'].min():.6f}` to `{chain_pred['mbr_l2'].max():.6f}`")
        lines.append(f"  - accuracy range: `{chain_pred['accuracy'].min():.6f}` to `{chain_pred['accuracy'].max():.6f}`")
        lines.append("")
    return lines


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def analyze_run(spec: RunSpec, out_dir: Path) -> tuple[list[str], dict[str, Path]]:
    _print("")
    _print("============================================================")
    _print(f"Analyzing {spec.label}")
    _print("============================================================")
    csv_files = _load_csv_files(spec.run_dir)
    if not csv_files:
        raise FileNotFoundError(f"No Stan chain CSVs found in {spec.run_dir}")
    _print(f"  found {len(csv_files)} chain csv files")
    for path in csv_files:
        _print(f"    - {path.name}")

    _print("  computing CmdStan summary...")
    summary = _load_fit_summary(csv_files)
    param_summary = _prepare_param_summary(summary)
    filtered_summary = _filtered_param_summary(param_summary)

    _print("  aggregating by parameter family...")
    family_summary = _family_summary(filtered_summary)
    _print("  aggregating i*j grouped families...")
    ij_summary = _ij_summary(filtered_summary)

    _print("  collecting sampler diagnostics...")
    sampler_diag = _sampler_diagnostics(csv_files)

    _print("  computing chain-level predictive metrics...")
    chain_pred = _chain_predictive_metrics(spec, csv_files)

    run_out = out_dir / spec.slug
    run_out.mkdir(parents=True, exist_ok=True)
    files = {
        "param_summary": run_out / "parameter_summary.csv",
        "filtered_param_summary": run_out / "filtered_parameter_summary.csv",
        "family_summary": run_out / "family_summary.csv",
        "ij_summary": run_out / "ij_group_summary.csv",
        "sampler_diagnostics": run_out / "sampler_diagnostics.csv",
        "chain_predictive": run_out / "chain_predictive_metrics.csv",
        "top20_rhat": run_out / "top20_rhat.csv",
        "top20_ess_bulk": run_out / "top20_ess_bulk.csv",
        "top20_ess_tail": run_out / "top20_ess_tail.csv",
        "hyperparameters": run_out / "hyperparameter_summary.csv",
    }
    _save_dataframe(param_summary, files["param_summary"])
    _save_dataframe(filtered_summary, files["filtered_param_summary"])
    _save_dataframe(family_summary, files["family_summary"])
    _save_dataframe(ij_summary, files["ij_summary"])
    _save_dataframe(sampler_diag, files["sampler_diagnostics"])
    _save_dataframe(chain_pred, files["chain_predictive"])
    _save_dataframe(_top_k(filtered_summary, "R_hat", ascending=False), files["top20_rhat"])
    _save_dataframe(_top_k(filtered_summary, "ESS_bulk", ascending=True), files["top20_ess_bulk"])
    _save_dataframe(_top_k(filtered_summary, "ESS_tail", ascending=True), files["top20_ess_tail"])
    _save_dataframe(_hyperparam_summary(filtered_summary), files["hyperparameters"])

    report_lines = _report_lines_for_run(spec, filtered_summary, family_summary, ij_summary, sampler_diag, chain_pred)
    return report_lines, files


def main() -> None:
    report_dir = OUT_ROOT / "REPORTS"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "stan_diagnostics_report.md"

    all_lines: list[str] = [
        "# DOMAIN3-SIGMOID Stan Diagnostics Report",
        "",
        f"Generated from `{ROOT}`",
        "",
    ]
    all_files: dict[str, dict[str, Path]] = {}

    for spec in RUN_SPECS:
        lines, files = analyze_run(spec, report_dir)
        all_lines.extend(lines)
        all_files[spec.slug] = files

    report_path.write_text("\n".join(all_lines) + "\n")

    _print("")
    _print("============================================================")
    _print("Saved diagnostics report")
    _print("============================================================")
    _print(f"  report : {report_path}")
    for spec in RUN_SPECS:
        _print(f"  csv dir: {report_dir / spec.slug}")

    _print("")
    _print("============================================================")
    _print("Report preview")
    _print("============================================================")
    preview = "\n".join(all_lines[:80])
    _print(preview)


if __name__ == "__main__":
    main()
