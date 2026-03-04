#!usr/bin/env python3
"""
Cross Stan-type experiment: for each stan_type generate one dataset (with its own
parameters), then fit every domain model on every dataset (N×N Stan runs), run imputer
on each dataset, then produce Stan N×N performance grid and imputer vs multiple
Stan baseline plots.

Edit the CONFIGURATION section below to:
  - Set which stan types to use (STAN_TYPES).
  - Set shared dimensions and noise (BASE_*).
  - Set per-stan_type parameters (DISCRETE_*, TENSOR_*, etc.) for data generation.

Usage:
    PYTHONPATH=. python scripts/cross_stan_type_experiment.py
    PYTHONPATH=. python scripts/cross_stan_type_experiment.py --quick
    PYTHONPATH=. python scripts/cross_stan_type_experiment.py --output-base OUTPUT/cross_stan_experiment --quick
"""

import argparse
import subprocess
import sys
from pathlib import Path

# =============================================================================
# CONFIGURATION — change these to run different experiments
# =============================================================================

# ---------- 1. Stan types (order = row/column order in grid) ----------
STAN_TYPES = [
    "normal-noise-dot-product",
    "factored-dot-product",
    "discrete",
    "tensor",
]

# ---------- 2. Shared dimensions (all data generation) ----------
BASE_K_TRAIN = 10
BASE_K_TEST = 10
BASE_I = 5
BASE_J = 12
BASE_C = 5

# ---------- 3. Shared noise/temperature (used by embedding types and discrete) ----------
BASE_D = 8
BASE_SIGMA_ANNOTATOR = 0.5
BASE_SIGMA_MEASUREMENT = 0.1
BASE_KAPPA = 10
BASE_TEMPERATURE = 0.5

# ---------- 4. Per–stan_type parameters (for data generation only) ----------
# discrete: M item prototypes, S annotator styles (required by discrete_type_data_generation.stan)
DISCRETE_M = 6
DISCRETE_S = 3

# tensor: CP factor decay (required by tensor_data_generation.stan)
TENSOR_FACTOR_DECAY = 0.9

# factored-dot-product: set True to derive rating thresholds from annotator embedding
FACTORED_DERIVE_THRESHOLDS_FROM_ANNOTATOR = False

# ---------- 5. Stan MCMC (domain model inference) ----------
STAN_CHAINS = 1
STAN_WARMUP = 200
STAN_SAMPLING = 500

# ---------- 6. Imputer training (match run_single_kp10_random_as_key_fresh_lightning.sh) ----------
IMPUTER_EPOCHS = 200
IMPUTER_LR = 2e-4
IMPUTER_MASKING_RATE = 0.15
IMPUTER_MASKED_LOSS_WEIGHT = 15.0
IMPUTER_OBSERVED_LOSS_WEIGHT = 1.0
IMPUTER_MASK_AUGMENTATIONS = 5
IMPUTER_EMBEDDING_DIM = 72
IMPUTER_LAYERS = 4
IMPUTER_HEADS = 4
IMPUTER_NUM_FFN_LAYERS = 2
IMPUTER_WEIGHT_DECAY = 0.01
IMPUTER_DROPOUT = 0.1
IMPUTER_BATCH_SIZE = 1
IMPUTER_GRADIENT_CLIP_VAL = 0.0
IMPUTER_USE_COSINE_SCHEDULE = True
IMPUTER_WARMUP_STEPS = 100

# =============================================================================
# End of configuration
# =============================================================================

# CSV pattern for each stan_type (CmdStanPy names CSVs after the .stan base name)
STAN_CSV_PATTERNS = {
    "normal-noise-dot-product": "domain_model-*.csv",
    "factored-dot-product": "domain_model-*.csv",
    "discrete": "discrete_type_domain_model-*.csv",
    "tensor": "tensor_domain_model-*.csv",
}


def run(cmd: list, cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd or Path.cwd())
    if r.returncode != 0:
        sys.exit(r.returncode)


def run_allow_fail(cmd: list, cwd: Path | None = None) -> bool:
    """Run command; return True if success, False otherwise (no sys.exit)."""
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd or Path.cwd())
    if r.returncode != 0:
        print(f"  [FAILED] exit code {r.returncode}")
        return False
    return True


def _generate_data_args(st: str) -> list:
    """Build argv for generate_data.py for the given stan_type (shared + type-specific)."""
    base = [
        sys.executable, "stan/scripts/generate_data.py",
        "--K-train", str(BASE_K_TRAIN),
        "--K-test", str(BASE_K_TEST),
        "--I", str(BASE_I),
        "--J", str(BASE_J),
        "--C", str(BASE_C),
        "--sigma-measurement", str(BASE_SIGMA_MEASUREMENT),
        "--kappa", str(BASE_KAPPA),
        "--temperature", str(BASE_TEMPERATURE),
        "--stan-type", st,
    ]
    if st == "discrete":
        base += ["--stan-arg", f"M={DISCRETE_M}", "--stan-arg", f"S={DISCRETE_S}"]
    elif st in ("normal-noise-dot-product", "factored-dot-product"):
        base += [
            "--D", str(BASE_D),
            "--sigma-annotator", str(BASE_SIGMA_ANNOTATOR),
        ]
        if st == "factored-dot-product" and FACTORED_DERIVE_THRESHOLDS_FROM_ANNOTATOR:
            base += ["--derive-thresholds-from-annotator"]
    elif st == "tensor":
        base += [
            "--D", str(BASE_D),
            "--sigma-annotator", str(BASE_SIGMA_ANNOTATOR),
            "--factor-decay", str(TENSOR_FACTOR_DECAY),
        ]
    return base


def main():
    parser = argparse.ArgumentParser(description="Cross Stan-type experiment (N×N grid)")
    parser.add_argument("--output-base", type=str, default="OUTPUT/cross_stan_experiment",
                        help="Base directory for all outputs")
    parser.add_argument("--quick", action="store_true", help="Use minimal chains/epochs for a quick test")
    parser.add_argument("--stan-chains", type=int, default=None, help="Override Stan chains (default: from config)")
    parser.add_argument("--stan-warmup", type=int, default=None, help="Override Stan warmup iterations")
    parser.add_argument("--stan-sampling", type=int, default=None, help="Override Stan sampling iterations")
    parser.add_argument("--imputer-epochs", type=int, default=None, help="Override imputer epochs (e.g. 200 for full run)")
    parser.add_argument("--only-marformer-grid", action="store_true",
                        help="Skip data/Stan/imputer runs and only build Stan+Marformer grid from existing metrics")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    import os
    os.chdir(project_root)

    root = Path(args.output_base)
    global STAN_CHAINS, STAN_WARMUP, STAN_SAMPLING, IMPUTER_EPOCHS
    if args.quick:
        STAN_CHAINS, STAN_WARMUP, STAN_SAMPLING = 1, 10, 20
        IMPUTER_EPOCHS = 5
    if args.stan_chains is not None:
        STAN_CHAINS = args.stan_chains
    if args.stan_warmup is not None:
        STAN_WARMUP = args.stan_warmup
    if args.stan_sampling is not None:
        STAN_SAMPLING = args.stan_sampling
    if args.imputer_epochs is not None:
        IMPUTER_EPOCHS = args.imputer_epochs

    root.mkdir(parents=True, exist_ok=True)
    run_id = "run"
    exp_dir = root / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Fast path: only build Stan+Marformer grid from existing metrics
    if args.only_marformer_grid:
        print("\n=== Only Marformer grid: using existing Stan + imputer metrics ===")
        n = len(STAN_TYPES)
        stan_eval_dirs = {
            (data_st, model_st): Path("OUTPUT/domain_model/eval") / f"cross_data_{data_st}_model_{model_st}_eval"
            for data_st in STAN_TYPES for model_st in STAN_TYPES
        }
        grid_paths = [
            str(stan_eval_dirs[(data_st, model_st)] / "predictive_metrics.json")
            for data_st in STAN_TYPES
            for model_st in STAN_TYPES
        ]
        imputer_dirs = {
            data_st: Path("OUTPUT/IMPUTER") / f"cross_data_{data_st}_imputer"
            for data_st in STAN_TYPES
        }
        imputer_metrics_paths = [str(imputer_dirs[data_st] / "train_metrics.json") for data_st in STAN_TYPES]
        run(
            [
                sys.executable, "scripts/visualize_cross_stan_grid.py",
                "--output", str(exp_dir / f"stan_plus_marformer_{n}x{n+1}_grid.png"),
                "--stan-types", ",".join(STAN_TYPES),
                "--metrics-paths", *grid_paths,
                "--imputer-metrics-paths", *imputer_metrics_paths,
            ]
        )
        print("\n=== Done (only Marformer grid) ===")
        print(f"  Stan+Marformer grid: {exp_dir / f'stan_plus_marformer_{n}x{n+1}_grid.png'}")
        return

    data_dirs = {}
    stan_run_dirs = {}
    stan_eval_dirs = {}
    imputer_dirs = {}

    # ---------- 1. Generate one dataset per stan_type ----------
    print("\n=== 1. Generate datasets ===")
    for st in STAN_TYPES:
        run_name = f"cross_data_{st}"
        data_dir = Path("OUTPUT/generated_data") / run_name
        bundle_path = data_dir / "data_bundle.json"
        if bundle_path.exists():
            print(f"  Reusing existing data for {st}: {data_dir}")
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            cmd = _generate_data_args(st) + [
                "--run-name", run_name,
                "--output-dir", "OUTPUT/generated_data",
                "--overwrite-existing-data",
            ]
            run(cmd)
        data_dirs[st] = data_dir
        print(f"  Data for {st}: {data_dir}")

    # ---------- 2. For each (data_st, model_st): run Stan inference (fallback on failure) ----------
    print("\n=== 2. Stan inference (N×N) ===")
    failed_stan_runs: set[tuple[str, str]] = set()
    for data_st in STAN_TYPES:
        bundle = data_dirs[data_st] / "data_bundle.json"
        for model_st in STAN_TYPES:
            run_name = f"cross_data_{data_st}_model_{model_st}"
            inference_cmd = [
                sys.executable, "stan/scripts/run_inference.py",
                "--data-bundle", str(bundle),
                "--stan-type", model_st,
                "--use-train-only",
                "--chains", str(STAN_CHAINS),
                "--iter-warmup", str(STAN_WARMUP),
                "--iter-sampling", str(STAN_SAMPLING),
                "--run-name", run_name,
                "--output-dir", "OUTPUT/domain_model/runs",
                "--overwrite-existing-data",
            ]
            # Single source of truth: pass all parameters required by the inference model (cross-type)
            if model_st == "discrete":
                inference_cmd.extend(["--stan-arg", f"M={DISCRETE_M}", "--stan-arg", f"S={DISCRETE_S}"])
            elif model_st == "tensor":
                inference_cmd.extend([
                    "--stan-arg", f"D={BASE_D}",
                    "--stan-arg", f"d_annotator={BASE_D}",
                    "--stan-arg", f"factor_decay={TENSOR_FACTOR_DECAY}",
                ])
            elif model_st == "normal-noise-dot-product":
                inference_cmd.extend([
                    "--stan-arg", f"D={BASE_D}",
                    "--stan-arg", f"d_annotator={BASE_D}",
                    "--stan-arg", "use_factored_annotator=0",
                    "--stan-arg", "derive_thresholds_from_annotator=0",
                ])
            elif model_st == "factored-dot-product":
                inference_cmd.extend([
                    "--stan-arg", f"D={BASE_D}",
                    "--stan-arg", f"d_annotator={BASE_D}",
                    "--stan-arg", "use_factored_annotator=1",
                    "--stan-arg", f"derive_thresholds_from_annotator={1 if FACTORED_DERIVE_THRESHOLDS_FROM_ANNOTATOR else 0}",
                ])
            run_dir = Path("OUTPUT/domain_model/runs") / run_name
            # Reuse existing Stan run if CSVs are already present
            csv_pattern = STAN_CSV_PATTERNS.get(model_st, "*-*.csv")
            existing_csvs = list(run_dir.glob(csv_pattern))
            if run_dir.exists() and existing_csvs:
                print(f"  Reusing existing Stan run for data={data_st}, model={model_st}: {run_dir}")
                ok = True
            else:
                ok = run_allow_fail(inference_cmd)
            stan_run_dirs[(data_st, model_st)] = run_dir
            if not ok:
                failed_stan_runs.add((data_st, model_st))

    # ---------- 3. Evaluate each Stan run (skip failed) ----------
    print("\n=== 3. Evaluate Stan predictions ===")
    for data_st in STAN_TYPES:
        for model_st in STAN_TYPES:
            run_name = f"cross_data_{data_st}_model_{model_st}_eval"
            eval_dir = Path("OUTPUT/domain_model/eval") / run_name
            stan_eval_dirs[(data_st, model_st)] = eval_dir
            if (data_st, model_st) in failed_stan_runs:
                print(f"  Skipping evaluation for failed run data={data_st} model={model_st}")
                continue
            metrics_path = eval_dir / "predictive_metrics.json"
            if metrics_path.exists():
                print(f"  Reusing existing Stan evaluation for data={data_st}, model={model_st}: {metrics_path}")
                continue
            mcmc_dir = stan_run_dirs[(data_st, model_st)]
            csv_pattern = STAN_CSV_PATTERNS.get(model_st, "*-*.csv")
            run(
                [
                    sys.executable, "stan/scripts/evaluate_predictions.py",
                    "--mcmc-dir", str(mcmc_dir),
                    "--data-bundle", str(data_dirs[data_st] / "data_bundle.json"),
                    "--output-dir", "OUTPUT/domain_model/eval",
                    "--run-name", run_name,
                    "--csv-pattern", csv_pattern,
                    "--use-train-only",
                    "--overwrite-existing-data",
                ]
            )

    # ---------- 4. Run imputer on each dataset ----------
    print("\n=== 4. Run imputer on each dataset ===")
    for data_st in STAN_TYPES:
        run_name = f"cross_data_{data_st}_imputer"
        run(
            [
                sys.executable, "imputer/run_imputer.py",
                "--data-dir", str(data_dirs[data_st]),
                "--run-name", run_name,
                "--epochs", str(IMPUTER_EPOCHS),
                "--lr", str(IMPUTER_LR),
                "--masking-rate", str(IMPUTER_MASKING_RATE),
                "--masked-loss-weight", str(IMPUTER_MASKED_LOSS_WEIGHT),
                "--observed-loss-weight", str(IMPUTER_OBSERVED_LOSS_WEIGHT),
                "--mask-augmentations", str(IMPUTER_MASK_AUGMENTATIONS),
                "--embedding-dim", str(IMPUTER_EMBEDDING_DIM),
                "--encoder-layers", str(IMPUTER_LAYERS),
                "--attention-heads", str(IMPUTER_HEADS),
                "--num_ffn_layers", str(IMPUTER_NUM_FFN_LAYERS),
                "--weight-decay", str(IMPUTER_WEIGHT_DECAY),
                "--dropout", str(IMPUTER_DROPOUT),
                "--no-final-norm",
                "--normalize-parameter",
                "--device", "cpu",
                "--batch-size", str(IMPUTER_BATCH_SIZE),
                "--gradient-clip-val", str(IMPUTER_GRADIENT_CLIP_VAL),
                "--use-cosine-schedule",
                "--warmup-steps", str(IMPUTER_WARMUP_STEPS),
                "--overwrite-existing-data",
            ]
        )
        imputer_dirs[data_st] = Path("OUTPUT/IMPUTER") / run_name

    # ---------- 5. Visualize Stan N×N grid (logloss, accuracy) ----------
    print("\n=== 5. Stan N×N performance grid ===")
    grid_paths = [
        str(stan_eval_dirs[(data_st, model_st)] / "predictive_metrics.json")
        for data_st in STAN_TYPES
        for model_st in STAN_TYPES
    ]
    n = len(STAN_TYPES)
    run(
        [
            sys.executable, "scripts/visualize_cross_stan_grid.py",
            "--output", str(exp_dir / f"stan_{n}x{n}_grid.png"),
            "--stan-types", ",".join(STAN_TYPES),
            "--metrics-paths", *grid_paths,
        ]
    )

    # ---------- 5b. Stan N×(N+1) grid including Marformer (train-instance metrics) ----------
    print("\n=== 5b. Stan + Marformer performance grid ===")
    # Use train_metrics.json so Marformer column reflects train-instance missing metrics,
    # matching Stan which is now evaluated with --use-train-only.
    imputer_metrics_paths = [str(imputer_dirs[data_st] / "train_metrics.json") for data_st in STAN_TYPES]
    run(
        [
            sys.executable, "scripts/visualize_cross_stan_grid.py",
            "--output", str(exp_dir / f"stan_plus_marformer_{n}x{n+1}_grid.png"),
            "--stan-types", ",".join(STAN_TYPES),
            "--metrics-paths", *grid_paths,
            "--imputer-metrics-paths", *imputer_metrics_paths,
        ]
    )

    # ---------- 6. Imputer vs multiple Stan baselines (per dataset) ----------
    print("\n=== 6. Imputer plots with multiple Stan baselines ===")
    for data_st in STAN_TYPES:
        stan_paths = [stan_eval_dirs[(data_st, model_st)] / "predictive_metrics.json" for model_st in STAN_TYPES]
        stan_labels = [f"Stan ({model_st})" for model_st in STAN_TYPES]
        run(
            [
                sys.executable, "utils/visualize.py",
                "--run-dir", str(imputer_dirs[data_st]),
                "--stan-metrics", *[str(p) for p in stan_paths],
                "--stan-labels", *stan_labels,
            ]
        )

    print("\n=== Done ===")
    print(f"  Experiment dir: {exp_dir}")
    print(f"  Stan grid: {exp_dir / f'stan_{n}x{n}_grid.png'}")
    for data_st in STAN_TYPES:
        print(f"  Imputer plots ({data_st}): {imputer_dirs[data_st] / 'plots'}")


if __name__ == "__main__":
    main()
