import json
import time
from pathlib import Path
from typing import Any


def new_run_dir(root: Path | str = "runs") -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root_path / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_configs(run_dir: Path, **configs: Any) -> None:
    cfg_dict = {}
    for k, v in configs.items():
        try:
            cfg_dict[k] = v.__dict__
        except AttributeError:
            cfg_dict[k] = v
    save_json(cfg_dict, run_dir / "configs.json")


def save_bundle(run_dir: Path, bundle_dict: dict) -> None:
    # Expect numpy converted to lists by caller
    save_json(bundle_dict, run_dir / "data_bundle.json")


def save_predictives(run_dir: Path, predictives: dict) -> None:
    save_json(predictives, run_dir / "predictives.json")


def save_metrics(run_dir: Path, metrics: dict) -> None:
    save_json(metrics, run_dir / "metrics.json")


def save_fit_csvs(run_dir: Path, fit) -> None:
    csv_dir = run_dir / "stan_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(fit.runset.csv_files):
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Stan CSV not found: {src}")
        dest = csv_dir / f"chain_{i+1}.csv"
        try:
            src.replace(dest)
        except Exception as e:
            raise RuntimeError(f"Failed to move Stan CSV from {src} to {dest}: {e}")


