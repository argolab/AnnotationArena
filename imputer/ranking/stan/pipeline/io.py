import json
import time
from pathlib import Path
from typing import Any


def new_run_dir(root: Path | str = "runs", run_name: str = None) -> Path:
    """
    Create a new run directory under the given root, with a timestamp.
    If run_name is provided, use it as a prefix; otherwise, use 'run_'.
    Returns the Path to the created directory.

    Raises:
        FileNotFoundError: if the directory could not be created.
        TypeError: if root is not a valid path or string.
    """
    # Validate root type
    if not isinstance(root, (str, Path)):
        raise TypeError(f"Expected root to be str or Path, got {type(root)}")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_dir = root_path / f"run_{ts}"
    else:
        # Validate run_name
        if not isinstance(run_name, str):
            raise TypeError(f"Expected run_name to be str or None, got {type(run_name)}")
        run_dir = root_path / f"{run_name}_{ts}"

    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Extremely unlikely, but possible if called twice in the same second with same run_name
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            f"Try again or use a different run_name."
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create run directory {run_dir}: {e}"
        )

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory was not created as expected: {run_dir}. "
            f"Check permissions and disk space."
        )
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


