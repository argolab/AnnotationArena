import json
import numpy as np
import time
from pathlib import Path
from typing import Any


def new_run_dir(root: Path | str = "runs", run_name: str = None) -> Path:
    """
    Create a new run directory under the given root.
    If run_name is provided, use it exactly as the folder name; otherwise, use 'run_' + timestamp.
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

    if run_name is None:
        # Auto-generate name with timestamp
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = root_path / f"run_{ts}"
    else:
        # Use exact custom name provided
        if not isinstance(run_name, str):
            raise TypeError(f"Expected run_name to be str or None, got {type(run_name)}")
        run_dir = root_path / run_name

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
    # Ensure double precision by converting numpy floats to Python float (which are double precision)
    def _default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def save_configs(run_dir: Path, **configs: Any) -> None:
    cfg_dict = {}
    for k, v in configs.items():
        try:
            cfg_dict[k] = v.__dict__
        except AttributeError:
            cfg_dict[k] = v
    save_json(cfg_dict, run_dir / "configs.json")


def save_bundle(run_dir: Path, bundle_dict: dict) -> None:
    # Expect numpy converted to lists by caller; sanitize any simplex arrays for exact sum-to-one
    # Specifically fix rating_probs rows (each row is a simplex over classes)
    if "rating_probs" in bundle_dict and isinstance(bundle_dict["rating_probs"], list):
        fixed_rows = []
        for row in bundle_dict["rating_probs"]:
            v = np.asarray(row, dtype=np.float64)
            # Project to simplex with exact sum 1, guarding against tiny negatives
            v = np.maximum(v, 0.0)
            s = v.sum()
            if s <= 0:
                v[:] = 1.0 / v.size
            else:
                v /= s
                v = np.maximum(v, 1e-12)
                v /= v.sum()
                k = int(np.argmax(v))
                v[k] += (1.0 - v.sum(dtype=np.float64))
            fixed_rows.append(v.tolist())
        bundle_dict = {**bundle_dict, "rating_probs": fixed_rows}

    save_json(bundle_dict, run_dir / "data_bundle.json")


def save_predictives(run_dir: Path, predictives: dict, filename: str = "predictives.json") -> None:
    """Save predictives dict to run_dir/filename. Default filename for backward compatibility."""
    save_json(predictives, run_dir / filename)


def save_test_metrics(run_dir: Path, metrics: dict) -> None:
    save_json(metrics, run_dir / "test_metrics.json")


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


