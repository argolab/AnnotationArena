from __future__ import annotations

"""
Plot training/test MSE curves for a single synthetic run.

Reads training_curves.json from a run directory and writes a PNG.

Usage:
  python -m imputer.entity_mf.synthetic.plot_curves \
    --run-dir OUTPUT/SYNTHETIC/tree/depth_vs_layers/d3_L3 \
    --plot-path OUTPUT/SYNTHETIC/tree/plots/depth_vs_layers_d3_L3.png
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser("Plot train/test MSE vs epoch for a synthetic run.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--plot-path", type=str, required=True)
    parser.add_argument("--title", type=str, default=None)

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    curves_path = run_dir / "training_curves.json"
    if not curves_path.exists():
        raise FileNotFoundError(f"training_curves.json not found in {run_dir}")

    data = json.loads(curves_path.read_text())
    train: List[float] = data.get("train_loss", [])
    test: List[float] = data.get("test_loss", [])
    epochs = list(range(1, max(len(train), len(test)) + 1))

    plt.figure(figsize=(4, 3))
    if train:
        plt.plot(epochs[: len(train)], train, label="train MSE")
    if test:
        plt.plot(epochs[: len(test)], test, label="test MSE")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    if args.title:
        plt.title(args.title)
    plt.legend()
    plt.tight_layout()

    plot_path = Path(args.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    main()

