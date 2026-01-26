#!/usr/bin/env python3
"""
Visualization utilities for imputer training runs.

Usage:
    python utils/visualize.py --run-dir OUTPUT/IMPUTER/run_20251008_120647 [--stan-metrics OUTPUT/domain_model/eval/run_20251007_221514/predictive_metrics.json]
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict


def load_run_data(run_dir: Path) -> Dict[str, Any]:
    """Load all JSON data from a run directory."""
    data = {}

    # Current Lightning naming convention
    training_loss_history_path = run_dir / "training_loss_history.json"
    train_instance_path = run_dir / "train_instance_training_history.json"
    test_instance_path = run_dir / "test_instance_training_history.json"

    if training_loss_history_path.exists():
        with open(training_loss_history_path, "r") as f:
            data["training_loss_history"] = json.load(f)
    if train_instance_path.exists():
        with open(train_instance_path, "r") as f:
            data["train_instance_history"] = json.load(f)
    if test_instance_path.exists():
        with open(test_instance_path, "r") as f:
            data["test_instance_history"] = json.load(f)

    # Load predictions
    predictives_path = run_dir / "predictives.json"
    if predictives_path.exists():
        with open(predictives_path, 'r') as f:
            data['predictives'] = json.load(f)

    # Load final metrics
    test_metrics_path = run_dir / "test_metrics.json"
    if test_metrics_path.exists():
        with open(test_metrics_path, 'r') as f:
            data['test_metrics'] = json.load(f)

    # Load config
    config_path = run_dir / "train_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            data['config'] = json.load(f)

    return data


def load_stan_metrics(stan_path: Optional[Path]) -> Optional[Dict[str, float]]:
    """Load Stan baseline metrics if path provided."""
    if stan_path is None or not stan_path.exists():
        return None

    with open(stan_path, 'r') as f:
        return json.load(f)


def plot_learning_curves_train(train_history: List[Dict], stan_metrics: Optional[Dict], output_dir: Path) -> None:
    """Plot learning curves for train instance (observed, missing, overall). Strict naming: never use total_loss for Missing/Observed."""
    assert train_history, "train_history must not be empty"
    assert "observed_metrics" in train_history[0], "train_history must contain observed_metrics (from EvaluationCallback)"
    assert "missing_metrics" in train_history[0], "train_history must contain missing_metrics (from EvaluationCallback)"
    
    epochs = [entry["epoch"] for entry in train_history]
    
    # Overall metrics - rating_loss should always be present (from EvaluationCallback)
    overall_loss = [entry["rating_loss"] for entry in train_history]
    overall_acc = [entry["rating_accuracy"] for entry in train_history]
    overall_rmse = [entry["rating_rmse"] for entry in train_history]

    # Observed and missing metrics are always present (from EvaluationCallback)
    observed_loss = [entry["observed_metrics"]["rating_loss"] for entry in train_history]
    observed_acc = [entry["observed_metrics"]["rating_accuracy"] for entry in train_history]
    observed_rmse = [entry["observed_metrics"]["rating_rmse"] for entry in train_history]

    missing_loss = [entry["missing_metrics"]["rating_loss"] for entry in train_history]
    missing_acc = [entry["missing_metrics"]["rating_accuracy"] for entry in train_history]
    missing_rmse = [entry["missing_metrics"]["rating_rmse"] for entry in train_history]

    # Plot 1: Rating Loss (Train)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, overall_loss, label="Overall", linewidth=2, color="black")
    ax.plot(epochs, observed_loss, label="Observed", linewidth=1.5, color="green", linestyle="--")
    ax.plot(epochs, missing_loss, label="Missing", linewidth=1.5, color="red", linestyle="--")

    if stan_metrics and 'rating_missing_log_likelihood' in stan_metrics:
        # Stan reports negative log likelihood, convert to loss
        stan_loss = -stan_metrics['rating_missing_log_likelihood']
        ax.axhline(y=stan_loss, color='blue', linestyle=':', linewidth=2, label=f'Stan Baseline (Missing): {stan_loss:.3f}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Rating Loss (Cross-Entropy)', fontsize=12)
    ax.set_title('Training Set: Rating Loss over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'train_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Rating Accuracy (Train)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, overall_acc, label="Overall", linewidth=2, color="black")
    ax.plot(epochs, observed_acc, label="Observed", linewidth=1.5, color="green", linestyle="--")
    ax.plot(epochs, missing_acc, label="Missing", linewidth=1.5, color="red", linestyle="--")

    if stan_metrics and 'rating_missing_accuracy' in stan_metrics:
        stan_acc = stan_metrics['rating_missing_accuracy']
        ax.axhline(y=stan_acc, color='blue', linestyle=':', linewidth=2, label=f'Stan Baseline (Missing): {stan_acc:.3f}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Rating Accuracy', fontsize=12)
    ax.set_title('Training Set: Rating Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_dir / 'train_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Rating RMSE (Train)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, overall_rmse, label="Overall", linewidth=2, color="black")
    ax.plot(epochs, observed_rmse, label="Observed", linewidth=1.5, color="green", linestyle="--")
    ax.plot(epochs, missing_rmse, label="Missing", linewidth=1.5, color="red", linestyle="--")

    if stan_metrics and 'rating_missing_mae' in stan_metrics:
        stan_mae = stan_metrics['rating_missing_mae']
        ax.axhline(y=stan_mae, color='blue', linestyle=':', linewidth=2, label=f'Stan MAE (Missing): {stan_mae:.3f}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Rating RMSE', fontsize=12)
    ax.set_title('Training Set: Rating RMSE over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'train_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Combined Loss + Accuracy (Train)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(epochs, overall_loss, label="Overall", linewidth=2, color="black")
    ax1.plot(epochs, observed_loss, label="Observed", linewidth=1.5, color="green", linestyle="--")
    ax1.plot(epochs, missing_loss, label="Missing", linewidth=1.5, color="red", linestyle="--")
    if stan_metrics and "rating_missing_log_likelihood" in stan_metrics:
        stan_loss = -stan_metrics["rating_missing_log_likelihood"]
        ax1.axhline(y=stan_loss, color="blue", linestyle=":", linewidth=2, label="Stan (Missing)")
    ax1.set_ylabel("Rating Loss", fontsize=12)
    ax1.set_title("Training Set: Loss and Accuracy", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, overall_acc, label="Overall", linewidth=2, color="black")
    ax2.plot(epochs, observed_acc, label="Observed", linewidth=1.5, color="green", linestyle="--")
    ax2.plot(epochs, missing_acc, label="Missing", linewidth=1.5, color="red", linestyle="--")
    if stan_metrics and 'rating_missing_accuracy' in stan_metrics:
        stan_acc = stan_metrics['rating_missing_accuracy']
        ax2.axhline(y=stan_acc, color='blue', linestyle=':', linewidth=2, label=f'Stan (Missing)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Rating Accuracy', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'train_loss_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curves_test(test_history: List[Dict], stan_metrics: Optional[Dict], output_dir: Path) -> None:
    """Plot learning curves for test instance (observed, missing, overall). Strict naming: never use total_loss for Missing/Observed."""
    assert test_history, "test_history must not be empty"
    assert "observed_metrics" in test_history[0], "test_history must contain observed_metrics (from EvaluationCallback)"
    assert "missing_metrics" in test_history[0], "test_history must contain missing_metrics (from EvaluationCallback)"
    
    epochs = [entry["epoch"] for entry in test_history]

    overall_loss = [entry["rating_loss"] for entry in test_history]
    overall_acc = [entry["rating_accuracy"] for entry in test_history]
    overall_rmse = [entry["rating_rmse"] for entry in test_history]
    
    # Observed and missing metrics are always present (from EvaluationCallback)
    observed_loss = [entry["observed_metrics"]["rating_loss"] for entry in test_history]
    observed_acc = [entry["observed_metrics"]["rating_accuracy"] for entry in test_history]
    observed_rmse = [entry["observed_metrics"]["rating_rmse"] for entry in test_history]
    missing_loss = [entry["missing_metrics"]["rating_loss"] for entry in test_history]
    missing_acc = [entry["missing_metrics"]["rating_accuracy"] for entry in test_history]
    missing_rmse = [entry["missing_metrics"]["rating_rmse"] for entry in test_history]

    # Plot 1: Rating Loss (Test)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, overall_loss, label="Overall", linewidth=2, color="black")
    ax.plot(epochs, observed_loss, label="Observed", linewidth=1.5, color="green", linestyle="--")
    ax.plot(epochs, missing_loss, label="Missing", linewidth=1.5, color="red", linestyle="--")

    if stan_metrics and 'rating_missing_log_likelihood' in stan_metrics:
        stan_loss = -stan_metrics['rating_missing_log_likelihood']
        ax.axhline(y=stan_loss, color='blue', linestyle=':', linewidth=2, label=f'Stan Baseline (Missing): {stan_loss:.3f}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Rating Loss (Cross-Entropy)', fontsize=12)
    ax.set_title('Test Set: Rating Loss over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'test_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Rating Accuracy (Test)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, overall_acc, label="Overall", linewidth=2, color="black")
    if observed_acc is not None:
        ax.plot(epochs, observed_acc, label="Observed", linewidth=1.5, color="green", linestyle="--")
    if missing_acc is not None:
        ax.plot(epochs, missing_acc, label="Missing", linewidth=1.5, color="red", linestyle="--")

    if stan_metrics and "rating_missing_accuracy" in stan_metrics:
        stan_acc = stan_metrics['rating_missing_accuracy']
        ax.axhline(y=stan_acc, color='blue', linestyle=':', linewidth=2, label=f'Stan Baseline (Missing): {stan_acc:.3f}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Rating Accuracy', fontsize=12)
    ax.set_title('Test Set: Rating Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_dir / 'test_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Rating RMSE (Test)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, overall_rmse, label="Overall", linewidth=2, color="black")
    ax.plot(epochs, observed_rmse, label="Observed", linewidth=1.5, color="green", linestyle="--")
    ax.plot(epochs, missing_rmse, label="Missing", linewidth=1.5, color="red", linestyle="--")

    if stan_metrics and "rating_missing_mae" in stan_metrics:
        stan_mae = stan_metrics['rating_missing_mae']
        ax.axhline(y=stan_mae, color='blue', linestyle=':', linewidth=2, label=f'Stan MAE (Missing): {stan_mae:.3f}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Rating RMSE', fontsize=12)
    ax.set_title('Test Set: Rating RMSE over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'test_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Combined Loss + Accuracy (Test)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(epochs, overall_loss, label="Overall", linewidth=2, color="black")
    ax1.plot(epochs, observed_loss, label="Observed", linewidth=1.5, color="green", linestyle="--")
    ax1.plot(epochs, missing_loss, label="Missing", linewidth=1.5, color="red", linestyle="--")
    if stan_metrics and "rating_missing_log_likelihood" in stan_metrics:
        stan_loss = -stan_metrics["rating_missing_log_likelihood"]
        ax1.axhline(y=stan_loss, color="blue", linestyle=":", linewidth=2, label="Stan (Missing)")
    ax1.set_ylabel("Rating Loss", fontsize=12)
    ax1.set_title("Test Set: Loss and Accuracy", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, overall_acc, label="Overall", linewidth=2, color="black")
    ax2.plot(epochs, observed_acc, label="Observed", linewidth=1.5, color="green", linestyle="--")
    ax2.plot(epochs, missing_acc, label="Missing", linewidth=1.5, color="red", linestyle="--")
    if stan_metrics and "rating_missing_accuracy" in stan_metrics:
        stan_acc = stan_metrics['rating_missing_accuracy']
        ax2.axhline(y=stan_acc, color='blue', linestyle=':', linewidth=2, label=f'Stan (Missing)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Rating Accuracy', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'test_loss_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration(predictions: List[Dict], output_dir: Path):
    """Plot calibration curves for rating predictions (missing data only)."""
    # Filter to rating predictions (not listwise) and missing status only
    rating_preds = [p for p in predictions if not p['is_listwise'] and p['status'] == 0]

    if len(rating_preds) == 0:
        print("Warning: No missing rating predictions found for calibration plot")
        return

    # Extract probabilities and correctness
    max_probs = []
    predicted_classes = []
    true_classes = []
    is_correct = []

    for pred in rating_preds:
        if 'rating_probabilities' in pred and 'true_rating_class' in pred:
            probs = np.array(pred['rating_probabilities'])
            max_prob = np.max(probs)
            pred_class = np.argmax(probs)
            true_class = pred['true_rating_class']

            max_probs.append(max_prob)
            predicted_classes.append(pred_class)
            true_classes.append(true_class)
            is_correct.append(pred_class == true_class)

    max_probs = np.array(max_probs)
    is_correct = np.array(is_correct)

    # Plot 1: Calibration Curve (binned)
    fig, ax = plt.subplots(figsize=(8, 8))

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        mask = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i+1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (max_probs >= bin_edges[i]) & (max_probs <= bin_edges[i+1])

        if np.sum(mask) > 0:
            bin_acc = np.mean(is_correct[mask])
            bin_conf = np.mean(max_probs[mask])
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)

    # Plot calibration curve
    valid_mask = ~np.isnan(bin_accuracies)
    ax.plot(np.array(bin_confidences)[valid_mask], np.array(bin_accuracies)[valid_mask],
            marker='o', markersize=8, linewidth=2, color='red', label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Perfect Calibration')

    ax.set_xlabel('Confidence (Max Probability)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Calibration Curve (Missing Ratings Only)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Confidence Histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    correct_probs = max_probs[is_correct]
    incorrect_probs = max_probs[~is_correct]

    ax.hist(correct_probs, bins=20, alpha=0.6, label=f'Correct (n={len(correct_probs)})', color='green', edgecolor='black')
    ax.hist(incorrect_probs, bins=20, alpha=0.6, label=f'Incorrect (n={len(incorrect_probs)})', color='red', edgecolor='black')

    ax.set_xlabel('Max Probability (Confidence)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_analysis(predictions: List[Dict], output_dir: Path):
    """Plot detailed confidence analysis (missing data only)."""
    # Filter to rating predictions (not listwise) and missing status only
    rating_preds = [p for p in predictions if not p['is_listwise'] and p['status'] == 0]

    if len(rating_preds) == 0:
        print("Warning: No missing rating predictions found for confidence analysis")
        return

    # Extract data
    max_probs = []
    entropies = []
    is_correct = []
    true_classes = []

    for pred in rating_preds:
        if 'rating_probabilities' in pred and 'true_rating_class' in pred:
            probs = np.array(pred['rating_probabilities'])
            max_prob = np.max(probs)
            pred_class = np.argmax(probs)
            true_class = pred['true_rating_class']

            # Compute entropy
            probs_safe = probs + 1e-10  # Avoid log(0)
            entropy = -np.sum(probs_safe * np.log(probs_safe))

            max_probs.append(max_prob)
            entropies.append(entropy)
            is_correct.append(pred_class == true_class)
            true_classes.append(true_class)

    max_probs = np.array(max_probs)
    entropies = np.array(entropies)
    is_correct = np.array(is_correct)
    true_classes = np.array(true_classes)

    # Plot 1: Box plot of confidence for correct vs incorrect
    fig, ax = plt.subplots(figsize=(8, 6))

    data_to_plot = [max_probs[is_correct], max_probs[~is_correct]]
    bp = ax.boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True)

    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.6)

    ax.set_ylabel('Max Probability (Confidence)', fontsize=12)
    ax.set_title('Confidence Distribution: Correct vs Incorrect', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Entropy distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    correct_entropy = entropies[is_correct]
    incorrect_entropy = entropies[~is_correct]

    ax.hist(correct_entropy, bins=20, alpha=0.6, label=f'Correct (n={len(correct_entropy)})', color='green', edgecolor='black')
    ax.hist(incorrect_entropy, bins=20, alpha=0.6, label=f'Incorrect (n={len(incorrect_entropy)})', color='red', edgecolor='black')

    ax.set_xlabel('Entropy (Uncertainty)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Prediction Uncertainty: Correct vs Incorrect', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Scatter plot - confidence vs correctness colored by true class
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for true_class in range(5):
        mask = true_classes == true_class
        if np.sum(mask) > 0:
            correct_mask = mask & is_correct
            incorrect_mask = mask & ~is_correct

            ax.scatter(np.where(correct_mask)[0], max_probs[correct_mask],
                      color=colors[true_class], marker='o', s=20, alpha=0.5, label=f'Class {true_class} (correct)')
            ax.scatter(np.where(incorrect_mask)[0], max_probs[incorrect_mask],
                      color=colors[true_class], marker='x', s=30, alpha=0.7)

    ax.set_xlabel('Prediction Index', fontsize=12)
    ax.set_ylabel('Max Probability (Confidence)', fontsize=12)
    ax.set_title('Confidence by True Class (x = incorrect)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize imputer training run results")
    parser.add_argument("--run-dir", required=True, help="Path to imputer run directory")
    parser.add_argument("--stan-metrics", default=None, help="Optional path to Stan predictive_metrics.json for baseline comparison")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Create plots subdirectory
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Loading data from {run_dir}...")
    data = load_run_data(run_dir)

    # Load Stan metrics if provided
    stan_metrics = None
    if args.stan_metrics:
        stan_path = Path(args.stan_metrics)
        stan_metrics = load_stan_metrics(stan_path)
        if stan_metrics:
            print(f"Loaded Stan baseline metrics from {stan_path}")

    # Generate plots
    print("\nGenerating learning curves...")
    if data.get("train_instance_history"):
        train_src, train_name = data["train_instance_history"], "train_instance_history"
    elif data.get("train_history"):
        train_src, train_name = data["train_history"], "train_history"
    elif data.get("training_loss_history"):
        train_src, train_name = data["training_loss_history"], "training_loss_history"
    else:
        train_src = train_name = None
    if train_src:
        plot_learning_curves_train(train_src, stan_metrics, plots_dir)
        print(f"  - Train learning curves saved (from {train_name})")

    if data.get("test_instance_history"):
        test_src, test_name = data["test_instance_history"], "test_instance_history"
    else:
        test_src = test_name = None
    if test_src:
        plot_learning_curves_test(test_src, stan_metrics, plots_dir)
        print(f"  - Test learning curves saved (from {test_name})")

    print("\nGenerating calibration plots...")
    if 'predictives' in data:
        assert 'predictions' in data['predictives'], "predictives must contain 'predictions' key (from build_predictives)"
        plot_calibration(data['predictives']['predictions'], plots_dir)
        print("  - Calibration plots saved")

    print("\nGenerating confidence analysis plots...")
    if 'predictives' in data:
        assert 'predictions' in data['predictives'], "predictives must contain 'predictions' key (from build_predictives)"
        plot_confidence_analysis(data['predictives']['predictions'], plots_dir)
        print("  - Confidence analysis plots saved")

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
