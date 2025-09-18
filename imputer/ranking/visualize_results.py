#!/usr/bin/env python3
"""
ICLR-Ready Visualization Script for Ranking Imputation Results

Creates publication-ready plots:
1. Pretraining Training Loss Trend (training + heldout)
2. Finetuning Loss Trends (separate for Pretrain_Finetuned and Finetuned)
3. Main Comparison Plot (test loss comparison across methods)
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# ICLR-style plotting configuration
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'legend.frameon': False,
    'legend.fontsize': 11,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color scheme for ICLR plots
COLORS = {
    'total': '#1f77b4',      # Blue
    'rating': '#ff7f0e',     # Orange
    'ranking': '#2ca02c',    # Green
    'heldout': '#d62728',    # Red
    'pretrain_finetuned': '#9467bd',  # Purple
    'finetuned': '#8c564b',  # Brown
    'mcmc': '#e377c2'        # Pink
}

LINE_STYLES = {
    'total': '-',
    'rating': '--',
    'ranking': ':',
    'heldout': '-'
}


def load_results(json_path: str) -> Dict[str, Any]:
    """Load experiment results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_pretraining_loss(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Plot 1: Pretraining Training Loss Trend
    Shows training loss splits (total, rating, ranking) + heldout total loss
    """
    pretraining_results = results.get('pretraining_results', {})
    training_data = pretraining_results.get('training_results', [])
    heldout_data = pretraining_results.get('heldout_evaluation_metrics', [])

    if not training_data:
        print("Warning: No pretraining training data found")
        return

    epochs = list(range(len(training_data)))

    # Extract training losses
    train_total = [entry['total_loss'] for entry in training_data]
    train_rating = [entry['rating_loss'] for entry in training_data]
    train_ranking = [entry['ranking_loss'] for entry in training_data]

    # Extract heldout total loss (if available)
    heldout_total = []
    heldout_epochs = []
    if heldout_data:
        for entry in heldout_data:
            if 'epoch' in entry and 'total_loss' in entry:
                heldout_epochs.append(entry['epoch'])
                heldout_total.append(entry['total_loss'])

    plt.figure(figsize=(10, 6))

    # Plot training losses
    plt.plot(epochs, train_total, color=COLORS['total'], linestyle=LINE_STYLES['total'],
             linewidth=2, label='Training Total Loss')
    plt.plot(epochs, train_rating, color=COLORS['rating'], linestyle=LINE_STYLES['rating'],
             linewidth=1.5, label='Training Rating Loss')
    plt.plot(epochs, train_ranking, color=COLORS['ranking'], linestyle=LINE_STYLES['ranking'],
             linewidth=1.5, label='Training Ranking Loss')

    # Plot heldout total loss if available
    if heldout_total:
        plt.plot(heldout_epochs, heldout_total, color=COLORS['heldout'], linestyle=LINE_STYLES['heldout'],
                 linewidth=2, label='Heldout Total Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Imputer Pretraining Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    output_path = output_dir / 'pretraining_loss_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_finetuning_loss(results: Dict[str, Any], strategy_name: str, output_dir: Path) -> None:
    """
    Plot 2: Finetuning Loss Trends (separate plots for each strategy)
    Shows finetuning loss splits for a specific strategy
    """
    test_instances = results.get('test_instance_results', {})

    # Collect all finetuning data across test instances
    all_training_data = []
    all_callback_data = []

    for instance_key, instance_data in test_instances.items():
        if strategy_name in instance_data:
            strategy_data = instance_data[strategy_name]

            # Get training progression data
            training_results = strategy_data.get('finetuning_results', [])
            if strategy_name == 'Finetuned_Imputer':
                training_results = strategy_data.get('training_results', [])

            if training_results:
                all_training_data.append(training_results)

            # Get callback data (heldout/test evaluation during training)
            callback_results = strategy_data.get('callback_results', [])
            if callback_results:
                all_callback_data.append(callback_results)

    if not all_training_data:
        print(f"Warning: No finetuning data found for {strategy_name}")
        return

    # Average across instances (assuming same length)
    max_epochs = max(len(data) for data in all_training_data)
    epochs = list(range(max_epochs))

    avg_total = np.zeros(max_epochs)
    avg_rating = np.zeros(max_epochs)
    avg_ranking = np.zeros(max_epochs)

    for training_data in all_training_data:
        for i, entry in enumerate(training_data):
            if i < max_epochs:
                avg_total[i] += entry['total_loss']
                avg_rating[i] += entry['rating_loss']
                avg_ranking[i] += entry['ranking_loss']

    # Average
    num_instances = len(all_training_data)
    avg_total /= num_instances
    avg_rating /= num_instances
    avg_ranking /= num_instances

    plt.figure(figsize=(10, 6))

    # Plot finetuning losses
    color = COLORS['pretrain_finetuned'] if 'Pretrain' in strategy_name else COLORS['finetuned']

    plt.plot(epochs, avg_total, color=color, linestyle=LINE_STYLES['total'],
             linewidth=2, label='Training Total Loss')
    plt.plot(epochs, avg_rating, color=COLORS['rating'], linestyle=LINE_STYLES['rating'],
             linewidth=1.5, label='Training Rating Loss')
    plt.plot(epochs, avg_ranking, color=COLORS['ranking'], linestyle=LINE_STYLES['ranking'],
             linewidth=1.5, label='Training Ranking Loss')

    # Add callback data if available (test evaluation - use every second entry)
    if all_callback_data:
        # Average callback results across instances
        callback_epochs = []
        callback_losses = []

        # Use first instance as reference for callback timing
        # Take every second entry (index 1, 3, 5, ...) to get test evaluations
        for i in range(1, len(all_callback_data[0]), 2):
            entry = all_callback_data[0][i]
            if isinstance(entry, dict) and 'total_loss' in entry and 'epoch' in entry:
                epoch = entry['epoch']
                if epoch < max_epochs:
                    callback_epochs.append(epoch)

                    # Average across all instances for this callback point
                    avg_callback_loss = 0
                    count = 0
                    for callback_data in all_callback_data:
                        if len(callback_data) > i:
                            cb_entry = callback_data[i]  # Use same index (test evaluation)
                            if isinstance(cb_entry, dict) and 'total_loss' in cb_entry:
                                avg_callback_loss += cb_entry['total_loss']
                                count += 1

                    if count > 0:
                        callback_losses.append(avg_callback_loss / count)
                    else:
                        callback_losses.append(entry['total_loss'])

        if callback_epochs and callback_losses:
            plt.plot(callback_epochs, callback_losses, color=COLORS['heldout'],
                     linestyle=LINE_STYLES['heldout'], linewidth=2, label='Test Total Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{strategy_name} Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    strategy_filename = strategy_name.lower().replace('_', '_')
    output_path = output_dir / f'{strategy_filename}_loss_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_test_comparison(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Plot 3: Main Comparison Plot
    X-axis: epochs, Y-axis: test loss
    Lines: Pretrain_Finetuned, Finetuned, MCMC baselines
    """
    test_instances = results.get('test_instance_results', {})
    domain_config = results.get('experiment_metadata', {}).get('config', {}).get('domain_config', {})
    sample_counts = domain_config.get('sample_counts', [])

    plt.figure(figsize=(12, 8))

    # Get finetuning progression data
    pretrain_finetuned_losses = []
    finetuned_losses = []
    max_epochs = 0

    # Collect data across all test instances
    for instance_key, instance_data in test_instances.items():
        # Pretrain_Finetuned_Imputer test loss progression
        if 'Pretrain_Finetuned_Imputer' in instance_data:
            callback_data = instance_data['Pretrain_Finetuned_Imputer'].get('callback_results', [])
            test_losses = []
            # Take every second entry (index 1, 3, 5, ...) to get test evaluations
            for i in range(1, len(callback_data), 2):
                entry = callback_data[i]
                if isinstance(entry, dict) and 'total_loss' in entry:
                    test_losses.append(entry['total_loss'])

            if test_losses:
                pretrain_finetuned_losses.append(test_losses)
                max_epochs = max(max_epochs, len(test_losses))

        # Finetuned_Imputer test loss progression
        if 'Finetuned_Imputer' in instance_data:
            callback_data = instance_data['Finetuned_Imputer'].get('callback_results', [])
            test_losses = []
            # Take every second entry (index 1, 3, 5, ...) to get test evaluations
            for i in range(1, len(callback_data), 2):
                entry = callback_data[i]
                if isinstance(entry, dict) and 'total_loss' in entry:
                    test_losses.append(entry['total_loss'])

            if test_losses:
                finetuned_losses.append(test_losses)
                max_epochs = max(max_epochs, len(test_losses))

    # If no callback data available, use final evaluation results only
    if max_epochs == 0:
        print("Warning: No callback progression data found, using final evaluation results")

        # Get final test results instead
        pretrain_final = []
        finetuned_final = []

        for instance_key, instance_data in test_instances.items():
            if 'Pretrain_Finetuned_Imputer' in instance_data:
                eval_results = instance_data['Pretrain_Finetuned_Imputer'].get('evaluation_results', {})
                if 'total_loss' in eval_results:
                    pretrain_final.append(eval_results['total_loss'])

            if 'Finetuned_Imputer' in instance_data:
                eval_results = instance_data['Finetuned_Imputer'].get('evaluation_results', {})
                if 'total_loss' in eval_results:
                    finetuned_final.append(eval_results['total_loss'])

        # Plot as horizontal lines
        if pretrain_final:
            avg_pretrain = np.mean(pretrain_final)
            plt.axhline(y=avg_pretrain, color=COLORS['pretrain_finetuned'],
                       linestyle='-', linewidth=2, label='Pretrained_Finetuned_Imputer')

        if finetuned_final:
            avg_finetuned = np.mean(finetuned_final)
            plt.axhline(y=avg_finetuned, color=COLORS['finetuned'],
                       linestyle='-', linewidth=2, label='Finetuned_Imputer')

        plt.xlim(0, 100)  # Default x range when no progression data

    else:
        # Plot progression curves
        epochs = list(range(max_epochs))

        if pretrain_finetuned_losses:
            # Average across instances
            avg_pretrain_losses = np.zeros(max_epochs)
            count = np.zeros(max_epochs)

            for losses in pretrain_finetuned_losses:
                for i, loss in enumerate(losses):
                    if i < max_epochs:
                        avg_pretrain_losses[i] += loss
                        count[i] += 1

            # Avoid division by zero
            avg_pretrain_losses = np.where(count > 0, avg_pretrain_losses / count, np.nan)

            plt.plot(epochs, avg_pretrain_losses, color=COLORS['pretrain_finetuned'],
                     linestyle='-', linewidth=2, label='Pretrained_Finetuned_Imputer')

        if finetuned_losses:
            # Average across instances
            avg_finetuned_losses = np.zeros(max_epochs)
            count = np.zeros(max_epochs)

            for losses in finetuned_losses:
                for i, loss in enumerate(losses):
                    if i < max_epochs:
                        avg_finetuned_losses[i] += loss
                        count[i] += 1

            avg_finetuned_losses = np.where(count > 0, avg_finetuned_losses / count, np.nan)

            plt.plot(epochs, avg_finetuned_losses, color=COLORS['finetuned'],
                     linestyle='-', linewidth=2, label='Finetuned_Imputer')

    # Add MCMC baseline horizontal lines
    mcmc_results = []
    for instance_key, instance_data in test_instances.items():
        if 'Domain_Model' in instance_data:
            domain_data = instance_data['Domain_Model']

            for sample_count in sample_counts:
                sample_key = f'samples_{sample_count}'
                if sample_key in domain_data:
                    eval_results = domain_data[sample_key].get('evaluation_results', {})
                    if 'total_loss' in eval_results:
                        mcmc_results.append((sample_count, eval_results['total_loss']))

    # Group by sample count and average
    mcmc_by_samples = {}
    for sample_count, loss in mcmc_results:
        if sample_count not in mcmc_by_samples:
            mcmc_by_samples[sample_count] = []
        mcmc_by_samples[sample_count].append(loss)

    # Plot MCMC lines
    for sample_count, losses in mcmc_by_samples.items():
        avg_loss = np.mean(losses)
        plt.axhline(y=avg_loss, color=COLORS['mcmc'], linestyle='--',
                   linewidth=1.5, alpha=0.8, label=f'MCMC_{sample_count}')

    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    output_path = output_dir / 'test_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function to generate all plots."""
    parser = argparse.ArgumentParser(description='Generate ICLR-ready plots from experiment results')
    parser.add_argument('json_path', type=str, help='Path to experiment results JSON file')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots (default: plots)')

    args = parser.parse_args()

    # Load results
    if not Path(args.json_path).exists():
        print(f"Error: Results file not found: {args.json_path}")
        return

    results = load_results(args.json_path)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Generating plots from: {args.json_path}")
    print(f"Output directory: {output_dir}")

    # Generate all plots
    print("\nGenerating plots...")

    # Plot 1: Pretraining loss curves
    plot_pretraining_loss(results, output_dir)

    # Plot 2a: Pretrain_Finetuned loss curves
    plot_finetuning_loss(results, 'Pretrain_Finetuned_Imputer', output_dir)

    # Plot 2b: Finetuned loss curves
    plot_finetuning_loss(results, 'Finetuned_Imputer', output_dir)

    # Plot 3: Main comparison plot
    plot_test_comparison(results, output_dir)

    print(f"\nAll plots saved to: {output_dir}")
    print("Generated files:")
    for plot_file in output_dir.glob('*.png'):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    main()