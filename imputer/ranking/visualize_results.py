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


def plot_pretraining_loss(results: Dict[str, Any], output_dir: Path, test_config: Optional[Dict] = None) -> None:
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

    # Extract training losses - compute unweighted total if config available
    train_rating = [entry['rating_loss'] for entry in training_data]
    train_ranking = [entry['ranking_loss'] for entry in training_data]

    # Compute unweighted total loss for original loss values
    if test_config and 'model_config' in test_config:
        # Use unweighted total = rating_loss + ranking_loss (original loss)
        train_total = [r + rk for r, rk in zip(train_rating, train_ranking)]
        total_label = 'Training Total Loss (Unweighted)'
    else:
        # Fallback to weighted total from results
        train_total = [entry['total_loss'] for entry in training_data]
        total_label = 'Training Total Loss'

    plt.figure(figsize=(10, 6))

    # Plot training losses only (remove heldout)
    plt.plot(epochs, train_total, color=COLORS['total'], linestyle=LINE_STYLES['total'],
             linewidth=2, label=total_label)
    plt.plot(epochs, train_rating, color=COLORS['rating'], linestyle=LINE_STYLES['rating'],
             linewidth=1.5, label='Training Rating Loss')
    plt.plot(epochs, train_ranking, color=COLORS['ranking'], linestyle=LINE_STYLES['ranking'],
             linewidth=1.5, label='Training Ranking Loss')

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


def plot_finetuning_loss(results: Dict[str, Any], strategy_name: str, output_dir: Path, test_config: Optional[Dict] = None) -> None:
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

    avg_rating = np.zeros(max_epochs)
    avg_ranking = np.zeros(max_epochs)

    for training_data in all_training_data:
        for i, entry in enumerate(training_data):
            if i < max_epochs:
                avg_rating[i] += entry['rating_loss']
                avg_ranking[i] += entry['ranking_loss']

    # Average
    num_instances = len(all_training_data)
    avg_rating /= num_instances
    avg_ranking /= num_instances

    # Compute unweighted total loss for original loss values
    if test_config and 'model_config' in test_config:
        avg_total = avg_rating + avg_ranking  # Unweighted original loss
        total_label = 'Training Total Loss (Unweighted)'
    else:
        # Fallback: compute weighted average from raw data
        avg_total = np.zeros(max_epochs)
        for training_data in all_training_data:
            for i, entry in enumerate(training_data):
                if i < max_epochs:
                    avg_total[i] += entry['total_loss']
        avg_total /= num_instances
        total_label = 'Training Total Loss'

    plt.figure(figsize=(10, 6))

    # Plot finetuning losses
    color = COLORS['pretrain_finetuned'] if 'Pretrain' in strategy_name else COLORS['finetuned']

    plt.plot(epochs, avg_total, color=color, linestyle=LINE_STYLES['total'],
             linewidth=2, label=total_label)
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

    # Clean up strategy name for title (remove underscores)
    if strategy_name == 'Pretrain_Finetuned_Imputer':
        title = 'Pretrained & Finetuned Imputer Loss Curves'
    elif strategy_name == 'Finetuned_Imputer':
        title = 'Finetuned Imputer Loss Curves'
    else:
        title = f'{strategy_name.replace("_", " ")} Loss Curves'

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    strategy_filename = strategy_name.lower().replace('_', '_')
    output_path = output_dir / f'{strategy_filename}_loss_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_test_comparison(results: Dict[str, Any], output_dir: Path, max_epochs: int = 100, test_config: Optional[Dict] = None) -> None:
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
    max_epochs_found = 0

    # Collect data across all test instances
    for instance_key, instance_data in test_instances.items():
        # Pretrain_Finetuned_Imputer test loss progression
        if 'Pretrain_Finetuned_Imputer' in instance_data:
            callback_data = instance_data['Pretrain_Finetuned_Imputer'].get('callback_results', [])
            test_losses = []
            print(f"DEBUG: Processing Pretrain_Finetuned_Imputer for {instance_key}")
            print(f"DEBUG: test_config is not None: {test_config is not None}")
            print(f"DEBUG: callback_data length: {len(callback_data)}")

            # Take every second entry (index 1, 3, 5, ...) to get test evaluations
            for i in range(1, len(callback_data), 2):
                entry = callback_data[i]
                if isinstance(entry, dict) and 'total_loss' in entry:
                    # Debug first few entries
                    if i <= 5:
                        print(f"DEBUG: Entry {i} keys: {list(entry.keys())}")
                        if 'rating_loss' in entry and 'ranking_loss' in entry:
                            print(f"DEBUG: Entry {i} - rating_loss: {entry['rating_loss']}, ranking_loss: {entry['ranking_loss']}, total_loss: {entry['total_loss']}")

                    # Compute unweighted loss if config available
                    if test_config and 'rating_loss' in entry and 'ranking_loss' in entry:
                        # Use the unweighted components from loss strategy
                        unweighted_loss = entry['rating_loss'] + entry['ranking_loss']
                        test_losses.append(unweighted_loss)
                        if i <= 5:
                            print(f"DEBUG: Entry {i} - Using unweighted: rating_loss({entry['rating_loss']}) + ranking_loss({entry['ranking_loss']}) = {unweighted_loss}")
                            print(f"DEBUG: Entry {i} - vs total_loss: {entry['total_loss']}")
                            print(f"DEBUG: Entry {i} - Reduction: {((entry['total_loss'] - unweighted_loss) / entry['total_loss'] * 100):.1f}%")
                    else:
                        test_losses.append(entry['total_loss'])
                        if i <= 5:
                            print(f"DEBUG: Entry {i} - Using total_loss: {entry['total_loss']}")
                            print(f"DEBUG: test_config: {test_config is not None}, rating_loss in entry: {'rating_loss' in entry}, ranking_loss in entry: {'ranking_loss' in entry}")

            if test_losses:
                pretrain_finetuned_losses.append(test_losses)
                max_epochs_found = max(max_epochs_found, len(test_losses))

        # Finetuned_Imputer test loss progression
        if 'Finetuned_Imputer' in instance_data:
            callback_data = instance_data['Finetuned_Imputer'].get('callback_results', [])
            test_losses = []
            print(f"DEBUG: Processing Finetuned_Imputer for {instance_key}")

            # Take every second entry (index 1, 3, 5, ...) to get test evaluations
            for i in range(1, len(callback_data), 2):
                entry = callback_data[i]
                if isinstance(entry, dict) and 'total_loss' in entry:
                    # Compute unweighted loss if config available
                    if test_config and 'rating_loss' in entry and 'ranking_loss' in entry:
                        # Use the unweighted components from loss strategy
                        unweighted_loss = entry['rating_loss'] + entry['ranking_loss']
                        test_losses.append(unweighted_loss)
                        if i <= 5:
                            print(f"DEBUG: Finetuned Entry {i} - Using unweighted: {unweighted_loss} vs total: {entry['total_loss']}")
                    else:
                        test_losses.append(entry['total_loss'])
                        if i <= 5:
                            print(f"DEBUG: Finetuned Entry {i} - Using total_loss: {entry['total_loss']}")

            if test_losses:
                finetuned_losses.append(test_losses)
                max_epochs_found = max(max_epochs_found, len(test_losses))

    # Use minimum of found epochs and requested max_epochs
    plot_epochs = min(max_epochs_found, max_epochs) if max_epochs_found > 0 else max_epochs

    # If no callback data available, use final evaluation results only
    if max_epochs_found == 0:
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
        epochs = list(range(plot_epochs))

        if pretrain_finetuned_losses:
            # Average across instances
            avg_pretrain_losses = np.zeros(plot_epochs)
            count = np.zeros(plot_epochs)

            for losses in pretrain_finetuned_losses:
                for i, loss in enumerate(losses):
                    if i < plot_epochs:
                        avg_pretrain_losses[i] += loss
                        count[i] += 1

            # Avoid division by zero
            avg_pretrain_losses = np.where(count > 0, avg_pretrain_losses / count, np.nan)

            plt.plot(epochs, avg_pretrain_losses, color=COLORS['pretrain_finetuned'],
                     linestyle='-', linewidth=2, label='Pretrained & Finetuned Imputer')

        if finetuned_losses:
            # Average across instances
            avg_finetuned_losses = np.zeros(plot_epochs)
            count = np.zeros(plot_epochs)

            for losses in finetuned_losses:
                for i, loss in enumerate(losses):
                    if i < plot_epochs:
                        avg_finetuned_losses[i] += loss
                        count[i] += 1

            avg_finetuned_losses = np.where(count > 0, avg_finetuned_losses / count, np.nan)

            plt.plot(epochs, avg_finetuned_losses, color=COLORS['finetuned'],
                     linestyle='-', linewidth=2, label='Finetuned Imputer')

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

    # Plot MCMC lines with different colors for each sample size
    mcmc_colors = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']  # Different colors for each sample size
    for i, (sample_count, losses) in enumerate(mcmc_by_samples.items()):
        avg_loss = np.mean(losses)
        color = mcmc_colors[i % len(mcmc_colors)]
        plt.axhline(y=avg_loss, color=color, linestyle='-',  # Solid lines for MCMC
                   linewidth=1.5, alpha=0.8, label=f'MCMC {sample_count} Samples')

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


def plot_test_accuracy(results: Dict[str, Any], output_dir: Path, max_epochs: int = 100) -> None:
    """
    Plot test accuracy for pretrained imputer only.
    Shows pretraining metrics with dotted line, then finetuning with trend.
    """
    test_instances = results.get('test_instance_results', {})
    pretraining_results = results.get('pretraining_results', {})

    plt.figure(figsize=(12, 8))

    # Get Pretrained Imputer final test accuracy (baseline to compare against)
    pretrained_imputer_rating_acc = []
    pretrained_imputer_ranking_acc = []

    # Collect Pretrained_Imputer final test results across all instances
    for instance_key, instance_data in test_instances.items():
        if 'Pretrained_Imputer' in instance_data:
            eval_results = instance_data['Pretrained_Imputer'].get('evaluation_results', {})
            if 'rating_accuracy' in eval_results:
                pretrained_imputer_rating_acc.append(eval_results['rating_accuracy'])
            if 'ranking_accuracy' in eval_results:
                pretrained_imputer_ranking_acc.append(eval_results['ranking_accuracy'])

    # Plot horizontal dotted lines for Pretrained Imputer performance (baseline)
    if pretrained_imputer_rating_acc and pretrained_imputer_ranking_acc:
        avg_pretrained_rating = sum(pretrained_imputer_rating_acc) / len(pretrained_imputer_rating_acc)
        avg_pretrained_ranking = sum(pretrained_imputer_ranking_acc) / len(pretrained_imputer_ranking_acc)
        avg_pretrained_overall = (avg_pretrained_rating + avg_pretrained_ranking) / 2

        plt.axhline(y=avg_pretrained_overall, color=COLORS['total'], linestyle=':',
                   linewidth=2, alpha=0.8, label='Pretrained Imputer Overall Accuracy')
        plt.axhline(y=avg_pretrained_rating, color=COLORS['rating'], linestyle=':',
                   linewidth=1.5, alpha=0.8, label='Pretrained Imputer Rating Accuracy')
        plt.axhline(y=avg_pretrained_ranking, color=COLORS['ranking'], linestyle=':',
                   linewidth=1.5, alpha=0.8, label='Pretrained Imputer Ranking Accuracy')

    # Get finetuning test accuracy progression for Pretrained & Finetuned Imputer
    pretrain_finetuned_rating_acc = []
    pretrain_finetuned_ranking_acc = []
    max_epochs_found = 0

    # Collect data across all test instances
    for instance_key, instance_data in test_instances.items():
        if 'Pretrain_Finetuned_Imputer' in instance_data:
            callback_data = instance_data['Pretrain_Finetuned_Imputer'].get('callback_results', [])
            rating_accs = []
            ranking_accs = []

            # Take every second entry (index 1, 3, 5, ...) to get test evaluations
            for i in range(1, len(callback_data), 2):
                entry = callback_data[i]
                if isinstance(entry, dict):
                    if 'rating_accuracy' in entry:
                        rating_accs.append(entry['rating_accuracy'])
                    if 'ranking_accuracy' in entry:
                        ranking_accs.append(entry['ranking_accuracy'])

            if rating_accs:
                pretrain_finetuned_rating_acc.append(rating_accs)
                max_epochs_found = max(max_epochs_found, len(rating_accs))
            if ranking_accs:
                pretrain_finetuned_ranking_acc.append(ranking_accs)
                max_epochs_found = max(max_epochs_found, len(ranking_accs))

    # Use minimum of found epochs and requested max_epochs
    plot_epochs = min(max_epochs_found, max_epochs) if max_epochs_found > 0 else max_epochs

    if pretrain_finetuned_rating_acc and pretrain_finetuned_ranking_acc:
        epochs = list(range(plot_epochs))

        # Average rating accuracy across instances
        avg_rating_acc = np.zeros(plot_epochs)
        rating_count = np.zeros(plot_epochs)

        for accs in pretrain_finetuned_rating_acc:
            for i, acc in enumerate(accs):
                if i < plot_epochs:
                    avg_rating_acc[i] += acc
                    rating_count[i] += 1

        avg_rating_acc = np.where(rating_count > 0, avg_rating_acc / rating_count, np.nan)

        # Average ranking accuracy across instances
        avg_ranking_acc = np.zeros(plot_epochs)
        ranking_count = np.zeros(plot_epochs)

        for accs in pretrain_finetuned_ranking_acc:
            for i, acc in enumerate(accs):
                if i < plot_epochs:
                    avg_ranking_acc[i] += acc
                    ranking_count[i] += 1

        avg_ranking_acc = np.where(ranking_count > 0, avg_ranking_acc / ranking_count, np.nan)

        # Overall accuracy (average of rating and ranking)
        avg_overall_acc = (avg_rating_acc + avg_ranking_acc) / 2

        # Plot finetuning accuracy trends
        plt.plot(epochs, avg_overall_acc, color=COLORS['total'], linestyle='-',
                linewidth=2, label='Pretrained & Finetuned Overall Accuracy')
        plt.plot(epochs, avg_rating_acc, color=COLORS['rating'], linestyle='-',
                linewidth=1.5, label='Pretrained & Finetuned Rating Accuracy')
        plt.plot(epochs, avg_ranking_acc, color=COLORS['ranking'], linestyle='-',
                linewidth=1.5, label='Pretrained & Finetuned Ranking Accuracy')

    plt.xlabel('Finetuning Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Pretrained Imputer vs Pretrained & Finetuned Imputer Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1

    # Save plot
    output_path = output_dir / 'test_accuracy_pretrained_imputer.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function to generate all plots."""
    parser = argparse.ArgumentParser(description='Generate ICLR-ready plots from experiment results')
    parser.add_argument('json_path', type=str, help='Path to experiment results JSON file')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--test-config', type=str, default=None,
                       help='Path to test config JSON file for unweighted loss calculation')
    parser.add_argument('--max-epochs', type=int, default=100,
                       help='Maximum epochs to display in progression plots (default: 100)')

    args = parser.parse_args()

    # Load results
    if not Path(args.json_path).exists():
        print(f"Error: Results file not found: {args.json_path}")
        return

    results = load_results(args.json_path)

    # Load test config if provided
    test_config = None
    if args.test_config and Path(args.test_config).exists():
        with open(args.test_config, 'r') as f:
            test_config = json.load(f)
        print(f"Loaded test config: {args.test_config}")
        print(f"DEBUG: test_config keys: {list(test_config.keys())}")
        if 'model_config' in test_config:
            print(f"DEBUG: model_config keys: {list(test_config['model_config'].keys())}")
    else:
        print(f"DEBUG: test_config not loaded. args.test_config: {args.test_config}, exists: {Path(args.test_config).exists() if args.test_config else 'N/A'}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Generating plots from: {args.json_path}")
    print(f"Output directory: {output_dir}")
    print(f"Max epochs for progression plots: {args.max_epochs}")

    # Generate all plots
    print("\nGenerating plots...")

    # Plot 1: Pretraining loss curves
    plot_pretraining_loss(results, output_dir, test_config)

    # Plot 2a: Pretrain_Finetuned loss curves
    plot_finetuning_loss(results, 'Pretrain_Finetuned_Imputer', output_dir, test_config)

    # Plot 2b: Finetuned loss curves
    plot_finetuning_loss(results, 'Finetuned_Imputer', output_dir, test_config)

    # Plot 3: Main comparison plot
    plot_test_comparison(results, output_dir, args.max_epochs, test_config)

    # Plot 4: Test accuracy plot
    plot_test_accuracy(results, output_dir, args.max_epochs)

    print(f"\nAll plots saved to: {output_dir}")
    print("Generated files:")
    for plot_file in output_dir.glob('*.png'):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    main()