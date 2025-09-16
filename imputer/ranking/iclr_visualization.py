#!/usr/bin/env python3
"""Visualization and reporting for ICLR experiments."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ICLRResultsAnalyzer:
    """Analyzes and visualizes ICLR experiment results."""

    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        with open(results_path, 'r') as f:
            self.results = json.load(f)

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def create_results_table(self) -> pd.DataFrame:
        """Create comprehensive results table with statistics."""
        logger.info("Creating results table...")

        # Extract results for each test instance and method
        table_data = []

        for test_idx, test_results in self.results['test_results'].items():
            instance_data = {'test_instance': int(test_idx)}

            # Method 1: Pretrained only
            pretrained = test_results['pretrained_only']
            instance_data.update({
                'pretrained_total_loss': pretrained['total_log_loss'],
                'pretrained_rating_loss': pretrained['rating_log_loss'],
                'pretrained_ranking_loss': pretrained['ranking_log_loss'],
                'pretrained_rating_accuracy': pretrained['rating_accuracy'],
                'pretrained_ranking_accuracy': pretrained['ranking_accuracy'],
                'pretrained_rating_rmse': pretrained['rating_rmse'],
                'pretrained_runtime': pretrained['wall_time']
            })

            # Method 2: Pretrained + Finetuned
            finetuned = test_results['pretrained_finetuned']
            instance_data.update({
                'finetuned_total_loss': finetuned['total_log_loss'],
                'finetuned_rating_loss': finetuned['rating_log_loss'],
                'finetuned_ranking_loss': finetuned['ranking_log_loss'],
                'finetuned_rating_accuracy': finetuned['rating_accuracy'],
                'finetuned_ranking_accuracy': finetuned['ranking_accuracy'],
                'finetuned_rating_rmse': finetuned['rating_rmse'],
                'finetuned_runtime': finetuned['wall_time']
            })

            # Method 3: No pretrain
            no_pretrain = test_results['no_pretrain_finetuned']
            instance_data.update({
                'no_pretrain_total_loss': no_pretrain['total_log_loss'],
                'no_pretrain_rating_loss': no_pretrain['rating_log_loss'],
                'no_pretrain_ranking_loss': no_pretrain['ranking_log_loss'],
                'no_pretrain_rating_accuracy': no_pretrain['rating_accuracy'],
                'no_pretrain_ranking_accuracy': no_pretrain['ranking_accuracy'],
                'no_pretrain_rating_rmse': no_pretrain['rating_rmse'],
                'no_pretrain_runtime': no_pretrain['wall_time']
            })

            # Method 4: Domain model (best MCMC result)
            domain_results = test_results['domain_model']
            best_domain = max(domain_results.values(), key=lambda x: x['mcmc_samples'])
            instance_data.update({
                'domain_total_loss': best_domain['total_log_loss'],
                'domain_rating_loss': best_domain['rating_log_loss'],
                'domain_ranking_loss': best_domain['ranking_log_loss'],
                'domain_rating_accuracy': best_domain['rating_accuracy'],
                'domain_ranking_accuracy': best_domain['ranking_accuracy'],
                'domain_rating_rmse': best_domain['rating_rmse'],
                'domain_mcmc_samples': best_domain['mcmc_samples']
            })

            table_data.append(instance_data)

        df = pd.DataFrame(table_data)

        # Calculate summary statistics
        methods = ['pretrained', 'finetuned', 'no_pretrain', 'domain']
        metrics = ['total_loss', 'rating_loss', 'ranking_loss', 'rating_accuracy',
                  'ranking_accuracy', 'rating_rmse', 'runtime']

        summary_stats = []
        for method in methods:
            for metric in metrics:
                col_name = f'{method}_{metric}'
                if col_name in df.columns:
                    mean_val = df[col_name].mean()
                    std_val = df[col_name].std()
                    summary_stats.append({
                        'method': method,
                        'metric': metric,
                        'mean': mean_val,
                        'std': std_val,
                        'mean_pm_std': f"{mean_val:.4f} ± {std_val:.4f}"
                    })

        summary_df = pd.DataFrame(summary_stats)

        return df, summary_df

    def plot_training_curves(self, output_dir: Path) -> None:
        """Plot training curves during pretraining."""
        logger.info("Creating training curves plot...")

        pretraining_history = self.results.get('pretraining_history', {})
        if not pretraining_history:
            logger.warning("No pretraining history found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Pretraining Progress on Combined Training Instances', fontsize=16)

        epochs = list(range(len(pretraining_history['train_loss'])))

        # Training loss
        axes[0, 0].plot(epochs, pretraining_history['train_loss'], 'b-', label='Train')
        if 'heldout_loss' in pretraining_history:
            axes[0, 0].plot(epochs, pretraining_history['heldout_loss'], 'g--', label='Training Heldout')
        if 'val_loss' in pretraining_history:
            axes[0, 0].plot(epochs, pretraining_history['val_loss'], 'r--', label='Test Instances')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Rating loss
        if 'train_rating_loss' in pretraining_history:
            axes[0, 1].plot(epochs, pretraining_history['train_rating_loss'], 'b-', label='Train')
            if 'heldout_rating_loss' in pretraining_history:
                axes[0, 1].plot(epochs, pretraining_history['heldout_rating_loss'], 'g--', label='Training Heldout')
            if 'val_rating_loss' in pretraining_history:
                axes[0, 1].plot(epochs, pretraining_history['val_rating_loss'], 'r--', label='Test Instances')
            axes[0, 1].set_title('Rating Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Ranking loss
        if 'train_ranking_loss' in pretraining_history:
            axes[1, 0].plot(epochs, pretraining_history['train_ranking_loss'], 'b-', label='Train')
            if 'heldout_ranking_loss' in pretraining_history:
                axes[1, 0].plot(epochs, pretraining_history['heldout_ranking_loss'], 'g--', label='Training Heldout')
            if 'val_ranking_loss' in pretraining_history:
                axes[1, 0].plot(epochs, pretraining_history['val_ranking_loss'], 'r--', label='Test Instances')
            axes[1, 0].set_title('Ranking Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Training time
        if 'epoch_times' in pretraining_history:
            cumulative_time = np.cumsum(pretraining_history['epoch_times'])
            axes[1, 1].plot(epochs, cumulative_time, 'g-')
            axes[1, 1].set_title('Cumulative Training Time')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_runtime_comparison(self, output_dir: Path) -> None:
        """Create runtime vs performance comparison plot."""
        logger.info("Creating runtime comparison plot...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Runtime vs Performance Comparison', fontsize=16)

        metrics = ['total_log_loss', 'rating_log_loss', 'ranking_log_loss']
        metric_names = ['Total Loss', 'Rating Loss', 'Ranking Loss']

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]

            # Extract data for each method across test instances
            for test_idx, test_results in self.results['test_results'].items():

                # Method 1: Pretrained only (single point)
                pretrained = test_results['pretrained_only']
                ax.scatter(pretrained['wall_time'], pretrained[metric],
                          color='blue', marker='o', s=100, alpha=0.7, label='Pretrained Only')

                # Method 2: Finetuned (single point)
                finetuned = test_results['finetuned']
                ax.scatter(finetuned['wall_time'], finetuned[metric],
                          color='green', marker='s', s=100, alpha=0.7, label='Pretrained + Finetuned')

                # Method 3: No pretrain (single point)
                no_pretrain = test_results['no_pretrain']
                ax.scatter(no_pretrain['wall_time'], no_pretrain[metric],
                          color='orange', marker='^', s=100, alpha=0.7, label='No Pretrain')

                # Method 4: Domain model (curve with MCMC samples)
                domain_results = test_results['domain_model']
                times = [result['wall_time'] for result in domain_results.values()]
                losses = [result[metric] for result in domain_results.values()]
                samples = [result['mcmc_samples'] for result in domain_results.values()]

                # Sort by MCMC samples for proper curve
                sorted_data = sorted(zip(times, losses, samples), key=lambda x: x[2])
                sorted_times, sorted_losses, sorted_samples = zip(*sorted_data)

                ax.plot(sorted_times, sorted_losses, 'r-', alpha=0.7, linewidth=2)
                ax.scatter(sorted_times, sorted_losses, color='red', marker='d',
                          s=60, alpha=0.7, label='Domain EM')

                # Annotate MCMC sample points
                for time, loss, sample in zip(sorted_times[-2:], sorted_losses[-2:], sorted_samples[-2:]):
                    ax.annotate(f'{sample}', (time, loss), xytext=(5, 5),
                               textcoords='offset points', fontsize=8, alpha=0.8)

            ax.set_xlabel('Runtime (seconds)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Runtime')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            # Only show legend on first subplot
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.tight_layout()
        plt.savefig(output_dir / 'runtime_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_method_comparison_boxplots(self, output_dir: Path) -> None:
        """Create box plots comparing methods across test instances."""
        logger.info("Creating method comparison box plots...")

        # Prepare data for plotting
        plot_data = []

        for test_idx, test_results in self.results['test_results'].items():
            # Pretrained only
            pretrained = test_results['pretrained_only']
            plot_data.extend([
                {'Method': 'Pretrained Only', 'Metric': 'Total Loss', 'Value': pretrained['total_log_loss']},
                {'Method': 'Pretrained Only', 'Metric': 'Rating Loss', 'Value': pretrained['rating_log_loss']},
                {'Method': 'Pretrained Only', 'Metric': 'Ranking Loss', 'Value': pretrained['ranking_log_loss']},
                {'Method': 'Pretrained Only', 'Metric': 'Rating Accuracy', 'Value': pretrained['rating_accuracy']},
                {'Method': 'Pretrained Only', 'Metric': 'Ranking Accuracy', 'Value': pretrained['ranking_accuracy']},
                {'Method': 'Pretrained Only', 'Metric': 'Rating RMSE', 'Value': pretrained['rating_rmse']}
            ])

            # Finetuned
            finetuned = test_results['finetuned']
            plot_data.extend([
                {'Method': 'Pretrained + Finetuned', 'Metric': 'Total Loss', 'Value': finetuned['total_log_loss']},
                {'Method': 'Pretrained + Finetuned', 'Metric': 'Rating Loss', 'Value': finetuned['rating_log_loss']},
                {'Method': 'Pretrained + Finetuned', 'Metric': 'Ranking Loss', 'Value': finetuned['ranking_log_loss']},
                {'Method': 'Pretrained + Finetuned', 'Metric': 'Rating Accuracy', 'Value': finetuned['rating_accuracy']},
                {'Method': 'Pretrained + Finetuned', 'Metric': 'Ranking Accuracy', 'Value': finetuned['ranking_accuracy']},
                {'Method': 'Pretrained + Finetuned', 'Metric': 'Rating RMSE', 'Value': finetuned['rating_rmse']}
            ])

            # No pretrain
            no_pretrain = test_results['no_pretrain']
            plot_data.extend([
                {'Method': 'No Pretrain', 'Metric': 'Total Loss', 'Value': no_pretrain['total_log_loss']},
                {'Method': 'No Pretrain', 'Metric': 'Rating Loss', 'Value': no_pretrain['rating_log_loss']},
                {'Method': 'No Pretrain', 'Metric': 'Ranking Loss', 'Value': no_pretrain['ranking_log_loss']},
                {'Method': 'No Pretrain', 'Metric': 'Rating Accuracy', 'Value': no_pretrain['rating_accuracy']},
                {'Method': 'No Pretrain', 'Metric': 'Ranking Accuracy', 'Value': no_pretrain['ranking_accuracy']},
                {'Method': 'No Pretrain', 'Metric': 'Rating RMSE', 'Value': no_pretrain['rating_rmse']}
            ])

            # Domain model (best result)
            domain_results = test_results['domain_model']
            best_domain = max(domain_results.values(), key=lambda x: x['mcmc_samples'])
            plot_data.extend([
                {'Method': 'Domain EM', 'Metric': 'Total Loss', 'Value': best_domain['total_log_loss']},
                {'Method': 'Domain EM', 'Metric': 'Rating Loss', 'Value': best_domain['rating_log_loss']},
                {'Method': 'Domain EM', 'Metric': 'Ranking Loss', 'Value': best_domain['ranking_log_loss']},
                {'Method': 'Domain EM', 'Metric': 'Rating Accuracy', 'Value': best_domain['rating_accuracy']},
                {'Method': 'Domain EM', 'Metric': 'Ranking Accuracy', 'Value': best_domain['ranking_accuracy']},
                {'Method': 'Domain EM', 'Metric': 'Rating RMSE', 'Value': best_domain['rating_rmse']}
            ])

        df = pd.DataFrame(plot_data)

        # Create subplots for different metric types
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Method Comparison Across Test Instances', fontsize=16)

        metrics = ['Total Loss', 'Rating Loss', 'Ranking Loss',
                  'Rating Accuracy', 'Ranking Accuracy', 'Rating RMSE']

        for idx, metric in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            metric_data = df[df['Metric'] == metric]
            sns.boxplot(data=metric_data, x='Method', y='Value', ax=ax)
            ax.set_title(metric)
            ax.set_xlabel('')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            # Use log scale for loss metrics
            if 'Loss' in metric:
                ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(output_dir / 'method_comparison_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_results_tables(self, output_dir: Path) -> None:
        """Save detailed results tables."""
        logger.info("Saving results tables...")

        df, summary_df = self.create_results_table()

        # Save detailed results
        df.to_csv(output_dir / 'detailed_results.csv', index=False)

        # Save summary statistics
        summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)

        # Create formatted summary table for paper
        pivot_df = summary_df.pivot(index='metric', columns='method', values='mean_pm_std')
        pivot_df.to_csv(output_dir / 'formatted_summary_table.csv')

        logger.info(f"Results tables saved to {output_dir}")

    def create_comprehensive_report(self, output_dir: Path) -> None:
        """Create comprehensive analysis report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating comprehensive report in {output_dir}")

        # Generate all visualizations
        self.plot_training_curves(output_dir)
        self.plot_runtime_comparison(output_dir)
        self.plot_method_comparison_boxplots(output_dir)
        self.save_results_tables(output_dir)

        # Create summary report
        self._create_summary_report(output_dir)

        logger.info("Comprehensive report created successfully!")

    def _create_summary_report(self, output_dir: Path) -> None:
        """Create a text summary report."""
        _, summary_df = self.create_results_table()

        with open(output_dir / 'summary_report.txt', 'w') as f:
            f.write("ICLR Experiment Results Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write("Configuration:\n")
            f.write(f"- Masking rate: {self.results.get('config', {}).get('masking_rate', 'N/A')}\n")
            f.write(f"- Number of test instances: {len(self.results['test_results'])}\n")
            f.write(f"- Pretraining time: {self.results.get('pretraining_time', 'N/A'):.2f} seconds\n")
            f.write(f"- Total experiment time: {self.results.get('total_time', 'N/A'):.2f} seconds\n\n")

            f.write("Performance Summary (Mean ± Std):\n")
            f.write("-" * 40 + "\n")

            # Group by metric for easier reading
            for metric in summary_df['metric'].unique():
                f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                metric_data = summary_df[summary_df['metric'] == metric]
                for _, row in metric_data.iterrows():
                    f.write(f"  {row['method']}: {row['mean_pm_std']}\n")

            f.write(f"\nGenerated files:\n")
            f.write(f"- training_curves.png\n")
            f.write(f"- runtime_comparison.png\n")
            f.write(f"- method_comparison_boxplots.png\n")
            f.write(f"- detailed_results.csv\n")
            f.write(f"- summary_statistics.csv\n")
            f.write(f"- formatted_summary_table.csv\n")


def main():
    """Test visualization with example results."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate ICLR experiment visualizations')
    parser.add_argument('results_file', help='Path to results JSON file')
    parser.add_argument('--output-dir', default='visualization_output',
                       help='Output directory for plots and tables')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    analyzer = ICLRResultsAnalyzer(args.results_file)
    analyzer.create_comprehensive_report(args.output_dir)

    print(f"Visualization complete! Check {args.output_dir} for results.")


if __name__ == "__main__":
    main()