#!/usr/bin/env python3
"""Unified experiment runner for single and multi-instance imputation experiments."""

import argparse
import logging
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import asdict

from config import ExperimentConfig, InstanceConfig, ModelConfig, TrainingConfig
from iclr_data_generator import ICLRDataGenerator, ICLRDatasetConfig
from imputer import DataConverter, MultiVariableImputer, ImputerTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Unified runner for imputation experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.trainer = None
        self.results = {}
        
        # Create output directories
        self.output_dir = config.output_dir
        self.plots_dir = self.output_dir / "plots"
        self.models_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        
        for dir_path in [self.plots_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config.save_to_file(self.output_dir / "config.json")
        
    def generate_data(self) -> None:
        """Generate data for all instances."""
        logger.info("Generating data for all instances...")
        
        for i, instance_config in enumerate(self.config.instances):
            instance_data_dir = self.config.get_instance_data_dir(i)
            instance_data_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Generating data for instance {i}...")
            
            # Convert to ICLR data config - filter to only accepted parameters
            instance_dict = asdict(instance_config)
            iclr_params = {
                'K': instance_dict['K'],
                'I': instance_dict['I'], 
                'J': instance_dict['J'],
                'D': instance_dict['D'],
                'C': instance_dict['C'],
                'max_pairs_per_tied_group': instance_dict['max_pairs_per_tied_group'],
                'min_group_size': instance_dict['min_group_size'],
                'max_group_size': instance_dict['max_group_size'],
                'sigma_annotator': instance_dict['sigma_annotator'],
                'sigma_measurement': instance_dict['sigma_measurement'],
                'alpha_dirichlet': instance_dict['alpha_dirichlet'],
                'temperature': instance_dict['temperature'],
                'train_fraction': self.config.train_fraction,
                'test_fraction': self.config.test_fraction
            }
            iclr_config = ICLRDatasetConfig(**iclr_params)
            
            # Generate data
            generator = ICLRDataGenerator()
            dataset = generator.generate_dataset(iclr_config)
            
            # Save data
            generator.save_dataset(dataset, instance_data_dir, "iclr_complete")
            logger.info(f"Instance {i} data saved to {instance_data_dir}")
            
        logger.info("Data generation completed!")
    
    def load_instance_data(self, instance_idx: int) -> Tuple[Dict, Dict]:
        """Load train/test data for specific instance."""
        instance_data_dir = self.config.get_instance_data_dir(instance_idx)
        train_path = instance_data_dir / "iclr_complete_train.json"
        test_path = instance_data_dir / "iclr_complete_test.json"
        
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Data not found for instance {instance_idx}. Run data generation first.")
        
        # Use first instance config for converter parameters
        converter = self._get_data_converter()
        train_data = converter.load_training_data(str(train_path))
        test_data = converter.load_training_data(str(test_path))
        
        return train_data, test_data
    
    def _get_data_converter(self) -> DataConverter:
        """Get data converter using first instance config."""
        instance_config = self.config.instances[0]
        return DataConverter(
            num_attributes=instance_config.I,
            num_annotators=instance_config.J,
            num_items=instance_config.K,
            num_likert_classes=instance_config.C,
            max_rank_size=2  # Always pairwise
        )
    
    def _initialize_model(self) -> None:
        """Initialize model and trainer."""
        instance_config = self.config.instances[0]
        model_config = self.config.model_config
        training_config = self.config.training_config
        
        self.model = MultiVariableImputer(
            num_attributes=instance_config.I,
            num_annotators=instance_config.J,
            num_items=instance_config.K,
            num_likert_classes=instance_config.C,
            max_rank_size=2,
            encoder_layers_num=model_config.encoder_layers,
            attention_heads=model_config.attention_heads,
            embedding_dim=model_config.embedding_dim,
            dropout=model_config.dropout,
            embedding_type=model_config.embedding_type,
            device=self.config.device
        )
        
        self.trainer = ImputerTrainer(
            self.model,
            learning_rate=training_config.learning_rate,
            device=self.config.device,
            embedding_anchor_reg=training_config.embedding_anchor_reg
        )
    
    def run_single_instance(self) -> Dict[str, Any]:
        """Run single instance experiment."""
        logger.info("Running single instance experiment...")
        
        # Generate data if needed - check for actual data files, not just directory
        instance_data_dir = self.config.get_instance_data_dir(0)
        train_path = instance_data_dir / "iclr_complete_train.json"
        test_path = instance_data_dir / "iclr_complete_test.json"
        
        if not train_path.exists() or not test_path.exists():
            self.generate_data()
        
        # Load data
        train_data, test_data = self.load_instance_data(0)
        
        # Initialize model
        self._initialize_model()
        converter = self._get_data_converter()
        
        # Prepare data
        rating_variables, ranking_variables = converter.create_variables_from_actual_data(train_data, test_data)
        rating_data, ranking_data = converter.process_training_data(train_data)
        test_rating_data, test_ranking_data = converter.process_training_data(test_data)
        
        # Create batches
        logger.info(f"Masking rate: {self.config.training_config.masking_rate}")
        
        train_batch = converter.create_batch(
            rating_variables, ranking_variables, rating_data, ranking_data, 
            masking_rate=self.config.training_config.masking_rate
        )
        
        test_batch = converter.create_batch(
            rating_variables, ranking_variables, test_rating_data, test_ranking_data,
            mode="test", masking_rate=self.config.training_config.masking_rate
        )
        
        # Training loop
        results = self._train_instance(
            train_batch, test_batch, test_data, converter, instance_idx=0
        )
        
        # Save results
        self._save_single_instance_results(results)
        
        return results
    
    def run_multi_instance(self) -> Dict[str, Any]:
        """Run multi-instance experiment."""
        logger.info("Running multi-instance experiment...")
        
        # Generate data if needed - check for actual data files
        data_exists = True
        for i in range(self.config.num_instances):
            instance_data_dir = self.config.get_instance_data_dir(i)
            train_path = instance_data_dir / "iclr_complete_train.json"
            test_path = instance_data_dir / "iclr_complete_test.json"
            if not train_path.exists() or not test_path.exists():
                data_exists = False
                break
        
        if not data_exists:
            self.generate_data()
        
        # Initialize model once
        self._initialize_model()
        converter = self._get_data_converter()
        
        # Prepare test instances (for evaluation throughout training)
        test_instances = []
        for test_idx in self.config.test_instance_indices:
            test_train_data, test_test_data = self.load_instance_data(test_idx)
            
            # For test instances: combine ALL data (train + test portions) since entire instance is held out
            full_test_data = {'ratings': test_train_data['ratings'] + test_test_data['ratings'],
                             'pairwise_rankings': test_train_data['pairwise_rankings'] + test_test_data['pairwise_rankings']}
            
            test_rating_variables, test_ranking_variables = converter.create_variables_from_actual_data(full_test_data, full_test_data)
            test_rating_data, test_ranking_data = converter.process_training_data(full_test_data)
            
            # Use full test instance data for evaluation
            test_batch = converter.create_batch(
                test_rating_variables, test_ranking_variables, test_rating_data, test_ranking_data,
                mode="test", masking_rate=self.config.training_config.masking_rate
            )
            test_instances.append((test_batch, full_test_data, test_idx))
        
        # Sequential training on train instances
        all_results = {
            'train_losses': {},
            'test_losses': {'epoch': [], 'test_rating_loss': [], 'test_ranking_loss': []},
            'instance_boundaries': [],
            'train_instances': self.config.train_instance_indices,
            'test_instances': self.config.test_instance_indices
        }
        
        # Set up global test losses collection for plotting
        self._global_test_losses = all_results['test_losses']
        
        global_epoch = 0
        
        for train_idx in self.config.train_instance_indices:
            logger.info(f"Training on instance {train_idx}...")
            
            # Load training instance data
            train_data, test_data = self.load_instance_data(train_idx)
            rating_variables, ranking_variables = converter.create_variables_from_actual_data(train_data, test_data)
            rating_data, ranking_data = converter.process_training_data(train_data)
            test_rating_data, test_ranking_data = converter.process_training_data(test_data)
            
            # Create batches
            train_batch = converter.create_batch(
                rating_variables, ranking_variables, rating_data, ranking_data,
                masking_rate=self.config.training_config.masking_rate
            )
            
            test_batch = converter.create_batch(
                rating_variables, ranking_variables, test_rating_data, test_ranking_data,
                mode="test", masking_rate=self.config.training_config.masking_rate
            )
            
            # Train on this instance
            instance_results = self._train_instance(
                train_batch, test_batch, test_data, converter, 
                instance_idx=train_idx, global_epoch_offset=global_epoch,
                test_instances=test_instances
            )
            
            # Store results
            all_results['train_losses'][train_idx] = instance_results['train_losses']
            
            # Update global tracking
            global_epoch += self.config.training_config.epochs
            all_results['instance_boundaries'].append(global_epoch)
            
            logger.info(f"Completed training on instance {train_idx}")
        
        # Final evaluation on test instances
        final_test_results = {}
        for test_batch, test_data, test_idx in test_instances:
            test_eval = self.trainer.evaluate_with_test_data(
                test_batch, test_data, converter, 
                masking_rate=self.config.training_config.masking_rate
            )
            final_test_results[test_idx] = test_eval
        
        all_results['final_test_results'] = final_test_results
        
        # Save results
        self._save_multi_instance_results(all_results)
        
        return all_results
    
    def _train_instance(
        self, 
        train_batch: Dict, 
        test_batch: Dict, 
        test_data: Dict,
        converter: DataConverter,
        instance_idx: int,
        global_epoch_offset: int = 0,
        test_instances: Optional[List] = None
    ) -> Dict[str, Any]:
        """Train on single instance."""
        
        train_losses = {
            'epoch': [], 'total_loss': [], 'rating_loss': [], 'ranking_loss': []
        }
        
        test_losses = {
            'epoch': [], 'test_rating_loss': [], 'test_ranking_loss': []
        }
        
        # Training loop
        for epoch in tqdm(range(self.config.training_config.epochs), 
                         desc=f"Training Instance {instance_idx}"):
            
            losses = self.trainer.train_step(train_batch)
            
            # Record training losses
            train_losses['epoch'].append(global_epoch_offset + epoch)
            for key in ['total_loss', 'rating_loss', 'ranking_loss']:
                train_losses[key].append(losses[key])
            
            # Evaluate periodically
            if epoch % self.config.training_config.evaluation_frequency == 0:
                
                # Evaluate on instance test set
                test_eval = self.trainer.evaluate_with_test_data(
                    test_batch, test_data, converter,
                    masking_rate=self.config.training_config.masking_rate, verbose=False
                )
                test_losses['epoch'].append(global_epoch_offset + epoch)
                test_losses['test_rating_loss'].append(test_eval['test_rating_loss'])
                test_losses['test_ranking_loss'].append(test_eval['test_ranking_loss'])
                
                # For multi-instance, also evaluate on held-out test instances
                if test_instances:
                    avg_test_rating_loss = 0.0
                    avg_test_ranking_loss = 0.0
                    
                    for test_batch_i, test_data_i, _ in test_instances:
                        test_eval_i = self.trainer.evaluate_with_test_data(
                            test_batch_i, test_data_i, converter,
                            masking_rate=self.config.training_config.masking_rate, verbose=False
                        )
                        avg_test_rating_loss += test_eval_i['test_rating_loss']
                        avg_test_ranking_loss += test_eval_i['test_ranking_loss']
                    
                    avg_test_rating_loss /= len(test_instances)
                    avg_test_ranking_loss /= len(test_instances)
                    
                    # Store in global results (will be handled by caller)
                    if hasattr(self, '_global_test_losses'):
                        self._global_test_losses['epoch'].append(global_epoch_offset + epoch)
                        self._global_test_losses['test_rating_loss'].append(avg_test_rating_loss)
                        self._global_test_losses['test_ranking_loss'].append(avg_test_ranking_loss)
                
                # Print training losses at evaluation frequency
                logger.info(f"Instance {instance_idx}, Epoch {epoch}: "
                           f"Total={losses['total_loss']:.4f}, "
                           f"Rating={losses['rating_loss']:.4f}, "
                           f"Ranking={losses['ranking_loss']:.4f}")
                
                # Print full test evaluation results like the old version
                logger.info(f"TEST LOSS & METRICS: {test_eval}")
        
        # Final evaluation
        final_test_eval = self.trainer.evaluate_with_test_data(
            test_batch, test_data, converter,
            masking_rate=self.config.training_config.masking_rate
        )
        
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_test_eval': final_test_eval,
            'instance_idx': instance_idx
        }
    
    def _save_single_instance_results(self, results: Dict[str, Any]) -> None:
        """Save single instance results and create plots."""
        
        # Save raw results
        results_path = self.results_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save model
        model_path = self.models_dir / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Create plots
        self._create_single_instance_plots(results)
        
        # Create table
        self._create_results_table(results['final_test_eval'])
        
        logger.info(f"Single instance results saved to {self.results_dir}")
    
    def _save_multi_instance_results(self, results: Dict[str, Any]) -> None:
        """Save multi-instance results and create plots."""
        
        # Save raw results
        results_path = self.results_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save model
        model_path = self.models_dir / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Create plots
        self._create_multi_instance_plots(results)
        
        # Create table (average over test instances)
        if results['final_test_results']:
            avg_results = self._average_test_results(results['final_test_results'])
            self._create_results_table(avg_results)
        
        logger.info(f"Multi-instance results saved to {self.results_dir}")
    
    def _create_single_instance_plots(self, results: Dict[str, Any]) -> None:
        """Create plots for single instance experiment."""
        
        train_losses = results['train_losses']
        test_losses = results['test_losses']
        
        # Training plot - single comprehensive plot (left plot was sufficient)
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses['epoch'], train_losses['total_loss'], 'b-', label='Total', linewidth=2)
        plt.plot(train_losses['epoch'], train_losses['rating_loss'], 'g--', label='Rating', linewidth=2)
        plt.plot(train_losses['epoch'], train_losses['ranking_loss'], 'r--', label='Ranking', linewidth=2)
        
        plt.title(f'Training Log Loss (Masking Rate {self.config.training_config.masking_rate:.1f})')
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.plots_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Test plot
        if len(test_losses['epoch']) > 0:
            plt.figure(figsize=(8, 6))
            plt.plot(test_losses['epoch'], test_losses['test_rating_loss'], 
                    'b-o', label='Rating Test Log Loss', markersize=4, linewidth=2)
            plt.plot(test_losses['epoch'], test_losses['test_ranking_loss'], 
                    'r-s', label='Ranking Test Log Loss', markersize=4, linewidth=2)
            
            plt.title(f'Test Log Loss (Masking Rate {self.config.training_config.masking_rate:.1f})')
            plt.xlabel('Epoch')
            plt.ylabel('Log Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(self.plots_dir / 'test_loss.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Single instance plots saved to {self.plots_dir}")
    
    def _create_multi_instance_plots(self, results: Dict[str, Any]) -> None:
        """Create segmented plots for multi-instance experiment."""
        
        # Combine all training losses with instance boundaries
        all_epochs = []
        all_total_loss = []
        all_rating_loss = []
        all_ranking_loss = []
        
        for train_idx in results['train_instances']:
            instance_results = results['train_losses'][train_idx]
            all_epochs.extend(instance_results['epoch'])
            all_total_loss.extend(instance_results['total_loss'])
            all_rating_loss.extend(instance_results['rating_loss'])
            all_ranking_loss.extend(instance_results['ranking_loss'])
        
        # Training plot with instance boundaries
        plt.figure(figsize=(12, 6))
        plt.plot(all_epochs, all_total_loss, 'b-', label='Total', linewidth=2)
        plt.plot(all_epochs, all_rating_loss, 'g--', label='Rating', linewidth=2)
        plt.plot(all_epochs, all_ranking_loss, 'r--', label='Ranking', linewidth=2)
        
        # Add vertical lines for instance boundaries
        for boundary in results['instance_boundaries'][:-1]:  # Skip the last boundary
            plt.axvline(x=boundary, color='gray', linestyle=':', alpha=0.7)
        
        plt.title(f'Multi-Instance Training Log Loss (Masking Rate {self.config.training_config.masking_rate:.1f})')
        plt.xlabel('Global Epoch')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add instance labels
        for i, (start, end) in enumerate(zip([0] + results['instance_boundaries'][:-1], results['instance_boundaries'])):
            mid = (start + end) / 2
            plt.text(mid, plt.ylim()[1] * 0.9, f'Inst {results["train_instances"][i]}', 
                    ha='center', fontsize=10, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'multi_instance_training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Test performance over instances (if available)
        if 'test_losses' in results and len(results['test_losses']['epoch']) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(results['test_losses']['epoch'], results['test_losses']['test_rating_loss'], 
                    'b-o', label='Rating Test Log Loss', markersize=3, linewidth=2)
            plt.plot(results['test_losses']['epoch'], results['test_losses']['test_ranking_loss'], 
                    'r-s', label='Ranking Test Log Loss', markersize=3, linewidth=2)
            
            # Add vertical lines for instance boundaries
            for boundary in results['instance_boundaries'][:-1]:
                plt.axvline(x=boundary, color='gray', linestyle=':', alpha=0.7)
            
            plt.title(f'Test Performance on Held-out Instances (Masking Rate {self.config.training_config.masking_rate:.1f})')
            plt.xlabel('Global Epoch')
            plt.ylabel('Log Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(self.plots_dir / 'multi_instance_test_loss.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Multi-instance plots saved to {self.plots_dir}")
    
    def _create_results_table(self, test_results: Dict[str, Any]) -> None:
        """Create results table with proper breakdown: (masked/observed/all) × (rating/ranking/overall)."""
        
        # Get detailed breakdown from evaluation
        breakdown_results = self._get_detailed_breakdown(test_results)
        
        # Create table with proper structure
        rows = ['Masked', 'Observed', 'All']
        columns = ['Rating Loss', 'Rating Accuracy', 'Ranking Loss', 'Ranking Accuracy', 'Overall Loss', 'Overall Accuracy']
        
        table_data = {}
        for col in columns:
            table_data[col] = []
        
        for row in rows:
            # For now, use available data until we implement detailed breakdown
            if row == 'All':
                table_data['Rating Loss'].append(f"{test_results.get('test_rating_loss', 0.0):.4f}")
                table_data['Rating Accuracy'].append(f"{test_results.get('rating_accuracy', 0.0):.4f}" if test_results.get('rating_accuracy') is not None else "N/A")
                table_data['Ranking Loss'].append(f"{test_results.get('test_ranking_loss', 0.0):.4f}")
                table_data['Ranking Accuracy'].append(f"{test_results.get('pairwise_accuracy', 0.0):.4f}" if test_results.get('pairwise_accuracy') is not None else "N/A")
                table_data['Overall Loss'].append(f"{test_results.get('total_test_loss', 0.0):.4f}")
                overall_acc = (test_results.get('rating_accuracy', 0.0) + test_results.get('pairwise_accuracy', 0.0)) / 2 if test_results.get('rating_accuracy') is not None and test_results.get('pairwise_accuracy') is not None else None
                table_data['Overall Accuracy'].append(f"{overall_acc:.4f}" if overall_acc is not None else "N/A")
            else:
                # Placeholder for masked/observed breakdown - needs implementation
                for col in columns:
                    table_data[col].append("TBD")
        
        # Add row labels
        table_data['Row'] = rows
        
        # Save as JSON
        table_path = self.results_dir / "results_table.json"
        with open(table_path, 'w') as f:
            json.dump(table_data, f, indent=2)
        
        logger.info(f"Results breakdown table saved to {table_path}")
        logger.info("Note: Masked/Observed breakdown not yet implemented - showing 'All' data only")
    
    def _get_detailed_breakdown(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed breakdown by masked/observed status. TODO: Implement."""
        # This would need to be implemented in the evaluation method
        # to track which predictions were on masked vs observed variables
        return test_results
    
    def _average_test_results(self, test_results: Dict[int, Dict]) -> Dict[str, float]:
        """Average results across test instances."""
        
        if not test_results:
            return {}
        
        # Get all metrics from first instance
        metrics = list(test_results[list(test_results.keys())[0]].keys())
        
        avg_results = {}
        for metric in metrics:
            values = [test_results[idx][metric] for idx in test_results.keys() if metric in test_results[idx]]
            if values:
                avg_results[metric] = float(np.mean(values))
        
        return avg_results
    
    def run(self) -> Dict[str, Any]:
        """Run experiment based on configuration."""
        
        if self.config.experiment_type == "single_instance":
            return self.run_single_instance()
        elif self.config.experiment_type == "multi_instance":
            return self.run_multi_instance()
        else:
            raise ValueError(f"Unknown experiment type: {self.config.experiment_type}")

def main():
    parser = argparse.ArgumentParser(description='Run imputation experiments')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to experiment configuration file')
    parser.add_argument('--generate_only', action='store_true',
                       help='Only generate data, do not run experiments')
    
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig.load_from_file(args.config)
    logger.info(f"Loaded configuration: {config.experiment_type}")
    
    # Create runner
    runner = ExperimentRunner(config)
    
    if args.generate_only:
        runner.generate_data()
        logger.info("Data generation completed!")
    else:
        results = runner.run()
        logger.info("Experiment completed successfully!")

if __name__ == "__main__":
    main()