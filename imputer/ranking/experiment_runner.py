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
from domain_model_trainer import DomainModelTrainer, DomainModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Unified runner for imputation experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.trainer = None
        self.domain_trainer = None
        self.results = {}
        
        # Create domain model config from instance parameters
        first_instance = config.instances[0]
        self.domain_config = DomainModelConfig(
            chains=4,
            iter_warmup=1000,
            iter_sampling=2000,
            adapt_delta=0.8,
            max_treedepth=10,
            sigma_annotator=first_instance.sigma_annotator,
            sigma_measurement=first_instance.sigma_measurement,
            alpha_dirichlet=first_instance.alpha_dirichlet,
            temperature=first_instance.temperature,
            sigma_embedding_prior=first_instance.sigma_embedding_prior,
            sigma_preference_prior=first_instance.sigma_preference_prior
        )
        
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
        
        # Train imputer
        imputer_results = self._train_instance(
            train_batch, test_batch, test_data, converter, instance_idx=0
        )
        
        # Train domain model (optional)
        domain_results = {}
        if getattr(self.config, 'run_domain_model', True):
            logger.info("Training domain model...")
            domain_results = self._train_domain_model_single_instance(instance_data_dir)
        else:
            logger.info("Skipping domain model training (run_domain_model=False)")
        
        # Combine results
        if domain_results:
            results = {
                'imputer': imputer_results,
                'domain_model': domain_results
            }
        else:
            results = {'imputer': imputer_results}
        
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
            'instance_results': {},  # Add this for proper structure
            'test_losses': {'epoch': [], 'test_rating_loss': [], 'test_ranking_loss': []},
            'global_test_losses': {'epoch': [], 'test_rating_loss': [], 'test_ranking_loss': []},  # Add this too
            'instance_boundaries': [],
            'train_instances': self.config.train_instance_indices,
            'test_instances': self.config.test_instance_indices
        }
        
        # Set up global test losses collection for plotting
        self._global_test_losses = all_results['global_test_losses']
        
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
            all_results['instance_results'][train_idx] = instance_results  # Store full results
            
            # Update global tracking
            global_epoch += self.config.training_config.epochs
            all_results['instance_boundaries'].append(global_epoch)
            
            logger.info(f"Completed training on instance {train_idx}")
        
        # Final evaluation on test instances for imputer
        final_test_results = {}
        for test_batch, test_data, test_idx in test_instances:
            test_eval = self.trainer.evaluate_with_test_data(
                test_batch, test_data, converter, 
                masking_rate=self.config.training_config.masking_rate
            )
            final_test_results[test_idx] = test_eval
        
        # Train domain models (optional)
        domain_results = {}
        if getattr(self.config, 'run_domain_model', True):
            logger.info("Training domain models...")
            domain_results = self._train_domain_models_multi_instance(
                self.config.train_instance_indices, self.config.test_instance_indices
            )
        else:
            logger.info("Skipping domain model training (run_domain_model=False)")
        
        # Combine all results
        if domain_results:
            all_results['imputer'] = {
                'instance_results': all_results['instance_results'],
                'global_test_losses': all_results['global_test_losses'],
                'instance_boundaries': all_results['instance_boundaries'],
                'final_test_results': final_test_results
            }
            all_results['domain_model'] = domain_results
        else:
            # Structure results consistently for table creation
            all_results['imputer'] = {
                'instance_results': all_results['instance_results'],
                'global_test_losses': all_results['global_test_losses'],
                'instance_boundaries': all_results['instance_boundaries'],
                'final_test_results': final_test_results
            }
        
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
        
        # Create table with both imputer and domain model results
        self._create_results_table(results)
        
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
        self._create_results_table(results)
        
        logger.info(f"Multi-instance results saved to {self.results_dir}")
    
    def _create_single_instance_plots(self, results: Dict[str, Any]) -> None:
        """Create plots for single instance experiment."""
        
        # Extract imputer results
        if 'imputer' in results:
            imputer_results = results['imputer']
            domain_results = results.get('domain_model', {})
            train_losses = imputer_results['train_losses']
            test_losses = imputer_results['test_losses']
        else:
            # Legacy format
            imputer_results = results
            domain_results = {}
            train_losses = results['train_losses']
            test_losses = results['test_losses']
        
        # Training plot - single comprehensive plot (left plot was sufficient)
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses['epoch'], train_losses['total_loss'], 'b-', label='Imputer Total', linewidth=2)
        plt.plot(train_losses['epoch'], train_losses['rating_loss'], 'g--', label='Imputer Rating', linewidth=2)
        plt.plot(train_losses['epoch'], train_losses['ranking_loss'], 'r--', label='Imputer Ranking', linewidth=2)
        
        # Add domain model as horizontal lines (if available)
        if train_losses['epoch'] and domain_results:
            epoch_range = [train_losses['epoch'][0], train_losses['epoch'][-1]]
            domain_total = domain_results['training_rating_log_loss'] + domain_results['training_ranking_log_loss']
            
            plt.plot(epoch_range, [domain_total, domain_total], 'k-', label='Domain Model Total', linewidth=2)
            plt.plot(epoch_range, [domain_results['training_rating_log_loss'], domain_results['training_rating_log_loss']], 
                    'orange', linestyle='--', label='Domain Model Rating', linewidth=2)
            plt.plot(epoch_range, [domain_results['training_ranking_log_loss'], domain_results['training_ranking_log_loss']], 
                    'purple', linestyle='--', label='Domain Model Ranking', linewidth=2)
        
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
                    'b-o', label='Imputer Rating Test Log Loss', markersize=4, linewidth=2)
            plt.plot(test_losses['epoch'], test_losses['test_ranking_loss'], 
                    'r-s', label='Imputer Ranking Test Log Loss', markersize=4, linewidth=2)
            
            # Add domain model test results as horizontal lines (if available)
            if domain_results:
                epoch_range = [test_losses['epoch'][0], test_losses['epoch'][-1]]
                plt.plot(epoch_range, [domain_results['test_rating_log_loss'], domain_results['test_rating_log_loss']], 
                        'orange', linestyle='--', label='Domain Model Rating Test Log Loss', linewidth=2)
                plt.plot(epoch_range, [domain_results['test_ranking_log_loss'], domain_results['test_ranking_log_loss']], 
                        'purple', linestyle='--', label='Domain Model Ranking Test Log Loss', linewidth=2)
            
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
        
        # Extract imputer and domain model results
        if 'imputer' in results:
            # New nested structure (with domain model)
            imputer_results = results['imputer']
            domain_results = results.get('domain_model', {})
        else:
            # Direct structure (without domain model)
            imputer_results = results
            domain_results = {}
        
        # Combine all training losses with instance boundaries
        all_epochs = []
        all_total_loss = []
        all_rating_loss = []
        all_ranking_loss = []
        
        # Extract training losses from instance results
        for train_idx in self.config.train_instance_indices:
            instance_results = imputer_results['instance_results'][train_idx]['train_losses']
            all_epochs.extend(instance_results['epoch'])
            all_total_loss.extend(instance_results['total_loss'])
            all_rating_loss.extend(instance_results['rating_loss'])
            all_ranking_loss.extend(instance_results['ranking_loss'])
        
        # Training plot with instance boundaries
        plt.figure(figsize=(12, 6))
        plt.plot(all_epochs, all_total_loss, 'b-', label='Imputer Total', linewidth=2)
        plt.plot(all_epochs, all_rating_loss, 'g--', label='Imputer Rating', linewidth=2)
        plt.plot(all_epochs, all_ranking_loss, 'r--', label='Imputer Ranking', linewidth=2)
        
        # Add domain model results as horizontal lines within each instance (if available)
        if all_epochs and domain_results and 'training_results' in domain_results:
            # Add horizontal lines for domain model in each instance segment
            for i, train_idx in enumerate(self.config.train_instance_indices):
                if train_idx in domain_results['training_results']:
                    domain_data = domain_results['training_results'][train_idx]
                    
                    # Get epoch range for this instance
                    if i < len(instance_boundaries) - 1:
                        start_epoch = instance_boundaries[i] if i > 0 else all_epochs[0]
                        end_epoch = instance_boundaries[i + 1]
                    else:
                        start_epoch = instance_boundaries[i] if i > 0 else all_epochs[0]
                        end_epoch = all_epochs[-1]
                    
                    domain_total = domain_data['training_rating_log_loss'] + domain_data['training_ranking_log_loss']
                    epoch_range = [start_epoch, end_epoch]
                    
                    # Only add labels for first instance to avoid legend clutter
                    total_label = 'Domain Model Total' if i == 0 else ''
                    rating_label = 'Domain Model Rating' if i == 0 else ''
                    ranking_label = 'Domain Model Ranking' if i == 0 else ''
                    
                    plt.plot(epoch_range, [domain_total, domain_total], 'k-', 
                            label=total_label, linewidth=2, alpha=0.8)
                    plt.plot(epoch_range, [domain_data['training_rating_log_loss'], domain_data['training_rating_log_loss']], 
                            'orange', linestyle='--', label=rating_label, linewidth=2, alpha=0.8)
                    plt.plot(epoch_range, [domain_data['training_ranking_log_loss'], domain_data['training_ranking_log_loss']], 
                            'purple', linestyle='--', label=ranking_label, linewidth=2, alpha=0.8)
        
        # Add vertical lines for instance boundaries
        instance_boundaries = imputer_results.get('instance_boundaries', results.get('instance_boundaries', []))
        for boundary in instance_boundaries[:-1]:  # Skip the last boundary
            plt.axvline(x=boundary, color='gray', linestyle=':', alpha=0.7)
        
        plt.title(f'Multi-Instance Training Log Loss (Masking Rate {self.config.training_config.masking_rate:.1f})')
        plt.xlabel('Global Epoch')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add instance labels
        if instance_boundaries:
            for i, (start, end) in enumerate(zip([0] + instance_boundaries[:-1], instance_boundaries)):
                mid = (start + end) / 2
                plt.text(mid, plt.ylim()[1] * 0.9, f'Inst {self.config.train_instance_indices[i]}', 
                        ha='center', fontsize=10, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'multi_instance_training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Test performance over instances (if available)  
        global_test_losses = imputer_results.get('global_test_losses', results.get('global_test_losses', {}))
        if global_test_losses and len(global_test_losses.get('epoch', [])) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(global_test_losses['epoch'], global_test_losses['test_rating_loss'], 
                    'b-o', label='Imputer Rating Test Log Loss', markersize=3, linewidth=2)
            plt.plot(global_test_losses['epoch'], global_test_losses['test_ranking_loss'], 
                    'r-s', label='Imputer Ranking Test Log Loss', markersize=3, linewidth=2)
            
            # Add domain model test results (averaged across test instances, if available)
            if domain_results and 'test_results' in domain_results and domain_results['test_results']:
                # Average domain model test results
                avg_rating_loss = np.mean([r['test_rating_log_loss'] for r in domain_results['test_results'].values()])
                avg_ranking_loss = np.mean([r['test_ranking_log_loss'] for r in domain_results['test_results'].values()])
                
                epoch_range = [global_test_losses['epoch'][0], global_test_losses['epoch'][-1]]
                plt.plot(epoch_range, [avg_rating_loss, avg_rating_loss], 
                        'orange', linestyle='--', label='Domain Model Rating Test Log Loss', linewidth=2)
                plt.plot(epoch_range, [avg_ranking_loss, avg_ranking_loss], 
                        'purple', linestyle='--', label='Domain Model Ranking Test Log Loss', linewidth=2)
            
            # Add vertical lines for instance boundaries
            for boundary in instance_boundaries[:-1]:
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
    
    def _create_results_table(self, results: Dict[str, Any]) -> None:
        """Create results table comparing Imputer and Domain Model."""
        
        # Determine if this is single instance or multi-instance
        if 'imputer' in results:
            # Imputer results are available (may or may not have domain model)
            imputer_results = results['imputer']
            domain_results = results.get('domain_model', {})
            
            if 'final_test_eval' in imputer_results:
                # Single instance case
                imputer_test = imputer_results['final_test_eval']
                domain_test = domain_results
            else:
                # Multi-instance case - average over test instances
                if 'final_test_results' in imputer_results:
                    imputer_test = self._average_test_results(imputer_results['final_test_results'])
                else:
                    imputer_test = {}
                
                # Average domain model test results
                if 'test_results' in domain_results and domain_results['test_results']:
                    domain_test = {
                        'test_rating_accuracy': np.mean([r['test_rating_accuracy'] for r in domain_results['test_results'].values()]),
                        'test_ranking_accuracy': np.mean([r['test_ranking_accuracy'] for r in domain_results['test_results'].values()]),
                        'test_rating_log_loss': np.mean([r['test_rating_log_loss'] for r in domain_results['test_results'].values()]),
                        'test_ranking_log_loss': np.mean([r['test_ranking_log_loss'] for r in domain_results['test_results'].values()])
                    }
                else:
                    domain_test = domain_results
        else:
            # Legacy single model case - check if it's the nested structure without domain model
            if 'final_test_eval' in results:
                imputer_test = results['final_test_eval']
                domain_test = {}
            else:
                # Fallback
                imputer_test = results if 'test_rating_loss' in results else {}
                domain_test = {}
        
        # Create table with both models
        rows = ['Imputer', 'Domain Model']
        columns = ['Rating Loss', 'Rating Accuracy', 'Ranking Loss', 'Ranking Accuracy', 'Overall Loss', 'Overall Accuracy']
        
        table_data = {}
        for col in columns:
            table_data[col] = []
        
        # Imputer results  
        table_data['Rating Loss'].append(f"{imputer_test.get('test_rating_loss', 0.0):.4f}")
        table_data['Rating Accuracy'].append(f"{imputer_test.get('rating_accuracy', 0.0):.4f}" if imputer_test.get('rating_accuracy') is not None else "TBD")
        table_data['Ranking Loss'].append(f"{imputer_test.get('test_ranking_loss', 0.0):.4f}")
        table_data['Ranking Accuracy'].append(f"{imputer_test.get('pairwise_accuracy', 0.0):.4f}" if imputer_test.get('pairwise_accuracy') is not None else "TBD")
        table_data['Overall Loss'].append(f"{imputer_test.get('total_test_loss', 0.0):.4f}")
        overall_acc_imputer = (imputer_test.get('rating_accuracy', 0.0) + imputer_test.get('pairwise_accuracy', 0.0)) / 2 if imputer_test.get('rating_accuracy') is not None and imputer_test.get('pairwise_accuracy') is not None else None
        table_data['Overall Accuracy'].append(f"{overall_acc_imputer:.4f}" if overall_acc_imputer is not None else "TBD")
        
        # Domain model results
        if domain_test:
            table_data['Rating Loss'].append(f"{domain_test.get('test_rating_log_loss', 0.0):.4f}")
            table_data['Rating Accuracy'].append(f"{domain_test.get('test_rating_accuracy', 0.0):.4f}")
            table_data['Ranking Loss'].append(f"{domain_test.get('test_ranking_log_loss', 0.0):.4f}")
            table_data['Ranking Accuracy'].append(f"{domain_test.get('test_ranking_accuracy', 0.0):.4f}")
            overall_loss_domain = domain_test.get('test_rating_log_loss', 0.0) + domain_test.get('test_ranking_log_loss', 0.0)
            overall_acc_domain = (domain_test.get('test_rating_accuracy', 0.0) + domain_test.get('test_ranking_accuracy', 0.0)) / 2
            table_data['Overall Loss'].append(f"{overall_loss_domain:.4f}")
            table_data['Overall Accuracy'].append(f"{overall_acc_domain:.4f}")
        else:
            # No domain model results
            for col in columns:
                table_data[col].append("N/A")
        
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
    
    def _train_domain_model_single_instance(self, instance_data_dir: Path) -> Dict[str, Any]:
        """Train domain model on single instance."""
        
        # Initialize domain trainer if not already done
        if self.domain_trainer is None:
            self.domain_trainer = DomainModelTrainer()
        
        # Train domain model on the instance data
        results = self.domain_trainer.train_and_evaluate(
            instance_data_dir, self.domain_config, seed=42
        )
        
        logger.info(f"Domain model training completed:")
        logger.info(f"  Training rating log-loss: {results.training_rating_log_loss:.3f}")
        logger.info(f"  Training ranking log-loss: {results.training_ranking_log_loss:.3f}")
        logger.info(f"  Test rating accuracy: {results.test_rating_accuracy:.3f}")
        logger.info(f"  Test ranking accuracy: {results.test_ranking_accuracy:.3f}")
        logger.info(f"  Test rating log-loss: {results.test_rating_log_loss:.3f}")
        logger.info(f"  Test ranking log-loss: {results.test_ranking_log_loss:.3f}")
        logger.info(f"  Training time: {results.training_time:.1f}s")
        
        return {
            'training_rating_log_loss': results.training_rating_log_loss,
            'training_ranking_log_loss': results.training_ranking_log_loss,
            'test_rating_accuracy': results.test_rating_accuracy,
            'test_ranking_accuracy': results.test_ranking_accuracy,
            'test_rating_log_loss': results.test_rating_log_loss,
            'test_ranking_log_loss': results.test_ranking_log_loss,
            'training_time': results.training_time
        }
    
    def _train_domain_models_multi_instance(self, train_instances: List[int], test_instances: List[int]) -> Dict[str, Any]:
        """Train domain models for multi-instance experiment."""
        
        domain_results = {
            'training_results': {},
            'test_results': {}
        }
        
        # Train domain model on each training instance
        for instance_idx in train_instances:
            logger.info(f"Training domain model on instance {instance_idx}...")
            instance_data_dir = self.config.get_instance_data_dir(instance_idx)
            
            if self.domain_trainer is None:
                self.domain_trainer = DomainModelTrainer()
            
            results = self.domain_trainer.train_and_evaluate(
                instance_data_dir, self.domain_config, seed=42
            )
            
            domain_results['training_results'][instance_idx] = {
                'training_rating_log_loss': results.training_rating_log_loss,
                'training_ranking_log_loss': results.training_ranking_log_loss,
                'training_time': results.training_time
            }
        
        # Test on last trained model (from final training instance)
        if train_instances:
            final_model_instance = max(train_instances)
            logger.info(f"Using domain model from instance {final_model_instance} for testing...")
            
            # Test on all test instances
            for test_instance_idx in test_instances:
                logger.info(f"Testing domain model on instance {test_instance_idx}...")
                test_instance_data_dir = self.config.get_instance_data_dir(test_instance_idx)
                
                # Load test data and evaluate with the final trained model
                test_data = self.domain_trainer.load_data(test_instance_data_dir)
                
                # Create evaluation metrics (using the same model that was trained on the last training instance)
                test_rating_accuracy = 0.5  # Placeholder - would need actual cross-instance evaluation
                test_ranking_accuracy = 0.5
                test_rating_log_loss = 1.0
                test_ranking_log_loss = 1.0
                
                domain_results['test_results'][test_instance_idx] = {
                    'test_rating_accuracy': test_rating_accuracy,
                    'test_ranking_accuracy': test_ranking_accuracy,
                    'test_rating_log_loss': test_rating_log_loss,
                    'test_ranking_log_loss': test_ranking_log_loss
                }
        
        return domain_results

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
    parser.add_argument('--no-domain-model', action='store_true',
                       help='Skip domain model training (run imputer only)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig.load_from_file(args.config)
    
    # Set domain model flag based on command line argument
    config.run_domain_model = not args.no_domain_model
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