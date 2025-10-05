"""
Partial Experiment Runner for STAN Experiments

Allows running individual parts of the experiment pipeline in isolation:
1. Data generation only
2. Pretraining only  
3. Individual strategy evaluation
4. Domain model evaluation with different configurations

Reuses existing code from new_experiment_runner.py via imports.
"""

import time
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import torch
import numpy as np
import argparse

from experiment_config import ExperimentConfig
from iclr_data_generator import ICLRDataGenerator, ICLRDatasetConfig
from imputer.data import DataConverter
from imputer.ranking_imputer import MultiVariableImputer
from imputer.multi_instance_trainer import SequentialMIT, MixedMIT, GeneralMIT
from imputer.eval import EvaluationEngine
from domain_model_trainer import DomainModelTrainer

# Import Timer and ExperimentRunner from the original file
from new_experiment_runner import Timer, ExperimentRunner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PartialExperimentRunner:
    """Partial experiment runner for isolated component testing."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Create timestamped data directory for generated data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = Path("generated_data") / f"{config.experiment_name}_{timestamp}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Results go to standard experiment_results directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        
        # Initialize components
        self.data_generator = None
        self.converter = None
        self.eval_engine = EvaluationEngine(self.config.model_config)
        
        logger.info(f"Initialized partial experiment runner: {config.experiment_name}")
        logger.info(f"Generated data directory: {self.data_dir}")
        logger.info(f"Results output directory: {self.output_dir}")

    def generate_data_only(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate training and test instances only."""
        logger.info("="*60)
        logger.info("GENERATING DATA ONLY")
        logger.info("="*60)
        
        with Timer("Data Generation") as timer:
            # Create data generator (only use parameters supported by ICLRDatasetConfig)
            data_config = ICLRDatasetConfig(
                K=self.config.data_config.K,
                I=self.config.data_config.I,
                J=self.config.data_config.J,
                C=self.config.data_config.C,
                sigma_annotator=self.config.data_config.sigma_annotator,
                sigma_measurement=self.config.data_config.sigma_measurement,
                alpha_dirichlet=self.config.data_config.alpha_dirichlet,
                temperature=self.config.data_config.temperature,
                max_pairs_per_tied_group=self.config.data_config.max_pairs_per_tied_group,
                min_group_size=self.config.data_config.min_group_size,
                max_group_size=self.config.data_config.max_group_size
            )

            self.data_generator = ICLRDataGenerator()

            # Generate instances
            all_instances = []
            for i in range(self.config.data_config.num_instances):
                dataset = self.data_generator.generate_dataset(data_config, seed=self.config.random_seed + i)
                # Convert to format expected by our system
                instance = {
                    'ratings': dataset.observed_ratings,
                    'pairwise_rankings': dataset.observed_pairwise_rankings
                }
                all_instances.append(instance)

            # Split into train/test
            num_train = int(len(all_instances) * self.config.data_config.train_test_split)
            train_instances = all_instances[:num_train]
            test_instances = all_instances[num_train:]

            # Initialize data converter
            self.converter = DataConverter(
                num_attributes=self.config.data_config.I,
                num_annotators=self.config.data_config.J,
                num_items=self.config.data_config.K,
                num_likert_classes=self.config.data_config.C,
                max_rank_size=self.config.model_config.max_rank_size
            )

            logger.info(f"Generated {len(train_instances)} train instances, {len(test_instances)} test instances")

            # Save generated data to timestamped directory
            self._save_generated_data(train_instances, test_instances, data_config)

        logger.info(f"Data generation completed in {timer.get_elapsed():.2f} seconds")
        logger.info("="*60)
        
        return train_instances, test_instances

    def run_pretraining_only(self, train_instances: List[Dict]) -> MultiVariableImputer:
        """Run pretraining only."""
        logger.info("="*60)
        logger.info("RUNNING PRETRAINING ONLY")
        logger.info("="*60)
        
        with Timer("Pretraining") as timer:
            # Create model
            model = MultiVariableImputer(
                num_attributes=self.config.data_config.I,
                num_annotators=self.config.data_config.J,
                num_items=self.config.data_config.K,
                num_likert_classes=self.config.data_config.C,
                max_rank_size=self.config.model_config.max_rank_size,
                encoder_layers_num=self.config.model_config.encoder_layers,
                attention_heads=self.config.model_config.attention_heads,
                embedding_dim=self.config.model_config.embedding_dim,
                dropout=self.config.model_config.dropout,
                embedding_type=self.config.model_config.embedding_type,
                device=self.config.device
            )

            # Create MIT trainer
            if self.config.pretraining_config.strategy == "sequential":
                mit = SequentialMIT(model, self.eval_engine, self.config.pretraining_config, self.converter, self.config.model_config)
            else:  # mixed
                mit = MixedMIT(model, self.eval_engine, self.config.pretraining_config, self.converter, self.config.model_config)

            # Run training
            training_results = mit.train_on_instances(train_instances, [])

            # Save model if requested
            model_path = None
            if self.config.save_models:
                model_path = self.output_dir / "pretrained_model.pt"
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved pretrained model to {model_path}")

        logger.info(f"Pretraining completed in {timer.get_elapsed():.2f} seconds")
        logger.info("="*60)
        
        return model

    def evaluate_single_strategy(self, strategy_name: str, test_instance: Dict, 
                               pretrained_model: Optional[MultiVariableImputer] = None,
                               instance_idx: int = 0) -> Dict[str, Any]:
        """Evaluate a single strategy on a test instance."""
        logger.info("="*60)
        logger.info(f"EVALUATING STRATEGY: {strategy_name}")
        logger.info("="*60)
        
        if strategy_name == "Pretrained_Imputer":
            if pretrained_model is None:
                raise ValueError("Pretrained_Imputer requires a pretrained model")
            return self._evaluate_pretrained_imputer(pretrained_model, test_instance, instance_idx)
        
        elif strategy_name == "Pretrain_Finetuned_Imputer":
            if pretrained_model is None:
                raise ValueError("Pretrain_Finetuned_Imputer requires a pretrained model")
            return self._evaluate_pretrain_finetuned_imputer(pretrained_model, test_instance, instance_idx)
        
        elif strategy_name == "Finetuned_Imputer":
            return self._evaluate_finetuned_imputer(test_instance, instance_idx)
        
        elif strategy_name == "Domain_Model":
            return self._evaluate_domain_model(test_instance, instance_idx)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def evaluate_domain_model_with_config(self, test_instance: Dict, 
                                        chains: int = None,
                                        iter_warmup: int = None,
                                        iter_sampling: int = None,
                                        adapt_delta: float = None,
                                        max_treedepth: int = None,
                                        instance_idx: int = 0) -> Dict[str, Any]:
        """Evaluate domain model with custom configuration."""
        logger.info("="*60)
        logger.info("EVALUATING DOMAIN MODEL WITH CUSTOM CONFIG")
        logger.info("="*60)
        
        # Use provided values or defaults from config
        chains = chains or self.config.domain_config.chains
        iter_warmup = iter_warmup or self.config.domain_config.iter_warmup
        iter_sampling = iter_sampling or self.config.domain_config.iter_sampling
        adapt_delta = adapt_delta or self.config.domain_config.adapt_delta
        max_treedepth = max_treedepth or self.config.domain_config.max_treedepth
        
        with Timer(f"Domain_Model evaluation (instance {instance_idx}, custom config)") as timer:
            # Create domain model trainer
            domain_trainer = DomainModelTrainer()

            # Prepare data config for domain model
            data_config_dict = {
                'K': self.config.data_config.K,
                'I': self.config.data_config.I,
                'J': self.config.data_config.J,
                'C': self.config.data_config.C,
                'D': self.config.model_config.embedding_dim,  # embedding dimension
                'ranking_size': self.config.model_config.max_rank_size,
                'sigma_annotator': self.config.data_config.sigma_annotator,
                'sigma_measurement': self.config.data_config.sigma_measurement,
                'alpha_dirichlet': self.config.data_config.alpha_dirichlet,
                'temperature': self.config.data_config.temperature,
                'sigma_embedding_prior': self.config.data_config.sigma_embedding_prior,
                'sigma_preference_prior': self.config.data_config.sigma_preference_prior
            }

            # Evaluate using domain model
            evaluation_results = domain_trainer.evaluate_test_instance(
                test_instance,
                data_config_dict,
                masking_rate=self.config.evaluation_config.test_masking_rate,
                chains=chains,
                iter_sampling=iter_sampling,
                iter_warmup=iter_warmup,
                adapt_delta=adapt_delta,
                max_treedepth=max_treedepth,
                seed=self.config.random_seed
            )

        result = {
            "evaluation_results": evaluation_results.__dict__,
            "wall_time": timer.get_elapsed(),
            "config_used": {
                "chains": chains,
                "iter_warmup": iter_warmup,
                "iter_sampling": iter_sampling,
                "adapt_delta": adapt_delta,
                "max_treedepth": max_treedepth
            }
        }
        
        logger.info(f"Domain model evaluation completed in {timer.get_elapsed():.2f} seconds")
        logger.info("="*60)
        
        return result

    def load_data_from_file(self, data_file: str) -> Tuple[List[Dict], List[Dict]]:
        """Load previously generated data from file."""
        logger.info(f"Loading data from {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Initialize data converter
        self.converter = DataConverter(
            num_attributes=self.config.data_config.I,
            num_annotators=self.config.data_config.J,
            num_items=self.config.data_config.K,
            num_likert_classes=self.config.data_config.C,
            max_rank_size=self.config.model_config.max_rank_size
        )
        
        # Handle both dictionary format and direct list format
        if isinstance(data, dict):
            return data.get('train_instances', []), data.get('test_instances', [])
        elif isinstance(data, list):
            # If it's a list, assume it's test instances (common for domain model evaluation)
            return [], data
        else:
            raise ValueError(f"Unexpected data format in {data_file}")

    def load_pretrained_model(self, model_file: str) -> MultiVariableImputer:
        """Load a previously trained model."""
        logger.info(f"Loading pretrained model from {model_file}")
        
        model = MultiVariableImputer(
            num_attributes=self.config.data_config.I,
            num_annotators=self.config.data_config.J,
            num_items=self.config.data_config.K,
            num_likert_classes=self.config.data_config.C,
            max_rank_size=self.config.model_config.max_rank_size,
            encoder_layers_num=self.config.model_config.encoder_layers,
            attention_heads=self.config.model_config.attention_heads,
            embedding_dim=self.config.model_config.embedding_dim,
            dropout=self.config.model_config.dropout,
            embedding_type=self.config.model_config.embedding_type,
            device=self.config.device
        )
        
        model.load_state_dict_with_warnings(torch.load(model_file, map_location=self.config.device), strict=False)
        return model

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            filename = f"{self.config.experiment_name}_partial_results.json"
        
        results_file = self.output_dir / filename
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")

    # Reuse methods from ExperimentRunner
    def _evaluate_pretrained_imputer(self, model: MultiVariableImputer, test_instance: Dict, instance_idx: int) -> Dict[str, Any]:
        """Strategy 1: Direct evaluation of pretrained model."""
        with Timer(f"Pretrained_Imputer evaluation (instance {instance_idx})") as timer:
            test_variables = self.converter.create_variables(test_instance)

            evaluation_results = self.eval_engine.evaluate_model(
                model=model,
                variables=test_variables,
                masking_rate=self.config.evaluation_config.test_masking_rate,
                converter=self.converter,
                device=self.config.device
            )

        return {
            "evaluation_results": evaluation_results.__dict__,
            "wall_time": timer.get_elapsed()
        }

    def _evaluate_pretrain_finetuned_imputer(self, pretrained_model: MultiVariableImputer, test_instance: Dict, instance_idx: int) -> Dict[str, Any]:
        """Strategy 2: Pretrained model finetuned on test instance."""
        with Timer(f"Pretrain_Finetuned_Imputer evaluation (instance {instance_idx})") as timer:
            # Create copy of pretrained model for finetuning
            model = MultiVariableImputer(
                num_attributes=self.config.data_config.I,
                num_annotators=self.config.data_config.J,
                num_items=self.config.data_config.K,
                num_likert_classes=self.config.data_config.C,
                max_rank_size=self.config.model_config.max_rank_size,
                encoder_layers_num=self.config.model_config.encoder_layers,
                attention_heads=self.config.model_config.attention_heads,
                embedding_dim=self.config.model_config.embedding_dim,
                dropout=self.config.model_config.dropout,
                embedding_type=self.config.model_config.embedding_type,
                device=self.config.device
            )
            model.load_state_dict(pretrained_model.state_dict())

            # Finetune using GeneralMIT
            mit = GeneralMIT(model, self.eval_engine, self.config.finetuning_config, self.converter, self.config.model_config)
            finetuning_results = mit.finetune_on_instance(
                model, test_instance,
                full_test_instances=None,
                eval_config=self.config.evaluation_config
            )

            # Save finetuned model if requested
            model_path = None
            if self.config.save_models:
                model_path = self.output_dir / f"pretrain_finetuned_model_instance_{instance_idx}.pt"
                torch.save(model.state_dict(), model_path)

        return {
            "finetuning_results": finetuning_results["finetuning_results"] if self.config.save_training_histories else "saved_separately",
            "callback_results": finetuning_results.get("callback_results", []),
            "evaluation_results": finetuning_results["final_evaluation"].__dict__,
            "model_path": str(model_path) if model_path else None,
            "wall_time": timer.get_elapsed()
        }

    def _evaluate_finetuned_imputer(self, test_instance: Dict, instance_idx: int) -> Dict[str, Any]:
        """Strategy 3: Fresh model trained only on test instance."""
        with Timer(f"Finetuned_Imputer evaluation (instance {instance_idx})") as timer:
            # Create fresh model
            model = MultiVariableImputer(
                num_attributes=self.config.data_config.I,
                num_annotators=self.config.data_config.J,
                num_items=self.config.data_config.K,
                num_likert_classes=self.config.data_config.C,
                max_rank_size=self.config.model_config.max_rank_size,
                encoder_layers_num=self.config.model_config.encoder_layers,
                attention_heads=self.config.model_config.attention_heads,
                embedding_dim=self.config.model_config.embedding_dim,
                dropout=self.config.model_config.dropout,
                embedding_type=self.config.model_config.embedding_type,
                device=self.config.device
            )

            # Train from scratch using GeneralMIT
            mit = GeneralMIT(model, self.eval_engine, self.config.finetuning_config, self.converter, self.config.model_config)
            training_results = mit.finetune_on_instance(
                model, test_instance,
                full_test_instances=None,
                eval_config=self.config.evaluation_config
            )

            # Save fresh model if requested
            model_path = None
            if self.config.save_models:
                model_path = self.output_dir / f"fresh_model_instance_{instance_idx}.pt"
                torch.save(model.state_dict(), model_path)

        return {
            "training_results": training_results["finetuning_results"] if self.config.save_training_histories else "saved_separately",
            "callback_results": training_results.get("callback_results", []),
            "evaluation_results": training_results["final_evaluation"].__dict__,
            "model_path": str(model_path) if model_path else None,
            "wall_time": timer.get_elapsed()
        }

    def _evaluate_domain_model(self, test_instance: Dict, instance_idx: int) -> Dict[str, Any]:
        """Strategy 4: Domain model with multiple sample counts."""
        results = {}

        for sample_count in self.config.domain_config.sample_counts:
            with Timer(f"Domain_Model evaluation (instance {instance_idx}, {sample_count} samples)") as timer:
                # Create domain model trainer
                domain_trainer = DomainModelTrainer()

                # Prepare data config for domain model
                data_config_dict = {
                    'K': self.config.data_config.K,
                    'I': self.config.data_config.I,
                    'J': self.config.data_config.J,
                    'C': self.config.data_config.C,
                    'D': self.config.model_config.embedding_dim,  # embedding dimension
                    'ranking_size': self.config.model_config.max_rank_size,
                    'sigma_annotator': self.config.data_config.sigma_annotator,
                    'sigma_measurement': self.config.data_config.sigma_measurement,
                    'alpha_dirichlet': self.config.data_config.alpha_dirichlet,
                    'temperature': self.config.data_config.temperature,
                    'sigma_embedding_prior': self.config.data_config.sigma_embedding_prior,
                    'sigma_preference_prior': self.config.data_config.sigma_preference_prior
                }

                # Evaluate using domain model
                evaluation_results = domain_trainer.evaluate_test_instance(
                    test_instance,
                    data_config_dict,
                    masking_rate=self.config.evaluation_config.test_masking_rate,
                    chains=self.config.domain_config.chains,
                    iter_sampling=sample_count,
                    iter_warmup=self.config.domain_config.iter_warmup,
                    adapt_delta=self.config.domain_config.adapt_delta,
                    max_treedepth=self.config.domain_config.max_treedepth,
                    seed=self.config.random_seed
                )

            results[f"samples_{sample_count}"] = {
                "evaluation_results": evaluation_results.__dict__,
                "wall_time": timer.get_elapsed()
            }

        return results

    def _save_generated_data(self, train_instances: List[Dict], test_instances: List[Dict], data_config):
        """Save generated data to timestamped directory."""
        try:
            # Save train instances
            train_file = self.data_dir / "train_instances.json"
            with open(train_file, 'w') as f:
                json.dump(train_instances, f, indent=2)

            # Save test instances
            test_file = self.data_dir / "test_instances.json"
            with open(test_file, 'w') as f:
                json.dump(test_instances, f, indent=2)

            # Save data generation config
            config_file = self.data_dir / "data_config.json"
            with open(config_file, 'w') as f:
                json.dump(data_config.__dict__, f, indent=2)

            # Save experiment metadata
            metadata_file = self.data_dir / "experiment_metadata.json"
            metadata = {
                "experiment_name": self.config.experiment_name,
                "timestamp": datetime.now().isoformat(),
                "num_train_instances": len(train_instances),
                "num_test_instances": len(test_instances),
                "data_config": data_config.__dict__
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Generated data saved to: {self.data_dir}")

        except Exception as e:
            logger.error(f"Failed to save generated data: {e}")


def main():
    """Main entry point for partial experiment runner."""
    parser = argparse.ArgumentParser(description="Run partial ranking imputation experiments")
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment configuration file")
    
    # Operation selection
    parser.add_argument("--operation", type=str, required=True,
                       choices=["generate_data", "pretrain", "evaluate_strategy", "evaluate_domain"],
                       help="Operation to perform")
    
    # Data loading
    parser.add_argument("--data_file", type=str,
                       help="Path to previously generated data file (for non-generate operations)")
    parser.add_argument("--model_file", type=str,
                       help="Path to pretrained model file (for strategy evaluation)")
    
    # Strategy evaluation
    parser.add_argument("--strategy", type=str,
                       choices=["Pretrained_Imputer", "Pretrain_Finetuned_Imputer", "Finetuned_Imputer", "Domain_Model"],
                       help="Strategy to evaluate")
    parser.add_argument("--instance_idx", type=int, default=0,
                       help="Test instance index to evaluate")
    
    # Domain model custom config
    parser.add_argument("--chains", type=int,
                       help="Number of MCMC chains")
    parser.add_argument("--iter_warmup", type=int,
                       help="Number of warmup iterations")
    parser.add_argument("--iter_sampling", type=int,
                       help="Number of sampling iterations")
    parser.add_argument("--adapt_delta", type=float,
                       help="Adaptation delta for MCMC")
    parser.add_argument("--max_treedepth", type=int,
                       help="Maximum tree depth for MCMC")
    
    # Output
    parser.add_argument("--output_file", type=str,
                       help="Output filename for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig.load(args.config)
    runner = PartialExperimentRunner(config)
    
    # Execute requested operation
    if args.operation == "generate_data":
        logger.info("Generating data only...")
        train_instances, test_instances = runner.generate_data_only()
        logger.info(f"Generated {len(train_instances)} train and {len(test_instances)} test instances")
        
    elif args.operation == "pretrain":
        if args.data_file is None:
            logger.error("--data_file required for pretrain operation")
            return
        
        logger.info("Running pretraining only...")
        train_instances, test_instances = runner.load_data_from_file(args.data_file)
        pretrained_model = runner.run_pretraining_only(train_instances)
        logger.info("Pretraining completed")
        
    elif args.operation == "evaluate_strategy":
        if args.data_file is None or args.strategy is None:
            logger.error("--data_file and --strategy required for evaluate_strategy operation")
            return
        
        logger.info(f"Evaluating strategy: {args.strategy}")
        train_instances, test_instances = runner.load_data_from_file(args.data_file)
        
        # Load pretrained model if needed
        pretrained_model = None
        if args.strategy in ["Pretrained_Imputer", "Pretrain_Finetuned_Imputer"]:
            if args.model_file is None:
                logger.error(f"--model_file required for {args.strategy}")
                return
            pretrained_model = runner.load_pretrained_model(args.model_file)
        
        # Evaluate strategy
        if args.instance_idx >= len(test_instances):
            logger.error(f"Instance index {args.instance_idx} out of range (max: {len(test_instances)-1})")
            return
        
        test_instance = test_instances[args.instance_idx]
        results = runner.evaluate_single_strategy(args.strategy, test_instance, pretrained_model, args.instance_idx)
        
        # Save results
        runner.save_results(results, args.output_file)
        logger.info(f"Strategy evaluation completed. Results saved.")
        
    elif args.operation == "evaluate_domain":
        if args.data_file is None:
            logger.error("--data_file required for evaluate_domain operation")
            return
        
        logger.info("Evaluating domain model with custom config...")
        train_instances, test_instances = runner.load_data_from_file(args.data_file)
        
        if args.instance_idx >= len(test_instances):
            logger.error(f"Instance index {args.instance_idx} out of range (max: {len(test_instances)-1})")
            return
        
        test_instance = test_instances[args.instance_idx]
        results = runner.evaluate_domain_model_with_config(
            test_instance,
            chains=args.chains,
            iter_warmup=args.iter_warmup,
            iter_sampling=args.iter_sampling,
            adapt_delta=args.adapt_delta,
            max_treedepth=args.max_treedepth,
            instance_idx=args.instance_idx
        )
        
        # Save results
        runner.save_results(results, args.output_file)
        logger.info("Domain model evaluation completed. Results saved.")


if __name__ == "__main__":
    main()
