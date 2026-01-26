"""
Phase 5 Experiment Runner

Implements all 4 evaluation strategies with comprehensive wall clock timing:
1. Pretrained_Imputer - Direct evaluation of pretrained model
2. Pretrain_Finetuned_Imputer - Pretrained model finetuned on test instance
3. Finetuned_Imputer - Fresh model trained only on test instance
4. Domain_Model - Bayesian domain model with MCMC

All operations are timed and results stored in structured JSON format.
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

from experiment_config import ExperimentConfig
from iclr_data_generator import ICLRDataGenerator, ICLRDatasetConfig
from imputer.data import DataConverter
from imputer.ranking_imputer import MultiVariableImputer
from imputer.legacy.multi_instance_trainer import SequentialMIT, MixedMIT, GeneralMIT
from imputer.eval import EvaluationEngine
from domain_model_trainer import DomainModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing operations."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting {self.operation_name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        logger.info(f"Completed {self.operation_name} in {self.elapsed:.2f} seconds")

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.elapsed


class ExperimentRunner:
    """New Phase 5 experiment runner with all 4 strategies and timing."""

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

        # Results storage
        self.results = {
            "experiment_metadata": {
                "config": config._to_dict(),
                "timestamp": datetime.now().isoformat(),
                "experiment_name": config.experiment_name
            },
            "timing_log": {},
            "pretraining_results": {},
            "test_instance_results": {},
            "summary_statistics": {}
        }

        logger.info(f"Initialized experiment: {config.experiment_name}")
        logger.info(f"Generated data directory: {self.data_dir}")
        logger.info(f"Results output directory: {self.output_dir}")
        logger.info(f"Enabled strategies: {config.enabled_strategies}")

    def run_full_experiment(self) -> Dict[str, Any]:
        """Execute complete experimental pipeline with all 4 strategies."""
        logger.info("="*60)
        logger.info("STARTING FULL EXPERIMENTAL PIPELINE")
        logger.info("="*60)

        with Timer("Full Experiment") as full_timer:
            # 1. Generate data and split train/test instances
            train_instances, test_instances = self.generate_data()

            # Store test instances for use in strategy evaluation
            self.test_instances = test_instances

            # 2. Run pretraining (for strategies 1 & 2)
            pretrained_model = None
            if self._strategy_needs_pretraining():
                pretrained_model = self.run_pretraining(train_instances)

            # 3. Execute all enabled strategies on test instances
            for i, test_instance in enumerate(test_instances):
                logger.info(f"Processing test instance {i+1}/{len(test_instances)}")
                instance_results = self.evaluate_all_strategies(test_instance, pretrained_model, i)
                self.results["test_instance_results"][f"instance_{i}"] = instance_results

            # 4. Compute summary statistics
            self.compute_summary_statistics()

            # 5. Save all results
            self.save_results()

        self.results["timing_log"]["full_experiment"] = full_timer.get_elapsed()
        logger.info("="*60)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*60)

        return self.results

    def generate_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate training and test instances."""
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

        self.results["timing_log"]["data_generation"] = timer.get_elapsed()
        self.results["experiment_metadata"]["data_instances"] = {
            "train": [f"instance_{i}" for i in range(len(train_instances))],
            "test": [f"instance_{i}" for i in range(len(train_instances), len(all_instances))]
        }

        return train_instances, test_instances

    def run_pretraining(self, train_instances: List[Dict]) -> MultiVariableImputer:
        """Run pretraining using Sequential or Mixed MIT."""
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

        self.results["timing_log"]["pretraining"] = timer.get_elapsed()
        self.results["pretraining_results"] = {
            "strategy": self.config.pretraining_config.strategy,
            "training_results": training_results["training_results"] if self.config.save_training_histories else "saved_separately",
            "heldout_evaluation_metrics": training_results.get("callback_results", []),
            "heldout_variables_info": f"Combined heldout set: {len(training_results.get('heldout_variables', []))} variables",
            "model_path": str(model_path) if model_path else None
        }

        return model

    def evaluate_all_strategies(self, test_instance: Dict, pretrained_model: Optional[MultiVariableImputer],
                               instance_idx: int) -> Dict[str, Any]:
        """Run all enabled strategies on a test instance."""
        results = {}

        # Strategy 1: Pretrained_Imputer
        if "Pretrained_Imputer" in self.config.enabled_strategies and pretrained_model is not None:
            results["Pretrained_Imputer"] = self.evaluate_pretrained_imputer(pretrained_model, test_instance, instance_idx)

        # Strategy 2: Pretrain_Finetuned_Imputer
        if "Pretrain_Finetuned_Imputer" in self.config.enabled_strategies and pretrained_model is not None:
            results["Pretrain_Finetuned_Imputer"] = self.evaluate_pretrain_finetuned_imputer(pretrained_model, test_instance, instance_idx)

        # Strategy 3: Finetuned_Imputer
        if "Finetuned_Imputer" in self.config.enabled_strategies:
            results["Finetuned_Imputer"] = self.evaluate_finetuned_imputer(test_instance, instance_idx)

        # Strategy 4: Domain_Model
        if "Domain_Model" in self.config.enabled_strategies:
            results["Domain_Model"] = self.evaluate_domain_model(test_instance, instance_idx)

        return results

    def evaluate_pretrained_imputer(self, model: MultiVariableImputer, test_instance: Dict, instance_idx: int) -> Dict[str, Any]:
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

    def evaluate_pretrain_finetuned_imputer(self, pretrained_model: MultiVariableImputer, test_instance: Dict, instance_idx: int) -> Dict[str, Any]:
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
                device=self.config.device
            )
            model.load_state_dict(pretrained_model.state_dict())

            # Finetune using GeneralMIT
            mit = GeneralMIT(model, self.eval_engine, self.config.finetuning_config, self.converter, self.config.model_config)
            finetuning_results = mit.finetune_on_instance(
                model, test_instance,
                full_test_instances=getattr(self, 'test_instances', None),
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

    def evaluate_finetuned_imputer(self, test_instance: Dict, instance_idx: int) -> Dict[str, Any]:
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
                device=self.config.device
            )

            # Train from scratch using GeneralMIT
            mit = GeneralMIT(model, self.eval_engine, self.config.finetuning_config, self.converter, self.config.model_config)
            training_results = mit.finetune_on_instance(
                model, test_instance,
                full_test_instances=getattr(self, 'test_instances', None),
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

    def evaluate_domain_model(self, test_instance: Dict, instance_idx: int) -> Dict[str, Any]:
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

    def compute_summary_statistics(self):
        """Compute summary statistics across all test instances and strategies."""
        logger.info("Computing summary statistics...")

        # TODO: Implement statistical analysis
        # - Average performance per strategy
        # - Error bars across instances
        # - Statistical significance tests
        # - Best performing strategy identification

        self.results["summary_statistics"] = {
            "computed_at": datetime.now().isoformat(),
            "note": "Summary statistics computation to be implemented"
        }

    def save_results(self):
        """Save all experimental results to JSON file."""
        results_file = self.output_dir / f"{self.config.experiment_name}_results.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")

    def _strategy_needs_pretraining(self) -> bool:
        """Check if any enabled strategy requires pretraining."""
        pretraining_strategies = ["Pretrained_Imputer", "Pretrain_Finetuned_Imputer"]
        return any(strategy in self.config.enabled_strategies for strategy in pretraining_strategies)


def main():
    """Main entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 5 ranking imputation experiment")
    parser.add_argument("--config", type=str, default="configs/test_config.json",
                       help="Path to experiment configuration file")
    parser.add_argument("--create-test-config", action="store_true",
                       help="Create a test configuration file and exit")

    args = parser.parse_args()

    if args.create_test_config:
        from experiment_config import create_test_config
        config = create_test_config()
        config_path = "configs/test_config.json"
        config.save(config_path)
        print(f"Test configuration saved to {config_path}")
        return

    # Load config and run experiment
    config = ExperimentConfig.load(args.config)
    runner = ExperimentRunner(config)
    results = runner.run_full_experiment()

    print(f"Experiment completed. Results saved to {runner.output_dir}")


if __name__ == "__main__":
    main()