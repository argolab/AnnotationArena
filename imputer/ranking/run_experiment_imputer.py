#!/usr/bin/env python3
"""
Legacy experiment runner for backwards compatibility.
For new experiments, use experiment_runner.py with JSON configs.
"""
import argparse
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import torch

# Import from the refactored imputer package
from imputer import (
    DataConverter, MultiVariableImputer, ImputerTrainer
)
from config import ExperimentConfig, InstanceConfig, ModelConfig, TrainingConfig
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def main():
    parser = argparse.ArgumentParser(description='Train ranking imputer with masking (Legacy Mode)')
    
    # Data parameters (paths only - dimensions from config)
    parser.add_argument('--train_data', type=str, default='generated_data/iclr_complete_train.json')
    parser.add_argument('--test_data', type=str, default='generated_data/iclr_complete_test.json')
    parser.add_argument('--config_path', type=str, default=None, help='Path to NEW JSON config file')
    
    # Training parameters (for backwards compatibility)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--embedding_anchor_reg', type=float, default=0.0)
    parser.add_argument('--masking_rate', type=float, default=0.5, help='Fraction of training variables to mask (0.0-1.0)')
    
    # Model parameters (for backwards compatibility)
    parser.add_argument('--encoder_layers', type=int, default=4)
    parser.add_argument('--attention_heads', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='OUTPUT/IMPUTER/legacy')
    parser.add_argument('--save_plots', action='store_true', help='Save training loss plots')
    parser.add_argument("--embedding_type", default="pairwise", help="Type of layer 0 representation to use")
    parser.add_argument("--device", default="cpu", help="Device use for training and testing")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load or create configuration
    if args.config_path:
        logger.info(f"Loading new JSON config from {args.config_path}")
        config = ExperimentConfig.load_from_file(args.config_path)
        
        # Force single instance mode for legacy script
        if config.experiment_type != "single_instance":
            logger.warning("Legacy script only supports single_instance experiments. Use experiment_runner.py for multi-instance.")
            config.experiment_type = "single_instance"
            config.train_instance_indices = [0]
            config.test_instance_indices = [0]
    else:
        logger.info("Creating configuration from command line arguments (Legacy Mode)")
        
        # Create config from args for backwards compatibility
        instance_config = InstanceConfig()  # Use defaults, will be compatible with existing data
        
        model_config = ModelConfig(
            encoder_layers=args.encoder_layers,
            attention_heads=args.attention_heads,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            embedding_type=args.embedding_type
        )
        
        training_config = TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            embedding_anchor_reg=args.embedding_anchor_reg,
            masking_rate=args.masking_rate,
            evaluation_frequency=2
        )
        
        config = ExperimentConfig.create_single_instance(
            instance_config=instance_config,
            model_config=model_config,
            training_config=training_config,
            base_output_dir=str(Path(args.output_dir).parent),
            save_plots=args.save_plots,
            device=args.device
        )
    
    legacy_props = config.get_legacy_properties()
    logger.info(f"Config: K={legacy_props['K']}, I={legacy_props['I']}, J={legacy_props['J']}, C={legacy_props['C']}, ranking_size=2")
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    train_path = script_dir / args.train_data
    test_path = script_dir / args.test_data
    
    # Initialize components using config
    converter = DataConverter(
        num_attributes=legacy_props['I'],
        num_annotators=legacy_props['J'],
        num_items=legacy_props['K'],
        num_likert_classes=legacy_props['C'],
        max_rank_size=2
    )
    
    # Load data
    logger.info("Loading training and test data...")
    train_data = converter.load_training_data(str(train_path))
    test_data = converter.load_training_data(str(test_path))
    
    logger.info(f"Training data: {len(train_data['ratings'])} ratings, {len(train_data['pairwise_rankings'])} rankings")
    logger.info(f"Test data: {len(test_data['ratings'])} ratings, {len(test_data['pairwise_rankings'])} rankings")
    
    # Create variables based on actual data
    rating_variables, ranking_variables = converter.create_variables_from_actual_data(train_data, test_data)
    logger.info(f"Total variables: {len(rating_variables)} ratings + {len(ranking_variables)} rankings")
    
    # Process data
    rating_data, ranking_data = converter.process_training_data(train_data)
    logger.info(f"Available training data: {len(rating_data)} ratings, {len(ranking_data)} rankings")
    
    # Create batch with masking
    batch = converter.create_batch(
        rating_variables, ranking_variables, rating_data, ranking_data, masking_rate=config.training_config.masking_rate
    )
    
    # Count masked entries
    train_rating_count = batch['rating_mask'].sum().item()
    train_ranking_count = batch['ranking_mask'].sum().item()
    masked_rating_count = batch['rating_masked'].sum().item()
    masked_ranking_count = batch['ranking_masked'].sum().item()
    
    logger.info(f"Training data: {train_rating_count} ratings ({masked_rating_count} masked), "
               f"{train_ranking_count} rankings ({masked_ranking_count} masked)")
    
    # Initialize model using config
    model = MultiVariableImputer(
        num_attributes=legacy_props['I'],
        num_annotators=legacy_props['J'],
        num_items=legacy_props['K'],
        num_likert_classes=legacy_props['C'],
        max_rank_size=2,
        encoder_layers_num=config.model_config.encoder_layers,
        attention_heads=config.model_config.attention_heads,
        embedding_dim=config.model_config.embedding_dim,
        dropout=config.model_config.dropout,
        embedding_type=config.model_config.embedding_type,
        device=config.device
    )
    
    # Initialize trainer
    trainer = ImputerTrainer(
        model, 
        learning_rate=config.training_config.learning_rate,
        device=config.device,
        embedding_anchor_reg=config.training_config.embedding_anchor_reg,
    )
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training for {config.training_config.epochs} epochs...")
    
    # Track training losses for plotting
    train_losses = {
        'epoch': [], 'total_loss': [], 'rating_loss': [], 'ranking_loss': []
    }
    
    # Track test losses for plotting
    test_losses_over_time = {
        'epoch': [], 'test_rating_loss': [], 'test_ranking_loss': []
    }

    test_rating_data, test_ranking_data = converter.process_training_data(test_data)
    logger.info(f"Available test data: {len(test_rating_data)} ratings, {len(test_ranking_data)} rankings")

    test_batch = converter.create_batch(
        rating_variables, ranking_variables, test_rating_data, test_ranking_data, mode="test", masking_rate=config.training_config.masking_rate
    )

    # Count masked entries
    train_rating_count = test_batch['rating_mask'].sum().item()
    train_ranking_count = test_batch['ranking_mask'].sum().item()
    masked_rating_count = test_batch['rating_masked'].sum().item()
    masked_ranking_count = test_batch['ranking_masked'].sum().item()
    
    logger.info(f"Testing data: {train_rating_count} ratings ({masked_rating_count} masked), "
               f"{train_ranking_count} rankings ({masked_ranking_count} masked)")
    
    # Training loop
    for epoch in tqdm(range(config.training_config.epochs), desc="Training"):
        losses = trainer.train_step(batch)
        
        # Record training losses
        train_losses['epoch'].append(epoch)
        for key in ['total_loss', 'rating_loss', 'ranking_loss']:
            train_losses[key].append(losses[key])
        
        # Evaluate on test set every N epochs
        if epoch % config.training_config.evaluation_frequency == 0:
            test_eval = trainer.evaluate_with_test_data(test_batch, test_data, converter, masking_rate=config.training_config.masking_rate, verbose=False)
            test_losses_over_time['epoch'].append(epoch)
            test_losses_over_time['test_rating_loss'].append(test_eval['test_rating_loss'])
            test_losses_over_time['test_ranking_loss'].append(test_eval['test_ranking_loss'])
            
            logger.info(f"Epoch {epoch}: Total={losses['total_loss']:.4f}, "
                       f"Rating={losses['rating_loss']:.4f}, "
                       f"Ranking={losses['ranking_loss']:.4f}")
            logger.info(f"TEST LOSS & METRICS: {test_eval}")
    
    logger.info("Training completed!")
    
    # Final evaluation
    logger.info("Evaluating on test data...")
    test_losses = trainer.evaluate_with_test_data(test_batch, test_data, converter, masking_rate=config.training_config.masking_rate)
    
    logger.info("Final Results:")
    logger.info(f"Test Rating Loss: {test_losses['test_rating_loss']:.4f}")
    logger.info(f"Test Ranking Loss: {test_losses['test_ranking_loss']:.4f}")
    logger.info(f"Total Test Loss: {test_losses['total_test_loss']:.4f}")
    
    # Save plots
    if config.save_plots:
        import matplotlib.pyplot as plt
        
        # Training plot - single comprehensive plot (removed redundancy)
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses['epoch'], train_losses['total_loss'], 'b-', label='Total', linewidth=2)
        plt.plot(train_losses['epoch'], train_losses['rating_loss'], 'g--', label='Rating', linewidth=2)  
        plt.plot(train_losses['epoch'], train_losses['ranking_loss'], 'r--', label='Ranking', linewidth=2)
        plt.title(f'Training Log Loss (Masking Rate {config.training_config.masking_rate:.1f})')
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
        logger.info(f"Training plot saved to {plots_dir / 'training_loss.png'}")
        plt.close()
        
        # Test loss plot (separate PNG)
        if len(test_losses_over_time['epoch']) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            
            plt.plot(test_losses_over_time['epoch'], test_losses_over_time['test_rating_loss'], 
                   'b-o', label='Rating Test Log Loss', markersize=4, linewidth=2)
            plt.plot(test_losses_over_time['epoch'], test_losses_over_time['test_ranking_loss'], 
                   'r-s', label='Ranking Test Log Loss', markersize=4, linewidth=2)
            plt.title(f'Test Log Loss (Masking Rate {config.training_config.masking_rate:.1f})')
            plt.xlabel('Epoch')
            plt.ylabel('Log Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'test_loss.png', dpi=300, bbox_inches='tight')
            logger.info(f"Test loss plot saved to {plots_dir / 'test_loss.png'}")
            plt.close()
    
    # Save model
    model_path = models_dir / f'imputer_e{config.training_config.epochs}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'test_losses_over_time': test_losses_over_time,
        'final_test_losses': test_losses
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()