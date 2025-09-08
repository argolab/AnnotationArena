#!/usr/bin/env python3
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
from config import ExperimentConfig
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def main():
    parser = argparse.ArgumentParser(description='Train ranking imputer with masking')
    
    # Data parameters (paths only - dimensions from config)
    parser.add_argument('--train_data', type=str, default='generated_data/test_complete_train.json')
    parser.add_argument('--test_data', type=str, default='generated_data/test_complete_test.json')
    parser.add_argument('--config_path', type=str, default=None, help='Path to config file')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    
    # Model parameters
    parser.add_argument('--encoder_layers', type=int, default=4)
    parser.add_argument('--attention_heads', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--save_plots', action='store_true', help='Save training loss plots')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config_path:
        # TODO: Add config loading from file
        config = ExperimentConfig()
        logger.info(f"Using config from {args.config_path}")
    else:
        config = ExperimentConfig()
        logger.info("Using default configuration")
    
    logger.info(f"Config: K={config.K}, I={config.I}, J={config.J}, C={config.C}, ranking_size={config.ranking_size}")
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    train_path = script_dir / args.train_data
    test_path = script_dir / args.test_data
    
    # Initialize components using config
    converter = DataConverter(
        num_attributes=config.I,
        num_annotators=config.J,
        num_items=config.K,
        num_likert_classes=config.C,
        max_rank_size=config.ranking_size
    )
    
    # Load data
    logger.info("Loading training and test data...")
    train_data = converter.load_training_data(str(train_path))
    test_data = converter.load_training_data(str(test_path))
    
    logger.info(f"Training data: {len(train_data['ratings'])} ratings, {len(train_data['rankings'])} rankings")
    logger.info(f"Test data: {len(test_data['ratings'])} ratings, {len(test_data['rankings'])} rankings")
    
    # Create variables based on actual data
    rating_variables, ranking_variables = converter.create_variables_from_actual_data(train_data, test_data)
    logger.info(f"Total variables: {len(rating_variables)} ratings + {len(ranking_variables)} rankings")
    
    # Process data
    rating_data, ranking_data = converter.process_training_data(train_data)
    logger.info(f"Available training data: {len(rating_data)} ratings, {len(ranking_data)} rankings")
    
    # Create batch with masking
    batch = converter.create_batch(
        rating_variables, ranking_variables, rating_data, ranking_data
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
        num_attributes=config.I,
        num_annotators=config.J,
        num_items=config.K,
        num_likert_classes=config.C,
        max_rank_size=config.ranking_size,
        encoder_layers_num=args.encoder_layers,
        attention_heads=args.attention_heads,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout
    )
    
    # Initialize trainer
    trainer = ImputerTrainer(
        model, 
        learning_rate=args.learning_rate
    )
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training for {args.epochs} epochs...")
    
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
        rating_variables, ranking_variables, test_rating_data, test_ranking_data, mode="test"
    )

    # Count masked entries
    train_rating_count = test_batch['rating_mask'].sum().item()
    train_ranking_count = test_batch['ranking_mask'].sum().item()
    masked_rating_count = test_batch['rating_masked'].sum().item()
    masked_ranking_count = test_batch['ranking_masked'].sum().item()
    
    logger.info(f"Testing data: {train_rating_count} ratings ({masked_rating_count} masked), "
               f"{train_ranking_count} rankings ({masked_ranking_count} masked)")
    
    # Training loop
    for epoch in tqdm(range(args.epochs), desc="Training"):
        losses = trainer.train_step(batch)
        
        # Record training losses
        train_losses['epoch'].append(epoch)
        for key in ['total_loss', 'rating_loss', 'ranking_loss']:
            train_losses[key].append(losses[key])
        
        # Evaluate on test set every 10 epochs
        if epoch % 2 == 0:
            test_eval = trainer.evaluate_with_test_data(test_batch, test_data, converter, verbose=False)
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
    test_losses = trainer.evaluate_with_test_data(batch, test_data, converter)
    
    logger.info("Final Results:")
    logger.info(f"Test Rating Loss: {test_losses['test_rating_loss']:.4f}")
    logger.info(f"Test Ranking Loss: {test_losses['test_ranking_loss']:.4f}")
    logger.info(f"Total Test Loss: {test_losses['total_test_loss']:.4f}")
    
    # Save plots
    if args.save_plots:
        import matplotlib.pyplot as plt
        
        # Training plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Top left: Training log loss (combined)
        ax1.plot(train_losses['epoch'], train_losses['total_loss'], 'b-', label='Total')
        ax1.plot(train_losses['epoch'], train_losses['rating_loss'], 'g--', label='Rating')  
        ax1.plot(train_losses['epoch'], train_losses['ranking_loss'], 'r--', label='Ranking')
        ax1.set_title('Training Log Loss (Combined)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Log Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Top right: Rating + Ranking losses 
        ax2.plot(train_losses['epoch'], train_losses['rating_loss'], 'g-', label='Rating Loss', alpha=0.7)
        ax2.plot(train_losses['epoch'], train_losses['ranking_loss'], 'r-', label='Ranking Loss', alpha=0.7)
        ax2.set_title('Training Log Loss by Type')
        ax2.set_xlabel('Epoch') 
        ax2.set_ylabel('Log Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_plots_conditional.png', dpi=300, bbox_inches='tight')
        logger.info(f"Training plots saved to {plots_dir / 'training_plots_conditional.png'}")
        plt.close()
        
        # Test loss plot (separate PNG)
        if len(test_losses_over_time['epoch']) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            
            ax.plot(test_losses_over_time['epoch'], test_losses_over_time['test_rating_loss'], 
                   'b-o', label='Rating Test Log Loss', markersize=4)
            ax.plot(test_losses_over_time['epoch'], test_losses_over_time['test_ranking_loss'], 
                   'r-s', label='Ranking Test Log Loss', markersize=4)
            ax.set_title('Test Set Imputation Log Loss (50% Masked)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Log Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'test_plots_conditional.png', dpi=300, bbox_inches='tight')
            logger.info(f"Test loss plot saved to {plots_dir / 'test_plots_conditional.png'}")
            plt.close()
    
    # Save model
    model_path = models_dir / f'imputer_e{args.epochs}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args,
        'config': config,
        'train_losses': train_losses,
        'test_losses_over_time': test_losses_over_time,
        'final_test_losses': test_losses
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()