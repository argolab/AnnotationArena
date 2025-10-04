"""
Tuned Lens and Logit Lens implementation for transformer interpretability.

Logit Lens: Projects hidden states at each layer directly to output space using final layer norm + prediction head
Tuned Lens: Learns optimal affine transformations per layer to project to output space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class LogitLens:
    """
    Logit Lens: Uses the model's final layer norm and output projection
    to decode intermediate layer representations.
    """
    
    def __init__(self, model):
        """
        Args:
            model: ImputerEmbedding model with encoder layers
        """
        self.model = model
        self.device = next(model.parameters()).device
        
    def analyze_layers(self, inputs, annotators, questions, target_position=None, true_probs=None):
        """
        Analyze what each layer predicts by projecting through final layers.
        
        Args:
            inputs: Input tensor [batch_size, seq_len, input_dim]
            annotators: Annotator indices
            questions: Question indices  
            target_position: Specific position to analyze (None = all positions)
            true_probs: True probability distribution [batch_size, seq_len, num_classes] or [num_classes]
            
        Returns:
            Dictionary with layer-wise predictions and statistics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get initial embeddings
            feature_x, param_x, query_x = self.model.encoder.position_encoder(
                inputs, annotators, questions
            )
            
            layer_predictions = []
            layer_entropies = []
            layer_confidences = []
            layer_kl_divergences = []
            
            # Analyze each encoder layer
            for layer_idx, layer in enumerate(self.model.encoder.layers):
                # Pass through this layer
                feature_x, param_x, query_x = layer(
                    feature_x, param_x, query_x, questions, inputs[:, :, 0]
                )
                
                # Project param_x (which becomes the output) to predictions
                # param_x is already in the right space for predictions
                predictions = param_x  # [batch_size, seq_len, max_choices]
                probs = F.softmax(predictions, dim=-1)
                
                if target_position is not None:
                    predictions = predictions[:, target_position:target_position+1, :]
                    probs = probs[:, target_position:target_position+1, :]
                
                # Compute statistics
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                confidence = torch.max(probs, dim=-1)[0]
                
                # Compute KL divergence if true distribution provided
                if true_probs is not None:
                    true_p = true_probs.to(self.device)
                    if target_position is not None and len(true_p.shape) > 1:
                        true_p = true_p[:, target_position:target_position+1, :]
                    
                    # Ensure proper shape
                    if len(true_p.shape) == 1:
                        true_p = true_p.unsqueeze(0).unsqueeze(0)
                    elif len(true_p.shape) == 2:
                        true_p = true_p.unsqueeze(1)
                    
                    # KL(true || predicted)
                    kl_div = F.kl_div(
                        torch.log(probs + 1e-10), 
                        true_p,
                        reduction='batchmean'
                    )
                    layer_kl_divergences.append(kl_div.cpu())
                
                layer_predictions.append(predictions.cpu())
                layer_entropies.append(entropy.cpu())
                layer_confidences.append(confidence.cpu())
        
        result = {
            'predictions': layer_predictions,
            'entropies': layer_entropies,
            'confidences': layer_confidences,
            'num_layers': len(layer_predictions)
        }
        
        if true_probs is not None:
            result['kl_divergences'] = layer_kl_divergences
        
        return result

class SimpleMLP(nn.Module):
    def __init__(self, param_dim, hidden_factor=2):
        super().__init__()
        hidden_dim = param_dim * hidden_factor
        
        self.linear1 = nn.Linear(param_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, param_dim)
        
        # Kaiming initialization
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.linear2.bias)
        
        # Scale down final layer for stability with residual connection
        with torch.no_grad():
            self.linear2.weight.mul_(0.1)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        return x + out  # Residual connection

class TunedLens(nn.Module):
    """
    Tuned Lens: Learns optimal affine transformations to project
    intermediate representations to output space.
    """
    
    def __init__(self, model, param_dim):
        """
        Args:
            model: ImputerEmbedding model
            param_dim: Dimension of param_x (should be max_choices)
        """
        super().__init__()
        self.model = model
        self.param_dim = param_dim
        self.num_layers = len(model.encoder.layers)
        
        # Learn affine transformation per layer: W*x + b
        self.layer_transforms = nn.ModuleList([
            SimpleMLP(param_dim, hidden_factor=2)
            for _ in range(self.num_layers)
        ])
        
    
    def train_lens(self, train_dataset, epochs=10, lr=1e-3, batch_size=32):
        """
        Train the lens transformations to match final layer outputs.
        
        Args:
            train_dataset: Training dataset
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
        """
        optimizer = torch.optim.Adam(self.layer_transforms.parameters(), lr=lr)
        
        self.model.eval()  # Keep base model frozen
        
        logger.info(f"Training Tuned Lens for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Create batches
            indices = torch.randperm(len(train_dataset))
            for batch_start in range(0, len(train_dataset), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                
                batch_inputs = []
                batch_annotators = []
                batch_questions = []
                batch_answers = []
                
                for idx in batch_indices:
                    known_q, inp, ans, annot, quest = train_dataset[idx]
                    batch_inputs.append(inp)
                    batch_annotators.append(annot)
                    batch_questions.append(quest)
                    batch_answers.append(ans)
                
                inputs = torch.stack(batch_inputs).to(self.model.device)
                annotators = torch.stack(batch_annotators).to(self.model.device)
                questions = torch.stack(batch_questions).to(self.model.device)
                answers = torch.stack(batch_answers).to(self.model.device)
                
                
                with torch.no_grad():
                    # Get final output as target
                    final_output, _ = self.model(inputs, annotators, questions)
                    # Normalize to probability distribution
                    final_probs = F.softmax(final_output, dim=-1)
                    
                    # Get intermediate representations
                    feature_x, param_x, query_x = self.model.encoder.position_encoder(
                        inputs, annotators, questions
                    )
                    
                    intermediate_params = []
                    for layer in self.model.encoder.layers:
                        feature_x, param_x, query_x = layer(
                            feature_x, param_x, query_x, questions, inputs[:, :, 0]
                        )
                        intermediate_params.append(param_x.clone())
                
                # Train transformations
                optimizer.zero_grad()
                layer_loss = 0.0
                
                for layer_idx, (transform, param_x) in enumerate(
                    zip(self.layer_transforms, intermediate_params)
                ):
                    # Apply learned transformation
                    transformed = transform(param_x)
                    # Normalize to probability distribution
                    transformed_probs = F.softmax(transformed, dim=-1)
                    
                    # KL divergence: KL(final || transformed)
                    # Use log of transformed_probs and final_probs as target
                    loss = F.kl_div(
                        torch.log(transformed_probs + 1e-10),
                        final_probs,
                        reduction='batchmean'
                    )
                    layer_loss += loss
                
                layer_loss = layer_loss / len(self.layer_transforms)
                layer_loss.backward()
                optimizer.step()
                
                total_loss += layer_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(1, num_batches)
            logger.info(f"Epoch {epoch+1}/{epochs}, KL Loss: {avg_loss:.6f}")
        
        logger.info("Tuned Lens training complete")
    
    def analyze_layers(self, inputs, annotators, questions, target_position=None, true_probs=None):
        """
        Analyze predictions using tuned transformations.
        
        Args:
            inputs: Input tensor
            annotators: Annotator indices
            questions: Question indices
            target_position: Specific position to analyze
            true_probs: True probability distribution for KL divergence
        
        Returns:
            Dictionary with layer-wise predictions and statistics
        """
        self.eval()
        self.model.eval()
        
        with torch.no_grad():
            feature_x, param_x, query_x = self.model.encoder.position_encoder(
                inputs, annotators, questions
            )
            
            layer_predictions = []
            layer_entropies = []
            layer_confidences = []
            layer_kl_divergences = []
            
            for layer_idx, (layer, transform) in enumerate(
                zip(self.model.encoder.layers, self.layer_transforms)
            ):
                feature_x, param_x, query_x = layer(
                    feature_x, param_x, query_x, questions, inputs[:, :, 0]
                )
                
                # Apply tuned transformation
                predictions = transform(param_x)
                probs = F.softmax(predictions, dim=-1)
                
                if target_position is not None:
                    predictions = predictions[:, target_position:target_position+1, :]
                    probs = probs[:, target_position:target_position+1, :]
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                confidence = torch.max(probs, dim=-1)[0]
                
                # Compute KL divergence if true distribution provided
                if true_probs is not None:
                    true_p = true_probs.to(self.model.device)
                    if target_position is not None and len(true_p.shape) > 1:
                        true_p = true_p[:, target_position:target_position+1, :]
                    
                    # Ensure proper shape
                    if len(true_p.shape) == 1:
                        true_p = true_p.unsqueeze(0).unsqueeze(0)
                    elif len(true_p.shape) == 2:
                        true_p = true_p.unsqueeze(1)
                    
                    kl_div = F.kl_div(
                        torch.log(probs + 1e-10), 
                        true_p,
                        reduction='batchmean'
                    )
                    layer_kl_divergences.append(kl_div.cpu())
                
                layer_predictions.append(predictions.cpu())
                layer_entropies.append(entropy.cpu())
                layer_confidences.append(confidence.cpu())
        
        result = {
            'predictions': layer_predictions,
            'entropies': layer_entropies,
            'confidences': layer_confidences,
            'num_layers': len(layer_predictions)
        }
        
        if true_probs is not None:
            result['kl_divergences'] = layer_kl_divergences
        
        return result


def plot_lens_analysis(logit_results, tuned_results, true_label=None, 
                      save_path='lens_analysis.png'):
    """
    Plot layer-wise analysis from both lens types.
    
    Args:
        logit_results: Results from LogitLens
        tuned_results: Results from TunedLens  
        true_label: Optional true label index
        save_path: Path to save plot
    """
    num_layers = logit_results['num_layers']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Extract data
    logit_entropies = [e.mean().item() for e in logit_results['entropies']]
    tuned_entropies = [e.mean().item() for e in tuned_results['entropies']]
    
    logit_confidences = [c.mean().item() for c in logit_results['confidences']]
    tuned_confidences = [c.mean().item() for c in tuned_results['confidences']]
    
    layers = list(range(num_layers))
    
    # Plot 1: Entropy over layers
    axes[0, 0].plot(layers, logit_entropies, 'o-', label='Logit Lens', linewidth=2)
    axes[0, 0].plot(layers, tuned_entropies, 's-', label='Tuned Lens', linewidth=2)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Average Entropy')
    axes[0, 0].set_title('Prediction Entropy by Layer')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Confidence over layers
    axes[0, 1].plot(layers, logit_confidences, 'o-', label='Logit Lens', linewidth=2)
    axes[0, 1].plot(layers, tuned_confidences, 's-', label='Tuned Lens', linewidth=2)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Average Confidence')
    axes[0, 1].set_title('Prediction Confidence by Layer')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Prediction distribution heatmap (Logit Lens)
    num_classes = logit_results['predictions'][0].shape[-1]
    logit_dist = np.zeros((num_layers, num_classes))
    for i, pred in enumerate(logit_results['predictions']):
        probs = F.softmax(pred[0, 0], dim=-1).numpy()
        logit_dist[i] = probs
    
    im1 = axes[0, 2].imshow(logit_dist.T, aspect='auto', cmap='viridis')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Class')
    axes[0, 2].set_title('Logit Lens: Class Probabilities')
    if true_label is not None:
        axes[0, 2].axhline(y=true_label, color='r', linestyle='--', linewidth=2, label='True Label')
        axes[0, 2].legend()
    plt.colorbar(im1, ax=axes[0, 2])
    
    # Plot 4: Prediction distribution heatmap (Tuned Lens)
    tuned_dist = np.zeros((num_layers, num_classes))
    for i, pred in enumerate(tuned_results['predictions']):
        probs = F.softmax(pred[0, 0], dim=-1).numpy()
        tuned_dist[i] = probs
    
    im2 = axes[1, 0].imshow(tuned_dist.T, aspect='auto', cmap='viridis')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Class')
    axes[1, 0].set_title('Tuned Lens: Class Probabilities')
    if true_label is not None:
        axes[1, 0].axhline(y=true_label, color='r', linestyle='--', linewidth=2, label='True Label')
        axes[1, 0].legend()
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot 5: Top prediction changes
    logit_top_preds = [torch.argmax(F.softmax(pred[0, 0], dim=-1)).item() 
                       for pred in logit_results['predictions']]
    tuned_top_preds = [torch.argmax(F.softmax(pred[0, 0], dim=-1)).item() 
                       for pred in tuned_results['predictions']]
    
    axes[1, 1].plot(layers, logit_top_preds, 'o-', label='Logit Lens', linewidth=2, markersize=8)
    axes[1, 1].plot(layers, tuned_top_preds, 's-', label='Tuned Lens', linewidth=2, markersize=8)
    if true_label is not None:
        axes[1, 1].axhline(y=true_label, color='r', linestyle='--', linewidth=2, label='True Label')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Predicted Class')
    axes[1, 1].set_title('Top Prediction by Layer')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Difference between lenses
    entropy_diff = np.array(logit_entropies) - np.array(tuned_entropies)
    conf_diff = np.array(logit_confidences) - np.array(tuned_confidences)
    
    ax6 = axes[1, 2]
    ax6_twin = ax6.twinx()
    
    l1 = ax6.plot(layers, entropy_diff, 'o-', color='blue', label='Entropy Diff', linewidth=2)
    l2 = ax6_twin.plot(layers, conf_diff, 's-', color='orange', label='Confidence Diff', linewidth=2)
    
    ax6.set_xlabel('Layer')
    ax6.set_ylabel('Entropy Difference', color='blue')
    ax6_twin.set_ylabel('Confidence Difference', color='orange')
    ax6.set_title('Logit - Tuned Lens Differences')
    ax6.tick_params(axis='y', labelcolor='blue')
    ax6_twin.tick_params(axis='y', labelcolor='orange')
    ax6.grid(True, alpha=0.3)
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Lens analysis plot saved to {save_path}")
    
    return fig


def analyze_model_with_lenses(model, dataset, train_dataset=None, 
                              example_idx=0, target_position=0,
                              train_tuned_lens=True):
    """
    Complete pipeline to analyze a model using both lens types.
    
    Args:
        model: Trained ImputerEmbedding model
        dataset: Dataset for analysis
        train_dataset: Dataset for training tuned lens (optional)
        example_idx: Index of example to analyze
        target_position: Position to analyze
        train_tuned_lens: Whether to train the tuned lens
        
    Returns:
        Tuple of (logit_results, tuned_results, figure)
    """
    # Get example data
    known_q, inputs, answers, annotators, questions = dataset[example_idx]
    inputs = inputs.unsqueeze(0).to(model.device)
    annotators = annotators.unsqueeze(0).to(model.device)
    questions = questions.unsqueeze(0).to(model.device)
    
    true_label = torch.argmax(answers[target_position]).item()
    
    # Logit Lens analysis
    logger.info("Running Logit Lens analysis...")
    logit_lens = LogitLens(model)
    logit_results = logit_lens.analyze_layers(
        inputs, annotators, questions, target_position
    )
    
    # Tuned Lens analysis
    logger.info("Running Tuned Lens analysis...")
    tuned_lens = TunedLens(model, model.max_choices)
    
    if train_tuned_lens and train_dataset is not None:
        tuned_lens.train_lens(train_dataset)
    
    tuned_results = tuned_lens.analyze_layers(
        inputs, annotators, questions, target_position
    )
    
    # Plot results
    fig = plot_lens_analysis(logit_results, tuned_results, true_label)
    
    return logit_results, tuned_results, fig


# Example usage function
def example_usage():
    """Example of how to use the lens analysis"""
    # Assuming you have a trained model and datasets
    # model = your_trained_model
    # train_dataset = your_train_dataset
    # eval_dataset = your_eval_dataset
    
    # Analyze a specific example
    # logit_results, tuned_results, fig = analyze_model_with_lenses(
    #     model=model,
    #     dataset=eval_dataset,
    #     train_dataset=train_dataset,
    #     example_idx=0,
    #     target_position=5,
    #     train_tuned_lens=True
    # )
    
    pass