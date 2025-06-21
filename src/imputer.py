"""
Imputer for Active Learner framework with fixed training queue system.
Based on original v1/src/imputer_embedding.py with ONLY data access fixes applied.

Author: Prabhav Singh / Haojun Shi
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Wandb not available, logging disabled")


class NormLayer(nn.Module):
    """Layer normalization with learnable parameters."""
    
    def __init__(self, d_model, eps=1e-6):
        """Initialize normalization layer."""
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        """Apply normalization to input tensor."""
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    """Simple feed-forward network with ReLU activation."""
    
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        """Initialize feed-forward network."""
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """Transform input through feed-forward layers."""
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class FullyVectorizedSimilaritySmoothing(nn.Module):
    """Fully vectorized similarity smoothing layer."""
    
    def __init__(self, hidden_dim, param_dim, num_question_types, dropout=0.1):
        """Initialize smoothing layer."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.param_dim = param_dim
        self.num_question_types = num_question_types
        
        # Q and K matrices for attention
        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Temperature parameter
        self.temp_projection = nn.Linear(hidden_dim, 1, bias=False)
        with torch.no_grad():
            self.temp_projection.weight.normal_(0, 0.01)
        
        with torch.no_grad():
            # Johnson-Lindenstrauss: entries ~ N(0, 1/k)
            jl_matrix = torch.randn(hidden_dim, hidden_dim) / math.sqrt(hidden_dim)
            self.Q.weight.copy_(jl_matrix)
            self.K.weight.copy_(jl_matrix.clone())
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, param_states, questions, mask):
        """Fully vectorized forward pass."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Ensure mask is boolean for proper operations
        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
        
        # Early exit if no masked positions
        if mask_bool.sum() == 0:
            return hidden_states, param_states
        
        # Compute Q and K projections
        Q = self.Q(hidden_states)  # [B, L, H]
        K = self.K(hidden_states)  # [B, L, H]
        
        variable_temps = F.relu(self.temp_projection(hidden_states)) + 0.01 # [B, L, 1]

        # Expand temperatures for broadcasting: [B, L, L]
        temp_matrix = variable_temps.expand(-1, -1, seq_len)  # [B, L, L]

        # Create attention scores with per-variable temperature
        scores = torch.bmm(Q, K.transpose(-2, -1)) / temp_matrix
        
        # Create question type mask - only allow attention within same question type
        question_mask = questions.unsqueeze(-1) == questions.unsqueeze(-2)  # [B, L, L]
        
        # Apply question type mask
        scores = scores.masked_fill(~question_mask, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [B, L, L]
        attention_weights = self.dropout(attention_weights)
        
        # Apply smoothing only to masked positions
        should_smooth = mask_bool.unsqueeze(-1).expand(-1, -1, seq_len)  # [B, L, L]
        
        # Zero out attention weights for positions that shouldn't be smoothed
        attention_weights = attention_weights * should_smooth.float()
        
        # For positions that shouldn't be smoothed, set self-attention to 1
        eye_mask = torch.eye(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1, -1)
        no_smooth_mask = (~mask_bool).unsqueeze(-1).expand(-1, -1, seq_len)
        attention_weights = attention_weights + eye_mask * no_smooth_mask.float()
        
        # Apply smoothing
        smoothed_params = torch.bmm(attention_weights, param_states)  # [B, L, P]
        
        return hidden_states, smoothed_params


class Positional_Encoder(nn.Module):
    """Encodes question and annotator information."""
    
    def __init__(self, question_num, max_choices, num_annotator, annotator_embedding_dim):
        """Initialize positional encoder."""
        super().__init__()
        self.question_num = question_num
        self.max_choices = max_choices
        self.annotator_embedding = nn.Parameter(torch.randn(num_annotator + 1, annotator_embedding_dim))
        self.question_embedding = nn.Parameter(torch.randn(question_num, annotator_embedding_dim))
        torch.nn.init.kaiming_normal_(self.annotator_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.question_embedding, mode='fan_out', nonlinearity='relu')
        self.num_annotator = num_annotator

    def forward(self, x, annotators, questions, embeddings):
        """Create encoded representations combining annotator and question features."""
        batch_size = x.shape[0]
        question_embeds = self.question_embedding[questions]
        annotators = torch.where(annotators < 0, torch.full_like(annotators, self.num_annotator), annotators)
        annotator_embeds = self.annotator_embedding[annotators]
        
        # Handle embeddings shape
        if len(embeddings.shape) == 4:
            embeddings = embeddings.squeeze(0)
            
        # Combine all embeddings
        combined_embeds = question_embeds + annotator_embeds
        feature_x = torch.cat((combined_embeds, embeddings, x[:,:,1:]), dim=-1)
        param_x = x[:,:,1:].clone()
        
        return feature_x, param_x


class EncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feed-forward networks."""
    
    def __init__(self, feature_dim, param_dim, attention_heads, num_question_types, dropout=0.3):
        """Initialize encoder layer."""
        super().__init__()
        self.feature_dim = feature_dim 
        self.param_dim = param_dim  
        self.attention_heads = attention_heads
        
        self.Q = nn.Linear(feature_dim, feature_dim)
        self.K = nn.Linear(feature_dim, feature_dim)
        self.V = nn.Linear(feature_dim, feature_dim)
        self.out = nn.Linear(feature_dim, feature_dim)
        
        self.norm_1 = NormLayer(feature_dim)
        self.norm_2 = NormLayer(feature_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
        self.ff = FeedForward(feature_dim, dropout=dropout)
        self.param_update = nn.Linear(feature_dim + param_dim, param_dim)

        # Add smoothing layer
        self.smoothing = FullyVectorizedSimilaritySmoothing(
            hidden_dim=feature_dim,
            param_dim=param_dim, 
            num_question_types=num_question_types,
            dropout=dropout
        )

    def multihead_attention(self, feature_x, batch_size):
        """Apply multi-head attention to the features."""
        Q = self.Q(feature_x).view(batch_size, -1, self.attention_heads, self.feature_dim // self.attention_heads).transpose(1, 2)
        K = self.K(feature_x).view(batch_size, -1, self.attention_heads, self.feature_dim // self.attention_heads).transpose(1, 2)
        V = self.V(feature_x).view(batch_size, -1, self.attention_heads, self.feature_dim // self.attention_heads).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.feature_dim // self.attention_heads)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_1(scores)
        scores = torch.matmul(scores, V)
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)
        output = self.out(scores)
        return output

    def forward(self, feature_x, param_x, questions, mask):
        """Process features through attention, feed-forward, and smoothing."""
        batch_size = feature_x.shape[0]
        
        # Standard transformer processing
        feature_x = self.norm_1(feature_x)
        attention_output = self.multihead_attention(feature_x, batch_size)
        feature_x = feature_x + self.dropout_1(attention_output)
        
        feature_x_ff = self.norm_2(feature_x)
        feature_x = feature_x + self.dropout_2(self.ff(feature_x_ff))
        
        # Update parameters
        combined = torch.cat([feature_x, param_x], dim=-1)
        param_x = self.param_update(combined)
        
        # Apply smoothing (this is the key addition)
        feature_x, param_x = self.smoothing(feature_x, param_x, questions, mask)
        
        return feature_x, param_x


class Encoder(nn.Module):
    """Full encoder consisting of multiple encoder layers."""

    def __init__(self, question_num, max_choices, encoder_num, attention_heads, 
             num_annotator, annotator_embedding_dim, dropout=0.1):
        """Initialize encoder with multiple layers."""
        super().__init__()
        self.feature_dim = annotator_embedding_dim + max_choices + 384
        self.param_dim = max_choices
        self.position_encoder = Positional_Encoder(question_num, max_choices, num_annotator, annotator_embedding_dim)

        self.layers = nn.ModuleList([
            EncoderLayer(self.feature_dim, self.param_dim, attention_heads, 
                        question_num, dropout)
            for _ in range(encoder_num)
        ])

        self.norm = NormLayer(self.feature_dim)
        self.annotator_embedding_dim = annotator_embedding_dim

    def forward(self, x, annotators, questions, embeddings):
        """Process input through all encoder layers with per-layer smoothing."""
        feature_x, param_x = self.position_encoder(x, annotators, questions, embeddings)
        
        # Create mask from input (1 where masked, 0 where observed)
        mask = x[:, :, 0]
        
        # Process through encoder layers (each applies smoothing)
        for layer in self.layers:
            feature_x, param_x = layer(feature_x, param_x, questions, mask)
        
        return param_x


class ImputerEmbedding(nn.Module):
    """
    Imputer model for predicting missing annotations with fixed training queue system.
    Based on original architecture with ONLY data access fixes applied.
    """
    
    def __init__(self, 
                 question_num=7, 
                 max_choices=5, 
                 encoder_layers_num=6, 
                 attention_heads=4, 
                 hidden_dim=64, 
                 num_annotator=15, 
                 annotator_embedding_dim=8, 
                 dropout=0.1):
        """Initialize Imputer model."""
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.question_num = question_num
        self.max_choices = max_choices
        self.encoder = Encoder(question_num, max_choices, encoder_layers_num, attention_heads, 
                               num_annotator, annotator_embedding_dim, dropout)
        
        # FIXED: Training queue system instead of training_examples
        self.training_queue = []
        self.prediction_history = []
        self.examples_to_revisit = set()
        self.training_losses = []
        self.dataset = None
        
        logger.info(f"ImputerEmbedding initialized: {question_num} questions, {max_choices} choices, {num_annotator} annotators")
    
    def set_dataset(self, dataset):
        """Set dataset reference for current data access."""
        self.dataset = dataset
        logger.debug(f"Dataset set with {len(dataset)} examples")
    
    def forward(self, x, annotators, questions, embeddings):
        """Forward pass through the model."""
        param_x = self.encoder(x, annotators, questions, embeddings)
        return param_x
    
    def predict(self, inputs, annotators, questions, embeddings, positions=None, train=True, weight=1.0, example_idx=None):
        """
        Predict distributions for specified positions.
        
        Args:
            inputs: Input tensor [batch_size, sequence_length, input_dim]
            annotators: Annotator indices [batch_size, sequence_length]
            questions: Question indices [batch_size, sequence_length]
            embeddings: Text embeddings [batch_size, sequence_length, embedding_dim]
            positions: Positions to predict (default: all)
            train: Whether to track this prediction for training
            weight: Weight of this example for training
            example_idx: Example index for training queue (required if train=True)
            
        Returns:
            Predicted distributions for specified positions
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self(inputs, annotators, questions, embeddings)
            
            # Extract predictions for specific positions if provided
            if positions is not None:
                if isinstance(positions, list):
                    predictions = outputs[:, positions, :]
                else:
                    predictions = outputs[:, positions:positions+1, :]
            else:
                predictions = outputs
        
        # Track this prediction for training if required
        if train and example_idx is not None:
            batch_size = inputs.shape[0]
            
            for i in range(batch_size):
                # FIXED: Store in training queue with example_idx for current data access
                queue_entry = {
                    'example_idx': example_idx,
                    'positions': positions if positions is not None else list(range(inputs.shape[1])),
                    'weight': weight,
                    'timestamp': len(self.training_queue),
                    'needs_revisit': False
                }
                self.training_queue.append(queue_entry)
                
                # Keep prediction history for analysis (preserve original functionality)
                history_entry = {
                    'example_idx': example_idx,
                    'inputs': inputs[i].detach().cpu().clone(),
                    'annotators': annotators[i].detach().cpu().clone(),
                    'questions': questions[i].detach().cpu().clone(),
                    'embeddings': None if embeddings is None else embeddings[i].detach().cpu().clone(),
                    'positions': positions if positions is not None else list(range(inputs.shape[1])),
                    'weight': weight,
                    'timestamp': len(self.prediction_history),
                    'loss': None,
                    'needs_revisit': False
                }
                self.prediction_history.append(history_entry)
            
            logger.debug(f"Added prediction for example {example_idx} to training queue, size: {len(self.training_queue)}")
        
        return predictions
    
    def update_training_supervision(self, observed_values, variable_ids, example_indices=None):
        """
        Update training queue when new observations are made.
        
        Args:
            observed_values: Observed values to update with
            variable_ids: Variable IDs that were observed
            example_indices: Indices of examples to update (unused, kept for compatibility)
            
        Returns:
            Number of queue entries updated
        """
        if not variable_ids:
            return 0
        
        # Parse variable_ids to get (example_idx, position) pairs
        observed_positions = []
        for var_id in variable_ids:
            if isinstance(var_id, str) and var_id.startswith('example_') and '_position_' in var_id:
                parts = var_id.split('_')
                if len(parts) >= 4:
                    try:
                        example_idx = int(parts[1])
                        position = int(parts[3])
                        observed_positions.append((example_idx, position))
                    except ValueError:
                        continue
        
        if not observed_positions:
            logger.debug("No valid variable_ids to process")
            return 0
        
        # FIXED: Mark relevant training queue entries for revisiting
        updated_count = 0
        for queue_idx, queue_entry in enumerate(self.training_queue):
            entry_example_idx = queue_entry['example_idx']
            entry_positions = queue_entry['positions']
            
            # Check if this queue entry is affected by any observation
            for obs_example_idx, obs_position in observed_positions:
                if entry_example_idx == obs_example_idx and obs_position in entry_positions:
                    queue_entry['needs_revisit'] = True
                    self.examples_to_revisit.add(queue_idx)
                    updated_count += 1
                    logger.debug(f"Marked queue entry {queue_idx} for revisit (example {entry_example_idx}, position {obs_position})")
                    break
        
        # Also update prediction history for analysis
        for history_entry in self.prediction_history:
            if history_entry['example_idx'] in [ex_idx for ex_idx, _ in observed_positions]:
                history_entry['needs_revisit'] = True
        
        logger.debug(f"Updated supervision for {len(variable_ids)} variables, marked {updated_count} queue entries for revisit")
        return updated_count
    
    def compute_log_loss(self, outputs, targets, weights=None):
        """
        Compute log loss (cross-entropy) for predicted distributions.
        
        Args:
            outputs: Predicted logits [batch_size, sequence_length, max_choices]
            targets: Target values [batch_size, sequence_length, max_choices]
            weights: Optional weights for examples [batch_size]
            
        Returns:
            Log loss value
        """
        batch_size, seq_len, num_classes = outputs.shape
        loss = torch.zeros(1, device=self.device)
        
        for i in range(batch_size):
            for j in range(seq_len):
                target_idx = torch.argmax(targets[i, j]).item()
                example_loss = F.cross_entropy(
                    outputs[i:i+1, j], 
                    torch.tensor([target_idx], device=self.device)
                )
                
                if weights is not None:
                    example_loss *= weights[i]
                    
                loss += example_loss
        
        # Normalize
        if weights is not None:
            total_weight = weights.sum().item()
            loss = loss / max(1.0, total_weight)
        else:
            loss = loss / (batch_size * seq_len)
            
        return loss
    
    def train_on_examples_basic(self, examples_indices=None, epochs=1, batch_size=8, lr=1e-4):
        """
        Train the model on stored examples with prioritized revisiting.
        
        Args:
            examples_indices: Indices of queue entries to train on (default: all)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            List of training losses
        """
        # FIXED: Use training queue instead of training_examples
        if examples_indices is None:
            examples_indices = list(range(len(self.training_queue)))
        
        if not examples_indices:
            return []
        
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            np.random.shuffle(examples_indices)
            
            for batch_start in range(0, len(examples_indices), batch_size):
                batch_indices = examples_indices[batch_start:batch_start + batch_size]
                
                # FIXED: Get current data from dataset using training queue
                batch_examples = []
                for queue_idx in batch_indices:
                    if queue_idx >= len(self.training_queue):
                        continue
                    queue_entry = self.training_queue[queue_idx]
                    example_idx = queue_entry['example_idx']
                    current_data = self.dataset[example_idx]
                    
                    known_questions, inputs, answers, annotators, questions, embeddings = current_data
                    batch_examples.append({
                        'inputs': inputs.clone(),
                        'annotators': annotators.clone(),
                        'questions': questions.clone(),
                        'embeddings': embeddings.clone() if embeddings is not None else None,
                        'weight': queue_entry.get('weight', 1.0)
                    })
                
                if not batch_examples:
                    continue
                
                # Extract batch data
                batch_inputs = torch.stack([e['inputs'] for e in batch_examples]).to(self.device)
                batch_annotators = torch.stack([e['annotators'] for e in batch_examples]).to(self.device)
                batch_questions = torch.stack([e['questions'] for e in batch_examples]).to(self.device)
                batch_embeddings = torch.stack([e['embeddings'] for e in batch_examples]).to(self.device) if batch_examples[0]['embeddings'] is not None else None
                batch_weights = torch.tensor([e['weight'] for e in batch_examples]).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self(batch_inputs, batch_annotators, batch_questions, batch_embeddings)
                
                # Compute loss on all positions
                batch_targets = batch_inputs[:, :, 1:].clone()
                loss = self.compute_log_loss(outputs, batch_targets, batch_weights)
                
                if loss > 0:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # Store individual losses for analysis
                    for i, queue_idx in enumerate(batch_indices):
                        if queue_idx < len(self.prediction_history):
                            individual_loss = self.compute_log_loss(
                                outputs[i:i+1], 
                                batch_targets[i:i+1]
                            ).item()
                            self.prediction_history[queue_idx]['loss'] = individual_loss
                    
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({"batch_loss": loss.item(), "epoch": epoch})
            
            avg_epoch_loss = epoch_loss / max(1, batch_count)
            losses.append(avg_epoch_loss)
            self.training_losses.append(avg_epoch_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})
        
        # Clear revisit flags for trained examples
        for queue_idx in examples_indices:
            if queue_idx < len(self.training_queue):
                self.training_queue[queue_idx]['needs_revisit'] = False
            if queue_idx < len(self.prediction_history):
                self.prediction_history[queue_idx]['needs_revisit'] = False
        self.examples_to_revisit.clear()
        
        logger.info(f"Training completed - Final loss: {losses[-1]:.4f}")
        return losses

    def train_on_examples_random_masking(self, examples_indices=None, epochs=1, batch_size=8, lr=1e-4):
        """Train with random masking patterns."""
        # FIXED: Use training queue instead of training_examples
        if examples_indices is None:
            examples_indices = list(range(len(self.training_queue)))
        
        if not examples_indices:
            return []
        
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        # Original complete masking patterns (30+ patterns)
        masking_patterns = [
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        ]

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            np.random.shuffle(examples_indices)
            
            for batch_start in range(0, len(examples_indices), batch_size):
                batch_indices = examples_indices[batch_start:batch_start + batch_size]
                
                # FIXED: Get current data from dataset using training queue
                batch_examples = []
                for queue_idx in batch_indices:
                    if queue_idx >= len(self.training_queue):
                        continue
                    queue_entry = self.training_queue[queue_idx]
                    example_idx = queue_entry['example_idx']
                    current_data = self.dataset[example_idx]
                    
                    known_questions, inputs, answers, annotators, questions, embeddings = current_data
                    batch_examples.append({
                        'inputs': inputs.clone(),
                        'annotators': annotators.clone(),
                        'questions': questions.clone(),
                        'embeddings': embeddings.clone() if embeddings is not None else None,
                        'weight': queue_entry.get('weight', 1.0)
                    })
                
                if not batch_examples:
                    continue
                
                # Extract batch data
                batch_inputs = torch.stack([e['inputs'] for e in batch_examples]).to(self.device)
                batch_annotators = torch.stack([e['annotators'] for e in batch_examples]).to(self.device)
                batch_questions = torch.stack([e['questions'] for e in batch_examples]).to(self.device)
                batch_embeddings = torch.stack([e['embeddings'] for e in batch_examples]).to(self.device) if batch_examples[0]['embeddings'] is not None else None
                
                # Apply random masking pattern
                temp_inputs = batch_inputs.clone()
                pattern_idx = np.random.randint(0, len(masking_patterns))
                pattern = masking_patterns[pattern_idx]
                
                for b in range(temp_inputs.shape[0]):
                    for i in range(temp_inputs.shape[1]):
                        q_idx = batch_questions[b, i].item()
                        is_llm = (batch_annotators[b, i].item() == -1)
                        
                        # Only mask positions that are currently observed
                        if temp_inputs[b, i, 0] == 0:  # Currently observed
                            pattern_pos = 2 * q_idx + (0 if is_llm else 1)
                            if pattern_pos < len(pattern) and pattern[pattern_pos] == 1:
                                temp_inputs[b, i, 0] = 1  # Mask it
                                temp_inputs[b, i, 1:] = 0  # Zero out
                
                optimizer.zero_grad()
                outputs = self(temp_inputs, batch_annotators, batch_questions, batch_embeddings)
                
                # Compute loss on ALL positions (using original labels)
                batch_targets = batch_inputs[:, :, 1:].clone()
                loss = self.compute_log_loss(outputs, batch_targets)
                
                if loss > 0:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_count += 1
            
            avg_epoch_loss = epoch_loss / max(1, batch_count)
            losses.append(avg_epoch_loss)
            self.training_losses.append(avg_epoch_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"epoch_loss_random": avg_epoch_loss, "epoch": epoch})
        
        # Clear revisit flags
        for queue_idx in examples_indices:
            if queue_idx < len(self.training_queue):
                self.training_queue[queue_idx]['needs_revisit'] = False
        self.examples_to_revisit.clear()
        
        return losses

    def train_on_examples_dynamic_masking(self, examples_indices=None, epochs=5, batch_size=32, lr=1e-4, 
                                     num_patterns_per_example=5, visible_ratio=0.5):
        """
        Train model using dynamic masking patterns based on observed variables.
        
        Args:
            examples_indices: Indices of queue entries to train on (default: all)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            num_patterns_per_example: Number of different masking patterns to generate per example
            visible_ratio: Ratio of observed positions to keep visible (vs masked)
            
        Returns:
            List of training losses
        """
        # FIXED: Use training queue instead of training_examples
        if examples_indices is None:
            examples_indices = list(range(len(self.training_queue)))
        
        if not examples_indices:
            return []
        
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # Generate augmented training instances with dynamic masking
            augmented_examples = []
            
            for queue_idx in examples_indices:
                if queue_idx >= len(self.training_queue):
                    continue
                    
                queue_entry = self.training_queue[queue_idx]
                example_idx = queue_entry['example_idx']
                current_data = self.dataset[example_idx]
                
                known_questions, inputs, answers, annotators, questions, embeddings = current_data
                
                # Identify originally observed positions (mask bit = 0)
                observed_positions = []
                for pos in range(inputs.shape[0]):
                    if inputs[pos, 0] == 0:
                        observed_positions.append(pos)
                
                if len(observed_positions) == 0:
                    continue
                
                # Generate multiple masking patterns for this example
                for pattern_idx in range(num_patterns_per_example):
                    # Create copy of original example
                    augmented_example = {
                        'inputs': inputs.clone(),
                        'annotators': annotators.clone(),
                        'questions': questions.clone(),
                        'embeddings': embeddings.clone() if embeddings is not None else None,
                        'weight': queue_entry.get('weight', 1.0),
                        'original_observed_mask': (inputs[:, 0] == 0).float(),
                        'original_targets': inputs[:, 1:].clone()
                    }
                    
                    # Randomly select which observed positions to keep visible
                    num_visible = max(1, int(len(observed_positions) * visible_ratio))
                    if num_visible >= len(observed_positions):
                        visible_positions = observed_positions.copy()
                    else:
                        visible_positions = np.random.choice(
                            observed_positions, size=num_visible, replace=False
                        ).tolist()
                    
                    # Mask the non-visible observed positions
                    for pos in observed_positions:
                        if pos not in visible_positions:
                            augmented_example['inputs'][pos, 0] = 1  # Set mask bit
                            augmented_example['inputs'][pos, 1:] = 0  # Clear answer distribution
                    
                    augmented_examples.append(augmented_example)
            
            # Shuffle augmented examples
            np.random.shuffle(augmented_examples)
            
            # Train in batches
            for batch_start in range(0, len(augmented_examples), batch_size):
                batch_examples = augmented_examples[batch_start:batch_start + batch_size]
                
                if not batch_examples:
                    continue
                
                # Extract batch data
                batch_inputs = torch.stack([e['inputs'] for e in batch_examples]).to(self.device)
                batch_annotators = torch.stack([e['annotators'] for e in batch_examples]).to(self.device)
                batch_questions = torch.stack([e['questions'] for e in batch_examples]).to(self.device)
                batch_embeddings = torch.stack([e['embeddings'] for e in batch_examples]).to(self.device) if batch_examples[0]['embeddings'] is not None else None
                batch_weights = torch.tensor([e['weight'] for e in batch_examples]).to(self.device)
                
                # Extract original targets and observed masks
                batch_targets = torch.stack([e['original_targets'] for e in batch_examples]).to(self.device)
                batch_observed_mask = torch.stack([e['original_observed_mask'] for e in batch_examples]).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self(batch_inputs, batch_annotators, batch_questions, batch_embeddings)
                
                # Compute loss only for originally observed AND currently visible positions
                batch_size_actual, seq_len, num_classes = outputs.shape
                outputs_flat = outputs.view(-1, num_classes)
                targets_flat = batch_targets.view(-1, num_classes)

                # Get current mask state (0 = visible, 1 = masked)
                current_mask = batch_inputs[:, :, 0]
                currently_visible = (current_mask == 0).float()

                # Combine: originally observed AND currently visible
                loss_mask = batch_observed_mask * currently_visible
                loss_mask_flat = loss_mask.view(-1)

                # Compute loss only where mask allows
                log_probs = F.log_softmax(outputs_flat, dim=-1)
                loss_per_position = kl_criterion(log_probs.unsqueeze(0), targets_flat.unsqueeze(0))

                # Apply masking and weights
                weighted_loss = loss_per_position * loss_mask_flat

                if batch_weights.numel() > 0:
                    batch_weights_expanded = batch_weights.unsqueeze(1).expand(-1, seq_len).contiguous().view(-1)
                    weighted_loss = weighted_loss * batch_weights_expanded

                # Average loss over valid positions
                total_valid = loss_mask_flat.sum()
                if total_valid > 0:
                    loss = weighted_loss.sum() / total_valid
                else:
                    loss = weighted_loss.sum()
                
                if loss > 0:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({"batch_loss": loss.item(), "epoch": epoch})
            
            avg_epoch_loss = epoch_loss / max(1, batch_count)
            epoch_losses.append(avg_epoch_loss)
            self.training_losses.append(avg_epoch_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"epoch_loss_dynamic": avg_epoch_loss, "epoch": epoch})
        
        # Clear revisit flags
        for queue_idx in examples_indices:
            if queue_idx < len(self.training_queue):
                self.training_queue[queue_idx]['needs_revisit'] = False
        self.examples_to_revisit.clear()
        
        return epoch_losses
    
    def compute_total_loss(self, outputs, labels, inputs, questions, embeddings, full_supervision=False):
        """
        Compute total loss over all positions based on supervision type.
        Maintained for backward compatibility with activelearner.py.
        """
        batch_size = outputs.shape[0]
        var_num = outputs.shape[1]
        device = self.device
        loss = torch.tensor(0.0, device=device)
        position_count = 0
        
        for i in range(batch_size):
            for j in range(var_num):
                if inputs[i, j, 0] == 0 and full_supervision:
                    target_idx = torch.argmax(labels[i, j]).item()
                    target = torch.tensor([target_idx], device=device)
                    position_loss = F.cross_entropy(outputs[i:i+1, j], target)
                else:
                    probs = F.softmax(outputs[i, j], dim=0)
                    expected_loss = torch.tensor(0.0, device=device)
                    for class_idx in range(self.max_choices):
                        class_loss = F.cross_entropy(
                            outputs[i:i+1, j], 
                            torch.tensor([class_idx], device=device)
                        )
                        expected_loss += probs[class_idx] * class_loss
                    
                    position_loss = expected_loss
                
                loss += position_loss
                position_count += 1
        
        # Average loss per position
        if position_count > 0:
            loss = loss / position_count
                
        return loss