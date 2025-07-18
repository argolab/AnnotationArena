"""
Imputer for Active Learner framework with integrated query stream for pattern generation.

Author: Prabhav Singh / Haojun Shi
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
from tqdm.auto import tqdm
import random
import json
import os

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
        
        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.temp_projection = nn.Linear(hidden_dim, 1, bias=False)
        with torch.no_grad():
            self.temp_projection.weight.normal_(0, 0.1)
        
        with torch.no_grad():
            jl_matrix = torch.randn(hidden_dim, hidden_dim) / math.sqrt(hidden_dim)
            self.Q.weight.copy_(jl_matrix)
            self.K.weight.copy_(jl_matrix.clone())
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, param_states, questions, mask):
        """Fully vectorized forward pass."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
        
        if mask_bool.sum() == 0:
            return hidden_states, param_states
        
        Q = self.Q(hidden_states)
        K = self.K(hidden_states)
        
        variable_temps = F.relu(self.temp_projection(hidden_states)) + 10.0

        temp_matrix = variable_temps.expand(-1, -1, seq_len)

        scores = torch.bmm(Q, K.transpose(-2, -1)) / temp_matrix
        
        question_mask = questions.unsqueeze(-1) == questions.unsqueeze(-2)

        scores = scores.masked_fill(~question_mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        should_smooth = mask_bool.unsqueeze(-1).expand(-1, -1, seq_len)
        
        attention_weights = attention_weights * should_smooth.float()
        
        eye_mask = torch.eye(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1, -1)
        no_smooth_mask = (~mask_bool).unsqueeze(-1).expand(-1, -1, seq_len)
        attention_weights = attention_weights + eye_mask * no_smooth_mask.float()
        
        smoothed_params = torch.bmm(attention_weights, param_states)
        
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
        
        if len(embeddings.shape) == 4:
            embeddings = embeddings.squeeze(0)
            
        combined_embeds = question_embeds + annotator_embeds
        feature_x = torch.cat((combined_embeds, embeddings, x[:,:,1:]), dim=-1)
        param_x = x[:,:,1:].clone()
        
        # Initialize query stream: [batch_size, seq_len, 2] - [binary_decision, weight]
        query_x = torch.zeros(batch_size, x.shape[1], 2, device=x.device)
        
        return feature_x, param_x, query_x


class EncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feed-forward networks."""

    def __init__(self, feature_dim, param_dim, attention_heads, num_question_types, dropout=0.3):
        """Initialize encoder layer."""
        super().__init__()
        self.feature_dim = feature_dim 
        self.param_dim = param_dim  
        self.attention_heads = attention_heads
        
        # Feature stream attention
        self.Q = nn.Linear(feature_dim, feature_dim)
        self.K = nn.Linear(feature_dim, feature_dim)
        self.V = nn.Linear(feature_dim, feature_dim)
        self.out = nn.Linear(feature_dim, feature_dim)
        
        # Query stream processing
        self.query_Q = nn.Linear(2, 2)
        self.query_K = nn.Linear(2, 2)
        self.query_V = nn.Linear(2, 2)
        self.query_out = nn.Linear(2, 2)
        
        self.norm_1 = NormLayer(feature_dim)
        self.norm_2 = NormLayer(feature_dim)
        self.norm_query = NormLayer(2)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_query = nn.Dropout(dropout)
        
        self.ff = FeedForward(feature_dim, dropout=dropout)
        self.param_update = nn.Linear(feature_dim + param_dim, param_dim)

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
    
    def query_attention(self, query_x):
        """Apply attention to query stream."""
        Q = self.query_Q(query_x)
        K = self.query_K(query_x)
        V = self.query_V(query_x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(2)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_query(scores)
        scores = torch.matmul(scores, V)
        output = self.query_out(scores)
        return output

    def forward(self, feature_x, param_x, query_x, questions, mask):
        """Process features through attention, feed-forward, and smoothing."""
        batch_size = feature_x.shape[0]
        
        # Feature stream processing
        feature_x = self.norm_1(feature_x)
        attention_output = self.multihead_attention(feature_x, batch_size)
        feature_x = feature_x + self.dropout_1(attention_output)
        
        feature_x_ff = self.norm_2(feature_x)
        feature_x = feature_x + self.dropout_2(self.ff(feature_x_ff))
        
        # Query stream self-attention
        query_x_norm = self.norm_query(query_x)
        query_attention_output = self.query_attention(query_x_norm)
        query_x = query_x + self.dropout_query(query_attention_output)
        
        query_x_norm = self.norm_query(query_x)
        query_attention_output = self.query_attention(query_x_norm)
        query_x = query_x + self.dropout_query(query_attention_output)

        # Param update and smoothing
        combined = torch.cat([feature_x, param_x], dim=-1)
        param_x = self.param_update(combined)
        
        feature_x, param_x = self.smoothing(feature_x, param_x, questions, mask)
        
        return feature_x, param_x, query_x


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
        feature_x, param_x, query_x = self.position_encoder(x, annotators, questions, embeddings)
        
        mask = x[:, :, 0]
        
        for layer in self.layers:
            feature_x, param_x, query_x = layer(feature_x, param_x, query_x, questions, mask)
        
        return feature_x, param_x, query_x


class ImputerEmbedding(nn.Module):
    """
    Imputer model for predicting missing annotations with intelligent masking pattern generation.
    """
    
    def __init__(self, 
                 question_num=7, 
                 max_choices=5, 
                 encoder_layers_num=6, 
                 attention_heads=4, 
                 hidden_dim=64, 
                 num_annotator=15, 
                 annotator_embedding_dim=8, 
                 dropout=0.1,
                 training_queue_size=None):
        """Initialize Imputer model with integrated query stream."""
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.question_num = question_num
        self.max_choices = max_choices
        self.encoder = Encoder(question_num, max_choices, encoder_layers_num, attention_heads, 
                               num_annotator, annotator_embedding_dim, dropout)

        self.training_queue = []
        self.recent_indicators = []
        self.prediction_history = []
        self.examples_to_revisit = set()
        self.training_losses = []
        self.dataset = None
        self.training_queue_size = training_queue_size
        self.unique_examples = []
        
        self.pattern_effectiveness_history = []
        self.current_cycle = 0
        
        self.historical_patterns = []
        
        logger.info(f"ImputerEmbedding initialized: {question_num} questions, {max_choices} choices, {num_annotator} annotators")
    
    def set_dataset(self, dataset):
        """Set dataset reference for current data access."""
        self.dataset = dataset
        logger.debug(f"Dataset set with {len(dataset)} examples")
    
    def forward(self, x, annotators, questions, embeddings):
        """Forward pass through the model."""
        feature_x, param_x, query_x = self.encoder(x, annotators, questions, embeddings)
        return param_x, query_x
    
    def set_current_cycle(self, cycle):
        """Set current cycle for tracking"""
        self.current_cycle = cycle
    
    def collect_pattern_effectiveness(self, example_idx, masking_patterns, pattern_losses):
        """Simple collection of what worked"""
        self.pattern_effectiveness_history.append({
            'example_idx': example_idx,
            'cycle': self.current_cycle,
            'patterns': masking_patterns,
            'losses': pattern_losses,
            'timestamp': time.time()
        })
    
    def export_pattern_logs(self, output_dir):
        """Simple JSON export"""
        log_file = os.path.join(output_dir, f"pattern_effectiveness_{self.current_cycle}.json")
        with open(log_file, 'w') as f:
            json.dump(self.pattern_effectiveness_history, f, indent=2, default=str)
        logger.info(f"Exported pattern effectiveness logs to {log_file}")
    
    def collect_historical_pattern(self, current_state, query_pattern, cycle):
        """
        Collect (current_state, historical_pattern) training pairs for professor's approach
        
        Args:
            current_state: Current observed positions [14] (0=masked, 1=observed)
            query_pattern: Which positions were queried to reach this state [14] (0=not_queried, 1=queried) 
            cycle: Current active learning cycle
        """
        self.historical_patterns.append({
            'current_state': current_state.tolist() if isinstance(current_state, torch.Tensor) else current_state,
            'query_pattern': query_pattern.tolist() if isinstance(query_pattern, torch.Tensor) else query_pattern,
            'cycle': cycle,
            'timestamp': time.time()
        })
    
    def compute_query_stream_loss(self, query_predictions, example_idx):
        """
        Train query stream to predict historical query patterns (professor's approach)
        
        Given current state, predict which positions were likely queried historically
        """
        if len(self.historical_patterns) < 10:
            return torch.tensor(0.0, device=self.device)
        
        current_data = self.dataset[example_idx]
        current_state = (current_data[1][:, 0] == 0).float().to(self.device)
        
        relevant_patterns = []
        for pattern in self.historical_patterns:
            historical_state = torch.tensor(pattern['current_state'], device=self.device).float()
            if torch.all(current_state <= historical_state):
                relevant_patterns.append(pattern)
        
        if not relevant_patterns:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0
        for pattern in relevant_patterns:
            target_query_pattern = torch.tensor(pattern['query_pattern'], device=self.device).float()
            
            # Use query stream predictions - binary decisions
            predicted_query_probs = torch.sigmoid(query_predictions[0, :, 0])
            
            loss = F.binary_cross_entropy(predicted_query_probs, target_query_pattern)
            total_loss += loss
        
        return total_loss / len(relevant_patterns)

    def generate_masking_pattern_from_query_stream(self, inputs, annotators, questions, embeddings, visible_ratio):
        """
        Generate intelligent masking pattern using query stream predictions.
        
        Args:
            inputs: Input tensor [sequence_length, input_dim]
            annotators: Annotator indices [sequence_length]
            questions: Question indices [sequence_length]
            embeddings: Text embeddings [sequence_length, embedding_dim]
            visible_ratio: Ratio of observed positions to keep visible
            
        Returns:
            List of positions to mask
        """
        observed_positions = [pos for pos in range(inputs.shape[0]) if inputs[pos, 0] == 0]
        
        if len(observed_positions) == 0:
            return []
        
        inputs_batch = inputs.unsqueeze(0).to(self.device)
        annotators_batch = annotators.unsqueeze(0).to(self.device)
        questions_batch = questions.unsqueeze(0).to(self.device)
        embeddings_batch = embeddings.unsqueeze(0).to(self.device) if embeddings is not None else None
        
        with torch.no_grad():
            _, query_predictions = self(inputs_batch, annotators_batch, questions_batch, embeddings_batch)
            
            # Get query decisions and weights
            query_probs = torch.sigmoid(query_predictions[0, :, 0])  # Binary decisions
            query_weights = torch.softmax(query_predictions[0, :, 1], dim=0)  # Weights
            
            # Combine probability and weight for masking decisions
            combined_scores = query_probs * query_weights
        
        # Apply constraint: only observed positions can be masked
        observed_mask = torch.zeros(inputs.shape[0], device=self.device)
        observed_mask[observed_positions] = 1.0
        
        constrained_scores = combined_scores * observed_mask + (1 - observed_mask) * (-1e9)
        
        num_to_mask = max(1, len(observed_positions) - int(len(observed_positions) * visible_ratio))
        
        if num_to_mask >= len(observed_positions):
            return observed_positions
        
        probs = F.softmax(constrained_scores, dim=0)
        
        try:
            masked_indices = torch.multinomial(probs, num_to_mask, replacement=False)
            return masked_indices.cpu().tolist()
        except:
            logger.info('ERROR! ERROR! Stop Code.')
            return random.sample(observed_positions, num_to_mask)

    def compute_log_loss(self, outputs, targets, weights=None):
        """
        Compute log loss for training.
        
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
        
        if weights is not None:
            total_weight = weights.sum().item()
            loss = loss / max(1.0, total_weight)
        else:
            loss = loss / (batch_size * seq_len)
            
        return loss

    def replace_training_queue_entry(self, new_entry, clear_buffer_size=1):
        old_indices = [i for i in range(len(self.unique_examples)) if not self.recent_indicators[i]]
        
        indices = random.sample(old_indices, clear_buffer_size)
        indices.sort(reverse=True)
        
        for index in indices:
            del self.unique_examples[index]
            del self.recent_indicators[index]
        
        index_to_remove = []
        for i, entry in enumerate(self.training_queue):
            if entry["example_idx"] in indices:
                index_to_remove.append(i)
        index_to_remove.sort(reverse=True)
        for index in index_to_remove:
            del self.training_queue[index]

        self.training_queue.append(new_entry)
    
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
            outputs, _ = self(inputs, annotators, questions, embeddings)
            
            if positions is not None:
                if isinstance(positions, list):
                    predictions = outputs[:, positions, :]
                else:
                    predictions = outputs[:, positions:positions+1, :]
            else:
                predictions = outputs
        
        if train and example_idx is not None:
            batch_size = inputs.shape[0]
            
            for i in range(batch_size):
                queue_entry = {
                    'example_idx': example_idx,
                    'positions': positions if positions is not None else list(range(inputs.shape[1])),
                    'weight': weight,
                    'timestamp': len(self.training_queue),
                    'needs_revisit': False
                }
                if example_idx not in self.unique_examples:
                    self.unique_examples.append(example_idx)
                    self.recent_indicators.append(True)

                if self.training_queue_size is not None and len(self.unique_examples) == self.training_queue_size:
                    self.replace_training_queue_entry(queue_entry)
                else:
                    self.training_queue.append(queue_entry)

                assert len(self.unique_examples) == len(self.recent_indicators)

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
            observed_values: List of observed values
            variable_ids: List of variable IDs that were observed
            example_indices: Optional list of example indices
            
        Returns:
            int: Number of training entries updated
        """
        updated_count = 0
        
        for variable_id in variable_ids:
            parts = variable_id.split('_')
            if len(parts) >= 4 and parts[0] == 'example' and parts[2] == 'position':
                try:
                    example_idx = int(parts[1])
                    position = int(parts[3])
                    
                    for entry in self.training_queue:
                        if entry['example_idx'] == example_idx:
                            if position in entry['positions']:
                                entry['needs_revisit'] = True
                                self.examples_to_revisit.add(example_idx)
                                updated_count += 1
                                
                except (ValueError, IndexError):
                    continue
        
        logger.debug(f"Updated {updated_count} training queue entries with new observations")
        return updated_count

    def train_on_examples_dynamic_masking(self, examples_indices=None, epochs=5, batch_size=32, lr=1e-4, num_patterns_per_example=5, visible_ratio=0.5, masking_lambda=0.1):
        """
        Train model using intelligent masking patterns based on query stream predictions.
        
        Args:
            examples_indices: Indices of queue entries to train on (default: all)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            num_patterns_per_example: Number of different masking patterns to generate per example
            visible_ratio: Ratio of observed positions to keep visible (vs masked)
            masking_lambda: Weight for query stream loss component
            
        Returns:
            List of training losses
        """
        if examples_indices is None:
            examples_indices = list(range(len(self.training_queue)))

        if not examples_indices:
            return []

        unique_examples = {}
        for queue_idx in examples_indices:
            if queue_idx < len(self.training_queue):
                queue_entry = self.training_queue[queue_idx]
                example_idx = queue_entry['example_idx']
                if example_idx not in unique_examples:
                    unique_examples[example_idx] = queue_entry

        if not unique_examples:
            return []
        
        logger.info(f'Unique Examples Training On : {len(unique_examples)}')

        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')

        epoch_losses = []

        start_time = time.time()
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_query_loss = 0.0
            batch_count = 0
            
            augmented_examples = []
            example_masking_patterns = {}
            
            for example_idx, queue_entry in unique_examples.items():
                current_data = self.dataset[example_idx]
                
                known_questions, inputs, answers, annotators, questions, embeddings = current_data
                
                observed_positions = [pos for pos in range(inputs.shape[0]) if inputs[pos, 0] == 0]
                
                if len(observed_positions) == 0:
                    continue
                
                example_patterns = []
                
                for pattern_idx in range(num_patterns_per_example):
                    augmented_example = {
                        'inputs': inputs.clone(),
                        'annotators': annotators.clone(),
                        'questions': questions.clone(),
                        'embeddings': embeddings.clone() if embeddings is not None else None,
                        'weight': queue_entry.get('weight', 1.0),
                        'original_observed_mask': (inputs[:, 0] == 0).float(),
                        'original_targets': inputs[:, 1:].clone(),
                        'example_idx': example_idx
                    }
                    
                    if epoch == 0:
                        num_visible = max(1, int(len(observed_positions) * visible_ratio))
                        if num_visible >= len(observed_positions):
                            visible_positions = observed_positions.copy()
                        else:
                            visible_positions = np.random.choice(
                                observed_positions, size=num_visible, replace=False
                            ).tolist()
                        
                        positions_to_mask = [pos for pos in observed_positions if pos not in visible_positions]
                    else:
                        positions_to_mask = self.generate_masking_pattern_from_query_stream(
                            inputs, annotators, questions, embeddings, visible_ratio
                        )
                    
                    masking_pattern = torch.zeros(inputs.shape[0])
                    for pos in positions_to_mask:
                        augmented_example['inputs'][pos, 0] = 1
                        augmented_example['inputs'][pos, 1:] = 0
                        masking_pattern[pos] = 1.0
                    
                    example_patterns.append(masking_pattern)
                    augmented_examples.append(augmented_example)
                
                example_masking_patterns[example_idx] = example_patterns
            
            np.random.shuffle(augmented_examples)
            
            for batch_start in range(0, len(augmented_examples), batch_size):
                batch_examples = augmented_examples[batch_start:batch_start + batch_size]
                
                if not batch_examples:
                    continue
                
                batch_inputs = torch.stack([e['inputs'] for e in batch_examples]).to(self.device)
                batch_annotators = torch.stack([e['annotators'] for e in batch_examples]).to(self.device)
                batch_questions = torch.stack([e['questions'] for e in batch_examples]).to(self.device)
                batch_embeddings = torch.stack([e['embeddings'] for e in batch_examples]).to(self.device) if batch_examples[0]['embeddings'] is not None else None
                batch_weights = torch.tensor([e['weight'] for e in batch_examples]).to(self.device)
                
                batch_targets = torch.stack([e['original_targets'] for e in batch_examples]).to(self.device)
                batch_observed_mask = torch.stack([e['original_observed_mask'] for e in batch_examples]).to(self.device)
                
                optimizer.zero_grad()
                outputs, query_predictions = self(batch_inputs, batch_annotators, batch_questions, batch_embeddings)
                
                batch_size_actual, seq_len, num_classes = outputs.shape
                outputs_flat = outputs.view(-1, num_classes)
                targets_flat = batch_targets.view(-1, num_classes)

                current_mask = batch_inputs[:, :, 0]
                currently_visible = (current_mask == 0).float()

                # Two-component loss: reconstruction (masked) + consistency (observed)
                artificially_masked = batch_observed_mask * (1 - currently_visible)
                still_observed = batch_observed_mask * currently_visible 

                # Combined loss with different weights
                reconstruction_weight = 0.5
                consistency_weight = 0.5
                loss_mask = reconstruction_weight * artificially_masked + consistency_weight * still_observed
                loss_mask_flat = loss_mask.view(-1)

                log_probs = F.log_softmax(outputs_flat, dim=-1)
                loss_per_position = kl_criterion(log_probs.unsqueeze(0), targets_flat.unsqueeze(0))

                weighted_loss = loss_per_position * loss_mask_flat

                if batch_weights.numel() > 0:
                    batch_weights_expanded = batch_weights.unsqueeze(1).expand(-1, seq_len).contiguous().view(-1)
                    weighted_loss = weighted_loss * batch_weights_expanded

                total_valid = loss_mask_flat.sum()
                if total_valid > 0:
                    main_loss = weighted_loss.sum() / total_valid
                else:
                    main_loss = weighted_loss.sum()
                
                query_loss = torch.tensor(0.0, device=self.device)
                if len(batch_examples) > 0:
                    for i, example in enumerate(batch_examples):
                        if i < batch_inputs.shape[0]:
                            example_idx = example['example_idx']
                            example_query_loss = self.compute_query_stream_loss(query_predictions[i:i+1], example_idx)
                            query_loss += example_query_loss
                
                total_loss = main_loss + masking_lambda * query_loss
                
                if total_loss > 0:
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += main_loss.item()
                    epoch_query_loss += query_loss.item()
                    batch_count += 1
                    
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({
                            "batch_loss": main_loss.item(), 
                            "batch_query_loss": query_loss.item(),
                            "epoch": epoch
                        })
            
            for example_idx in unique_examples.keys():
                if example_idx in example_masking_patterns:
                    patterns_as_lists = [pattern.tolist() if isinstance(pattern, torch.Tensor) else pattern 
                                       for pattern in example_masking_patterns[example_idx]]
                    pattern_losses = [epoch_loss / max(1, batch_count)] * len(patterns_as_lists)
                    self.collect_pattern_effectiveness(example_idx, patterns_as_lists, pattern_losses)
            
            avg_epoch_loss = epoch_loss / max(1, batch_count)
            avg_query_loss = epoch_query_loss / max(1, batch_count)
            epoch_losses.append(avg_epoch_loss)
            self.training_losses.append(avg_epoch_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Query Loss: {avg_query_loss:.4f}")
            
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "epoch_loss_dynamic": avg_epoch_loss, 
                    "epoch_query_loss": avg_query_loss,
                    "epoch": epoch
                })

        end_time = time.time()
        logger.info(f'Time taken for all Epoch: {end_time - start_time}')

        for queue_idx in examples_indices:
            if queue_idx < len(self.training_queue):
                self.training_queue[queue_idx]['needs_revisit'] = False
        for i in range(len(self.recent_indicators)):
            self.recent_indicators[i] = False
        logger.info(f"Current examples in the training buffer: {self.unique_examples}")
        logger.debug(f"Indicators for recent examples: {self.recent_indicators}")
        self.examples_to_revisit.clear()

        return epoch_losses
    
    def compute_total_loss(self, outputs, labels, inputs, questions, embeddings, full_supervision=False):
        """
        Compute total loss over all positions based on supervision type.
        Maintained for backward compatibility with activelearner.py.
        """
        if full_supervision:
            batch_size, seq_len, num_classes = outputs.shape
            
            total_loss = 0.0
            total_positions = 0
            
            for i in range(batch_size):
                for j in range(seq_len):
                    target_idx = torch.argmax(labels[i, j]).item()
                    position_loss = F.cross_entropy(
                        outputs[i:i+1, j], 
                        torch.tensor([target_idx], device=self.device)
                    )
                    total_loss += position_loss
                    total_positions += 1
            
            return total_loss / max(1, total_positions)
        else:
            return self.compute_log_loss(outputs, labels)