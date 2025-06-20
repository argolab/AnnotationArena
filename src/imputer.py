"""
Imputer {Training and Prediction} for Active Learner framework.

Author: Prabhav Singh / Haojun Shi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import copy
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Wandb not available, logging disabled")

class NormLayer(nn.Module):
    """Normalization layer."""
    
    def __init__(self, features, eps=1e-6):
        super(NormLayer, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class FeedForward(nn.Module):
    """Feed forward network."""
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class FullyVectorizedSimilaritySmoothing(nn.Module):
    """Fully vectorized similarity smoothing layer."""
    
    def __init__(self, hidden_dim, param_dim, num_question_types, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.param_dim = param_dim
        self.num_question_types = num_question_types
        
        self.similarity_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.confidence_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, feature_x, param_x, questions):
        batch_size, seq_len, _ = feature_x.shape
        device = feature_x.device
        
        similarities = self.similarity_network(feature_x).squeeze(-1)
        confidences = self.confidence_network(feature_x).squeeze(-1)
        
        question_mask = questions.unsqueeze(-1) == questions.unsqueeze(-2)
        similarities_masked = similarities.unsqueeze(-1) * question_mask.float()
        
        attention_weights = F.softmax(similarities_masked, dim=-2)
        smoothed_params = torch.bmm(attention_weights.transpose(-2, -1), param_x)
        
        final_params = confidences.unsqueeze(-1) * param_x + (1 - confidences.unsqueeze(-1)) * smoothed_params
        
        return final_params

class Positional_Encoder(nn.Module):
    """Positional encoder for embedding questions and annotators."""
    
    def __init__(self, question_num, max_choices, num_annotator, annotator_embedding_dim):
        super().__init__()
        self.question_num = question_num
        self.max_choices = max_choices
        self.annotator_embedding = nn.Parameter(torch.randn(num_annotator + 1, annotator_embedding_dim))
        self.question_embedding = nn.Parameter(torch.randn(question_num, annotator_embedding_dim))
        torch.nn.init.kaiming_normal_(self.annotator_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.question_embedding, mode='fan_out', nonlinearity='relu')
        self.num_annotator = num_annotator

    def forward(self, x, annotators, questions, embeddings):
        question_embeds = self.question_embedding[questions]
        annotators = torch.where(annotators < 0, torch.full_like(annotators, self.num_annotator), annotators)
        annotator_embeds = self.annotator_embedding[annotators]
        
        if len(embeddings.shape) == 4:
            embeddings = embeddings.squeeze(0)
            
        combined_embeds = question_embeds + annotator_embeds
        feature_x = torch.cat((combined_embeds, embeddings, x[:,:,1:]), dim=-1)
        param_x = x[:,:,1:].clone()
        
        return feature_x, param_x

class EncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feed-forward networks."""
    
    def __init__(self, feature_dim, param_dim, attention_heads, num_question_types, dropout=0.3):
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

        self.smoothing = FullyVectorizedSimilaritySmoothing(
            hidden_dim=feature_dim,
            param_dim=param_dim, 
            num_question_types=num_question_types,
            dropout=dropout
        )

    def multihead_attention(self, feature_x, batch_size):
        q = self.Q(feature_x).view(batch_size, -1, self.attention_heads, self.feature_dim // self.attention_heads).transpose(1, 2)
        k = self.K(feature_x).view(batch_size, -1, self.attention_heads, self.feature_dim // self.attention_heads).transpose(1, 2)
        v = self.V(feature_x).view(batch_size, -1, self.attention_heads, self.feature_dim // self.attention_heads).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.feature_dim // self.attention_heads)
        scores = F.softmax(scores, dim=-1)
        
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)
        return self.out(out)

    def forward(self, feature_x, param_x, questions, mask):
        batch_size = feature_x.shape[0]
        
        attended_features = self.multihead_attention(feature_x, batch_size)
        feature_x = self.norm_1(feature_x + self.dropout_1(attended_features))
        
        ff_output = self.ff(feature_x)
        feature_x = self.norm_2(feature_x + self.dropout_2(ff_output))
        
        combined = torch.cat([feature_x, param_x], dim=-1)
        param_x = self.param_update(combined)
        
        param_x = self.smoothing(feature_x, param_x, questions)
        
        return feature_x, param_x

class Encoder(nn.Module):
    """Multi-layer encoder."""
    
    def __init__(self, question_num, max_choices, encoder_num, attention_heads, 
                 num_annotator, annotator_embedding_dim, dropout=0.1):
        super().__init__()
        self.feature_dim = annotator_embedding_dim + max_choices + 384
        self.param_dim = max_choices
        self.position_encoder = Positional_Encoder(question_num, max_choices, num_annotator, annotator_embedding_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(self.feature_dim, self.param_dim, attention_heads, question_num, dropout)
            for _ in range(encoder_num)
        ])
        self.annotator_embedding_dim = annotator_embedding_dim

    def forward(self, x, annotators, questions, embeddings):
        feature_x, param_x = self.position_encoder(x, annotators, questions, embeddings)
        
        mask = x[:, :, 0]
        
        for layer in self.layers:
            feature_x, param_x = layer(feature_x, param_x, questions, mask)
        
        return param_x

class ImputerEmbedding(nn.Module):
    """Imputer model for predicting missing annotations with fixed training queue system."""
    
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
        
        # Fixed training queue system
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
        """Predict distributions for specified positions."""
        self.eval()
        
        with torch.no_grad():
            outputs = self(inputs, annotators, questions, embeddings)
            
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
                # Store in training queue with example_idx for current data access
                queue_entry = {
                    'example_idx': example_idx,
                    'positions': positions if positions is not None else list(range(inputs.shape[1])),
                    'weight': weight,
                    'timestamp': len(self.training_queue),
                    'needs_revisit': False
                }
                self.training_queue.append(queue_entry)
                
                # Keep prediction history for analysis (original functionality)
                history_entry = {
                    'example_idx': example_idx,
                    'inputs_snapshot': inputs[i].detach().cpu().clone(),
                    'annotators': annotators[i].detach().cpu().clone(),
                    'questions': questions[i].detach().cpu().clone(),
                    'embeddings': None if embeddings is None else embeddings[i].detach().cpu().clone(),
                    'positions': positions,
                    'weight': weight,
                    'timestamp': len(self.prediction_history),
                    'loss': None,
                    'needs_revisit': False
                }
                self.prediction_history.append(history_entry)
            
            logger.debug(f"Added prediction for example {example_idx} to training queue, size: {len(self.training_queue)}")
        
        return predictions
    
    def update_training_supervision(self, observed_values, variable_ids, example_indices=None):
        """Update training queue when new observations are made."""
        
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
        
        # Mark relevant training queue entries for revisiting
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
    
    def train_on_examples_basic(self, examples_indices=None, epochs=1, batch_size=8, lr=1e-4):
        """Train the model on stored examples with prioritized revisiting."""
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
                
                # Get current data from dataset using training queue
                batch_examples = []
                for queue_idx in batch_indices:
                    if queue_idx >= len(self.training_queue):
                        continue
                    queue_entry = self.training_queue[queue_idx]
                    example_idx = queue_entry['example_idx']
                    current_data = self.dataset[example_idx]
                    
                    # Prepare example data
                    known_questions, inputs, answers, annotators, questions, embeddings = current_data
                    example = {
                        'inputs': inputs.clone(),
                        'annotators': annotators.clone(), 
                        'questions': questions.clone(),
                        'embeddings': embeddings.clone() if embeddings is not None else torch.zeros(inputs.shape[0], 384),
                        'weight': queue_entry.get('weight', 1.0)
                    }
                    batch_examples.append(example)
                
                if not batch_examples:
                    continue
                    
                # Extract batch data
                batch_inputs = torch.stack([e['inputs'] for e in batch_examples]).to(self.device)
                batch_annotators = torch.stack([e['annotators'] for e in batch_examples]).to(self.device)
                batch_questions = torch.stack([e['questions'] for e in batch_examples]).to(self.device)
                batch_embeddings = torch.stack([e['embeddings'] for e in batch_examples]).to(self.device)
                batch_weights = torch.tensor([e['weight'] for e in batch_examples]).to(self.device)
                
                optimizer.zero_grad()
                outputs = self(batch_inputs, batch_annotators, batch_questions, batch_embeddings)
                
                # Compute loss on observed positions (where mask bit is 0)
                batch_targets = batch_inputs[:, :, 1:].clone()
                observed_mask = (batch_inputs[:, :, 0] == 0).float().unsqueeze(-1).expand_as(outputs)
                masked_outputs = outputs * observed_mask
                masked_targets = batch_targets * observed_mask
                
                if observed_mask.sum() > 0:
                    loss = self.compute_log_loss(masked_outputs, masked_targets, batch_weights)
                    loss.backward()
                    optimizer.step()
                    
                    # Record individual losses
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
        """Train with random masking patterns using current dataset state."""
        if examples_indices is None:
            examples_indices = list(range(len(self.training_queue)))
        
        if not examples_indices:
            return []
        
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        losses = []
        
        # Fixed masking patterns from original
        masking_patterns = [
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        ]
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            np.random.shuffle(examples_indices)
            
            for batch_start in range(0, len(examples_indices), batch_size):
                batch_indices = examples_indices[batch_start:batch_start + batch_size]
                
                # Get current data from dataset using training queue
                batch_examples = []
                for queue_idx in batch_indices:
                    if queue_idx >= len(self.training_queue):
                        continue
                    queue_entry = self.training_queue[queue_idx]
                    example_idx = queue_entry['example_idx']
                    current_data = self.dataset[example_idx]
                    
                    known_questions, inputs, answers, annotators, questions, embeddings = current_data
                    example = {
                        'inputs': inputs.clone(),
                        'annotators': annotators.clone(),
                        'questions': questions.clone(), 
                        'embeddings': embeddings.clone() if embeddings is not None else torch.zeros(inputs.shape[0], 384),
                        'weight': queue_entry.get('weight', 1.0)
                    }
                    batch_examples.append(example)
                
                if not batch_examples:
                    continue
                
                batch_inputs = torch.stack([e['inputs'] for e in batch_examples]).to(self.device)
                batch_annotators = torch.stack([e['annotators'] for e in batch_examples]).to(self.device)
                batch_questions = torch.stack([e['questions'] for e in batch_examples]).to(self.device)
                batch_embeddings = torch.stack([e['embeddings'] for e in batch_examples]).to(self.device)
                
                # Apply random masking pattern
                pattern = masking_patterns[np.random.randint(len(masking_patterns))]
                temp_inputs = batch_inputs.clone()
                
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
        """Train model using dynamic masking patterns based on observed variables."""
        if examples_indices is None:
            examples_indices = list(range(len(self.training_queue)))
        
        if not examples_indices:
            return []
        
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        epoch_losses = []
        kl_criterion = nn.KLDivLoss(reduction='none', log_target=False)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            np.random.shuffle(examples_indices)
            
            for batch_start in range(0, len(examples_indices), batch_size):
                batch_indices = examples_indices[batch_start:batch_start + batch_size]
                
                # Get current data from dataset using training queue
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
                        'embeddings': embeddings.clone() if embeddings is not None else torch.zeros(inputs.shape[0], 384),
                        'weight': queue_entry.get('weight', 1.0),
                        'original_targets': answers.clone()
                    })
                
                if not batch_examples:
                    continue
                
                batch_inputs = torch.stack([ex['inputs'] for ex in batch_examples]).to(self.device)
                batch_annotators = torch.stack([ex['annotators'] for ex in batch_examples]).to(self.device)
                batch_questions = torch.stack([ex['questions'] for ex in batch_examples]).to(self.device)
                batch_embeddings = torch.stack([ex['embeddings'] for ex in batch_examples]).to(self.device)
                batch_targets = torch.stack([ex['original_targets'] for ex in batch_examples]).to(self.device)
                batch_weights = torch.tensor([ex['weight'] for ex in batch_examples]).to(self.device)
                
                # Track which positions were originally observed
                batch_observed_mask = (batch_inputs[:, :, 0] == 0).float()
                
                for pattern_idx in range(num_patterns_per_example):
                    visible_positions = torch.rand_like(batch_observed_mask) < visible_ratio
                    currently_visible = visible_positions.float()
                    
                    # Create masked inputs
                    temp_inputs = batch_inputs.clone()
                    temp_inputs[:, :, 0] = (1.0 - currently_visible)  # 1 = masked, 0 = visible
                    temp_inputs[:, :, 1:] = temp_inputs[:, :, 1:] * currently_visible.unsqueeze(-1)
                    
                    optimizer.zero_grad()
                    outputs = self(temp_inputs, batch_annotators, batch_questions, batch_embeddings)
                    
                    # Reshape for loss computation
                    outputs_flat = outputs.view(-1, outputs.shape[-1])
                    targets_flat = batch_targets.view(-1, batch_targets.shape[-1])
                    
                    # Compute loss only on originally observed AND currently visible positions
                    loss_mask = batch_observed_mask * currently_visible
                    loss_mask_flat = loss_mask.view(-1)
                    
                    # Use KL divergence loss like original
                    log_probs = F.log_softmax(outputs_flat, dim=-1)
                    loss_per_position = kl_criterion(log_probs.unsqueeze(0), targets_flat.unsqueeze(0))
                    
                    # Apply masking and weights
                    weighted_loss = loss_per_position * loss_mask_flat
                    
                    if batch_weights.numel() > 0:
                        batch_weights_expanded = batch_weights.unsqueeze(1).expand(-1, temp_inputs.shape[1]).contiguous().view(-1)
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
    
    def compute_log_loss(self, outputs, targets, weights=None):
        """Compute log loss (cross-entropy) for predicted distributions."""
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
    
    def train_on_dataset_basic(self, dataset, epochs=1, batch_size=8, lr=1e-4):
        """Train model on entire dataset using current data state."""
        logger.info(f"Training on full dataset - {len(dataset)} examples")
        
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            batch_count = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                known_questions, inputs, answers, annotators, questions, embeddings = batch
                
                inputs = inputs.to(self.device)
                annotators = annotators.to(self.device)
                questions = questions.to(self.device)
                
                if embeddings is not None:
                    embeddings = embeddings.to(self.device)
                else:
                    seq_len = inputs.shape[1]
                    embeddings = torch.zeros(inputs.shape[0], seq_len, 384).to(self.device)
                
                optimizer.zero_grad()
                outputs = self(inputs, annotators, questions, embeddings)
                
                targets = inputs[:, :, 1:]
                loss = self.compute_log_loss(outputs, targets)
                
                if loss > 0:
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    batch_count += 1
            
            avg_loss = running_loss / max(1, batch_count)
            losses.append(avg_loss)
            self.training_losses.append(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"dataset_loss": avg_loss, "epoch": epoch})
        
        return losses
    
    def save(self, path):
        """Save model state."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'question_num': self.question_num,
                'max_choices': self.max_choices,
                'num_annotator': self.encoder.position_encoder.num_annotator,
                'annotator_embedding_dim': self.encoder.annotator_embedding_dim
            },
            'training_losses': self.training_losses,
            'training_queue_size': len(self.training_queue)
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load model from file."""
        checkpoint = torch.load(path)
        config = checkpoint['config']
        
        model = cls(
            question_num=config['question_num'],
            max_choices=config['max_choices'],
            num_annotator=config['num_annotator'],
            annotator_embedding_dim=config['annotator_embedding_dim']
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.training_losses = checkpoint.get('training_losses', [])
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def get_training_stats(self):
        """Get training statistics for logging/wandb."""
        return {
            'training_queue_size': len(self.training_queue),
            'prediction_history_size': len(self.prediction_history),
            'examples_to_revisit': len(self.examples_to_revisit),
            'total_training_losses': len(self.training_losses),
            'latest_loss': self.training_losses[-1] if self.training_losses else 0.0
        }
    
    def compute_total_loss(self, outputs, labels, inputs, questions, embeddings, full_supervision=False):
        """Compute total loss over all positions based on supervision type."""
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
        
        if position_count > 0:
            loss = loss / position_count
                
        return loss