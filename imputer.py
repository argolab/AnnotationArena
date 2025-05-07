import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
import os
import json

# Base paths (for consistency with utils.py)
BASE_PATH = "/export/fs06/psingh54/ActiveRubric-Internal/outputs"
MODELS_PATH = os.path.join(BASE_PATH, "models")

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

    def forward(self, x, annotators, questions):
        """Create encoded representations combining annotator and question features."""
        batch_size = x.shape[0]
        question_embeds = self.question_embedding[questions]
        annotators = torch.where(annotators < 0, torch.full_like(annotators, self.num_annotator), annotators)
        annotator_embeds = self.annotator_embedding[annotators]
        feature_x = torch.cat((question_embeds + annotator_embeds, x[:,:,1:]), dim=-1)
        param_x = x[:,:,1:].clone()
        return feature_x, param_x


class EncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feed-forward networks."""
    
    def __init__(self, feature_dim, param_dim, attention_heads, dropout=0.3):
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

    def forward(self, feature_x, param_x):
        """Process features through attention and feed-forward layers."""
        batch_size = feature_x.shape[0]
        
        feature_x = self.norm_1(feature_x)
        attention_output = self.multihead_attention(feature_x, batch_size)
        feature_x = feature_x + self.dropout_1(attention_output)
        
        feature_x_ff = self.norm_2(feature_x)
        feature_x = feature_x + self.dropout_2(self.ff(feature_x_ff))
        
        combined = torch.cat([feature_x, param_x], dim=-1)
        param_x = self.param_update(combined)
        
        return feature_x, param_x


class Encoder(nn.Module):
    """Full encoder consisting of multiple encoder layers."""
    
    def __init__(self, question_num, max_choices, encoder_num, attention_heads, 
                 num_annotator, annotator_embedding_dim, dropout=0.1):
        """Initialize encoder with multiple layers."""
        super().__init__()
        self.feature_dim = annotator_embedding_dim + max_choices
        self.param_dim = max_choices
        self.position_encoder = Positional_Encoder(question_num, max_choices, num_annotator, annotator_embedding_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(self.feature_dim, self.param_dim, attention_heads, dropout)
            for _ in range(encoder_num)
        ])
        self.norm = NormLayer(self.feature_dim)
        self.annotator_embedding_dim = annotator_embedding_dim

    def forward(self, x, annotators, questions):
        """Process input through all encoder layers."""
        feature_x, param_x = self.position_encoder(x, annotators, questions)
        for layer in self.layers:
            feature_x, param_x = layer(feature_x, param_x)
        return param_x


class Imputer(nn.Module):
    """
    Imputer model for predicting missing annotations.
    Based on JointModel from activelearner.py with enhanced training capabilities.
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
        
        # Tracking prediction examples
        self.training_examples = []
        # Track which examples need revisiting due to updated supervision
        self.examples_to_revisit = set()
        # Track training losses over time
        self.training_losses = []
    
    def forward(self, x, annotators, questions):
        """Forward pass through the model."""
        param_x = self.encoder(x, annotators, questions)
        return param_x
    
    def predict(self, inputs, annotators, questions, positions=None, train=True, weight=1.0):
        """
        Predict distributions for specified positions.
        
        Args:
            inputs: Input tensor [batch_size, sequence_length, input_dim]
            annotators: Annotator indices [batch_size, sequence_length]
            questions: Question indices [batch_size, sequence_length]
            positions: Positions to predict (default: all)
            train: Whether to track this prediction for training
            weight: Weight of this example for training
            
        Returns:
            Predicted distributions for specified positions
        """
        self.eval()  # Set to eval mode for prediction
        
        with torch.no_grad():
            outputs = self(inputs, annotators, questions)
            
            # Extract predictions for specific positions if provided
            if positions is not None:
                if isinstance(positions, list):
                    predictions = outputs[:, positions, :]
                else:  # Single position
                    predictions = outputs[:, positions:positions+1, :]
            else:
                predictions = outputs
        
        # Track this prediction for training if required
        if train:
            batch_size = inputs.shape[0]
            
            for i in range(batch_size):
                example = {
                    'inputs': inputs[i].detach().cpu().clone(),
                    'annotators': annotators[i].detach().cpu().clone(),
                    'questions': questions[i].detach().cpu().clone(),
                    'positions': positions if positions is not None else list(range(inputs.shape[1])),
                    'weight': weight,
                    'timestamp': len(self.training_examples),
                    'loss': None,  # Will be filled during training
                    'needs_revisit': False  # Will be set to True when new observations arrive
                }
                self.training_examples.append(example)
        
        return predictions
    
    def update_training_supervision(self, observed_values, positions, example_indices=None):
        """
        Update training examples with observed values.
        
        Args:
            observed_values: Observed values to update with
            positions: Positions of the observed values
            example_indices: Indices of examples to update (default: all)
            
        Returns:
            Number of examples updated
        """
        if example_indices is None:
            example_indices = list(range(len(self.training_examples)))
        
        count = 0
        for idx in example_indices:
            example = self.training_examples[idx]
            
            # Check if any positions in this example match the observed positions
            for i, pos in enumerate(positions):
                if pos in example['positions']:
                    # Update the example's inputs to reflect the observed value
                    pos_idx = example['positions'].index(pos)
                    example['inputs'][pos][0] = 0  # Mark as observed
                    example['inputs'][pos][1:] = observed_values[i]
                    count += 1
                    
                    # Mark this example for revisiting during next training
                    example['needs_revisit'] = True
                    self.examples_to_revisit.add(idx)
        
        return count
    
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
    
    def train_on_examples(self, examples_indices=None, epochs=1, batch_size=8, lr=1e-4):
        """
        Train the model on stored examples with prioritized revisiting.
        
        Args:
            examples_indices: Indices of examples to train on (default: all)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            List of training losses
        """
        self.train()  # Set to train mode
        
        # If no specific examples are provided, prioritize those that need revisiting
        if examples_indices is None:
            # First include examples that need revisiting
            examples_to_revisit = list(self.examples_to_revisit)
            
            # Then add other examples up to a reasonable batch count
            other_examples = [i for i in range(len(self.training_examples)) 
                             if i not in self.examples_to_revisit]
            
            # Determine how many additional examples to include
            target_examples = batch_size * 10  # Train on approximately 10 batches
            num_additional = max(0, target_examples - len(examples_to_revisit))
            
            if num_additional > 0 and other_examples:
                additional_examples = np.random.choice(
                    other_examples, 
                    min(num_additional, len(other_examples)), 
                    replace=False
                ).tolist()
                examples_indices = examples_to_revisit + additional_examples
            else:
                examples_indices = examples_to_revisit
                
            if not examples_indices:
                # If still no examples, just use all available
                examples_indices = list(range(len(self.training_examples)))
        
        if len(examples_indices) == 0:
            return []
        
        # Create optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # Shuffle examples
            np.random.shuffle(examples_indices)
            
            # Process in batches
            for batch_start in range(0, len(examples_indices), batch_size):
                batch_indices = examples_indices[batch_start:batch_start + batch_size]
                batch_examples = [self.training_examples[i] for i in batch_indices]
                
                # Extract batch data
                batch_inputs = torch.stack([e['inputs'] for e in batch_examples]).to(self.device)
                batch_annotators = torch.stack([e['annotators'] for e in batch_examples]).to(self.device)
                batch_questions = torch.stack([e['questions'] for e in batch_examples]).to(self.device)
                batch_weights = torch.tensor([e['weight'] for e in batch_examples]).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self(batch_inputs, batch_annotators, batch_questions)
                
                # Compute loss on observed positions (where mask bit is 0)
                batch_targets = batch_inputs[:, :, 1:].clone()  # Exclude mask bit
                observed_mask = (batch_inputs[:, :, 0] == 0).float().unsqueeze(-1).expand_as(outputs)
                masked_outputs = outputs * observed_mask
                masked_targets = batch_targets * observed_mask
                
                if observed_mask.sum() > 0:
                    loss = self.compute_log_loss(masked_outputs, masked_targets, batch_weights)
                    loss.backward()
                    optimizer.step()
                    
                    # Record example losses
                    for i, idx in enumerate(batch_indices):
                        example_outputs = outputs[i:i+1]
                        example_targets = batch_targets[i:i+1]
                        example_mask = observed_mask[i:i+1]
                        if example_mask.sum() > 0:
                            example_loss = self.compute_log_loss(
                                example_outputs * example_mask, 
                                example_targets * example_mask
                            ).item()
                            self.training_examples[idx]['loss'] = example_loss
                    
                    epoch_loss += loss.item()
                    batch_count += 1
            
            # Calculate average loss for this epoch
            avg_epoch_loss = epoch_loss / max(1, batch_count)
            losses.append(avg_epoch_loss)
            
            # Record for history
            self.training_losses.append({
                'epoch': len(self.training_losses),
                'loss': avg_epoch_loss,
                'examples_trained': len(examples_indices),
                'examples_revisited': len(self.examples_to_revisit.intersection(examples_indices))
            })
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Clear the revisit flags for trained examples
        for idx in examples_indices:
            if idx in self.examples_to_revisit:
                self.training_examples[idx]['needs_revisit'] = False
                self.examples_to_revisit.remove(idx)
        
        return losses
    
    def train_with_revisiting(self, dataset, epochs=1, batch_size=8, lr=1e-4):
        """
        Train the model with revisiting examples.
        
        Args:
            dataset: The dataset to train on
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            List of training losses
        """
        self.train()  # Set to train mode
        
        # Create optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Masking patterns for random masking during training
        masking_patterns = [
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0],
        ]
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            batch_count = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                known_questions, inputs, answers, annotators, questions = batch
                inputs, answers, annotators, questions = (
                    inputs.to(self.device), answers.to(self.device), 
                    annotators.to(self.device), questions.to(self.device)
                )
                
                temp_inputs = inputs.clone()
                
                # Apply masking pattern for artificial masking during training
                pattern_idx = np.random.randint(0, len(masking_patterns))
                pattern = masking_patterns[pattern_idx]
                
                for b in range(inputs.shape[0]):
                    for i in range(annotators.shape[1]):
                        q_idx = questions[b, i].item()
                        is_llm = (annotators[b, i].item() == -1)
                        
                        # Only mask positions that are currently observed
                        if temp_inputs[b, i, 0] == 0:  # Currently observed
                            pattern_pos = 2 * q_idx + (0 if is_llm else 1)
                            if pattern_pos < len(pattern) and pattern[pattern_pos] == 1:
                                temp_inputs[b, i, 0] = 1
                                temp_inputs[b, i, 1:] = torch.zeros_like(temp_inputs[b, i, 1:])
                
                optimizer.zero_grad()
                outputs = self(temp_inputs, annotators, questions)
                
                # Compute loss on all positions
                loss = 0.0
                position_count = 0
                
                for i in range(batch_size):
                    if i >= outputs.shape[0]:
                        continue
                        
                    for j in range(outputs.shape[1]):
                        # Only compute loss for observed positions (in the original inputs)
                        if inputs[i, j, 0] == 0:
                            target_idx = torch.argmax(answers[i, j]).item()
                            target = torch.tensor([target_idx], device=self.device)
                            position_loss = F.cross_entropy(outputs[i:i+1, j], target)
                            
                            loss += position_loss
                            position_count += 1
                
                # Average loss
                if position_count > 0:
                    loss = loss / position_count
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    batch_count += 1
            
            # Calculate average loss for this epoch
            avg_epoch_loss = running_loss / max(1, batch_count)
            losses.append(avg_epoch_loss)
            
            # Record for history
            self.training_losses.append({
                'epoch': len(self.training_losses),
                'loss': avg_epoch_loss,
                'examples_trained': len(dataset) * batch_count // batch_size
            })
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        return losses
    
    def clear_training_examples(self):
        """Clear all stored training examples."""
        self.training_examples = []
        self.examples_to_revisit = set()
    
    def get_loss_history(self):
        """Get the history of training losses."""
        return self.training_losses
    
    def save(self, path):
        """Save model to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'question_num': self.question_num,
                'max_choices': self.max_choices,
                'num_annotator': self.encoder.position_encoder.num_annotator,
                'annotator_embedding_dim': self.encoder.annotator_embedding_dim
            },
            'training_losses': self.training_losses
        }, path)
    
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
        
        return model
    
    def compute_total_loss(self, outputs, labels, inputs, questions, full_supervision=False):
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
                if inputs[i, j, 0] == 0 or full_supervision:
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