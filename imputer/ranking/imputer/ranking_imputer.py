import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from itertools import combinations
from collections import defaultdict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class NormLayer(nn.Module):
    """Layer normalization with learnable parameters."""
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    """Simple feed-forward network with ReLU activation."""
    
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class MultiVariableEncoder(nn.Module):
    """Encodes both attribute and ranking variables with appropriate embeddings."""
    
    def __init__(self, num_attributes, num_annotators, num_items, 
                 embedding_dim, num_likert_classes, max_rank_size):
        super().__init__()
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_likert_classes = num_likert_classes
        self.max_rank_size = max_rank_size
        
        # Embeddings for each component
        self.attribute_embedding = nn.Parameter(torch.randn(num_attributes, embedding_dim))
        self.annotator_embedding = nn.Parameter(torch.randn(num_annotators, embedding_dim))
        self.item_embedding = nn.Parameter(torch.randn(num_items, embedding_dim))
        
        # Initialize embeddings
        torch.nn.init.kaiming_normal_(self.attribute_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.annotator_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.item_embedding, mode='fan_out', nonlinearity='relu')
        
        # Projection layer for ranking variables (outer product results)
        self.ranking_projection = nn.Sequential(
            nn.Linear(embedding_dim * embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, variable_data, variable_types, attribute_ids, annotator_ids, item_ids):
        batch_size, num_variables = variable_types.shape
        
        # Get base embeddings
        attr_embeds = self.attribute_embedding[attribute_ids]  # [batch, vars, emb_dim]
        annot_embeds = self.annotator_embedding[annotator_ids]  # [batch, vars, emb_dim]
        
        # Initialize feature embeddings
        feature_embeddings = torch.zeros(batch_size, num_variables, self.embedding_dim, 
                                       device=variable_data.device)
        
        # Initialize parameter data
        max_param_dim = max(self.num_likert_classes, self.max_rank_size)
        param_data = torch.zeros(batch_size, num_variables, max_param_dim, 
                               device=variable_data.device)
        
        for b in range(batch_size):
            for v in range(num_variables):
                var_type = variable_types[b, v].item()
                
                if var_type == 0:  # Rating variable
                    # Simple sum: attribute + annotator + item embeddings
                    primary_item_id = item_ids[b, v, 0].item()
                    if primary_item_id >= 0 and primary_item_id < self.num_items:
                        item_embed = self.item_embedding[primary_item_id]
                        feature_embeddings[b, v] = (attr_embeds[b, v] + 
                                                   annot_embeds[b, v] + 
                                                   item_embed)
                    else:
                        feature_embeddings[b, v] = attr_embeds[b, v] + annot_embeds[b, v]
                    
                    # Copy rating data
                    param_data[b, v, :self.num_likert_classes] = variable_data[b, v, :self.num_likert_classes]
                
                elif var_type == 1:  # Ranking variable
                    # Get embeddings for all items being ranked
                    ranked_items = item_ids[b, v]  # [max_items_per_var]
                    valid_items = ranked_items[ranked_items >= 0]
                    
                    if len(valid_items) > 1:
                        # Get attribute embeddings for each item
                        item_attr_embeddings = []
                        for item_id in valid_items:
                            if item_id < self.num_items:
                                item_embed = self.item_embedding[item_id]
                                combined_embed = (attr_embeds[b, v] + 
                                               annot_embeds[b, v] + 
                                               item_embed)
                                item_attr_embeddings.append(combined_embed)
                        
                        if len(item_attr_embeddings) > 1:
                            # Compute sum of pairwise outer products
                            total_outer_product = torch.zeros(self.embedding_dim, self.embedding_dim, device=item_attr_embeddings[0].device)
                            for i in range(len(item_attr_embeddings)):
                                for j in range(i+1, len(item_attr_embeddings)):
                                    total_outer_product += torch.outer(item_attr_embeddings[i], item_attr_embeddings[j])
                            feature_embeddings[b, v] = self.ranking_projection(total_outer_product.flatten())
                        else:
                            feature_embeddings[b, v] = attr_embeds[b, v] + annot_embeds[b, v]
                    else:
                        feature_embeddings[b, v] = attr_embeds[b, v] + annot_embeds[b, v]
                    
                    # Copy ranking data
                    rank_data = variable_data[b, v, :self.max_rank_size]
                    param_data[b, v, :self.max_rank_size] = rank_data
        
        return feature_embeddings, param_data


class EncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feed-forward networks."""

    def __init__(self, feature_dim, param_dim, attention_heads, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim 
        self.param_dim = param_dim  
        self.attention_heads = attention_heads
        
        # Feature stream attention
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
        Q = self.Q(feature_x).view(batch_size, -1, self.attention_heads, 
                                  self.feature_dim // self.attention_heads).transpose(1, 2)
        K = self.K(feature_x).view(batch_size, -1, self.attention_heads, 
                                  self.feature_dim // self.attention_heads).transpose(1, 2)
        V = self.V(feature_x).view(batch_size, -1, self.attention_heads, 
                                  self.feature_dim // self.attention_heads).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.feature_dim // self.attention_heads)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_1(scores)
        scores = torch.matmul(scores, V)
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)
        output = self.out(scores)
        return output

    def forward(self, feature_x, param_x):
        """Process features through attention, feed-forward."""
        batch_size = feature_x.shape[0]
        
        # Feature stream processing
        feature_x_norm = self.norm_1(feature_x)
        attention_output = self.multihead_attention(feature_x_norm, batch_size)
        feature_x = feature_x + self.dropout_1(attention_output)
        
        feature_x_ff = self.norm_2(feature_x)
        feature_x = feature_x + self.dropout_2(self.ff(feature_x_ff))

        # Param update
        combined = torch.cat([feature_x, param_x], dim=-1)
        param_x = self.param_update(combined)
        
        return feature_x, param_x


class Encoder(nn.Module):
    """Full encoder consisting of multiple encoder layers."""

    def __init__(self, num_attributes, num_annotators, num_items, encoder_num, attention_heads, 
                 embedding_dim, num_likert_classes, max_rank_size, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_param_dim = max(num_likert_classes, max_rank_size)
        
        self.variable_encoder = MultiVariableEncoder(
            num_attributes, num_annotators, num_items, 
            embedding_dim, num_likert_classes, max_rank_size
        )

        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, self.max_param_dim, attention_heads, dropout)
            for _ in range(encoder_num)
        ])

        self.norm = NormLayer(embedding_dim)

    def forward(self, variable_data, variable_types, attribute_ids, annotator_ids, item_ids):
        """Process input through all encoder layers."""
        feature_x, param_x = self.variable_encoder(
            variable_data, variable_types, attribute_ids, annotator_ids, item_ids
        )
        
        for layer in self.layers:
            feature_x, param_x = layer(feature_x, param_x)
        
        return feature_x, param_x


class MultiVariableImputer(nn.Module):
    """Imputer model for predicting missing annotations."""
    
    def __init__(self, 
                 num_attributes=8, 
                 num_annotators=8, 
                 num_items=8,
                 num_likert_classes=5,
                 max_rank_size=3,
                 encoder_layers_num=2, 
                 attention_heads=4, 
                 embedding_dim=64,
                 dropout=0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.num_likert_classes = num_likert_classes
        self.max_rank_size = max_rank_size
        
        self.encoder = Encoder(
            num_attributes, num_annotators, num_items, encoder_layers_num, attention_heads, 
            embedding_dim, num_likert_classes, max_rank_size, dropout
        )

        # Two output heads
        self.attribute_head = nn.Linear(embedding_dim, num_likert_classes)  # For Likert scale
        self.ranking_head = nn.Linear(embedding_dim, max_rank_size)  # For rankings
        
    def forward(self, variable_data, variable_types, attribute_ids, annotator_ids, item_ids):
        # Encode through transformer
        feature_x, param_x = self.encoder(
            variable_data, variable_types, attribute_ids, annotator_ids, item_ids
        )
        
        # Apply output heads
        attribute_logits = self.attribute_head(feature_x)  # [batch, vars, C]
        ranking_logits = self.ranking_head(feature_x)      # [batch, vars, r]
        
        return attribute_logits, ranking_logits




# Import your model (assuming it's in a separate file)
# from your_model_file import MultiVariableImputer

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from itertools import combinations
from collections import defaultdict
import logging
from tqdm import tqdm

# Import your model (assuming it's in a separate file)
# from your_model_file import MultiVariableImputer

class DataConverter:
    def __init__(self, num_attributes=10, num_annotators=5, num_items=10, 
                 num_likert_classes=5, max_rank_size=3):
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators  
        self.num_items = num_items
        self.num_likert_classes = num_likert_classes
        self.max_rank_size = max_rank_size
        
    def load_training_data(self, json_file):
        """Load training data from JSON file and filter items <= num_items"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Filter ratings and rankings to only include items <= num_items
        filtered_ratings = []
        for rating in data['ratings']:
            if rating['item'] <= self.num_items:
                filtered_ratings.append(rating)
        
        filtered_rankings = []
        for ranking in data['rankings']:
            # Check if all items in the ranking are <= num_items
            items_to_check = ranking['items'][:self.max_rank_size]
            if all(item <= self.num_items for item in items_to_check):
                filtered_rankings.append(ranking)
        
        return {
            'ratings': filtered_ratings,
            'rankings': filtered_rankings
        }
    
    def create_variables_from_actual_data(self, train_data, test_data):
        """Create variables based on actual data that exists in train+test files"""
        rating_variables = []
        ranking_variables = []
        
        # Collect all unique rating variables from both train and test data
        all_ratings = train_data['ratings'] + test_data['ratings']
        rating_keys = set()
        
        for rating in all_ratings:
            key = (rating['attribute'], rating['annotator'], rating['item'])
            rating_keys.add(key)
        
        # Create rating variables from actual data
        for attr, annot, item in rating_keys:
            rating_variables.append({
                'type': 'rating',
                'attribute': attr,
                'annotator': annot,
                'item': item
            })
        
        # Collect all unique ranking variables from both train and test data
        all_rankings = train_data['rankings'] + test_data['rankings']
        ranking_keys = set()
        
        for ranking in all_rankings:
            # Use the exact items from the data (don't sort for key)
            items = ranking['items'][:self.max_rank_size]  # Truncate if needed
            key = (ranking['attribute'], ranking['annotator'], tuple(items))
            ranking_keys.add(key)
        
        # Create ranking variables from actual data
        for attr, annot, items_tuple in ranking_keys:
            ranking_variables.append({
                'type': 'ranking',
                'attribute': attr,
                'annotator': annot,
                'items': list(items_tuple)
            })
        
        return rating_variables, ranking_variables
    
    def process_training_data(self, data):
        """Process training data into lookup dictionaries"""
        rating_data = {}
        ranking_data = {}
        
        # Process ratings
        for rating in data['ratings']:
            key = (rating['attribute'], rating['annotator'], rating['item'])
            rating_data[key] = rating['value']
        
        # Process rankings (truncate to max_rank_size)
        for ranking in data['rankings']:
            # Truncate items and order to max_rank_size
            items = ranking['items'][:self.max_rank_size]
            order = ranking['order'][:self.max_rank_size]
            
            key = (ranking['attribute'], ranking['annotator'], tuple(items))
            ranking_data[key] = {
                'items': items,
                'order': order
            }
        
        return rating_data, ranking_data
    
    def create_training_batch(self, rating_variables, ranking_variables, 
                            rating_data, ranking_data, test_data=None, mask_rate=0.5):
        """Create a single training batch with all variables"""
        all_variables = rating_variables + ranking_variables
        num_variables = len(all_variables)
        
        # Initialize tensors
        variable_data = torch.zeros(1, num_variables, 
                                  max(self.num_likert_classes, self.max_rank_size))
        variable_types = torch.zeros(1, num_variables, dtype=torch.long)
        attribute_ids = torch.zeros(1, num_variables, dtype=torch.long)
        annotator_ids = torch.zeros(1, num_variables, dtype=torch.long)
        
        # For items, we need to handle both single items and multiple items
        max_items_per_var = self.max_rank_size
        item_ids = torch.full((1, num_variables, max_items_per_var), -1, dtype=torch.long)
        
        # Targets for training
        rating_targets = torch.zeros(1, num_variables, self.num_likert_classes)
        ranking_targets = torch.zeros(1, num_variables, self.max_rank_size)
        
        # Masks to indicate which variables have data
        rating_mask = torch.zeros(1, num_variables, dtype=torch.bool)
        ranking_mask = torch.zeros(1, num_variables, dtype=torch.bool)
        
        # Masks to indicate which variables are masked for imputation
        rating_masked = torch.zeros(1, num_variables, dtype=torch.bool)
        ranking_masked = torch.zeros(1, num_variables, dtype=torch.bool)
        
        # Process test data if provided to create exclusion set
        test_exclusions = set()
        if test_data is not None:
            test_rating_data, test_ranking_data = self.process_training_data(test_data)
            test_exclusions.update(test_rating_data.keys())
            test_exclusions.update(test_ranking_data.keys())
        
        # Collect available training variables for balanced masking
        available_rating_vars = []
        available_ranking_vars = []
        
        for i, var in enumerate(all_variables):
            if var['type'] == 'rating':
                key = (var['attribute'], var['annotator'], var['item'])
                if key in rating_data and key not in test_exclusions:
                    available_rating_vars.append(i)
            elif var['type'] == 'ranking':
                items = var['items']
                key = (var['attribute'], var['annotator'], tuple(items))
                if key in ranking_data and key not in test_exclusions:
                    available_ranking_vars.append(i)
        
        # Balanced random masking
        import random
        num_rating_masked = int(len(available_rating_vars) * mask_rate)
        num_ranking_masked = int(len(available_ranking_vars) * mask_rate)
        
        masked_rating_indices = set(random.sample(available_rating_vars, num_rating_masked)) if available_rating_vars else set()
        masked_ranking_indices = set(random.sample(available_ranking_vars, num_ranking_masked)) if available_ranking_vars else set()
        
        for i, var in enumerate(all_variables):
            attribute_ids[0, i] = var['attribute'] - 1  # Convert to 0-indexed for model
            annotator_ids[0, i] = var['annotator'] - 1  # Convert to 0-indexed for model
            
            if var['type'] == 'rating':
                variable_types[0, i] = 0  # Rating type
                item_ids[0, i, 0] = var['item'] - 1  # Convert to 0-indexed for model
                
                # Check if this rating exists in training data and not in test data
                key = (var['attribute'], var['annotator'], var['item'])
                if key in rating_data and key not in test_exclusions:
                    rating_value = rating_data[key] - 1  # Convert to 0-indexed
                    rating_targets[0, i, rating_value] = 1.0  # One-hot encoding
                    rating_mask[0, i] = True
                    
                    # Check if this variable is masked for imputation
                    if i in masked_rating_indices:
                        rating_masked[0, i] = True
                    else:
                        # Only set input data for observed (non-masked) variables
                        variable_data[0, i, rating_value] = 1.0
                    
            elif var['type'] == 'ranking':
                variable_types[0, i] = 1  # Ranking type
                items = var['items']
                for j, item in enumerate(items):
                    item_ids[0, i, j] = item - 1  # Convert to 0-indexed for model
                
                # Check if this ranking exists in training data and not in test data
                key = (var['attribute'], var['annotator'], tuple(items))
                if key in ranking_data and key not in test_exclusions:
                    ranking_info = ranking_data[key]
                    order = ranking_info['order']
                    
                    # Convert order to Plackett-Luce format (position-based scores)
                    # Higher score = better rank (1st place gets highest score)
                    for j, pos in enumerate(order):
                        if j < self.max_rank_size:
                            ranking_targets[0, i, j] = self.max_rank_size - pos + 1
                    
                    ranking_mask[0, i] = True
                    
                    # Check if this variable is masked for imputation
                    if i in masked_ranking_indices:
                        ranking_masked[0, i] = True
                    else:
                        # Only set input data for observed (non-masked) variables
                        for j, pos in enumerate(order):
                            if j < self.max_rank_size:
                                variable_data[0, i, j] = self.max_rank_size - pos + 1
        
        return {
            'variable_data': variable_data,
            'variable_types': variable_types,
            'attribute_ids': attribute_ids,
            'annotator_ids': annotator_ids,
            'item_ids': item_ids,
            'rating_targets': rating_targets,
            'ranking_targets': ranking_targets,
            'rating_mask': rating_mask,
            'ranking_mask': ranking_mask,
            'rating_masked': rating_masked,
            'ranking_masked': ranking_masked,
            'all_variables': all_variables
        }

class PlackettLuceLoss(nn.Module):
    """Plackett-Luce loss for ranking data"""
    def __init__(self):
        super().__init__()
        
    def forward(self, logits, targets, mask):
        """
        Args:
            logits: [batch, variables, max_rank_size] - raw scores
            targets: [batch, variables, max_rank_size] - position scores (higher = better rank)
            mask: [batch, variables] - which variables have ranking data
        """
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)
            
        loss = 0.0
        count = 0
        
        for b in range(logits.size(0)):
            for v in range(logits.size(1)):
                if mask[b, v]:
                    # Get the scores for this ranking
                    scores = logits[b, v]  # [max_rank_size]
                    target_ranks = targets[b, v]  # [max_rank_size]
                    
                    # Find valid positions (non-zero targets)
                    valid_positions = target_ranks > 0
                    if not valid_positions.any():
                        continue
                        
                    valid_scores = scores[valid_positions]
                    valid_targets = target_ranks[valid_positions]
                    
                    # Sort by target rank (descending order of preference)
                    _, sort_indices = torch.sort(valid_targets, descending=True)
                    sorted_scores = valid_scores[sort_indices]
                    
                    # Compute Plackett-Luce loss
                    pl_loss = 0.0
                    for i in range(len(sorted_scores)):
                        # Log probability of choosing item i from remaining items
                        remaining_scores = sorted_scores[i:]
                        log_prob = sorted_scores[i] - torch.logsumexp(remaining_scores, dim=0)
                        pl_loss -= log_prob
                    
                    loss += pl_loss
                    count += 1
        
        return loss / max(count, 1)

class ImputerTrainer:
    def __init__(self, model, learning_rate=1e-3, alpha=1.0, beta=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.rating_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.ranking_loss_fn = PlackettLuceLoss()
        self.alpha = alpha  # Weight for observed variables
        self.beta = beta    # Weight for masked variables
        
    def train_step(self, batch):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Move batch to device
        variable_data = batch['variable_data'].to(self.device)
        variable_types = batch['variable_types'].to(self.device)
        attribute_ids = batch['attribute_ids'].to(self.device)
        annotator_ids = batch['annotator_ids'].to(self.device)
        item_ids = batch['item_ids'].to(self.device)
        rating_targets = batch['rating_targets'].to(self.device)
        ranking_targets = batch['ranking_targets'].to(self.device)
        rating_mask = batch['rating_mask'].to(self.device)
        ranking_mask = batch['ranking_mask'].to(self.device)
        rating_masked = batch['rating_masked'].to(self.device)
        ranking_masked = batch['ranking_masked'].to(self.device)
        
        # Forward pass
        rating_logits, ranking_logits = self.model(
            variable_data, variable_types, attribute_ids, annotator_ids, item_ids
        )
        
        # Compute losses with weighted approach (Option C)
        rating_loss_observed = 0.0
        rating_loss_masked = 0.0
        ranking_loss_observed = 0.0
        ranking_loss_masked = 0.0
        
        # Rating loss (cross-entropy)
        if rating_mask.any():
            rating_indices = rating_mask.nonzero(as_tuple=False)
            if len(rating_indices) > 0:
                valid_rating_logits = rating_logits[rating_indices[:, 0], rating_indices[:, 1]]
                valid_rating_targets = rating_targets[rating_indices[:, 0], rating_indices[:, 1]]
                valid_rating_masked = rating_masked[rating_indices[:, 0], rating_indices[:, 1]]
                target_classes = torch.argmax(valid_rating_targets, dim=1)
                
                losses = self.rating_loss_fn(valid_rating_logits, target_classes)
                
                # Separate observed and masked losses
                masked_indices = valid_rating_masked.nonzero(as_tuple=False).squeeze(-1)
                observed_indices = (~valid_rating_masked).nonzero(as_tuple=False).squeeze(-1)
                
                if len(masked_indices) > 0:
                    rating_loss_masked = losses[masked_indices].mean()
                if len(observed_indices) > 0:
                    rating_loss_observed = losses[observed_indices].mean()
        
        # Ranking loss (Plackett-Luce) - split observed and masked
        if ranking_mask.any():
            # Separate observed and masked ranking losses
            observed_mask = ranking_mask & (~ranking_masked)
            masked_mask = ranking_mask & ranking_masked
            
            if observed_mask.any():
                ranking_loss_observed = self.ranking_loss_fn(ranking_logits, ranking_targets, observed_mask)
            if masked_mask.any():
                ranking_loss_masked = self.ranking_loss_fn(ranking_logits, ranking_targets, masked_mask)
        
        # Weighted total loss (Option C)
        observed_loss = self.alpha * (rating_loss_observed + ranking_loss_observed)
        masked_loss = self.beta * (rating_loss_masked + ranking_loss_masked)
        total_loss = observed_loss + masked_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'observed_loss': observed_loss.item() if isinstance(observed_loss, torch.Tensor) else observed_loss,
            'masked_loss': masked_loss.item() if isinstance(masked_loss, torch.Tensor) else masked_loss,
            'rating_loss_observed': rating_loss_observed.item() if isinstance(rating_loss_observed, torch.Tensor) else rating_loss_observed,
            'rating_loss_masked': rating_loss_masked.item() if isinstance(rating_loss_masked, torch.Tensor) else rating_loss_masked,
            'ranking_loss_observed': ranking_loss_observed.item() if isinstance(ranking_loss_observed, torch.Tensor) else ranking_loss_observed,
            'ranking_loss_masked': ranking_loss_masked.item() if isinstance(ranking_loss_masked, torch.Tensor) else ranking_loss_masked
        }
    
    def evaluate_with_test_data(self, batch, test_data, converter, mask_rate=0.5, verbose=True):
        """Evaluate model on test data with proper imputation masking"""
        self.model.eval()
        
        with torch.no_grad():
            # Process test data
            test_rating_data, test_ranking_data = converter.process_training_data(test_data)
            
            # Create test variables and apply masking for imputation
            all_variables = batch['all_variables']
            
            # Collect test variables that have data
            test_rating_vars = []
            test_ranking_vars = []
            
            for i, var in enumerate(all_variables):
                if var['type'] == 'rating':
                    key = (var['attribute'], var['annotator'], var['item'])
                    if key in test_rating_data:
                        test_rating_vars.append(i)
                elif var['type'] == 'ranking':
                    items = var['items']
                    key = (var['attribute'], var['annotator'], tuple(items))
                    if key in test_ranking_data:
                        test_ranking_vars.append(i)
            
            # Randomly mask 50% of test data for imputation evaluation
            import random
            random.seed(42)  # Reproducible masking
            
            num_rating_masked = int(len(test_rating_vars) * mask_rate)
            num_ranking_masked = int(len(test_ranking_vars) * mask_rate)
            
            masked_test_rating_vars = set(random.sample(test_rating_vars, num_rating_masked)) if test_rating_vars else set()
            masked_test_ranking_vars = set(random.sample(test_ranking_vars, num_ranking_masked)) if test_ranking_vars else set()
            
            # Create input data for imputer (with masked positions set to zero)
            test_variable_data = batch['variable_data'].clone()
            
            # Mask the selected test variables (set their data to zero)
            for i in masked_test_rating_vars:
                test_variable_data[0, i, :] = 0.0
            for i in masked_test_ranking_vars:
                test_variable_data[0, i, :] = 0.0
            
            # Move batch to device
            test_variable_data = test_variable_data.to(self.device)
            variable_types = batch['variable_types'].to(self.device)
            attribute_ids = batch['attribute_ids'].to(self.device)
            annotator_ids = batch['annotator_ids'].to(self.device)
            item_ids = batch['item_ids'].to(self.device)
            
            # Forward pass with masked input
            rating_logits, ranking_logits = self.model(
                test_variable_data, variable_types, attribute_ids, annotator_ids, item_ids
            )
            
            # Create targets for ONLY the masked test variables
            test_rating_mask = torch.zeros(1, len(all_variables), dtype=torch.bool)
            test_ranking_mask = torch.zeros(1, len(all_variables), dtype=torch.bool)
            test_rating_targets = torch.zeros(1, len(all_variables), converter.num_likert_classes)
            test_ranking_targets = torch.zeros(1, len(all_variables), converter.max_rank_size)
            
            # Fill targets ONLY for masked variables (this is what we're evaluating imputation on)
            for i in masked_test_rating_vars:
                var = all_variables[i]
                key = (var['attribute'], var['annotator'], var['item'])
                if key in test_rating_data:
                    test_rating_mask[0, i] = True
                    rating_value = test_rating_data[key] - 1  # Convert to 0-indexed
                    test_rating_targets[0, i, rating_value] = 1.0
                        
            for i in masked_test_ranking_vars:
                var = all_variables[i]
                items = var['items']
                key = (var['attribute'], var['annotator'], tuple(items))
                if key in test_ranking_data:
                    test_ranking_mask[0, i] = True
                    ranking_info = test_ranking_data[key]
                    order = ranking_info['order']
                    
                    for j, pos in enumerate(order):
                        if j < converter.max_rank_size:
                            test_ranking_targets[0, i, j] = converter.max_rank_size - pos + 1
            
            # Move test data to device
            test_rating_targets = test_rating_targets.to(self.device)
            test_ranking_targets = test_ranking_targets.to(self.device)
            test_rating_mask = test_rating_mask.to(self.device)
            test_ranking_mask = test_ranking_mask.to(self.device)
            
            # Compute test losses
            test_rating_loss = 0.0
            test_ranking_loss = 0.0
            
            if test_rating_mask.any():
                rating_indices = test_rating_mask.nonzero(as_tuple=False)
                if len(rating_indices) > 0:
                    valid_rating_logits = rating_logits[rating_indices[:, 0], rating_indices[:, 1]]
                    valid_rating_targets = test_rating_targets[rating_indices[:, 0], rating_indices[:, 1]]
                    target_classes = torch.argmax(valid_rating_targets, dim=1)
                    test_rating_loss = self.rating_loss_fn(valid_rating_logits, target_classes).mean()
            
            if test_ranking_mask.any():
                test_ranking_loss = self.ranking_loss_fn(ranking_logits, test_ranking_targets, test_ranking_mask)
            
            # Print predictions vs ground truth for each attribute
            if verbose:
                self.print_predictions_by_attribute(
                    rating_logits, ranking_logits, test_rating_targets, test_ranking_targets,
                    test_rating_mask, test_ranking_mask, all_variables, converter
                )
            
            return {
                'test_rating_loss': test_rating_loss.item() if isinstance(test_rating_loss, torch.Tensor) else test_rating_loss,
                'test_ranking_loss': test_ranking_loss.item() if isinstance(test_ranking_loss, torch.Tensor) else test_ranking_loss,
                'total_test_loss': (test_rating_loss.item() if isinstance(test_rating_loss, torch.Tensor) else test_rating_loss) + 
                                 (test_ranking_loss.item() if isinstance(test_ranking_loss, torch.Tensor) else test_ranking_loss)
            }
    
    def print_predictions_by_attribute(self, rating_logits, ranking_logits, test_rating_targets, 
                                     test_ranking_targets, test_rating_mask, test_ranking_mask, 
                                     all_variables, converter):
        """Print predictions and ground truth organized by attribute"""
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS BY ATTRIBUTE")
        print("="*80)
        
        for attr in range(converter.num_attributes):
            print(f"\n--- ATTRIBUTE {attr} ---")
            
            # Rating predictions for this attribute
            rating_found = False
            for i, var in enumerate(all_variables):
                if (var['type'] == 'rating' and var['attribute'] == attr and 
                    test_rating_mask[0, i]):
                    if not rating_found:
                        print(f"Ratings:")
                        rating_found = True
                    
                    # Get prediction
                    pred_probs = torch.softmax(rating_logits[0, i], dim=0)
                    pred_class = torch.argmax(pred_probs).item() + 1  # Convert back to 1-indexed
                    
                    # Get true class
                    true_class = torch.argmax(test_rating_targets[0, i]).item() + 1  # Convert back to 1-indexed
                    
                    print(f"  Annotator {var['annotator']}, Item {var['item']}: "
                          f"Pred={pred_class}, True={true_class}, "
                          f"Confidence={pred_probs[pred_class-1]:.3f}")
            
            # Ranking predictions for this attribute
            ranking_found = False
            for i, var in enumerate(all_variables):
                if (var['type'] == 'ranking' and var['attribute'] == attr and 
                    test_ranking_mask[0, i]):
                    if not ranking_found:
                        print(f"Rankings:")
                        ranking_found = True
                    
                    # Get prediction - convert logits to ranking
                    pred_scores = ranking_logits[0, i, :converter.max_rank_size]
                    pred_ranking_indices = torch.argsort(pred_scores, descending=True)
                    pred_items = [var['items'][idx] for idx in pred_ranking_indices]
                    
                    # Get true ranking
                    true_scores = test_ranking_targets[0, i, :converter.max_rank_size]
                    true_ranking_indices = torch.argsort(true_scores, descending=True)
                    # Only consider non-zero positions
                    valid_positions = true_scores > 0
                    if valid_positions.any():
                        true_items = [var['items'][idx] for idx in true_ranking_indices]
                    else:
                        true_items = var['items']
                    
                    print(f"  Annotator {var['annotator']}, Items {var['items']}: "
                          f"Pred={pred_items}, True={true_items}")
            
            if not rating_found and not ranking_found:
                print(f"  No test data for this attribute")

def main():
    # Initialize components with smaller dataset
    converter = DataConverter(num_attributes=10, num_annotators=5, num_items=10)
    
    # Load training and test data
    train_data = converter.load_training_data('test_complete_train.json')
    test_data = converter.load_training_data('test_complete_test.json')
    
    print(f"Filtered training data: {len(train_data['ratings'])} ratings, {len(train_data['rankings'])} rankings")
    print(f"Filtered test data: {len(test_data['ratings'])} ratings, {len(test_data['rankings'])} rankings")
    
    # Create variables from actual data
    rating_variables, ranking_variables = converter.create_variables_from_actual_data(train_data, test_data)
    print(f"Total rating variables: {len(rating_variables)}")
    print(f"Total ranking variables: {len(ranking_variables)}")
    print(f"Total variables: {len(rating_variables) + len(ranking_variables)}")
    
    # Process training data
    rating_data, ranking_data = converter.process_training_data(train_data)
    print(f"Available training rating data points: {len(rating_data)}")
    print(f"Available training ranking data points: {len(ranking_data)}")
    
    # Create training batch with masking
    batch = converter.create_training_batch(rating_variables, ranking_variables,
                                          rating_data, ranking_data, test_data=test_data, mask_rate=0.5)
    
    # Count masked and non-masked entries
    train_rating_count = batch['rating_mask'].sum().item()
    train_ranking_count = batch['ranking_mask'].sum().item()
    masked_rating_count = batch['rating_masked'].sum().item()
    masked_ranking_count = batch['ranking_masked'].sum().item()
    print(f"Training data: {train_rating_count} ratings ({masked_rating_count} masked), "
          f"{train_ranking_count} rankings ({masked_ranking_count} masked)")
    
    # Initialize model
    model = MultiVariableImputer(
        num_attributes=10,
        num_annotators=5,
        num_items=10,  # Updated for smaller dataset
        num_likert_classes=5,
        max_rank_size=3,
        encoder_layers_num=2,
        attention_heads=4,
        embedding_dim=64,
        dropout=0.1
    )
    
    # Initialize trainer
    trainer = ImputerTrainer(model, learning_rate=1e-3)
    
    # Training loop
    num_epochs = 10
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in tqdm(range(num_epochs)):
        losses = trainer.train_step(batch)
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Total Loss: {losses['total_loss']:.4f}, "
                  f"Rating Loss: {losses['rating_loss']:.4f}, "
                  f"Ranking Loss: {losses['ranking_loss']:.4f}")
    
    print("Training completed!")
    
    # Evaluation on test data
    print("\nEvaluating on test data...")
    test_losses = trainer.evaluate_with_test_data(batch, test_data, converter)
    
    print(f"\nTest Results:")
    print(f"Test Rating Loss: {test_losses['test_rating_loss']:.4f}")
    print(f"Test Ranking Loss: {test_losses['test_ranking_loss']:.4f}")
    print(f"Total Test Loss: {test_losses['total_test_loss']:.4f}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_attributes': 10,
        'num_annotators': 5,
        'num_items': 10,
        'num_likert_classes': 5,
        'max_rank_size': 3
    }, 'trained_imputer_small.pth')
    
    print("\nModel saved as 'trained_imputer_small.pth'")

if __name__ == "__main__":
    main()