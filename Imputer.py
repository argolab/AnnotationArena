import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from variables import *
from typing import List, Dict, Optional, Union, Tuple

VARIABLE_CLASSES = {
    'CategoricalNumericVariable': CategoricalNumericVariable,
    'OrdinalVariable': OrdinalVariable,
    # Add others as needed
}

class EncoderLayer(nn.Module):
    def __init__(self, d_model, attention_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.attention_heads = attention_heads
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def multihead_attention(self, x, batch_size):
        # More efficient reshaping
        x_flat = x.view(batch_size * x.shape[1], -1)
        
        # Compute Q, K, V in one go
        qkv = torch.cat([
            self.Q(x_flat), 
            self.K(x_flat), 
            self.V(x_flat)
        ], dim=-1)
        
        # Split and reshape
        d_head = self.d_model // self.attention_heads
        qkv = qkv.view(batch_size, -1, 3, self.attention_heads, d_head)
        Q, K, V = qkv.unbind(dim=2)
        
        Q = Q.transpose(1, 2)  # [batch, heads, seq_len, d_head]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        # Apply attention
        output = torch.matmul(scores, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(output)

    def forward(self, x, normalization=True):
        batch_size = x.shape[0]
        if normalization:
            x = self.norm_1(x)
        x = x + self.dropout_1(self.multihead_attention(x, batch_size))
        x = x + self.dropout_2(self.ff(self.norm_2(x)))
        return x

class NormLayer(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        # More numerically stable implementation
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True) + self.eps)
        return self.alpha * (x - mean) / std + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
import json

class Imputer(nn.Module):
    def __init__(self, total_embedding_dimension: int = 30, num_heads: int = 8, 
                 num_layers: int = 6, ff_dim: int = 1024, dropout: float = 0.1):
        
        super().__init__()
        
        self.embedding_dimension = total_embedding_dimension
        self.num_heads = num_heads
        self.encoder_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        # Variables storage
        self.variables = {}  # name -> Variable instance
        self.variable_order = []  # List of variable names in order they were added
        self.variable_embeddings = nn.ParameterDict()  # name -> embedding parameter
        self.norm_layers = nn.ModuleDict()
        self.layers = nn.ModuleList([EncoderLayer(self.embedding_dimension, self.num_heads, dropout) 
                                      for _ in range(self.encoder_layers)])
        self.question_num_to_question_name = {}
        self.var_to_index = {}
    
    def add_variable(self, variable: Variable) -> None:
        if variable.name in self.variables:
            raise ValueError(f"Variable '{variable.name}' already exists.")
        
        # Store the variable
        self.variables[variable.name] = variable
        self.variable_order.append(variable.name)
        
        param_dim = variable.param_dim()
        embedding = nn.Parameter(torch.randn(1, self.embedding_dimension - param_dim) * 0.01)
        self.variable_embeddings[variable.name] = embedding
        
        norm = NormLayer(self.embedding_dimension - param_dim)
        self.norm_layers[variable.name] = norm
    
    def remove_variable(self, name: str) -> None:
        if name not in self.variables:
            raise ValueError(f"Variable '{name}' does not exist.")
        
        del self.variables[name]
        self.variable_order.remove(name)
        
        del self.variable_embeddings[name]
        del self.norm_layers[name]
    
    def set_loss_function(self, variable_name: str, loss_fn) -> None:
        if variable_name not in self.variables:
            raise ValueError(f"Variable '{variable_name}' does not exist.")
        
        self.variables[variable_name].set_loss_function(loss_fn)
    
    def set_query_variables(self, variable_names: List[str], weights: Optional[List[float]] = None) -> None:
        """
        Set which variables are query variables (used for VOI).
        
        Args:
            variable_names: List of variable names to set as queries
            weights: Optional weights for each query variable
        """
        if weights is None:
            weights = [1.0] * len(variable_names)
        
        # Reset all variables to non-query
        for var in self.variables.values():
            var.set_as_query(False)
        
        # Set specified variables as queries
        for name, weight in zip(variable_names, weights):
            if name not in self.variables:
                raise ValueError(f"Variable '{name}' does not exist.")
            self.variables[name].set_as_query(True, weight)
    
    def observe_variable(self, name: str, value) -> None:
        if name not in self.variables:
            raise ValueError(f"Variable '{name}' does not exist.")
        
        self.variables[name].observed_value = value
    
    def unobserve_variable(self, name: str) -> None:
        if name not in self.variables:
            raise ValueError(f"Variable '{name}' does not exist.")
        
        # Remove the observation
        if hasattr(self.variables[name], 'observed_value'):
            delattr(self.variables[name], 'observed_value')
    
    def _prepare_input_features(self, variables: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        batch_size = len(variables)
        features = torch.zeros(batch_size, self.embedding_dimension, device=device)
        
        for i, name in enumerate(variables):
            var = self.variables[name]
            embedding = self.variable_embeddings[name]
            var_dim = var.param_dim()
            
            # Copy embedding part for all variables at once
            features[i, :self.embedding_dimension - var_dim] = embedding
            
            # Handle observed vs unobserved efficiently
            if hasattr(var, 'observed_value'):
                observed_value = var.observed_value
                if not isinstance(observed_value, torch.Tensor):
                    observed_value = torch.tensor(observed_value, device=device)
                features[i, -var_dim:] = observed_value.to(device)
            else:
                features[i, -var_dim:] = var.get_mask_value().to(device)
        
        return features
    
    def forward(self, inputs: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        var_num = inputs.shape[1]
        device = inputs.device
        
        # Create a mapping from question ID to variable info once
        q_to_var_info = {}
        for q_id in questions.unique().tolist():
            if q_id not in self.question_num_to_question_name:
                var_name = f"Q{q_id}"
                self.question_num_to_question_name[q_id] = var_name
                self.var_to_index[var_name] = q_id
                
                # Determine number of categories
                num_categories = inputs.shape[2] - 1
                self.add_variable(CategoricalNumericVariable(var_name, [i+1 for i in range(num_categories)]))
            
            var_name = self.question_num_to_question_name[q_id]
            var = self.variables[var_name]
            var_dim = var.param_dim()
            embedding = self.variable_embeddings[var_name]
            norm_layer = self.norm_layers[var_name]
            
            q_to_var_info[q_id] = {
                'name': var_name,
                'var': var,
                'dim': var_dim,
                'embedding': embedding,
                'norm': norm_layer
            }
        
        # Pre-allocate x tensor
        x = torch.zeros(batch_size, var_num, self.embedding_dimension, device=device)
        
        # Prepare all variable features at once
        observed_vars = set()  # Track which variables were observed
        
        # Populate initial embeddings and handle observations
        for i in range(batch_size):
            for j in range(var_num):
                q_id = questions[i, j].item()
                var_info = q_to_var_info[q_id]
                var_name = var_info['name']
                var = var_info['var']
                var_dim = var_info['dim']
                embedding = var_info['embedding']
                
                # Copy embedding part
                x[i, j, :self.embedding_dimension - var_dim] = embedding
                
                # Handle observation
                if inputs[i, j, 0] == 0:  # Observed
                    observed_vars.add((i, j, var_name))
                    x[i, j, -var_dim:] = inputs[i, j, 1:]
                else:  # Unobserved
                    x[i, j, -var_dim:] = var.get_mask_value().to(device)
        
        # Process through encoder layers
        for layer_idx, layer in enumerate(self.layers):
            # Save original for residual connection
            prev_x = x
            
            # Forward through attention and feed-forward
            x = layer(x)
            
            # Process through variable-specific components - use tensor operations where possible
            new_x = torch.zeros_like(x)
            
            # Apply normalization and combine with param parts
            for i in range(batch_size):
                for j in range(var_num):
                    q_id = questions[i, j].item()
                    var_info = q_to_var_info[q_id]
                    var_dim = var_info['dim']
                    norm_layer = var_info['norm']
                    
                    # Split tensor
                    embedding_part = x[i, j, :-var_dim]
                    param_part = x[i, j, -var_dim:]
                    
                    # Apply normalization
                    normalized_emb = norm_layer(embedding_part)
                    
                    # Combine parts and add residual
                    new_x[i, j] = torch.cat([normalized_emb, param_part], dim=-1) + prev_x[i, j]
            
            # Update x with new tensor
            x = new_x
            
            # Convert logits to distribution parameters at intermediate layers
            if layer_idx < len(self.layers) - 1:
                for i in range(batch_size):
                    for j in range(var_num):
                        q_id = questions[i, j].item()
                        var_info = q_to_var_info[q_id]
                        var = var_info['var']
                        var_dim = var_info['dim']
                        
                        logits = x[i, j, -var_dim:]
                        distribution = var.to_features(logits)
                        x[i, j, -var_dim:] = distribution
        
        return x
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        state['_custom_variable_order'] = self.variable_order
        state['_custom_question_map'] = self.question_num_to_question_name
        state['_custom_var_to_index'] = self.var_to_index

        variable_metadata = {}
        for name in self.variable_order:
            var = self.variables[name]
            metadata = {
                'type': type(var).__name__,
                'name': name,
            }

            if isinstance(var, CategoricalNumericVariable):
                metadata['numeric_categories'] = var.domain  # or list(var.numeric_values.cpu().numpy())
            
            # Other variable types here as needed

            variable_metadata[name] = metadata

        state['_custom_variable_metadata'] = variable_metadata
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        self.variable_order = state_dict.get('_custom_variable_order', [])
        self.question_num_to_question_name = state_dict.get('_custom_question_map', {})
        self.var_to_index = state_dict.get('_custom_var_to_index', {})
        var_metadata = state_dict.get('_custom_variable_metadata', {})

        for var_name in self.variable_order:
            if var_name not in self.variables:
                metadata = var_metadata[var_name]
                var_class_name = metadata['type']

                if var_class_name == 'CategoricalNumericVariable':
                    var = CategoricalNumericVariable(
                        metadata['name'],
                        metadata['numeric_categories']  # <- previously was missing
                    )
                else:
                    raise ValueError(f"Unsupported variable type: {var_class_name}")

                self.add_variable(var)

        # Now load all the parameters
        super().load_state_dict({k: v for k, v in state_dict.items() if not k.startswith('_custom_')}, strict)
    
    def _compute_total_loss(self, outputs: torch.Tensor, labels: torch.Tensor, 
                           inputs: torch.Tensor, questions: torch.Tensor, 
                           observe_only: bool = False, variables: Optional[List[str]] = None, 
                           parametric: bool = False) -> torch.Tensor:
        
        batch_size = outputs.shape[0]
        var_num = outputs.shape[1]
        device = outputs.device
        loss = torch.tensor(0.0, device=device)
        
        for i in range(batch_size):
            for j in range(var_num):
                # Skip if observed and we only want loss for masked
                if inputs[i, j, 0] == 0 and observe_only:
                    continue
                
                # Get variable name
                q_id = questions[i, j].item()
                var_name = self.question_num_to_question_name[q_id]
                
                # Skip if not in target variables
                if variables is not None and var_name not in variables:
                    continue
                
                # Get the variable
                var = self.variables[var_name]
                var_dim = var.param_dim()
                
                # Compute loss
                var_loss = var.compute_loss(outputs[i, j, -var_dim:], labels[i, j], parametric=parametric)
                loss += var_loss
        
        return loss
    
    def compute_voi(self, query_variable: str, candidate_variable: str, 
                   inputs: torch.Tensor, labels: torch.Tensor, questions: torch.Tensor, 
                   device: torch.device, added_variables: Optional[List[str]] = None, 
                   oracle: bool = False) -> float:
        
        self.eval()
        
        # Find indices for query and candidate variables
        query_idx = self.var_to_index.get(query_variable)
        candidate_idx = self.var_to_index.get(candidate_variable)
        
        if query_idx is None or candidate_idx is None:
            raise ValueError(f"Could not find indices for query variable {query_variable} or candidate variable {candidate_variable}")
        
        # Track which positions in the batch correspond to our variables of interest
        query_pos = None
        candidate_pos = None
        
        for j in range(inputs.shape[1]):
            if questions[0, j].item() == query_idx:
                query_pos = j
            if questions[0, j].item() == candidate_idx:
                candidate_pos = j
        
        if query_pos is None or candidate_pos is None:
            raise ValueError(f"Could not find positions for query variable {query_variable} or candidate variable {candidate_variable}")
        
        with torch.no_grad():
            # Add known variables to the input if any
            if added_variables:
                inputs_copy = inputs.clone()
                for var_name in added_variables:
                    var_idx = self.var_to_index[var_name]
                    for j in range(inputs.shape[1]):
                        if questions[0, j].item() == var_idx:
                            inputs_copy[0, j, 0] = 0  # Mark as observed
                            inputs_copy[0, j, 1:] = labels[0, j]  # Set to true value
                inputs = inputs_copy
            
            # Forward pass
            outputs = self(inputs, questions)
            
            # Get query variable info
            var = self.variables[query_variable]
            num_classes = var.param_dim()
            query_outputs = var.to_features(outputs[0, query_pos, -num_classes:])
            query_outputs = query_outputs.unsqueeze(0)  # [1, num_classes]
            
            # Generate possible label values
            possible_labels = torch.arange(1, num_classes + 1, dtype=torch.float, device=device)
            
            # Calculate expected rating
            expected_rating = torch.sum(query_outputs * possible_labels, dim=1)
            
            # Calculate squared errors for each possible true label
            squared_errors = torch.zeros(1, num_classes, device=device)
            for i in range(num_classes):
                true_label = i + 1 
                squared_errors[0, i] = (expected_rating - true_label) ** 2
            
            # Initial expected loss
            initial_loss = torch.sum(query_outputs * squared_errors, dim=1)
            initial_loss_mean = initial_loss.item()
            
            if oracle:
                # Oracle knows the true answer for the candidate variable
                inputs_updated = inputs.clone()
                inputs_updated[0, candidate_pos, 0] = 0  # Set as observed
                inputs_updated[0, candidate_pos, 1:] = labels[0, candidate_pos]  # Set to true value
                
                # Forward pass with oracle knowledge
                outputs_updated = self(inputs_updated, questions)
                
                # Get updated query predictions
                query_outputs_updated = var.to_features(outputs_updated[0, query_pos, -num_classes:])
                query_outputs_updated = query_outputs_updated.unsqueeze(0)
                
                # Calculate updated expected rating
                expected_rating_updated = torch.sum(query_outputs_updated * possible_labels, dim=1)
                
                # Calculate updated squared errors
                squared_errors_updated = torch.zeros(1, num_classes, device=device)
                for i in range(num_classes):
                    true_label = i + 1
                    squared_errors_updated[0, i] = (expected_rating_updated - true_label) ** 2
                
                # Final expected loss with oracle knowledge
                updated_loss = torch.sum(query_outputs_updated * squared_errors_updated, dim=1)
                expected_loss = updated_loss.item()
                
                # Value of information is the reduction in expected loss
                voi = initial_loss_mean - expected_loss
            else:
                # Without oracle, calculate expected loss over possible values
                candidate_var = self.variables[candidate_variable]
                candidate_dim = candidate_var.param_dim()
                
                # Get distribution for candidate variable
                candidate_output = candidate_var.to_features(outputs[0, candidate_pos, -candidate_dim:])
                candidate_distribution = candidate_output.unsqueeze(0)  # [1, candidate_dim]
                
                # Store expected losses for each possible value
                expected_losses = []
                
                for class_idx in range(candidate_dim):
                    # Create inputs with this class value
                    inputs_with_class = inputs.clone()
                    inputs_with_class[0, candidate_pos, 0] = 0  # Set as observed
                    
                    # Create one-hot encoded answer
                    one_hot = torch.zeros(candidate_dim, device=device)
                    one_hot[class_idx] = 1
                    
                    inputs_with_class[0, candidate_pos, 1:1+candidate_dim] = one_hot
                    
                    # Forward pass with this variable value
                    outputs_with_class = self(inputs_with_class, questions)
                    
                    # Extract outputs for the query variable
                    query_outputs_with_class = var.to_features(outputs_with_class[0, query_pos, -num_classes:])
                    query_outputs_with_class = query_outputs_with_class.unsqueeze(0)  # [1, num_classes]
                    
                    # Calculate expected rating
                    expected_rating_with_class = torch.sum(query_outputs_with_class * possible_labels, dim=1)
                    
                    # Calculate squared errors
                    squared_errors_with_class = torch.zeros(1, num_classes, device=device)
                    for i in range(num_classes):
                        true_label = i + 1
                        squared_errors_with_class[0, i] = (expected_rating_with_class - true_label) ** 2
                    
                    class_loss = torch.sum(query_outputs_with_class * squared_errors_with_class, dim=1).item()
                    expected_losses.append(class_loss)
                
                expected_losses = torch.tensor(expected_losses, device=device)
                
                # Calculate weighted loss based on candidate distribution
                weighted_loss = torch.sum(candidate_distribution[0] * expected_losses)
                expected_loss = weighted_loss.item()
                
                # Value of information is the reduction in expected loss
                voi = initial_loss_mean - expected_loss

        return voi