import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional

class FeatureRecorder:
    """
    Records features for each example at each cycle to train a regression model
    for predicting example scores in active learning.
    """
    
    def __init__(self, model, device=None):
        """
        Initialize feature recorder.
        
        Args:
            model: The imputer model (ImputerEmbedding)
            device: Device to use for computations
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature storage
        self.cycle_features = []  # List of feature dicts for each cycle
        self.selection_history = defaultdict(int)  # Track variable selection frequency
        self.annotated_embeddings = []  # Store embeddings of annotated examples
        
        # Cache for validation embeddings (computed once)
        self.validation_embeddings_cache = None
        self.validation_embeddings_computed = False
        
    def extract_top_layer_features(self, inputs, annotators, questions, embeddings):
        """
        Extract features from the top layer of the imputer model.
        
        Args:
            inputs: Input tensor [batch_size, seq_len, input_dim]
            annotators: Annotator tensor [batch_size, seq_len]
            questions: Question tensor [batch_size, seq_len] 
            embeddings: Text embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            Top layer features [batch_size, seq_len, feature_dim]
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get features from the encoder's final layer
            feature_x, param_x = self.model.encoder.position_encoder(inputs, annotators, questions, embeddings)
            
            # Pass through all but the last encoder layer
            for layer in self.model.encoder.layers[:-1]:
                feature_x, param_x = layer(feature_x, param_x)
            
            # Get features from the last layer (before final output)
            final_feature_x, final_param_x = self.model.encoder.layers[-1](feature_x, param_x)
            
            return final_feature_x
    
    def compute_posterior_entropy(self, inputs, annotators, questions, embeddings, masked_positions):
        """
        Compute average posterior entropy for masked positions.
        
        Args:
            inputs: Input tensor [1, seq_len, input_dim]
            annotators: Annotator tensor [1, seq_len]
            questions: Question tensor [1, seq_len]
            embeddings: Text embeddings [1, seq_len, embedding_dim]
            masked_positions: List of masked position indices
            
        Returns:
            Average entropy across masked positions
        """
        if not masked_positions:
            return 0.0
            
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(inputs, annotators, questions, embeddings)
            
            entropies = []
            for pos in masked_positions:
                logits = outputs[0, pos]
                probs = F.softmax(logits, dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                entropies.append(entropy.item())
            
            return np.mean(entropies) if entropies else 0.0
    
    def compute_text_embedding_similarities(self, example_embedding, validation_embeddings, annotated_embeddings=None):
        """
        Compute cosine similarities with validation and annotated examples.
        
        Args:
            example_embedding: Embedding of current example [seq_len, embedding_dim]
            validation_embeddings: List of validation embeddings
            annotated_embeddings: List of annotated example embeddings
            
        Returns:
            Tuple of (avg_val_similarity, avg_annotated_similarity)
        """
        # Average the sequence embeddings to get example-level embedding
        if len(example_embedding.shape) == 2:
            example_embed = example_embedding.mean(dim=0).cpu().numpy()
        else:
            example_embed = example_embedding.cpu().numpy()
        
        # Compute similarities with validation set
        val_similarities = []
        for val_embed in validation_embeddings:
            if len(val_embed.shape) == 2:
                val_embed_avg = val_embed.mean(dim=0).cpu().numpy()
            else:
                val_embed_avg = val_embed.cpu().numpy()
            
            similarity = cosine_similarity([example_embed], [val_embed_avg])[0, 0]
            val_similarities.append(similarity)
        
        avg_val_similarity = np.mean(val_similarities) if val_similarities else 0.0
        
        # Compute similarities with annotated examples
        avg_annotated_similarity = 0.0
        if annotated_embeddings:
            annotated_similarities = []
            for ann_embed in annotated_embeddings:
                if len(ann_embed.shape) == 2:
                    ann_embed_avg = ann_embed.mean(dim=0).cpu().numpy()
                else:
                    ann_embed_avg = ann_embed.cpu().numpy()
                
                similarity = cosine_similarity([example_embed], [ann_embed_avg])[0, 0]
                annotated_similarities.append(similarity)
            
            avg_annotated_similarity = np.mean(annotated_similarities)
        
        return avg_val_similarity, avg_annotated_similarity
    
    def extract_masked_observed_pattern(self, inputs, questions):
        """
        Extract the masked/observed pattern for each variable.
        
        Args:
            inputs: Input tensor [1, seq_len, input_dim]
            questions: Question tensor [1, seq_len]
            
        Returns:
            Dictionary mapping question_id to pattern info
        """
        pattern_info = {}
        
        for pos in range(inputs.shape[1]):
            question_id = questions[0, pos].item()
            is_masked = inputs[0, pos, 0].item() == 1
            
            if question_id not in pattern_info:
                pattern_info[question_id] = {
                    'total_positions': 0,
                    'masked_positions': 0,
                    'observed_positions': 0
                }
            
            pattern_info[question_id]['total_positions'] += 1
            if is_masked:
                pattern_info[question_id]['masked_positions'] += 1
            else:
                pattern_info[question_id]['observed_positions'] += 1
        
        return pattern_info
    
    def update_selection_frequency(self, selected_variables):
        """
        Update the frequency count for selected variables.
        
        Args:
            selected_variables: List of variable identifiers that were selected
        """
        for var in selected_variables:
            self.selection_history[var] += 1
    
    def get_variable_selection_frequencies(self, example_idx, masked_positions):
        """
        Get selection frequencies for variables in the current example.
        
        Args:
            example_idx: Index of the current example
            masked_positions: List of masked positions in the example
            
        Returns:
            List of selection frequencies for each masked position
        """
        frequencies = []
        for pos in masked_positions:
            var_id = f"example_{example_idx}_position_{pos}"
            frequencies.append(self.selection_history.get(var_id, 0))
        
        return frequencies
    
    def compute_validation_embeddings(self, val_dataset):
        """
        Compute and cache validation set embeddings.
        
        Args:
            val_dataset: Validation dataset
        """
        if self.validation_embeddings_computed:
            return
        
        self.validation_embeddings_cache = []
        
        for idx in range(len(val_dataset)):
            entry = val_dataset.get_data_entry(idx)
            embeddings = entry.get('embeddings', None)
            if embeddings is not None:
                self.validation_embeddings_cache.append(embeddings)
        
        self.validation_embeddings_computed = True
    
    def record_cycle_features(self, cycle_num, dataset, active_pool, annotated_examples, 
                            val_dataset, selected_examples_with_scores):
        """
        Record features for all examples in the active pool for the current cycle.
        
        Args:
            cycle_num: Current cycle number
            dataset: Training dataset
            active_pool: List of active pool example indices
            annotated_examples: List of annotated example indices
            val_dataset: Validation dataset
            selected_examples_with_scores: List of (example_idx, score) tuples for selected examples
        """
        print(f"Recording features for cycle {cycle_num}")
        
        # Compute validation embeddings if not already done
        self.compute_validation_embeddings(val_dataset)
        
        # Update annotated embeddings
        for ann_idx in annotated_examples:
            if ann_idx < len(dataset):
                entry = dataset.get_data_entry(ann_idx)
                embeddings = entry.get('embeddings', None)
                if embeddings is not None and ann_idx not in [ae[0] if isinstance(ae, tuple) else ae for ae in self.annotated_embeddings]:
                    self.annotated_embeddings.append((ann_idx, embeddings))
        
        cycle_features = {
            'cycle': cycle_num,
            'features': [],
            'scores': [],
            'example_indices': []
        }
        
        # Create score mapping for selected examples
        score_mapping = {ex_idx: score for ex_idx, score in selected_examples_with_scores}
        
        for pool_idx, example_idx in enumerate(active_pool):
            try:
                # Get example data
                entry = dataset.get_data_entry(example_idx)
                inputs = torch.tensor(entry['input']).unsqueeze(0).to(self.device)  # Add batch dimension
                annotators = torch.tensor(entry['annotators']).unsqueeze(0).to(self.device)
                questions = torch.tensor(entry['questions']).unsqueeze(0).to(self.device)
                embeddings = entry.get('embeddings', None)
                
                if embeddings is not None:
                    embeddings = torch.tensor(embeddings).unsqueeze(0).to(self.device)
                else:
                    # Create dummy embeddings if not available
                    embeddings = torch.zeros(1, inputs.shape[1], 384).to(self.device)
                
                # Get masked positions
                masked_positions = []
                for j in range(inputs.shape[1]):
                    if inputs[0, j, 0] == 1:
                        masked_positions.append(j)
                
                if not masked_positions:
                    continue
                
                # Extract features
                features = {}
                
                # 1. Posterior entropy
                features['posterior_entropy'] = self.compute_posterior_entropy(
                    inputs, annotators, questions, embeddings, masked_positions
                )
                
                # 2. Top layer features
                top_layer_features = self.extract_top_layer_features(inputs, annotators, questions, embeddings)
                # Average across sequence length and flatten
                features['top_layer_features'] = top_layer_features[0].mean(dim=0).cpu().numpy().tolist()
                
                # 3. Text embedding similarities
                val_sim, ann_sim = self.compute_text_embedding_similarities(
                    embeddings[0], 
                    self.validation_embeddings_cache,
                    [emb for _, emb in self.annotated_embeddings]
                )
                print(val_sim)
                print(ann_sim)
                features['avg_val_embedding_similarity'] = val_sim
                features['avg_annotated_embedding_similarity'] = ann_sim
                
                # 4. Variable selection frequencies
                selection_frequencies = self.get_variable_selection_frequencies(example_idx, masked_positions)
                features['avg_selection_frequency'] = np.mean(selection_frequencies) if selection_frequencies else 0.0
                features['max_selection_frequency'] = np.max(selection_frequencies) if selection_frequencies else 0.0
                features['min_selection_frequency'] = np.min(selection_frequencies) if selection_frequencies else 0.0
                
                # 5. Masked/observed pattern
                pattern_info = self.extract_masked_observed_pattern(inputs, questions)
                features['total_masked_positions'] = len(masked_positions)
                features['total_observed_positions'] = inputs.shape[1] - len(masked_positions)
                features['masking_ratio'] = len(masked_positions) / inputs.shape[1]
                
                # Add pattern info for each question type
                for q_id, pattern in pattern_info.items():
                    features[f'question_{q_id}_masked_ratio'] = (
                        pattern['masked_positions'] / pattern['total_positions'] 
                        if pattern['total_positions'] > 0 else 0.0
                    )
                
                # 6. Additional features
                features['num_annotated_examples'] = len(self.annotated_embeddings)
                features['cycle_number'] = cycle_num
                features['example_index'] = example_idx
                
                # Store features and score
                cycle_features['features'].append(features)
                cycle_features['example_indices'].append(example_idx)
                
                # Get score if this example was selected
                score = score_mapping.get(example_idx, 0.0)
                cycle_features['scores'].append(score)
                
            except Exception as e:
                continue
        
        self.cycle_features.append(cycle_features)
        print(f"Recorded features for {len(cycle_features['features'])} examples in cycle {cycle_num}")
    
    def update_from_selections(self, selected_examples, selected_variables_info):
        """
        Update internal state based on selections made in the current cycle.
        
        Args:
            selected_examples: List of selected example indices
            selected_variables_info: List of (example_idx, position) tuples for selected variables
        """
        # Update selection frequency
        selected_vars = []
        for ex_idx, pos in selected_variables_info:
            var_id = f"example_{ex_idx}_position_{pos}"
            selected_vars.append(var_id)
        
        self.update_selection_frequency(selected_vars)
    
    def get_feature_matrix(self, cycle_nums=None):
        """
        Get feature matrix and scores for training a regression model.
        
        Args:
            cycle_nums: List of cycle numbers to include (None for all)
            
        Returns:
            Tuple of (feature_matrix, scores, metadata)
        """
        if cycle_nums is None:
            cycles_to_use = self.cycle_features
        else:
            cycles_to_use = [cf for cf in self.cycle_features if cf['cycle'] in cycle_nums]
        
        all_features = []
        all_scores = []
        all_metadata = []
        
        for cycle_data in cycles_to_use:
            for i, features in enumerate(cycle_data['features']):
                # Convert features to flat vector
                feature_vector = []
                
                # Add scalar features
                scalar_features = [
                    'posterior_entropy', 'avg_val_embedding_similarity', 
                    'avg_annotated_embedding_similarity', 'avg_selection_frequency',
                    'max_selection_frequency', 'min_selection_frequency',
                    'total_masked_positions', 'total_observed_positions', 
                    'masking_ratio', 'num_annotated_examples', 'cycle_number'
                ]
                
                for feat_name in scalar_features:
                    feature_vector.append(features.get(feat_name, 0.0))
                
                # Add question-specific masking ratios (assuming 7 questions max)
                for q_id in range(7):
                    feat_name = f'question_{q_id}_masked_ratio'
                    feature_vector.append(features.get(feat_name, 0.0))
                
                # Add top layer features (flattened)
                top_layer_feats = features.get('top_layer_features', [])
                if isinstance(top_layer_feats, list):
                    feature_vector.extend(top_layer_feats)
                else:
                    # If it's a numpy array or tensor
                    feature_vector.extend(top_layer_feats.tolist())
                
                all_features.append(feature_vector)
                all_scores.append(cycle_data['scores'][i])
                all_metadata.append({
                    'cycle': cycle_data['cycle'],
                    'example_idx': cycle_data['example_indices'][i]
                })
        
        return np.array(all_features), np.array(all_scores), all_metadata
    
    def save_features(self, filepath):
        """Save recorded features to file."""
        data = {
            'cycle_features': self.cycle_features,
            'selection_history': dict(self.selection_history),
            'annotated_embeddings': [(idx, emb.cpu() if torch.is_tensor(emb) else emb) 
                                   for idx, emb in self.annotated_embeddings]
        }
        print(self.cycle_features)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_features(self, filepath):
        """Load recorded features from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.cycle_features = data['cycle_features']
        self.selection_history = defaultdict(int, data['selection_history'])
        self.annotated_embeddings = [(idx, emb.to(self.device) if torch.is_tensor(emb) else emb) 
                                   for idx, emb in data['annotated_embeddings']]
