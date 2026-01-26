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
from typing import List, Dict
import numpy as np
from itertools import combinations
from collections import defaultdict
import logging
from tqdm import tqdm

# New modular components (use absolute imports)
from imputer.embedding import AtomCompositonalEmbeddingProvider
from imputer.transformer import TransformerBlock, NormLayer as _NormLayer
from imputer.data import RankingData, DataConverter
from imputer.legacy.trainer import ImputerTrainer

# Export for backward compatibility with experiment runner
__all__ = ['MultiVariableImputer', 'DataConverter', 'ImputerTrainer', 'RankingData']

logger = logging.getLogger(__name__)


class MultiVariableImputer(nn.Module):
    """Imputer managing embeddings, transformer blocks, and output heads.

    Encapsulates all learnable params for easy checkpointing. Supports returning
    per-head logits and optionally intermediate hidden states for layerwise supervision.
    """

    def __init__(self,
                 num_attributes=8,
                 num_annotators=8,
                 num_items=8,
                 num_likert_classes=5,
                 max_rank_size=3,
                 encoder_layers_num=2,
                 attention_heads=4,
                 embedding_dim=64,
                 dropout=0.1,
                 device="cuda",
                 embedding_dropout: float | None = None,
                 use_gelu_after_attention=False,
                 use_final_norm=True,
                 normalize_parameter=False,
                 num_ffn_layers=4,
                 temperature=1.0,
                 use_concat_embedding: bool = False,
                 batch_size: int = 1,
                 enable_pointer_mechanism: bool = True):
        super().__init__()
        self.device = torch.device(device)
        # Defer moving to device until after all submodules are created
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.num_likert_classes = num_likert_classes
        self.max_rank_size = max_rank_size
        self.embedding_dim = embedding_dim
        self.embedding_type = "atom"
        self.use_gelu_after_attention = use_gelu_after_attention
        self.use_final_norm = use_final_norm
        self.num_ffn_layers = num_ffn_layers
        self.temperature = temperature
        self.use_concat_embedding = use_concat_embedding
        self.batch_size = batch_size
        self.enable_pointer_mechanism = enable_pointer_mechanism
        if embedding_dropout is None:
            embedding_dropout = dropout

        self.embedding_provider = AtomCompositonalEmbeddingProvider(
            num_attributes,
            num_annotators,
            num_items,
            embedding_dim,
            num_likert_classes,
            max_rank_size,
            self.device,
            embedding_dropout=embedding_dropout,
            use_concat_embedding=use_concat_embedding,
        )

        # Use provider-declared parameter dimension (includes missing-status bit)
        param_dim = self.embedding_provider.parameter_dimension
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, param_dim, attention_heads, dropout, use_gelu_after_attention, 
                           normalize_parameter=normalize_parameter, num_ffn_layers=num_ffn_layers,
                           enable_pointer_mechanism=enable_pointer_mechanism)
            for _ in range(encoder_layers_num)
        ])

        # Final normalization for Pre-LN architecture (CRITICAL for stability, but optional for experiments)
        if use_final_norm:
            self.final_norm = _NormLayer(param_dim)
        else:
            self.final_norm = None

        self.to(self.device)

    def load_state_dict_with_warnings(self, state_dict, strict=True):
        """Load state dict with warnings for extra/missing keys."""
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        if unexpected_keys:
            print(f"Warning: Found unexpected keys in checkpoint (ignoring): {unexpected_keys}")
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(f"State dict loading failed. Missing: {missing_keys}, Unexpected: {unexpected_keys}")
            
        return missing_keys, unexpected_keys

    def read_prediction_logits_from_param(self, head_key: str, hidden: torch.Tensor) -> torch.Tensor:
        """Apply a named head to hidden states [B, N, D] -> logits [B, N, *]. Skip the missing-status bit."""
        if head_key == "rating":
            return hidden[:, :, 1:1 + self.num_likert_classes]
        else:
            return hidden[:, :, 1:1 + self.max_rank_size]


    def forward(self, ranking_data_list, attn_mask: torch.Tensor | None = None, return_intermediate: bool = False):
        # Support both structured list inputs and legacy tensor inputs
        
        # Debug: Count tokens by type and status
        total_tokens = len(ranking_data_list)
        rating_tokens = sum(1 for var in ranking_data_list if not var.is_listwise)
        pairwise_tokens = sum(1 for var in ranking_data_list if var.is_listwise)
        status_counts = {0: 0, 1: 0, 2: 0}
        for var in ranking_data_list:
            if var.status in status_counts:
                status_counts[var.status] += 1
        print(
            f"\033[96m[DEBUG] Forward pass: {total_tokens} total tokens "
            f"({rating_tokens} ratings, {pairwise_tokens} pairwise rankings) | "
            f"by status: missing={status_counts[0]}, masked={status_counts[1]}, observed={status_counts[2]}"
            f"\033[0m"
        )
        # import traceback
        # stack = traceback.format_stack(limit=5)
        # print("\033[96m[DEBUG] Stack Trace (most recent call last):\033[0m")
        # for line in stack:
        #     print(f"\033[96m{line.strip()}\033[0m")

        features, params = self.embedding_provider(ranking_data_list, batch_size=self.batch_size)

        # Precompute indicator matrices for pointer mechanism
        # Shape: [B, N, N, 3] where [b, i, j, :] = [attr_match, annot_match, item_match] for query i and key j
        N = len(ranking_data_list)
        device = features.device
        
        if self.enable_pointer_mechanism:
            # Extract metadata for each token
            # Debug: uncomment the line below to enable interactive debugging
            # When running from bash script, use: DEBUG=1 ./run_single_kp10_random_as_key_fresh_lightning.sh
            # import pdb; pdb.set_trace()
            
            attr_ids = torch.tensor([var.attribute_id for var in ranking_data_list], device=device)  # [N]
            annot_ids = torch.tensor([var.annotator_id for var in ranking_data_list], device=device)  # [N]
            # For item_ids, handle both rating (single item) and ranking (multiple items)
            # For pointer mechanism, we'll use the first item for rankings
            item_ids = torch.tensor([var.item_ids[0] if len(var.item_ids) > 0 else -1 for var in ranking_data_list], device=device)  # [N]
            
            # Create indicator matrices [N, N] for attribute/annotator/item sameness
            # [i, j] = 1 if token i and token j share the same attribute/annotator/item
            attr_indicator = (attr_ids.unsqueeze(0) == attr_ids.unsqueeze(1)).float()  # [N, N]
            annot_indicator = (annot_ids.unsqueeze(0) == annot_ids.unsqueeze(1)).float()  # [N, N]
            item_indicator = (item_ids.unsqueeze(0) == item_ids.unsqueeze(1)).float()  # [N, N]
            
            # Stack to get [N, N, 3] - for each query i and key j, we have [attr_match, annot_match, item_match]
            # Then add batch dimension: [B, N, N, 3]
            B = features.shape[0]
            K_aug = torch.stack([attr_indicator, annot_indicator, item_indicator], dim=-1)  # [N, N, 3]
            K_aug = K_aug.unsqueeze(0).expand(B, -1, -1, -1)  # [B, N, N, 3]
        else:
            K_aug = None

        hidden_intermediates = []
        if return_intermediate:
            # Add embeddings as the first entry
            hidden_intermediates.append([features, params])

        for block in self.blocks:
            features, params = block(features, params, attn_mask=attn_mask, K_aug=K_aug)
            if return_intermediate:
                hidden_intermediates.append([features, params])

        # Apply final normalization (critical for Pre-LN transformer stability)
        if self.final_norm is not None:
            params = self.final_norm(params)

        # Read logits from parameters
        rating_logits = self.read_prediction_logits_from_param('rating', params)
        ranking_logits = self.read_prediction_logits_from_param('ranking', params)

        # Apply temperature scaling to soften/sharpen predictions
        logits = {
            'rating': rating_logits / self.temperature,
            'ranking': ranking_logits / self.temperature,
        }
        if return_intermediate:
            return logits, hidden_intermediates
        return logits


def main():
    from imputer.data import DataConverter, RankingData
    from imputer.legacy.trainer import ImputerTrainer
    
    converter = DataConverter(
        num_attributes=2, num_annotators=2, num_items=3,
        num_likert_classes=5, max_rank_size=2
    )
    
    data = {
        'ratings': [
            {'annotator': 1, 'attribute': 1, 'item': 1, 'value': 2},
            {'annotator': 2, 'attribute': 1, 'item': 2, 'value': 3},
            {'annotator': 1, 'attribute': 2, 'item': 1, 'value': 1},
            {'annotator': 2, 'attribute': 2, 'item': 2, 'value': 4},
        ],
        'pairwise_rankings': [
            {'annotator': 1, 'attribute': 1, 'items': [1, 2], 'order': [1, 2]},
            {'annotator': 2, 'attribute': 2, 'items': [1, 2], 'order': [2, 1]},
        ]
    }
    variables = converter.create_variables(data)
    
    batch = converter.create_masked_batch(variables, 0.5, 10)
    
    # Initialize model
    model = MultiVariableImputer(
        num_attributes=2,
        num_annotators=2,
        num_items=3,  # Updated for smaller dataset
        num_likert_classes=5,
        max_rank_size=2,
        encoder_layers_num=2,
        attention_heads=4,
        embedding_dim=64,
        dropout=0.1
    )

    out = model(batch)

    print("Forward pass is succssful")
    
    # # Initialize trainer
    # trainer = ImputerTrainer(model, learning_rate=1e-3)
    
    # # Training loop
    # num_epochs = 10
    # print(f"\nStarting training for {num_epochs} epochs...")
    
    # for epoch in tqdm(range(num_epochs)):
    #     losses = trainer.train_step(batch)
        
    #     if epoch % 1 == 0:
    #         print(f"Epoch {epoch}: Total Loss: {losses['total_loss']:.4f}, "
    #               f"Rating Loss: {losses['rating_loss']:.4f}, "
    #               f"Ranking Loss: {losses['ranking_loss']:.4f}")
    
    # print("Training completed!")
    
    # # Evaluation on test data
    # print("\nEvaluating on test data...")
    # test_losses = trainer.evaluate_with_test_data(batch, test_data, converter)
    
    # print(f"\nTest Results:")
    # print(f"Test Rating Loss: {test_losses['test_rating_loss']:.4f}")
    # print(f"Test Ranking Loss: {test_losses['test_ranking_loss']:.4f}")
    # print(f"Total Test Loss: {test_losses['total_test_loss']:.4f}")
    
    # # Save model
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'num_attributes': 10,
    #     'num_annotators': 5,
    #     'num_items': 10,
    #     'num_likert_classes': 5,
    #     'max_rank_size': 3
    # }, 'trained_imputer_small.pth')
    
    # print("\nModel saved as 'trained_imputer_small.pth'")

if __name__ == "__main__":
    main()
