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
import sys
from typing import List, Dict
import numpy as np
from itertools import combinations
from collections import defaultdict
import logging
from tqdm import tqdm

# New modular components (use relative imports only)
from .embedding import OuterProductRankingEmbeddingProvider, PairwiseRankingProjectionEmbeddingProvider
from .transformer import TransformerBlock, NormLayer as _NormLayer
from .data import RankingData, DataConverter
from .trainer import ImputerTrainer

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
                 embedding_type="pairwise",
                 device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.num_likert_classes = num_likert_classes
        self.max_rank_size = max_rank_size
        self.embedding_dim = embedding_dim

        if embedding_type == "pairwise":
            self.embedding_provider = PairwiseRankingProjectionEmbeddingProvider(
                num_attributes, num_annotators, num_items, embedding_dim, num_likert_classes, max_rank_size, self.device
            )
        # Probably not needed for ICLR but will need this for later.
        elif embedding_type == "outer_product":
            self.embedding_provider = OuterProductRankingEmbeddingProvider(
                num_attributes, num_annotators, num_items, embedding_dim, num_likert_classes, max_rank_size, self.device
            )
            print("WARNING - You shouldn't be here!")
            sys.exit()
        else:
            print("WARNING - You shouldn't be here also!")
            sys.exit()
            pass
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, attention_heads, dropout)
            for _ in range(encoder_layers_num)
        ])
        # use transformer.NormLayer implementation
        self.norm = _NormLayer(embedding_dim)

        # Output heads in a ModuleDict for extensibility
        self.heads = nn.ModuleDict({
            'rating': nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim // 2, num_likert_classes)
            ),
            'ranking': nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim // 2, max_rank_size)
            ),
        })

    def apply_head(self, head_key: str, hidden: torch.Tensor) -> torch.Tensor:
        """Apply a named head to hidden states [B, N, D] -> logits [B, N, *]."""
        return self.heads[head_key](hidden)

    def _convert_legacy_tensors_to_ranking_data(self, variable_data, variable_types, attribute_ids, annotator_ids, item_ids) -> List[RankingData]:
        """Convert legacy tensor format to List[RankingData]."""
        batch_size, num_variables = variable_types.shape
        assert batch_size == 1, "Only batch size 1 supported for conversion"
        
        variables = []
        for i in range(num_variables):
            var_type = variable_types[0, i].item()
            attr_id = attribute_ids[0, i].item()
            annot_id = annotator_ids[0, i].item()
            
            if var_type == 0:  # Rating
                item_id = item_ids[0, i, 0].item()
                # Check if this rating has supervision (non-zero data)
                rating_value = None
                data_vec = variable_data[0, i]
                if data_vec.sum() > 0:
                    rating_value = torch.argmax(data_vec).item()
                
                variables.append(RankingData(
                    annotator_id=annot_id,
                    attribute_id=attr_id,
                    is_listwise=False,
                    item_ids=[item_id],
                    rating_value=rating_value
                ))
            else:  # Ranking
                # Extract valid item IDs (non-negative)
                item_list = []
                for j in range(self.max_rank_size):
                    item_id = item_ids[0, i, j].item()
                    if item_id >= 0:
                        item_list.append(item_id)
                
                # Check if this ranking has supervision (non-zero data)
                ranking_order = None
                data_vec = variable_data[0, i]
                if data_vec.sum() > 0:
                    ranking_order = []
                    for j in range(len(item_list)):
                        if j < data_vec.shape[0]:
                            rank_pos = int(data_vec[j].item())
                            if rank_pos > 0:
                                ranking_order.append(rank_pos)
                
                variables.append(RankingData(
                    annotator_id=annot_id,
                    attribute_id=attr_id,
                    is_listwise=True,
                    item_ids=item_list,
                    ranking_order=ranking_order
                ))
        
        return variables

    def forward_hidden(self, variable_data, variable_types=None, attribute_ids=None, annotator_ids=None, item_ids=None, attn_mask: torch.Tensor | None = None):
        """Return intermediate hidden states from each transformer block and the final normalized features.

        Returns list[Tensor] of length = num_blocks + 1, where the last is post-norm features.
        """
        # Support both structured list inputs and legacy tensor inputs
        if isinstance(variable_data, list):
            features = self.embedding_provider(variable_data)
        else:
            # Convert legacy tensor format to List[RankingData]
            ranking_data_list = self._convert_legacy_tensors_to_ranking_data(
                variable_data, variable_types, attribute_ids, annotator_ids, item_ids
            )
            features = self.embedding_provider(ranking_data_list)

        hiddens = []
        for block in self.blocks:
            features = block(features, attn_mask=attn_mask)
            hiddens.append(features)
        features = self.norm(features)
        hiddens.append(features)
        return hiddens

    def forward(self, ranking_data_list, attn_mask: torch.Tensor | None = None, return_hidden: bool = False):
        # Support both structured list inputs and legacy tensor inputs

        features = self.embedding_provider(ranking_data_list)

        hidden_states = []
        for block in self.blocks:
            features = block(features, attn_mask=attn_mask)
            if return_hidden:
                hidden_states.append(features)
        features = self.norm(features)
        if return_hidden:
            hidden_states.append(features)

        logits = {
            'rating': self.apply_head('rating', features),
            'ranking': self.apply_head('ranking', features),
        }
        if return_hidden:
            return logits, hidden_states
        return logits


class MultiVariableImputerWithExternalEmbeddings(MultiVariableImputer):
    """Subclass that provides a convenient constructor for external/ground-truth embeddings.

    Use this when you wish to initialize the model with known embeddings and optionally
    freeze some or all of them while training the remaining modules.
    """

    @classmethod
    def from_true_embedding(
        cls,
        *,
        attribute_embedding=None,
        annotator_embedding=None,
        item_embedding=None,
        attribute_embedding_size: tuple | None = None,
        annotator_embedding_size: tuple | None = None,
        item_embedding_size: tuple | None = None,
        num_likert_classes: int,
        max_rank_size: int,
        encoder_layers_num: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
        freeze: bool | dict = False,
    ) -> "MultiVariableImputerWithExternalEmbeddings":
        # Build provider from external embeddings (supports partial + size hints)
        provider = OuterProductRankingEmbeddingProvider._from_true_embedding(
            attribute_embedding=attribute_embedding,
            annotator_embedding=annotator_embedding,
            item_embedding=item_embedding,
            attribute_embedding_size=attribute_embedding_size,
            annotator_embedding_size=annotator_embedding_size,
            item_embedding_size=item_embedding_size,
            num_likert_classes=num_likert_classes,
            max_rank_size=max_rank_size,
            freeze=freeze,
        )

        # Construct model using inferred sizes from the provider
        instance = cls(
            num_attributes=provider.num_attributes,
            num_annotators=provider.num_annotators,
            num_items=provider.num_items,
            num_likert_classes=num_likert_classes,
            max_rank_size=max_rank_size,
            encoder_layers_num=encoder_layers_num,
            attention_heads=attention_heads,
            embedding_dim=provider.embedding_dim,
            dropout=dropout,
        )
        # Inject the prepared provider
        instance.embedding_provider = provider
        return instance

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
    
    # Create training batch
    batch = converter.create_training_batch(rating_variables, ranking_variables,
                                          rating_data, ranking_data, test_data=test_data)
    
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
