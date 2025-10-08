# Ranking Imputer — Architecture Overview

This directory contains a modular, extensible implementation for ranking/rating imputation using transformer-based neural networks. The design emphasizes clean separation of concerns, structured data flow, and model-agnostic losses for mixed annotation types.

## Core Architecture

### Transformer-Based Imputation Model

The neural imputer uses a transformer encoder with mixed annotation heads to predict both ratings and rankings from partially observed data.

**Key Components**:
- **Input**: Mixed rating and ranking variables with conditional masking
- **Embedding**: Additive compositional embeddings (attribute + annotator + item(s))
- **Encoder**: 4-layer transformer with 8 attention heads (configurable)
- **Heads**: Rating classifier (C-way) + Ranking utilities (pairwise)
- **Training**: Progressive masking with configurable masking rates

## Core Concepts

### Data Structures (data.py)

**RankingData**: Structured variable unit with 0-indexed ids and optional targets
- Fields: `annotator_id`, `attribute_id`, `is_listwise`, `item_ids`, `rating_value?`, `ranking_order?`
- Conventions: rank 1 is best (ascending ranks)

**DataConverter**: Utilities to load JSON data and produce:
- Lists of variables (`rating_variables`, `ranking_variables`)
- Structured variables (`List[RankingData]`) with optional targets
- Legacy tensor batches (`create_batch`) with conditional masking support

### Embedding System (abstractions.py, embedding.py)

**RankingEmbeddingProviderBase**: Abstract base providing `forward(List[RankingData]) -> [B,N,D]`

Subclasses implement:
- `get_rating_embedding(attr_id, annot_id, item_id, rating_value)`
- `get_ranking_embedding(attr_id, annot_id, item_ids, ranking_order)`

**OuterProductRankingEmbeddingProvider**: Concrete provider using:
- **Rating**: `emb_attr + emb_annot + emb_item` (additive composition)
- **Ranking**: `emb_attr + emb_annot + pairwise_pool(emb_items)` (outer-product pooling)

**External Embedding Support**:
```python
provider = OuterProductRankingEmbeddingProvider._from_true_embedding(
    attribute_embedding=attr_emb,  # [I, D] or None
    annotator_embedding=annot_emb, # [J, D] or None  
    item_embedding=item_emb,       # [K, D] or None
    attribute_embedding_size=(I, D),  # Required if attr_emb is None
    annotator_embedding_size=(J, D),  # Required if annot_emb is None
    item_embedding_size=(K, D),       # Required if item_emb is None
    num_likert_classes=C,
    max_rank_size=R,
    freeze=False  # Optional freezing of embedding components
)
```

### Transformer Architecture (transformer.py)

**TransformerBlock**: Single-stream encoder block with:
- Pre-normalization
- Multi-head attention with optional attention mask `[B,N]` (True=valid)
- Feed-forward network with GELU activation
- Residual connections

**NormLayer**: Layer normalization helper
**FeedForward**: MLP with configurable dropout

### Model Architecture (ranking_imputer.py)

**MultiVariableImputer** (nn.Module): Main learnable model for mixed annotation imputation

Components:
- **Embedding Provider**: Converts structured variables to embeddings
- **Transformer Stack**: `ModuleList[TransformerBlock]` (configurable depth)
- **Prediction Heads**: `ModuleDict` with rating and ranking heads
  - `'rating'`: `Linear(D, C)` for categorical ratings
  - `'ranking'`: `Linear(D, R)` for ranking utilities

**Forward Pass**:
- Inputs: `List[RankingData]` or legacy tensors; optional `attn_mask`
- Returns: logits dict `{'rating': [B,N,C], 'ranking': [B,N,R]}`
- Optional: intermediate hidden states (`return_hidden=True`)

**External Embedding Constructor**:
```python
model = MultiVariableImputer.from_true_embedding(
    num_attributes=I, num_annotators=J, num_items=K,
    num_likert_classes=C, max_rank_size=R,
    encoder_layers_num=4, attention_heads=8, embedding_dim=D,
    attribute_embedding=attr_emb,  # Optional external embeddings
    annotator_embedding=annot_emb, 
    item_embedding=item_emb,
    freeze=False,  # Freeze external embeddings
    device='cpu'
)
```

### Loss Functions (losses.py)

**PlackettLuceLoss**: Ranking loss over masked positions
- Expects ranks where 1 is best (ascending order)
- Handles variable-length rankings with masking

**CrossEntropyLoss**: Standard rating classification loss

**LossStrategyBase / DefaultLossStrategy**:
- `compute(predictions, references, masked_flags) -> metrics`
- Converts structured lists to batched tensors
- Computes separate losses for observed and masked variables
- Returns detailed metrics: `rating_loss`, `ranking_loss`, `total_loss`

### Training System (trainer.py)

**ImputerTrainer**: Orchestrates training and evaluation

Key Features:
- **Mixed Loss Computation**: Combines rating and ranking losses
- **Embedding Regularization**: `embedding_anchor_reg` parameter to regularize toward initialization
- **Evaluation Modes**: Supports both conditional masking and pure imputation evaluation
- **Test Data Integration**: Can evaluate on separate test datasets

**Training Loop**:
```python
trainer = ImputerTrainer(model, learning_rate=1e-4, device='cpu')

# Single training step
losses = trainer.train_step(batch)
# Returns: {'total_loss': ..., 'rating_loss': ..., 'ranking_loss': ...}

# Evaluation with test data
test_metrics = trainer.evaluate_with_test_data(
    test_batch, test_data, converter, masking_rate=0.5
)
```

## Training Protocols

### Conditional Masking Training

**Purpose**: Train the model to predict masked variables from observed variables

**Process**:
1. Load training data with all annotations
2. Create batch with conditional masking (e.g., 50% masking rate)
3. Train model to predict masked variables from observed variables
4. Evaluate on masked test variables

**Implementation**:
```python
batch = converter.create_batch(
    rating_variables, ranking_variables, 
    rating_data, ranking_data, 
    mode="train", masking_rate=0.5
)
```

### Pure Imputation Evaluation

**Purpose**: Evaluate model's ability to predict ALL test variables without any observed test data

**Process**:
1. Train on ALL training data (no masking)
2. Test on ALL test data (100% masking - pure imputation)
3. Compare with domain model baseline

**Implementation**:
```python
test_batch = converter.create_batch(
    rating_variables, ranking_variables,
    test_rating_data, test_ranking_data,
    mode="test", masking_rate=1.0  # 100% masking for pure imputation
)
```

## Data & Rank Conventions

### Indexing
- **Model Internal**: `annotator_id`, `attribute_id`, `item_ids` are 0-indexed
- **JSON Data**: 1-indexed in input files, converted to 0-indexed internally

### Ranking Targets
- **Format**: `ranking_order` uses ascending ranks (1=best, 2=second, etc.)
- **Loss**: Plackett-Luce loss sorts in ascending order of ranks
- **Prediction**: Model outputs utilities, higher = better preference

### Attention Masks
- **Format**: `[B, N_max]` boolean tensor (True=valid, False=padded)
- **Usage**: When batching multiple variable lists, pad to `[B, N_max, D]`

## Configuration and Usage

### Model Configuration

**Standard Configuration**:
```python
from imputer import MultiVariableImputer, ImputerTrainer
from config import ExperimentConfig

config = ExperimentConfig()
model = MultiVariableImputer(
    num_attributes=config.I,
    num_annotators=config.J, 
    num_items=config.K,
    num_likert_classes=config.C,
    max_rank_size=config.ranking_size,
    encoder_layers_num=4,
    attention_heads=8,
    embedding_dim=64,
    dropout=0.1,
    embedding_type="pairwise",
    device="cpu"
)
```

### Training Script Usage

**Command Line**:
```bash
python run_experiment_imputer.py \
    --epochs 50 \
    --learning_rate 1e-4 \
    --masking_rate 0.5 \
    --encoder_layers 4 \
    --attention_heads 8 \
    --embedding_dim 64 \
    --save_plots
```

### Save/Load

**Saving**:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'train_losses': train_losses,
    'final_test_losses': test_losses
}, 'model.pth')
```

**Loading**:
```python
checkpoint = torch.load('model.pth')
model = MultiVariableImputer(**config_args)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Extensibility

### New Embeddings
Subclass `RankingEmbeddingProviderBase`:
```python
class CustomEmbeddingProvider(RankingEmbeddingProviderBase):
    def get_rating_embedding(self, attr_id, annot_id, item_id, rating_value):
        # Custom rating embedding logic
        pass
    
    def get_ranking_embedding(self, attr_id, annot_id, item_ids, ranking_order):
        # Custom ranking embedding logic  
        pass
```

### New Losses
Subclass `LossStrategyBase`:
```python
class CustomLossStrategy(LossStrategyBase):
    def compute(self, predictions, references, masked_flags):
        # Custom loss computation
        return metrics_dict
```

### New Heads or Multi-task
Add to `self.heads` in model and update trainer supervision:
```python
self.heads = nn.ModuleDict({
    'rating': nn.Linear(embedding_dim, num_likert_classes),
    'ranking': nn.Linear(embedding_dim, max_rank_size),
    'custom': nn.Linear(embedding_dim, custom_output_size)  # New head
})
```

## Performance and Metrics

### Training Metrics
- **Total Loss**: Combined rating + ranking loss
- **Rating Loss**: Cross-entropy loss on rating predictions
- **Ranking Loss**: Plackett-Luce loss on ranking predictions

### Evaluation Metrics
- **Test Rating Accuracy**: Fraction of correctly predicted ratings
- **Test Ranking Accuracy**: Fraction of correctly predicted pairwise preferences
- **Test Log-Loss**: Negative log-likelihood on test predictions

### Current Performance
- **Training**: Progressive masking enables learning from partial observations
- **Testing**: Pure imputation demonstrates generalization capability
- **Comparison**: Direct comparison with Bayesian domain model baseline

## Notes

### Legacy Support
- Legacy tensor batch path supported via `DataConverter.create_batch`
- Structured `List[RankingData]` preferred for new work
- Backward compatibility maintained for existing experiments

### Future Enhancements
- Structured batch processing (`[B,N,...]` batches directly)
- Extended loss strategies for multi-task learning
- Advanced attention mechanisms for annotation-specific modeling

---

*For usage examples and training scripts, see the parent directory's main README and training scripts.*