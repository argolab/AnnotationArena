# Complete Codebase Reference Documentation - Deep Dive

**Last Updated**: October 2025
**Purpose**: Comprehensive technical reference with implementation details
**Audience**: Self-reference for understanding every aspect of the codebase

---

## Executive Summary

This project compares **Bayesian (Stan) vs Neural (Transformer)** approaches for imputing missing ratings and rankings. The core insight is using **artificial masking** during training to create a self-supervised learning task that teaches the model to predict missing values.

**Key Innovation**: The transformer uses a **unified-stream architecture** where features and parameters flow through the same attention mechanism, with item embeddings randomized at each forward pass to prevent memorization.

---

## Part 1: Stan Pipeline - Data Generation

### 1.1 Core Data Structure: GroundTruthBundle

Location: `stan/pipeline/bundle.py`

```python
@dataclass
class GroundTruthBundle:
    # Ground truth parameters (K = K_train + K_test)
    embeddings: np.ndarray               # [K, D] item embeddings
    mean_preferences: np.ndarray         # [I, D] attribute preferences
    annotator_preferences: np.ndarray    # [I*J, D] annotator-attribute preferences
    rating_probs: np.ndarray            # [I*J, C] rating probabilities
    rating_cumprobs: np.ndarray         # [I*J, C] cumulative
    rating_thresholds_z: np.ndarray     # [I*J, C+1] threshold z-values
    base_scores: np.ndarray             # [I*J, K] dot products v_ij · e_k

    # Annotations (1-indexed in JSON)
    all_ratings: List[Dict]             # All ratings
    all_pairwise: List[Dict]            # All pairwise rankings

    # Partitions
    observed_ratings: List[Dict]        # Marked as observed
    missing_ratings: List[Dict]         # Marked as missing
    observed_pairwise: List[Dict]
    missing_pairwise: List[Dict]
    missing_ratings_indexes_in_test_instance: List[int]

    stats: Dict[str, Any]
```

**Critical Field**: Each rating/pairwise dict includes `"instance": "train"` or `"instance": "test"` to distinguish train vs test items.

**Rating dict schema**:
```python
{
    "attribute": 1,      # 1-indexed (i ∈ [1..I])
    "annotator": 1,      # 1-indexed (j ∈ [1..J])
    "item": 1,          # 1-indexed (k ∈ [1..K])
    "value": 3,         # 1-indexed (c ∈ [1..C])
    "instance": "train" # or "test"
}
```

**Pairwise dict schema**:
```python
{
    "attribute": 1,
    "annotator": 1,
    "items": [1, 2],         # Two items to compare
    "order": [1, 2],         # [1,2] means first item wins, [2,1] means second wins
    "tied_rating": 3,        # The rating bin where they tied
    "instance": "train"
}
```

### 1.2 Data Generation Process

Location: `stan/pipeline/data_gen.py`

**Step-by-step**:

1. **Compile Stan model**: `models/iclr_data_generation.stan`
2. **Sample ground truth**:
   - Item embeddings: `e_k ~ N(0, I_D)`
   - Mean preferences: `v_i ~ N(0, I_D)`
   - Annotator preferences: `v_ij ~ N(v_i, sigma_annotator^2 * I_D)`
   - Rating thresholds: Generated via Dirichlet process

3. **Generate observations**:
   ```
   For each (i, j, k):
       base_score = v_ij · e_k
       noisy_score = base_score + N(0, sigma_measurement^2)
       rating = bin(noisy_score, thresholds)
       observed = apply_observation_protocol()
   ```

4. **Generate pairwise rankings**:
   ```
   For tied ratings (two annotators give same rating to different items):
       score_diff = (v_ij · e_1) - (v_ij · e_2)
       P(item1 > item2) = sigmoid(score_diff / temperature)
       ranking = sample(P)
   ```

5. **Partition**:
   - Split into train (K_train items) vs test (K_test items)
   - Mark some as observed, some as missing

**Command**:
```bash
python stan/scripts/generate_data.py \
    --K-train 10 --K-test 10 \
    --I 10 --J 5 --D 64 --C 5 \
    --seed 42 \
    --output-dir OUTPUT/generated_data
```

**Output**: `OUTPUT/generated_data/run_YYYYMMDD_HHMMSS/data_bundle.json`

### 1.3 Stan Inference (Baseline)

Location: `stan/pipeline/inference.py`, `stan/scripts/run_inference.py`

**Purpose**: MCMC inference to learn posterior over embeddings/preferences for comparison baseline.

**Key function**: `prepare_stan_data_for_inference()`
- Filters observed ratings/pairwise by instance (train/test)
- Supports three modes:
  - `use_train_only=True`: Only train instance (inductive)
  - `use_test_only=True`: Only test instance
  - Default: Both train and test (transductive)

**Command**:
```bash
python stan/scripts/run_inference.py \
    --data-bundle OUTPUT/generated_data/run_*/data_bundle.json \
    --chains 8 --iter-warmup 1000 --iter-sampling 2000 \
    --init-strategy ground_truth  # or "random"
```

---

## Part 2: Imputer Architecture

### 2.1 RankingData: The Fundamental Unit

Location: `imputer/data.py`

```python
@dataclass
class RankingData:
    annotator_id: int          # 0-indexed
    attribute_id: int          # 0-indexed
    is_listwise: bool          # True=ranking, False=rating
    item_ids: List[int]        # 0-indexed
    status: int                # 0=missing, 1=masked, 2=observed
    instance: str              # "train" or "test"
    rating_value: Optional[int] = None      # 0-indexed (0-4 for C=5)
    ranking_order: Optional[List[int]] = None  # [1,2] means first wins

    @property
    def is_missing(self) -> bool:
        return self.status == 0

    @property
    def is_masked(self) -> bool:
        return self.status == 1

    @property
    def is_observed(self) -> bool:
        return self.status == 2
```

**Status encoding** (CRITICAL):
- `status=0` (missing): Never observed, target for pure imputation
- `status=1` (masked): Observed but artificially hidden during training
- `status=2` (observed): Available for conditioning

### 2.2 DataConverter

Location: `imputer/data.py`

**Key method**:
```python
def create_variables_from_bundle(
    self,
    bundle: GroundTruthBundle,
    partition: str,  # "train", "test", or "all"
    status: str      # "observed", "missing", or "all"
) -> List[RankingData]:
```

**Usage example**:
```python
converter = DataConverter(
    num_attributes=I,
    num_annotators=J,
    num_items=K_train + K_test,
    num_likert_classes=C,
    max_rank_size=2
)

bundle = converter.load_bundle_data("data_bundle.json")

# Get different partitions
train_observed = converter.create_variables_from_bundle(bundle, "train", "observed")
train_missing = converter.create_variables_from_bundle(bundle, "train", "missing")
test_observed = converter.create_variables_from_bundle(bundle, "test", "observed")
test_missing = converter.create_variables_from_bundle(bundle, "test", "missing")
```

**Important**: Converts 1-indexed JSON to 0-indexed Python automatically!

### 2.3 MultiVariableImputer Architecture

Location: `imputer/ranking_imputer.py`

```python
class MultiVariableImputer(nn.Module):
    def __init__(self,
                 num_attributes,
                 num_annotators,
                 num_items,
                 num_likert_classes,
                 max_rank_size,
                 encoder_layers_num=6,
                 attention_heads=8,
                 embedding_dim=128,
                 dropout=0.1,
                 embedding_type="atom",
                 randomness=True,
                 use_gelu_after_attention=False,
                 use_final_norm=True,        # CRITICAL for stability
                 normalize_parameter=False,
                 num_ffn_layers=4,
                 temperature=1.0):
```

**Architecture flow**:
```
Input: List[RankingData] (N variables)
    ↓
Embedding Provider
    - Creates features: [B, N, D] (semantic embeddings)
    - Creates params: [B, N, 1+C+R] (status bit + prediction logits)
    ↓
For each TransformerBlock:
    1. Concatenate: z = [features | params]
    2. Pre-LN: z_norm = LayerNorm(z)
    3. Attention: z = z + MHA(z_norm)
    4. Pre-LN: z_norm = LayerNorm(z)
    5. FFN: z = z + FFN(z_norm)
    6. Split back: features, params
    ↓
Final LayerNorm (if use_final_norm=True)  # Recommended!
    ↓
Read predictions from params:
    - Rating logits: params[:, :, 1:1+C]
    - Ranking logits: params[:, :, 1+C:1+C+R]
    ↓
Apply temperature scaling: logits / temperature
    ↓
Output: {'rating': [B, N, C], 'ranking': [B, N, R]}
```

**Critical architectural choice**: **Unified stream** - features and params flow through same attention!

### 2.4 Atom Compositional Embedding Provider

Location: `imputer/embedding.py`

The default embedding provider (`embedding_type="atom"`):

```python
class AtomCompositonalEmbeddingProvider(RankingEmbeddingProviderBase):
    def __init__(self, ..., randomness: bool):
        # Learnable embeddings
        self.attribute_embedding: [I, D-3]           # Fully learnable
        self.annotator_embedding_learnable: [J, (D-3)//2]  # Half learnable
        self.annotator_embedding_random: [J, (D-3)//2]     # Half random
        self.item_embedding: [K, D-3]                # Fully random (regenerated!)

        # Type indicators (not learnable)
        # attribute: [1, 0, 0]
        # annotator: [0, 1, 0]
        # item: [0, 0, 1]
```

**Key insight**: The `internal_dimension = embedding_dim - 3` to leave room for atom type indicators.

**For ratings**:
```python
def get_rating_embedding(self, attribute_id, annotator_id, item_id,
                         rating_value, is_missing):
    attr_vec = [1, 0, 0, self.attribute_embedding[attribute_id]]
    annot_vec = [0, 1, 0, learnable[annotator_id], random[annotator_id]]
    item_vec = [0, 0, 1, self.item_embedding[item_id]]

    # Features: sum of atom embeddings
    features = attr_vec + annot_vec + item_vec

    # Parameters: status bit + rating logits + ranking logits
    param = torch.zeros(1 + C + R)
    if is_missing:
        param[0] = 1.0  # Missing bit
        # Rest are zeros (uniform distribution)
    else:
        param[0] = 0.0  # Not missing
        param[rating_value + 1] = 20.0  # One-hot with high logit (LOGIT_HIGH=20.0)

    return torch.cat([features, param], dim=-1)
```

**For rankings**:
```python
def get_ranking_embedding(self, attribute_id, annotator_id, item_ids,
                          ranking_order, is_missing):
    attr_vec = [1, 0, 0, self.attribute_embedding[attribute_id]]
    annot_vec = [0, 1, 0, learnable[annotator_id], random[annotator_id]]

    # Combine items via pairwise relation
    item1_vec = [0, 0, 1, self.item_embedding[item_ids[0]]]
    item2_vec = [0, 0, 1, self.item_embedding[item_ids[1]]]
    item_combined = item1_vec + item2_vec @ self.pairwise_relation

    features = attr_vec + annot_vec + item_combined

    param = torch.zeros(1 + C + R)
    if is_missing:
        param[0] = 1.0
    else:
        param[0] = 0.0
        if ranking_order[0] < ranking_order[1]:  # First item wins
            param[1] = 20.0
            param[2] = 0.0
        else:  # Second item wins
            param[1] = 0.0
            param[2] = 20.0

    return torch.cat([features, param], dim=-1)
```

**Critical feature**: `on_forward_start()` regenerates item and half of annotator embeddings!

```python
def on_forward_start(self, variables):
    if self.randomness:
        self.partial_reset_embedding()

def partial_reset_embedding(self):
    # Regenerate random components at each forward pass
    self.annotator_embedding_random = torch.rand(J, (D-3)//2)
    self.item_embedding = torch.rand(K, D-3)
```

**Why randomness?**:
- Prevents memorization of specific items
- Forces model to learn attribute-annotator relationships
- Item embeddings become "slots" not tied to specific item identity

### 2.5 TransformerBlock

Location: `imputer/transformer.py`

```python
class TransformerBlock(nn.Module):
    def __init__(self, feature_dim, param_dim, attention_heads, dropout,
                 use_gelu_after_attention=False, normalize_parameter=False,
                 num_ffn_layers=4):
        self.feature_dim = feature_dim  # D
        self.param_dim = param_dim      # 1 + C + R
        self.total_dim = feature_dim + param_dim
        self.model_dim = ceil(total_dim / attention_heads) * attention_heads
```

**Forward pass**:
```python
def forward(self, feature_x, param_x, attn_mask=None):
    # Concatenate
    combined = torch.cat([feature_x, param_x], dim=-1)  # [B, N, total_dim]
    z = self.proj_in(combined)  # [B, N, model_dim]

    # Pre-LN + Attention + Residual
    if normalize_parameter:
        z_norm = self.norm_1(z)
    else:
        # Only normalize features, keep params unchanged
        z_norm = torch.cat([
            self.norm_1(z[:, :, :feature_dim]),
            z[:, :, feature_dim:]
        ], dim=-1)

    attn_out = self._multihead_attention(z_norm, attn_mask)

    if use_gelu_after_attention:
        attn_out = F.gelu(attn_out)

    z = z + self.dropout_1(attn_out)

    # Pre-LN + FFN + Residual
    z_ff_in = self.norm_2(z)
    z_ff = self.ff(z_ff_in)  # FeedForward with num_ffn_layers
    z = z + z_ff

    back = self.proj_out(z)
    combined = combined + self.dropout_2(back)

    # Split back
    feature_x = combined[:, :, :feature_dim]
    param_x = combined[:, :, feature_dim:]
    return feature_x, param_x
```

**Key innovation**: The model processes `[features | params]` as a unified stream, allowing cross-attention between semantic embeddings and prediction targets.

---

## Part 3: Training Process - The Magic

### 3.1 The Training Data Conversion (CRITICAL)

Location: `imputer/run_imputer.py:167-186`

**EXPERIMENTAL CHANGE** in run_imputer.py:
```python
# Convert all training missing to observed for fully observed training
train_missing_as_observed = []
for var in train_missing:
    train_missing_as_observed.append(RankingData(
        annotator_id=var.annotator_id,
        attribute_id=var.attribute_id,
        is_listwise=var.is_listwise,
        item_ids=var.item_ids,
        status=2,  # ← observed instead of missing!
        instance=var.instance,
        rating_value=var.rating_value,
        ranking_order=var.ranking_order,
    ))

# Use the converted missing data as additional observed training data
train_observed_full = train_observed + train_missing_as_observed
```

**Why?**: This makes ALL training annotations available as "observed", allowing us to mask 15% of EVERYTHING rather than just the originally observed subset.

### 3.2 Training Step

Location: `imputer/trainer.py`

```python
def train_step(self, train_observed_vars, train_missing_vars, masking_rate):
    # 1. Apply random masking to observed variables
    masked_or_observed = self._apply_training_mask(train_observed_vars, masking_rate)

    # 2. Build batch: [masked/observed] + [missing]
    batch_list = masked_or_observed + train_missing_vars

    # 3. Forward pass
    out = model(batch_list)

    # 4. Compute loss ONLY on masked + observed (not on missing)
    losses = self._compute_loss_for_batch(out, masked_or_observed)

    # 5. Backpropagation
    total_loss_tensor = losses['_total_loss_tensor']
    total_loss_tensor.backward()
    optimizer.step()

    return losses
```

**Critical distinction**:
- **Batch includes**: masked (status=1) + observed (status=2) + missing (status=0)
- **Loss computed on**: ONLY masked + observed
- **Missing included for**: Allowing model to see them, but no supervision

### 3.3 Masking Function

```python
def _apply_training_mask(self, observed_vars, masking_rate):
    num_to_mask = int(len(observed_vars) * masking_rate)
    masked_indices = random.sample(range(len(observed_vars)), num_to_mask)

    out = []
    for idx, var in enumerate(observed_vars):
        status = 1 if idx in masked_indices else 2  # 1=masked, 2=observed
        out.append(RankingData(
            ...var.fields...,
            status=status,
        ))
    return out
```

**Result**:
- M% of variables have status=1 (artificially masked)
- (1-M)% have status=2 (remain observed)
- Model must predict masked from observed context

### 3.4 Loss Computation

Location: `imputer/losses.py`

**DefaultLossStrategy**:
```python
class DefaultLossStrategy(LossStrategyBase):
    def __init__(self, masked_loss_weight=8.0, observed_loss_weight=1.0):
        self.masked_loss_weight = masked_loss_weight
        self.observed_loss_weight = observed_loss_weight
```

**Loss breakdown**:
```python
def compute(self, predictions, references):
    # Separate by status
    masked_rating_loss = mean(rating_losses where ref.is_masked)
    observed_rating_loss = mean(rating_losses where ref.is_observed)
    masked_ranking_loss = PL_loss(rankings where ref.is_masked)
    observed_ranking_loss = PL_loss(rankings where ref.is_observed)

    # Combine
    masked_total = masked_rating_loss + masked_ranking_loss
    observed_total = observed_rating_loss + observed_ranking_loss

    # Weighted total for backprop
    total_loss_tensor = (
        masked_loss_weight * masked_total +
        observed_loss_weight * observed_total
    )

    return {
        'total_loss': float(masked_total + observed_total),  # Unweighted for logging
        'masked_total_loss': float(masked_total),
        'observed_total_loss': float(observed_total),
        '_total_loss_tensor': total_loss_tensor  # Weighted for backprop
    }
```

**Why weight masked loss higher?**:
- Masked entries are the "hard" prediction task
- Observed entries are easier (model can condition on them)
- Default: masked_weight=8.0, observed_weight=1.0

**Optional loss decay**:
```python
if decay_observed_weight:
    # Linearly decay observed weight to 0 over first N epochs
    current_observed_weight = initial_weight * (1.0 - epoch / decay_epochs)
    loss_strategy.update_weights(masked_weight, current_observed_weight)
```

### 3.5 Training Loop

```python
def train(self, train_observed_vars, train_missing_vars, masking_rate, epochs,
          mask_augmentations=1, early_stopping=None, decay_observed_weight=False):
    for epoch in range(epochs):
        # Update loss weights if decay enabled
        if decay_observed_weight:
            update_loss_weights(epoch)

        # Multiple masking patterns per epoch (augmentation)
        epoch_losses = []
        for aug_idx in range(mask_augmentations):
            loss_dict = self.train_step(train_observed_vars, train_missing_vars, masking_rate)
            epoch_losses.append(loss_dict)

        # Average losses across augmentations
        loss_dict = average(epoch_losses)

        # Callbacks (evaluation without masking)
        if epoch % call_callbacks_every == 0:
            callback_results = self._call_epoch_end_callbacks(epoch)

            # Early stopping check
            if early_stopping:
                metric_value = callback_results['missing_metrics'][metric_name]
                if early_stopping.should_stop(metric_value, model):
                    early_stopping.restore_best_model(model)
                    break
```

**Mask augmentation**: Generate multiple random masking patterns per epoch for data augmentation.

---

## Part 4: Evaluation

### 4.1 Evaluation Engine

Location: `imputer/eval.py`

```python
class EvaluationEngine:
    def evaluate_model(self, model, variables, converter, device):
        model.eval()
        with torch.no_grad():
            # Forward pass (NO masking applied)
            model_output = model(variables)

            # Partition by status
            observed_idx = [i for i, v in enumerate(variables) if v.is_observed]
            missing_idx = [i for i, v in enumerate(variables) if v.is_missing]
            masked_idx = [i for i, v in enumerate(variables) if v.is_masked]

            # Compute metrics separately for each partition
            observed_metrics = compute_subset(observed_idx)
            missing_metrics = compute_subset(missing_idx)
            masked_metrics = compute_subset(masked_idx)

            return EvaluationResults(
                total_loss=...,
                observed_metrics=observed_metrics,
                missing_metrics=missing_metrics,    # ← Most important!
                masked_metrics=masked_metrics,
                ...
            )
```

**Key insight**: Evaluation uses variables **as-is** with no artificial masking. The `missing_metrics` show pure imputation performance!

### 4.2 Metrics

**Rating accuracy**:
```python
pred_class = argmax(rating_logits[0, i])  # 0-indexed (0-4)
accuracy = mean(pred_class == true_class)
```

**Rating RMSE**:
```python
pred_rating = pred_class + 1  # Convert to 1-5 scale
true_rating = true_class + 1
rmse = sqrt(mean((pred - true)^2))
```

**Ranking accuracy** (pairwise):
```python
scores = ranking_logits[0, i]
probs = softmax(scores[:2])
pred_first_wins = probs[0] > probs[1]
true_first_wins = ranking_order[0] < ranking_order[1]
accuracy = pred_first_wins == true_first_wins
```

### 4.3 Evaluation Callback

Location: `imputer/trainer.py`

```python
class EvaluationCallback:
    def on_epoch_end(self, model, epoch):
        results = self.eval_engine.evaluate_model(
            model=model,
            variables=self.test_variables,  # test_observed + test_missing
            converter=self.converter,
            device=self.device
        )
        return {
            'epoch': epoch,
            'name': self.name,
            'total_loss': results.total_loss,
            'rating_accuracy': results.rating_accuracy,
            'masked_metrics': results.masked_metrics,
            'observed_metrics': results.observed_metrics,
            'missing_metrics': results.missing_metrics,  # ← Key metric!
        }
```

**Usage**:
```python
# Register callback to evaluate on test set every epoch
trainer.register_callback(EvaluationCallback(
    eval_engine=eval_engine,
    test_variables=test_all,  # test_observed + test_missing
    name="test_all_evaluation"
))
```

---

## Part 5: Transductive Learning

### 5.1 What Is It?

**Inductive learning** (standard ML):
- Train on: training data only
- Test on: completely unseen test data

**Transductive learning**:
- Train on: training data + test data features (but not labels)
- Test on: test data labels

In this codebase:
- **Inductive**: `train_vars = train_observed`
- **Transductive**: `train_vars = train_observed + test_observed`

### 5.2 Implementation

Location: `imputer/run_imputer.py:254-256`

```python
train_vars = train_observed_full

if args.transductive_learning:
    print("Using transductive learning")
    train_vars += test_observed  # ← Add test observed to training!

trainer.train(
    train_observed_vars=train_vars,
    train_missing_vars=[],  # Empty (converted to observed earlier)
    masking_rate=args.masking_rate,
    epochs=args.epochs
)
```

### 5.3 Why Does It Help?

**Scenario**: Train items (1-10) are different from test items (11-20).

**Inductive**:
- Model only sees train items (1-10) during training
- Must predict test_missing for items (11-20) without ever seeing those items
- Hard generalization problem!

**Transductive**:
- Model sees train items (1-10) + test_observed annotations for items (11-20)
- Can learn embeddings for test items from partial observations
- Predicting test_missing becomes interpolation, not extrapolation

**Trade-off**: Transductive is "easier" but less realistic (assumes test data is available).

---

## Part 6: run_imputer.py Command-Line Arguments

### 6.1 Complete Argument List

Location: `imputer/run_imputer.py:86-128`

```bash
python imputer/run_imputer.py \
    # Required
    --data-dir OUTPUT/generated_data/run_YYYYMMDD_HHMMSS \

    # Output
    --output-root OUTPUT/IMPUTER \

    # Training hyperparameters
    --epochs 50 \
    --masking-rate 0.15 \
    --lr 1e-4 \
    --device cuda \

    # Data configuration
    --max-rank-size 2 \

    # Transductive learning
    --transductive_learning \

    # Model architecture
    --encoder-layers 6 \
    --attention-heads 8 \
    --embedding-dim 128 \
    --dropout 0.1 \
    --num_ffn_layers 4 \

    # Loss weighting
    --masked-loss-weight 8.0 \
    --observed-loss-weight 1.0 \
    --decay-observed-weight \
    --decay-observed-epochs 20 \

    # Architectural improvements
    --use-gelu-after-attention \
    --use-final-norm \          # Default True, recommended
    --no-final-norm \            # Disable (not recommended)
    --normalize-parameter \

    # Temperature scaling
    --temperature 1.5 \          # T > 1 softens, T < 1 sharpens

    # Augmentation
    --mask-augmentations 5 \     # Train with N masking patterns per epoch

    # Early stopping
    --early-stopping \
    --early-stopping-metric loss \    # or "accuracy"
    --early-stopping-patience 10 \
    --early-stopping-min-delta 1e-4 \

    # Checkpointing
    --save-checkpoints \
    --checkpoint-every 10 \

    # Experimental
    --full_random                # Use frozen random embeddings
```

### 6.2 Argument Details

**Masking rate**:
- `--masking-rate 0.15`: Artificially mask 15% of training observed data
- Higher values (0.3-0.5) create harder self-supervised task
- Lower values (0.05-0.15) provide more supervision

**Loss weighting**:
- `--masked-loss-weight 8.0`: Weight for masked entry loss
- `--observed-loss-weight 1.0`: Weight for observed entry loss
- Emphasizes learning to predict masked (unknown) entries

**Loss decay**:
- `--decay-observed-weight`: Enable linear decay of observed weight
- `--decay-observed-epochs 20`: Decay over first 20 epochs
- Gradually shifts focus from observed to masked prediction

**Temperature scaling**:
- `--temperature T`:
  - `T > 1`: Softer predictions (more uncertain, better calibration)
  - `T < 1`: Sharper predictions (more confident)
  - `T = 1`: No scaling (default)

**Final LayerNorm**:
- `--use-final-norm`: Apply final LayerNorm after all blocks (DEFAULT, RECOMMENDED)
- `--no-final-norm`: Disable final LayerNorm (can cause instability)
- **Critical for Pre-LN transformer stability!**

**Mask augmentation**:
- `--mask-augmentations N`: Generate N different masking patterns per epoch
- Creates data augmentation by varying which entries are masked
- Increases training time proportionally

---

## Part 7: Key Insights and Gotchas

### 7.1 The Three Status Codes

| Code | Name | Meaning | Training Loss? | Eval Metrics? |
|------|------|---------|----------------|---------------|
| 0 | missing | Never observed (protocol) | ❌ NO | ✅ YES (missing_metrics) |
| 1 | masked | Artificially hidden | ✅ YES (self-supervised) | ❌ Usually empty |
| 2 | observed | Available for conditioning | ✅ YES (supervised) | ✅ YES (observed_metrics) |

**Critical distinction**:
- **Protocol-driven missing** (status=0): Truly unobserved in the data generation process
- **Artificially masked** (status=1): We hide them for training, but they were originally observed

### 7.2 Index Conversion

| Context | Convention | Example |
|---------|-----------|---------|
| JSON data | 1-indexed | `"item": 1` (first item) |
| Python/Model | 0-indexed | `item_ids=[0]` (first item) |
| DataConverter | Converts automatically | Handles 1→0 conversion |

### 7.3 Training vs Evaluation Masking

**CRITICAL**: Masking behavior is different!

**Training**:
```python
# Apply artificial masking
masked_or_observed = _apply_training_mask(train_observed, masking_rate)
batch = masked_or_observed + train_missing
loss = compute_loss(batch, supervised=masked_or_observed)  # Only on non-missing
```

**Evaluation**:
```python
# NO masking applied
batch = test_observed + test_missing  # Variables used as-is
results = evaluate(batch)
# Metrics computed separately for observed, missing, masked
```

### 7.4 Experimental Training Data Conversion

**WATCH OUT**: In `run_imputer.py:167-186`, there's code that converts ALL `train_missing` to `status=2` (observed):

```python
train_missing_as_observed = [
    RankingData(..., status=2, ...)
    for var in train_missing
]
train_observed_full = train_observed + train_missing_as_observed
```

**Effect**: ALL training data becomes "observed", allowing masking of entire training set rather than just originally observed subset.

**Then in training**:
```python
trainer.train(
    train_observed_vars=train_observed_full,  # ALL training data
    train_missing_vars=[],                    # Empty!
    masking_rate=0.15,
    ...
)
```

### 7.5 Item Embedding Randomization

**Key feature** in AtomCompositionalEmbeddingProvider:

```python
def on_forward_start(self, variables):
    if self.randomness:
        self.partial_reset_embedding()

def partial_reset_embedding(self):
    # Regenerate at EVERY forward pass!
    self.annotator_embedding_random = torch.rand(J, (D-3)//2)
    self.item_embedding = torch.rand(K, D-3)
```

**Why?**:
- Prevents model from memorizing specific item identities
- Forces learning of attribute-annotator relationships
- Item embeddings become "slots" filled randomly each time

### 7.6 Final LayerNorm is Critical

**Pre-LN Transformer architecture** requires final normalization:

```python
for block in self.blocks:
    features, params = block(features, params)

# CRITICAL for stability!
if self.final_norm is not None:
    params = self.final_norm(params)
```

**Why?**: Pre-LN means the last block's output is NOT normalized. Without final norm, gradients can explode/vanish.

---

## Part 8: Research Questions to Explore

### 8.1 Comparison Studies

1. **Bayesian vs Neural**: Which performs better on `missing_metrics`?
2. **Inductive vs Transductive**: How much does test_observed help?
3. **Masking rate**: Optimal value (0.15 vs 0.3 vs 0.5)?
4. **Architecture depth**: 4 vs 6 vs 8 transformer layers?
5. **Loss weighting**: Effect of masked_loss_weight (4.0 vs 8.0 vs 16.0)?

### 8.2 Ablation Studies

1. **Item randomization**: Effect of randomizing vs learning item embeddings
2. **Temperature scaling**: Effect on calibration and accuracy
3. **Mask augmentation**: Effect of multiple masking patterns per epoch
4. **Loss decay**: Effect of decaying observed loss weight
5. **Final LayerNorm**: Critical for stability? (yes!)

### 8.3 Generalization Studies

1. **Unseen items**: Performance on items not in training set
2. **Unseen annotators**: Can model predict for new annotators?
3. **Domain shift**: What if test distribution differs from train?

---

## Part 9: Typical Workflow

### 9.1 Generate Data

```bash
cd imputer/ranking
export PYTHONPATH=.

python stan/scripts/generate_data.py \
    --K-train 10 --K-test 10 \
    --I 10 --J 5 --D 64 --C 5 \
    --seed 42 \
    --output-dir OUTPUT/generated_data
```

### 9.2 Run Stan Baseline (Transductive)

```bash
python stan/scripts/run_inference.py \
    --data-bundle OUTPUT/generated_data/run_*/data_bundle.json \
    --chains 8 --iter-warmup 1000 --iter-sampling 2000 \
    --init-strategy ground_truth
```

### 9.3 Train Neural Imputer (Inductive)

```bash
python imputer/run_imputer.py \
    --data-dir OUTPUT/generated_data/run_YYYYMMDD_HHMMSS \
    --epochs 50 --masking-rate 0.15 --lr 1e-4 \
    --encoder-layers 6 --attention-heads 8 --embedding-dim 128 \
    --use-final-norm \
    --device cuda
```

### 9.4 Train Neural Imputer (Transductive)

```bash
python imputer/run_imputer.py \
    --data-dir OUTPUT/generated_data/run_YYYYMMDD_HHMMSS \
    --transductive_learning \
    --epochs 50 --masking-rate 0.15 --lr 1e-4 \
    --encoder-layers 6 --attention-heads 8 --embedding-dim 128 \
    --use-final-norm \
    --device cuda
```

### 9.5 Compare Results

**Stan results**: `OUTPUT/domain_model/runs/run_*/`
- `domain_model-*.csv`: MCMC samples
- `predictive_metrics.json`: Evaluation metrics

**Imputer results**: `OUTPUT/IMPUTER/run_*/`
- `model.pt`: Trained model
- `test_metrics.json`: Evaluation metrics
- `predictives.json`: Predictions
- `train_config.json`: Configuration
- `test_training_history.json`: Training history

**Key metrics to compare**:
- `missing_metrics['rating_accuracy']`: Pure imputation accuracy
- `missing_metrics['rating_rmse']`: Average error on imputed ratings
- `missing_metrics['ranking_accuracy']`: Pairwise ranking accuracy

---

## Part 10: File Reference

### Stan Pipeline

| File | Purpose |
|------|---------|
| `stan/pipeline/bundle.py` | GroundTruthBundle definition |
| `stan/pipeline/configs.py` | Configuration dataclasses |
| `stan/pipeline/data_gen.py` | Data generation wrapper |
| `stan/pipeline/inference.py` | MCMC inference |
| `stan/pipeline/predictives.py` | Posterior predictive evaluation |
| `stan/pipeline/io.py` | File I/O utilities |
| `stan/scripts/generate_data.py` | CLI for data generation |
| `stan/scripts/run_inference.py` | CLI for MCMC inference |
| `models/iclr_data_generation.stan` | Stan data generation model |
| `models/domain_model.stan` | Stan inference model |

### Imputer

| File | Purpose |
|------|---------|
| `imputer/run_imputer.py` | **Main training script** |
| `imputer/ranking_imputer.py` | Model architecture |
| `imputer/transformer.py` | Transformer blocks |
| `imputer/embedding.py` | Embedding providers |
| `imputer/trainer.py` | Training loop |
| `imputer/losses.py` | Loss functions |
| `imputer/eval.py` | Evaluation engine |
| `imputer/data.py` | Data structures (RankingData, DataConverter) |
| `imputer/abstractions.py` | Abstract base classes |

---

## Summary: The Big Picture

**Problem**: Predict missing ratings and rankings from partial observations.

**Bayesian approach** (Stan):
- Sample from posterior over embeddings/preferences
- Principled uncertainty quantification
- Slow, doesn't scale

**Neural approach** (Transformer):
- Learn to predict masked entries from observed context
- Fast inference after training
- Scales to large datasets

**Key innovation**:
1. **Unified-stream transformer**: Features and parameters flow through same attention
2. **Artificial masking**: Self-supervised learning by hiding known values
3. **Item randomization**: Prevent memorization, force relational learning
4. **Transductive option**: Leverage partial test observations

**Critical implementation details**:
- Status codes (0=missing, 1=masked, 2=observed)
- Training converts train_missing to observed for masking
- Loss computed only on non-missing (observed + masked)
- Evaluation has no masking, computes metrics separately by status
- Final LayerNorm is critical for Pre-LN transformer stability

**Bottom line**: The transformer learns attribute-annotator relationships from partial observations and uses them to predict missing annotations for test items, with optional transductive learning to leverage partial test observations.

---

**END OF COMPREHENSIVE REFERENCE**

*Use this document to understand every aspect of the codebase. Think harder when needed!*
