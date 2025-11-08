# Comprehensive Codebase Reference Documentation

**Generated**: 2025-10-06
**Purpose**: Complete reference for the ranking imputation system combining Stan-based data generation with transformer-based neural imputation.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Stan Pipeline: Data Generation and Domain Model](#stan-pipeline)
3. [Imputer: Transformer-Based Neural Imputation](#imputer)
4. [run_imputer.py: Main Training Script](#run_imputer)
5. [Training Process: Masking Strategy](#training-process)
6. [Evaluation and Metrics](#evaluation)
7. [Transductive Learning](#transductive-learning)
8. [Data Flow Diagram](#data-flow)
9. [Key Concepts and Terminology](#key-concepts)

---

## 1. High-Level Overview {#high-level-overview}

This codebase implements a **ranking/rating imputation system** that combines:

1. **Stan-based data generation**: Generates synthetic datasets with known ground truth
2. **Transformer-based neural imputation**: Learns to predict missing ratings and rankings

### The Task

**Input**: Partially observed ratings and pairwise rankings from multiple annotators rating items on multiple attributes.

**Goal**: Predict missing ratings and rankings using a transformer model that learns relationships between annotators, attributes, and items.

**Data Structure**:
- **Ratings**: Annotator `j` rates item `k` on attribute `i` with a Likert scale value (1-5)
- **Rankings**: Annotator `j` ranks items on attribute `i` (pairwise comparisons)

### Key Innovation

The system uses **artificial masking** during training to create a self-supervised learning task:
- Take observed training data
- **Mask** M% randomly (status=1, artificially hidden)
- **Keep** (1-M)% observed (status=2, available for training)
- **Append** truly missing data (status=0, prediction targets)
- **Train** to predict masked + observed entries
- **Evaluate** on test set missing entries

---

## 2. Stan Pipeline: Data Generation and Domain Model {#stan-pipeline}

### 2.1 Purpose

The Stan pipeline generates synthetic datasets with known ground truth for:
1. Training and evaluating the neural imputer
2. Comparing neural imputer against Bayesian domain model baseline

### 2.2 Core Components

#### Location: `stan/pipeline/`

**Key Files**:
- `configs.py`: Configuration dataclasses
- `bundle.py`: `GroundTruthBundle` data structure
- `data_gen.py`: Stan-based data generation
- `inference.py`: MCMC inference for domain model
- `predictives.py`: Posterior predictive evaluation
- `io.py`: File I/O utilities

### 2.3 Data Generation Config

```python
@dataclass
class DataGenConfig:
    K_train: int          # Number of training items
    K_test: int           # Number of test items
    I: int                # Number of attributes
    J: int                # Number of annotators
    D: int                # Embedding dimension
    C: int                # Number of Likert classes (1-5)

    enable_pairwise_rankings: bool = True
    pairwise_cap_per_item: int = 10

    # Noise parameters
    sigma_annotator: float = 0.3      # Annotator preference noise
    sigma_measurement: float = 0.1    # Measurement noise
    alpha_dirichlet: float = 2.0      # Dirichlet concentration
    temperature: float = 0.5          # Pairwise ranking temperature

    seed: Optional[int] = None
```

### 2.4 GroundTruthBundle Structure

The fundamental data container used throughout the system:

```python
@dataclass
class GroundTruthBundle:
    # Ground truth parameters (K = K_train + K_test)
    embeddings: np.ndarray               # [K, D] - item embeddings
    mean_preferences: np.ndarray         # [I, D] - attribute preferences
    annotator_preferences: np.ndarray    # [I*J, D] - annotator preferences
    rating_probs: np.ndarray            # [I*J, C] - rating probabilities
    rating_cumprobs: np.ndarray         # [I*J, C] - cumulative probs
    rating_thresholds_z: np.ndarray     # [I*J, C+1] - z-cutpoints
    base_scores: np.ndarray             # [I*J, K] - base utility scores

    # All data (observed + missing)
    all_ratings: List[Dict]             # All rating observations
    all_pairwise: List[Dict]            # All pairwise rankings

    # Partitioned by observed/missing
    observed_ratings: List[Dict]
    missing_ratings: List[Dict]
    observed_pairwise: List[Dict]
    missing_pairwise: List[Dict]

    # Each rating dict: {'attribute': int(1..I), 'annotator': int(1..J),
    #                    'item': int(1..K), 'value': int(1..C), 'instance': "train"/"test"}

    # Each pairwise dict: {'attribute': int, 'annotator': int,
    #                      'items': [int, int], 'order': [1,2] or [2,1],
    #                      'instance': "train"/"test"}

    # Statistics
    stats: Dict[str, Any]

    # Optional posterior predictions
    train_posterior_rating_probs: Optional[np.ndarray] = None  # [I*J, K_train, C]
    test_posterior_rating_probs: Optional[np.ndarray] = None   # [I*J, K_test, C]
```

**Important Convention**: All indices in JSON data are **1-indexed**, but converted to **0-indexed** when loaded into the imputer.

### 2.5 Data Generation Process

**Script**: `stan/scripts/generate_data.py`

```bash
python stan/scripts/generate_data.py \
    --K-train 10 --K-test 10 \
    --I 3 --J 9 \
    --D 64 --C 5 \
    --output-dir OUTPUT/data \
    --seed 42
```

**What happens**:
1. Compiles `models/iclr_data_generation.stan`
2. Generates ground truth embeddings and preferences
3. Simulates ratings and pairwise rankings with noise
4. Splits data into observed/missing, train/test
5. Saves `data_bundle.json` and `configs.json`

**Output Structure**:
```
OUTPUT/data/run_YYYYMMDD_HHMMSS/
├── configs.json           # Generation config
└── data_bundle.json       # Complete GroundTruthBundle
```

### 2.6 Key Stan Models

#### `models/iclr_data_generation.stan`

Generative model:
1. **Sample embeddings**: Item embeddings from N(0, 1)
2. **Sample preferences**: Attribute and annotator preferences
3. **Compute base scores**: dot products of preferences and embeddings
4. **Generate ratings**: Ordinal regression with noise
5. **Generate pairwise rankings**: Bradley-Terry model on tied ratings
6. **Apply observation protocol**: Randomly hide some entries

#### `models/domain_model.stan`

Bayesian inference model for comparison baseline:
- Infers embeddings and preferences from observed data
- Generates posterior predictives for missing entries
- Used to benchmark neural imputer performance

### 2.7 Inference Pipeline

**Script**: `stan/scripts/run_inference.py`

```bash
python stan/scripts/run_inference.py \
    --data-bundle OUTPUT/data/.../data_bundle.json \
    --output-dir OUTPUT/inference \
    --chains 8 \
    --iter-sampling 2000
```

Runs MCMC to infer posteriors and evaluate predictions on missing data.

---

## 3. Imputer: Transformer-Based Neural Imputation {#imputer}

### 3.1 Architecture Overview

The imputer is a **vanilla transformer encoder** with custom embedding and prediction heads for mixed rating/ranking tasks.

**Location**: `imputer/`

**Key Components**:
1. **Embedding Provider**: Converts structured variables to embeddings
2. **Transformer Blocks**: Multi-head attention with residual connections
3. **Prediction via Parameters**: Rating and ranking logits read from parameter stream

### 3.2 Data Structures

#### RankingData

The fundamental unit representing a single rating or ranking observation:

```python
@dataclass
class RankingData:
    annotator_id: int          # 0-indexed
    attribute_id: int          # 0-indexed
    is_listwise: bool          # True=ranking, False=rating
    item_ids: List[int]        # 0-indexed item indices
    status: int                # 0=missing, 1=masked, 2=observed
    instance: str              # "train" or "test"
    rating_value: Optional[int] = None      # 0-indexed (0-4 for C=5)
    ranking_order: Optional[List[int]] = None  # [1,2] means first item wins

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

**Status Encoding**:
- `0 = missing`: Not observed, target for prediction
- `1 = masked`: Observed but artificially hidden during training
- `2 = observed`: Available for training

#### DataConverter

Converts `GroundTruthBundle` to `List[RankingData]`:

```python
converter = DataConverter(
    num_attributes=I,
    num_annotators=J,
    num_items=K_train + K_test,
    num_likert_classes=C,
    max_rank_size=2
)

# Load bundle
bundle = converter.load_bundle_data("data_bundle.json")

# Create variables by partition and status
train_observed = converter.create_variables_from_bundle(bundle, "train", "observed")
train_missing = converter.create_variables_from_bundle(bundle, "train", "missing")
test_observed = converter.create_variables_from_bundle(bundle, "test", "observed")
test_missing = converter.create_variables_from_bundle(bundle, "test", "missing")
```

**Important**: Conversion handles 1-indexed → 0-indexed automatically.

### 3.3 Model Architecture

#### MultiVariableImputer

**File**: `imputer/ranking_imputer.py`

```python
model = MultiVariableImputer(
    num_attributes=I,
    num_annotators=J,
    num_items=K,
    num_likert_classes=C,
    max_rank_size=2,
    encoder_layers_num=6,      # Number of transformer blocks
    attention_heads=8,
    embedding_dim=128,
    dropout=0.1,
    embedding_type="atom",     # Options: "atom", "pairwise", "fully_random"
    device="cuda",
    randomness=False           # If True, freezes embeddings
)
```

**Architecture**:

```
Input: List[RankingData]
    ↓
[Embedding Provider]
    - Computes features (D-dim vectors)
    - Computes parameters (C+R+1 dim: status bit + rating logits + ranking logits)
    ↓
[Transformer Blocks] × encoder_layers_num
    - Multi-head attention on [features | params]
    - Feed-forward network
    - Pre-normalization + residual connections
    ↓
[Read Predictions from Parameters]
    - Rating logits: params[:, 1:1+C]
    - Ranking logits: params[:, 1+C:1+C+R]
    ↓
Output: {'rating': [B, N, C], 'ranking': [B, N, R]}
```

#### Transformer Block

**File**: `imputer/transformer.py`

```python
class TransformerBlock(nn.Module):
    """Unified-stream transformer over concatenated [features | params]."""

    def __init__(self, feature_dim: int, param_dim: int,
                 attention_heads: int, dropout: float):
        # feature_dim = D (embedding dimension)
        # param_dim = 1 + C + R (status + rating + ranking)
        # total_dim = D + (1 + C + R)
```

**Forward pass**:
1. Concatenate `[features, params]` → shape `[B, N, D + param_dim]`
2. Project to attention space if needed (ensure divisible by num_heads)
3. **Pre-norm** → Multi-head attention → **Residual**
4. **Pre-norm** → Feed-forward → **Residual**
5. Split back to `features` and `params` streams

**Key Innovation**: Unified stream allows features and parameters to interact through attention.

#### Embedding Providers

**File**: `imputer/embedding.py`

Several embedding strategies are available:

1. **AtomCompositionalEmbeddingProvider** (default, `embedding_type="atom"`):
   - **Rating**: `attr_emb + annot_emb + item_emb`
   - **Ranking**: `attr_emb + annot_emb + mean(item_embs)`
   - Learnable embeddings with additive composition

2. **PairwiseRankingProjectionEmbeddingProvider** (`embedding_type="pairwise"`):
   - Uses outer product pooling for pairwise rankings

3. **FullyRandomizedEmbeddingProvider** (`embedding_type="fully_random"`):
   - Fixed random embeddings (non-trainable)

All providers inherit from `RankingEmbeddingProviderBase` and return:
- `features`: `[B, N, D]` - semantic embeddings
- `params`: `[B, N, 1+C+R]` - status bit + prediction logits

---

## 4. run_imputer.py: Main Training Script {#run_imputer}

### 4.1 Command-Line Arguments

**File**: `imputer/run_imputer.py`

```bash
python imputer/run_imputer.py \
    --data-dir OUTPUT/data/run_YYYYMMDD_HHMMSS \
    --output-root OUTPUT/IMPUTER \
    --epochs 50 \
    --masking-rate 0.15 \
    --lr 1e-3 \
    --device cuda \
    --max-rank-size 2 \
    --transductive_learning \
    --save-checkpoints \
    --checkpoint-every 10
```

**Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | str | **required** | Directory with `data_bundle.json` and `configs.json` |
| `--output-root` | str | `OUTPUT/IMPUTER` | Root directory for saving results |
| `--epochs` | int | `50` | Number of training epochs |
| `--masking-rate` | float | `0.15` | Fraction of training observed data to mask (M%) |
| `--lr` | float | `1e-3` | Learning rate for Adam optimizer |
| `--device` | str | `cuda` | Device for training (`cuda` or `cpu`) |
| `--max-rank-size` | int | `2` | Maximum pairwise ranking size |
| `--transductive_learning` | flag | `False` | If set, includes test_observed in training |
| `--full_random` | flag | `False` | If set, uses frozen random embeddings |
| `--save-checkpoints` | flag | `False` | Enable checkpoint saving |
| `--checkpoint-every` | int | `10` | Save checkpoint every N epochs |

### 4.2 What the Script Does

1. **Load Data**:
   ```python
   bundle = GroundTruthBundle.from_dict(bundle_dict)
   ```

2. **Create Variables**:
   ```python
   train_observed = converter.create_variables_from_bundle(bundle, "train", "observed")
   train_missing = converter.create_variables_from_bundle(bundle, "train", "missing")
   test_observed = converter.create_variables_from_bundle(bundle, "test", "observed")
   test_missing = converter.create_variables_from_bundle(bundle, "test", "missing")
   ```

3. **Initialize Model**:
   ```python
   model = MultiVariableImputer(
       num_attributes=I, num_annotators=J, num_items=K,
       num_likert_classes=C, max_rank_size=2,
       encoder_layers_num=6, attention_heads=8, embedding_dim=128,
       device=device
   )
   ```

4. **Setup Trainer**:
   ```python
   trainer = ImputerTrainer(
       model=model, learning_rate=lr, device=device
   )
   ```

5. **Register Evaluation Callbacks**:
   ```python
   trainer.register_callback(
       EvaluationCallback(
           eval_engine=eval_engine,
           test_variables=test_all,  # test_observed + test_missing
           converter=converter,
           device=device,
           name="test_all_evaluation"
       )
   )
   ```

6. **Train**:
   ```python
   trainer.train(
       train_observed_vars=train_vars,  # train_observed (or + test_observed if transductive)
       train_missing_vars=train_missing,
       masking_rate=masking_rate,
       epochs=epochs
   )
   ```

7. **Evaluate**:
   ```python
   results = eval_engine.evaluate_model(
       model=model,
       variables=test_all,
       converter=converter,
       device=device
   )
   ```

8. **Save Outputs**:
   - `model.pt`: Trained model checkpoint
   - `test_metrics.json`: Evaluation metrics
   - `predictives.json`: Predictions on test set
   - `train_config.json`: Full training configuration

### 4.3 Model Architecture Details

The model is initialized with:

```python
model_config = {
    'num_attributes': I,
    'num_annotators': J,
    'num_items': K,
    'num_likert_classes': C,
    'max_rank_size': 2,
    'encoder_layers_num': 6,      # Fixed in run_imputer.py:139
    'attention_heads': 8,          # Fixed in run_imputer.py:140
    'embedding_dim': 128,          # Fixed in run_imputer.py:141
    'dropout': 0.1,
    'embedding_type': 'atom',
    'device': device
}
```

**Note**: These hyperparameters are currently hardcoded in `run_imputer.py` and not exposed as CLI arguments.

---

## 5. Training Process: Masking Strategy {#training-process}

### 5.1 The Masking Strategy

The key innovation is **artificial masking** for self-supervised learning:

**Training Input**:
- `train_observed_vars`: Observed training data (status=2)
- `train_missing_vars`: Missing training data (status=0)
- `masking_rate`: Fraction M% to artificially mask

**Training Step** (`trainer.py:ImputerTrainer.train_step`):

```python
def train_step(self, train_observed_vars, train_missing_vars, masking_rate):
    # 1. Apply artificial masking
    masked_or_observed = self._apply_training_mask(train_observed_vars, masking_rate)
    # Randomly selects M% of observed vars and sets status=1 (masked)
    # Remaining (1-M)% keep status=2 (observed)

    # 2. Build batch: [masked/observed] + [missing]
    batch_list = masked_or_observed + train_missing_vars

    # 3. Forward pass
    out = model(batch_list)

    # 4. Compute loss ONLY on masked + observed (NOT on missing)
    loss = compute_loss(out, masked_or_observed)

    # 5. Backprop
    loss.backward()
    optimizer.step()
```

**Critical Distinction**:

| Status | Description | Training Loss | Evaluation Loss |
|--------|-------------|---------------|-----------------|
| `status=0` (missing) | Truly unobserved | **NO** | Only for missing metrics |
| `status=1` (masked) | Artificially hidden | **YES** (self-supervised) | N/A (not used in eval) |
| `status=2` (observed) | Available supervision | **YES** | YES |

### 5.2 Why This Works

1. **Self-Supervised Learning**: Model learns to reconstruct masked entries from context
2. **Generalization**: Forces model to use relationships between variables, not memorization
3. **Calibration**: Observed entries provide supervised signal alongside masked reconstruction

### 5.3 Training Loop

**File**: `trainer.py:ImputerTrainer.train`

```python
for epoch in range(epochs):
    # Training step with masking
    loss_dict = trainer.train_step(train_observed_vars, train_missing_vars, masking_rate)

    # Callback evaluation (no masking during eval)
    if (epoch + 1) % call_callbacks_every == 0:
        callback_results = trainer._call_epoch_end_callbacks(epoch)
        # Evaluates on test_all without any masking
```

### 5.4 Difference: Training vs Evaluation

**Training**:
- **Input**: `train_observed` (with M% masked → status=1) + `train_missing` (status=0)
- **Loss computed on**: masked (status=1) + observed (status=2) entries
- **Purpose**: Learn to predict masked from observed context

**Evaluation**:
- **Input**: All test variables as-is (observed + missing)
- **No masking applied**
- **Metrics computed separately for**:
  - `observed_metrics`: status=2
  - `missing_metrics`: status=0
  - `masked_metrics`: status=1 (usually empty in eval)

---

## 6. Evaluation and Metrics {#evaluation}

### 6.1 Evaluation Engine

**File**: `imputer/eval.py:EvaluationEngine`

```python
results = eval_engine.evaluate_model(
    model=model,
    variables=test_all,  # test_observed + test_missing
    converter=converter,
    device=device
)
```

**Output**: `EvaluationResults` dataclass

```python
@dataclass
class EvaluationResults:
    # Overall metrics
    total_loss: float
    rating_loss: float
    ranking_loss: float
    rating_accuracy: float
    rating_rmse: float
    ranking_accuracy: float
    num_rating_evaluations: int
    num_ranking_evaluations: int

    # Breakdown by status
    observed_metrics: Dict[str, Any]   # Metrics on status=2
    missing_metrics: Dict[str, Any]    # Metrics on status=0
    masked_metrics: Dict[str, Any]     # Metrics on status=1 (usually empty)
```

### 6.2 Loss Functions

**File**: `imputer/losses.py`

#### Rating Loss: Cross-Entropy

```python
rating_loss_fn = nn.CrossEntropyLoss(reduction='none')
```

Computed per rating variable:
- **Input**: `[N, C]` logits
- **Target**: One-hot encoded rating class
- **Output**: Cross-entropy loss per variable

#### Ranking Loss: Plackett-Luce

```python
class PlackettLuceLoss(nn.Module):
    """Plackett-Luce ranking loss."""

    def forward(self, logits, targets, mask):
        # logits: [B, V, K] - predicted utilities
        # targets: [B, V, K] - rank positions (1=best, 2=second, etc.)
        # mask: [B, V] - which rankings to compute loss on

        # For each ranking:
        #   1. Sort items by target rank (ascending)
        #   2. Compute PL probability for observed order
        #   3. Return negative log-probability
```

**Plackett-Luce Model**:
- Probability of ranking item `i` at position `k`:
  ```
  p_k = exp(score_i) / sum_j exp(score_j for j in remaining items)
  ```
- Loss: `-sum_k log(p_k)`

### 6.3 Weighted Loss Strategy

**File**: `losses.py:DefaultLossStrategy`

```python
strategy = DefaultLossStrategy(
    masked_loss_weight=8.0,    # Weight for masked entries
    observed_loss_weight=1.0   # Weight for observed entries
)
```

**Loss Breakdown**:

```python
# Separate losses by status
masked_rating_loss = mean(rating_losses where status=1)
observed_rating_loss = mean(rating_losses where status=2)
masked_ranking_loss = PL_loss(rankings where status=1)
observed_ranking_loss = PL_loss(rankings where status=2)

# Combined
masked_total = masked_rating_loss + masked_ranking_loss
observed_total = observed_rating_loss + observed_ranking_loss

# Weighted combination for backprop
total_loss = (masked_loss_weight * masked_total +
              observed_loss_weight * observed_total)
```

### 6.4 Metrics Computation

**Rating Accuracy**:
```python
pred_class = argmax(rating_logits)
accuracy = mean(pred_class == true_class)
```

**Rating RMSE** (on 1-5 scale):
```python
pred_rating = pred_class + 1  # Convert 0-4 to 1-5
true_rating = true_class + 1
rmse = sqrt(mean((pred_rating - true_rating)^2))
```

**Ranking Accuracy** (pairwise):
```python
# For pairwise rankings [items=[k1, k2], order=[1,2] or [2,1]]
pred_first_wins = softmax(logits)[0] > softmax(logits)[1]
true_first_wins = (order[0] < order[1])
accuracy = mean(pred_first_wins == true_first_wins)
```

---

## 7. Transductive Learning {#transductive-learning}

### 7.1 What is Transductive Learning?

**Transductive learning** means using test set features (but not labels) during training.

In this codebase:
- **Standard (inductive)**: Train only on `train_observed`
- **Transductive**: Train on `train_observed` + `test_observed`

### 7.2 Implementation

**In run_imputer.py**:

```python
train_vars = train_observed

if args.transductive_learning:
    print("Using transductive learning")
    train_vars += test_observed  # Include test observed in training!

trainer.train(
    train_observed_vars=train_vars,
    train_missing_vars=train_missing,
    masking_rate=masking_rate,
    epochs=epochs
)
```

### 7.3 Why Transductive Learning?

**Benefits**:
1. Model sees test set items during training (even if not labels)
2. Can learn better item embeddings for test items
3. Improves prediction of test set missing entries

**Use Case**:
When test items are **different** from train items (e.g., `K_train=10, K_test=10`), transductive learning helps the model learn embeddings for test items using the observed test set ratings.

**Comparison**:
- **Inductive**: Predict test missing entries using only train data
- **Transductive**: Predict test missing entries using train + test observed data

---

## 8. Data Flow Diagram {#data-flow}

### End-to-End Pipeline

```
[1. Data Generation (Stan)]
    ↓
models/iclr_data_generation.stan
    ↓ (samples ground truth + observations)
    ↓
GroundTruthBundle
    - embeddings, preferences
    - all_ratings, all_pairwise
    - observed_ratings, missing_ratings
    - observed_pairwise, missing_pairwise
    - stats
    ↓
OUTPUT/data/run_*/data_bundle.json

---

[2. Data Loading (Imputer)]
    ↓
DataConverter.load_bundle_data()
    ↓
DataConverter.create_variables_from_bundle()
    ↓
List[RankingData] for each partition/status:
    - train_observed (status=2, instance="train")
    - train_missing (status=0, instance="train")
    - test_observed (status=2, instance="test")
    - test_missing (status=0, instance="test")

---

[3. Training]
    ↓
ImputerTrainer.train_step():
    1. Apply masking to train_observed → masked/observed mix
    2. Build batch: [masked/observed] + [missing]
    3. Forward: model(batch) → logits
    4. Loss: compute on masked + observed only
    5. Backprop
    ↓
Repeat for epochs

---

[4. Evaluation]
    ↓
EvaluationEngine.evaluate_model():
    - No masking applied
    - Forward on test_all (observed + missing)
    - Compute metrics separately:
        * observed_metrics (status=2)
        * missing_metrics (status=0)
    ↓
EvaluationResults

---

[5. Save Outputs]
    ↓
OUTPUT/IMPUTER/run_*/
    - model.pt
    - test_metrics.json
    - predictives.json
    - train_config.json
```

---

## 9. Key Concepts and Terminology {#key-concepts}

### 9.1 Index Conventions

| Context | Convention | Example |
|---------|-----------|---------|
| **JSON data** | 1-indexed | `item=1` is first item |
| **Model internal** | 0-indexed | `item_ids=[0]` is first item |
| **DataConverter** | Converts 1→0 | Automatic during loading |

### 9.2 Status Encoding

| Code | Name | Meaning | Used in Training Loss? | Used in Eval? |
|------|------|---------|------------------------|---------------|
| `0` | missing | Not observed | ❌ No | ✅ Yes (missing_metrics) |
| `1` | masked | Artificially hidden | ✅ Yes | ❌ No (usually empty) |
| `2` | observed | Available | ✅ Yes | ✅ Yes (observed_metrics) |

### 9.3 Instance Convention

| Value | Meaning | Usage |
|-------|---------|-------|
| `"train"` | Training instance | Items from K_train |
| `"test"` | Test instance | Items from K_test |

### 9.4 Ranking Order Convention

For pairwise rankings with `items=[k1, k2]`:
- `order=[1, 2]`: k1 is preferred (k1 > k2)
- `order=[2, 1]`: k2 is preferred (k2 > k1)

**Note**: Lower rank = better (rank 1 is best)

### 9.5 Model I/O

**Input**: `List[RankingData]` of length N

**Output**:
```python
{
    'rating': torch.Tensor([B, N, C]),   # Rating logits
    'ranking': torch.Tensor([B, N, R])   # Ranking logits
}
```

Where:
- `B=1` (batch size always 1 in current implementation)
- `N` = number of variables
- `C` = num_likert_classes
- `R` = max_rank_size

### 9.6 Important Files Reference

| File | Purpose |
|------|---------|
| `stan/scripts/generate_data.py` | Generate synthetic data |
| `imputer/run_imputer.py` | Main training script |
| `imputer/ranking_imputer.py` | Model definition |
| `imputer/transformer.py` | Transformer blocks |
| `imputer/trainer.py` | Training loop with masking |
| `imputer/eval.py` | Evaluation engine |
| `imputer/losses.py` | Loss functions |
| `imputer/data.py` | Data structures |
| `imputer/embedding.py` | Embedding providers |
| `stan/pipeline/bundle.py` | GroundTruthBundle definition |
| `stan/pipeline/data_gen.py` | Stan data generation wrapper |

---

## Quick Start Commands

### Generate Data
```bash
cd imputer/ranking
export PYTHONPATH=.

python stan/scripts/generate_data.py \
    --K-train 10 --K-test 10 \
    --I 3 --J 9 --D 64 --C 5 \
    --output-dir OUTPUT/data \
    --seed 42
```

### Train Imputer (Inductive)
```bash
python imputer/run_imputer.py \
    --data-dir OUTPUT/data/run_YYYYMMDD_HHMMSS \
    --output-root OUTPUT/IMPUTER \
    --epochs 50 \
    --masking-rate 0.15 \
    --lr 1e-3 \
    --device cuda
```

### Train Imputer (Transductive)
```bash
python imputer/run_imputer.py \
    --data-dir OUTPUT/data/run_YYYYMMDD_HHMMSS \
    --output-root OUTPUT/IMPUTER \
    --epochs 50 \
    --masking-rate 0.15 \
    --lr 1e-3 \
    --device cuda \
    --transductive_learning  # Include test_observed in training
```

---

## Summary of Key Insights

1. **Data is 1-indexed in JSON, 0-indexed in model** - DataConverter handles conversion
2. **Three status codes**: missing(0), masked(1), observed(2)
3. **Training uses artificial masking**: Randomly mask M% of observed to create self-supervised task
4. **Loss is computed ONLY on non-missing entries** (masked + observed), NOT on truly missing
5. **Evaluation has no masking**: Directly evaluate on observed and missing test data
6. **Transductive learning**: Optionally include test_observed in training for better test predictions
7. **Unified stream transformer**: Features and parameters flow through the same attention mechanism
8. **Predictions read from parameters**: Rating/ranking logits are extracted from parameter stream
9. **Two instance types**: train (K_train items) and test (K_test items)
10. **Ground truth is known**: Stan generates data with known embeddings/preferences for evaluation

---

**End of Documentation**
