# Implementation Plan: Revised Experimental Framework

## Overview
Implement a new experimental framework supporting different imputer types with pretraining/finetuning capabilities and proper test instance evaluation with masking.

## Core Changes Summary

### Data Organization
- **Train Instances (7)**: Rating/ranking data split into train/dev within each instance
- **Test Instances (3)**: M% variables masked (Test_M), remaining observed (Test_O)
- **Mixed Training**: IID sampling from combined variable pool across all training instances

### Imputer Types
1. **Pretrained**: Train on training instances, evaluate directly on test instances
2. **Pretrained + Finetuned**: Pretrained + finetune on Test_O, evaluate on Test_M
3. **Fresh**: No pretraining, train only on Test_O, evaluate on Test_M
4. **Domain Model**: Fit to Test_O, evaluate on Test_M

---

## Phase 1: Evaluation Infrastructure ✅ COMPLETED

### File: `imputer/eval.py` ✅ IMPLEMENTED

**Implemented Classes:**
- `EvaluationResults` - Container for evaluation metrics with masked/observed breakdown
- `EvaluationEngine` - Main evaluation engine with comprehensive functionality

**Key Functions & Flow:**

**File: `imputer/eval.py`**
- `EvaluationEngine.evaluate_model()` → main entry point for evaluation
- `EvaluationEngine.create_evaluation_mask()` → creates M% random mask
- `EvaluationEngine.split_variables()` → splits into Test_M/Test_O
- `EvaluationEngine._create_evaluation_batch()` → builds evaluation batch with masking
- `EvaluationEngine._compute_comprehensive_metrics()` → computes all metrics

**Function Call Flow:**
```
EvaluationEngine.evaluate_model(model, variables, data, masking_rate, converter, device)
├── create_evaluation_mask(variables, masking_rate)
├── split_variables(variables, mask) → (masked_vars, observed_vars)
├── _create_evaluation_batch(variables, rating_data, ranking_data, mask, converter)
│   ├── Apply masking to supervision
│   └── Return evaluation batch
├── model.forward(evaluation_batch)
└── _compute_comprehensive_metrics(model_output, batch, variables, mask, converter)
    ├── compute_losses() → total, rating, ranking losses
    ├── compute_accuracies() → rating/ranking accuracies
    ├── compute_rmse() → RMSE for ratings
    └── evaluate_by_type() → breakdown by type and mask status
```

**Testing Completed:**
- ✅ Masking logic verified (0%, 20%, 50%, 80%, 100% masking rates)
- ✅ Metric computations tested (RMSE, accuracy)
- ✅ Test_M/Test_O splitting verified (no overlap, preserves counts)
- ✅ Evaluation batch creation tested (proper supervision masking)
- ✅ End-to-end evaluation with real model verified

**Integration Points Ready:**
- Compatible with existing `DataConverter` and model architecture
- Ready for trainer callbacks (Phase 2)
- API designed for mixed instance trainers (Phase 4)

---

## Phase 2: Trainer Refactoring ✅ COMPLETED

### File: `imputer/trainer.py` ✅ IMPLEMENTED

**Implemented Classes:**
- `EvaluationCallback` - Callback for evaluation during training using EvaluationEngine
- `ImputerTrainer` (Refactored) - Clean trainer with callback system, no evaluation logic
- `EarlyStopping` - Utility for early stopping (unchanged)
- `calculate_rmse` - Utility function (unchanged)

**Key Functions & Flow:**

**File: `imputer/trainer.py`**
- `EvaluationCallback.__init__()` → stores evaluation components
- `EvaluationCallback.on_epoch_end()` → calls `EvaluationEngine.evaluate_model()`
- `ImputerTrainer.register_callback()` → adds callback to list
- `ImputerTrainer.train()` → training loop with callback support
- `ImputerTrainer._call_epoch_end_callbacks()` → calls all registered callbacks

**Function Call Flow:**
```
ImputerTrainer.train(train_batches, epochs, call_callbacks_every, verbose)
├── For each epoch:
│   ├── For each batch:
│   │   └── train_step(batch) → forward/backward pass
│   └── _call_epoch_end_callbacks(epoch)
│       └── EvaluationCallback.on_epoch_end(model, epoch)
│           └── EvaluationEngine.evaluate_model(model, test_variables, test_data, masking_rate, converter, device)
└── Return training_history + callback_history
```

**Old Evaluation Methods:**
- ✅ `evaluate_with_test_data()` - REMOVED (replaced with EvaluationEngine + callbacks)
- ✅ `print_predictions_by_attribute()` - REMOVED (functionality moved to EvaluationEngine)

**Testing Completed:**
- ✅ Callback registration and storage works correctly
- ✅ EvaluationCallback creation and on_epoch_end functionality verified
- ✅ Training loop with callbacks works (epochs=3, different frequencies tested)
- ✅ Callback frequency control works (every 1, 2, 3 epochs)
- ✅ Training step functionality unchanged (loss computation verified)
- ✅ Integration with EvaluationEngine works correctly

**Backwards Compatibility:**
- ImputerTrainer constructor accepts callbacks parameter (optional)
- train_step() method unchanged - existing code will work
- New train() method provides modern training loop with callbacks
- EvaluationCallback integrates seamlessly with EvaluationEngine from Phase 1

**Integration Points Ready:**
- Ready for mixed instance trainers (Phase 4) - callback system is extensible
- Ready for configuration updates (Phase 5) - masking rates configurable in callbacks
- Clean separation of concerns - training logic vs evaluation logic

---

## Phase 3: Random Embedding Provider ✅ COMPLETED

### File: `imputer/embedding.py` ✅ IMPLEMENTED

**Implemented Classes:**
- `FullyRandomizedEmbeddingProvider` - Fully randomized embedding provider that generates new random embeddings for each forward pass
- `BaseRankingEmbeddingProvider.reset_embedding()` - Method to reset embeddings with new random values

**Key Functions & Flow:**

**File: `imputer/embedding.py`**
- `FullyRandomizedEmbeddingProvider.on_forward_start()` → calculates required dimensions → calls `reset_embedding()`
- `BaseRankingEmbeddingProvider.reset_embedding()` → creates new random embeddings → sets `requires_grad=False`
- Inherits: `get_rating_embedding()`, `get_ranking_embedding()` from `PairwiseRankingProjectionEmbeddingProvider`

**File: `imputer/abstractions.py`**
- `RankingEmbeddingProviderBase.forward()` → calls `on_forward_start()` → processes variables → returns embeddings

**Function Call Flow:**
```
model.forward(variables)
├── RankingEmbeddingProviderBase.forward()
│   ├── on_forward_start(variables)  # Hook for randomization
│   │   └── FullyRandomizedEmbeddingProvider.on_forward_start()
│   │       ├── Calculate max IDs from variables
│   │       └── reset_embedding(required_dims)
│   │           ├── Create new random embeddings
│   │           ├── Apply kaiming initialization
│   │           └── Set requires_grad=False
│   └── Process each variable
│       ├── get_rating_embedding() or get_ranking_embedding()
│       └── Return feature embeddings
```

**Key Differences from Original Plan:**
- **Inheritance Approach**: Extends `PairwiseRankingProjectionEmbeddingProvider` instead of `RankingEmbeddingProviderBase` directly
- **Dynamic Sizing**: Calculates required dimensions based on actual IDs in the batch
- **Hook Integration**: Uses `on_forward_start()` hook called by the base `forward()` method
- **Same Composition Logic**: Reuses existing `get_rating_embedding` and `get_ranking_embedding` methods
- **Data Validation**: Added assertions for `max_rank_size` and `num_likert_classes` consistency

**Testing Completed:**
- ✅ Different embeddings generated for each forward pass
- ✅ Dynamic sizing based on batch content works correctly
- ✅ Same interface and behavior as parent class maintained
- ✅ Data validation assertions work (max_rank_size, num_likert_classes)
- ✅ Integration with existing forward() method verified
- ✅ Randomized embeddings correctly set as non-trainable (requires_grad=False)
- ✅ Only projection layers remain trainable (parameter_projection, pairwise_relation)

**Integration Points Ready:**
- Ready for use in `ranking_imputer.py` (can be added as embedding_type option)
- Compatible with existing trainer and evaluation infrastructure
- Maintains same API as other embedding providers

**Code Simplifications Applied:**
- ✅ `max_rank_size` assertion: Ensures `len(ranking_order) == max_rank_size`
- ✅ `num_likert_classes` assertion: Ensures `0 <= rating_value < num_likert_classes`
- ✅ Cleaner, more robust code with early data validation

---

## Phase 4: Mixed Instance Training

**Key Functions & Flow:**

**File: `imputer/mixed_instance_trainer.py`**
- `MixedInstanceTrainerBase.__init__()` → initializes base trainer with eval_engine
- `MixedInstanceTrainerBase.setup_evaluation_callback()` → creates callback for test instances
- `SequentialMIT.train_on_instances()` → trains instance by instance
- `MixedMIT.train_on_instances()` → combines instances, samples IID
- `MixedMIT.combine_instances()` → merges all instance variables
- `GeneralMIT.finetune_on_instance()` → finetunes on single instance

**Function Call Flow:**
```
SequentialMIT.train_on_instances(train_instances, test_instances)
├── For each train_instance:
│   ├── train_single_instance(instance)
│   └── evaluate_on_test_instances(test_instances)

MixedMIT.train_on_instances(train_instances, test_instances)
├── combine_instances(train_instances) → combined_variables
├── For each epoch:
│   ├── sample_iid_batch(combined_variables)
│   ├── trainer.train_step(batch)
│   └── evaluate_on_test_instances(test_instances)

GeneralMIT.finetune_on_instance(pretrained_model, instance_data)
├── Split instance → Test_O_Observed, Test_O_Masked
├── Train on Test_O_Observed
└── Evaluate on Test_O_Masked
```

### File: `imputer/mixed_instance_trainer.py`

```python
class MixedInstanceTrainerBase:
    def __init__(self, model, eval_engine, config):
        self.model = model
        self.eval_engine = eval_engine
        self.config = config
        self.trainer = ImputerTrainer(model, config.learning_rate)

    def setup_evaluation_callback(self, test_instances):
        """Setup callback for test instance evaluation"""

class SequentialMIT(MixedInstanceTrainerBase):
    def train_on_instances(self, train_instances, test_instances):
        """Train instance by instance, evaluate on test instances after each"""
        for i, instance in enumerate(train_instances):
            # Train on instance with M% masking
            self.train_single_instance(instance)
            # Evaluate on all test instances
            results = self.evaluate_on_test_instances(test_instances)

class MixedMIT(MixedInstanceTrainerBase):
    def train_on_instances(self, train_instances, test_instances):
        """Create big combined instance, sample IID"""
        combined_variables = self.combine_instances(train_instances)

        for epoch in range(self.config.epochs):
            # Sample IID batch from combined variables
            batch = self.sample_iid_batch(combined_variables)
            # Train with M% masking
            self.trainer.train_step(batch)
            # Evaluate on test instances

    def combine_instances(self, instances):
        """Combine all instance variables (keep distinct IDs)"""

class GeneralMIT(MixedInstanceTrainerBase):
    def finetune_on_instance(self, pretrained_model, instance_data):
        """Finetune on single instance (for test instance finetuning)"""
        # Split instance into Test_O_Masked and Test_O_Observed
        # Train on observed, validate on masked
```

### Test Script: `test_mixed_trainers.py`
- Test sequential training
- Test mixed instance combination preserves IDs
- Test IID sampling from combined pool
- Test finetuning logic

---

## Phase 5: Configuration Updates

**Key Functions & Flow:**

**File: `config.py`**
- `ModelConfig.embedding_type` → add "random" option
- `ExperimentConfig.test_masking_rate` → M% for test instances
- `ExperimentConfig.pretraining_mode` → "sequential" or "mixed"
- `ExperimentConfig.evaluation_types` → list of evaluation strategies

### File: `config.py` (Modified)

```python
@dataclass
class ModelConfig:
    encoder_layers: int = 4
    attention_heads: int = 8
    embedding_dim: int = 64
    dropout: float = 0.1
    embedding_type: str = "pairwise"  # Add "random" option

@dataclass
class ExperimentConfig:
    # Existing fields...

    # New fields
    test_masking_rate: float = 0.3  # M% for test instances
    pretraining_mode: str = "sequential"  # "sequential" or "mixed"

    evaluation_types: List[str] = field(default_factory=lambda: [
        "pretrained",           # Direct evaluation
        "pretrained_finetuned", # Pretrained + finetune on Test_O
        "fresh",               # No pretraining, train on Test_O only
        "domain"               # Domain model on Test_O
    ])
```

### Update config files:
- `configs/single_instance.json`
- `configs/multi_instance_demo.json`

---

## Phase 6: Main Experiment Runner

**Key Functions & Flow:**

**File: `experiment_runner_v2.py`**
- `ExperimentRunnerV2.__init__()` → initializes with config and eval_engine
- `ExperimentRunnerV2.run_experiment()` → main entry point
- `ExperimentRunnerV2.run_pretrained_evaluation()` → trains on train_instances
- `ExperimentRunnerV2.run_finetuned_evaluation()` → pretrain + finetune on Test_O
- `ExperimentRunnerV2.run_fresh_evaluation()` → train fresh on Test_O
- `ExperimentRunnerV2.run_domain_evaluation()` → domain model evaluation
- `ExperimentRunnerV2.evaluate_all_test_instances()` → evaluate on all test instances

**Function Call Flow:**
```
ExperimentRunnerV2.run_experiment()
├── load_instances() → train_instances, test_instances
├── For each evaluation_type in config.evaluation_types:
│   ├── "pretrained" → run_pretrained_evaluation()
│   │   ├── Choose trainer (SequentialMIT or MixedMIT)
│   │   ├── trainer.train_on_instances(train_instances, test_instances)
│   │   └── evaluate_all_test_instances(pretrained_model, test_instances)
│   ├── "pretrained_finetuned" → run_finetuned_evaluation()
│   │   ├── get_pretrained_model(train_instances)
│   │   └── For each test_instance:
│   │       ├── GeneralMIT.finetune_on_instance(pretrained_model, test_instance)
│   │       └── eval_engine.evaluate_model(finetuned_model, test_instance, masking_rate)
│   ├── "fresh" → run_fresh_evaluation(test_instances)
│   └── "domain" → run_domain_evaluation(test_instances)
└── Return results dictionary
```

### File: `experiment_runner_v2.py`

```python
class ExperimentRunnerV2:
    def __init__(self, config):
        self.config = config
        self.eval_engine = EvaluationEngine(config)

    def run_experiment(self):
        """Main experiment entry point"""
        train_instances, test_instances = self.load_instances()
        results = {}

        if "pretrained" in self.config.evaluation_types:
            results['pretrained'] = self.run_pretrained_evaluation(
                train_instances, test_instances
            )

        if "pretrained_finetuned" in self.config.evaluation_types:
            results['pretrained_finetuned'] = self.run_finetuned_evaluation(
                train_instances, test_instances
            )

        if "fresh" in self.config.evaluation_types:
            results['fresh'] = self.run_fresh_evaluation(test_instances)

        if "domain" in self.config.evaluation_types:
            results['domain'] = self.run_domain_evaluation(test_instances)

        return results

    def run_pretrained_evaluation(self, train_instances, test_instances):
        """Train on training instances, evaluate on test instances"""
        if self.config.pretraining_mode == "sequential":
            trainer = SequentialMIT(model, self.eval_engine, self.config)
        else:
            trainer = MixedMIT(model, self.eval_engine, self.config)

        pretrained_model = trainer.train_on_instances(train_instances, test_instances)
        return self.evaluate_all_test_instances(pretrained_model, test_instances)

    def run_finetuned_evaluation(self, train_instances, test_instances):
        """Pretrain + finetune on Test_O of each test instance"""
        pretrained_model = self.get_pretrained_model(train_instances)
        results = {}

        for test_idx, test_instance in enumerate(test_instances):
            finetuner = GeneralMIT(pretrained_model, self.eval_engine, self.config)
            finetuned_model = finetuner.finetune_on_instance(test_instance)
            results[test_idx] = self.eval_engine.evaluate_model(
                finetuned_model, test_instance, self.config.test_masking_rate
            )

        return results

    def run_fresh_evaluation(self, test_instances):
        """Train fresh model on Test_O, evaluate on Test_M"""

    def run_domain_evaluation(self, test_instances):
        """Domain model evaluation"""

    def evaluate_all_test_instances(self, model, test_instances):
        """Evaluate model on all test instances separately"""
        results = {}
        for test_idx, test_instance in enumerate(test_instances):
            results[test_idx] = self.eval_engine.evaluate_model(
                model, test_instance, self.config.test_masking_rate
            )
        return results
```

### Test Script: `test_end_to_end.py`
- Test complete experimental pipeline
- Verify all evaluation types work
- Check results format consistency

---

## Phase 7: Integration and Testing

### Backwards Compatibility
- Ensure existing `experiment_runner.py` still works
- Add migration path for old configs
- Maintain existing API where possible

### Comprehensive Testing
- Integration tests with small datasets
- Performance testing
- Error handling and edge cases

### Documentation
- Update README with new experimental design
- Document new configuration options
- Add examples of different evaluation types

---

## Implementation Order

1. **Week 1**: Phase 1-2 (Evaluation infrastructure + trainer refactoring)
2. **Week 1**: Phase 3 (Random embeddings)
3. **Week 2**: Phase 4 (Mixed instance trainers)
4. **Week 2**: Phase 5-6 (Configuration + main runner)
5. **Week 3**: Phase 7 (Integration, testing, documentation)

Each phase includes immediate testing before proceeding to ensure functionality and catch issues early.