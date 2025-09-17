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

## Phase 4: Multi-Instance Training ✅ COMPLETED

### File: `imputer/multi_instance_trainer.py` ✅ IMPLEMENTED

**Implemented Classes:**
- `MultiInstanceTrainerBase` - Base class with generator-based training loop
- `SequentialMIT` - Sequential multi-instance trainer (exhausts each instance)
- `MixedMIT` - Mixed multi-instance trainer (IID sampling from all instances)
- `GeneralMIT` - General multi-instance trainer (placeholder for finetuning)

**Key Functions & Flow:**

**File: `imputer/multi_instance_trainer.py`**
- `MultiInstanceTrainerBase.__init__()` → initializes base trainer with eval_engine, config, converter
- `MultiInstanceTrainerBase.train_on_instances()` → unified training loop using generator
- `MultiInstanceTrainerBase.create_training_generator()` → abstract method for data generation
- `MultiInstanceTrainerBase.create_masked_batch()` → creates masked batches using DataConverter
- `SequentialMIT.create_training_generator()` → exhausts each instance before next
- `MixedMIT.create_training_generator()` → IID sampling from all instances
- `GeneralMIT.finetune_on_instance()` → placeholder for finetuning (not implemented)

**Function Call Flow:**
```
MultiInstanceTrainerBase.train_on_instances(train_instances, test_instances)
├── create_training_generator(train_instances, total_batches, batch_size) → generator
├── For each batch in generator:
│   ├── trainer.train_step(batch)
│   └── evaluate_on_test_instances(test_instances) if should_evaluate(step)

SequentialMIT.create_training_generator(train_instances, total_batches, batch_size)
├── batches_per_instance = total_batches // len(train_instances)
├── For each instance:
│   └── For each batch in range(batches_per_instance):
│       ├── masking_rate = random.choice(config.masking_rates)
│       └── yield create_masked_batch(instance, masking_rate, batch_size)

MixedMIT.create_training_generator(train_instances, total_batches, batch_size)
├── For each batch in range(total_batches):
│   ├── instance = random.choice(train_instances)
│   ├── masking_rate = random.choice(config.masking_rates)
│   └── yield create_masked_batch(instance, masking_rate, batch_size)

GeneralMIT.finetune_on_instance(pretrained_model, instance_data)
├── Split instance → Test_O_Observed, Test_O_Masked
├── Train on Test_O_Observed
└── Evaluate on Test_O_Masked
```

**Key Implementation Details:**
- **Generator Pattern**: Clean separation of data generation and training logic
- **DataConverter Integration**: Uses existing `DataConverter.create_batch()` for batch creation
- **Flexible Control**: `total_batches` and `batch_size` parameters control training length
- **Random Masking**: Random masking rate per batch from `config.masking_rates`
- **Evaluation Hook**: `evaluate_on_test_instances()` placeholder for future evaluation integration

**Testing Completed:**
- ✅ SequentialMIT: Generates batches per instance sequentially
- ✅ MixedMIT: Generates exact number of batches with IID sampling
- ✅ Both generators produce valid batches with proper structure
- ✅ Integration with existing data pipeline works correctly
- ✅ Batch creation with masking works properly

**Not Yet Implemented:**
- ❌ `MultiInstanceTrainerBase.evaluate_on_test_instances()` - placeholder only
- ❌ `GeneralMIT.finetune_on_instance()` - placeholder only
- ❌ Integration with EvaluationEngine for test instance evaluation

---

## Phase 5: Configuration Updates

**Key Functions & Flow:**

**File: `config.py`**
- `ModelConfig.embedding_type` → add "random" option
- `ExperimentConfig.test_masking_rate` → M% for test instances
- `ExperimentConfig.pretraining_mode` → "sequential" or "mixed"
- `ExperimentConfig.total_batches` → total number of batches to generate
- `ExperimentConfig.batch_size` → size of each batch
- `ExperimentConfig.masking_rates` → list of masking rates for training
- `ExperimentConfig.eval_frequency` → evaluation frequency during training
- `ExperimentConfig.evaluation_types` → list of evaluation strategies

### File: `config.py` (To Be Modified)

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
    total_batches: int = 1000  # Total number of batches to generate
    batch_size: int = 32  # Size of each batch
    masking_rates: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.5, 0.8, 1.0])  # Training masking rates
    eval_frequency: int = 100  # Evaluate every N steps

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

## Implementation Status

### ✅ COMPLETED PHASES
1. **Phase 1**: Evaluation Infrastructure ✅ COMPLETED
   - `EvaluationEngine` with comprehensive metrics
   - `EvaluationResults` container
   - Masking and splitting functionality
   - Full testing completed

2. **Phase 2**: Trainer Refactoring ✅ COMPLETED
   - `EvaluationCallback` system
   - Refactored `ImputerTrainer` with callbacks
   - Clean separation of training and evaluation
   - Full testing completed

3. **Phase 3**: Random Embedding Provider ✅ COMPLETED
   - `FullyRandomizedEmbeddingProvider` class
   - `reset_embedding()` method in base class
   - Non-trainable randomized embeddings
   - Data validation assertions
   - Full testing completed

4. **Phase 4**: Multi-Instance Training ✅ COMPLETED
   - `MultiInstanceTrainerBase` with generator pattern
   - `SequentialMIT` and `MixedMIT` implementations
   - `GeneralMIT` placeholder for finetuning
   - Integration with `DataConverter`
   - Full testing completed

### ❌ REMAINING PHASES
5. **Phase 5**: Configuration Updates ❌ NOT IMPLEMENTED
   - Add new config parameters
   - Update existing config files
   - Backwards compatibility

6. **Phase 6**: Main Experiment Runner ❌ NOT IMPLEMENTED
   - `experiment_runner_v2.py`
   - Integration with all components
   - End-to-end testing

7. **Phase 7**: Integration and Testing ❌ NOT IMPLEMENTED
   - Backwards compatibility
   - Comprehensive testing
   - Documentation updates

### NEXT STEPS
- **Phase 5**: Configuration updates (add new parameters to config.py)
- **Phase 6**: Main experiment runner (integrate all components)
- **Phase 7**: Integration testing and documentation