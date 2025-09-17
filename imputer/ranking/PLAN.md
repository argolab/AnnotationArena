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

**Key Features Implemented:**
```python
class EvaluationEngine:
    def __init__(self, config):
        self.config = config
        self.loss_strategy = DefaultLossStrategy()

    def evaluate_model(self, model, variables, data, masking_rate, converter, device):
        """Main evaluation with M% masking - FULLY IMPLEMENTED"""
        # - Creates M% evaluation mask across all variables
        # - Builds evaluation batch with proper masking
        # - Runs model forward pass
        # - Computes comprehensive metrics with masked/observed breakdown

    def create_evaluation_mask(self, variables, masking_rate):
        """Create M% random mask across all variables - IMPLEMENTED"""
        # Randomly selects M% of variables to mask for evaluation

    def split_variables(self, variables, mask):
        """Split into Test_M (masked) and Test_O (observed) - IMPLEMENTED"""
        # Returns tuple of (masked_variables, observed_variables)

    def compute_losses(self, model_output, targets, mask):
        """Total, rating, ranking log losses - IMPLEMENTED"""
        # Uses structured loss strategy for accurate loss computation

    def compute_accuracies(self, predictions, targets, variable_types):
        """Rating and ranking accuracies - IMPLEMENTED"""
        # Separate accuracy computation for each annotation type

    def compute_rmse(self, rating_predictions, rating_targets):
        """RMSE for ratings - IMPLEMENTED"""
        # Converts 0-indexed to 1-5 scale for proper RMSE calculation

    def evaluate_by_type(self, predictions, targets, masks):
        """Separate metrics for ratings vs rankings - IMPLEMENTED"""
        # Provides nested breakdown by annotation type and mask status

    def _create_evaluation_batch(self, variables, rating_data, ranking_data, evaluation_mask, converter):
        """Creates evaluation batch with masking applied - IMPLEMENTED"""
        # Key feature: Masked variables get no supervision in variable_data
        # Unmasked variables get normal supervision for model input

    def _compute_comprehensive_metrics(self, model_output, batch, variables, evaluation_mask, converter):
        """End-to-end metric computation - IMPLEMENTED"""
        # Structured approach using existing loss strategy
        # Separate tracking for masked vs observed performance
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

## Phase 2: Trainer Refactoring

### File: `imputer/trainer.py` (Modified)

```python
class EvaluationCallback:
    def __init__(self, eval_engine, test_data, masking_rate):
        self.eval_engine = eval_engine
        self.test_data = test_data
        self.masking_rate = masking_rate

    def on_epoch_end(self, model, epoch):
        return self.eval_engine.evaluate_model(
            model, self.test_data, self.masking_rate
        )

class ImputerTrainer:
    def __init__(self, model, learning_rate, callbacks=None):
        # Remove all evaluation logic
        # Add callback system

    def train_step(self, batch):
        # Keep existing training logic
        # Remove evaluation calls

    def register_callback(self, callback):
        """Register evaluation callback"""
```

### Test Script: `test_trainer_callbacks.py`
- Verify callbacks called at epoch end
- Check evaluation removed from trainer
- Test training still works correctly

---

## Phase 3: Random Embedding Provider

### File: `imputer/embedding.py` (Add class)

```python
class RandomEmbeddingProvider(RankingEmbeddingProviderBase):
    def __init__(self, embedding_dim, num_likert_classes, max_rank_size, device):
        super().__init__(embedding_dim, num_likert_classes, max_rank_size, device)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, ranking_data_list):
        """Generate fresh random embeddings each forward pass"""
        batch_size = len(ranking_data_list)
        embeddings = []

        for data in ranking_data_list:
            # Generate fresh random embeddings for this data point
            attr_emb = torch.randn(1, self.embedding_dim, device=self.device)
            annot_emb = torch.randn(1, self.embedding_dim, device=self.device)

            if data.is_listwise:
                # Random embeddings for each item in ranking
                item_embs = [torch.randn(1, self.embedding_dim, device=self.device)
                           for _ in data.item_ids]
                # Composition logic like PairwiseRankingProjectionEmbeddingProvider
            else:
                # Single item embedding for rating
                item_emb = torch.randn(1, self.embedding_dim, device=self.device)

            # Add and project (same composition as existing providers)
            combined = attr_emb + annot_emb + item_representation
            projected = self.projection(combined)
            embeddings.append(projected)

        return torch.cat(embeddings, dim=0)

    def parameters(self):
        """Only projection layer is trainable"""
        return self.projection.parameters()
```

### File: `ranking_imputer.py` (Add option)
```python
elif embedding_type == "random":
    self.embedding_provider = RandomEmbeddingProvider(
        embedding_dim, num_likert_classes, max_rank_size, device
    )
```

### Test Script: `test_random_embeddings.py`
- Test embeddings different each forward pass
- Test no shared embeddings between data points
- Verify composition logic works
- Check only projection trainable

---

## Phase 4: Mixed Instance Training

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