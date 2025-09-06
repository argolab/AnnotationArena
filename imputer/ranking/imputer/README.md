Ranking Imputer — Architecture Overview

This directory contains a modular, extensible implementation for ranking/rating imputation. The design emphasizes clean separation of concerns, structured data flow, and model-agnostic losses.

Core Concepts

- Data (data.py)
  - RankingData: Structured variable unit with 0-indexed ids and optional targets.
    - Fields: annotator_id, attribute_id, is_listwise, item_ids, rating_value?, ranking_order?
    - Conventions: rank 1 is best (ascending ranks).
  - DataConverter: Utilities to load JSON data and produce:
    - Lists of variables (rating_variables, ranking_variables)
    - Structured variables (List[RankingData]) with optional targets
    - Legacy tensor batches (create_training_batch) for compatibility with existing pipelines

- Embeddings (abstractions.py, embedding.py)
  - RankingEmbeddingProviderBase: Abstract base providing forward(List[RankingData]) -> [B,N,D]. Subclasses implement:
    - get_rating_embedding(attr_id, annot_id, item_id, rating_value)
    - get_ranking_embedding(attr_id, annot_id, item_ids, ranking_order)
  - OuterProductRankingEmbeddingProvider: Concrete provider using pairwise outer-product pooling for listwise items.

- Transformer (transformer.py)
  - TransformerBlock: Single-stream encoder block with pre-norm, MHA, FFN.
    - Supports optional attention mask `[B,N]` (True=valid). Padded tokens neither attend nor update.
  - NormLayer, FeedForward helpers.

- Model (ranking_imputer.py)
  - MultiVariableImputer (nn.Module): Encapsulates the full learnable model for easy save/load.
    - Components: embedding provider + ModuleList[TransformerBlock] + ModuleDict heads
    - Heads: {'rating': Linear(D,C), 'ranking': Linear(D,R)}
    - Forward:
      - Inputs: List[RankingData] or legacy tensors; optional `attn_mask`
      - Returns: logits dict {'rating': [B,N,C], 'ranking': [B,N,R]}
      - Optionally returns intermediate hidden states (return_hidden=True)

- Losses (losses.py)
  - PlackettLuceLoss: Ranking loss over (masked) positions; expects ranks where 1 is best.
  - PredictionResult: Lightweight container for per-variable logits.
  - adapt_batched_logits_to_predictions: Converts logits dict {'rating','ranking'} to List[PredictionResult] for B=1.
  - LossStrategyBase / DefaultLossStrategy:
    - compute(predictions, references, masked_flags) -> metrics
    - Converts structured lists to batched tensors, then reuses CrossEntropy and Plackett–Luce with observed/masked splits.

- Trainer (trainer.py)
  - ImputerTrainer: Orchestrates forward, builds structured predictions/references, computes loss with DefaultLossStrategy, and steps the optimizer.
  - Uses the legacy tensor batch format for now; can migrate to fully structured batching.

Data & Rank Conventions

- Indices: annotator_id, attribute_id, item_ids are 0-indexed inside the model.
- Ranking targets: ranking_order uses ascending ranks (1=best). PL loss sorts ascending.
- Attention masks: when batching multiple variable lists, pad to `[B,N_max,D]` and pass `attn_mask: [B,N_max]`.

Save/Load

- Save: `torch.save({'config': cfg, 'state_dict': model.state_dict()}, path)`
- Load: `model = MultiVariableImputer(**cfg); model.load_state_dict(ckpt['state_dict'])`

Extensibility

- New embeddings: subclass RankingEmbeddingProviderBase.
- New losses: subclass LossStrategyBase; the trainer remains unchanged.
- New heads or multi-task: add to `self.heads` and consume in the trainer’s supervision plan.

Notes

- The legacy tensor batch path is supported via DataConverter.create_training_batch; structured `List[RankingData]` is preferred for new work.
- A future `compute_batch` can extend DefaultLossStrategy to accept `[B,N,...]` structured batches directly.

