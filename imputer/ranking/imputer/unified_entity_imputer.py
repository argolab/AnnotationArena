"""
Unified Entity Imputer — entities as tokens in self-attention sequence.

Architecture:
  1. Entity tokens (I attrs + J annotators + K items) in same sequence as observations
  2. Edge-label key dimensions (IS_ATTR, IS_ANNOT, IS_ITEM) for entity tokens
  3. Pointer mechanism (SAME_ATTR, SAME_ANNOT, SAME_ITEM) between observations
  4. Type embeddings + optional per-entity deviation with dropout
  5. Data-dependent item embeddings from mean-pooled observed [features|params]
  6. Layer-0 observations: attribute + annotator composition
  7. Prediction head reads from [features | params] concatenation

External interface is identical to MultiVariableImputer:
  forward(ranking_data_list) → {'rating': [B,N,C], 'ranking': [B,N,R]}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict

from imputer.data import RankingData
from imputer.unified_entity_transformer import UnifiedEntityBlock
from imputer.transformer import NormLayer

logger = logging.getLogger(__name__)


class _DropoutController:
    """Minimal compatibility shim for eval.py / lightning_trainer.py."""

    def __init__(self):
        self.full_dropout_enabled = False

    def set_full_dropout(self, enabled: bool):
        self.full_dropout_enabled = enabled


class UnifiedEntityImputer(nn.Module):
    """Imputer with unified entity+observation self-attention.

    Sequence structure: [observations (N) | attributes (I) | annotators (J) | items (K)]

    Drop-in replacement for MultiVariableImputer.
    """

    def __init__(
        self,
        num_attributes: int = 8,
        num_annotators: int = 8,
        num_items: int = 8,
        num_likert_classes: int = 5,
        max_rank_size: int = 3,
        encoder_layers_num: int = 2,
        attention_heads: int = 4,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        device: str = "cuda",
        embedding_dropout: float | None = None,
        use_gelu_after_attention: bool = False,
        use_final_norm: bool = True,
        normalize_parameter: bool = False,
        num_ffn_layers: int = 4,
        d_ff: int = 512,
        temperature: float = 1.0,
        batch_size: int = 1,
        use_prediction_head: bool = False,
        logit_high: float = 20.0,
        # Entity-specific architecture parameters
        use_annotator_deviation: bool = True,
        annotator_dropout: float = 0.5,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.num_likert_classes = num_likert_classes
        self.max_rank_size = max_rank_size
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.batch_size = batch_size
        self.use_prediction_head = use_prediction_head
        self.LOGIT_HIGH = logit_high
        self.embedding_type = "unified_entity"
        self.use_annotator_deviation = use_annotator_deviation
        self.annotator_dropout = annotator_dropout

        # Feature dimension = embedding_dim
        self.feature_dim = embedding_dim

        # Parameter stream dimension: 1 (mask bit) + max(num_likert_classes, max_rank_size)
        self.param_dim = 1 + max(num_likert_classes, max_rank_size)

        if embedding_dropout is None:
            embedding_dropout = dropout

        # Compatibility shim
        self.embedding_provider = _DropoutController()

        # ════════════════════════════════════════════════════════════
        # TYPE EMBEDDINGS (shared across all entities of same type)
        # ════════════════════════════════════════════════════════════
        # Attribute type embedding
        self.attr_type_emb = nn.Parameter(torch.empty(1, embedding_dim))
        # Annotator type embedding
        self.annot_type_emb = nn.Parameter(torch.empty(1, embedding_dim))
        # Item type embedding
        self.item_type_emb = nn.Parameter(torch.empty(1, embedding_dim))

        nn.init.kaiming_normal_(self.attr_type_emb, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.annot_type_emb, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.item_type_emb, mode='fan_out', nonlinearity='relu')

        # ════════════════════════════════════════════════════════════
        # PER-ENTITY DEVIATION EMBEDDINGS (optional, with dropout)
        # ════════════════════════════════════════════════════════════
        # Per-attribute deviation (always used for attributes)
        self.attr_deviation = nn.Parameter(torch.empty(num_attributes, embedding_dim))
        nn.init.zeros_(self.attr_deviation)  # Start with zero deviation

        # Per-annotator deviation (optional, with dropout regularization)
        if use_annotator_deviation:
            self.annot_deviation = nn.Parameter(torch.empty(num_annotators, embedding_dim))
            nn.init.zeros_(self.annot_deviation)
            self.annot_deviation_dropout = nn.Dropout(annotator_dropout)
        else:
            self.annot_deviation = None
            self.annot_deviation_dropout = None

        # NO per-item deviation — items are always new at test time
        # Items get type embedding + data-dependent init from observed observations

        # ════════════════════════════════════════════════════════════
        # OBSERVATION COMPOSITION (layer-0 only)
        # ════════════════════════════════════════════════════════════
        # W_I: project attribute entity to observation feature space
        # W_J: project annotator entity to observation feature space
        # NO W_K in obs composition: items don't contribute to observation features
        self.W_I = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_J = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # W_K_init: project mean-pooled observed [features|params] → item init embedding
        self.W_K_init = nn.Linear(embedding_dim + self.param_dim, embedding_dim, bias=False)

        # ════════════════════════════════════════════════════════════
        # TRANSFORMER BLOCKS
        # ════════════════════════════════════════════════════════════
        self.blocks = nn.ModuleList([
            UnifiedEntityBlock(
                feature_dim=embedding_dim,
                param_dim=self.param_dim,
                attention_heads=attention_heads,
                dropout=dropout,
                use_gelu_after_attention=use_gelu_after_attention,
                normalize_parameter=normalize_parameter,
                num_ffn_layers=num_ffn_layers,
                d_ff=d_ff,
            )
            for _ in range(encoder_layers_num)
        ])

        # ════════════════════════════════════════════════════════════
        # FINAL NORMALIZATION / PREDICTION
        # ════════════════════════════════════════════════════════════
        if use_final_norm and not use_prediction_head:
            self.final_norm = NormLayer(self.param_dim)
        else:
            self.final_norm = None

        if use_prediction_head:
            head_input_dim = embedding_dim + self.param_dim
            self.rating_head = nn.Sequential(
                nn.Linear(head_input_dim, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, num_likert_classes),
            )
            self.ranking_head = nn.Sequential(
                nn.Linear(head_input_dim, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, max_rank_size),
            )
        else:
            self.rating_head = None
            self.ranking_head = None

    # ════════════════════════════════════════════════════════════════
    # BUILD ENTITY EMBEDDINGS
    # ════════════════════════════════════════════════════════════════

    def _get_entity_embeddings(self, B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build entity embeddings: type + optional deviation.

        Returns:
            attr_emb: [B, I, D]
            annot_emb: [B, J, D]
            item_emb: [B, K, D]
        """
        I, J, K = self.num_attributes, self.num_annotators, self.num_items
        device = self.attr_type_emb.device

        # Attributes: type + deviation
        attr_emb = self.attr_type_emb.expand(I, -1) + self.attr_deviation  # [I, D]

        # Annotators: type + optional deviation with dropout
        if self.use_annotator_deviation and self.annot_deviation is not None:
            annot_dev = self.annot_deviation_dropout(self.annot_deviation) if self.training else self.annot_deviation
            annot_emb = self.annot_type_emb.expand(J, -1) + annot_dev  # [J, D]
        else:
            annot_emb = self.annot_type_emb.expand(J, -1)  # [J, D]

        # Items: type only (no deviation — items always new at test)
        item_emb = self.item_type_emb.expand(K, -1)  # [K, D]

        # Expand for batch
        attr_emb = attr_emb.unsqueeze(0).expand(B, -1, -1)    # [B, I, D]
        annot_emb = annot_emb.unsqueeze(0).expand(B, -1, -1)  # [B, J, D]
        item_emb = item_emb.unsqueeze(0).expand(B, -1, -1)    # [B, K, D]

        return attr_emb, annot_emb, item_emb

    # ════════════════════════════════════════════════════════════════
    # BUILD OBSERVATION FEATURES (layer-0 only)
    # ════════════════════════════════════════════════════════════════

    def _build_observation_features(
        self,
        attr_emb: torch.Tensor,
        annot_emb: torch.Tensor,
        attr_ids: torch.Tensor,
        annot_ids: torch.Tensor,
        B: int,
        N: int,
    ) -> torch.Tensor:
        """Compose observation features from attribute and annotator entities.

        NO item contribution — items are always new at test time.

        Returns:
            features: [B, N, feature_dim]
        """
        # Gather entities for each observation
        attr_gathered = attr_emb[:, attr_ids, :]    # [B, N, D]
        annot_gathered = annot_emb[:, annot_ids, :]  # [B, N, D]

        # Project and sum (no W_K — no item contribution)
        features = self.W_I(attr_gathered) + self.W_J(annot_gathered)  # [B, N, D]
        return features

    # ════════════════════════════════════════════════════════════════
    # INITIALIZE ITEM EMBEDDINGS FROM DATA
    # ════════════════════════════════════════════════════════════════

    def _init_item_embeddings(
        self,
        item_emb_base: torch.Tensor,
        obs_features: torch.Tensor,
        obs_params: torch.Tensor,
        item_ids: torch.Tensor,
        status: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        """Initialize item embeddings from observed observation data.

        For each item k, mean-pool [features | params] of all observed
        observations involving item k, project with W_K_init, and add to base.

        Args:
            item_emb_base: [B, K, D] — base item embeddings (type embedding only)
            obs_features: [B, N, D] — observation features
            obs_params: [B, N, param_dim] — observation params
            item_ids: [N] — item id for each observation
            status: [N] — 0=missing, 1=masked, 2=observed
            B: batch size

        Returns:
            item_emb: [B, K, D] — data-dependent item embeddings
        """
        K = self.num_items
        N = obs_features.shape[1]
        device = item_emb_base.device

        # Combine features and params for richer representation
        obs_combined = torch.cat([obs_features, obs_params], dim=-1)  # [B, N, D+param_dim]

        # Only observed observations contribute (status != 0, valid item_id)
        observed_mask = (status != 0).float()  # [N]
        valid_item_mask = (item_ids >= 0).float()  # [N]
        contrib_mask = observed_mask * valid_item_mask  # [N]
        obs_combined = obs_combined * contrib_mask.view(1, N, 1)  # zero out non-contributing

        # Scatter-add by item_id
        safe_item_ids = item_ids.clamp(min=0, max=K - 1)
        ids_expanded = safe_item_ids.view(1, N, 1).expand(B, N, obs_combined.shape[-1])

        item_sum = torch.zeros(B, K, obs_combined.shape[-1], device=device)
        item_sum.scatter_add_(1, ids_expanded, obs_combined)

        # Count per item for mean
        count_ids = safe_item_ids.view(1, N, 1).expand(B, N, 1)
        item_count = torch.zeros(B, K, 1, device=device)
        item_count.scatter_add_(1, count_ids, contrib_mask.view(1, N, 1).expand(B, N, 1))

        item_mean = item_sum / (item_count + 1e-8)  # [B, K, D+param_dim]

        # Project and add to base embedding
        item_init = self.W_K_init(item_mean)  # [B, K, D]
        return item_emb_base + item_init

    # ════════════════════════════════════════════════════════════════
    # BUILD PARAMETER STREAM
    # ════════════════════════════════════════════════════════════════

    def _build_params(self, ranking_data_list: List[RankingData], B: int) -> torch.Tensor:
        """Construct initial parameter stream for observations.

        Returns:
            params: [B, N, param_dim]
        """
        N = len(ranking_data_list)
        device = self.attr_type_emb.device

        params_list = []
        for var in ranking_data_list:
            p = torch.zeros(self.param_dim, device=device)
            is_missing = var.is_missing or var.is_masked

            if is_missing:
                p[0] = 1.0  # mask bit
            else:
                p[0] = 0.0
                if var.is_listwise:
                    assert var.ranking_order is not None
                    if var.ranking_order[0] < var.ranking_order[1]:
                        p[1] = self.LOGIT_HIGH
                    else:
                        p[2] = self.LOGIT_HIGH
                else:
                    assert var.rating_value is not None and 0 <= var.rating_value < self.num_likert_classes
                    p[var.rating_value + 1] = self.LOGIT_HIGH
            params_list.append(p)

        params = torch.stack(params_list, dim=0).unsqueeze(0)  # [1, N, param_dim]
        if B > 1:
            params = params.expand(B, -1, -1).contiguous()
        return params

    def _build_entity_params(self, B: int, num_entities: int) -> torch.Tensor:
        """Build dummy params for entity tokens (no loss computed on these).

        Returns:
            params: [B, num_entities, param_dim]
        """
        device = self.attr_type_emb.device
        # Dummy params: all zeros (mask bit = 0, no logits)
        return torch.zeros(B, num_entities, self.param_dim, device=device)

    # ════════════════════════════════════════════════════════════════
    # BUILD KEY EDGE FEATURES
    # ════════════════════════════════════════════════════════════════

    def _build_key_edge_features(
        self,
        N: int,
        B: int,
    ) -> torch.Tensor:
        """Build fixed edge features for key positions.

        Sequence structure: [observations (N) | attributes (I) | annotators (J) | items (K)]

        Per Jason's proposal, edge labels should be properties of the KEY only:
          0: IS_ATTR — key token is an attribute entity
          1: IS_ANNOT — key token is an annotator entity
          2: IS_ITEM — key token is an item entity

        Note: SAME_ATTR/SAME_ANNOT/SAME_ITEM between observations are handled
        separately via the pointer mechanism (K_aug), not through key-only edges.

        Returns:
            key_edge_features: [B, L, 3] where L = N + I + J + K
        """
        I, J, K = self.num_attributes, self.num_annotators, self.num_items
        L = N + I + J + K
        device = self.attr_type_emb.device

        key_edges = torch.zeros(L, 3, device=device)

        # IS_ATTR (dim 0): positions N to N+I are attribute entities
        key_edges[N:N+I, 0] = 1.0

        # IS_ANNOT (dim 1): positions N+I to N+I+J are annotator entities
        key_edges[N+I:N+I+J, 1] = 1.0

        # IS_ITEM (dim 2): positions N+I+J to end are item entities
        key_edges[N+I+J:, 2] = 1.0

        # Expand for batch
        return key_edges.unsqueeze(0).expand(B, -1, -1)  # [B, L, 3]

    # ════════════════════════════════════════════════════════════════
    # BUILD PAIRWISE OBS-TO-OBS INDICATORS (K_aug)
    # ════════════════════════════════════════════════════════════════

    def _build_K_aug(
        self,
        attr_ids: torch.Tensor,
        annot_ids: torch.Tensor,
        item_ids: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        """Build pairwise SAME_ATTR/SAME_ANNOT/SAME_ITEM indicators for observations.

        Same mechanism as the old Marformer's pointer mechanism (K_aug).

        Args:
            attr_ids: [N] — attribute id per observation
            annot_ids: [N] — annotator id per observation
            item_ids: [N] — item id per observation
            B: batch size

        Returns:
            K_aug: [B, N, N, 3] where [b, i, j, :] = [same_attr, same_annot, same_item]
        """
        attr_indicator = (attr_ids.unsqueeze(0) == attr_ids.unsqueeze(1)).float()  # [N, N]
        annot_indicator = (annot_ids.unsqueeze(0) == annot_ids.unsqueeze(1)).float()  # [N, N]
        item_indicator = (item_ids.unsqueeze(0) == item_ids.unsqueeze(1)).float()  # [N, N]

        # Don't count invalid item_ids (-1) as matching
        invalid = (item_ids == -1)
        item_indicator[invalid, :] = 0
        item_indicator[:, invalid] = 0

        K_aug = torch.stack([attr_indicator, annot_indicator, item_indicator], dim=-1)  # [N, N, 3]
        return K_aug.unsqueeze(0).expand(B, -1, -1, -1)  # [B, N, N, 3]

    # ════════════════════════════════════════════════════════════════
    # BUILD ATTENTION MASK
    # ════════════════════════════════════════════════════════════════

    def _build_attn_mask(
        self,
        status: torch.Tensor,
        B: int,
        N: int,
    ) -> torch.Tensor:
        """Build attention mask for unified sequence.

        - Missing observations (status=0) are masked out
        - All entity tokens are valid

        Returns:
            attn_mask: [B, L] — True for valid tokens
        """
        I, J, K = self.num_attributes, self.num_annotators, self.num_items
        L = N + I + J + K
        device = status.device

        # Observations: valid if status != 0 (not missing)
        obs_valid = (status != 0)  # [N]

        # Entity tokens: always valid
        entity_valid = torch.ones(I + J + K, dtype=torch.bool, device=device)

        # Concatenate
        attn_mask = torch.cat([obs_valid, entity_valid])  # [L]
        return attn_mask.unsqueeze(0).expand(B, -1)  # [B, L]

    # ════════════════════════════════════════════════════════════════
    # PREDICTION READOUT
    # ════════════════════════════════════════════════════════════════

    def read_prediction_logits_from_param(self, head_key: str, params: torch.Tensor) -> torch.Tensor:
        """Read logits from param stream (for observations only)."""
        if head_key == "rating":
            return params[:, :, 1:1 + self.num_likert_classes]
        else:
            return params[:, :, 1:1 + self.max_rank_size]

    # ════════════════════════════════════════════════════════════════
    # FORWARD PASS
    # ════════════════════════════════════════════════════════════════

    def forward(
        self,
        ranking_data_list: List[RankingData],
        attn_mask: torch.Tensor | None = None,  # Unused, we build our own
        return_intermediate: bool = False,
        verbose: bool = False,
    ):
        N = len(ranking_data_list)
        device = self.attr_type_emb.device
        B = self.batch_size
        I, J, K = self.num_attributes, self.num_annotators, self.num_items
        L = N + I + J + K

        # Debug logging
        rating_tokens = sum(1 for var in ranking_data_list if not var.is_listwise)
        pairwise_tokens = sum(1 for var in ranking_data_list if var.is_listwise)
        status_counts = {0: 0, 1: 0, 2: 0}
        for var in ranking_data_list:
            if var.status in status_counts:
                status_counts[var.status] += 1
        print(
            f"\033[96m[DEBUG UnifiedEntity] Forward pass: {N} observations + {I+J+K} entities = {L} total tokens | "
            f"({rating_tokens} ratings, {pairwise_tokens} pairwise) | "
            f"status: missing={status_counts[0]}, masked={status_counts[1]}, observed={status_counts[2]}"
            f"\033[0m"
        )

        # ── Extract IDs ──
        attr_ids = torch.tensor([var.attribute_id for var in ranking_data_list], device=device)
        annot_ids = torch.tensor([var.annotator_id for var in ranking_data_list], device=device)
        item_ids = torch.tensor(
            [var.item_ids[0] if len(var.item_ids) > 0 else -1 for var in ranking_data_list],
            device=device,
        )
        status = torch.tensor([var.status for var in ranking_data_list], device=device)

        # ── Build entity embeddings ──
        attr_emb, annot_emb, item_emb = self._get_entity_embeddings(B)  # [B, I/J/K, D]

        # ── Build observation features (layer-0 composition) ──
        obs_features = self._build_observation_features(
            attr_emb, annot_emb, attr_ids, annot_ids, B, N
        )  # [B, N, D]

        # ── Build params ──
        obs_params = self._build_params(ranking_data_list, B)  # [B, N, param_dim]
        entity_params = self._build_entity_params(B, I + J + K)  # [B, I+J+K, param_dim]

        # ── Initialize item embeddings from observed data ──
        item_emb = self._init_item_embeddings(
            item_emb, obs_features, obs_params, item_ids, status, B
        )  # [B, K, D]

        # ── Build pairwise obs-to-obs indicators ──
        K_aug = self._build_K_aug(attr_ids, annot_ids, item_ids, B)  # [B, N, N, 3]

        # ── Concatenate into unified sequence ──
        # Features: [obs | attr | annot | item]
        all_features = torch.cat([obs_features, attr_emb, annot_emb, item_emb], dim=1)  # [B, L, D]
        # Params: [obs_params | entity_params]
        all_params = torch.cat([obs_params, entity_params], dim=1)  # [B, L, param_dim]

        # ── Build unified tensor [features | params] ──
        x = torch.cat([all_features, all_params], dim=-1)  # [B, L, D + param_dim]

        # ── Build key edge features and attention mask ──
        key_edge_features = self._build_key_edge_features(N, B)  # [B, L, 3]
        attn_mask = self._build_attn_mask(status, B, N)  # [B, L]

        # ── Intermediate states ──
        hidden_intermediates = []
        if return_intermediate:
            # Split back for logging
            features = x[:, :, :self.feature_dim]
            params = x[:, :, self.feature_dim:]
            hidden_intermediates.append([features[:, :N, :], params[:, :N, :]])

        # ── Run transformer blocks ──
        for block in self.blocks:
            x = block(x, key_edge_features, attn_mask, K_aug=K_aug, num_obs=N)
            if return_intermediate:
                features = x[:, :, :self.feature_dim]
                params = x[:, :, self.feature_dim:]
                hidden_intermediates.append([features[:, :N, :], params[:, :N, :]])

        # ── Extract observation tokens for prediction ──
        obs_x = x[:, :N, :]  # [B, N, D + param_dim]
        obs_features = obs_x[:, :, :self.feature_dim]  # [B, N, D]
        obs_params = obs_x[:, :, self.feature_dim:]    # [B, N, param_dim]

        # ── Prediction ──
        if self.use_prediction_head:
            obs_combined = torch.cat([obs_features, obs_params], dim=-1)  # [B, N, D+param_dim]
            rating_logits = self.rating_head(obs_combined)   # [B, N, C]
            ranking_logits = self.ranking_head(obs_combined)  # [B, N, R]
        else:
            if self.final_norm is not None:
                obs_params = self.final_norm(obs_params)
            rating_logits = self.read_prediction_logits_from_param('rating', obs_params)
            ranking_logits = self.read_prediction_logits_from_param('ranking', obs_params)

        # Temperature scaling
        logits = {
            'rating': rating_logits / self.temperature,
            'ranking': ranking_logits / self.temperature,
        }

        if return_intermediate:
            return logits, hidden_intermediates
        return logits
