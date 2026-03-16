from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch

from imputer.data import RankingData

from .types import EntityType


@dataclass
class Token:
    """
    Single sequence element for the Entity Marformer.

    Attributes:
        type_name: Name of the EntityType.
        entity_id: Index within that type (e.g., annotator id, attribute id).
        status: 0=missing, 1=masked, 2=observed.
        raw_data: Type-specific payload (e.g., rating_value, ranking_order).
    """

    type_name: str
    entity_id: int
    status: int
    raw_data: Dict[str, Any] | None = None


@dataclass
class Relationship:
    """
    Labeled edge type between source and target types.

    Example labels for domain 3:
        - ATTR
        - ATTR_INV
        - ANNOT
        - ANNOT_INV
        - ITEM
        - ITEM_INV
    """

    name: str
    source_type: str
    target_type: str
    inverse: str | None = None


class EntityGraph:
    """
    Typed token graph + labeled edges for relational attention.

    This is intentionally simple for the MVP: we assume a single instance
    (batch size 1) and store a flat list of tokens with integer indices.
    """

    def __init__(
        self,
        types: Dict[str, EntityType],
        relationships: List[Relationship],
        tokens: List[Token],
        edges: List[Tuple[int, int, str]],  # specify relationships externally
    ):
        self.types = types
        self.relationships = relationships
        self.tokens = tokens
        self.edges = edges

        self._rel_index: Dict[str, int] = {rel.name: i for i, rel in enumerate(self.relationships)}

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def num_relationships(self) -> int:
        return len(self.relationships)

    def build_edge_masks(self, device: torch.device) -> torch.Tensor:
        """
        Build binary edge mask tensor of shape [L, L, R].

        TODO: think of ways we can tensorize this. It might be faster to compute in parallelbased on some rules.
        mask[q, k, r] = 1 if there is an edge of relationship r from q -> k.
        """
        L = self.num_tokens
        R = self.num_relationships
        edge_mask = torch.zeros(L, L, R, device=device)
        for src, tgt, rel_name in self.edges:
            r_idx = self._rel_index.get(rel_name, None)
            if r_idx is None:
                continue
            if 0 <= src < L and 0 <= tgt < L:
                edge_mask[src, tgt, r_idx] = 1.0
        return edge_mask


def variable_list_to_entity_graph(
    ranking_vars: List[RankingData],
    types: Dict[str, EntityType],
) -> EntityGraph:
    """
    Convert a list of RankingData variables into a single EntityGraph.

    For domain 3, we create:
      - One token per rating / pairwise ranking variable.
      - One token per attribute, annotator, and item entity.
      - Edges of types ATTR / ATTR_INV / ANNOT / ANNOT_INV / ITEM / ITEM_INV.
    """
    tokens: List[Token] = []
    edges: List[Tuple[int, int, str]] = []

    num_attributes = types["attribute"].variation.num_entities
    num_annotators = types["annotator"].variation.num_entities
    num_items = types["item"].variation.num_entities

    # 1) Variable tokens (ratings + pairwise rankings)
    for var in ranking_vars:
        type_name = "ranking_pairwise" if var.is_listwise else "rating"
        raw_data: Dict[str, Any] = {
            "is_missing": var.is_missing,
            "is_masked": var.is_masked,
            # Store obs identity for K_aug pointer mechanism
            "attribute_id": var.attribute_id,
            "annotator_id": var.annotator_id,
            "item_ids": list(var.item_ids),
        }
        if var.is_listwise:
            raw_data["ranking_order"] = list(var.ranking_order or [])
        else:
            raw_data["rating_value"] = var.rating_value
            raw_data["rating_dist"] = var.rating_dist

        tokens.append(
            Token(
                type_name=type_name,
                entity_id=-1,
                status=var.status,
                raw_data=raw_data,
            )
        )

    num_variable_tokens = len(tokens)

    # 2) Attribute entity tokens
    attr_token_start = len(tokens)
    for a in range(num_attributes):
        tokens.append(Token(type_name="attribute", entity_id=a, status=2, raw_data=None))

    # 3) Annotator entity tokens
    annot_token_start = len(tokens)
    for j in range(num_annotators):
        tokens.append(Token(type_name="annotator", entity_id=j, status=2, raw_data=None))

    # 4) Item entity tokens
    item_token_start = len(tokens)
    for k in range(num_items):
        tokens.append(Token(type_name="item", entity_id=k, status=2, raw_data=None))

    # 5) Edges
    #    For each variable token v with (i, j, k):
    #      v --ATTR--> attribute_i, attribute_i --ATTR_INV--> v
    #      v --ANNOT--> annotator_j, annotator_j --ANNOT_INV--> v
    #      v --ITEM--> item_k, item_k --ITEM_INV--> v
    for idx, var in enumerate(ranking_vars):
        attr_id = var.attribute_id
        annot_id = var.annotator_id
        # Ratings: one item; rankings: two items; we connect to both items.
        item_ids = var.item_ids

        # Attribute edges
        if 0 <= attr_id < num_attributes:
            attr_token_idx = attr_token_start + attr_id
            edges.append((idx, attr_token_idx, "ATTR"))
            edges.append((attr_token_idx, idx, "ATTR_INV"))

        # Annotator edges
        if 0 <= annot_id < num_annotators:
            annot_token_idx = annot_token_start + annot_id
            edges.append((idx, annot_token_idx, "ANNOT"))
            edges.append((annot_token_idx, idx, "ANNOT_INV"))

        # Item edges
        for item_id in item_ids:
            if 0 <= item_id < num_items:
                item_token_idx = item_token_start + item_id
                edges.append((idx, item_token_idx, "ITEM"))
                edges.append((item_token_idx, idx, "ITEM_INV"))

    relationships = [
        Relationship(name="ATTR", source_type="rating_or_ranking", target_type="attribute", inverse="ATTR_INV"),
        Relationship(name="ATTR_INV", source_type="attribute", target_type="rating_or_ranking", inverse="ATTR"),
        Relationship(name="ANNOT", source_type="rating_or_ranking", target_type="annotator", inverse="ANNOT_INV"),
        Relationship(name="ANNOT_INV", source_type="annotator", target_type="rating_or_ranking", inverse="ANNOT"),
        Relationship(name="ITEM", source_type="rating_or_ranking", target_type="item", inverse="ITEM_INV"),
        Relationship(name="ITEM_INV", source_type="item", target_type="rating_or_ranking", inverse="ITEM"),
    ]

    return EntityGraph(types=types, relationships=relationships, tokens=tokens, edges=edges)

