"""
Semantic ID tools for generative-retrieval-aware agents.

Loads artifacts produced by scripts/build_semantic_ids.py and exposes lookup,
history tokenization, prefix search, and candidate ranking utilities.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .interaction_tool import InteractionTool
from .cache_interaction_tool import CacheInteractionTool

logger = logging.getLogger("websocietysimulator")

InteractionToolType = Union[InteractionTool, CacheInteractionTool]


def _normalize_sid(sid: Sequence[int]) -> List[int]:
    return [int(value) for value in sid]


def _sid_key(sid: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in sid)


class SemanticIDIndex:
    """In-memory index for one dataset source (amazon, yelp, ...)."""

    def __init__(self, artifact_dir: str, source: str):
        self.artifact_dir = artifact_dir
        self.source = source

        metadata_path = os.path.join(artifact_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        else:
            self.metadata = {}

        self.item_id_to_sid: Dict[str, torch.Tensor] = torch.load(
            os.path.join(artifact_dir, "item_id_to_sid.pt"),
            map_location="cpu",
        )
        sid_map_path = os.path.join(artifact_dir, "sid_map.pt")
        self.sid_map: Optional[torch.Tensor] = (
            torch.load(sid_map_path, map_location="cpu")
            if os.path.exists(sid_map_path)
            else None
        )

        sid_json_path = os.path.join(artifact_dir, "sid_to_item_ids.json")
        if os.path.exists(sid_json_path):
            with open(sid_json_path, "r", encoding="utf-8") as handle:
                self.sid_to_item_ids: Dict[str, List[str]] = json.load(handle)
        else:
            self.sid_to_item_ids = defaultdict(list)
            for item_id, sid_tensor in self.item_id_to_sid.items():
                self.sid_to_item_ids[_sid_key(sid_tensor.tolist())].append(item_id)

        self.num_hierarchies = int(
            self.metadata.get(
                "num_hierarchies",
                next(iter(self.item_id_to_sid.values())).numel() - 1,
            )
        )
        self.num_hierarchies_with_dedup = next(
            iter(self.item_id_to_sid.values())
        ).numel()

        self._prefix_to_items: Dict[Tuple[int, ...], List[str]] = defaultdict(list)
        for item_id, sid_tensor in self.item_id_to_sid.items():
            sid = _normalize_sid(sid_tensor.tolist())
            for depth in range(1, len(sid) + 1):
                self._prefix_to_items[tuple(sid[:depth])].append(item_id)

        embeddings_path = os.path.join(artifact_dir, "merged_predictions_tensor.pt")
        ordered_ids_path = os.path.join(artifact_dir, "ordered_item_ids.pt")
        self.item_id_to_embedding: Dict[str, torch.Tensor] = {}
        if os.path.exists(embeddings_path) and os.path.exists(ordered_ids_path):
            embeddings = torch.load(embeddings_path, map_location="cpu")
            ordered_item_ids = torch.load(ordered_ids_path, map_location="cpu")
            if isinstance(ordered_item_ids, torch.Tensor):
                ordered_item_ids = ordered_item_ids.tolist()
            for index, item_id in enumerate(ordered_item_ids):
                self.item_id_to_embedding[str(item_id)] = embeddings[index]

        logger.info(
            "Loaded SemanticIDIndex for %s: %d items, %d hierarchies (+dedup=%d)",
            source,
            len(self.item_id_to_sid),
            self.num_hierarchies,
            self.num_hierarchies_with_dedup,
        )

    def has_item(self, item_id: str) -> bool:
        return item_id in self.item_id_to_sid

    def get_sid(self, item_id: str) -> Optional[List[int]]:
        sid_tensor = self.item_id_to_sid.get(item_id)
        if sid_tensor is None:
            return None
        return _normalize_sid(sid_tensor.tolist())

    def get_items_by_sid(self, sid: Sequence[int], exact: bool = True) -> List[str]:
        sid = _normalize_sid(sid)
        if exact:
            return list(self.sid_to_item_ids.get(_sid_key(sid), []))
        return self.get_items_by_sid_prefix(sid, top_k=None)

    def get_items_by_sid_prefix(
        self,
        prefix: Sequence[int],
        top_k: Optional[int] = 50,
    ) -> List[str]:
        prefix_tuple = tuple(_normalize_sid(prefix))
        items = list(dict.fromkeys(self._prefix_to_items.get(prefix_tuple, [])))
        if top_k is not None:
            return items[:top_k]
        return items

    def validate_sid_prefix(self, prefix: Sequence[int]) -> Dict[str, Any]:
        prefix = _normalize_sid(prefix)
        items = self.get_items_by_sid_prefix(prefix, top_k=5)
        return {
            "valid": len(items) > 0,
            "prefix": prefix,
            "num_extensions": len(self.get_items_by_sid_prefix(prefix, top_k=None)),
            "example_items": items[:5],
        }

    def get_sid_neighbors(self, item_id: str, level: int = 1) -> List[str]:
        sid = self.get_sid(item_id)
        if sid is None:
            return []
        level = max(1, min(level, len(sid)))
        prefix = sid[:level]
        neighbors = [
            candidate
            for candidate in self.get_items_by_sid_prefix(prefix, top_k=None)
            if candidate != item_id
        ]
        return neighbors

    def sid_similarity(self, sid_a: Sequence[int], sid_b: Sequence[int]) -> float:
        sid_a = _normalize_sid(sid_a)
        sid_b = _normalize_sid(sid_b)
        content_levels = min(self.num_hierarchies, len(sid_a), len(sid_b))
        if content_levels == 0:
            return 0.0
        matches = sum(1 for index in range(content_levels) if sid_a[index] == sid_b[index])
        return matches / content_levels

    def embedding_similarity(self, item_id_a: str, item_id_b: str) -> Optional[float]:
        embedding_a = self.item_id_to_embedding.get(item_id_a)
        embedding_b = self.item_id_to_embedding.get(item_id_b)
        if embedding_a is None or embedding_b is None:
            return None
        similarity = torch.nn.functional.cosine_similarity(
            embedding_a.unsqueeze(0),
            embedding_b.unsqueeze(0),
        ).item()
        return float(similarity)


class SemanticIDTool:
    """
    Multi-source semantic ID tool layer for AgentRecBench agents.

    Parameters
    ----------
    artifact_dirs:
        Mapping like {"amazon": "artifacts/amazon", "yelp": "artifacts/yelp"}.
    interaction_tool:
        Existing InteractionTool used to fetch user history and item metadata.
    default_source:
        Fallback source when item metadata does not include a ``source`` field.
    """

    def __init__(
        self,
        artifact_dirs: Dict[str, str],
        interaction_tool: Optional[InteractionToolType] = None,
        default_source: str = "amazon",
    ):
        self.interaction_tool = interaction_tool
        self.default_source = default_source
        self.indices: Dict[str, SemanticIDIndex] = {}

        for source, artifact_dir in artifact_dirs.items():
            if not os.path.isdir(artifact_dir):
                raise FileNotFoundError(
                    f"Semantic ID artifact directory not found for {source}: {artifact_dir}"
                )
            self.indices[source] = SemanticIDIndex(artifact_dir=artifact_dir, source=source)

    def set_interaction_tool(self, interaction_tool: InteractionToolType) -> None:
        self.interaction_tool = interaction_tool

    def set_default_source(self, source: str) -> None:
        if source not in self.indices:
            raise ValueError(
                f"Unknown source {source!r}. Loaded sources: {list(self.indices.keys())}"
            )
        self.default_source = source

    def available_sources(self) -> List[str]:
        return list(self.indices.keys())

    def _resolve_source(
        self,
        item_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> str:
        if source is not None:
            if source not in self.indices:
                raise ValueError(
                    f"Unknown source {source!r}. Loaded sources: {list(self.indices.keys())}"
                )
            return source

        if item_id and self.interaction_tool is not None:
            item = self.interaction_tool.get_item(item_id=item_id)
            if item and item.get("source") in self.indices:
                return item["source"]

        if self.default_source in self.indices:
            return self.default_source

        return next(iter(self.indices.keys()))

    def _index(
        self,
        item_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> SemanticIDIndex:
        resolved = self._resolve_source(item_id=item_id, source=source)
        return self.indices[resolved]

    def get_sid(self, item_id: str, source: Optional[str] = None) -> Optional[List[int]]:
        """Map an item ID to its hierarchical semantic ID."""
        return self._index(item_id=item_id, source=source).get_sid(item_id)

    def get_items_by_sid(
        self,
        sid: Sequence[int],
        exact: bool = True,
        source: Optional[str] = None,
    ) -> List[str]:
        """Map a semantic ID back to one or more item IDs."""
        resolved = self._resolve_source(source=source)
        return self.indices[resolved].get_items_by_sid(sid, exact=exact)

    def get_items_by_sid_prefix(
        self,
        prefix: Sequence[int],
        top_k: int = 50,
        source: Optional[str] = None,
    ) -> List[str]:
        """Retrieve candidate items sharing a semantic ID prefix."""
        resolved = self._resolve_source(source=source)
        return self.indices[resolved].get_items_by_sid_prefix(prefix, top_k=top_k)

    def validate_sid_prefix(
        self,
        prefix: Sequence[int],
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check whether a partial semantic ID prefix maps to any catalog item."""
        resolved = self._resolve_source(source=source)
        return self.indices[resolved].validate_sid_prefix(prefix)

    def get_sid_neighbors(
        self,
        item_id: str,
        level: int = 1,
        source: Optional[str] = None,
    ) -> List[str]:
        """Return items that share the first ``level`` SID digits with ``item_id``."""
        return self._index(item_id=item_id, source=source).get_sid_neighbors(
            item_id=item_id,
            level=level,
        )

    def decode_sid(
        self,
        sid: Sequence[int],
        source: Optional[str] = None,
        include_item_names: bool = True,
    ) -> Dict[str, Any]:
        """Return a structured, LLM-friendly description of a semantic ID."""
        resolved = self._resolve_source(source=source)
        index = self.indices[resolved]
        sid = _normalize_sid(sid)
        exact_items = index.get_items_by_sid(sid, exact=True)
        prefix_info = index.validate_sid_prefix(sid)

        item_summaries = []
        if include_item_names and self.interaction_tool is not None:
            for item_id in exact_items[:5]:
                item = self.interaction_tool.get_item(item_id=item_id)
                if item is None:
                    continue
                name = item.get("title") or item.get("name") or item_id
                item_summaries.append({"item_id": item_id, "name": name})

        return {
            "source": resolved,
            "sid": sid,
            "num_hierarchies": index.num_hierarchies_with_dedup,
            "exact_item_ids": exact_items,
            "exact_match_count": len(exact_items),
            "prefix_valid": prefix_info["valid"],
            "prefix_extension_count": prefix_info["num_extensions"],
            "example_items": item_summaries or prefix_info["example_items"],
        }

    def encode_user_history(
        self,
        user_id: str,
        max_items: Optional[int] = 20,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert a user's review history into flattened semantic ID tokens.

        Returns
        -------
        dict with keys:
            user_id, source, item_ids, sids, flat_sid_tokens, num_hierarchies
        """
        if self.interaction_tool is None:
            raise RuntimeError(
                "encode_user_history requires InteractionTool. "
                "Call set_interaction_tool() first."
            )

        reviews = self.interaction_tool.get_reviews(user_id=user_id)
        reviews = sorted(
            reviews,
            key=lambda review: review.get("timestamp", 0),
        )

        item_ids: List[str] = []
        seen = set()
        for review in reviews:
            item_id = review.get("item_id")
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            item_ids.append(item_id)
            if max_items is not None and len(item_ids) >= max_items:
                break

        resolved_source = source
        if resolved_source is None and item_ids:
            resolved_source = self._resolve_source(item_id=item_ids[0])
        elif resolved_source is None:
            resolved_source = self.default_source

        index = self.indices[resolved_source]
        sids: List[List[int]] = []
        flat_sid_tokens: List[int] = []
        missing_items: List[str] = []

        for item_id in item_ids:
            sid = index.get_sid(item_id)
            if sid is None:
                missing_items.append(item_id)
                continue
            sids.append(sid)
            flat_sid_tokens.extend(sid)

        return {
            "user_id": user_id,
            "source": resolved_source,
            "item_ids": item_ids,
            "sids": sids,
            "flat_sid_tokens": flat_sid_tokens,
            "num_hierarchies": index.num_hierarchies_with_dedup,
            "missing_items": missing_items,
        }

    def rank_candidates_by_sid(
        self,
        user_id: str,
        candidate_item_ids: Sequence[str],
        source: Optional[str] = None,
        max_history_items: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Score and rank candidate items using semantic ID similarity to user history.

        Higher score is better. Uses hierarchy overlap plus optional embedding cosine
        similarity when embeddings are available in the artifact bundle.
        """
        history = self.encode_user_history(
            user_id=user_id,
            max_items=max_history_items,
            source=source,
        )
        resolved_source = history["source"]
        index = self.indices[resolved_source]
        history_sids = history["sids"]

        if not history_sids:
            return [(item_id, 0.0) for item_id in candidate_item_ids]

        ranked: List[Tuple[str, float]] = []
        for candidate_id in candidate_item_ids:
            candidate_sid = index.get_sid(candidate_id)
            if candidate_sid is None:
                ranked.append((candidate_id, 0.0))
                continue

            sid_scores = [
                index.sid_similarity(candidate_sid, history_sid)
                for history_sid in history_sids
            ]
            score = max(sid_scores)

            embedding_scores = []
            for history_item_id in history["item_ids"]:
                embedding_score = index.embedding_similarity(candidate_id, history_item_id)
                if embedding_score is not None:
                    embedding_scores.append(embedding_score)
            if embedding_scores:
                score = 0.7 * score + 0.3 * max(embedding_scores)

            ranked.append((candidate_id, float(score)))

        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked
