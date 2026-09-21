"""
Agentic generative-retrieval tools over metadata semantic ID artifacts.

These tools implement prefix search, collision resolution, and candidate scoring
without requiring the LLM to one-shot generate full hierarchical SID strings.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .semantic_id_tool import SemanticIDTool


def _content_prefix(sid: Sequence[int], depth: int, num_hierarchies: int) -> Tuple[int, ...]:
    depth = max(1, min(depth, num_hierarchies, len(sid)))
    return tuple(int(value) for value in sid[:depth])


@dataclass
class PrefixHit:
    prefix: List[int]
    depth: int
    matched_candidate_ids: List[str]
    extension_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CollisionNote:
    sid: List[int]
    item_ids: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalToolTrace:
    """Structured output from the tool-augmented GR retrieval protocol."""

    source: str
    history_item_ids: List[str]
    selected_prefixes: List[List[int]]
    prefix_hits: List[PrefixHit] = field(default_factory=list)
    collisions: List[CollisionNote] = field(default_factory=list)
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    tool_order: List[str] = field(default_factory=list)
    prefix_matched_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["prefix_hits"] = [hit.to_dict() for hit in self.prefix_hits]
        payload["collisions"] = [note.to_dict() for note in self.collisions]
        return payload


class GRRetrievalToolkit:
    """
    Tool layer for agentic generative retrieval on top of SemanticIDTool.

    Typical protocol:
      1. derive history SID prefixes
      2. search_by_sid_prefix within the task candidate pool
      3. resolve_sid / note collisions
      4. score_candidates for all task items
    """

    def __init__(self, semantic_id_tool: SemanticIDTool):
        self.semantic_id_tool = semantic_id_tool

    def derive_history_prefixes(
        self,
        user_id: str,
        source: Optional[str] = None,
        prefix_depth: int = 2,
        max_prefixes: int = 3,
    ) -> Tuple[List[List[int]], List[str], str]:
        history = self.semantic_id_tool.encode_user_history(
            user_id=user_id,
            source=source,
        )
        resolved_source = history["source"]
        index = self.semantic_id_tool.indices[resolved_source]
        depth = max(1, min(prefix_depth, index.num_hierarchies))

        prefix_counts: Counter[Tuple[int, ...]] = Counter()
        for sid in history["sids"]:
            if sid:
                prefix_counts[_content_prefix(sid, depth, index.num_hierarchies)] += 1

        selected = [
            list(prefix)
            for prefix, _count in prefix_counts.most_common(max_prefixes)
        ]
        if not selected and history["sids"]:
            fallback_sid = history["sids"][-1]
            selected = [list(_content_prefix(fallback_sid, depth, index.num_hierarchies))]

        return selected, history["item_ids"], resolved_source

    def search_by_sid_prefix(
        self,
        prefix: Sequence[int],
        candidate_item_ids: Sequence[str],
        source: Optional[str] = None,
    ) -> List[str]:
        """Return candidate IDs whose SID shares the given content prefix."""
        index = self.semantic_id_tool._index(source=source)
        prefix_tuple = _content_prefix(prefix, len(prefix), index.num_hierarchies)
        matched: List[str] = []
        for item_id in candidate_item_ids:
            sid = index.get_sid(str(item_id))
            if sid is None:
                continue
            if _content_prefix(sid, len(prefix_tuple), index.num_hierarchies) == prefix_tuple:
                matched.append(str(item_id))
        return matched

    def resolve_sid(
        self,
        sid: Sequence[int],
        candidate_item_ids: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Decode a SID and optionally restrict to a candidate pool."""
        decoded = self.semantic_id_tool.decode_sid(
            sid=sid,
            source=source,
            include_item_names=False,
        )
        item_ids = [str(item_id) for item_id in decoded.get("exact_item_ids", [])]
        if candidate_item_ids is not None:
            pool = {str(item_id) for item_id in candidate_item_ids}
            item_ids = [item_id for item_id in item_ids if item_id in pool]
        return {
            "sid": list(sid),
            "item_ids": item_ids,
            "collision_count": len(item_ids),
            "prefix_extension_count": decoded.get("prefix_extension_count", 0),
        }

    def score_candidates(
        self,
        user_id: str,
        candidate_item_ids: Sequence[str],
        source: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        return self.semantic_id_tool.rank_candidates_by_sid(
            user_id=user_id,
            candidate_item_ids=candidate_item_ids,
            source=source,
        )

    def run_retrieval_protocol(
        self,
        user_id: str,
        candidate_item_ids: Sequence[str],
        source: Optional[str] = None,
        prefix_depth: int = 2,
        max_prefixes: int = 3,
    ) -> RetrievalToolTrace:
        candidates = [str(item_id) for item_id in candidate_item_ids]
        selected_prefixes, history_item_ids, resolved_source = self.derive_history_prefixes(
            user_id=user_id,
            source=source,
            prefix_depth=prefix_depth,
            max_prefixes=max_prefixes,
        )

        prefix_hits: List[PrefixHit] = []
        prefix_matched: Set[str] = set()
        index = self.semantic_id_tool.indices[resolved_source]

        for prefix in selected_prefixes:
            matched = self.search_by_sid_prefix(
                prefix=prefix,
                candidate_item_ids=candidates,
                source=resolved_source,
            )
            prefix_hits.append(
                PrefixHit(
                    prefix=list(prefix),
                    depth=len(prefix),
                    matched_candidate_ids=matched,
                    extension_count=len(
                        index.get_items_by_sid_prefix(prefix, top_k=None)
                    ),
                )
            )
            prefix_matched.update(matched)

        collisions: List[CollisionNote] = []
        seen_sids: Set[str] = set()
        for item_id in prefix_matched:
            sid = index.get_sid(item_id)
            if sid is None:
                continue
            sid_key = ",".join(str(value) for value in sid)
            if sid_key in seen_sids:
                continue
            seen_sids.add(sid_key)
            resolved = self.resolve_sid(
                sid=sid,
                candidate_item_ids=candidates,
                source=resolved_source,
            )
            if resolved["collision_count"] > 1:
                collisions.append(
                    CollisionNote(
                        sid=list(sid),
                        item_ids=list(resolved["item_ids"]),
                    )
                )

        ranked = self.score_candidates(
            user_id=user_id,
            candidate_item_ids=candidates,
            source=resolved_source,
        )
        candidate_scores = {item_id: score for item_id, score in ranked}
        tool_order = [item_id for item_id, _score in ranked]

        return RetrievalToolTrace(
            source=resolved_source,
            history_item_ids=history_item_ids,
            selected_prefixes=selected_prefixes,
            prefix_hits=prefix_hits,
            collisions=collisions,
            candidate_scores=candidate_scores,
            tool_order=tool_order,
            prefix_matched_ids=sorted(prefix_matched),
        )

    def format_trace_for_prompt(self, trace: RetrievalToolTrace) -> str:
        lines = [
            "Tool-augmented generative retrieval trace:",
            f"- source: {trace.source}",
            f"- history items: {trace.history_item_ids}",
            f"- selected SID prefixes: {trace.selected_prefixes}",
        ]
        for hit in trace.prefix_hits:
            lines.append(
                "- search_sid_prefix "
                f"{hit.prefix} -> {len(hit.matched_candidate_ids)} candidate matches "
                f"(catalog extensions={hit.extension_count}): {hit.matched_candidate_ids[:8]}"
            )
        if trace.collisions:
            lines.append("- resolve_sid collisions in candidate pool:")
            for note in trace.collisions[:5]:
                lines.append(f"  SID {note.sid} -> {note.item_ids}")
        lines.append(
            "- score_candidates top matches: "
            + str(
                [
                    {"item_id": item_id, "tool_score": round(trace.candidate_scores[item_id], 4)}
                    for item_id in trace.tool_order[:8]
                ]
            )
        )
        return "\n".join(lines)
