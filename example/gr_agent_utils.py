"""Shared helpers for GR-aware baseline agents."""

from __future__ import annotations

import ast
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

ITEM_FIELD_KEYS = [
    "item_id",
    "name",
    "title",
    "stars",
    "average_rating",
    "review_count",
    "rating_number",
    "attributes",
    "description",
    "categories",
    "type",
    "source",
]


def truncate_text(text: str, max_tokens: int = 12000) -> str:
    if not text or tiktoken is None:
        return text
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def filter_item_fields(item: Optional[dict]) -> dict:
    if not item:
        return {}
    return {key: item[key] for key in ITEM_FIELD_KEYS if key in item and item[key] is not None}


def resolve_sid_source(
    interaction_tool,
    candidate_list: Sequence[str],
    default_source: str = "amazon",
) -> str:
    for item_id in candidate_list:
        item = interaction_tool.get_item(item_id=item_id)
        if item and item.get("source"):
            return item["source"]
    return default_source


def parse_ranked_item_list(text: str, valid_ids: Set[str]) -> List[str]:
    """Extract a ranking of candidate IDs from model text.

    Primary path: a Python list literal ``[...]`` (what the prompt asks for).
    Fallbacks exist because some backbones (e.g. Llama) often return numbered
    lists / prose that still mention valid IDs in rank order. Empty / garbage
    text still returns ``[]`` so the caller can fall back to SID / candidate order.
    """
    if not text or not valid_ids:
        return []

    def _filter(seq) -> List[str]:
        ranked: List[str] = []
        seen: Set[str] = set()
        for item_id in seq:
            item_id = str(item_id).strip()
            if item_id in valid_ids and item_id not in seen:
                ranked.append(item_id)
                seen.add(item_id)
        return ranked

    # 1) Non-nested bracket lists (preferred; avoids greedy DOTALL eating the whole reply).
    for match in re.finditer(r"\[[^\[\]]*\]", text):
        try:
            parsed = ast.literal_eval(match.group())
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, list):
            ranked = _filter(parsed)
            if ranked:
                return ranked

    # 2) Greedy bracket span (legacy path for multi-line lists with nested quotes).
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = ast.literal_eval(match.group())
            if isinstance(parsed, list):
                ranked = _filter(parsed)
                if ranked:
                    return ranked
        except (SyntaxError, ValueError):
            pass

    # 3) Quoted tokens that are valid IDs, in appearance order.
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
    ranked = _filter(quoted)
    if len(ranked) >= 3:
        return ranked

    # 4) Any valid ID substring in appearance order (numbered lists / prose).
    #    Longer IDs first so short IDs don't steal prefixes of longer ones.
    positions: List[Tuple[int, str]] = []
    for item_id in sorted(valid_ids, key=len, reverse=True):
        start = 0
        while True:
            pos = text.find(item_id, start)
            if pos < 0:
                break
            positions.append((pos, item_id))
            start = pos + len(item_id)
            break  # first occurrence only (rank order = first mention)
    if positions:
        positions.sort(key=lambda x: x[0])
        ranked = _filter(vid for _, vid in positions)
        if ranked:
            return ranked
    return []


def merge_ranking(
    primary: Sequence[str],
    fallback: Sequence[str],
    valid_ids: Set[str],
) -> List[str]:
    merged: List[str] = []
    seen: Set[str] = set()
    for sequence in (primary, fallback):
        for item_id in sequence:
            if item_id in valid_ids and item_id not in seen:
                merged.append(item_id)
                seen.add(item_id)
    for item_id in fallback:
        if item_id in valid_ids and item_id not in seen:
            merged.append(item_id)
            seen.add(item_id)
    return merged


def complete_ranking(
    primary: Sequence[str],
    candidate_order: Sequence[str],
    valid_ids: Set[str],
) -> List[str]:
    """Fill a partial ranking using the original candidate order (no SID merge)."""
    ranked: List[str] = []
    seen: Set[str] = set()
    for item_id in primary:
        if item_id in valid_ids and item_id not in seen:
            ranked.append(item_id)
            seen.add(item_id)
    for item_id in candidate_order:
        if item_id in valid_ids and item_id not in seen:
            ranked.append(item_id)
            seen.add(item_id)
    return ranked


def build_sid_candidate_summaries(
    semantic_id_tool,
    candidate_ids: Sequence[str],
    sid_scores: Dict[str, float],
    source: Optional[str] = None,
    top_n: Optional[int] = None,
) -> List[dict]:
    summaries: List[dict] = []
    ordered_ids = sorted(
        candidate_ids,
        key=lambda item_id: sid_scores.get(item_id, 0.0),
        reverse=True,
    )
    if top_n is not None:
        ordered_ids = ordered_ids[:top_n]

    for item_id in ordered_ids:
        sid = semantic_id_tool.get_sid(item_id, source=source)
        summary = {
            "item_id": item_id,
            "sid": sid,
            "sid_score": sid_scores.get(item_id, 0.0),
        }
        if sid is not None:
            decoded = semantic_id_tool.decode_sid(
                sid,
                source=source,
                include_item_names=False,
            )
            summary["prefix_extension_count"] = decoded["prefix_extension_count"]
        summaries.append(summary)
    return summaries
