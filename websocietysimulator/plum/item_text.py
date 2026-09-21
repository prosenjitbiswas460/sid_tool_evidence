"""Item text helpers shared by SID-v2 and PLUM data builders."""

from __future__ import annotations

from typing import List, Optional


def item_to_text(item: dict) -> str:
    source = item.get("source", "")

    if source == "amazon":
        parts = [
            str(item.get("title", "") or ""),
            str(item.get("description", "") or ""),
            str(item.get("features", "") or ""),
        ]
    elif source == "yelp":
        parts = [
            str(item.get("name", "") or ""),
            str(item.get("categories", "") or ""),
            str(item.get("attributes", "") or ""),
        ]
    elif source == "goodreads":
        parts = [
            str(item.get("title", "") or ""),
            str(item.get("description", "") or ""),
            str(item.get("authors", "") or ""),
        ]
    else:
        parts = [
            str(item.get("title", item.get("name", "")) or ""),
            str(item.get("description", "") or ""),
        ]

    text = " ".join(part.strip() for part in parts if part and str(part).strip())
    return text if text else str(item.get("item_id", ""))


def load_items_jsonl(
    item_json: str,
    source: Optional[str] = None,
    max_items: Optional[int] = None,
) -> List[dict]:
    items: List[dict] = []
    with open(item_json, "r", encoding="utf-8") as handle:
        for line in handle:
            item = __import__("json").loads(line)
            if source is not None and item.get("source") != source:
                continue
            items.append(item)
            if max_items is not None and len(items) >= max_items:
                break
    if not items:
        raise ValueError(f"No items loaded from {item_json} (source={source!r})")
    return items
