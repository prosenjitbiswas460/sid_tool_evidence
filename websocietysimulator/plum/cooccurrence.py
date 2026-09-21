"""Co-occurrence pair mining for SID-v2 contrastive alignment."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from websocietysimulator.conditions.task_conditions import parse_review_timestamp


def _sequence_sort_key(review: dict, line_index: int) -> Tuple[int, float]:
    timestamp = parse_review_timestamp(review)
    if timestamp is not None:
        return (0, float(timestamp))
    return (1, float(line_index))


def load_user_item_sequences(
    review_json: str,
    source: str,
    *,
    max_users: Optional[int] = None,
    min_sequence_length: int = 2,
) -> Dict[str, List[str]]:
    """Chronological deduplicated item-id sequences per user."""
    events: Dict[str, List[Tuple[Tuple[int, float], str]]] = defaultdict(list)
    with open(review_json, "r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            review = json.loads(line)
            if review.get("source") != source:
                continue
            user_id = review.get("user_id")
            item_id = review.get("item_id")
            if not user_id or not item_id:
                continue
            events[str(user_id)].append(
                (_sequence_sort_key(review, line_index), str(item_id))
            )

    sequences: Dict[str, List[str]] = {}
    for user_id, rows in events.items():
        rows.sort(key=lambda row: row[0])
        seen: Set[str] = set()
        seq: List[str] = []
        for _, item_id in rows:
            if item_id in seen:
                continue
            seen.add(item_id)
            seq.append(item_id)
        if len(seq) >= min_sequence_length:
            sequences[user_id] = seq
        if max_users is not None and len(sequences) >= max_users:
            break
    return sequences


def mine_cooccurrence_pairs(
    sequences: Dict[str, List[str]],
    *,
    window: int = 3,
    max_pairs: int = 500_000,
    seed: int = 42,
    holdout_last: int = 0,
) -> List[Tuple[str, str]]:
    """Positive item pairs within a sliding window along user trajectories.

    ``holdout_last`` trims the final N items from each user's trajectory before
    mining pairs. Set it to >=1 to prevent the benchmark's held-out target
    interaction from shaping the SID space (leave-last-out; avoids train/eval
    co-occurrence leakage).
    """
    rng = random.Random(seed)
    pairs: Set[Tuple[str, str]] = set()
    for seq in sequences.values():
        if holdout_last > 0:
            seq = seq[: len(seq) - holdout_last]
            if len(seq) < 2:
                continue
        for start in range(len(seq)):
            anchor = seq[start]
            end = min(len(seq), start + window + 1)
            for offset in range(start + 1, end):
                partner = seq[offset]
                if anchor == partner:
                    continue
                pair = (anchor, partner) if anchor < partner else (partner, anchor)
                pairs.add(pair)
                if len(pairs) >= max_pairs:
                    return list(pairs)
    pair_list = list(pairs)
    rng.shuffle(pair_list)
    return pair_list[:max_pairs]


def item_id_to_index(item_ids: Sequence[str]) -> Dict[str, int]:
    return {item_id: index for index, item_id in enumerate(item_ids)}


def pair_indices(
    pairs: Sequence[Tuple[str, str]],
    lookup: Dict[str, int],
) -> List[Tuple[int, int]]:
    indexed: List[Tuple[int, int]] = []
    for left, right in pairs:
        if left in lookup and right in lookup:
            indexed.append((lookup[left], lookup[right]))
    return indexed


def batch_cooccurrence(
    indexed_pairs: Sequence[Tuple[int, int]],
    num_items: int,
    batch_size: int,
    *,
    seed: int = 42,
) -> Iterator[Tuple[List[int], List[int], List[int]]]:
    """Yield (anchor_idx, positive_idx, negative_idx) batches."""
    rng = random.Random(seed)
    if not indexed_pairs:
        return
    while True:
        anchors: List[int] = []
        positives: List[int] = []
        negatives: List[int] = []
        for _ in range(batch_size):
            anchor_idx, pos_idx = rng.choice(indexed_pairs)
            neg_idx = rng.randrange(num_items)
            while neg_idx == anchor_idx or neg_idx == pos_idx:
                neg_idx = rng.randrange(num_items)
            anchors.append(anchor_idx)
            positives.append(pos_idx)
            negatives.append(neg_idx)
        yield anchors, positives, negatives
