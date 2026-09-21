"""Build SASRec training data from processed review.json."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..conditions.task_conditions import parse_review_timestamp


@dataclass
class SASRecDataBundle:
    user_train: Dict[str, List[int]]
    user_valid: Dict[str, List[int]]
    user_test: Dict[str, List[int]]
    item_id_to_idx: Dict[str, int]
    idx_to_item_id: Dict[int, str]
    num_items: int
    num_users: int

    def to_metadata(self) -> dict:
        return {
            "item_id_to_idx": self.item_id_to_idx,
            "idx_to_item_id": {str(k): v for k, v in self.idx_to_item_id.items()},
            "num_items": self.num_items,
            "num_users": self.num_users,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "SASRecDataBundle":
        item_id_to_idx = metadata["item_id_to_idx"]
        idx_to_item_id = {int(k): v for k, v in metadata["idx_to_item_id"].items()}
        return cls(
            user_train={},
            user_valid={},
            user_test={},
            item_id_to_idx=item_id_to_idx,
            idx_to_item_id=idx_to_item_id,
            num_items=int(metadata["num_items"]),
            num_users=int(metadata.get("num_users", 0)),
        )


def load_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_eval_exclusion_pairs(task_dir: str, groundtruth_dir: str) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    if not os.path.isdir(task_dir) or not os.path.isdir(groundtruth_dir):
        return pairs
    for filename in os.listdir(task_dir):
        if not filename.startswith("task_") or not filename.endswith(".json"):
            continue
        task_path = os.path.join(task_dir, filename)
        gt_path = os.path.join(groundtruth_dir, filename.replace("task_", "groundtruth_"))
        if not os.path.isfile(gt_path):
            continue
        with open(task_path, "r", encoding="utf-8") as handle:
            task = json.load(handle)
        with open(gt_path, "r", encoding="utf-8") as handle:
            groundtruth = json.load(handle)
        if task.get("type") != "recommendation":
            continue
        user_id = task.get("user_id")
        gt_item = groundtruth.get("ground truth")
        if user_id and gt_item:
            pairs.add((str(user_id), str(gt_item)))
    return pairs


def build_user_item_sequences(
    reviews: Sequence[dict],
    source: str,
    exclusion_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[str, List[str]]:
    """Return time-ordered unique item-id sequences per user.

    Reviews without a parseable timestamp keep file order (needed for Goodreads,
    where date fields are often missing/unreliable).
    """
    exclusion_pairs = exclusion_pairs or set()
    # (sort_key, seq_idx, item_id) — seq_idx breaks ties / missing timestamps
    events: Dict[str, List[Tuple[float, int, str]]] = {}
    seq_idx = 0
    n_source = 0
    n_kept = 0
    for review in reviews:
        if review.get("source") != source:
            continue
        n_source += 1
        user_id = review.get("user_id")
        item_id = review.get("item_id")
        if not user_id or not item_id:
            continue
        if (str(user_id), str(item_id)) in exclusion_pairs:
            continue
        timestamp = parse_review_timestamp(review)
        # Missing timestamps: use large offset + file order so sequences are non-empty
        sort_key = float(timestamp) if timestamp is not None else (1e18 + seq_idx)
        events.setdefault(str(user_id), []).append((sort_key, seq_idx, str(item_id)))
        seq_idx += 1
        n_kept += 1

    if n_source == 0:
        raise ValueError(
            f"No reviews with source={source!r} in review.json. "
            "Check domain tags (amazon/yelp/goodreads)."
        )
    if n_kept == 0:
        raise ValueError(
            f"Found {n_source} reviews for source={source!r} but kept 0 after "
            "user/item/exclusion filters."
        )

    sequences: Dict[str, List[str]] = {}
    for user_id, user_events in events.items():
        user_events.sort(key=lambda pair: (pair[0], pair[1]))
        seen: Set[str] = set()
        ordered_items: List[str] = []
        for _, _, item_id in user_events:
            if item_id in seen:
                continue
            seen.add(item_id)
            ordered_items.append(item_id)
        if ordered_items:
            sequences[user_id] = ordered_items
    return sequences


def build_item_vocab(item_sequences: Dict[str, List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    item_ids = sorted({item_id for seq in item_sequences.values() for item_id in seq})
    item_id_to_idx = {item_id: idx + 1 for idx, item_id in enumerate(item_ids)}  # 0 = pad
    idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}
    return item_id_to_idx, idx_to_item_id


def to_index_sequences(
    item_sequences: Dict[str, List[str]],
    item_id_to_idx: Dict[str, int],
) -> Dict[str, List[int]]:
    indexed: Dict[str, List[int]] = {}
    for user_id, items in item_sequences.items():
        seq = [item_id_to_idx[item_id] for item_id in items if item_id in item_id_to_idx]
        if seq:
            indexed[user_id] = seq
    return indexed


def leave_one_out_split(
    indexed_sequences: Dict[str, List[int]],
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
    """Match kang205/SASRec split: train[:-2], valid[-2], test[-1] when len >= 3."""
    user_train: Dict[str, List[int]] = {}
    user_valid: Dict[str, List[int]] = {}
    user_test: Dict[str, List[int]] = {}
    for user_id, seq in indexed_sequences.items():
        if len(seq) < 3:
            user_train[user_id] = seq
            user_valid[user_id] = []
            user_test[user_id] = []
        else:
            user_train[user_id] = seq[:-2]
            user_valid[user_id] = [seq[-2]]
            user_test[user_id] = [seq[-1]]
    return user_train, user_valid, user_test


def build_sasrec_data(
    data_dir: str,
    source: str = "amazon",
    eval_task_dir: Optional[str] = None,
    eval_groundtruth_dir: Optional[str] = None,
) -> SASRecDataBundle:
    review_path = os.path.join(data_dir, "review.json")
    reviews = load_jsonl(review_path)
    exclusion_pairs: Set[Tuple[str, str]] = set()
    if eval_task_dir and eval_groundtruth_dir:
        exclusion_pairs = load_eval_exclusion_pairs(eval_task_dir, eval_groundtruth_dir)

    item_sequences = build_user_item_sequences(
        reviews=reviews,
        source=source,
        exclusion_pairs=exclusion_pairs,
    )
    item_id_to_idx, idx_to_item_id = build_item_vocab(item_sequences)
    indexed = to_index_sequences(item_sequences, item_id_to_idx)
    user_train, user_valid, user_test = leave_one_out_split(indexed)
    return SASRecDataBundle(
        user_train=user_train,
        user_valid=user_valid,
        user_test=user_test,
        item_id_to_idx=item_id_to_idx,
        idx_to_item_id=idx_to_item_id,
        num_items=len(item_id_to_idx),
        num_users=len(indexed),
    )


def build_padded_sequence(
    item_indices: Sequence[int],
    maxlen: int,
) -> np.ndarray:
    seq = np.zeros(maxlen, dtype=np.int64)
    if not item_indices:
        return seq
    tail = list(item_indices[-maxlen:])
    seq[-len(tail) :] = tail
    return seq


def build_user_history_indices(
    interaction_tool,
    user_id: str,
    item_id_to_idx: Dict[str, int],
    maxlen: int,
    exclude_item_ids: Optional[Sequence[str]] = None,
    source: Optional[str] = None,
) -> np.ndarray:
    """Build padded history indices for SASRec ranking.

    ``exclude_item_ids`` should include the held-out Track-2 GT so the model is
    not conditioned on the target (matches leave-one-out training).
    """
    exclude = {str(x) for x in (exclude_item_ids or [])}
    reviews = interaction_tool.get_reviews(user_id=user_id)

    def _sort_key(review: dict) -> float:
        ts = parse_review_timestamp(review)
        if ts is not None:
            return float(ts)
        raw = review.get("timestamp", 0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    reviews = sorted(reviews, key=_sort_key)
    seen: Set[str] = set()
    history: List[int] = []
    for review in reviews:
        if source is not None:
            rev_src = review.get("source")
            if rev_src is not None and rev_src != source:
                continue
        item_id = review.get("item_id")
        if not item_id:
            continue
        item_id = str(item_id)
        if item_id in exclude or item_id in seen:
            continue
        seen.add(item_id)
        idx = item_id_to_idx.get(item_id)
        if idx is not None:
            history.append(idx)
    return build_padded_sequence(history, maxlen=maxlen)


class SASRecTrainSampler:
    """Batch sampler matching SASRec pairwise training."""

    def __init__(
        self,
        user_train: Dict[str, List[int]],
        num_items: int,
        maxlen: int,
        batch_size: int,
        seed: int = 42,
    ):
        self.users = [user_id for user_id, seq in user_train.items() if len(seq) >= 1]
        self.user_train = user_train
        self.num_items = num_items
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return max(1, len(self.users) // self.batch_size)

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_users = self.rng.sample(self.users, k=min(self.batch_size, len(self.users)))
        seqs = np.zeros((len(batch_users), self.maxlen), dtype=np.int64)
        pos = np.zeros((len(batch_users), self.maxlen), dtype=np.int64)
        neg = np.zeros((len(batch_users), self.maxlen), dtype=np.int64)

        for row, user_id in enumerate(batch_users):
            user_seq = self.user_train[user_id]
            if len(user_seq) < 2:
                input_items = user_seq[:1]
                target_items = user_seq[:1]
            else:
                end = self.rng.randint(1, min(len(user_seq) - 1, self.maxlen))
                input_items = user_seq[:end]
                target_items = user_seq[1 : end + 1]

            tail_inputs = input_items[-self.maxlen :]
            tail_targets = target_items[-self.maxlen :]
            seqs[row, -len(tail_inputs) :] = tail_inputs
            pos[row, -len(tail_targets) :] = tail_targets

            rated = set(user_seq)
            for col in range(self.maxlen):
                if pos[row, col] == 0:
                    continue
                sampled = self.rng.randint(1, self.num_items)
                while sampled in rated:
                    sampled = self.rng.randint(1, self.num_items)
                neg[row, col] = sampled
        return seqs, pos, neg, np.asarray(batch_users, dtype=object)


def evaluate_leave_one_out(
    model,
    user_train: Dict[str, List[int]],
    user_eval: Dict[str, List[int]],
    num_items: int,
    maxlen: int,
    device: torch.device,
    sample_users: Optional[int] = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Original SASRec-style valid/test eval with 100 random negatives + target."""
    rng = random.Random(seed)
    users = [u for u in user_eval if user_eval[u] and user_train.get(u)]
    if sample_users and len(users) > sample_users:
        users = rng.sample(users, sample_users)

    ndcg_total = 0.0
    hr_total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for user_id in tqdm(users, desc="SASRec LOO eval", leave=False):
            target = user_eval[user_id][0]
            train_seq = user_train[user_id]
            seq = build_padded_sequence(train_seq, maxlen=maxlen)
            rated = set(train_seq)
            negatives: List[int] = []
            while len(negatives) < 100:
                sampled = rng.randint(1, num_items)
                if sampled not in rated:
                    negatives.append(sampled)
            candidates = [target] + negatives
            input_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device).unsqueeze(0)
            logits = model.sequence_logits(input_tensor, candidate_tensor).squeeze(0)
            rank = torch.argsort(logits, descending=True).tolist().index(0)
            if rank < 10:
                hr_total += 1.0
                ndcg_total += 1.0 / np.log2(rank + 2)
            count += 1
    if count == 0:
        return {"ndcg@10": 0.0, "hr@10": 0.0, "count": 0.0}
    return {
        "ndcg@10": ndcg_total / count,
        "hr@10": hr_total / count,
        "count": float(count),
    }


def save_data_bundle_metadata(path: str, bundle: SASRecDataBundle) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(bundle.to_metadata(), handle, indent=2)


def load_data_bundle_metadata(path: str) -> SASRecDataBundle:
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return SASRecDataBundle.from_metadata(metadata)
