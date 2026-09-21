#!/usr/bin/env python3
"""
Temporary patch: relabel all manifest conditions from review.json counts.

Use when Goodreads (or other) manifests have broken time-based labels because
reviews lack reliable timestamps. Recomputes:

  - classic (always true)
  - cold_start_user / cold_start_item (from interaction counts before task)
  - evolving_interest (last-N interaction window)

At eval time, evolving_interest uses evolving_interest_last_n from manifest
config (see ConditionalInteractionTool.history_window_count).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from websocietysimulator.conditions.task_conditions import (  # noqa: E402
    load_condition_manifest,
    parse_review_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Existing manifest JSON (e.g. manifests/goodreads_track2.json)",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Processed data dir with review.json (default: manifest metadata data_dir)",
    )
    parser.add_argument("--cold_start_user_k", type=int, default=None)
    parser.add_argument("--cold_start_item_k", type=int, default=None)
    parser.add_argument(
        "--last_n",
        type=int,
        default=10,
        help="Short-term window = last N user interactions before task",
    )
    parser.add_argument(
        "--min_recent_reviews",
        type=int,
        default=2,
        help="Min interactions required inside the last-N window",
    )
    parser.add_argument(
        "--min_outside_reviews",
        type=int,
        default=1,
        help="Min interactions required before the last-N window",
    )
    parser.add_argument(
        "--review_source",
        default=None,
        help="Optional review source filter (e.g. goodreads)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: overwrite --manifest)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print counts only; do not write manifest",
    )
    return parser.parse_args()


def _review_sort_key(review: dict, fallback_index: int) -> Tuple[int, float]:
    timestamp = parse_review_timestamp(review)
    if timestamp is None:
        return (1, float(fallback_index))
    return (0, float(timestamp))


def load_review_indices(
    review_path: str,
    user_ids: Set[str],
    item_ids: Set[str],
    review_source: Optional[str],
) -> Tuple[
    Dict[str, List[dict]],
    Dict[str, List[dict]],
    Dict[Tuple[str, str], float],
]:
    user_histories: Dict[str, List[Tuple[int, dict]]] = {uid: [] for uid in user_ids}
    item_histories: Dict[str, List[Tuple[int, dict]]] = {iid: [] for iid in item_ids}
    pair_times: Dict[Tuple[str, str], float] = {}

    with open(review_path, "r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            review = json.loads(line)
            if review_source and review.get("source") != review_source:
                continue

            user_id = review.get("user_id")
            item_id = review.get("item_id")
            timestamp = parse_review_timestamp(review)

            if user_id is not None and item_id is not None and timestamp is not None:
                pair_key = (user_id, item_id)
                previous = pair_times.get(pair_key)
                if previous is None or timestamp > previous:
                    pair_times[pair_key] = timestamp

            if user_id in user_ids:
                user_histories[user_id].append((line_index, review))
            if item_id in item_ids:
                item_histories[item_id].append((line_index, review))

    def _finalize(
        raw: Dict[str, List[Tuple[int, dict]]],
    ) -> Dict[str, List[dict]]:
        ordered: Dict[str, List[dict]] = {}
        for key, entries in raw.items():
            entries.sort(key=lambda pair: _review_sort_key(pair[1], pair[0]))
            ordered[key] = [review for _, review in entries]
        return ordered

    return _finalize(user_histories), _finalize(item_histories), pair_times


def reviews_before_task(
    reviews: Sequence[dict],
    task_time: Optional[float],
) -> List[dict]:
    if task_time is None:
        return list(reviews)
    pool: List[dict] = []
    for review in reviews:
        timestamp = parse_review_timestamp(review)
        if timestamp is not None and timestamp < task_time:
            pool.append(review)
    return pool or list(reviews)


def count_before(reviews: Sequence[dict], task_time: Optional[float]) -> int:
    return len(reviews_before_task(reviews, task_time))


def count_last_n_window(
    reviews: Sequence[dict],
    last_n: int,
) -> Tuple[int, int, int]:
    total = len(reviews)
    recent = min(last_n, total)
    outside = max(0, total - last_n)
    return total, recent, outside


def resolve_task_time(
    user_id: str,
    target_item_id: str,
    pair_times: Dict[Tuple[str, str], float],
    user_reviews: Sequence[dict],
) -> Optional[float]:
    pair_time = pair_times.get((user_id, target_item_id))
    if pair_time is not None:
        return pair_time
    timestamps = [
        parse_review_timestamp(review)
        for review in user_reviews
        if parse_review_timestamp(review) is not None
    ]
    if timestamps:
        return max(timestamps)
    return None


def label_conditions(
    *,
    user_count: int,
    item_count: int,
    recent_count: int,
    outside_count: int,
    cold_start_user_k: int,
    cold_start_item_k: int,
    min_recent_reviews: int,
    min_outside_reviews: int,
) -> Dict[str, bool]:
    evolving_interest = (
        recent_count >= min_recent_reviews
        and outside_count >= min_outside_reviews
    )
    return {
        "classic": True,
        "cold_start_user": user_count <= cold_start_user_k,
        "cold_start_item": item_count <= cold_start_item_k,
        "evolving_interest": evolving_interest,
    }


def rebuild_condition_indices(tasks: Sequence[dict]) -> Dict[str, List[int]]:
    indices = {
        "classic": [],
        "cold_start_user": [],
        "cold_start_item": [],
        "evolving_interest": [],
    }
    for task in tasks:
        for name, flag in task["conditions"].items():
            if flag:
                indices[name].append(task["index"])
    return indices


def patch_manifest(args: argparse.Namespace) -> dict:
    manifest = load_condition_manifest(args.manifest)
    patched = copy.deepcopy(manifest)
    data_dir = args.data_dir or manifest["metadata"]["data_dir"]
    review_path = os.path.join(data_dir, "review.json")
    if not os.path.isfile(review_path):
        raise FileNotFoundError(f"Missing review.json at {review_path}")

    config = patched["metadata"].setdefault("config", {})
    cold_start_user_k = (
        args.cold_start_user_k
        if args.cold_start_user_k is not None
        else int(config.get("cold_start_user_k", 5))
    )
    cold_start_item_k = (
        args.cold_start_item_k
        if args.cold_start_item_k is not None
        else int(config.get("cold_start_item_k", 5))
    )

    review_source = args.review_source or manifest["metadata"].get("dataset")
    user_ids = {task["user_id"] for task in patched["tasks"]}
    item_ids: Set[str] = set()
    for task in patched["tasks"]:
        item_ids.add(task["target_item_id"])
        item_ids.update(task.get("candidate_item_ids") or [])

    user_histories, item_histories, pair_times = load_review_indices(
        review_path,
        user_ids,
        item_ids,
        review_source,
    )

    for task in patched["tasks"]:
        user_history = user_histories.get(task["user_id"], [])
        task_time = resolve_task_time(
            task["user_id"],
            task["target_item_id"],
            pair_times,
            user_history,
        )
        user_before = reviews_before_task(user_history, task_time)
        user_count = len(user_before)
        recent_count, outside_count = count_last_n_window(user_before, args.last_n)[1:]

        target_history = item_histories.get(task["target_item_id"], [])
        item_count = count_before(target_history, task_time)

        candidate_counts = []
        for candidate_id in task.get("candidate_item_ids") or []:
            candidate_history = item_histories.get(candidate_id, [])
            candidate_counts.append(count_before(candidate_history, task_time))

        task["task_time"] = task_time
        task["user_reviews_before_task"] = user_count
        task["user_reviews_in_recent_window"] = recent_count
        task["user_reviews_outside_window"] = outside_count
        task["item_reviews_before_task"] = item_count
        task["min_candidate_item_reviews_before_task"] = (
            min(candidate_counts) if candidate_counts else None
        )
        task["conditions"] = label_conditions(
            user_count=user_count,
            item_count=item_count,
            recent_count=recent_count,
            outside_count=outside_count,
            cold_start_user_k=cold_start_user_k,
            cold_start_item_k=cold_start_item_k,
            min_recent_reviews=args.min_recent_reviews,
            min_outside_reviews=args.min_outside_reviews,
        )

    patched["condition_indices"] = rebuild_condition_indices(patched["tasks"])
    patched["condition_counts"] = {
        name: len(indices) for name, indices in patched["condition_indices"].items()
    }

    config["cold_start_user_k"] = cold_start_user_k
    config["cold_start_item_k"] = cold_start_item_k
    config["evolving_interest_last_n"] = args.last_n
    config["evolving_interest_min_recent_reviews"] = args.min_recent_reviews
    config["evolving_interest_min_outside_reviews"] = args.min_outside_reviews

    definitions = patched["metadata"].setdefault("condition_definitions", {})
    definitions["classic"] = "All tasks; agents receive full available user history."
    definitions["cold_start_user"] = (
        f"Users with <= {cold_start_user_k} {review_source or 'dataset'} reviews "
        "strictly before task time (or total reviews when task_time is missing)."
    )
    definitions["cold_start_item"] = (
        f"Ground-truth item with <= {cold_start_item_k} {review_source or 'dataset'} "
        "reviews before task time."
    )
    definitions["evolving_interest"] = (
        "TEMPORARY count-based slice: agent sees only the user's last "
        f"{args.last_n} interactions before task. Included when "
        f"recent_window>={args.min_recent_reviews} and "
        f"outside_window>={args.min_outside_reviews}."
    )
    patched["metadata"]["condition_label_mode"] = "count_based_last_n_interactions"
    patched["metadata"].pop("evolving_interest_label_mode", None)

    return patched


def main() -> None:
    args = parse_args()
    patched = patch_manifest(args)
    print(json.dumps(patched["condition_counts"], indent=2))
    print("condition_label_mode:", patched["metadata"].get("condition_label_mode"))
    print("evolving_interest_last_n:", patched["metadata"]["config"].get("evolving_interest_last_n"))

    if args.dry_run:
        print("Dry run — manifest not written.")
        return

    output_path = args.output or args.manifest
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(patched, handle, indent=2)
    print(f"Wrote patched manifest to {output_path}")


if __name__ == "__main__":
    main()
