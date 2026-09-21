"""Build disjoint listwise calibration pools for EAA threshold tuning."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .task_conditions import (
    ConditionConfig,
    build_condition_manifest,
    parse_review_timestamp,
)

CONDITION_NAMES = (
    "classic",
    "cold_start_user",
    "cold_start_item",
    "evolving_interest",
)


def load_eval_exclusion_pairs(task_dir: str, groundtruth_dir: str) -> Set[Tuple[str, str]]:
    """Return benchmark (user_id, ground_truth_item) pairs to exclude from calibration."""
    pairs: Set[Tuple[str, str]] = set()
    if not os.path.isdir(task_dir) or not os.path.isdir(groundtruth_dir):
        return pairs
    for filename in os.listdir(task_dir):
        if not filename.startswith("task_") or not filename.endswith(".json"):
            continue
        task_path = os.path.join(task_dir, filename)
        with open(task_path, "r", encoding="utf-8") as handle:
            task = json.load(handle)
        gt_path = os.path.join(groundtruth_dir, filename.replace("task_", "groundtruth_"))
        if not os.path.isfile(gt_path):
            continue
        with open(gt_path, "r", encoding="utf-8") as handle:
            groundtruth = json.load(handle)
        if task.get("type") != "recommendation":
            continue
        user_id = task.get("user_id")
        gt_item = groundtruth.get("ground truth")
        if user_id and gt_item:
            pairs.add((str(user_id), str(gt_item)))
    return pairs


def load_catalog_item_ids(data_dir: str, source: str) -> List[str]:
    item_path = os.path.join(data_dir, "item.json")
    if not os.path.isfile(item_path):
        raise FileNotFoundError(f"Missing item catalog at {item_path}")
    item_ids: List[str] = []
    with open(item_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("source") == source and item.get("item_id"):
                item_ids.append(str(item["item_id"]))
    if not item_ids:
        raise ValueError(f"No items found for source={source!r} in {item_path}")
    return item_ids


def _label_conditions(
    *,
    user_count: int,
    item_count: int,
    recent_count: int,
    outside_count: int,
    has_task_time: bool,
    config: ConditionConfig,
) -> Dict[str, bool]:
    if config.evolving_interest_last_n is not None:
        evolving_interest = (
            recent_count >= config.evolving_interest_min_recent_reviews
            and outside_count >= config.evolving_interest_min_outside_reviews
        )
    else:
        evolving_interest = has_task_time and (
            recent_count >= config.evolving_interest_min_recent_reviews
            and outside_count >= config.evolving_interest_min_outside_reviews
        )
    return {
        "classic": True,
        "cold_start_user": user_count <= config.cold_start_user_k,
        "cold_start_item": item_count <= config.cold_start_item_k,
        "evolving_interest": evolving_interest,
    }


@dataclass
class CalibrationCandidate:
    user_id: str
    target_item_id: str
    candidate_item_ids: List[str]
    task_time: float
    user_reviews_before_task: int
    user_reviews_in_recent_window: int
    user_reviews_outside_window: int
    item_reviews_before_task: int
    conditions: Dict[str, bool]

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.user_id, self.target_item_id)


def _build_user_events(
    review_path: str,
    source: str,
    *,
    order_mode: str = "timestamp",
) -> Dict[str, List[Tuple[float, str]]]:
    """Build per-user interaction timelines ordered by timestamp or file order."""
    if order_mode not in {"timestamp", "count"}:
        raise ValueError(f"Unsupported order_mode={order_mode!r}; use 'timestamp' or 'count'")

    events: Dict[str, List[Tuple[float, str]]] = {}
    with open(review_path, "r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            review = json.loads(line)
            if review.get("source") != source:
                continue
            user_id = review.get("user_id")
            item_id = review.get("item_id")
            if not user_id or not item_id:
                continue

            if order_mode == "count":
                timestamp = float(line_index)
            else:
                timestamp = parse_review_timestamp(review)
                if timestamp is None:
                    continue

            user_key = str(user_id)
            events.setdefault(user_key, []).append((float(timestamp), str(item_id)))
    for user_id in events:
        events[user_id].sort(key=lambda row: row[0])
    return events


def _unique_history_items(events: Sequence[Tuple[float, str]]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for _, item_id in events:
        if item_id in seen:
            continue
        seen.add(item_id)
        ordered.append(item_id)
    return ordered


def _count_before(reviews: Sequence[Tuple[float, str]], task_time: float) -> int:
    return sum(1 for timestamp, _ in reviews if timestamp < task_time)


def _count_in_window(
    reviews: Sequence[Tuple[float, str]],
    task_time: float,
    window_days: int,
) -> int:
    cutoff = task_time - float(window_days) * 86400.0
    return sum(1 for timestamp, _ in reviews if cutoff <= timestamp < task_time)


def _count_outside_window(
    reviews: Sequence[Tuple[float, str]],
    task_time: float,
    window_days: int,
) -> int:
    cutoff = task_time - float(window_days) * 86400.0
    return sum(1 for timestamp, _ in reviews if timestamp < cutoff)


def _count_last_n_window(total: int, last_n: int) -> Tuple[int, int]:
    recent = min(last_n, total)
    outside = max(0, total - last_n)
    return recent, outside


def generate_calibration_candidates(
    *,
    data_dir: str,
    source: str,
    catalog_items: Sequence[str],
    exclusion_pairs: Set[Tuple[str, str]],
    config: ConditionConfig,
    min_history: int,
    num_candidates: int,
    max_candidates: Optional[int],
    seed: int,
    order_mode: str = "timestamp",
) -> List[CalibrationCandidate]:
    """Enumerate listwise calibration candidates disjoint from benchmark pairs."""
    review_path = os.path.join(data_dir, "review.json")
    user_events = _build_user_events(review_path, source, order_mode=order_mode)
    rng = random.Random(seed)
    num_negatives = num_candidates - 1
    candidates: List[CalibrationCandidate] = []

    item_review_times: Dict[str, List[float]] = {}
    with open(review_path, "r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            review = json.loads(line)
            if review.get("source") != source:
                continue
            item_id = review.get("item_id")
            if not item_id:
                continue
            if order_mode == "count":
                timestamp = float(line_index)
            else:
                timestamp = parse_review_timestamp(review)
                if timestamp is None:
                    continue
            item_review_times.setdefault(str(item_id), []).append(float(timestamp))

    for event_list in item_review_times.values():
        event_list.sort()

    def item_count_before(item_id: str, task_time: float) -> int:
        times = item_review_times.get(item_id, [])
        return sum(1 for timestamp in times if timestamp < task_time)

    catalog = list(catalog_items)
    for user_id, events in user_events.items():
        history_items = _unique_history_items(events)
        if len(history_items) <= min_history:
            continue
        for split_index in range(min_history, len(history_items)):
            target_item_id = history_items[split_index]
            pair = (user_id, target_item_id)
            if pair in exclusion_pairs:
                continue

            prefix_events = events[: split_index + 1]
            history_prefix = _unique_history_items(prefix_events)
            task_time = prefix_events[-1][0]
            history_set = set(history_prefix)
            forbidden = set(history_set)
            pool = [item_id for item_id in catalog if item_id not in forbidden]
            if len(pool) < num_negatives:
                continue
            negatives = rng.sample(pool, num_negatives)
            candidate_ids = [target_item_id, *negatives]
            rng.shuffle(candidate_ids)

            user_count = split_index
            if config.evolving_interest_last_n is not None:
                recent_count, outside_count = _count_last_n_window(
                    len(prefix_events),
                    config.evolving_interest_last_n,
                )
            else:
                recent_count = _count_in_window(
                    prefix_events, task_time, config.evolving_interest_days
                )
                outside_count = _count_outside_window(
                    prefix_events, task_time, config.evolving_interest_days
                )
            item_count = item_count_before(target_item_id, task_time)
            conditions = _label_conditions(
                user_count=user_count,
                item_count=item_count,
                recent_count=recent_count,
                outside_count=outside_count,
                has_task_time=True,
                config=config,
            )
            candidates.append(
                CalibrationCandidate(
                    user_id=user_id,
                    target_item_id=target_item_id,
                    candidate_item_ids=candidate_ids,
                    task_time=task_time,
                    user_reviews_before_task=user_count,
                    user_reviews_in_recent_window=recent_count,
                    user_reviews_outside_window=outside_count,
                    item_reviews_before_task=item_count,
                    conditions=conditions,
                )
            )
            if max_candidates is not None and len(candidates) >= max_candidates:
                return candidates
    return candidates


def sample_calibration_tasks(
    candidates: Sequence[CalibrationCandidate],
    *,
    cal_tasks_per_condition: int,
    seed: int,
) -> List[CalibrationCandidate]:
    """Pick up to N disjoint tasks per condition, unioned into one calibration set."""
    rng = random.Random(seed)
    used_pairs: Set[Tuple[str, str]] = set()
    selected: List[CalibrationCandidate] = []

    for condition in CONDITION_NAMES:
        pool = [candidate for candidate in candidates if candidate.conditions.get(condition)]
        rng.shuffle(pool)
        picked = 0
        for candidate in pool:
            if candidate.pair in used_pairs:
                continue
            selected.append(candidate)
            used_pairs.add(candidate.pair)
            picked += 1
            if picked >= cal_tasks_per_condition:
                break
    return selected


def write_calibration_task_files(
    tasks: Sequence[CalibrationCandidate],
    *,
    task_dir: str,
    groundtruth_dir: str,
) -> None:
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)
    for index, task in enumerate(tasks):
        task_payload = {
            "type": "recommendation",
            "user_id": task.user_id,
            "candidate_category": "product",
            "candidate_list": list(task.candidate_item_ids),
            "loc": [-1, -1],
        }
        gt_payload = {"ground truth": task.target_item_id}
        task_path = os.path.join(task_dir, f"task_{index}.json")
        gt_path = os.path.join(groundtruth_dir, f"groundtruth_{index}.json")
        with open(task_path, "w", encoding="utf-8") as handle:
            json.dump(task_payload, handle, indent=2)
        with open(gt_path, "w", encoding="utf-8") as handle:
            json.dump(gt_payload, handle, indent=2)


def build_calibration_pool_manifest(
    *,
    data_dir: str,
    benchmark_task_dir: str,
    benchmark_groundtruth_dir: str,
    calibration_task_dir: str,
    calibration_groundtruth_dir: str,
    dataset: str,
    track: int,
    output_path: str,
    config: Optional[ConditionConfig] = None,
    cal_tasks_per_condition: int = 300,
    min_history: int = 5,
    num_candidates: int = 20,
    max_candidate_scan: Optional[int] = None,
    seed: int = 42,
    order_mode: str = "timestamp",
) -> dict:
    """Sample disjoint calibration tasks and write manifest + task files."""
    config = config or ConditionConfig()
    exclusion_pairs = load_eval_exclusion_pairs(benchmark_task_dir, benchmark_groundtruth_dir)
    catalog_items = load_catalog_item_ids(data_dir, dataset)

    candidates = generate_calibration_candidates(
        data_dir=data_dir,
        source=dataset,
        catalog_items=catalog_items,
        exclusion_pairs=exclusion_pairs,
        config=config,
        min_history=min_history,
        num_candidates=num_candidates,
        max_candidates=max_candidate_scan,
        seed=seed,
        order_mode=order_mode,
    )
    selected = sample_calibration_tasks(
        candidates,
        cal_tasks_per_condition=cal_tasks_per_condition,
        seed=seed + 1,
    )
    if not selected:
        if not candidates:
            hint = (
                "No calibration candidates were generated. "
                "If review timestamps are missing (common on Goodreads), rerun with "
                "--order_mode count --evolving_interest_last_n 10 "
                "--evolving_interest_min_outside_reviews 1."
            )
        else:
            hint = (
                f"Generated {len(candidates)} candidates but sampled 0 tasks. "
                "Lower --cal_tasks_per_condition or increase --max_candidate_scan."
            )
        raise RuntimeError(hint)

    write_calibration_task_files(
        selected,
        task_dir=calibration_task_dir,
        groundtruth_dir=calibration_groundtruth_dir,
    )

    manifest = build_condition_manifest(
        data_dir=data_dir,
        task_dir=calibration_task_dir,
        groundtruth_dir=calibration_groundtruth_dir,
        dataset=dataset,
        track=track,
        output_path=output_path,
        config=config,
    )
    manifest["metadata"]["manifest_kind"] = "eaa_calibration_pool"
    manifest["metadata"]["benchmark_task_dir"] = os.path.abspath(benchmark_task_dir)
    manifest["metadata"]["benchmark_groundtruth_dir"] = os.path.abspath(
        benchmark_groundtruth_dir
    )
    manifest["metadata"]["calibration_protocol"] = (
        "Threshold tuning on this disjoint pool; evaluate agents on the official "
        "benchmark manifest with identical task counts."
    )
    manifest["metadata"]["cal_tasks_per_condition"] = cal_tasks_per_condition
    manifest["metadata"]["excluded_benchmark_pairs"] = len(exclusion_pairs)
    manifest["metadata"]["calibration_candidates_scanned"] = len(candidates)
    manifest["metadata"]["calibration_tasks_written"] = len(selected)
    manifest["metadata"]["seed"] = seed
    manifest["metadata"]["order_mode"] = order_mode
    if config.evolving_interest_last_n is not None:
        manifest["metadata"]["condition_label_mode"] = "count_based_last_n_interactions"

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
