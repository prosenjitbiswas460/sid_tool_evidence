"""Profile competition tasks into AgentRecBench-style evaluation conditions."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("websocietysimulator")


def _parse_date_value(date_value) -> Optional[float]:
    if date_value is None:
        return None
    if isinstance(date_value, (int, float)):
        timestamp = float(date_value)
        return timestamp / 1000.0 if timestamp > 1e12 else timestamp

    text = str(date_value).strip()
    if not text or text.lower() in {"none", "nan", "nat"}:
        return None

    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        for candidate in (text, text[:19], text[:10]):
            try:
                return datetime.strptime(candidate, fmt).timestamp()
            except ValueError:
                continue

    try:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def parse_review_timestamp(review: dict) -> Optional[float]:
    """Return review time as Unix seconds when available."""
    if review.get("timestamp") is not None:
        timestamp = float(review["timestamp"])
        if timestamp > 1e12:
            return timestamp / 1000.0
        return timestamp

    for field in ("date", "date_updated", "date_added", "read_at", "started_at"):
        parsed = _parse_date_value(review.get(field))
        if parsed is not None:
            return parsed
    return None


@dataclass
class ConditionConfig:
    """Thresholds used to label tasks."""

    cold_start_user_k: int = 5
    cold_start_item_k: int = 5
    # Recency window T (days): standard short-term history horizon in time-aware rec.
    evolving_interest_days: int = 90
    # Min interactions in [task_time - T, task_time) to define a non-trivial
    # short-term profile (cf. session/min-history thresholds in sequential rec).
    evolving_interest_min_recent_reviews: int = 2
    # Optional: require hidden long-term history (outside window). Default 0 =
    # include all users with sufficient recent activity. Set to 1+ for a stricter
    # "preference drift" slice where full-history vs recent-only inputs differ.
    evolving_interest_min_outside_reviews: int = 0
    # Optional count-based recency window (last N interactions). Used when timestamps
    # are missing or unreliable (e.g. temporary Goodreads manifests).
    evolving_interest_last_n: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskProfile:
    index: int
    task_file: str
    task_type: str
    user_id: str
    target_item_id: str
    candidate_item_ids: List[str] = field(default_factory=list)
    task_time: Optional[float] = None
    user_reviews_before_task: int = 0
    user_reviews_in_recent_window: int = 0
    user_reviews_outside_window: int = 0
    item_reviews_before_task: int = 0
    min_candidate_item_reviews_before_task: Optional[int] = None
    conditions: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class TaskConditionProfiler:
    """Compute per-task metadata and condition membership from processed review.json."""

    def __init__(self, data_dir: str, config: Optional[ConditionConfig] = None):
        self.data_dir = data_dir
        self.config = config or ConditionConfig()
        self.review_path = os.path.join(data_dir, "review.json")
        if not os.path.exists(self.review_path):
            raise FileNotFoundError(
                f"Processed review.json not found at {self.review_path}. "
                "Run data_process.py first."
            )

    def load_task_records(
        self,
        task_dir: str,
        groundtruth_dir: str,
    ) -> List[dict]:
        task_files = sorted(
            [
                file_name
                for file_name in os.listdir(task_dir)
                if file_name.startswith("task_") and file_name.endswith(".json")
            ],
            key=lambda name: int(name.split("_")[1].split(".")[0]),
        )

        records: List[dict] = []
        for task_index, task_file in enumerate(task_files):
            task_path = os.path.join(task_dir, task_file)
            groundtruth_path = os.path.join(
                groundtruth_dir,
                f"groundtruth_{task_index}.json",
            )
            if not os.path.exists(groundtruth_path):
                logger.warning("Missing groundtruth for %s", task_file)
                continue

            with open(task_path, "r", encoding="utf-8") as handle:
                task_data = json.load(handle)
            with open(groundtruth_path, "r", encoding="utf-8") as handle:
                groundtruth_data = json.load(handle)

            if task_data["type"] == "user_behavior_simulation":
                target_item_id = task_data["item_id"]
                candidate_item_ids: List[str] = [target_item_id]
            else:
                target_item_id = groundtruth_data["ground truth"]
                candidate_item_ids = list(task_data["candidate_list"])

            records.append(
                {
                    "index": task_index,
                    "task_file": task_file,
                    "task_type": task_data["type"],
                    "user_id": task_data["user_id"],
                    "target_item_id": target_item_id,
                    "candidate_item_ids": candidate_item_ids,
                }
            )
        return records

    def _collect_relevant_ids(self, task_records: Sequence[dict]) -> Tuple[Set[str], Set[str]]:
        user_ids = {record["user_id"] for record in task_records}
        item_ids: Set[str] = set()
        for record in task_records:
            item_ids.update(record["candidate_item_ids"])
        return user_ids, item_ids

    def _build_review_indices(
        self,
        user_ids: Set[str],
        item_ids: Set[str],
        review_source: Optional[str] = None,
    ) -> Tuple[
        Dict[str, List[Tuple[float, str, str]]],
        Dict[str, List[Tuple[float, str, str]]],
        Dict[Tuple[str, str], float],
    ]:
        user_reviews: Dict[str, List[Tuple[float, str, str]]] = {uid: [] for uid in user_ids}
        item_reviews: Dict[str, List[Tuple[float, str, str]]] = {iid: [] for iid in item_ids}
        pair_times: Dict[Tuple[str, str], float] = {}
        use_count_fallback = self.config.evolving_interest_last_n is not None

        with open(self.review_path, "r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                review = json.loads(line)
                if review_source and review.get("source") != review_source:
                    continue
                user_id = review.get("user_id")
                item_id = review.get("item_id")
                if user_id is None or item_id is None:
                    continue

                timestamp = parse_review_timestamp(review)
                if timestamp is None:
                    if not use_count_fallback:
                        continue
                    timestamp = float(line_index)

                pair_key = (user_id, item_id)
                previous = pair_times.get(pair_key)
                if previous is None or timestamp > previous:
                    pair_times[pair_key] = timestamp

                if user_id in user_ids:
                    user_reviews[user_id].append((timestamp, item_id, review.get("review_id", "")))
                if item_id in item_ids:
                    item_reviews[item_id].append((timestamp, user_id, review.get("review_id", "")))

        for reviews in user_reviews.values():
            reviews.sort(key=lambda entry: entry[0])
        for reviews in item_reviews.values():
            reviews.sort(key=lambda entry: entry[0])

        return user_reviews, item_reviews, pair_times

    @staticmethod
    def _count_before(reviews: Sequence[Tuple[float, str, str]], task_time: float) -> int:
        return sum(1 for timestamp, _, _ in reviews if timestamp < task_time)

    @staticmethod
    def _count_in_recent_window(
        reviews: Sequence[Tuple[float, str, str]],
        task_time: float,
        window_days: int,
    ) -> int:
        cutoff = task_time - float(window_days) * 86400.0
        return sum(
            1 for timestamp, _, _ in reviews if cutoff <= timestamp < task_time
        )

    @staticmethod
    def _count_outside_recent_window(
        reviews: Sequence[Tuple[float, str, str]],
        task_time: float,
        window_days: int,
    ) -> int:
        cutoff = task_time - float(window_days) * 86400.0
        return sum(1 for timestamp, _, _ in reviews if timestamp < cutoff)

    def _resolve_task_time(
        self,
        user_id: str,
        target_item_id: str,
        pair_times: Dict[Tuple[str, str], float],
        user_reviews: Sequence[Tuple[float, str, str]],
    ) -> Optional[float]:
        pair_time = pair_times.get((user_id, target_item_id))
        if pair_time is not None:
            return pair_time
        if user_reviews:
            return max(timestamp for timestamp, _, _ in user_reviews)
        return None

    @staticmethod
    def _count_last_n_window(total: int, last_n: int) -> Tuple[int, int]:
        recent = min(last_n, total)
        outside = max(0, total - last_n)
        return recent, outside

    def profile_tasks(
        self,
        task_dir: str,
        groundtruth_dir: str,
        review_source: Optional[str] = None,
    ) -> List[TaskProfile]:
        task_records = self.load_task_records(task_dir, groundtruth_dir)
        user_ids, item_ids = self._collect_relevant_ids(task_records)
        user_reviews, item_reviews, pair_times = self._build_review_indices(
            user_ids,
            item_ids,
            review_source=review_source,
        )

        profiles: List[TaskProfile] = []
        for record in task_records:
            user_history = user_reviews.get(record["user_id"], [])
            task_time = self._resolve_task_time(
                record["user_id"],
                record["target_item_id"],
                pair_times,
                user_history,
            )

            user_count = (
                self._count_before(user_history, task_time)
                if task_time is not None
                else len(user_history)
            )
            recent_count = 0
            outside_count = 0
            if self.config.evolving_interest_last_n is not None:
                reviews_before = (
                    [entry for entry in user_history if entry[0] < task_time]
                    if task_time is not None
                    else list(user_history)
                )
                recent_count, outside_count = self._count_last_n_window(
                    len(reviews_before),
                    self.config.evolving_interest_last_n,
                )
            elif task_time is not None:
                recent_count = self._count_in_recent_window(
                    user_history,
                    task_time,
                    self.config.evolving_interest_days,
                )
                outside_count = self._count_outside_recent_window(
                    user_history,
                    task_time,
                    self.config.evolving_interest_days,
                )

            target_item_history = item_reviews.get(record["target_item_id"], [])
            item_count = (
                self._count_before(target_item_history, task_time)
                if task_time is not None
                else len(target_item_history)
            )

            candidate_counts = []
            for candidate_id in record["candidate_item_ids"]:
                candidate_history = item_reviews.get(candidate_id, [])
                if task_time is None:
                    candidate_counts.append(len(candidate_history))
                else:
                    candidate_counts.append(self._count_before(candidate_history, task_time))

            conditions = self._label_conditions(
                user_count=user_count,
                item_count=item_count,
                recent_count=recent_count,
                outside_count=outside_count,
                has_task_time=task_time is not None,
            )
            profiles.append(
                TaskProfile(
                    index=record["index"],
                    task_file=record["task_file"],
                    task_type=record["task_type"],
                    user_id=record["user_id"],
                    target_item_id=record["target_item_id"],
                    candidate_item_ids=record["candidate_item_ids"],
                    task_time=task_time,
                    user_reviews_before_task=user_count,
                    user_reviews_in_recent_window=recent_count,
                    user_reviews_outside_window=outside_count,
                    item_reviews_before_task=item_count,
                    min_candidate_item_reviews_before_task=min(candidate_counts)
                    if candidate_counts
                    else None,
                    conditions=conditions,
                )
            )
        return profiles

    def _label_conditions(
        self,
        user_count: int,
        item_count: int,
        recent_count: int,
        outside_count: int,
        has_task_time: bool,
    ) -> Dict[str, bool]:
        """
        Label task conditions.

        Evolving interest follows the common *recency-window* evaluation used in
        time-aware / dynamic recommendation: at task time t the agent should act
        on short-term history (t - T, t) instead of full lifetime history.

        A task is eligible when the user has enough recent interactions to form
        a short-term preference signal (>= min_recent_reviews in the window).
        Optionally require min_outside_reviews > 0 to restrict to users whose
        long-term history is hidden by the window (stronger drift contrast).
        """
        evolving_interest = (
            recent_count >= self.config.evolving_interest_min_recent_reviews
            and outside_count >= self.config.evolving_interest_min_outside_reviews
        )
        if self.config.evolving_interest_last_n is None:
            evolving_interest = has_task_time and evolving_interest
        return {
            "classic": True,
            "cold_start_user": user_count <= self.config.cold_start_user_k,
            "cold_start_item": item_count <= self.config.cold_start_item_k,
            "evolving_interest": evolving_interest,
        }


def build_condition_manifest(
    data_dir: str,
    task_dir: str,
    groundtruth_dir: str,
    dataset: str,
    track: int,
    output_path: str,
    config: Optional[ConditionConfig] = None,
) -> dict:
    profiler = TaskConditionProfiler(data_dir=data_dir, config=config)
    profiles = profiler.profile_tasks(task_dir, groundtruth_dir, review_source=dataset)
    condition_indices = {
        "classic": [profile.index for profile in profiles if profile.conditions["classic"]],
        "cold_start_user": [
            profile.index for profile in profiles if profile.conditions["cold_start_user"]
        ],
        "cold_start_item": [
            profile.index for profile in profiles if profile.conditions["cold_start_item"]
        ],
        "evolving_interest": [
            profile.index for profile in profiles if profile.conditions["evolving_interest"]
        ],
    }

    config_dict = (config or ConditionConfig()).to_dict()
    if config_dict.get("evolving_interest_last_n") is not None:
        evolving_definition = (
            "Count-based short-term slice: agent sees only the user's last "
            f"{config_dict['evolving_interest_last_n']} interactions before task. "
            "Included when recent_window>="
            f"{config_dict.get('evolving_interest_min_recent_reviews', 2)} and "
            "outside_window>="
            f"{config_dict.get('evolving_interest_min_outside_reviews', 0)}."
        )
        condition_label_mode = "count_based_last_n_interactions"
    else:
        evolving_definition = (
            "Recency-window (short-term) evaluation: agent sees only user "
            "interactions in the last evolving_interest_days before task "
            "time — the usual dynamic-preference setup in time-aware and "
            "session-based recommendation (full history in classic). "
            "Task is included if the user has at least "
            "evolving_interest_min_recent_reviews interactions in that "
            "window (non-trivial short-term profile). Optionally set "
            "evolving_interest_min_outside_reviews >= 1 to require "
            "hidden long-term history for a stricter drift subset."
        )
        condition_label_mode = None

    manifest = {
        "metadata": {
            "dataset": dataset,
            "track": track,
            "data_dir": os.path.abspath(data_dir),
            "task_dir": os.path.abspath(task_dir),
            "groundtruth_dir": os.path.abspath(groundtruth_dir),
            "config": config_dict,
            "condition_definitions": {
                "classic": "All tasks; agents receive full available user history.",
                "cold_start_user": (
                    "Users with <= cold_start_user_k reviews strictly before task time."
                ),
                "cold_start_item": (
                    "Ground-truth item with <= cold_start_item_k reviews before task time."
                ),
                "evolving_interest": evolving_definition,
            },
            "total_tasks": len(profiles),
        },
        "tasks": [profile.to_dict() for profile in profiles],
        "condition_indices": condition_indices,
        "condition_counts": {
            name: len(indices) for name, indices in condition_indices.items()
        },
    }
    if condition_label_mode:
        manifest["metadata"]["condition_label_mode"] = condition_label_mode

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def load_condition_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_condition_task_indices(manifest: dict, condition: str) -> List[int]:
    if condition not in manifest["condition_indices"]:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Expected one of: {sorted(manifest['condition_indices'].keys())}"
        )
    return list(manifest["condition_indices"][condition])


def build_task_contexts(
    manifest: dict,
    condition: str,
) -> Dict[int, dict]:
    """Return per-task context for ConditionalInteractionTool."""
    config = manifest["metadata"]["config"]
    window_days = config.get("evolving_interest_days", 90)
    window_count = config.get("evolving_interest_last_n")
    contexts: Dict[int, dict] = {}
    for task in manifest["tasks"]:
        history_window_days = None
        history_window_count = None
        if condition == "evolving_interest":
            if window_count is not None:
                history_window_count = int(window_count)
            else:
                history_window_days = window_days
        contexts[task["index"]] = {
            "task_time": task.get("task_time"),
            "history_window_days": history_window_days,
            "history_window_count": history_window_count,
        }
    return contexts
