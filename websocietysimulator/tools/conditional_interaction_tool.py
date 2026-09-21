"""Wrap InteractionTool with per-task history visibility rules."""

from __future__ import annotations

from typing import List, Optional

from ..conditions.task_conditions import parse_review_timestamp
from .cache_interaction_tool import CacheInteractionTool
from .interaction_tool import InteractionTool

InteractionToolType = InteractionTool | CacheInteractionTool


class ConditionalInteractionTool:
    """
    Proxy around InteractionTool / CacheInteractionTool.

    - classic / cold-start: full available history (block_set still applies on base tool)
    - evolving_interest: only reviews within the last ``history_window_days`` before
      task time, or the last ``history_window_count`` interactions when set
    """

    def __init__(
        self,
        base_tool: InteractionToolType,
        history_window_days: Optional[float] = None,
        history_window_count: Optional[int] = None,
    ):
        self._base = base_tool
        self.history_window_days = history_window_days
        self.history_window_count = history_window_count
        self._task_time: Optional[float] = None

    def set_task_context(
        self,
        task_time: Optional[float] = None,
        history_window_days: Optional[float] = None,
        history_window_count: Optional[int] = None,
    ):
        self._task_time = task_time
        if history_window_days is not None:
            self.history_window_days = history_window_days
        if history_window_count is not None:
            self.history_window_count = history_window_count

    @staticmethod
    def _review_sort_key(review: dict, fallback_index: int) -> tuple:
        timestamp = parse_review_timestamp(review)
        if timestamp is None:
            return (1, float(fallback_index))
        return (0, float(timestamp))

    def _filter_reviews(self, reviews: List[dict]) -> List[dict]:
        if self.history_window_count is not None:
            indexed = list(enumerate(reviews))
            indexed.sort(key=lambda pair: self._review_sort_key(pair[1], pair[0]))
            ordered = [review for _, review in indexed]
            if self._task_time is not None:
                before_task = []
                for review in ordered:
                    timestamp = parse_review_timestamp(review)
                    if timestamp is not None and timestamp < self._task_time:
                        before_task.append(review)
                pool = before_task or ordered
            else:
                pool = ordered
            window = max(int(self.history_window_count), 0)
            return pool[-window:] if window else pool

        if self.history_window_days is None:
            return reviews
        if self._task_time is None:
            return reviews

        cutoff = self._task_time - float(self.history_window_days) * 86400.0
        filtered: List[dict] = []
        for review in reviews:
            timestamp = parse_review_timestamp(review)
            if timestamp is None:
                continue
            if timestamp < self._task_time and timestamp >= cutoff:
                filtered.append(review)
        return filtered

    def get_user(self, user_id: str):
        return self._base.get_user(user_id)

    def get_item(self, item_id: str):
        return self._base.get_item(item_id)

    def get_reviews(
        self,
        item_id: Optional[str] = None,
        user_id: Optional[str] = None,
        review_id: Optional[str] = None,
    ) -> List[dict]:
        reviews = self._base.get_reviews(
            item_id=item_id,
            user_id=user_id,
            review_id=review_id,
        )
        if review_id:
            return reviews
        return self._filter_reviews(reviews)
