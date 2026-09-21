"""
Evidence quality signals and selection policy for EAA.

Design principle (v4):
  Estimate evidence quality → apply documented policy → select modules

No scenario/condition labels are required for selection. Condition is logged
only for benchmark stratification.

Policy thresholds were calibrated on Amazon track-2 ablations:
  - H,M beats H,M,S_tool when history is dense and metadata is rich (SID redundant)
  - S_tool helps when history is sparse (cold-start), even with moderate SID margin
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gr_agent_utils import ITEM_FIELD_KEYS, filter_item_fields

EVIDENCE_MODULE_NAMES = ("H", "M", "S_prompt", "S_tool")


@dataclass(frozen=True)
class EvidenceSignals:
    """Task-level evidence quality (computed before module selection)."""

    review_count: int
    history_chars: int
    metadata_richness: float
    sid_top1_score: float
    sid_top1_margin: float
    sid_score_std: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidencePolicyConfig:
    """
    Documented, reproducible thresholds for v4 quality policy.

    history_dense_reviews / history_dense_chars:
        History is informative enough that text evidence dominates.
    metadata_rich_threshold:
        Mean metadata fill ratio across candidates (0–1).
    sid_margin_threshold / sid_top1_threshold:
        SID tool confidence — top1−top2 margin or absolute top score.
    redundancy_skip_sid:
        When history is dense AND metadata is rich, skip SID (redundancy hypothesis).
    sparse_history_reviews:
        Below this, always consider S_tool even if redundancy would otherwise skip.
    """

    history_dense_reviews: int = 5
    history_dense_chars: int = 2000
    metadata_rich_threshold: float = 0.45
    sid_margin_threshold: float = 0.02
    sid_top1_threshold: float = 0.15
    redundancy_skip_sid: bool = True
    sparse_history_reviews: int = 5
    policy_version: str = "v4_quality"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_POLICY_CONFIG = EvidencePolicyConfig()


def policy_config_from_dict(payload: Dict[str, Any]) -> EvidencePolicyConfig:
    """Load thresholds from a calibration JSON file."""
    thresholds = payload.get("thresholds", payload)
    fields = EvidencePolicyConfig.__dataclass_fields__
    kwargs = {key: thresholds[key] for key in fields if key in thresholds}
    if "policy_version" in thresholds:
        kwargs["policy_version"] = thresholds["policy_version"]
    elif "policy_version" in payload:
        kwargs["policy_version"] = payload["policy_version"]
    return EvidencePolicyConfig(**kwargs)


def load_policy_config(path: str) -> EvidencePolicyConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return policy_config_from_dict(payload)


def save_policy_config(
    path: str,
    config: EvidencePolicyConfig,
    *,
    calibration: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "policy_version": config.policy_version,
        "thresholds": config.to_dict(),
    }
    if calibration is not None:
        payload["calibration"] = calibration
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def signals_from_dict(payload: Dict[str, Any]) -> EvidenceSignals:
    return EvidenceSignals(
        review_count=int(payload["review_count"]),
        history_chars=int(payload.get("history_chars", 0)),
        metadata_richness=float(payload["metadata_richness"]),
        sid_top1_score=float(payload["sid_top1_score"]),
        sid_top1_margin=float(payload["sid_top1_margin"]),
        sid_score_std=float(payload.get("sid_score_std", 0.0)),
    )


def compute_metadata_richness(
    interaction_tool,
    candidate_ids: Sequence[str],
    *,
    limit: int = 20,
) -> float:
    """Mean fraction of metadata fields populated per candidate (0–1)."""
    if not candidate_ids:
        return 0.0
    ratios: List[float] = []
    denom = max(len(ITEM_FIELD_KEYS), 1)
    for item_id in candidate_ids[:limit]:
        item = filter_item_fields(interaction_tool.get_item(item_id=item_id))
        if not item:
            ratios.append(0.0)
            continue
        filled = sum(1 for key in ITEM_FIELD_KEYS if item.get(key))
        ratios.append(filled / denom)
    return sum(ratios) / len(ratios)


def compute_sid_confidence(sid_scores: Sequence[float]) -> Tuple[float, float, float]:
    """Return (top1_score, top1_margin, score_std) from candidate SID scores."""
    if not sid_scores:
        return 0.0, 0.0, 0.0
    ordered = sorted(float(score) for score in sid_scores)
    top1 = ordered[-1]
    top2 = ordered[-2] if len(ordered) > 1 else 0.0
    margin = top1 - top2
    mean = sum(ordered) / len(ordered)
    variance = sum((score - mean) ** 2 for score in ordered) / len(ordered)
    return top1, margin, math.sqrt(variance)


@dataclass
class EvidenceContext:
    signals: EvidenceSignals
    sid_scores: Dict[str, float]
    sid_order: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signals": self.signals.to_dict(),
            "sid_order_preview": self.sid_order[:5],
        }


def compute_evidence_signals(
    *,
    interaction_tool,
    user_id: str,
    candidate_ids: Sequence[str],
    history_reviews_text: str,
    sid_scores: Sequence[float],
) -> EvidenceSignals:
    review_count = len(interaction_tool.get_reviews(user_id=user_id))
    metadata_richness = compute_metadata_richness(
        interaction_tool,
        candidate_ids,
    )
    top1, margin, std = compute_sid_confidence(sid_scores)
    return EvidenceSignals(
        review_count=review_count,
        history_chars=len(history_reviews_text or ""),
        metadata_richness=metadata_richness,
        sid_top1_score=top1,
        sid_top1_margin=margin,
        sid_score_std=std,
    )


def build_evidence_context(
    *,
    interaction_tool,
    semantic_id_tool,
    user_id: str,
    candidate_ids: Sequence[str],
    history_reviews_text: str,
    source: Optional[str] = None,
) -> EvidenceContext:
    sid_ranked = semantic_id_tool.rank_candidates_by_sid(
        user_id=user_id,
        candidate_item_ids=list(candidate_ids),
        source=source,
    )
    sid_scores_map = {item_id: score for item_id, score in sid_ranked}
    sid_order = [item_id for item_id, _ in sid_ranked]
    score_values = list(sid_scores_map.values())
    signals = compute_evidence_signals(
        interaction_tool=interaction_tool,
        user_id=user_id,
        candidate_ids=candidate_ids,
        history_reviews_text=history_reviews_text,
        sid_scores=score_values,
    )
    return EvidenceContext(
        signals=signals,
        sid_scores=sid_scores_map,
        sid_order=sid_order,
    )


def _history_dense(signals: EvidenceSignals, config: EvidencePolicyConfig) -> bool:
    return (
        signals.review_count >= config.history_dense_reviews
        or signals.history_chars >= config.history_dense_chars
    )


def _metadata_rich(signals: EvidenceSignals, config: EvidencePolicyConfig) -> bool:
    return signals.metadata_richness >= config.metadata_rich_threshold


def _sid_confident(signals: EvidenceSignals, config: EvidencePolicyConfig) -> bool:
    return (
        signals.sid_top1_margin >= config.sid_margin_threshold
        or signals.sid_top1_score >= config.sid_top1_threshold
    )


def select_modules_quality(
    signals: EvidenceSignals,
    config: EvidencePolicyConfig = DEFAULT_POLICY_CONFIG,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    v4: evidence quality → module set + explainable decision record.

    Rules (in order):
      1. H if any reviews; else omit H
      2. M always (closed 20-candidate listwise)
      3. Skip SID if redundancy_skip_sid and history dense and metadata rich
      4. Else add S_tool if history sparse OR SID confident
    """
    modules: List[str] = []
    if signals.review_count >= 1:
        modules.append("H")
    modules.append("M")

    history_dense = _history_dense(signals, config)
    metadata_rich = _metadata_rich(signals, config)
    sid_confident = _sid_confident(signals, config)
    history_sparse = signals.review_count <= config.sparse_history_reviews

    redundant_sid = (
        config.redundancy_skip_sid and history_dense and metadata_rich
    )
    use_sid_tool = False
    if redundant_sid:
        reason = "skip_sid_redundant_text_evidence"
    elif history_sparse:
        use_sid_tool = True
        reason = "add_sid_sparse_history"
    elif sid_confident:
        use_sid_tool = True
        reason = "add_sid_high_confidence"
    else:
        reason = "skip_sid_low_confidence"

    if use_sid_tool:
        modules.append("S_tool")

    decision = {
        "policy_version": config.policy_version,
        "reason": reason,
        "history_dense": history_dense,
        "metadata_rich": metadata_rich,
        "sid_confident": sid_confident,
        "history_sparse": history_sparse,
        "redundant_sid": redundant_sid,
        "signals": signals.to_dict(),
        "thresholds": config.to_dict(),
    }
    return modules, decision


def select_modules_v3(
    review_count: int,
    condition: Optional[str],
    *,
    sparse_threshold: int = 5,
) -> Tuple[List[str], Dict[str, Any]]:
    """Legacy scenario-conditioned policy (reproducibility baseline)."""
    if review_count == 0:
        modules = ["M", "S_tool"]
    elif condition == "evolving_interest":
        modules = ["H", "M"]
    elif condition == "cold_start_user":
        modules = ["H", "M", "S_tool"]
    elif condition in ("classic", "cold_start_item"):
        modules = (
            ["H", "M", "S_tool"]
            if review_count <= sparse_threshold
            else ["H", "M"]
        )
    elif review_count <= sparse_threshold:
        modules = ["H", "M", "S_tool"]
    else:
        modules = ["H", "M"]
    decision = {
        "policy_version": "v3_scenario",
        "reason": f"scenario_table:{condition}",
        "condition": condition,
    }
    return modules, decision


def select_modules(
    *,
    signals: EvidenceSignals,
    condition: Optional[str],
    policy_mode: str = "quality",
    config: EvidencePolicyConfig = DEFAULT_POLICY_CONFIG,
) -> Tuple[List[str], Dict[str, Any]]:
    if policy_mode == "v3_scenario":
        modules, decision = select_modules_v3(
            signals.review_count,
            condition,
            sparse_threshold=config.sparse_history_reviews,
        )
        decision["signals"] = signals.to_dict()
        return modules, decision
    return select_modules_quality(signals, config)
