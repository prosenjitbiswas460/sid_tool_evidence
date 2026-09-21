"""
Evidence-Adaptive Agent (EAA): module selection then listwise ranking.

Pipeline (v4 default):
  Estimate evidence quality (history density, metadata richness, SID confidence)
    → Documented selection policy
    → Rank 20 candidates using ONLY selected modules

Legacy v3 scenario policy available via policy_mode='v3_scenario'.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Type

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.llm import LLMBase

from eaa_evidence_policy import (
    DEFAULT_POLICY_CONFIG,
    EvidencePolicyConfig,
    build_evidence_context,
    load_policy_config,
    select_modules,
)
from gr_agent_utils import (
    build_sid_candidate_summaries,
    complete_ranking,
    filter_item_fields,
    merge_ranking,
    parse_ranked_item_list,
    resolve_sid_source,
    truncate_text,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVIDENCE_MODULES: Dict[str, str] = {
    "H": "User profile and review history text.",
    "M": "Candidate item metadata (title, categories, attributes).",
    "S_prompt": "Semantic ID tokens for user history and candidates in the prompt.",
    "S_tool": "Semantic ID tool scores ranking candidates vs user history.",
}


@dataclass(frozen=True)
class EAAConfig:
    agent_id: str = "eaa"
    label: str = "Evidence-Adaptive Agent (evidence quality → rank)"
    llm_candidate_limit: int = 20
    rank_max_tokens: int = 1000
    llm_temperature: float = 0.1
    use_sid_merge_fallback: bool = True
    trust_selection: bool = False
    condition: Optional[str] = None
    policy_mode: str = "quality"
    policy_config: EvidencePolicyConfig = DEFAULT_POLICY_CONFIG
    # Regime-B (evidence-scarce): drop review TEXT from history and present the
    # user's interaction history as bare item titles instead. Metadata (M) and the
    # SID tool (S_tool) are unaffected. Used to test whether adaptive evidence
    # selection matters when free-text reviews are unavailable at inference time.
    no_reviews: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["policy_config"] = self.policy_config.to_dict()
        return payload


class EAAReasoning(ReasoningBase):
    def __init__(self, llm: LLMBase):
        super().__init__(profile_type_prompt="", memory=None, llm=llm)

    def __call__(self, task_description: str, *, max_tokens: int, temperature: float) -> str:
        messages = [{"role": "user", "content": task_description}]
        return self.llm(messages=messages, temperature=temperature, max_tokens=max_tokens)


class EAARecommendationAgent(RecommendationAgent):
    """Select evidence modules from quality signals, then rank the candidate set."""

    eaa_config: EAAConfig = EAAConfig()
    last_trace: Dict[str, Any]

    def __init__(
        self,
        llm: LLMBase,
        *,
        eaa_config: Optional[EAAConfig] = None,
        force_modules: Optional[List[str]] = None,
    ):
        super().__init__(llm=llm)
        self.config = eaa_config or self.eaa_config
        self.force_modules = force_modules
        self.reasoning = EAAReasoning(llm=self.llm)
        self.last_trace = {}

    def _review_free_history(self, user_id: str, *, max_items: int = 30) -> str:
        """Regime-B history: the user's interacted item titles, no review text.

        Uses the same interaction source as the review-rich path (get_reviews gives
        the ordered set of items the user engaged with) but exposes only item
        titles/metadata names, never the free-text review body or rating prose.
        """
        reviews = self.interaction_tool.get_reviews(user_id=user_id) or []
        seen: Set[str] = set()
        titles: List[str] = []
        for review in reviews:
            item_id = review.get("item_id") if isinstance(review, dict) else None
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            item = self.interaction_tool.get_item(item_id=item_id) or {}
            title = (
                item.get("title")
                or item.get("name")
                or item.get("app_name")
                or item_id
            )
            titles.append(str(title).strip())
            if len(titles) >= max_items:
                break
        if not titles:
            return "No prior interaction history is available for this user."
        listed = "; ".join(titles)
        return (
            "The user previously interacted with these items "
            "(titles only; no review text available):\n" + listed
        )

    def workflow(self) -> List[str]:
        if self.semantic_id_tool is None:
            raise RuntimeError(
                "EAARecommendationAgent requires semantic_id_tool. "
                "Pass --artifact_dir to run_condition_eval."
            )

        user_id = self.task["user_id"]
        candidate_list = list(self.task["candidate_list"])
        valid_ids: Set[str] = set(candidate_list)
        source = resolve_sid_source(
            self.interaction_tool,
            candidate_list,
            default_source=self.semantic_id_tool.default_source,
        )

        user_info = truncate_text(str(self.interaction_tool.get_user(user_id=user_id)))
        if self.config.no_reviews:
            history_reviews = truncate_text(self._review_free_history(user_id))
        else:
            history_reviews = truncate_text(
                str(self.interaction_tool.get_reviews(user_id=user_id))
            )

        evidence = build_evidence_context(
            interaction_tool=self.interaction_tool,
            semantic_id_tool=self.semantic_id_tool,
            user_id=user_id,
            candidate_ids=candidate_list,
            history_reviews_text=history_reviews,
            source=source,
        )

        if self.force_modules:
            selected = [m for m in self.force_modules if m in EVIDENCE_MODULES]
            if not selected:
                selected = ["H", "M", "S_tool"]
            decision = {"policy_version": "forced", "reason": "force_modules"}
        else:
            selected, decision = select_modules(
                signals=evidence.signals,
                condition=self.config.condition,
                policy_mode=self.config.policy_mode,
                config=self.config.policy_config,
            )

        sid_scores = evidence.sid_scores
        sid_order = evidence.sid_order

        rank_prompt = self._build_rank_prompt(
            selected_modules=selected,
            user_info=user_info,
            history_reviews=history_reviews,
            candidate_list=candidate_list,
            user_id=user_id,
            source=source,
            sid_scores=sid_scores,
        )
        rank_raw = self.reasoning(
            rank_prompt,
            max_tokens=self.config.rank_max_tokens,
            temperature=self.config.llm_temperature,
        )
        llm_ranked = parse_ranked_item_list(rank_raw, valid_ids)

        uses_sid = "S_tool" in selected or "S_prompt" in selected
        if (
            self.config.use_sid_merge_fallback
            and uses_sid
            and sid_order != list(candidate_list)
        ):
            final_ranking = merge_ranking(llm_ranked, sid_order, valid_ids)
        else:
            final_ranking = complete_ranking(llm_ranked, candidate_list, valid_ids)

        if not final_ranking or len(llm_ranked) < 3:
            # When merge/repair is disabled (--eaa_no_sid_merge), do not silently
            # replace a weak parse with the SID order either — that would reintroduce
            # the post-processing confound the flag is meant to remove.
            use_sid_repair = (
                uses_sid and bool(sid_order) and self.config.use_sid_merge_fallback
            )
            logger.warning(
                "[%s] weak LLM parse (%d ids); falling back to %s",
                self.config.agent_id,
                len(llm_ranked),
                "SID order" if use_sid_repair else "candidate_list order",
            )
            if use_sid_repair:
                final_ranking = complete_ranking(sid_order, candidate_list, valid_ids)
            else:
                final_ranking = complete_ranking(candidate_list, candidate_list, valid_ids)

        self.last_trace = {
            "selected_modules": selected,
            "selection_decision": decision,
            "review_count": evidence.signals.review_count,
            "condition": self.config.condition,
            "policy_mode": self.config.policy_mode,
            "uses_sid_tool": "S_tool" in selected,
            "uses_sid_prompt": "S_prompt" in selected,
            "llm_calls": 1,
            "llm_ranked_count": len(llm_ranked),
            "evidence_signals": evidence.signals.to_dict(),
        }
        logger.info(
            "[%s] modules=%s reason=%s preview=%s",
            self.config.agent_id,
            selected,
            decision.get("reason"),
            final_ranking[:5],
        )
        return final_ranking

    def _build_rank_prompt(
        self,
        *,
        selected_modules: List[str],
        user_info: str,
        history_reviews: str,
        candidate_list: List[str],
        user_id: str,
        source: str,
        sid_scores: Dict[str, float],
    ) -> str:
        blocks: List[str] = []
        if "H" in selected_modules:
            blocks.append(
                f"User profile:\n{user_info}\n\nUser review history:\n{history_reviews}"
            )

        history_sids_block = ""
        sid_summaries_block = ""
        if "S_prompt" in selected_modules:
            history_sids = self.semantic_id_tool.encode_user_history(
                user_id=user_id,
                source=source,
            )
            sid_summaries = build_sid_candidate_summaries(
                semantic_id_tool=self.semantic_id_tool,
                candidate_ids=candidate_list,
                sid_scores=sid_scores,
                source=source,
                top_n=min(10, len(candidate_list)),
            )
            history_sids_block = (
                f"\nUser history encoded as semantic IDs:\n{history_sids}\n"
            )
            sid_summaries_block = (
                f"\nCandidate SID summaries (sid_score):\n{sid_summaries}\n"
            )

        if "S_tool" in selected_modules and "S_prompt" not in selected_modules:
            sid_summaries = build_sid_candidate_summaries(
                semantic_id_tool=self.semantic_id_tool,
                candidate_ids=candidate_list,
                sid_scores=sid_scores,
                source=source,
                top_n=min(10, len(candidate_list)),
            )
            sid_summaries_block = (
                f"\nSemantic-ID tool pre-ranking (higher sid_score is better):\n{sid_summaries}\n"
            )

        item_details = []
        if "M" in selected_modules or "S_prompt" in selected_modules:
            for item_id in candidate_list[: self.config.llm_candidate_limit]:
                item = filter_item_fields(
                    self.interaction_tool.get_item(item_id=item_id)
                )
                entry: Dict[str, Any] = {"item_id": item_id, "item": item}
                if "S_prompt" in selected_modules:
                    entry["sid"] = self.semantic_id_tool.get_sid(item_id, source=source)
                    entry["sid_score"] = sid_scores.get(item_id, 0.0)
                item_details.append(entry)

        evidence_section = "\n".join(blocks)
        if history_sids_block:
            evidence_section += history_sids_block
        if sid_summaries_block:
            evidence_section += sid_summaries_block
        if item_details:
            evidence_section += f"\nDetailed candidate information:\n{item_details}\n"

        if "S_prompt" in selected_modules or "S_tool" in selected_modules:
            prior_instruction = (
                "Use semantic-ID scores as a strong prior, but apply your own judgment using "
                "the review history and item metadata."
            )
        else:
            prior_instruction = (
                "Use the review history and item metadata to rank candidates."
            )

        return f"""
You are a recommendation agent on an online review platform.

Selected evidence modules: {selected_modules}

{evidence_section}

Candidate item IDs (you must rank ONLY these IDs):
{candidate_list}

Task:
Rank all candidate item IDs from most to least relevant for this user.
{prior_instruction}

Output ONLY a Python list of item IDs, with the most recommended item first.
Do not include any item outside the candidate list.
Do not explain your reasoning.

Example format:
['item id1', 'item id2', 'item id3']
"""


def make_eaa_agent_class(
    *,
    force_modules: Optional[List[str]] = None,
    use_sid_merge_fallback: bool = True,
    trust_selection: bool = False,
    condition: Optional[str] = None,
    policy_mode: str = "quality",
    policy_config: Optional[EvidencePolicyConfig] = None,
    no_reviews: bool = False,
) -> Type[EAARecommendationAgent]:
    resolved_policy = policy_config or DEFAULT_POLICY_CONFIG
    config = EAAConfig(
        use_sid_merge_fallback=use_sid_merge_fallback,
        trust_selection=trust_selection,
        condition=condition,
        policy_mode=policy_mode,
        policy_config=resolved_policy,
        no_reviews=no_reviews,
    )
    trace_log: List[Dict[str, Any]] = []

    class _Agent(EAARecommendationAgent):
        eaa_config = config

        def __init__(self, llm: LLMBase):
            super().__init__(
                llm=llm,
                eaa_config=config,
                force_modules=force_modules,
            )

        def workflow(self) -> List[str]:
            ranking = super().workflow()
            trace_log.append(dict(self.last_trace))
            return ranking

    _Agent.__name__ = "EAARecommendationAgent"
    _Agent.__qualname__ = _Agent.__name__
    _Agent.trace_log = trace_log  # type: ignore[attr-defined]
    _Agent.eaa_config_obj = config  # type: ignore[attr-defined]
    return _Agent
