"""Load trained SASRec checkpoints and rank candidate lists."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .sasrec_data import (
    SASRecDataBundle,
    build_padded_sequence,
    build_user_history_indices,
    load_data_bundle_metadata,
)
from .sasrec_model import SASRec, SASRecConfig


@dataclass
class SASRecCheckpoint:
    model: SASRec
    config: SASRecConfig
    item_id_to_idx: Dict[str, int]
    idx_to_item_id: Dict[int, str]
    device: torch.device

    @classmethod
    def from_dir(cls, model_dir: str, device: Optional[str] = None) -> "SASRecCheckpoint":
        model_dir = os.path.abspath(model_dir)
        config_path = os.path.join(model_dir, "sasrec_config.json")
        metadata_path = os.path.join(model_dir, "data_maps.json")
        weights_path = os.path.join(model_dir, "sasrec.pt")
        for path in (config_path, metadata_path, weights_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing SASRec artifact: {path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
        bundle = load_data_bundle_metadata(metadata_path)
        config = SASRecConfig(**config_dict)
        resolved_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model = SASRec(config)
        state_dict = torch.load(weights_path, map_location=resolved_device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(resolved_device)
        model.eval()
        return cls(
            model=model,
            config=config,
            item_id_to_idx=bundle.item_id_to_idx,
            idx_to_item_id=bundle.idx_to_item_id,
            device=resolved_device,
        )

    def rank_candidates(
        self,
        history_item_ids: Sequence[int],
        candidate_item_ids: Sequence[str],
    ) -> Tuple[List[str], Dict[str, float]]:
        candidate_indices: List[int] = []
        candidate_ids: List[str] = []
        unknown: List[str] = []
        for item_id in candidate_item_ids:
            idx = self.item_id_to_idx.get(str(item_id))
            if idx is None:
                unknown.append(str(item_id))
            else:
                candidate_indices.append(idx)
                candidate_ids.append(str(item_id))

        if not candidate_indices:
            return list(candidate_item_ids), {}

        seq = build_padded_sequence(history_item_ids, maxlen=self.config.maxlen)
        input_tensor = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
        candidate_tensor = torch.tensor(
            [candidate_indices],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            logits = self.model.sequence_logits(input_tensor, candidate_tensor).squeeze(0)
        scores = logits.detach().cpu().numpy().tolist()
        ranked_known = [
            item_id
            for item_id, _ in sorted(
                zip(candidate_ids, scores),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]
        final_ranking = ranked_known + [item_id for item_id in unknown if item_id not in ranked_known]
        score_map = {item_id: float(score) for item_id, score in zip(candidate_ids, scores)}
        return final_ranking, score_map

    def rank_from_interaction_tool(
        self,
        interaction_tool,
        user_id: str,
        candidate_item_ids: Sequence[str],
        exclude_item_ids: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
    ) -> List[str]:
        history = build_user_history_indices(
            interaction_tool=interaction_tool,
            user_id=user_id,
            item_id_to_idx=self.item_id_to_idx,
            maxlen=self.config.maxlen,
            exclude_item_ids=exclude_item_ids,
            source=source,
        )
        ranking, _ = self.rank_candidates(
            history_item_ids=history.tolist(),
            candidate_item_ids=candidate_item_ids,
        )
        return ranking
