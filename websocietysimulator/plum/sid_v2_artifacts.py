"""Persist SID-v2 artifacts compatible with SemanticIDTool."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import torch


@dataclass
class SidV2BuildConfig:
    item_json: str
    review_json: str
    output_dir: str
    source: str
    embedding_model: str
    codebook_sizes: Sequence[int]
    hidden_dim: int
    latent_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    cooccurrence_weight: float
    embed_batch_size: int
    max_length: int
    max_items: Optional[int]
    max_users: Optional[int]
    device: str
    seed: int
    # --- anti-collapse (opt-in; recorded for reproducibility) ---
    use_ema: bool = False
    ema_decay: float = 0.99
    kmeans_init: bool = False
    kmeans_iters: int = 10
    revive_dead: bool = False
    dead_threshold: float = 1.0
    revive_every: int = 20
    # Leave-last-out for co-occurrence mining (anti-leakage; recorded for repro).
    cooccurrence_holdout_last: int = 0


def add_deduplication_digit(cluster_ids: torch.Tensor) -> torch.Tensor:
    unique_rows, inverse_indices, counts = torch.unique(
        cluster_ids, dim=0, return_inverse=True, return_counts=True
    )
    dedup = torch.zeros(cluster_ids.shape[0], dtype=torch.long)
    duplicate_indices = torch.where(counts > 1)[0]
    for duplicate_idx in duplicate_indices:
        row_indices = torch.where(inverse_indices == duplicate_idx)[0]
        dedup[row_indices] = torch.arange(1, row_indices.numel() + 1)
    return torch.cat([cluster_ids, dedup.unsqueeze(1)], dim=1)


def save_sid_v2_artifacts(
    *,
    output_dir: str,
    item_ids: List[str],
    embeddings: torch.Tensor,
    cluster_ids: torch.Tensor,
    cluster_ids_with_dedup: torch.Tensor,
    model_state: dict,
    build_config: SidV2BuildConfig,
    codebook_sizes: Sequence[int],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    item_id_to_sid = {
        item_id: cluster_ids_with_dedup[index].long()
        for index, item_id in enumerate(item_ids)
    }
    sid_to_item_ids: Dict[str, List[str]] = {}
    for item_id, sid_tensor in item_id_to_sid.items():
        key = ",".join(str(int(value)) for value in sid_tensor.tolist())
        sid_to_item_ids.setdefault(key, []).append(item_id)

    sid_map = cluster_ids_with_dedup.long().t().contiguous()
    torch.save(embeddings, os.path.join(output_dir, "merged_predictions_tensor.pt"))
    torch.save(item_id_to_sid, os.path.join(output_dir, "item_id_to_sid.pt"))
    torch.save(sid_map, os.path.join(output_dir, "sid_map.pt"))
    torch.save(model_state, os.path.join(output_dir, "rq_vae.pt"))
    torch.save(item_ids, os.path.join(output_dir, "ordered_item_ids.pt"))

    with open(os.path.join(output_dir, "sid_to_item_ids.json"), "w", encoding="utf-8") as handle:
        json.dump(sid_to_item_ids, handle, indent=2)

    metadata = asdict(build_config)
    metadata["sid_version"] = "sid_v2"
    metadata["sid_mode"] = "sid_v2"
    metadata["num_items"] = len(item_ids)
    metadata["embedding_dim"] = int(embeddings.shape[1])
    metadata["num_hierarchies"] = len(codebook_sizes)
    metadata["codebook_sizes"] = list(codebook_sizes)
    metadata["num_unique_sid_prefixes"] = int(cluster_ids.unique(dim=0).shape[0])
    metadata["num_collisions"] = int(
        cluster_ids_with_dedup.shape[0] - metadata["num_unique_sid_prefixes"]
    )
    metadata["files"] = {
        "item_id_to_sid": "item_id_to_sid.pt",
        "sid_map": "sid_map.pt",
        "rq_vae": "rq_vae.pt",
        "embeddings": "merged_predictions_tensor.pt",
        "ordered_item_ids": "ordered_item_ids.pt",
        "sid_to_item_ids": "sid_to_item_ids.json",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
