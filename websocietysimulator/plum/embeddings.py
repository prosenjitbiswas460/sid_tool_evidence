"""Frozen encoder embeddings for SID-v2 item vectors."""

from __future__ import annotations

from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from websocietysimulator.plum.item_text import item_to_text


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


@torch.no_grad()
def embed_items(
    items: List[dict],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> Tuple[List[str], torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = T5EncoderModel.from_pretrained(model_name).to(device)
    encoder.eval()

    item_ids: List[str] = []
    embeddings: List[torch.Tensor] = []
    batch_texts: List[str] = []
    batch_ids: List[str] = []

    def flush_batch() -> None:
        if not batch_texts:
            return
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = encoder(**encoded)
        pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"]).cpu()
        item_ids.extend(batch_ids)
        embeddings.append(pooled)
        batch_texts.clear()
        batch_ids.clear()

    for item in tqdm(items, desc="Embedding items"):
        batch_ids.append(str(item["item_id"]))
        batch_texts.append(item_to_text(item))
        if len(batch_texts) >= batch_size:
            flush_batch()
    flush_batch()
    return item_ids, torch.cat(embeddings, dim=0)
