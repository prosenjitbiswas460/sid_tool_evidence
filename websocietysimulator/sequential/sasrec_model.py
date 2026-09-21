"""
PyTorch SASRec (Self-Attentive Sequential Recommendation).

Architecture aligned with kang205/SASRec (ICDM 2018):
https://github.com/kang205/SASRec
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SASRecConfig:
    num_items: int
    maxlen: int = 50
    hidden_units: int = 64
    num_blocks: int = 2
    num_heads: int = 1
    dropout_rate: float = 0.2
    l2_emb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_units)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(-1, -2)
        outputs = self.conv2(self.dropout(F.relu(self.conv1(outputs))))
        outputs = outputs.transpose(-1, -2)
        outputs = self.dropout(outputs)
        return self.layer_norm(outputs + inputs)


class SASRec(nn.Module):
    """Self-attentive sequential recommender over item-id sequences."""

    def __init__(self, config: SASRecConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_units
        self.item_embedding = nn.Embedding(
            config.num_items + 1,
            hidden,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(config.maxlen, hidden)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.attention_layernorms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(config.num_blocks)]
        )
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden,
                    num_heads=config.num_heads,
                    dropout=config.dropout_rate,
                    batch_first=True,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.forward_layernorms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(config.num_blocks)]
        )
        self.forward_layers = nn.ModuleList(
            [
                PointWiseFeedForward(hidden, config.dropout_rate)
                for _ in range(config.num_blocks)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(hidden)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def encode_sequence(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Return hidden states for all positions: [B, L, H]."""
        batch_size, seq_len = input_seq.shape
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0).expand(
            batch_size, seq_len
        )
        seqs = self.item_embedding(input_seq) * math.sqrt(self.config.hidden_units)
        seqs = seqs + self.position_embedding(positions)
        seqs = self.dropout(seqs)

        timeline_mask = (input_seq == 0).unsqueeze(-1)
        attention_mask = self._causal_attention_mask(seq_len, input_seq.device)

        for idx in range(self.config.num_blocks):
            normalized = self.attention_layernorms[idx](seqs)
            attn_output, _ = self.attention_layers[idx](
                normalized,
                normalized,
                normalized,
                attn_mask=attention_mask,
                need_weights=False,
            )
            seqs = seqs + attn_output
            seqs = self.forward_layers[idx](seqs)
            seqs = seqs.masked_fill(timeline_mask, 0.0)

        return self.final_layer_norm(seqs)

    @staticmethod
    def _causal_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def sequence_logits(self, input_seq: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """
        Score candidates using the last non-padded position.

        Parameters
        ----------
        input_seq : [B, L] padded item indices
        candidate_ids : [B, C] candidate item indices
        """
        hidden = self.encode_sequence(input_seq)
        last_indices = self._last_non_pad_indices(input_seq)
        batch_indices = torch.arange(input_seq.size(0), device=input_seq.device)
        seq_repr = hidden[batch_indices, last_indices, :]
        candidate_emb = self.item_embedding(candidate_ids)
        return torch.sum(seq_repr.unsqueeze(1) * candidate_emb, dim=-1)

    @staticmethod
    def _last_non_pad_indices(input_seq: torch.Tensor) -> torch.Tensor:
        non_pad = (input_seq != 0).long()
        lengths = non_pad.sum(dim=1).clamp(min=1)
        return lengths - 1

    def training_step(
        self,
        input_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encode_sequence(input_seq)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        pos_logits = torch.sum(hidden * pos_emb, dim=-1)
        neg_logits = torch.sum(hidden * neg_emb, dim=-1)
        istarget = (pos_items != 0).float()
        loss = (
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        )
        loss = torch.sum(loss) / torch.sum(istarget).clamp(min=1.0)
        if self.config.l2_emb > 0:
            loss = loss + self.config.l2_emb * torch.sum(self.item_embedding.weight ** 2)
        return loss

    def score_candidates(
        self,
        input_seq: torch.Tensor,
        candidate_ids: Sequence[int],
    ) -> torch.Tensor:
        if not candidate_ids:
            return torch.empty(0)
        candidate_tensor = torch.tensor(
            [candidate_ids],
            dtype=torch.long,
            device=input_seq.device,
        )
        logits = self.sequence_logits(input_seq, candidate_tensor)
        return logits.squeeze(0)
