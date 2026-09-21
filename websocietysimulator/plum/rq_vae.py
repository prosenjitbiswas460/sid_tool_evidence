"""Residual-quantized VAE with co-occurrence contrastive alignment (SID-v2 core).

Anti-collapse machinery (opt-in, off by default so legacy artifacts stay
reproducible):
  * k-means codebook initialization from the first training batch,
  * EMA codebook updates (VQ-VAE-2 style) instead of pure gradient updates,
  * dead-code revival (reset rarely-used codes to live encoder outputs).

Yelp collapsed under the legacy path (256^3 codebook but only ~1.8k distinct
SIDs) because tightly-clustered business-text embeddings let a handful of codes
win the argmin forever while the rest received no gradient. The three techniques
above are the standard fixes and restore high codebook utilization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RQVAEConfig:
    input_dim: int
    hidden_dim: int = 512
    latent_dim: int = 256
    codebook_sizes: Tuple[int, ...] = (256, 256, 256)
    commitment_beta: float = 0.25
    cooccurrence_weight: float = 0.1
    cooccurrence_margin: float = 0.2
    # --- anti-collapse (opt-in) ---
    use_ema: bool = False
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    kmeans_init: bool = False
    kmeans_iters: int = 10
    revive_dead: bool = False
    dead_threshold: float = 1.0
    revive_every: int = 20


def _kmeans(data: torch.Tensor, k: int, iters: int) -> torch.Tensor:
    """Lightweight Lloyd's k-means; returns [k, D] centroids on data.device."""
    n, d = data.shape
    if n <= k:
        # Not enough points: sample with replacement and jitter duplicates.
        idx = torch.randint(0, n, (k,), device=data.device)
        centroids = data[idx].clone()
        centroids += 1e-4 * torch.randn_like(centroids)
        return centroids
    idx = torch.randperm(n, device=data.device)[:k]
    centroids = data[idx].clone()
    for _ in range(max(iters, 1)):
        dists = torch.cdist(data, centroids)          # [n, k]
        assign = dists.argmin(dim=1)                  # [n]
        counts = torch.bincount(assign, minlength=k).clamp(min=1).unsqueeze(1)
        new_centroids = torch.zeros_like(centroids)
        new_centroids.index_add_(0, assign, data)
        new_centroids = new_centroids / counts
        # Revive any empty cluster with a random data point.
        empty = (torch.bincount(assign, minlength=k) == 0)
        if bool(empty.any()):
            ridx = torch.randint(0, n, (int(empty.sum()),), device=data.device)
            new_centroids[empty] = data[ridx]
        centroids = new_centroids
    return centroids


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        *,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        revive_dead: bool = False,
        dead_threshold: float = 1.0,
        revive_every: int = 20,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.revive_dead = revive_dead
        self.dead_threshold = dead_threshold
        self.revive_every = max(int(revive_every), 1)

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)
        # EMA / init state carried as buffers so they persist in state_dict.
        self.register_buffer("_initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("embed_avg", self.embedding.weight.data.clone())
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _maybe_init(self, flat: torch.Tensor) -> None:
        if bool(self._initialized):
            return
        if self.kmeans_init:
            centroids = _kmeans(flat.detach(), self.num_embeddings, self.kmeans_iters)
        else:
            n = flat.shape[0]
            idx = torch.randint(0, n, (self.num_embeddings,), device=flat.device)
            centroids = flat[idx].detach().clone()
        self.embedding.weight.data.copy_(centroids)
        self.embed_avg.data.copy_(centroids)
        self.cluster_size.data.fill_(1.0)
        self._initialized.fill_(True)

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        onehot = F.one_hot(indices, self.num_embeddings).type(flat.dtype)  # [N, K]
        batch_cluster = onehot.sum(dim=0)                                  # [K]
        embed_sum = onehot.t() @ flat                                      # [K, D]
        self.cluster_size.mul_(self.ema_decay).add_(batch_cluster, alpha=1 - self.ema_decay)
        self.embed_avg.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.ema_eps) / (n + self.num_embeddings * self.ema_eps) * n
        self.embedding.weight.data.copy_(self.embed_avg / smoothed.unsqueeze(1))

    @torch.no_grad()
    def _revive(self, flat: torch.Tensor) -> None:
        dead = self.cluster_size < self.dead_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return
        n = flat.shape[0]
        ridx = torch.randint(0, n, (n_dead,), device=flat.device)
        samples = flat[ridx].detach()
        self.embedding.weight.data[dead] = samples
        self.embed_avg.data[dead] = samples
        self.cluster_size.data[dead] = 1.0

    # ------------------------------------------------------------------ #
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = inputs.reshape(-1, self.embedding_dim)
        if self.training and (self.use_ema or self.kmeans_init):
            self._maybe_init(flat)

        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view_as(inputs)

        if self.use_ema:
            # Codebook is updated by EMA, not gradient; keep only the encoder
            # commitment term (pull encoder outputs toward the chosen codes).
            commitment = self.beta * F.mse_loss(inputs, quantized.detach())
        else:
            commitment = F.mse_loss(inputs, quantized.detach()) + self.beta * F.mse_loss(
                inputs.detach(), quantized
            )

        if self.training and self.use_ema:
            self._ema_update(flat, indices)
            self._step.add_(1)
            if self.revive_dead and int(self._step) % self.revive_every == 0:
                self._revive(flat)

        quantized = inputs + (quantized - inputs).detach()
        return quantized, commitment, indices.view(inputs.shape[:-1])


class ResidualVQ(nn.Module):
    def __init__(self, latent_dim: int, codebook_sizes: Sequence[int], beta: float, config: "RQVAEConfig | None" = None):
        super().__init__()
        extra = {}
        if config is not None:
            extra = dict(
                use_ema=config.use_ema,
                ema_decay=config.ema_decay,
                ema_eps=config.ema_eps,
                kmeans_init=config.kmeans_init,
                kmeans_iters=config.kmeans_iters,
                revive_dead=config.revive_dead,
                dead_threshold=config.dead_threshold,
                revive_every=config.revive_every,
            )
        self.layers = nn.ModuleList(
            [VectorQuantizer(size, latent_dim, beta=beta, **extra) for size in codebook_sizes]
        )

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = latent
        total_commitment = latent.new_zeros(())
        codes: List[torch.Tensor] = []
        for layer in self.layers:
            quantized, commitment, indices = layer(residual)
            total_commitment = total_commitment + commitment
            codes.append(indices)
            residual = residual - quantized
        stacked = torch.stack(codes, dim=-1)
        return latent - residual, total_commitment, stacked

    def encode_indices(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, _, codes = self.forward(latent)
        return codes


class RQVAE(nn.Module):
    def __init__(self, config: RQVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )
        self.quantizer = ResidualVQ(
            config.latent_dim,
            config.codebook_sizes,
            beta=config.commitment_beta,
            config=config,
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(inputs)
        quantized, commitment, codes = self.quantizer(latent)
        recon = self.decoder(quantized)
        recon_loss = F.mse_loss(recon, inputs)
        return recon_loss, commitment, codes, quantized

    def cooccurrence_loss(
        self,
        embeddings: torch.Tensor,
        anchor_idx: torch.Tensor,
        positive_idx: torch.Tensor,
        negative_idx: torch.Tensor,
    ) -> torch.Tensor:
        anchor = F.normalize(embeddings[anchor_idx], dim=-1)
        positive = F.normalize(embeddings[positive_idx], dim=-1)
        negative = F.normalize(embeddings[negative_idx], dim=-1)
        pos_score = (anchor * positive).sum(dim=-1)
        neg_score = (anchor * negative).sum(dim=-1)
        margin = self.config.cooccurrence_margin
        return F.relu(margin - pos_score + neg_score).mean()

    @torch.no_grad()
    def assign_codes(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        return self.quantizer.encode_indices(latent)
