#!/usr/bin/env python3
"""
Build SID-v2 artifacts (PLUM stage 1).

RQ-VAE on content embeddings + co-occurrence contrastive alignment from review trajectories.
Writes item_id_to_sid.pt for SemanticIDTool.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

from plum_path import ensure_plum_importable

ensure_plum_importable()

from websocietysimulator.plum.cooccurrence import (
    batch_cooccurrence,
    item_id_to_index,
    load_user_item_sequences,
    mine_cooccurrence_pairs,
    pair_indices,
)
from websocietysimulator.plum.embeddings import embed_items
from websocietysimulator.plum.item_text import load_items_jsonl
from websocietysimulator.plum.rq_vae import RQVAE, RQVAEConfig
from websocietysimulator.plum.sid_v2_artifacts import SidV2BuildConfig, add_deduplication_digit, save_sid_v2_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_sid_v2")


def parse_args() -> SidV2BuildConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item_json", required=True)
    parser.add_argument("--review_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source", default="amazon", choices=["amazon", "yelp", "goodreads"])
    parser.add_argument("--embedding_model", default="google/flan-t5-base")
    parser.add_argument("--codebook_sizes", default="256,256,256")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--cooccurrence_weight", type=float, default=0.1)
    parser.add_argument("--embed_batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    # --- anti-collapse (opt-in) ---
    parser.add_argument("--kmeans_init", action="store_true",
                        help="Initialize each codebook level with k-means on the first batch.")
    parser.add_argument("--kmeans_iters", type=int, default=10)
    parser.add_argument("--ema", dest="use_ema", action="store_true",
                        help="Update codebooks with EMA (VQ-VAE-2 style) instead of gradient.")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--revive_dead", action="store_true",
                        help="Periodically reset rarely-used codes to live encoder outputs.")
    parser.add_argument("--dead_threshold", type=float, default=1.0,
                        help="EMA cluster-size below which a code is considered dead.")
    parser.add_argument("--revive_every", type=int, default=20,
                        help="Revive dead codes every N optimizer steps.")
    parser.add_argument("--cooccurrence_holdout_last", type=int, default=0,
                        help="Drop the last N items of each user trajectory before "
                             "mining co-occurrence pairs (leave-last-out; prevents the "
                             "benchmark's held-out target from leaking into the SID space).")
    args = parser.parse_args()
    sizes = tuple(int(part.strip()) for part in args.codebook_sizes.split(",") if part.strip())
    return SidV2BuildConfig(
        item_json=args.item_json,
        review_json=args.review_json,
        output_dir=args.output_dir,
        source=args.source,
        embedding_model=args.embedding_model,
        codebook_sizes=sizes,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        cooccurrence_weight=args.cooccurrence_weight,
        embed_batch_size=args.embed_batch_size,
        max_length=args.max_length,
        max_items=args.max_items,
        max_users=args.max_users,
        device=args.device,
        seed=args.seed,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        revive_dead=args.revive_dead,
        dead_threshold=args.dead_threshold,
        revive_every=args.revive_every,
        cooccurrence_holdout_last=args.cooccurrence_holdout_last,
    )


def train_rq_vae(
    config: SidV2BuildConfig,
    embeddings: torch.Tensor,
    indexed_pairs: list[tuple[int, int]],
) -> RQVAE:
    torch.manual_seed(config.seed)
    model = RQVAE(
        RQVAEConfig(
            input_dim=embeddings.shape[1],
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            codebook_sizes=config.codebook_sizes,
            cooccurrence_weight=config.cooccurrence_weight,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay,
            kmeans_init=config.kmeans_init,
            kmeans_iters=config.kmeans_iters,
            revive_dead=config.revive_dead,
            dead_threshold=config.dead_threshold,
            revive_every=config.revive_every,
        )
    ).to(config.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(
        TensorDataset(embeddings),
        batch_size=min(config.batch_size, len(embeddings)),
        shuffle=True,
    )
    cooc_iter = batch_cooccurrence(
        indexed_pairs,
        num_items=len(embeddings),
        batch_size=min(512, max(len(indexed_pairs), 1)),
        seed=config.seed,
    )

    # Warm-up init: give k-means / EMA a large population (not just one minibatch)
    # so every residual level's codebook is seeded from real encoder outputs.
    if config.kmeans_init or config.use_ema:
        model.train()
        with torch.no_grad():
            n_init = min(len(embeddings), 8192)
            init_idx = torch.randperm(len(embeddings))[:n_init]
            model(embeddings[init_idx].to(config.device))
        logger.info("Codebook warm-up init from %d encoder outputs", n_init)

    for epoch in range(config.epochs):
        running = 0.0
        for (batch,) in loader:
            batch = batch.to(config.device)
            recon_loss, commitment, _codes, _quantized = model(batch)
            loss = recon_loss + commitment
            if indexed_pairs:
                anchors, positives, negatives = next(cooc_iter)
                anchor_idx = torch.tensor(anchors, device=config.device)
                pos_idx = torch.tensor(positives, device=config.device)
                neg_idx = torch.tensor(negatives, device=config.device)
                with torch.no_grad():
                    latent = model.encode(embeddings.to(config.device))
                cooc = model.cooccurrence_loss(latent, anchor_idx, pos_idx, neg_idx)
                loss = loss + config.cooccurrence_weight * cooc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        logger.info("Epoch %d/%d loss=%.4f", epoch + 1, config.epochs, running / max(len(loader), 1))
    return model


def _report_utilization(codes: torch.Tensor, codebook_sizes) -> None:
    """Print per-level codebook utilization and collision stats (collapse check)."""
    num_items = codes.shape[0]
    logger.info("=" * 60)
    logger.info("CODEBOOK UTILIZATION (higher = healthier, collapse = low)")
    for level, size in enumerate(codebook_sizes):
        used = int(codes[:, level].unique().numel())
        logger.info(
            "  level %d: %d/%d codes used (%.1f%%)",
            level, used, size, 100.0 * used / max(size, 1),
        )
    distinct_prefix = int(codes.unique(dim=0).shape[0])
    _uniq, counts = codes.unique(dim=0, return_counts=True)
    logger.info(
        "  distinct SID prefixes: %d / %d items  (mean_leaf=%.2f, max_leaf=%d)",
        distinct_prefix, num_items,
        num_items / max(distinct_prefix, 1), int(counts.max()),
    )
    logger.info(
        "  collisions needing dedup digit: %d (%.1f%% of items)",
        num_items - distinct_prefix,
        100.0 * (num_items - distinct_prefix) / max(num_items, 1),
    )
    logger.info("=" * 60)


def main() -> None:
    config = parse_args()
    os.makedirs(config.output_dir, exist_ok=True)

    items = load_items_jsonl(config.item_json, source=config.source, max_items=config.max_items)
    item_ids, embeddings = embed_items(
        items=items,
        model_name=config.embedding_model,
        batch_size=config.embed_batch_size,
        max_length=config.max_length,
        device=config.device,
    )

    sequences = load_user_item_sequences(
        config.review_json,
        config.source,
        max_users=config.max_users,
    )
    pairs = mine_cooccurrence_pairs(
        sequences, holdout_last=config.cooccurrence_holdout_last
    )
    lookup = item_id_to_index(item_ids)
    indexed_pairs = pair_indices(pairs, lookup)
    logger.info(
        "Training SID-v2 on %d items, %d co-occurrence pairs",
        len(item_ids),
        len(indexed_pairs),
    )

    model = train_rq_vae(config, embeddings, indexed_pairs)
    model.eval()  # freeze EMA / dead-code revival before final assignment
    with torch.no_grad():
        codes = model.assign_codes(embeddings.to(config.device)).cpu()
    codes_with_dedup = add_deduplication_digit(codes)

    _report_utilization(codes, config.codebook_sizes)

    save_sid_v2_artifacts(
        output_dir=config.output_dir,
        item_ids=item_ids,
        embeddings=embeddings,
        cluster_ids=codes,
        cluster_ids_with_dedup=codes_with_dedup,
        model_state=model.state_dict(),
        build_config=config,
        codebook_sizes=config.codebook_sizes,
    )
    logger.info("Saved SID-v2 artifacts to %s", config.output_dir)


if __name__ == "__main__":
    main()
