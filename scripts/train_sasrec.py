#!/usr/bin/env python3
"""
Train SASRec on processed review.json.

PyTorch reimplementation aligned with kang205/SASRec:
https://github.com/kang205/SASRec
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from websocietysimulator.sequential.sasrec_data import (
    SASRecTrainSampler,
    build_sasrec_data,
    evaluate_leave_one_out,
    save_data_bundle_metadata,
)
from websocietysimulator.sequential.sasrec_model import SASRec, SASRecConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_sasrec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source", default="amazon")
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--l2_emb", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--eval_task_dir",
        default=os.path.join(ROOT, "example", "track2", "amazon", "tasks"),
    )
    parser.add_argument(
        "--eval_groundtruth_dir",
        default=os.path.join(ROOT, "example", "track2", "amazon", "groundtruth"),
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" or (device_arg == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    output_dir: str,
    model: SASRec,
    config: SASRecConfig,
    bundle,
    metrics: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "sasrec.pt"))
    with open(os.path.join(output_dir, "sasrec_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    save_data_bundle_metadata(os.path.join(output_dir, "data_maps.json"), bundle)
    with open(os.path.join(output_dir, "train_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    logger.info("Building SASRec dataset from %s (source=%s)", args.data_dir, args.source)
    bundle = build_sasrec_data(
        data_dir=args.data_dir,
        source=args.source,
        eval_task_dir=args.eval_task_dir,
        eval_groundtruth_dir=args.eval_groundtruth_dir,
    )
    avg_len = sum(len(seq) for seq in bundle.user_train.values()) / max(len(bundle.user_train), 1)
    logger.info(
        "Users=%d items=%d avg train len=%.2f",
        bundle.num_users,
        bundle.num_items,
        avg_len,
    )
    if bundle.num_users == 0 or bundle.num_items == 0:
        raise SystemExit(
            f"Empty SASRec dataset for source={args.source!r} "
            f"(users={bundle.num_users}, items={bundle.num_items}). "
            "Refusing to train a vacuous checkpoint. Check review.json source tags "
            "and timestamps."
        )

    config = SASRecConfig(
        num_items=bundle.num_items,
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        l2_emb=args.l2_emb,
    )
    model = SASRec(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    sampler = SASRecTrainSampler(
        user_train=bundle.user_train,
        num_items=bundle.num_items,
        maxlen=args.maxlen,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    num_batches = len(sampler)
    best_valid = None
    best_metrics = {}

    epoch_bar = tqdm(range(1, args.num_epochs + 1), desc="Training SASRec", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        seen = 0
        batch_bar = tqdm(range(num_batches), desc=f"Epoch {epoch}", leave=False, unit="batch")
        for _ in batch_bar:
            seqs, pos, neg, _ = sampler.next_batch()
            seq_tensor = torch.tensor(seqs, dtype=torch.long, device=device)
            pos_tensor = torch.tensor(pos, dtype=torch.long, device=device)
            neg_tensor = torch.tensor(neg, dtype=torch.long, device=device)
            loss = model.training_step(seq_tensor, pos_tensor, neg_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * seq_tensor.size(0)
            seen += seq_tensor.size(0)
            batch_bar.set_postfix(loss=f"{running_loss / max(seen, 1):.4f}")

        train_loss = running_loss / max(seen, 1)
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            valid_metrics = evaluate_leave_one_out(
                model=model,
                user_train=bundle.user_train,
                user_eval=bundle.user_valid,
                num_items=bundle.num_items,
                maxlen=args.maxlen,
                device=device,
            )
            test_metrics = evaluate_leave_one_out(
                model=model,
                user_train={
                    user_id: bundle.user_train.get(user_id, [])
                    + bundle.user_valid.get(user_id, [])
                    for user_id in bundle.user_test
                    if bundle.user_test.get(user_id)
                },
                user_eval=bundle.user_test,
                num_items=bundle.num_items,
                maxlen=args.maxlen,
                device=device,
            )
            logger.info(
                "Epoch %d train_loss=%.4f valid_ndcg@10=%.4f valid_hr@10=%.4f test_ndcg@10=%.4f test_hr@10=%.4f",
                epoch,
                train_loss,
                valid_metrics["ndcg@10"],
                valid_metrics["hr@10"],
                test_metrics["ndcg@10"],
                test_metrics["hr@10"],
            )
            if best_valid is None or valid_metrics["ndcg@10"] >= best_valid:
                best_valid = valid_metrics["ndcg@10"]
                best_metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid": valid_metrics,
                    "test": test_metrics,
                }
                save_checkpoint(
                    output_dir=os.path.abspath(args.output_dir),
                    model=model,
                    config=config,
                    bundle=bundle,
                    metrics=best_metrics,
                )

    logger.info("Saved best SASRec checkpoint to %s", args.output_dir)
    logger.info("Best metrics: %s", json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
