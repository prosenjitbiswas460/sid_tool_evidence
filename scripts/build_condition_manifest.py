#!/usr/bin/env python3
"""Build per-task condition manifests for AgentRecBench-style evaluation slices."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from websocietysimulator.conditions import ConditionConfig, build_condition_manifest

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Processed dataset directory containing review.json",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["amazon", "yelp", "goodreads"],
        help="Dataset name under example/track{1,2}/",
    )
    parser.add_argument(
        "--track",
        type=int,
        required=True,
        choices=[1, 2],
        help="1 = simulation track, 2 = recommendation track",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output manifest path (default: manifests/{dataset}_track{track}.json)",
    )
    parser.add_argument("--cold_start_user_k", type=int, default=5)
    parser.add_argument("--cold_start_item_k", type=int, default=5)
    parser.add_argument("--evolving_interest_days", type=int, default=90)
    parser.add_argument(
        "--evolving_interest_min_recent_reviews",
        type=int,
        default=2,
        help="Min interactions in the last T days to qualify (short-term profile)",
    )
    parser.add_argument(
        "--evolving_interest_min_outside_reviews",
        type=int,
        default=0,
        help="Min older interactions outside the window (0=standard recency slice; 1+=drift subset)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_dir = os.path.join(ROOT, "example", f"track{args.track}", args.dataset, "tasks")
    groundtruth_dir = os.path.join(
        ROOT,
        "example",
        f"track{args.track}",
        args.dataset,
        "groundtruth",
    )
    output_path = args.output or os.path.join(
        ROOT,
        "manifests",
        f"{args.dataset}_track{args.track}.json",
    )

    config = ConditionConfig(
        cold_start_user_k=args.cold_start_user_k,
        cold_start_item_k=args.cold_start_item_k,
        evolving_interest_days=args.evolving_interest_days,
        evolving_interest_min_recent_reviews=args.evolving_interest_min_recent_reviews,
        evolving_interest_min_outside_reviews=args.evolving_interest_min_outside_reviews,
    )
    manifest = build_condition_manifest(
        data_dir=args.data_dir,
        task_dir=task_dir,
        groundtruth_dir=groundtruth_dir,
        dataset=args.dataset,
        track=args.track,
        output_path=output_path,
        config=config,
    )

    print(json.dumps(manifest["condition_counts"], indent=2))
    print(f"Wrote manifest to {output_path}")

    if manifest["condition_counts"].get("evolving_interest", 0) == 0:
        sample = next(
            (task for task in manifest["tasks"] if task.get("task_time") is not None),
            None,
        )
        if sample is None:
            logging.warning(
                "No tasks have a resolvable task_time. Time-based conditions "
                "(evolving_interest) will be empty. For Goodreads, ensure "
                "review.json includes date_added/date_updated or timestamp "
                "(re-run data_process.py after updating it)."
            )


if __name__ == "__main__":
    main()
