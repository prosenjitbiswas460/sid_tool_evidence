#!/usr/bin/env python3
"""Build a disjoint EAA calibration pool from processed Amazon/Yelp data."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from websocietysimulator.conditions import ConditionConfig, build_calibration_pool_manifest

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset", required=True, choices=["amazon", "yelp", "goodreads"])
    parser.add_argument("--track", type=int, default=2, choices=[1, 2])
    parser.add_argument(
        "--benchmark_task_dir",
        default=None,
        help="Official eval tasks (default: example/track{track}/{dataset}/tasks)",
    )
    parser.add_argument(
        "--benchmark_groundtruth_dir",
        default=None,
        help="Official eval groundtruth (default: example/track{track}/{dataset}/groundtruth)",
    )
    parser.add_argument(
        "--calibration_root",
        default=os.path.join(ROOT, "manifests", "calibration"),
        help="Directory root for generated calibration task files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output manifest path (default: manifests/calibration/{dataset}_track{track}_calibration.json)",
    )
    parser.add_argument(
        "--cal_tasks_per_condition",
        type=int,
        default=300,
        help="Sample up to this many disjoint tasks per condition",
    )
    parser.add_argument("--min_history", type=int, default=5)
    parser.add_argument("--num_candidates", type=int, default=20)
    parser.add_argument(
        "--max_candidate_scan",
        type=int,
        default=50000,
        help="Stop scanning synthetic splits after this many candidates",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cold_start_user_k", type=int, default=5)
    parser.add_argument("--cold_start_item_k", type=int, default=5)
    parser.add_argument("--evolving_interest_days", type=int, default=90)
    parser.add_argument("--evolving_interest_min_recent_reviews", type=int, default=2)
    parser.add_argument("--evolving_interest_min_outside_reviews", type=int, default=0)
    parser.add_argument(
        "--evolving_interest_last_n",
        type=int,
        default=None,
        help="Count-based evolving-interest window (last N interactions). "
        "Use for Goodreads when review timestamps are missing.",
    )
    parser.add_argument(
        "--order_mode",
        choices=["timestamp", "count"],
        default="timestamp",
        help="Order user/item histories by parsed timestamps or review.json file order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_task_dir = args.benchmark_task_dir or os.path.join(
        ROOT, "example", f"track{args.track}", args.dataset, "tasks"
    )
    benchmark_groundtruth_dir = args.benchmark_groundtruth_dir or os.path.join(
        ROOT, "example", f"track{args.track}", args.dataset, "groundtruth"
    )
    calibration_task_dir = os.path.join(
        args.calibration_root, f"{args.dataset}_track{args.track}", "tasks"
    )
    calibration_groundtruth_dir = os.path.join(
        args.calibration_root, f"{args.dataset}_track{args.track}", "groundtruth"
    )
    output_path = args.output or os.path.join(
        args.calibration_root,
        f"{args.dataset}_track{args.track}_calibration.json",
    )

    config = ConditionConfig(
        cold_start_user_k=args.cold_start_user_k,
        cold_start_item_k=args.cold_start_item_k,
        evolving_interest_days=args.evolving_interest_days,
        evolving_interest_min_recent_reviews=args.evolving_interest_min_recent_reviews,
        evolving_interest_min_outside_reviews=args.evolving_interest_min_outside_reviews,
        evolving_interest_last_n=args.evolving_interest_last_n,
    )

    manifest = build_calibration_pool_manifest(
        data_dir=args.data_dir,
        benchmark_task_dir=benchmark_task_dir,
        benchmark_groundtruth_dir=benchmark_groundtruth_dir,
        calibration_task_dir=calibration_task_dir,
        calibration_groundtruth_dir=calibration_groundtruth_dir,
        dataset=args.dataset,
        track=args.track,
        output_path=output_path,
        config=config,
        cal_tasks_per_condition=args.cal_tasks_per_condition,
        min_history=args.min_history,
        num_candidates=args.num_candidates,
        max_candidate_scan=args.max_candidate_scan,
        seed=args.seed,
        order_mode=args.order_mode,
    )

    print(json.dumps(manifest["condition_counts"], indent=2))
    print(f"Wrote calibration manifest to {output_path}")
    print(f"Calibration tasks dir: {calibration_task_dir}")


if __name__ == "__main__":
    main()
