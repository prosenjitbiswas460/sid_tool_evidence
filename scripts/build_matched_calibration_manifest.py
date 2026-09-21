#!/usr/bin/env python3
"""
Build a *distribution-matched* EAA calibration pool (fast, torch-free).

Motivation
----------
The default calibration pool is severely covariate-shifted from the benchmark:
generate_calibration_candidates emits one task per history split, so long-history
power users dominate and the calibration `review_count` distribution (mean
~200-430) barely overlaps the benchmark (mean ~7.6). A selector trained on it
fails to transfer.

This builder fixes the *sampling*:
  1. density = each user's TOTAL review count (exactly the review_count signal,
     since get_reviews(user_id) returns all of a user's reviews),
  2. iterate users in SHUFFLED order (no file-order bias) and, for each, emit up
     to --max_per_user listwise tasks,
  3. fill per-density-stratum quotas that MATCH the POOLED benchmark density
     histogram (NOT per condition -- rarer conditions correlate with different
     densities, which inflates the pool); structurally unmatchable strata
     (review_count<=1) are zeroed out so the loop terminates promptly,
  4. write task/groundtruth files + manifest to a DISTINCT location; never
     overwrite existing artifacts unless --overwrite.

Negatives are drawn by rejection sampling from the catalog (O(k) per task), not
by filtering the whole catalog per candidate -- this is why it is fast.

Final condition labels/counts are authoritative from build_condition_manifest
(re-derived from the data); the labels used during sampling only balance quotas.

Output manifest is a drop-in for scripts/run_eaa_calibration_oracles.py; run the
oracles with a distinct --prediction_prefix (e.g. eaa_calm) so nothing collides
with the existing eaa_cal_* predictions.

Example:
  python scripts/build_matched_calibration_manifest.py \
    --data_dir websocietysimulator/processed_data --dataset amazon --track 2
"""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("matched_calibration")

CONDITION_NAMES = ("classic", "cold_start_user", "cold_start_item", "evolving_interest")


def load_task_conditions_module():
    """Load task_conditions.py directly, bypassing the torch-heavy package __init__."""
    path = os.path.join(ROOT, "websocietysimulator", "conditions", "task_conditions.py")
    spec = importlib.util.spec_from_file_location("_eaa_task_conditions", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # required so @dataclass can resolve __module__
    spec.loader.exec_module(module)
    return module


TC = load_task_conditions_module()
ConditionConfig = TC.ConditionConfig
build_condition_manifest = TC.build_condition_manifest
parse_review_timestamp = TC.parse_review_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset", required=True, choices=["amazon", "yelp", "goodreads"])
    parser.add_argument("--track", type=int, default=2, choices=[1, 2])
    parser.add_argument("--benchmark_task_dir", default=None)
    parser.add_argument("--benchmark_groundtruth_dir", default=None)
    parser.add_argument(
        "--calibration_root",
        default=os.path.join(ROOT, "manifests", "calibration_matched"),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--cal_tasks_per_condition", type=int, default=300)
    parser.add_argument(
        "--total_tasks", type=int, default=None,
        help="Total matched tasks to emit (distributed across density strata to "
        "match the pooled benchmark histogram). Defaults to the benchmark task count.",
    )
    parser.add_argument("--min_history", type=int, default=1,
                        help="Min unique history items for eligibility (low = include sparse users).")
    parser.add_argument("--num_candidates", type=int, default=20)
    parser.add_argument("--density_bins", type=int, default=8)
    parser.add_argument(
        "--strata_mode",
        choices=["quantile", "integer"],
        default="quantile",
        help="quantile bins (good for less-skewed data) or one bin per exact "
        "review_count up to --integer_cap (better for heavily sparse data e.g. yelp/goodreads).",
    )
    parser.add_argument("--integer_cap", type=int, default=15)
    parser.add_argument("--max_per_user", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cold_start_user_k", type=int, default=5)
    parser.add_argument("--cold_start_item_k", type=int, default=5)
    parser.add_argument("--evolving_interest_days", type=int, default=90)
    parser.add_argument("--evolving_interest_min_recent_reviews", type=int, default=2)
    parser.add_argument("--evolving_interest_min_outside_reviews", type=int, default=0)
    parser.add_argument("--evolving_interest_last_n", type=int, default=None)
    parser.add_argument("--order_mode", choices=["timestamp", "count"], default="timestamp")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Data scans (single pass each; pure stdlib)                                   #
# --------------------------------------------------------------------------- #
def scan_reviews(
    review_path: str, source: str, order_mode: str
) -> Tuple[Dict[str, int], Dict[str, List[Tuple[float, str]]], Dict[str, List[float]]]:
    """Return (user_total_counts, user_events, item_times) in one pass."""
    user_totals: Counter = Counter()
    user_events: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    item_times: Dict[str, List[float]] = defaultdict(list)
    with open(review_path, "r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            review = json.loads(line)
            if review.get("source") != source:
                continue
            user_id = review.get("user_id")
            item_id = review.get("item_id")
            if not user_id or not item_id:
                continue
            user_id, item_id = str(user_id), str(item_id)
            user_totals[user_id] += 1  # density = full review_count signal
            if order_mode == "count":
                timestamp = float(line_index)
            else:
                timestamp = parse_review_timestamp(review)
                if timestamp is None:
                    continue
            user_events[user_id].append((float(timestamp), item_id))
            item_times[item_id].append(float(timestamp))
    for events in user_events.values():
        events.sort(key=lambda row: row[0])
    for times in item_times.values():
        times.sort()
    return dict(user_totals), user_events, item_times


def load_catalog_item_ids(data_dir: str, source: str) -> List[str]:
    item_path = os.path.join(data_dir, "item.json")
    if not os.path.isfile(item_path):
        raise FileNotFoundError(f"Missing item catalog at {item_path}")
    item_ids: List[str] = []
    with open(item_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("source") == source and item.get("item_id"):
                item_ids.append(str(item["item_id"]))
    if not item_ids:
        raise ValueError(f"No items for source={source!r} in {item_path}")
    return item_ids


def load_exclusion_pairs(task_dir: str, groundtruth_dir: str) -> set:
    pairs = set()
    if not (os.path.isdir(task_dir) and os.path.isdir(groundtruth_dir)):
        return pairs
    for filename in os.listdir(task_dir):
        if not (filename.startswith("task_") and filename.endswith(".json")):
            continue
        with open(os.path.join(task_dir, filename), "r", encoding="utf-8") as handle:
            task = json.load(handle)
        gt_path = os.path.join(groundtruth_dir, filename.replace("task_", "groundtruth_"))
        if not os.path.isfile(gt_path) or task.get("type") != "recommendation":
            continue
        with open(gt_path, "r", encoding="utf-8") as handle:
            gt = json.load(handle)
        user_id, gt_item = task.get("user_id"), gt.get("ground truth")
        if user_id and gt_item:
            pairs.add((str(user_id), str(gt_item)))
    return pairs


def benchmark_densities(task_dir: str, user_totals: Dict[str, int]) -> List[int]:
    out: List[int] = []
    for filename in sorted(os.listdir(task_dir)):
        if not (filename.startswith("task_") and filename.endswith(".json")):
            continue
        with open(os.path.join(task_dir, filename), "r", encoding="utf-8") as handle:
            task = json.load(handle)
        if task.get("type") != "recommendation":
            continue
        out.append(user_totals.get(str(task.get("user_id")), 0))
    return out


# --------------------------------------------------------------------------- #
# Condition labels (mirror task_conditions._label_conditions)                  #
# --------------------------------------------------------------------------- #
def label_conditions(user_count, item_count, recent, outside, has_task_time, cfg) -> Dict[str, bool]:
    evolving = (
        recent >= cfg.evolving_interest_min_recent_reviews
        and outside >= cfg.evolving_interest_min_outside_reviews
    )
    if cfg.evolving_interest_last_n is None:
        evolving = has_task_time and evolving
    return {
        "classic": True,
        "cold_start_user": user_count <= cfg.cold_start_user_k,
        "cold_start_item": item_count <= cfg.cold_start_item_k,
        "evolving_interest": evolving,
    }


def unique_history(events: Sequence[Tuple[float, str]]) -> List[str]:
    seen, ordered = set(), []
    for _, item_id in events:
        if item_id not in seen:
            seen.add(item_id)
            ordered.append(item_id)
    return ordered


def window_counts(prefix_events, task_time, cfg) -> Tuple[int, int]:
    if cfg.evolving_interest_last_n is not None:
        total = len(prefix_events)
        recent = min(cfg.evolving_interest_last_n, total)
        return recent, max(0, total - cfg.evolving_interest_last_n)
    cutoff = task_time - float(cfg.evolving_interest_days) * 86400.0
    recent = sum(1 for t, _ in prefix_events if cutoff <= t < task_time)
    outside = sum(1 for t, _ in prefix_events if t < cutoff)
    return recent, outside


# --------------------------------------------------------------------------- #
# Matched sampling                                                             #
# --------------------------------------------------------------------------- #
def stratum_of(value: float, inner_edges: np.ndarray) -> int:
    return int(np.digitize([value], inner_edges)[0])


def sample_matched(
    *,
    user_events, item_times, user_totals, catalog, exclusion_pairs,
    inner_edges, bench_props, min_history, num_candidates,
    max_per_user, total_tasks, seed,
):
    """Match the POOLED benchmark density histogram with per-stratum quotas.

    Condition labels are NOT used for selection (they are re-derived later by
    build_condition_manifest); matching per condition would inflate density
    because rarer conditions correlate with different densities.
    """
    rng = random.Random(seed)
    n_strata = len(inner_edges) + 1
    num_negatives = num_candidates - 1

    # Eligibility requires a user with > min_history unique history items (to hold
    # one out), i.e. review_count >= min_history + 1. With --min_history 0 an
    # rc=1 user yields a valid empty-history cold-start task (target = their only
    # item), which is how the benchmark's rc=1 mass is represented. Zero out only
    # strata that cannot contain any eligible user.
    min_fillable = stratum_of(float(min_history + 1), inner_edges)
    need = [int(round(bench_props[s] * total_tasks)) for s in range(n_strata)]
    for s in range(n_strata):
        if s < min_fillable:
            need[s] = 0
    target = list(need)

    def all_met() -> bool:
        return all(n <= 0 for n in need)

    users = list(user_events.keys())
    rng.shuffle(users)

    used_pairs = set()
    per_user = defaultdict(int)
    selected: List[dict] = []
    achieved = [0] * n_strata
    scanned = 0

    for user_id in users:
        if all_met():
            break
        density = user_totals.get(user_id, 0)
        s = stratum_of(density, inner_edges)
        if need[s] <= 0:
            continue
        events = user_events[user_id]
        history_items = unique_history(events)
        if len(history_items) <= min_history:
            continue
        scanned += 1

        split_indices = list(range(min_history, len(history_items)))
        rng.shuffle(split_indices)
        for split_index in split_indices:
            if per_user[user_id] >= max_per_user or need[s] <= 0:
                break
            item = history_items[split_index]
            pair = (user_id, item)
            if pair in exclusion_pairs or pair in used_pairs:
                continue

            history_set = set(history_items[: split_index + 1])
            negatives = set()
            guard = 0
            while len(negatives) < num_negatives and guard < num_negatives * 50:
                cand = catalog[rng.randrange(len(catalog))]
                guard += 1
                if cand == item or cand in history_set:
                    continue
                negatives.add(cand)
            if len(negatives) < num_negatives:
                continue
            candidate_ids = [item, *negatives]
            rng.shuffle(candidate_ids)

            selected.append(
                {"user_id": user_id, "target_item_id": item,
                 "candidate_item_ids": candidate_ids}
            )
            used_pairs.add(pair)
            per_user[user_id] += 1
            need[s] -= 1
            achieved[s] += 1

    report = {
        "target_per_stratum": target,
        "achieved_per_stratum": achieved,
        "achieved_total": sum(achieved),
        "min_fillable_stratum": int(min_fillable),
    }
    return selected, report, scanned


def write_task_files(tasks: Sequence[dict], task_dir: str, groundtruth_dir: str) -> None:
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)
    # Clear any stale task_*/groundtruth_* files so re-runs don't leave a mixed
    # pool that build_condition_manifest would then (wrongly) index.
    for directory, prefix in ((task_dir, "task_"), (groundtruth_dir, "groundtruth_")):
        for filename in os.listdir(directory):
            if filename.startswith(prefix) and filename.endswith(".json"):
                os.remove(os.path.join(directory, filename))
    for index, task in enumerate(tasks):
        with open(os.path.join(task_dir, f"task_{index}.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "type": "recommendation",
                    "user_id": task["user_id"],
                    "candidate_category": "product",
                    "candidate_list": list(task["candidate_item_ids"]),
                    "loc": [-1, -1],
                },
                handle,
                indent=2,
            )
        with open(os.path.join(groundtruth_dir, f"groundtruth_{index}.json"), "w", encoding="utf-8") as handle:
            json.dump({"ground truth": task["target_item_id"]}, handle, indent=2)


def summarize(values: Sequence[int]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "median": float(np.median(arr)), "p90": float(np.percentile(arr, 90))}


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
        args.calibration_root, f"{args.dataset}_track{args.track}_calibration_matched.json"
    )

    existing = [
        p for p in (output_path, calibration_task_dir)
        if os.path.exists(p) and (os.path.isfile(p) or os.listdir(p))
    ]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite existing artifacts (use --overwrite):\n  " + "\n  ".join(existing)
        )

    cfg = ConditionConfig(
        cold_start_user_k=args.cold_start_user_k,
        cold_start_item_k=args.cold_start_item_k,
        evolving_interest_days=args.evolving_interest_days,
        evolving_interest_min_recent_reviews=args.evolving_interest_min_recent_reviews,
        evolving_interest_min_outside_reviews=args.evolving_interest_min_outside_reviews,
        evolving_interest_last_n=args.evolving_interest_last_n,
    )

    review_path = os.path.join(args.data_dir, "review.json")
    if not os.path.isfile(review_path):
        raise SystemExit(f"Missing {review_path}")

    LOGGER.info("[1/5] Scanning review.json (density + timelines)...")
    user_totals, user_events, item_times = scan_reviews(review_path, args.dataset, args.order_mode)
    LOGGER.info("      users=%d  items=%d", len(user_totals), len(item_times))

    LOGGER.info("[2/5] Reading benchmark density distribution...")
    bench_dens = benchmark_densities(benchmark_task_dir, user_totals)
    if not bench_dens:
        raise SystemExit(f"No benchmark recommendation tasks in {benchmark_task_dir}")
    if args.strata_mode == "integer":
        # one bin per exact review_count in [2 .. integer_cap], plus a >cap tail;
        # (review_count<=1 is unmatchable since a task must hold out an item).
        inner_edges = np.arange(1.5, args.integer_cap + 0.5 + 1e-9, 1.0)
    else:
        qs = np.linspace(0.0, 1.0, args.density_bins + 1)
        edges = np.unique(np.quantile(np.asarray(bench_dens, dtype=float), qs))
        inner_edges = edges[1:-1] if len(edges) > 2 else np.array([float(np.median(bench_dens))])
    n_strata = len(inner_edges) + 1
    bench_bins = np.digitize(bench_dens, inner_edges)
    bench_props = np.array([(bench_bins == s).mean() for s in range(n_strata)], dtype=float)

    LOGGER.info("[3/5] Loading item catalog + benchmark exclusion pairs...")
    catalog = load_catalog_item_ids(args.data_dir, args.dataset)
    exclusion_pairs = load_exclusion_pairs(benchmark_task_dir, benchmark_groundtruth_dir)

    total_tasks = args.total_tasks if args.total_tasks is not None else len(bench_dens)

    LOGGER.info("[4/5] Density-matched sampling (shuffled, bounded)...")
    selected, report, scanned = sample_matched(
        user_events=user_events, item_times=item_times, user_totals=user_totals,
        catalog=catalog, exclusion_pairs=exclusion_pairs, inner_edges=inner_edges,
        bench_props=bench_props, min_history=args.min_history,
        num_candidates=args.num_candidates, max_per_user=args.max_per_user,
        total_tasks=total_tasks, seed=args.seed + 1,
    )
    if not selected:
        raise SystemExit("Selected 0 tasks; loosen constraints (min_history/max_per_user).")
    LOGGER.info("      scanned %d eligible users -> %d tasks", scanned, len(selected))

    matched_dens = [user_totals.get(t["user_id"], 0) for t in selected]

    LOGGER.info("[5/5] Writing task files + manifest...")
    write_task_files(selected, calibration_task_dir, calibration_groundtruth_dir)
    manifest = build_condition_manifest(
        data_dir=args.data_dir, task_dir=calibration_task_dir,
        groundtruth_dir=calibration_groundtruth_dir, dataset=args.dataset,
        track=args.track, output_path=output_path, config=cfg,
    )
    manifest["metadata"].update(
        {
            "manifest_kind": "eaa_calibration_pool_matched",
            "benchmark_task_dir": os.path.abspath(benchmark_task_dir),
            "benchmark_groundtruth_dir": os.path.abspath(benchmark_groundtruth_dir),
            "matching_protocol": (
                "Stratified by per-user total review_count to match the benchmark "
                "density histogram; shuffled users, per-user cap for diversity."
            ),
            "density_signal": "review_count (user total reviews)",
            "density_bins": args.density_bins,
            "density_inner_edges": [float(e) for e in inner_edges],
            "benchmark_density_proportions": [float(p) for p in bench_props],
            "benchmark_density_summary": summarize(bench_dens),
            "matched_density_summary": summarize(matched_dens),
            "match_report": report,
            "max_per_user": args.max_per_user,
            "min_history": args.min_history,
            "cal_tasks_per_condition": args.cal_tasks_per_condition,
            "excluded_benchmark_pairs": len(exclusion_pairs),
            "eligible_users_scanned": scanned,
            "calibration_tasks_written": len(selected),
            "seed": args.seed,
            "order_mode": args.order_mode,
        }
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    bsum, msum = summarize(bench_dens), summarize(matched_dens)
    print("\n" + "=" * 70)
    print(f"MATCHED CALIBRATION POOL: {args.dataset} track{args.track}")
    print("=" * 70)
    print(json.dumps(manifest.get("condition_counts", {}), indent=2))
    print(f"\n  density (review_count)   benchmark        matched")
    print(f"    mean               {bsum['mean']:>10.1f}   {msum['mean']:>12.1f}")
    print(f"    median             {bsum['median']:>10.1f}   {msum['median']:>12.1f}")
    print(f"    p90                {bsum['p90']:>10.1f}   {msum['p90']:>12.1f}")
    print("\n  per-stratum fill (density strata, pooled match to benchmark):")
    edges_disp = [f"<= {e:g}" for e in inner_edges] + [f"> {inner_edges[-1]:g}"]
    for s, label in enumerate(edges_disp):
        tgt = report["target_per_stratum"][s]
        got = report["achieved_per_stratum"][s]
        if tgt == 0 and got == 0:
            continue
        print(f"    stratum {s:<2} ({label:<8}) target {tgt:>4}   got {got:>4}")
    print("\n  condition tasks (authoritative, re-derived by manifest):")
    for cond, count in manifest.get("condition_counts", {}).items():
        print(f"    {cond:<20} {count:>4}")
    print(f"\n  tasks written : {len(selected)}")
    print(f"  manifest      : {output_path}")
    print(f"  tasks dir     : {calibration_task_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
