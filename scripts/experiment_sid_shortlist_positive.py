#!/usr/bin/env python3
"""Positive-case experiment: SID as catalog shortlister vs LLM bottleneck.

Motivation
----------
Closed-set Track-2 (GT always in a fixed 20) stacks the deck against SID: the
vanilla LLM already *sees* the answer.  A fairer comparison — and the regime
where SID can win — is:

  Universe U of size M (Amazon subset: GT + random catalog negatives)
  Vanilla LLM can only listwise-rank L items (context limit), L << M.
  So vanilla must first form a shortlist of size L.

Protocols compared (CPU; no LLM required for the decision metric)
-----------------------------------------------------------------
  A. Random shortlist of size L from U
       Recall@L = L/M   (even if the LLM ranks that shortlist *perfectly*)
  B. SID tool ranks all of U, keep top-L
       Recall@L = fraction of tasks with GT in SID top-L

Success criterion (pre-registered)
---------------------------------
  SID wins if Recall@L(SID) > L/M  (beats perfect-LLM-on-random-shortlist).
  Secondary: HR@5 / NDCG@5 on the SID ranking of U (tool-alone quality).

This is the situation where SID is useful: *retrieving a shortlist from a
larger Amazon subset that the LLM cannot score in one shot*.

Example:
  OMP_NUM_THREADS=1 python scripts/experiment_sid_shortlist_positive.py \\
    --data_dir websocietysimulator/processed_data \\
    --universe_sizes 50 100 200 500 \\
    --shortlist_size 20 \\
    --max_tasks 200
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ARTIFACT = ROOT / "websocietysimulator/artifacts/amazon_sid_v2"
DEFAULT_TASK_DIR = ROOT / "example/track2/amazon/tasks"
DEFAULT_GT_DIR = ROOT / "example/track2/amazon/groundtruth"
DEFAULT_OUT = ROOT / "results/sid_shortlist_positive"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact_dir", type=Path, default=DEFAULT_ARTIFACT)
    p.add_argument("--task_dir", type=Path, default=DEFAULT_TASK_DIR)
    p.add_argument("--gt_dir", type=Path, default=DEFAULT_GT_DIR)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--universe_sizes", type=int, nargs="+", default=[50, 100, 200, 500])
    p.add_argument("--shortlist_size", type=int, default=20, help="L: size vanilla LLM can rank")
    p.add_argument("--max_tasks", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k_report", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def load_tasks(task_dir: Path, gt_dir: Path, max_tasks: int) -> List[dict]:
    rows = []
    for i in range(max_tasks):
        tp, gp = task_dir / f"task_{i}.json", gt_dir / f"groundtruth_{i}.json"
        if not tp.is_file() or not gp.is_file():
            continue
        task = json.loads(tp.read_text())
        gt = str(json.loads(gp.read_text())["ground truth"])
        cands = list(task["candidate_list"])
        if gt not in cands:
            continue
        rows.append({"task_index": i, "user_id": task["user_id"], "candidate_list": cands, "ground_truth": gt})
    return rows


def load_catalog(artifact_dir: Path) -> List[str]:
    import torch

    ordered = torch.load(artifact_dir / "ordered_item_ids.pt", map_location="cpu")
    if isinstance(ordered, torch.Tensor):
        ordered = ordered.tolist()
    return [str(x) for x in ordered]


def expand_universe(base: Sequence[str], gt: str, m: int, catalog: Sequence[str], rng: random.Random) -> List[str]:
    pool, seen = [], set()
    pool.append(gt)
    seen.add(gt)
    for c in base:
        if c not in seen and len(pool) < m:
            pool.append(c)
            seen.add(c)
    guard = 0
    while len(pool) < m and guard < m * 50:
        guard += 1
        c = catalog[rng.randrange(len(catalog))]
        if c not in seen:
            pool.append(c)
            seen.add(c)
    rng.shuffle(pool)
    return pool


def ndcg_at_k(ranking: Sequence[str], gt: str, k: int) -> float:
    if gt not in ranking[:k]:
        return 0.0
    return 1.0 / math.log2(ranking[:k].index(gt) + 2)


def main() -> None:
    args = parse_args()
    assert args.data_dir.joinpath("review.json").is_file(), f"missing review.json under {args.data_dir}"

    from websocietysimulator.tools.cache_interaction_tool import CacheInteractionTool
    from websocietysimulator.tools.semantic_id_tool import SemanticIDTool

    tasks = load_tasks(args.task_dir, args.gt_dir, args.max_tasks)
    catalog = load_catalog(args.artifact_dir)
    print(f"tasks={len(tasks)}  catalog={len(catalog)}  L={args.shortlist_size}")

    tool = CacheInteractionTool(str(args.data_dir))
    sid = SemanticIDTool(artifact_dirs={"amazon": str(args.artifact_dir)}, default_source="amazon")
    sid.set_interaction_tool(tool)

    L = args.shortlist_size
    rows = []
    print("\n=== SID shortlist vs random shortlist (LLM bottleneck) ===")
    print(f"{'M':>5}  {'R@L_rand':>9}  {'R@L_SID':>9}  {'win?':>5}  {'HR@5_SID':>9}  {'mean_rank':>10}")

    for m in args.universe_sizes:
        if m < L:
            print(f"skip M={m} < L={L}")
            continue
        rng = random.Random(args.seed + m)
        recalls: Dict[int, List[float]] = {k: [] for k in args.k_report}
        recall_L, hr5, ndcg5, ranks = [], [], [], []
        emp_rand_L = []  # empirical: GT in a fresh random shortlist of L

        for t in tasks:
            gt = t["ground_truth"]
            universe = expand_universe(t["candidate_list"], gt, m, catalog, rng)
            ranked = [item for item, _ in sid.rank_candidates_by_sid(t["user_id"], universe, source="amazon")]
            rank = ranked.index(gt) + 1 if gt in ranked else m
            ranks.append(rank)
            recall_L.append(1.0 if rank <= L else 0.0)
            hr5.append(1.0 if rank <= 5 else 0.0)
            ndcg5.append(ndcg_at_k(ranked, gt, 5))
            for k in args.k_report:
                recalls[k].append(1.0 if rank <= k else 0.0)

            short = rng.sample(universe, L)
            emp_rand_L.append(1.0 if gt in short else 0.0)

        r_rand_analytic = L / m
        r_rand_emp = float(np.mean(emp_rand_L))
        r_sid = float(np.mean(recall_L))
        win = r_sid > r_rand_analytic
        row = {
            "universe_size_M": m,
            "shortlist_L": L,
            "n_tasks": len(tasks),
            "recall_L_random_analytic": r_rand_analytic,
            "recall_L_random_empirical": r_rand_emp,
            "recall_L_sid": r_sid,
            "sid_beats_perfect_llm_on_random_shortlist": win,
            "lift_recall_L": r_sid - r_rand_analytic,
            "hr@5_sid": float(np.mean(hr5)),
            "ndcg@5_sid": float(np.mean(ndcg5)),
            "mean_gt_rank_sid": float(np.mean(ranks)),
            "median_gt_rank_sid": float(np.median(ranks)),
        }
        for k in args.k_report:
            row[f"recall@{k}_sid"] = float(np.mean(recalls[k]))
        rows.append(row)
        print(
            f"{m:5d}  {r_rand_analytic:9.3f}  {r_sid:9.3f}  "
            f"{'YES' if win else 'no':>5}  {row['hr@5_sid']:9.3f}  {row['mean_gt_rank_sid']:10.1f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "sid_vs_random_shortlist.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_wins = sum(1 for r in rows if r["sid_beats_perfect_llm_on_random_shortlist"])
    print(f"\nWrote {out}")
    print(f"Positive cells: {n_wins}/{len(rows)} universe sizes where SID Recall@L > L/M.")
    print(
        "\nInterpretation: YES = SID shortlisting beats even a *perfect* LLM that "
        "only ever sees a random L-subset of the Amazon universe (the realistic "
        "vanilla bottleneck when M >> L)."
    )


if __name__ == "__main__":
    main()
