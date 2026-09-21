#!/usr/bin/env python3
"""Materialize end-to-end shortlist→LLM experiment task dirs (Amazon / Yelp / Goodreads).

For each universe size M and each Track-2 classic task:
  1. Build universe U = {GT} ∪ fillers, |U|=M
  2. Random shortlist L=20 from U  (may omit GT)
  3. SID-tool shortlist = top-L of rank_candidates_by_sid over U

Writes, under --output_root / --dataset:

  {dataset}/m{M}/
    meta.json
    universes.jsonl
    arms/
      rand20_hm/{tasks,groundtruth,manifest.json}
      sid20_hm/...
      sid20_hmt/...   # same shortlists as sid20_hm; eval flags differ

Ground-truth files always store the true GT. Candidate lists are the shortlists
only — if GT is missing, standard HR/NDCG eval yields 0 (compelling miss penalty).

Example:
  OMP_NUM_THREADS=1 python scripts/build_sid_e2e_shortlist_tasks.py \\
    --data_dir websocietysimulator/processed_data \\
    --dataset yelp \\
    --universe_sizes 100 200 500 \\
    --max_tasks 200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("amazon", "yelp", "goodreads")
DEFAULT_OUT = ROOT / "outputs/sid_e2e_shortlist"


def default_artifact(dataset: str) -> Path:
    if dataset == "yelp":
        return ROOT / "websocietysimulator" / "artifacts" / "yelp_sid_v2_fixed"
    return ROOT / "websocietysimulator" / "artifacts" / f"{dataset}_sid_v2"


def default_task_dir(dataset: str) -> Path:
    return ROOT / "example" / "track2" / dataset / "tasks"


def default_gt_dir(dataset: str) -> Path:
    return ROOT / "example" / "track2" / dataset / "groundtruth"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--dataset", choices=DATASETS, default="amazon")
    p.add_argument("--artifact_dir", type=Path, default=None,
                   help="Default: websocietysimulator/artifacts/{dataset}_sid_v2")
    p.add_argument("--task_dir", type=Path, default=None)
    p.add_argument("--gt_dir", type=Path, default=None)
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUT,
                   help="Tasks written under {output_root}/{dataset}/m{M}/")
    p.add_argument("--universe_sizes", type=int, nargs="+", default=[100, 200, 500])
    p.add_argument("--shortlist_size", type=int, default=20)
    p.add_argument("--max_tasks", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.artifact_dir is None:
        args.artifact_dir = default_artifact(args.dataset)
    if args.task_dir is None:
        args.task_dir = default_task_dir(args.dataset)
    if args.gt_dir is None:
        args.gt_dir = default_gt_dir(args.dataset)
    return args


def load_base_tasks(task_dir: Path, gt_dir: Path, max_tasks: int) -> List[dict]:
    rows = []
    for i in range(max_tasks):
        tp, gp = task_dir / f"task_{i}.json", gt_dir / f"groundtruth_{i}.json"
        if not tp.is_file() or not gp.is_file():
            continue
        task = json.loads(tp.read_text())
        gt = str(json.loads(gp.read_text())["ground truth"])
        cands = [str(x) for x in task["candidate_list"]]
        if gt not in cands:
            continue
        rows.append(
            {
                "src_index": i,
                "user_id": task["user_id"],
                "candidate_category": task.get("candidate_category", "product"),
                "base_candidates": cands,
                "ground_truth": gt,
                "loc": task.get("loc", [-1, -1]),
            }
        )
    return rows


def load_catalog(artifact_dir: Path) -> List[str]:
    import torch

    ordered = torch.load(artifact_dir / "ordered_item_ids.pt", map_location="cpu")
    if hasattr(ordered, "tolist"):
        ordered = ordered.tolist()
    return [str(x) for x in ordered]


def expand_universe(
    base: Sequence[str], gt: str, m: int, catalog: Sequence[str], rng: random.Random
) -> List[str]:
    pool, seen = [], set()
    pool.append(gt)
    seen.add(gt)
    for c in base:
        if c not in seen and len(pool) < m:
            pool.append(c)
            seen.add(c)
    guard = 0
    while len(pool) < m and guard < m * 80:
        guard += 1
        c = catalog[rng.randrange(len(catalog))]
        if c not in seen:
            pool.append(c)
            seen.add(c)
    rng.shuffle(pool)
    return pool


def write_arm(arm_dir: Path, rows: List[dict], data_dir: Path, dataset: str) -> Path:
    task_dir = arm_dir / "tasks"
    gt_dir = arm_dir / "groundtruth"
    task_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    for j, r in enumerate(rows):
        (task_dir / f"task_{j}.json").write_text(
            json.dumps(
                {
                    "type": "recommendation",
                    "user_id": r["user_id"],
                    "candidate_category": r["candidate_category"],
                    "candidate_list": r["shortlist"],
                    "loc": r["loc"],
                },
                indent=2,
            )
        )
        (gt_dir / f"groundtruth_{j}.json").write_text(
            json.dumps({"ground truth": r["ground_truth"]}, indent=2)
        )

    n = len(rows)
    manifest = {
        "metadata": {
            "dataset": dataset,
            "track": 2,
            "data_dir": str(data_dir.resolve()),
            "task_dir": str(task_dir.resolve()),
            "groundtruth_dir": str(gt_dir.resolve()),
            "config": {
                "cold_start_user_k": 5,
                "cold_start_item_k": 5,
                "evolving_interest_days": 90,
                "evolving_interest_last_n": None,
                "evolving_interest_min_recent_reviews": 2,
                "evolving_interest_min_outside_reviews": 0,
            },
            "total_tasks": n,
            "experiment": "sid_e2e_shortlist",
        },
        "tasks": [{"index": j, "task_time": None} for j in range(n)],
        "condition_indices": {
            "classic": list(range(n)),
            "cold_start_user": [],
            "cold_start_item": [],
            "evolving_interest": [],
        },
        "condition_counts": {
            "classic": n,
            "cold_start_user": 0,
            "cold_start_item": 0,
            "evolving_interest": 0,
        },
    }
    man_path = arm_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    return man_path


def main() -> None:
    args = parse_args()
    assert (args.data_dir / "review.json").is_file(), f"need review.json in {args.data_dir}"
    assert args.artifact_dir.is_dir(), f"missing artifact dir: {args.artifact_dir}"
    assert args.task_dir.is_dir(), f"missing task dir: {args.task_dir}"
    L = args.shortlist_size
    dataset = args.dataset

    from websocietysimulator.tools.cache_interaction_tool import CacheInteractionTool
    from websocietysimulator.tools.semantic_id_tool import SemanticIDTool

    base_tasks = load_base_tasks(args.task_dir, args.gt_dir, args.max_tasks)
    catalog = load_catalog(args.artifact_dir)
    print(f"dataset={dataset} base_tasks={len(base_tasks)} catalog={len(catalog)} L={L}")
    if not base_tasks:
        raise SystemExit("No eligible base tasks.")

    tool = CacheInteractionTool(str(args.data_dir))
    sid = SemanticIDTool(
        artifact_dirs={dataset: str(args.artifact_dir)}, default_source=dataset
    )
    sid.set_interaction_tool(tool)

    ds_root = args.output_root / dataset
    ds_root.mkdir(parents=True, exist_ok=True)

    for m in args.universe_sizes:
        if m < L:
            raise ValueError(f"universe_size {m} < shortlist_size {L}")
        rng = random.Random(args.seed + m)
        out_m = ds_root / f"m{m}"
        arms_root = out_m / "arms"
        meta_rows = []
        rand_rows, sid_rows = [], []

        n_rand_hit = n_sid_hit = 0
        for t in base_tasks:
            gt = t["ground_truth"]
            universe = expand_universe(t["base_candidates"], gt, m, catalog, rng)
            ranked = [
                item
                for item, _sc in sid.rank_candidates_by_sid(
                    t["user_id"], universe, source=dataset
                )
            ]
            sid_short = ranked[:L]
            rand_short = rng.sample(universe, L)
            sid_rank = ranked.index(gt) + 1 if gt in ranked else m
            rand_hit = gt in rand_short
            sid_hit = gt in sid_short
            n_rand_hit += int(rand_hit)
            n_sid_hit += int(sid_hit)

            common = {
                "src_index": t["src_index"],
                "user_id": t["user_id"],
                "candidate_category": t["candidate_category"],
                "ground_truth": gt,
                "loc": t["loc"],
                "universe_size": m,
                "sid_gt_rank": sid_rank,
                "rand_hit": rand_hit,
                "sid_hit": sid_hit,
                "universe": universe,
                "sid_shortlist": sid_short,
                "rand_shortlist": rand_short,
            }
            meta_rows.append(common)
            rand_rows.append({**t, "shortlist": rand_short, "ground_truth": gt})
            sid_rows.append({**t, "shortlist": sid_short, "ground_truth": gt})

        man_rand = write_arm(arms_root / "rand20_hm", rand_rows, args.data_dir, dataset)
        man_sid = write_arm(arms_root / "sid20_hm", sid_rows, args.data_dir, dataset)
        man_hmt = write_arm(arms_root / "sid20_hmt", sid_rows, args.data_dir, dataset)

        with (out_m / "universes.jsonl").open("w") as f:
            for row in meta_rows:
                f.write(json.dumps(row) + "\n")

        meta = {
            "dataset": dataset,
            "universe_size": m,
            "shortlist_size": L,
            "n_tasks": len(base_tasks),
            "seed": args.seed,
            "recall_L_random": n_rand_hit / len(base_tasks),
            "recall_L_sid": n_sid_hit / len(base_tasks),
            "artifact_dir": str(args.artifact_dir.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "arms": {
                "rand20_hm": {
                    "force_modules": "H,M",
                    "eaa_no_sid_merge": True,
                    "manifest": str(man_rand.resolve()),
                    "desc": "Random L-shortlist → LLM text-only rank",
                },
                "sid20_hm": {
                    "force_modules": "H,M",
                    "eaa_no_sid_merge": True,
                    "manifest": str(man_sid.resolve()),
                    "desc": "SID L-shortlist → LLM text-only rank",
                },
                "sid20_hmt": {
                    "force_modules": "H,M,S_tool",
                    "eaa_no_sid_merge": False,
                    "manifest": str(man_hmt.resolve()),
                    "desc": "SID L-shortlist → LLM + SID scores + merge/fallback",
                },
            },
        }
        (out_m / "meta.json").write_text(json.dumps(meta, indent=2))
        print(
            f"[{dataset}] M={m}: recall@L rand={meta['recall_L_random']:.3f} "
            f"sid={meta['recall_L_sid']:.3f}  → {out_m}"
        )

    print(f"\nDone. Root: {ds_root}")
    print(
        f"Next: DATASET={dataset} bash scripts/run_sid_e2e_shortlist_eval.sh"
    )


if __name__ == "__main__":
    main()
