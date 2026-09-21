#!/usr/bin/env python3
"""Add sasrec20_hm shortlist arm to an existing Regime-C universe build.

Reuses the same per-task universes as rand20_hm / sid20_hm (from universes.jsonl),
ranks each universe with a trained SASRec checkpoint, and writes:

  {output_root}/{dataset}/m{M}/arms/sasrec20_hm/{tasks,groundtruth,manifest.json}

Also updates meta.json with recall_L_sasrec and the new arm entry.

This is the strong sequential baseline for: "why not SASRec top-20 → LLM?"
(LightGCN is not implemented in this repo; SASRec is the available trained
collaborative/sequential shortlister.)

Example:
  python scripts/add_sasrec_shortlist_arm.py \\
    --data_dir websocietysimulator/processed_data \\
    --dataset amazon \\
    --sasrec_model_dir websocietysimulator/artifacts/sasrec/amazon \\
    --output_root outputs/sid_e2e_shortlist \\
    --universe_sizes 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("amazon", "yelp", "goodreads")


def write_arm(arm_dir: Path, rows: list, data_dir: Path, dataset: str) -> Path:
    """Minimal classic-only manifest + task/gt files (same layout as SID arms)."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--dataset", choices=DATASETS, default="amazon")
    p.add_argument("--sasrec_model_dir", type=Path, required=True)
    p.add_argument("--output_root", type=Path, default=ROOT / "outputs/sid_e2e_shortlist")
    p.add_argument("--universe_sizes", type=int, nargs="+", default=[100])
    p.add_argument("--shortlist_size", type=int, default=20)
    p.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    p.add_argument("--max_tasks", type=int, default=None,
                   help="Optional cap when reading universes.jsonl (default: all)")
    return p.parse_args()


def resolve_task_root(output_root: Path, dataset: str) -> Path:
    nested = output_root / dataset
    if any(nested.glob("m*/universes.jsonl")):
        return nested
    if dataset == "amazon" and any(output_root.glob("m*/universes.jsonl")):
        return output_root
    return nested


def main() -> None:
    args = parse_args()
    assert args.sasrec_model_dir.is_dir(), f"missing SASRec dir: {args.sasrec_model_dir}"
    assert (args.data_dir / "review.json").is_file(), f"need review.json in {args.data_dir}"

    from websocietysimulator.tools.cache_interaction_tool import CacheInteractionTool
    from websocietysimulator.sequential.sasrec_runtime import SASRecCheckpoint

    tool = CacheInteractionTool(str(args.data_dir))
    ckpt = SASRecCheckpoint.from_dir(str(args.sasrec_model_dir), device=args.device)
    print(f"Loaded SASRec from {args.sasrec_model_dir} (device={ckpt.device})")

    task_root = resolve_task_root(args.output_root, args.dataset)
    L = args.shortlist_size

    for m in args.universe_sizes:
        uni_path = task_root / f"m{m}" / "universes.jsonl"
        meta_path = task_root / f"m{m}" / "meta.json"
        if not uni_path.is_file():
            raise SystemExit(
                f"Missing {uni_path}. Build Regime-C universes first:\n"
                f"  python scripts/build_sid_e2e_shortlist_tasks.py "
                f"--data_dir {args.data_dir} --dataset {args.dataset} "
                f"--universe_sizes {m}"
            )

        # Always read the FULL universe file so we never truncate it on rewrite.
        all_rows = [json.loads(line) for line in uni_path.open()]
        rows_in = all_rows
        if args.max_tasks is not None:
            rows_in = all_rows[: args.max_tasks]

        sasrec_rows = []
        n_hit = 0
        # Map src index in rows_in back into all_rows for safe rewrite
        for r in rows_in:
            universe = r["universe"]
            gt = r["ground_truth"]
            ranked = ckpt.rank_from_interaction_tool(
                interaction_tool=tool,
                user_id=r["user_id"],
                candidate_item_ids=universe,
                exclude_item_ids=[gt],
                source=args.dataset,
            )
            short = ranked[:L]
            hit = gt in short
            n_hit += int(hit)
            r["sasrec_shortlist"] = short
            r["sasrec_hit"] = hit
            r["sasrec_gt_rank"] = (ranked.index(gt) + 1) if gt in ranked else m
            sasrec_rows.append(
                {
                    "user_id": r["user_id"],
                    "candidate_category": r["candidate_category"],
                    "loc": r["loc"],
                    "shortlist": short,
                    "ground_truth": gt,
                }
            )

        arm_dir = task_root / f"m{m}" / "arms" / "sasrec20_hm"
        man = write_arm(arm_dir, sasrec_rows, args.data_dir, args.dataset)

        # Merge sasrec fields into full universes.jsonl (never drop rows).
        by_src = {(r.get("src_index"), r.get("user_id"), r.get("ground_truth")): r for r in rows_in}
        with uni_path.open("w") as f:
            for r in all_rows:
                key = (r.get("src_index"), r.get("user_id"), r.get("ground_truth"))
                if key in by_src:
                    upd = by_src[key]
                    r["sasrec_shortlist"] = upd.get("sasrec_shortlist")
                    r["sasrec_hit"] = upd.get("sasrec_hit")
                    r["sasrec_gt_rank"] = upd.get("sasrec_gt_rank")
                f.write(json.dumps(r) + "\n")

        recall = n_hit / len(rows_in)
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        meta["recall_L_sasrec"] = recall
        meta["sasrec_model_dir"] = str(args.sasrec_model_dir.resolve())
        meta.setdefault("arms", {})
        meta["arms"]["sasrec20_hm"] = {
            "force_modules": "H,M",
            "eaa_no_sid_merge": True,
            "manifest": str(man.resolve()),
            "desc": "SASRec top-L shortlist → LLM text-only rank",
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        r_rand = meta.get("recall_L_random")
        r_sid = meta.get("recall_L_sid")
        print(
            f"[{args.dataset}] M={m}: n={len(rows_in)}  "
            f"recall@L rand={r_rand} sid={r_sid} sasrec={recall:.3f}  → {arm_dir}"
        )

    print("\nNext: evaluate sasrec20_hm with run_sasrec_shortlist_e2e.sh (or run_condition_eval).")


if __name__ == "__main__":
    main()
