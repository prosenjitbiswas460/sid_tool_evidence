#!/usr/bin/env python3
"""
Item-level rescoring of SID-resolved candidates.

The SID tool ranks candidates by rank_candidates_by_sid, whose dominant term is
`sid_similarity` (coarse hierarchy-prefix overlap). This script re-ranks the
same candidates by embedding cosine to the user's history items, with SID
similarity as a tie-break, and compares NDCG@k for:

    HM       — LLM ranking without the tool
    HMT      — LLM ranking with the tool merged
    sid_tool — the tool's own ranking
    rescore  — embedding rescoring of the same candidates

Example:
  python scripts/analyze_sid_rescoring.py \
    --data_dir websocietysimulator/processed_data \
    --artifact_map amazon:websocietysimulator/artifacts/amazon_sid_v2,\
yelp:websocietysimulator/artifacts/yelp_sid_v2,\
goodreads:websocietysimulator/artifacts/goodreads_sid_v2 \
    --predictions_dir outputs \
    --output_dir results/sid_rescoring
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONDITIONS = ["classic", "cold_start_user", "cold_start_item", "evolving_interest"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data_dir", required=True)
    p.add_argument(
        "--artifact_map", required=True,
        help="Comma-separated dataset:artifact_dir pairs.",
    )
    p.add_argument("--predictions_dir", default="outputs")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--track", type=int, default=2)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--hm_prefix", default="eaa_oracle_hm")
    p.add_argument("--hmt_prefix", default="eaa_oracle_hmt")
    p.add_argument("--max_history_items", type=int, default=20)
    p.add_argument("--emb_topk", type=int, default=3,
                   help="Aggregate the top-k candidate<->history cosine sims.")
    p.add_argument("--output_dir", default="results/sid_rescoring")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# metric + IO helpers                                                          #
# --------------------------------------------------------------------------- #
def ndcg_single(ranking: Sequence[str], target: str, k: int) -> float:
    for pos, item in enumerate(ranking[:k]):
        if item == target:
            return 1.0 / math.log2(pos + 2)
    return 0.0


def load_records(path: str) -> Dict[int, dict]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return {int(r["run_index"]): r for r in payload.get("records", [])}


def target_of(rec: dict) -> Optional[str]:
    gt = rec.get("groundtruth") or {}
    t = gt.get("ground truth") or gt.get("ground_truth")
    return str(t) if t is not None else None


# --------------------------------------------------------------------------- #
# item-level rescoring                                                         #
# --------------------------------------------------------------------------- #
def _unit(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return mat / norm


def rescore_ranking(
    index, candidate_ids: Sequence[str], history_item_ids: Sequence[str],
    emb_topk: int,
) -> Optional[List[str]]:
    """Rank candidates by item-level embedding relevance to history.

    Returns None if embeddings are unavailable for this source.
    """
    emb = index.item_id_to_embedding
    if not emb:
        return None

    hist_vecs = [emb[h].numpy() for h in history_item_ids if h in emb]
    if not hist_vecs:
        return None
    H = _unit(np.vstack(hist_vecs).astype(np.float64))

    scored: List[Tuple[float, float, str]] = []
    for c in candidate_ids:
        cand_sid = index.get_sid(c)
        sid_tie = 0.0
        if cand_sid is not None:
            # coarse SID similarity to nearest history item, used only as tie-break
            sid_tie = max(
                (index.sid_similarity(cand_sid, index.get_sid(h))
                 for h in history_item_ids if index.get_sid(h) is not None),
                default=0.0,
            )
        cv = emb.get(c)
        if cv is None:
            scored.append((-1.0, sid_tie, c))
            continue
        cu = cv.numpy().astype(np.float64)
        cu = cu / (np.linalg.norm(cu) or 1.0)
        sims = H @ cu
        k = min(emb_topk, sims.shape[0])
        top = np.sort(sims)[-k:]
        scored.append((float(top.mean()), sid_tie, c))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [c for _, _, c in scored]


# --------------------------------------------------------------------------- #
# per-cell evaluation                                                          #
# --------------------------------------------------------------------------- #
def eval_cell(
    sid_tool, dataset: str, condition: str, args,
) -> List[dict]:
    hm = load_records(os.path.join(
        args.predictions_dir,
        f"{args.hm_prefix}_{dataset}_track{args.track}_{condition}_predictions.json"))
    hmt = load_records(os.path.join(
        args.predictions_dir,
        f"{args.hmt_prefix}_{dataset}_track{args.track}_{condition}_predictions.json"))
    if not hm:
        return []
    index = sid_tool.indices[dataset]

    rows: List[dict] = []
    for ri, rec in hm.items():
        target = target_of(rec)
        task = rec.get("task", {})
        user_id = task.get("user_id")
        cand = [str(c) for c in task.get("candidate_list", [])]
        if target is None or not user_id or not cand:
            continue

        ndcg_hm = ndcg_single(rec.get("output") or [], target, args.k)
        h = hmt.get(ri)
        ndcg_hmt = ndcg_single((h or {}).get("output") or [], target, args.k) if h else None

        # current SID tool ranking
        try:
            ranked = sid_tool.rank_candidates_by_sid(
                user_id=user_id, candidate_item_ids=cand,
                source=dataset, max_history_items=args.max_history_items)
            sid_rank = [c for c, _ in ranked]
        except Exception:
            sid_rank = cand
        ndcg_sid = ndcg_single(sid_rank, target, args.k)

        # Item-level rescore (needs the user's history item ids)
        try:
            hist = sid_tool.encode_user_history(
                user_id=user_id, max_items=args.max_history_items, source=dataset)
            # Defensive: never allow the held-out target into history (no leakage).
            hist_ids = [str(i) for i in hist.get("item_ids", []) if str(i) != target]
        except Exception:
            hist_ids = []
        rescored = rescore_ranking(index, cand, hist_ids, args.emb_topk)
        ndcg_rescore = ndcg_single(rescored, target, args.k) if rescored else None

        rows.append({
            "dataset": dataset, "condition": condition, "run_index": ri,
            "ndcg_hm": ndcg_hm, "ndcg_hmt": ndcg_hmt,
            "ndcg_sid_tool": ndcg_sid, "ndcg_rescore": ndcg_rescore,
        })
    return rows


def mean(vals: List[Optional[float]]) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def summarize(rows: List[dict], label: str) -> dict:
    n = len(rows)
    hm = mean([r["ndcg_hm"] for r in rows])
    hmt = mean([r["ndcg_hmt"] for r in rows])
    sid = mean([r["ndcg_sid_tool"] for r in rows])
    resc = mean([r["ndcg_rescore"] for r in rows])
    voi_tool = (hmt - hm) if (hm is not None and hmt is not None) else None
    gain = (resc - sid) if (resc is not None and sid is not None) else None
    lift_over_hm = (resc - hm) if (resc is not None and hm is not None) else None
    def f(x): return f"{x:>7.3f}" if x is not None else f"{'--':>7}"
    def s(x): return f"{x:>+7.3f}" if x is not None else f"{'--':>7}"
    print(f"  {label:<26} n={n:<5} HM={f(hm)} HMT={f(hmt)} "
          f"SIDtool={f(sid)} Rescore={f(resc)} | "
          f"VoI={s(voi_tool)} rescore-SID={s(gain)} rescore-HM={s(lift_over_hm)}")
    return {"n": n, "hm": hm, "hmt": hmt, "sid_tool": sid, "rescore": resc,
            "voi_tool": voi_tool, "rescore_minus_sid": gain,
            "rescore_minus_hm": lift_over_hm}


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from websocietysimulator.tools.interaction_tool import InteractionTool
    from websocietysimulator.tools.semantic_id_tool import SemanticIDTool

    amap: Dict[str, str] = {}
    for pair in args.artifact_map.split(","):
        pair = pair.strip()
        if not pair:
            continue
        ds, _, path = pair.partition(":")
        amap[ds.strip()] = os.path.abspath(path.strip())

    print("[load] InteractionTool + SemanticIDTool ...")
    interaction_tool = InteractionTool(data_dir=args.data_dir)
    sid_tool = SemanticIDTool(
        artifact_dirs=amap, interaction_tool=interaction_tool,
        default_source=next(iter(amap)))
    for ds in amap:
        n_emb = len(sid_tool.indices[ds].item_id_to_embedding)
        print(f"       {ds}: embeddings for {n_emb} items "
              f"({'OK' if n_emb else 'MISSING -> rescore unavailable'})")

    all_rows: List[dict] = []
    per_domain: Dict[str, List[dict]] = defaultdict(list)
    for ds in amap:
        for cond in args.conditions:
            rows = eval_cell(sid_tool, ds, cond, args)
            per_domain[ds].extend(rows)
            all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No rows. Check --predictions_dir / prefixes.")

    csv_path = os.path.join(args.output_dir, "sid_rescoring_per_task.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "dataset", "condition", "run_index",
            "ndcg_hm", "ndcg_hmt", "ndcg_sid_tool", "ndcg_rescore"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print("\n" + "=" * 100)
    print("ITEM-LEVEL RESCORING vs CURRENT SID TOOL  (NDCG@%d, standalone tool rankings)" % args.k)
    print("=" * 100)
    summary = {"overall": summarize(all_rows, "POOLED")}
    for ds, rows in per_domain.items():
        print(f"  -- {ds} --")
        summary[ds] = summarize(rows, f"{ds} (all conditions)")
        for cond in args.conditions:
            crows = [r for r in rows if r["condition"] == cond]
            if crows:
                summary[f"{ds}/{cond}"] = summarize(crows, f"{ds}/{cond}")

    print("\nRead: 'rescore-SID' > 0 means item-level rescoring beats the current tool ranking;")
    print("      'rescore-HM'  > 0 means a rescoring tool would BEAT the no-tool LLM (would stop hurting).")

    with open(os.path.join(args.output_dir, "sid_rescoring_summary.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nwrote: {csv_path}")
    print(f"wrote: {os.path.join(args.output_dir, 'sid_rescoring_summary.json')}")


if __name__ == "__main__":
    main()
