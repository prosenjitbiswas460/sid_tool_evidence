#!/usr/bin/env python3
"""Aggregate SID end-to-end shortlist→LLM results with bootstrap CIs.

Primary contrast (compelling attribution):
  sid20_hm − rand20_hm   (same LLM/text-only; only shortlist source differs)

Secondary:
  sid20_hmt − rand20_hm  (full SID pipeline vs random shortlist + text LLM)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("amazon", "yelp", "goodreads")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, default="amazon")
    p.add_argument("--output_root", type=Path, default=ROOT / "outputs/sid_e2e_shortlist")
    p.add_argument("--results_root", type=Path, default=None,
                   help="Default: results/sid_e2e_shortlist/{dataset}/{model_tag}")
    p.add_argument("--predictions_dir", type=Path, default=ROOT / "outputs")
    p.add_argument("--ollama_model", default="qwen2.5:7b-instruct")
    p.add_argument("--universe_sizes", type=int, nargs="+", default=[100, 200, 500])
    p.add_argument(
        "--arms",
        nargs="+",
        default=[
            "rand20_hm",
            "sid20_hm",
            "sid20_hmt",
            "bm25_20_hm",
            "embnn20_hm",
            "sasrec20_hm",
        ],
        help="Arms to load if prediction files exist (missing arms are skipped)",
    )
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.results_root is None:
        args.results_root = (
            ROOT / "results/sid_e2e_shortlist" / args.dataset / model_tag(args.ollama_model)
        )
    return args


def model_tag(model: str) -> str:
    return re.sub(r"[:/]", "_", model)


def ndcg_at_k(pred: Sequence[str], gt: str, k: int) -> float:
    top = list(pred)[:k]
    if gt not in top:
        return 0.0
    return 1.0 / math.log2(top.index(gt) + 2)


def hr_at_k(pred: Sequence[str], gt: str, k: int) -> float:
    return 1.0 if gt in list(pred)[:k] else 0.0


def load_preds(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    records = data["records"] if isinstance(data, dict) and "records" in data else data
    out = []
    for r in records:
        gt = r["groundtruth"]["ground truth"] if isinstance(r.get("groundtruth"), dict) else r.get("groundtruth")
        pred = r.get("output") or []
        if isinstance(pred, dict):
            pred = pred.get("ranking") or pred.get("prediction") or []
        out.append(
            {
                "task_index": r.get("task_index", r.get("run_index")),
                "gt": str(gt),
                "pred": [str(x) for x in pred],
                "error": r.get("error"),
            }
        )
    out.sort(key=lambda x: int(x["task_index"]) if x["task_index"] is not None else 0)
    return out


def resolve_task_root(output_root: Path, dataset: str) -> Path:
    """Prefer nested {output_root}/{dataset}; fall back to legacy amazon flat layout."""
    nested = output_root / dataset
    if (nested / "m100" / "meta.json").is_file() or any(nested.glob("m*/meta.json")):
        return nested
    if dataset == "amazon" and any(output_root.glob("m*/meta.json")):
        return output_root
    return nested


def find_pred(
    predictions_dir: Path, dataset: str, m: int, arm: str, model: str
) -> Optional[Path]:
    """Resolve prediction file; never cross-match other models/datasets."""
    mt = model_tag(model)
    candidates = [
        predictions_dir / f"sid_e2e_{dataset}_m{m}_{arm}_{mt}_predictions.json",
    ]
    if dataset == "amazon":
        candidates.append(
            predictions_dir / f"sid_e2e_m{m}_{arm}_{mt}_predictions.json"
        )
    for p in candidates:
        if p.is_file():
            return p
    return None


def bootstrap_mean_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> Tuple[float, float, float, float]:
    """Return (diff, lo, hi, p_two_sided) for mean(a-b)."""
    d = a - b
    diff = float(d.mean())
    n = len(d)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = d[idx].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2.0 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    p = min(p, 1.0)
    return diff, float(lo), float(hi), p


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    task_root = resolve_task_root(args.output_root, args.dataset)

    summary_rows = []
    pairwise_rows = []

    print(
        f"dataset={args.dataset}  model={args.ollama_model}  task_root={task_root}"
    )
    print(
        f"{'M':>5}  {'arm':<10}  {'R@20':>6}  {'HR@5':>6}  {'NDCG@5':>7}  "
        f"{'n':>4}  pred"
    )

    per_m_metrics: Dict[int, Dict[str, dict]] = {}

    for m in args.universe_sizes:
        meta_path = task_root / f"m{m}" / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        per_m_metrics[m] = {}

        for arm in args.arms:
            pred_path = find_pred(
                args.predictions_dir, args.dataset, m, arm, args.ollama_model
            )
            if pred_path is None:
                print(f"{m:5d}  {arm:<10}  MISSING predictions")
                continue
            recs = load_preds(pred_path)
            hrs, ndcgs, in_short = [], [], []
            for r in recs:
                gt, pred = r["gt"], r["pred"]
                hrs.append(hr_at_k(pred, gt, 5))
                ndcgs.append(ndcg_at_k(pred, gt, 5))
                in_short.append(1.0 if gt in pred else 0.0)

            row = {
                "dataset": args.dataset,
                "model": args.ollama_model,
                "model_tag": model_tag(args.ollama_model),
                "universe_size": m,
                "arm": arm,
                "n_tasks": len(recs),
                "recall_in_ranking": float(np.mean(in_short)),
                "hr@5": float(np.mean(hrs)),
                "ndcg@5": float(np.mean(ndcgs)),
                "predictions": str(pred_path),
                "meta_recall_L_random": meta.get("recall_L_random"),
                "meta_recall_L_sid": meta.get("recall_L_sid"),
            }
            summary_rows.append(row)
            per_m_metrics[m][arm] = {
                "hr": np.asarray(hrs, dtype=float),
                "ndcg": np.asarray(ndcgs, dtype=float),
                "recall": np.asarray(in_short, dtype=float),
                "path": str(pred_path),
            }
            print(
                f"{m:5d}  {arm:<10}  {row['recall_in_ranking']:6.3f}  "
                f"{row['hr@5']:6.3f}  {row['ndcg@5']:7.3f}  {row['n_tasks']:4d}  {pred_path.name}"
            )

        contrasts = [
            ("sid20_hm", "rand20_hm", "sid20_hm - rand20_hm  (SID vs random)"),
            ("sid20_hmt", "rand20_hm", "sid20_hmt - rand20_hm (SID pipeline vs random)"),
            ("bm25_20_hm", "rand20_hm", "bm25_20_hm - rand20_hm (BM25 vs random)"),
            ("bm25_20_hm", "sid20_hm", "bm25_20_hm - sid20_hm (BM25 vs SID)"),
            ("sid20_hm", "bm25_20_hm", "sid20_hm - bm25_20_hm (SID vs BM25)"),
            ("embnn20_hm", "rand20_hm", "embnn20_hm - rand20_hm (EmbNN vs random)"),
            ("embnn20_hm", "sid20_hm", "embnn20_hm - sid20_hm (EmbNN vs SID)"),
            ("sasrec20_hm", "rand20_hm", "sasrec20_hm - rand20_hm (SASRec vs random)"),
            ("sid20_hm", "sasrec20_hm", "sid20_hm - sasrec20_hm (SID vs SASRec)"),
            ("sasrec20_hm", "sid20_hm", "sasrec20_hm - sid20_hm (SASRec vs SID)"),
        ]
        for arm_a, arm_b, label in contrasts:
            a = per_m_metrics[m].get(arm_a)
            b = per_m_metrics[m].get(arm_b)
            if a is None or b is None:
                continue
            n = min(len(a["hr"]), len(b["hr"]))
            for metric in ("hr", "ndcg", "recall"):
                diff, lo, hi, p = bootstrap_mean_diff(
                    a[metric][:n], b[metric][:n], args.n_boot, rng
                )
                helps = diff > 0 and lo > 0
                pairwise_rows.append(
                    {
                        "dataset": args.dataset,
                        "model": args.ollama_model,
                        "model_tag": model_tag(args.ollama_model),
                        "universe_size": m,
                        "contrast": label,
                        "metric": metric,
                        "delta": diff,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "p_boot": p,
                        "compelling_helps": helps,
                        "n": n,
                    }
                )
                flag = "HELPS*" if helps else ("hurts*" if diff < 0 and hi < 0 else "ns")
                print(
                    f"       {label[:44]:<44}  {metric:6}  "
                    f"Δ={diff:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  {flag}"
                )

    if summary_rows:
        with (args.results_root / "summary_table.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    if pairwise_rows:
        with (args.results_root / "pairwise_bootstrap.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
            w.writeheader()
            w.writerows(pairwise_rows)

    n_compelling = sum(
        1 for r in pairwise_rows if r["compelling_helps"] and r["metric"] in ("hr", "ndcg")
    )
    print(f"\nWrote {args.results_root / 'summary_table.csv'}")
    print(f"Wrote {args.results_root / 'pairwise_bootstrap.csv'}")
    print(f"Compelling HELPS* cells (HR/NDCG, CI>0): {n_compelling}")


if __name__ == "__main__":
    main()
