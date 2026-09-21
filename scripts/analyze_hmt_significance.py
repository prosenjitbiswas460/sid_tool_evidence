#!/usr/bin/env python3
"""
Paired-bootstrap significance for the SID-tool effect (HMT − HM) per cell.

For each (dataset, backbone, condition) pairs HMT vs HM on shared run_index
and reports mean Δ, a 95% bootstrap CI, and a two-sided bootstrap p-value
for NDCG@5 and HR@5.

--backbones takes `label:prefix` tokens; oracle files are
`{prefix}_oracle_hm_*` / `{prefix}_oracle_hmt_*`. Prefix `eaa` matches
unprefixed 7B runs (`eaa_oracle_hm_*`).

--pairs takes `agentA,agentB` tokens for extra paired deltas.

Example:
  python scripts/analyze_hmt_significance.py \
    --predictions_dirs outputs \
    --datasets amazon goodreads yelp \
    --backbones 1.5b:eaa_1_5b 3b:eaa_3b 7b:eaa \
    --output_dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

K = 5
METRICS = [f"ndcg@{K}", f"hr@{K}"]
CONDITIONS = ["classic", "cold_start_user", "cold_start_item", "evolving_interest"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--predictions_dirs", nargs="+", default=["outputs"])
    p.add_argument("--datasets", nargs="+", default=["amazon", "goodreads", "yelp"])
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--track", type=int, default=2)
    p.add_argument("--backbones", nargs="*", default=[],
                   help="label:prefix tokens; HMT-HM computed for each.")
    p.add_argument("--pairs", nargs="*", default=[],
                   help="agentA,agentB tokens; extra paired (A-B) deltas.")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fdr_alpha", type=float, default=0.05,
        help="Benjamini–Hochberg FDR level "
             "(applied within each metric across all reported cells).",
    )
    p.add_argument("--output_dir", default="results/hmt_significance")
    return p.parse_args()


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Monotone BH-adjusted p-values (same length/order as input)."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = int(order[i])
        rank = i + 1
        val = min(prev, float(p_values[idx]) * n / rank)
        adj[idx] = val
        prev = val
    return [float(x) for x in adj]


def find_pred(dirs, agent, dataset, track, condition) -> Optional[str]:
    name = f"{agent}_{dataset}_track{track}_{condition}_predictions.json"
    for d in dirs:
        path = os.path.join(d, name)
        if os.path.isfile(path):
            return path
    return None


def _target(rec: dict) -> Optional[str]:
    gt = rec.get("groundtruth") or {}
    t = gt.get("ground truth") or gt.get("ground_truth")
    return str(t) if t is not None else None


def _hr(r, t): return 1.0 if t in list(r)[:K] else 0.0
def _ndcg(r, t):
    for pos, it in enumerate(list(r)[:K]):
        if it == t:
            return 1.0 / math.log2(pos + 2)
    return 0.0


def load_per_task(path: str) -> Dict[int, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    out: Dict[int, Dict[str, float]] = {}
    for rec in payload.get("records", []):
        t = _target(rec)
        ranking = [str(x) for x in (rec.get("output") or [])]
        if t is None or not ranking:
            continue
        out[int(rec["run_index"])] = {f"ndcg@{K}": _ndcg(ranking, t),
                                      f"hr@{K}": _hr(ranking, t)}
    return out


def paired(a: Dict[int, Dict[str, float]], b: Dict[int, Dict[str, float]],
           metric: str, *, bootstrap: int, rng) -> Optional[dict]:
    common = sorted(set(a) & set(b))
    if len(common) < 5:
        return None
    va = np.array([a[i][metric] for i in common])
    vb = np.array([b[i][metric] for i in common])
    diff = va - vb
    n = len(diff)
    obs = float(diff.mean())
    idx = rng.integers(0, n, size=(bootstrap, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided bootstrap p: reflect the null at 0
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    p = min(p, 1.0)
    verdict = "helps*" if lo > 0 else ("hurts*" if hi < 0 else "ns")
    return {"n": n, "a_mean": float(va.mean()), "b_mean": float(vb.mean()),
            "delta": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "p": p, "verdict": verdict}


def concat(store, src, off) -> int:
    for i, m in src.items():
        store[off + i] = m
    return off + (max(src) + 1 if src else 0)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dirs = [os.path.abspath(d) for d in args.predictions_dirs]

    # build list of comparisons: (label, agentA, agentB) where delta = A - B
    comparisons: List[Tuple[str, str, str]] = []
    for tok in args.backbones:
        label, _, prefix = tok.partition(":")
        comparisons.append((f"HMT-HM[{label}]",
                            f"{prefix}_oracle_hmt", f"{prefix}_oracle_hm"))
    for tok in args.pairs:
        a, _, b = tok.partition(",")
        comparisons.append((f"{a}-{b}", a, b))

    if not comparisons:
        raise SystemExit("Nothing to compare: pass --backbones and/or --pairs.")

    rows: List[dict] = []
    for dataset in args.datasets:
        for label, agent_a, agent_b in comparisons:
            pooled_a: Dict[int, Dict[str, float]] = {}
            pooled_b: Dict[int, Dict[str, float]] = {}
            off = 0
            for cond in args.conditions:
                pa = find_pred(dirs, agent_a, dataset, args.track, cond)
                pb = find_pred(dirs, agent_b, dataset, args.track, cond)
                if not pa or not pb:
                    continue
                ma, mb = load_per_task(pa), load_per_task(pb)
                for metric in METRICS:
                    res = paired(ma, mb, metric, bootstrap=args.bootstrap, rng=rng)
                    if res:
                        rows.append({"dataset": dataset, "comparison": label,
                                     "scope": cond, "metric": metric, **res})
                # accumulate pooled with a shared offset so run_index stays aligned
                nxt = max(concat(pooled_a, ma, off), concat(pooled_b, mb, off))
                off = nxt
            for metric in METRICS:
                res = paired(pooled_a, pooled_b, metric, bootstrap=args.bootstrap, rng=rng)
                if res:
                    rows.append({"dataset": dataset, "comparison": label,
                                 "scope": "POOLED", "metric": metric, **res})

    if not rows:
        raise SystemExit("No comparisons built. Check --predictions_dirs / prefixes.")

    # Benjamini–Hochberg within each metric.
    # Primary inference remains the 95% CI / unadjusted verdict; p_bh is secondary.
    for metric in METRICS:
        idxs = [i for i, r in enumerate(rows) if r["metric"] == metric]
        if not idxs:
            continue
        adj = benjamini_hochberg([rows[i]["p"] for i in idxs])
        for i, p_bh in zip(idxs, adj):
            rows[i]["p_bh"] = p_bh
            d = rows[i]["delta"]
            if p_bh < args.fdr_alpha and d > 0:
                rows[i]["verdict_bh"] = "helps*"
            elif p_bh < args.fdr_alpha and d < 0:
                rows[i]["verdict_bh"] = "hurts*"
            else:
                rows[i]["verdict_bh"] = "ns"

    csv_path = os.path.join(args.output_dir, "hmt_significance.csv")
    fields = ["dataset", "comparison", "scope", "metric", "n",
              "a_mean", "b_mean", "delta", "ci_lo", "ci_hi", "p", "verdict",
              "p_bh", "verdict_bh"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # console
    print("\n" + "=" * 100)
    print("PAIRED BOOTSTRAP  (delta = A - B; helps*/hurts* = 95% CI excludes 0)")
    print(f"BH FDR: p_bh / verdict_bh within each metric (alpha={args.fdr_alpha})")
    print("Prefer pooled + replicated CI effects over isolated cells.")
    print("=" * 100)
    for dataset in args.datasets:
        drows = [r for r in rows if r["dataset"] == dataset]
        if not drows:
            continue
        print(f"\n[{dataset}]")
        print(f"  {'comparison':<16}{'scope':<18}{'metric':<8}{'A':>7}{'B':>7}"
              f"{'delta':>9}{'95% CI':>20}{'p':>8}{'p_bh':>8}  verdict  bh")
        for r in drows:
            ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
            print(f"  {r['comparison']:<16}{r['scope']:<18}{r['metric']:<8}"
                  f"{r['a_mean']:>7.3f}{r['b_mean']:>7.3f}{r['delta']:>+9.3f}"
                  f"{ci:>20}{r['p']:>8.3f}{r.get('p_bh', float('nan')):>8.3f}"
                  f"  {r['verdict']:<7} {r.get('verdict_bh', 'ns')}")
    print(f"\nwrote: {csv_path}")


if __name__ == "__main__":
    main()
