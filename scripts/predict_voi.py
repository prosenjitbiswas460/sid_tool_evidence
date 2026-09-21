#!/usr/bin/env python3
"""
Predict SID-tool value of information (VoI) from pre-admission features.

    VoI(task) = NDCG@k(H,M,S_tool) − NDCG@k(H,M)

Features (all known before the tool is admitted):

  1. backbone capability — text-only NDCG@k or a parameter count
  2. codebook health — collision rate, leaf size, codebook utilization
  3. history sparsity — history_chars, review_count, metadata_richness
  4. candidate discriminability — SID geometry of the candidate set

Fits Ridge and HistGradientBoosting, then evaluates under leave-one-dataset-out,
leave-one-backbone-out, and k-fold CV. Reports cell-level sign accuracy and
Spearman(pred, actual). Torch is used only to read SID artifacts.

Example:
  python scripts/predict_voi.py \
    --backbones 1.5b:outputs_1_5b 3b:outputs_3b 7b:outputs \
    --artifact_map amazon:websocietysimulator/artifacts/amazon_sid_v2,\
yelp:websocietysimulator/artifacts/yelp_sid_v2_fixed,\
goodreads:websocietysimulator/artifacts/goodreads_sid_v2 \
    --output_dir results/voi_predictive
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CONDITIONS = ["classic", "cold_start_user", "cold_start_item", "evolving_interest"]

# feature groups (order matters for reporting)
CAP_FEATS = ["cap", "cap_sq"]
CODEBOOK_FEATS = ["collision_rate", "log_mean_leaf", "level_util"]
SPARSITY_FEATS = ["log_history_chars", "log_review_count", "metadata_richness"]
# Deployable before admission: no ground-truth target required.
DEPLOYABLE_TOOL_FEATS = [
    "sid_top1_margin", "sid_score_std", "candidate_sid_diversity",
]
# Oracle / post-hoc diagnostics only (use ground-truth target SID).
ORACLE_DISCRIM_FEATS = [
    "separation", "mean_prefix_overlap", "frac_exact_collision",
    "candidate_spread", "log_target_leaf_size",
]
# Default modeling features: deployable only (anti-leakage for Tables 3–4).
FEATURES = CAP_FEATS + CODEBOOK_FEATS + SPARSITY_FEATS + DEPLOYABLE_TOOL_FEATS
ALL_CSV_FEATS = FEATURES + ORACLE_DISCRIM_FEATS


# --------------------------------------------------------------------------- #
# args                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--backbones", nargs="+", required=True,
        help="One token PER backbone (separate shell words, NOT one quoted blob). "
        "Format: label:dir:hm_prefix:hmt_prefix  (4 colon-separated fields), e.g. "
        "1.5b:outputs:eaa_oracle_hm_1_5b:eaa_oracle_hmt_1_5b . "
        "A 2-field token label:dir uses the default --hm_prefix/--hmt_prefix "
        "(only valid when backbones live in separate dirs). Capability is measured "
        "from the data by default; use --params to supply param counts instead.",
    )
    p.add_argument(
        "--params", default="",
        help="Optional comma-separated label=value pairs for --capability params, "
        "e.g. '1.5b=1.5,3b=3,7b=7'.",
    )
    p.add_argument(
        "--artifact_map", required=True,
        help="Comma-separated dataset:artifact_dir pairs (codebook health).",
    )
    p.add_argument(
        "--artifact_overrides", default="",
        help="Per-cell SID artifact overrides for when a backbone used a different "
        "SID on a dataset (e.g. yelp fixed vs collapsed). Comma-separated "
        "'backbone/dataset=artifact_dir', e.g. '7b/yelp=artifacts/yelp_sid_v2_fixed'.",
    )
    p.add_argument(
        "--prefix_overrides", default="",
        help="Per-cell prediction-prefix overrides. Comma-separated "
        "'backbone/dataset=hm_prefix:hmt_prefix', e.g. "
        "'7b/yelp=eaa_oracle_hm:eaa_oracle_hmt'.",
    )
    p.add_argument("--datasets", nargs="+", default=["amazon", "yelp", "goodreads"])
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--track", type=int, default=2)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--hm_prefix", default="eaa_oracle_hm")
    p.add_argument("--hmt_prefix", default="eaa_oracle_hmt")
    p.add_argument(
        "--capability", choices=["measured", "params"], default="params",
        help="params = nominal parameter count from --params (default; leak-free). "
        "measured = mean text-only NDCG@k; for OOS protocols this is recomputed "
        "from the training fold only (never from the held-out pool).",
    )
    p.add_argument(
        "--feature_set", choices=["deployable", "all"], default="deployable",
        help="deployable = pre-admission features only (default). "
        "all = also include oracle target-relative SID geometry (diagnostics only).",
    )
    p.add_argument("--protocol", choices=["lodo", "lobo", "cv", "all"], default="all")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/voi_predictive")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# SID artifact -> codebook health                                             #
# --------------------------------------------------------------------------- #
def load_sid_index(artifact_dir: str) -> Tuple[Dict[str, Tuple[int, ...]], Counter, int, Dict]:
    import torch  # local; artifact only

    path = os.path.join(artifact_dir, "item_id_to_sid.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}")
    raw = torch.load(path, map_location="cpu")

    meta: Dict = {}
    meta_path = os.path.join(artifact_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    num_h = meta.get("num_hierarchies")

    item_to_sem: Dict[str, Tuple[int, ...]] = {}
    for item_id, sid_tensor in raw.items():
        sid = [int(v) for v in sid_tensor.tolist()]
        if num_h is None:
            num_h = len(sid) - 1  # last token is the dedup digit
        item_to_sem[str(item_id)] = tuple(sid[:num_h])

    leaf_size: Counter = Counter(item_to_sem.values())
    return item_to_sem, leaf_size, int(num_h), meta


def codebook_health(item_to_sem: Dict[str, Tuple[int, ...]], leaf_size: Counter,
                    num_h: int, meta: Dict) -> Dict[str, float]:
    n_items = len(item_to_sem)
    distinct = len(leaf_size)
    collision_rate = 1.0 - distinct / max(n_items, 1)
    mean_leaf = n_items / max(distinct, 1)

    sizes = meta.get("codebook_sizes")
    utils: List[float] = []
    for lvl in range(num_h):
        used = len({sem[lvl] for sem in item_to_sem.values()})
        if sizes and lvl < len(sizes) and sizes[lvl]:
            utils.append(used / float(sizes[lvl]))
    level_util = float(np.mean(utils)) if utils else float("nan")
    return {
        "collision_rate": float(collision_rate),
        "log_mean_leaf": float(math.log1p(mean_leaf)),
        "level_util": level_util,
        "_n_items": float(n_items),
        "_distinct": float(distinct),
        "_mean_leaf": float(mean_leaf),
        "_max_leaf": float(max(leaf_size.values()) if leaf_size else 0),
    }


# --------------------------------------------------------------------------- #
# prediction-log parsing (VoI + per-task features)                            #
# --------------------------------------------------------------------------- #
def ndcg_single(ranking: Sequence[str], target: str, k: int) -> float:
    for pos, item in enumerate(list(ranking)[:k]):
        if item == target:
            return 1.0 / math.log2(pos + 2)
    return 0.0


def load_preds(path: str) -> Tuple[Dict[int, dict], list]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    out: Dict[int, dict] = {}
    for rec in payload.get("records", []):
        out[int(rec["run_index"])] = rec
    traces = payload.get("metadata", {}).get("eaa_traces") or []
    return out, traces


def target_of(rec: dict) -> Optional[str]:
    gt = rec.get("groundtruth") or {}
    t = gt.get("ground truth") or gt.get("ground_truth")
    return str(t) if t is not None else None


def prefix_overlap(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _f(v) -> Optional[float]:
    return float(v) if v is not None else None


def build_cell_rows(
    backbone: str, dataset: str, condition: str, pdir: str, track: int, k: int,
    hm_prefix: str, hmt_prefix: str,
    item_to_sem: Dict[str, Tuple[int, ...]], leaf_size: Counter, num_h: int,
    health: Dict[str, float],
) -> List[dict]:
    hm_path = os.path.join(pdir, f"{hm_prefix}_{dataset}_track{track}_{condition}_predictions.json")
    hmt_path = os.path.join(pdir, f"{hmt_prefix}_{dataset}_track{track}_{condition}_predictions.json")
    if not (os.path.isfile(hm_path) and os.path.isfile(hmt_path)):
        return []
    hm_recs, hm_traces = load_preds(hm_path)
    hmt_recs, _ = load_preds(hmt_path)

    rows: List[dict] = []
    for ri, hm in hm_recs.items():
        hmt = hmt_recs.get(ri)
        if hmt is None:
            continue
        hm_out, hmt_out = hm.get("output"), hmt.get("output")
        target = target_of(hm)
        if not hm_out or not hmt_out or target is None:
            continue
        ndcg_hm = ndcg_single(hm_out, target, k)
        ndcg_hmt = ndcg_single(hmt_out, target, k)

        row: dict = {
            "backbone": backbone, "dataset": dataset, "condition": condition,
            "run_index": ri, "ndcg_hm": ndcg_hm, "ndcg_hmt": ndcg_hmt,
            "voi": ndcg_hmt - ndcg_hm,
            # codebook (per-dataset, broadcast)
            "collision_rate": health["collision_rate"],
            "log_mean_leaf": health["log_mean_leaf"],
            "level_util": health["level_util"],
        }

        # --- candidate geometry ---
        cand = [str(c) for c in hm.get("task", {}).get("candidate_list", [])]
        cand_in = [c for c in cand if c in item_to_sem]
        # Deployable: diversity among candidates only (no ground-truth target).
        if cand_in:
            sems_c = {item_to_sem[c] for c in cand_in}
            row["candidate_sid_diversity"] = float(len(sems_c) / len(cand_in))
        else:
            row["candidate_sid_diversity"] = None

        # Oracle diagnostics: relative to ground-truth target (not for deployment).
        tgt_sem = item_to_sem.get(target)
        in_idx = [c for c in cand_in if c != target]
        if tgt_sem is not None and in_idx:
            overlaps = [prefix_overlap(tgt_sem, item_to_sem[c]) for c in in_idx]
            n_exact = sum(1 for o in overlaps if o >= num_h)
            sems = {item_to_sem[c] for c in in_idx} | {tgt_sem}
            row["separation"] = float(num_h - max(overlaps))
            row["mean_prefix_overlap"] = float(np.mean(overlaps))
            row["frac_exact_collision"] = float(n_exact / len(overlaps))
            row["candidate_spread"] = float(len(sems) / (len(in_idx) + 1))
            row["log_target_leaf_size"] = float(math.log1p(leaf_size.get(tgt_sem, 1)))
        else:
            for f in ORACLE_DISCRIM_FEATS:
                row[f] = None

        # --- user-side sparsity + tool summary from the HM trace ---
        sig = (hm_traces[ri].get("evidence_signals") if ri < len(hm_traces) else {}) or {}
        hc = _f(sig.get("history_chars"))
        rc = _f(sig.get("review_count"))
        row["log_history_chars"] = math.log1p(hc) if hc is not None else None
        row["log_review_count"] = math.log1p(rc) if rc is not None else None
        row["metadata_richness"] = _f(sig.get("metadata_richness"))
        row["sid_top1_margin"] = _f(sig.get("sid_top1_margin"))
        row["sid_score_std"] = _f(sig.get("sid_score_std"))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# modeling                                                                     #
# --------------------------------------------------------------------------- #
def make_models(seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    linear = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    gbm = HistGradientBoostingRegressor(  # native NaN handling
        max_depth=3, learning_rate=0.05, max_iter=400,
        l2_regularization=1.0, random_state=seed,
    )
    return {"linear": linear, "gbm": gbm}


def matrix(
    rows: Sequence[dict], features: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    feats = list(features) if features is not None else list(FEATURES)
    X = np.array([[r.get(f, np.nan) if r.get(f) is not None else np.nan
                   for f in feats] for r in rows], dtype=float)
    y = np.array([r["voi"] for r in rows], dtype=float)
    return X, y


def _attach_capability(
    rows: List[dict],
    *,
    capability_mode: str,
    params_lookup: Dict[str, Optional[float]],
    train_rows: Optional[Sequence[dict]] = None,
) -> None:
    """Write cap/cap_sq onto rows. For measured mode, use train_rows only."""
    source = list(train_rows) if train_rows is not None else rows
    measured: Dict[str, float] = {}
    for label in {r["backbone"] for r in source}:
        hm = [r["ndcg_hm"] for r in source if r["backbone"] == label]
        measured[label] = float(np.mean(hm)) if hm else 0.0
    for r in rows:
        bb = r["backbone"]
        if capability_mode == "params" and params_lookup.get(bb) is not None:
            cap = float(params_lookup[bb])
        elif bb in measured:
            cap = measured[bb]
        elif params_lookup.get(bb) is not None:
            # LOBO + measured: held-out backbone has no train HM — use params.
            cap = float(params_lookup[bb])
        else:
            cap = 0.0
        r["cap"] = cap
        r["cap_sq"] = cap ** 2


def spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 3 or len(set(a.tolist())) < 2 or len(set(b.tolist())) < 2:
        return None
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)
    except Exception:
        ar = np.argsort(np.argsort(a))
        br = np.argsort(np.argsort(b))
        return float(np.corrcoef(ar, br)[0, 1])


def cell_key(r: dict) -> str:
    return f"{r['backbone']}/{r['dataset']}/{r['condition']}"


def evaluate_split(
    rows: Sequence[dict], train_idx, test_idx, model, seed: int,
    *,
    features: Optional[Sequence[str]] = None,
    capability_mode: str = "params",
    params_lookup: Optional[Dict[str, Optional[float]]] = None,
) -> dict:
    feats = list(features) if features is not None else list(FEATURES)
    params_lookup = params_lookup or {}
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    # Anti-leakage: measured capability from training fold only, then attach to both.
    _attach_capability(
        train_rows + test_rows,
        capability_mode=capability_mode,
        params_lookup=params_lookup,
        train_rows=train_rows,
    )
    Xtr, ytr = matrix(train_rows, feats)
    Xte, yte = matrix(test_rows, feats)
    if len(ytr) < 20 or len(yte) < 5:
        return {}
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    # instance-level
    inst_r = spearman(pred, yte)
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    # cell-level aggregation (the "law" metric)
    agg_pred: Dict[str, List[float]] = defaultdict(list)
    agg_true: Dict[str, List[float]] = defaultdict(list)
    for r, p, t in zip(test_rows, pred, yte):
        agg_pred[cell_key(r)].append(float(p))
        agg_true[cell_key(r)].append(float(t))
    cells = sorted(agg_true)
    cp = np.array([np.mean(agg_pred[c]) for c in cells])
    ct = np.array([np.mean(agg_true[c]) for c in cells])
    cell_r = spearman(cp, ct)
    # sign accuracy over cells whose true mean VoI is non-trivial
    mask = np.abs(ct) > 1e-4
    sign_acc = (float(np.mean(np.sign(cp[mask]) == np.sign(ct[mask])))
                if mask.sum() else None)
    return {
        "n_train": len(ytr), "n_test": len(yte), "n_cells": len(cells),
        "instance_spearman": inst_r, "instance_r2": r2,
        "cell_spearman": cell_r, "cell_sign_acc": sign_acc,
        "cells": {c: {"pred": float(np.mean(agg_pred[c])),
                      "true": float(np.mean(agg_true[c]))} for c in cells},
    }


def run_protocol(
    rows: List[dict], group_key: str, seed: int,
    *,
    features: Optional[Sequence[str]] = None,
    capability_mode: str = "params",
    params_lookup: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, dict]:
    groups = sorted({r[group_key] for r in rows})
    out: Dict[str, dict] = {}
    for model_name, model in make_models(seed).items():
        folds = {}
        pooled_true, pooled_pred = [], []
        for held in groups:
            test_idx = [i for i, r in enumerate(rows) if r[group_key] == held]
            train_idx = [i for i, r in enumerate(rows) if r[group_key] != held]
            res = evaluate_split(
                rows, train_idx, test_idx, make_models(seed)[model_name], seed,
                features=features, capability_mode=capability_mode,
                params_lookup=params_lookup,
            )
            if res:
                folds[held] = {k: v for k, v in res.items() if k != "cells"}
                for c, d in res["cells"].items():
                    pooled_true.append(d["true"]); pooled_pred.append(d["pred"])
        pt, pp = np.array(pooled_true), np.array(pooled_pred)
        mask = np.abs(pt) > 1e-4
        fold_sp = [f["cell_spearman"] for f in folds.values()
                   if f.get("cell_spearman") is not None]
        fold_sa = [f["cell_sign_acc"] for f in folds.values()
                   if f.get("cell_sign_acc") is not None]
        out[model_name] = {
            "folds": folds,
            # mean-of-folds is the generalization metric (each fold = one held-out
            # group); pooled conflates between-group level offsets (Simpson) and is
            # reported only as a secondary diagnostic.
            "mean_fold_spearman": float(np.mean(fold_sp)) if fold_sp else None,
            "mean_fold_sign_acc": float(np.mean(fold_sa)) if fold_sa else None,
            "min_fold_spearman": float(np.min(fold_sp)) if fold_sp else None,
            "pooled_cell_spearman": spearman(pp, pt),
            "pooled_cell_sign_acc": (float(np.mean(np.sign(pp[mask]) == np.sign(pt[mask])))
                                     if mask.sum() else None),
            "pooled_cells_evaluated": int(len(pt)),
        }
    return out


def run_cv(
    rows: List[dict], folds: int, seed: int,
    *,
    features: Optional[Sequence[str]] = None,
    capability_mode: str = "params",
    params_lookup: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, dict]:
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    idx = np.arange(len(rows))
    out: Dict[str, dict] = {}
    for model_name in ("linear", "gbm"):
        inst_rs, cell_signs = [], []
        for tr, te in kf.split(idx):
            res = evaluate_split(
                rows, tr.tolist(), te.tolist(), make_models(seed)[model_name], seed,
                features=features, capability_mode=capability_mode,
                params_lookup=params_lookup,
            )
            if res:
                if res["instance_spearman"] is not None:
                    inst_rs.append(res["instance_spearman"])
                if res["cell_sign_acc"] is not None:
                    cell_signs.append(res["cell_sign_acc"])
        out[model_name] = {
            "mean_instance_spearman": float(np.mean(inst_rs)) if inst_rs else None,
            "mean_cell_sign_acc": float(np.mean(cell_signs)) if cell_signs else None,
        }
    return out


def partial_dependence_capability(
    rows: List[dict], seed: int, features: Optional[Sequence[str]] = None,
) -> List[dict]:
    """PD of predicted VoI vs capability (GBM on all data) -> shows the U-shape."""
    feats = list(features) if features is not None else list(FEATURES)
    X, y = matrix(rows, feats)
    if len(y) < 30 or "cap" not in feats:
        return []
    model = make_models(seed)["gbm"]
    model.fit(X, y)
    cap_col = feats.index("cap")
    capsq_col = feats.index("cap_sq")
    caps = np.linspace(np.nanmin(X[:, cap_col]), np.nanmax(X[:, cap_col]), 25)
    base = np.nanmedian(X, axis=0)
    grid = np.tile(base, (len(caps), 1))
    grid[:, cap_col] = caps
    grid[:, capsq_col] = caps ** 2
    pd_pred = model.predict(grid)
    return [{"cap": float(c), "predicted_voi": float(p)} for c, p in zip(caps, pd_pred)]


def permutation_importance_all(
    rows: List[dict], seed: int, features: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    feats = list(features) if features is not None else list(FEATURES)
    X, y = matrix(rows, feats)
    if len(y) < 40:
        return {}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = make_models(seed)["gbm"]
    model.fit(Xtr, ytr)
    r = permutation_importance(model, Xte, yte, n_repeats=10,
                               random_state=seed, scoring="r2")
    return {feats[i]: float(r.importances_mean[i]) for i in range(len(feats))}


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # parse backbone map: label:dir:hm_prefix:hmt_prefix (or label:dir)
    backbones: List[Tuple[str, str, str, str]] = []
    for tok in args.backbones:
        parts = tok.split(":")
        if len(parts) == 2:
            label, pdir = parts
            hm_pref, hmt_pref = args.hm_prefix, args.hmt_prefix
        elif len(parts) == 4:
            label, pdir, hm_pref, hmt_pref = parts
        else:
            raise SystemExit(
                f"Bad --backbones token '{tok}'. Use label:dir:hm_prefix:hmt_prefix "
                "(4 fields) or label:dir (2 fields). Remember: one token per "
                "backbone as SEPARATE shell words, not one quoted string."
            )
        backbones.append((label, pdir, hm_pref, hmt_pref))
    if len({b[0] for b in backbones}) < len(backbones):
        raise SystemExit("Duplicate backbone labels in --backbones.")

    # optional explicit param counts
    params_lookup: Dict[str, Optional[float]] = {b[0]: None for b in backbones}
    for pair in args.params.split(","):
        pair = pair.strip()
        if not pair:
            continue
        lbl, _, val = pair.partition("=")
        params_lookup[lbl.strip()] = float(val) if val.strip() else None

    # parse artifact map + compute codebook health once per dataset
    amap: Dict[str, str] = {}
    for pair in args.artifact_map.split(","):
        pair = pair.strip()
        if not pair:
            continue
        ds, _, path = pair.partition(":")
        amap[ds.strip()] = path.strip()

    # per-cell overrides
    artifact_ov: Dict[str, str] = {}
    for pair in args.artifact_overrides.split(","):
        pair = pair.strip()
        if pair:
            key, _, val = pair.partition("=")
            artifact_ov[key.strip()] = val.strip()
    prefix_ov: Dict[str, Tuple[str, str]] = {}
    for pair in args.prefix_overrides.split(","):
        pair = pair.strip()
        if pair:
            key, _, val = pair.partition("=")
            hm_p, _, hmt_p = val.partition(":")
            prefix_ov[key.strip()] = (hm_p.strip(), hmt_p.strip())

    # SID index + codebook health cached by artifact PATH (a backbone may use a
    # different SID on a dataset, so health is resolved per (backbone,dataset)).
    sid_by_path: Dict[str, Tuple] = {}
    health_by_path: Dict[str, Dict[str, float]] = {}

    def get_sid(path: str):
        if path not in sid_by_path:
            item_to_sem, leaf_size, num_h, meta = load_sid_index(path)
            sid_by_path[path] = (item_to_sem, leaf_size, num_h)
            h = codebook_health(item_to_sem, leaf_size, num_h, meta)
            health_by_path[path] = h
            print(f"[codebook] {path}: items={int(h['_n_items'])} "
                  f"distinct={int(h['_distinct'])} collision_rate={h['collision_rate']:.3f} "
                  f"mean_leaf={h['_mean_leaf']:.2f} level_util={h['level_util']:.3f}")
        return sid_by_path[path], health_by_path[path]

    # build rows for every (backbone, dataset, condition)
    rows: List[dict] = []
    for label, pdir, hm_pref, hmt_pref in backbones:
        for ds in args.datasets:
            art = artifact_ov.get(f"{label}/{ds}") or amap.get(ds)
            if not art:
                print(f"[warn] no artifact for {label}/{ds} — skipping.")
                continue
            (item_to_sem, leaf_size, num_h), health = get_sid(art)
            hmp, hmtp = prefix_ov.get(f"{label}/{ds}", (hm_pref, hmt_pref))
            for cond in args.conditions:
                cell = build_cell_rows(
                    label, ds, cond, pdir, args.track, args.k,
                    hmp, hmtp, item_to_sem, leaf_size, num_h, health,
                )
                rows.extend(cell)
        n_bb = sum(1 for r in rows if r["backbone"] == label)
        print(f"[rows] backbone {label} ({pdir}, {hm_pref}/{hmt_pref}): {n_bb} tasks")

    if not rows:
        raise SystemExit("No rows built. Check --backbones dirs and --artifact_map.")

    # Coverage: condition–task evaluations (overlapping slices) vs unique classic.
    cov: Counter = Counter((r["backbone"], r["dataset"]) for r in rows)
    classic_unique: Dict[Tuple[str, str], int] = {}
    for bb, ds in sorted({(r["backbone"], r["dataset"]) for r in rows}):
        classic_unique[(bb, ds)] = len({
            r["run_index"] for r in rows
            if r["backbone"] == bb and r["dataset"] == ds and r["condition"] == "classic"
        })
    print("[coverage] condition-task evaluations per backbone/dataset "
          "(conditions overlap; not unique user-item pairs):")
    for (bb, ds), n in sorted(cov.items()):
        u = classic_unique.get((bb, ds), 0)
        print(f"           {bb}/{ds}: {n} condition-tasks "
              f"(unique classic run_index={u})")

    # direct VoI summary per (backbone,dataset) — the headline for contrast runs
    voi_by_cell: Dict[str, dict] = {}
    grp: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        grp[(r["backbone"], r["dataset"])].append(r)
    print("\n[VoI by cell]  mean(HMT-HM), %help/%hurt, mean NDCG@k")
    print(f"  {'backbone/dataset':<24}{'n':>6}{'meanVoI':>10}{'%help':>8}{'%hurt':>8}"
          f"{'HM':>8}{'HMT':>8}")
    for (bb, ds), rs in sorted(grp.items()):
        v = np.array([r["voi"] for r in rs])
        hm = float(np.mean([r["ndcg_hm"] for r in rs]))
        hmt = float(np.mean([r["ndcg_hmt"] for r in rs]))
        rec = {
            "n": len(rs), "mean_voi": float(v.mean()),
            "pct_help": float(np.mean(v > 1e-9)), "pct_hurt": float(np.mean(v < -1e-9)),
            "mean_ndcg_hm": hm, "mean_ndcg_hmt": hmt,
        }
        voi_by_cell[f"{bb}/{ds}"] = rec
        print(f"  {bb + '/' + ds:<24}{len(rs):>6}{v.mean():>+10.4f}"
              f"{rec['pct_help']:>8.1%}{rec['pct_hurt']:>8.1%}{hm:>8.4f}{hmt:>8.4f}")

    # Active modeling features (deployable by default).
    if args.feature_set == "all":
        active_feats = list(FEATURES) + [
            f for f in ORACLE_DISCRIM_FEATS if f not in FEATURES
        ]
    else:
        active_feats = list(FEATURES)
    print(f"[features] set={args.feature_set} n={len(active_feats)}: {active_feats}")

    # Capability for CSV / PD: params by default; measured reported as diagnostic.
    measured_cap: Dict[str, float] = {}
    for label in {r["backbone"] for r in rows}:
        hm = [r["ndcg_hm"] for r in rows if r["backbone"] == label]
        measured_cap[label] = float(np.mean(hm)) if hm else 0.0
    _attach_capability(
        rows, capability_mode=args.capability, params_lookup=params_lookup,
    )
    print("[capability] mode=" + args.capability + "  " + "  ".join(
        f"{b}: measured={measured_cap[b]:.4f}"
        + (f" params={params_lookup[b]}" if params_lookup.get(b) else "")
        + f" used_cap={next(r['cap'] for r in rows if r['backbone']==b):.4f}"
        for b in sorted(measured_cap)))

    # per-task CSV: always store deployable + oracle diagnostics
    csv_path = os.path.join(args.output_dir, "voi_features_per_task.csv")
    cols = (["backbone", "dataset", "condition", "run_index",
             "ndcg_hm", "ndcg_hmt", "voi"] + ALL_CSV_FEATS)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # majority-class base rate for cell-sign prediction (sign_acc must beat this)
    cell_true_acc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        cell_true_acc[cell_key(r)].append(r["voi"])
    cell_means = {c: float(np.mean(v)) for c, v in cell_true_acc.items()}
    nontrivial = [m for m in cell_means.values() if abs(m) > 1e-4]
    frac_neg = float(np.mean([m < 0 for m in nontrivial])) if nontrivial else 0.0
    base_rate = max(frac_neg, 1.0 - frac_neg)
    print(f"[baseline] majority cell-sign base rate = {base_rate:.3f} "
          f"({'hurts' if frac_neg >= 0.5 else 'helps'} in "
          f"{max(frac_neg, 1 - frac_neg):.0%} of {len(nontrivial)} non-trivial cells)")

    results: Dict[str, dict] = {
        "n_tasks": len(rows),
        "n_cells": len({cell_key(r) for r in rows}),
        "n_cells_nontrivial": len(nontrivial),
        "cell_sign_base_rate": base_rate,
        "voi_by_cell": voi_by_cell,
        "backbones": sorted(measured_cap),
        "datasets": sorted({r["dataset"] for r in rows}),
        "measured_capability": measured_cap,
        "capability_mode": args.capability,
        "feature_set": args.feature_set,
        "features": active_feats,
        "oracle_diagnostic_features": ORACLE_DISCRIM_FEATS,
    }

    protocols = (["lodo", "lobo", "cv"] if args.protocol == "all" else [args.protocol])
    proto_kw = dict(
        features=active_feats,
        capability_mode=args.capability,
        params_lookup=params_lookup,
    )
    if "lodo" in protocols:
        print("\n[protocol] LODO (leave-one-dataset-out)")
        results["lodo"] = run_protocol(rows, "dataset", args.seed, **proto_kw)
    if "lobo" in protocols:
        print("[protocol] LOBO (leave-one-backbone-out)")
        results["lobo"] = run_protocol(rows, "backbone", args.seed, **proto_kw)
    if "cv" in protocols:
        print("[protocol] within-sample CV (reference)")
        results["cv"] = run_cv(rows, args.cv_folds, args.seed, **proto_kw)

    results["partial_dependence_capability"] = partial_dependence_capability(
        rows, args.seed, active_feats)
    results["permutation_importance"] = permutation_importance_all(
        rows, args.seed, active_feats)

    out_json = os.path.join(args.output_dir, "voi_predictive_summary.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)

    # optional PD plot
    pd_png = None
    pd_curve = results["partial_dependence_capability"]
    if pd_curve:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            caps = [d["cap"] for d in pd_curve]
            vals = [d["predicted_voi"] for d in pd_curve]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(caps, vals, marker="o", color="#3b7dd8")
            ax.axhline(0, color="black", lw=1)
            ax.set_xlabel("backbone capability (mean NDCG@k of H,M)")
            ax.set_ylabel("predicted SID-tool VoI")
            ax.set_title("Partial dependence: does the model recover the U-shape?")
            fig.tight_layout()
            pd_png = os.path.join(args.output_dir, "voi_pd_capability.png")
            fig.savefig(pd_png, dpi=150)
            plt.close(fig)
        except Exception:
            pd_png = None

    # ---- console summary ----
    def fmt(v):
        return f"{v:+.3f}" if isinstance(v, float) else " n/a"

    print("\n" + "=" * 72)
    print(f"PREDICTIVE VoI  |  {results['n_tasks']} tasks, {results['n_cells']} cells")
    print(f"majority cell-sign baseline = {base_rate:.3f}  "
          f"(sign_acc must beat this to be meaningful)")
    print("=" * 72)
    for proto in ("lodo", "lobo"):
        if proto not in results:
            continue
        print(f"\n[{proto.upper()}]  cell-level out-of-sample (mean over held-out folds)")
        for model_name, d in results[proto].items():
            msa = d.get("mean_fold_sign_acc")
            beat = ""
            if msa is not None:
                beat = "  >baseline" if msa > base_rate + 1e-9 else "  <=baseline"
            print(f"  {model_name:<7} mean_fold: sign_acc={fmt(msa)}  "
                  f"spearman={fmt(d.get('mean_fold_spearman'))} "
                  f"(min_fold_ρ={fmt(d.get('min_fold_spearman'))}){beat}")
            print(f"          pooled(diag): sign_acc={fmt(d['pooled_cell_sign_acc'])}  "
                  f"spearman={fmt(d['pooled_cell_spearman'])}")
            # per-fold detail
            for held, fd in d.get("folds", {}).items():
                print(f"      fold[{held}] sign_acc={fmt(fd.get('cell_sign_acc'))} "
                      f"spearman={fmt(fd.get('cell_spearman'))} "
                      f"n_test={fd.get('n_test')} cells={fd.get('n_cells')}")
    if "cv" in results:
        print("\n[CV] within-sample reference")
        for model_name, d in results["cv"].items():
            print(f"  {model_name:<7} inst_spearman={fmt(d['mean_instance_spearman'])}  "
                  f"cell_sign_acc={fmt(d['mean_cell_sign_acc'])}")
    imp = results.get("permutation_importance") or {}
    if imp:
        print("\n[importance] top features (permutation, GBM R2 drop)")
        for f, v in sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:8]:
            print(f"  {f:<22} {v:+.4f}")
    print("\nwrote:")
    print(f"  {csv_path}")
    print(f"  {out_json}")
    print(f"  {pd_png if pd_png else '(PD plot skipped)'}")


if __name__ == "__main__":
    main()
