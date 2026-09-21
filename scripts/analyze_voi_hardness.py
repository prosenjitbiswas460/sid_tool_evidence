#!/usr/bin/env python3
"""
Instance-level hardness of SID-tool admission.

Population-level VoI can be predictable while remaining hard to exploit per task.
This script reports oracle headroom, signal ceiling, and recovered fraction.

Setup. Within a deployment scope (default: a (backbone,dataset) pair, since you
deploy one model on one catalog), a per-task selector may admit the SID tool
using only observable signals S. Define, relative to the best fixed policy
(always-admit vs always-skip):

    oracle headroom      H*  = E[VoI_+] - best_fixed        (perfect per-task routing)
    signal ceiling       G*(S) = E[(E[VoI|S])_+] - best_fixed   (Bayes selector on S)
    realized (cross-fit) G_hat = E[VoI * 1{ mu_oof(S) > 0 }] - best_fixed
    information gap      H* - G_hat            (VoI variance not explained by S)
    recoverable fraction G_hat / H*            (share of routable headroom captured)

If E[VoI|S] is near-flat (signals weakly informative per task) the ceiling
collapses even though the per-task oracle headroom H* is large (VoI is bimodal).

Shift. A selector is trained on a disjoint calibration pool and deployed on the
benchmark. With a calibration CSV we TRAIN mu on calibration and apply to the
benchmark (the honest cross-distribution number) and also report the covariate-
shift AUC (a classifier separating calibration vs benchmark signals). Without
it, pass --shift_auc to report the shift magnitude and a degraded-ceiling note.

Verification. Given the deployed calibrated-EAA predictions, we compute its
actual recovered headroom on the benchmark and check it lands at ~G_hat (shifted)
-- i.e. the deployed policy is already at the signal ceiling, which is ~0.

Input: the voi_features_per_task.csv produced by scripts/predict_voi.py.

Example:
  python scripts/analyze_voi_hardness.py \
    --benchmark_csv results/voi_predictive/voi_features_per_task.csv \
    --calibration_csv results/voi_calibration/voi_features_per_task.csv \
    --eaa_predictions_dir outputs --track 2 \
    --eaa_map "1.5b=eaa_1_5b,3b=eaa_3b,7b=eaa" \
    --output_dir results/voi_hardness
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

# Deployable pre-admission signals only (no ground-truth target).
# Cell-constant features (capability / codebook health) are excluded: within one
# deployment they cannot discriminate task-to-task.
# Oracle target-relative geometry is intentionally omitted here; use --signals
# explicitly if you need a post-hoc diagnostic ceiling with leakage.
DEFAULT_SIGNALS = [
    "log_history_chars", "log_review_count", "metadata_richness",
    "sid_top1_margin", "sid_score_std", "candidate_sid_diversity",
]
# Within-scope varying features for the cal-only two-feature rule.
# (collision_rate is cell-constant under backbone_dataset scope, so it cannot
# discriminate tasks inside a matched cal→bench scope.)
RULE_FEATURES = ["sid_top1_margin", "sid_score_std"]
CONDITIONS = ["classic", "cold_start_user", "cold_start_item", "evolving_interest"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--benchmark_csv", required=True,
                   help="voi_features_per_task.csv from predict_voi.py (benchmark).")
    p.add_argument("--calibration_csv", default="",
                   help="Optional voi_features_per_task.csv built from the calibration "
                        "oracles (train-on-cal / apply-on-bench shift test).")
    p.add_argument("--signals", nargs="+", default=DEFAULT_SIGNALS)
    p.add_argument("--scope", choices=["backbone_dataset", "backbone", "pooled"],
                   default="backbone_dataset",
                   help="Deployment unit within which per-task exploitability is measured.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_scope_n", type=int, default=60)
    p.add_argument("--shift_auc", type=float, default=None,
                   help="Optional covariate-shift AUC (cal vs bench) if no calibration_csv.")
    p.add_argument(
        "--stronger_routers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also report Ridge/HGB/MLP separately + a cal-only two-feature rule "
             "(default: on). Default Ridge/HGB aggregate keys are unchanged.",
    )
    # EAA verification
    p.add_argument("--eaa_predictions_dir", default="")
    p.add_argument("--track", type=int, default=2)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--eaa_map", default="",
                   help="Comma-separated backbone=eaa_prefix (the calibrated adaptive "
                        "agent), e.g. '1.5b=eaa_1_5b,3b=eaa_3b,7b=eaa'.")
    p.add_argument("--output_dir", default="results/voi_hardness")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# CSV IO                                                                       #
# --------------------------------------------------------------------------- #
def _num(v) -> float:
    if v is None or v == "" or v == "None":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def load_rows(path: str, signals: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    extras = [f for f in RULE_FEATURES if f not in signals]
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = {
                "backbone": r.get("backbone", "na"),
                "dataset": r.get("dataset", "na"),
                "condition": r.get("condition", "na"),
                "run_index": int(_num(r.get("run_index"))) if r.get("run_index") not in (None, "") else -1,
                "voi": _num(r.get("voi")),
                "ndcg_hm": _num(r.get("ndcg_hm")),
                "ndcg_hmt": _num(r.get("ndcg_hmt")),
            }
            for s in list(signals) + extras:
                row[s] = _num(r.get(s))
            if not math.isnan(row["voi"]):
                rows.append(row)
    return rows


def scope_key(row: dict, scope: str) -> str:
    if scope == "pooled":
        return "ALL"
    if scope == "backbone":
        return row["backbone"]
    return f"{row['backbone']}/{row['dataset']}"


# --------------------------------------------------------------------------- #
# ceiling estimation                                                           #
# --------------------------------------------------------------------------- #
def _matrix(rows: Sequence[dict], signals: Sequence[str]) -> np.ndarray:
    return np.array([[r[s] for s in signals] for r in rows], dtype=float)


def _model_registry(seed: int) -> Dict[str, object]:
    """Admit/skip regressors used for default selection and the stronger-router ablation."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor

    ridge = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=1.0)),
        ]
    )
    gbr = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        random_state=seed,
    )
    mlp = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "m",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=500,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]
    )
    return {"ridge": ridge, "hgb": gbr, "mlp": mlp}


def crossfit_mu_by_model(
    X: np.ndarray, y: np.ndarray, folds: int, seed: int
) -> Dict[str, np.ndarray]:
    """Out-of-fold μ predictions per model name."""
    from sklearn.model_selection import KFold, cross_val_predict

    n = len(y)
    kf = KFold(n_splits=min(folds, max(2, n // 20)), shuffle=True, random_state=seed)
    out: Dict[str, np.ndarray] = {}
    for name, model in _model_registry(seed).items():
        try:
            out[name] = np.asarray(cross_val_predict(model, X, y, cv=kf), dtype=float)
        except Exception:
            continue
    return out


def crossfit_mu(X: np.ndarray, y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Best out-of-fold E[VoI|S] over Ridge / HGB only.

    MLP is reported separately under stronger_routers.
    """
    by_model = crossfit_mu_by_model(X, y, folds, seed)
    best_mu, best_gain = None, -np.inf
    for name in ("ridge", "hgb"):
        mu = by_model.get(name)
        if mu is None:
            continue
        gain = float(np.mean(y * (mu > 0)))
        if gain > best_gain:
            best_gain, best_mu = gain, mu
    if best_mu is None:
        best_mu = np.full_like(y, y.mean(), dtype=float)
    return best_mu


def _selector_stats(y: np.ndarray, admit: np.ndarray) -> dict:
    best_fixed = max(0.0, float(y.mean()))
    oracle = float(np.maximum(y, 0).mean()) - best_fixed
    realized = float((y * admit.astype(float)).mean()) - best_fixed
    return {
        "n": int(len(y)),
        "best_fixed_gain": best_fixed,
        "oracle_headroom": oracle,
        "realized_gain_over_bf": realized,
        "recoverable_fraction": realized / oracle if oracle > 1e-9 else float("nan"),
        "admit_rate": float(np.mean(admit)),
    }


def ceiling_for_scope(rows: Sequence[dict], signals: Sequence[str],
                      folds: int, seed: int) -> Optional[dict]:
    y = np.array([r["voi"] for r in rows], dtype=float)
    n = len(y)
    if n < 20:
        return None
    best_fixed = max(0.0, float(y.mean()))          # always-skip(0) vs always-admit(E[VoI])
    oracle_gain = float(np.maximum(y, 0).mean())     # perfect per-task routing (over skip)
    headroom = oracle_gain - best_fixed              # max any router adds over best fixed

    X = _matrix(rows, signals)
    mu = crossfit_mu(X, y, folds, seed)
    admit = mu > 0
    realized_gain = float((y * admit).mean())
    theoretical_ceiling = float(np.maximum(mu, 0).mean())  # if mu were the truth
    realized_over_bf = realized_gain - best_fixed
    frac = realized_over_bf / headroom if headroom > 1e-9 else float("nan")
    return {
        "n": n, "mean_voi": float(y.mean()),
        "pct_help": float(np.mean(y > 1e-9)), "pct_hurt": float(np.mean(y < -1e-9)),
        "best_fixed_gain": best_fixed,
        "oracle_headroom": headroom,
        "signal_ceiling_theoretical": theoretical_ceiling - best_fixed,
        "realized_gain_over_bf": realized_over_bf,
        "information_gap": headroom - realized_over_bf,
        "recoverable_fraction": frac,
        "admit_rate": float(admit.mean()),
    }


def aggregate(scopes: Dict[str, dict]) -> dict:
    """n-weighted aggregate of gains, then recompute fraction."""
    tot = sum(s["n"] for s in scopes.values())
    if tot == 0:
        return {}
    w = lambda key: sum(s[key] * s["n"] for s in scopes.values()) / tot
    headroom = w("oracle_headroom")
    realized = w("realized_gain_over_bf")
    return {
        "n": tot,
        "oracle_headroom": headroom,
        "realized_gain_over_bf": realized,
        "signal_ceiling_theoretical": w("signal_ceiling_theoretical"),
        "information_gap": headroom - realized,
        "recoverable_fraction": realized / headroom if headroom > 1e-9 else float("nan"),
        "mean_voi": w("mean_voi"),
    }


# --------------------------------------------------------------------------- #
# shift: train on calibration, apply on benchmark                             #
# --------------------------------------------------------------------------- #
def covariate_shift_auc(cal: Sequence[dict], bench: Sequence[dict],
                        signals: Sequence[str], seed: int) -> Optional[float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    Xc, Xb = _matrix(cal, signals), _matrix(bench, signals)
    if len(Xc) < 20 or len(Xb) < 20:
        return None
    X = np.vstack([Xc, Xb])
    y = np.concatenate([np.zeros(len(Xc)), np.ones(len(Xb))])
    clf = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("m", LogisticRegression(max_iter=1000))])
    try:
        proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, proba))
    except Exception:
        return None


def shift_realized_by_model(
    cal_rows: Sequence[dict],
    bench_rows: Sequence[dict],
    signals: Sequence[str],
    seed: int,
) -> Dict[str, dict]:
    """Train each model on calibration, apply to benchmark."""
    yc = np.array([r["voi"] for r in cal_rows], dtype=float)
    yb = np.array([r["voi"] for r in bench_rows], dtype=float)
    if len(yc) < 30 or len(yb) < 20:
        return {}
    Xc, Xb = _matrix(cal_rows, signals), _matrix(bench_rows, signals)
    out: Dict[str, dict] = {}
    for name, model in _model_registry(seed).items():
        try:
            model.fit(Xc, yc)
            mu = np.asarray(model.predict(Xb), dtype=float)
        except Exception:
            continue
        out[name] = _selector_stats(yb, mu > 0)
    return out


def shift_realized(cal_rows: Sequence[dict], bench_rows: Sequence[dict],
                   signals: Sequence[str], seed: int) -> Optional[dict]:
    """Train mu on calibration, apply to benchmark; default models are Ridge/HGB."""
    by_model = shift_realized_by_model(cal_rows, bench_rows, signals, seed)
    locked = {k: v for k, v in by_model.items() if k in ("ridge", "hgb")}
    if not locked:
        return None
    best_name = max(locked, key=lambda k: locked[k]["realized_gain_over_bf"])
    best = locked[best_name]
    return {
        "oracle_headroom": best["oracle_headroom"],
        "realized_gain_over_bf": best["realized_gain_over_bf"],
        "recoverable_fraction": best["recoverable_fraction"],
        "selected_model": best_name,
    }


def _rule_admit(
    rows: Sequence[dict],
    form: str,
    t_margin: float,
    t_std: float,
) -> np.ndarray:
    margin = np.array([r.get("sid_top1_margin", float("nan")) for r in rows], dtype=float)
    std = np.array([r.get("sid_score_std", float("nan")) for r in rows], dtype=float)
    m_ok = np.isfinite(margin)
    s_ok = np.isfinite(std)
    if form == "margin_ge":
        return m_ok & (margin >= t_margin)
    if form == "std_ge":
        return s_ok & (std >= t_std)
    if form == "margin_ge_and_std_ge":
        return m_ok & s_ok & (margin >= t_margin) & (std >= t_std)
    raise ValueError(form)


def tune_two_feature_rule(
    cal_rows: Sequence[dict],
    seed: int,
    n_grid: int = 9,
    min_admit: float = 0.05,
    max_admit: float = 0.95,
) -> Optional[dict]:
    """Tune a simple margin+score-std admit rule on calibration only.

    Rejects degenerate always-skip / always-admit thresholds so the rule is a
    real selector when transferred to the benchmark.
    """
    del seed  # grid is deterministic given percentiles
    yc = np.array([r["voi"] for r in cal_rows], dtype=float)
    if len(yc) < 30:
        return None
    margins = np.array(
        [r.get("sid_top1_margin", float("nan")) for r in cal_rows], dtype=float
    )
    stds = np.array(
        [r.get("sid_score_std", float("nan")) for r in cal_rows], dtype=float
    )
    m_fin = margins[np.isfinite(margins)]
    s_fin = stds[np.isfinite(stds)]
    m_vals = np.unique(np.nanpercentile(m_fin, np.linspace(10, 90, n_grid))) if len(m_fin) else np.array([0.0])
    s_vals = np.unique(np.nanpercentile(s_fin, np.linspace(10, 90, n_grid))) if len(s_fin) else np.array([0.0])

    best = None
    best_key = None  # (gain, -abs(admit-0.5)) for tie-break
    forms = ("margin_ge", "std_ge", "margin_ge_and_std_ge")
    for form in forms:
        for t_m in m_vals:
            for t_s in s_vals:
                if form == "margin_ge" and t_s != s_vals[0]:
                    continue
                if form == "std_ge" and t_m != m_vals[0]:
                    continue
                admit = _rule_admit(cal_rows, form, float(t_m), float(t_s))
                admit_rate = float(np.mean(admit))
                if admit_rate < min_admit or admit_rate > max_admit:
                    continue
                stats = _selector_stats(yc, admit)
                g = stats["realized_gain_over_bf"]
                key = (g, -abs(admit_rate - 0.5))
                if best_key is None or key > best_key:
                    best_key = key
                    best = {
                        "form": form,
                        "t_sid_top1_margin": float(t_m),
                        "t_sid_score_std": float(t_s),
                        "calibration": stats,
                    }
    return best


def apply_two_feature_rule(rule: dict, bench_rows: Sequence[dict]) -> dict:
    yb = np.array([r["voi"] for r in bench_rows], dtype=float)
    admit = _rule_admit(
        bench_rows,
        rule["form"],
        rule["t_sid_top1_margin"],
        rule["t_sid_score_std"],
    )
    stats = _selector_stats(yb, admit)
    return {
        "form": rule["form"],
        "t_sid_top1_margin": rule["t_sid_top1_margin"],
        "t_sid_score_std": rule["t_sid_score_std"],
        "calibration": rule["calibration"],
        "benchmark": stats,
    }


def stronger_router_ablation(
    bench_rows: Sequence[dict],
    cal_rows: Optional[Sequence[dict]],
    scoped_bench: Dict[str, List[dict]],
    cal_by: Dict[str, List[dict]],
    signals: Sequence[str],
    folds: int,
    seed: int,
) -> dict:
    """Per-model in-dist / shift + cal-only two-feature rule."""
    # ---- in-distribution per model (pooled over scopes, n-weighted) ----
    indist_parts: Dict[str, List[Tuple[int, dict]]] = {
        "ridge": [], "hgb": [], "mlp": []
    }
    for _key, rs in scoped_bench.items():
        y = np.array([r["voi"] for r in rs], dtype=float)
        if len(y) < 20:
            continue
        X = _matrix(rs, signals)
        by_model = crossfit_mu_by_model(X, y, folds, seed)
        for name, mu in by_model.items():
            indist_parts.setdefault(name, []).append(
                (len(y), _selector_stats(y, mu > 0))
            )

    def _pool(parts: List[Tuple[int, dict]]) -> dict:
        if not parts:
            return {}
        tot = sum(n for n, _ in parts)
        keys = [
            "best_fixed_gain",
            "oracle_headroom",
            "realized_gain_over_bf",
            "admit_rate",
        ]
        out = {"n": tot}
        for k in keys:
            out[k] = sum(n * s[k] for n, s in parts) / tot
        hr = out["oracle_headroom"]
        out["recoverable_fraction"] = (
            out["realized_gain_over_bf"] / hr if hr > 1e-9 else float("nan")
        )
        return out

    in_dist = {name: _pool(parts) for name, parts in indist_parts.items() if parts}

    # ---- shift per model ----
    shift_models: Dict[str, dict] = {}
    if cal_rows:
        shift_parts: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)
        for key, brs in scoped_bench.items():
            crs = cal_by.get(key, [])
            if not crs:
                continue
            by_model = shift_realized_by_model(crs, brs, signals, seed)
            for name, stats in by_model.items():
                shift_parts[name].append((len(brs), stats))
        shift_models = {name: _pool(parts) for name, parts in shift_parts.items() if parts}

        # ---- two-feature rule: tune per matched scope on that scope's cal ----
        rule_parts_cal: List[Tuple[int, dict]] = []
        rule_parts_bench: List[Tuple[int, dict]] = []
        rule_examples = []
        for key, brs in scoped_bench.items():
            crs = cal_by.get(key, [])
            if len(crs) < 30 or len(brs) < 20:
                continue
            rule = tune_two_feature_rule(crs, seed=seed)
            if rule is None:
                continue
            applied = apply_two_feature_rule(rule, brs)
            rule_parts_cal.append((len(crs), rule["calibration"]))
            rule_parts_bench.append((len(brs), applied["benchmark"]))
            rule_examples.append(
                {
                    "scope": key,
                    "form": rule["form"],
                    "t_sid_top1_margin": rule["t_sid_top1_margin"],
                    "t_sid_score_std": rule["t_sid_score_std"],
                    "cal_frac": rule["calibration"].get("recoverable_fraction"),
                    "bench_frac": applied["benchmark"].get("recoverable_fraction"),
                    "bench_admit": applied["benchmark"].get("admit_rate"),
                }
            )
        rule_shift = None
        if rule_parts_bench:
            rule_shift = {
                "protocol": "per_scope_thresholds_tuned_on_calibration_only",
                "features": ["sid_top1_margin", "sid_score_std"],
                "calibration": _pool(rule_parts_cal),
                "benchmark": _pool(rule_parts_bench),
                "per_scope": rule_examples,
                # Representative display fields from first matched scope
                "form": rule_examples[0]["form"],
                "t_sid_top1_margin": rule_examples[0]["t_sid_top1_margin"],
                "t_sid_score_std": rule_examples[0]["t_sid_score_std"],
            }
    else:
        rule_shift = None

    return {
        "models": ["ridge", "hgb", "mlp"],
        "in_distribution": in_dist,
        "shift": shift_models,
        "two_feature_rule": rule_shift,
        "note": (
            "Per-model and two-feature-rule ablations. Aggregate and shift "
            "keys elsewhere in this JSON are unchanged. Two-feature "
            "thresholds are tuned on calibration only (collision_rate, "
            "sid_score_std)."
        ),
    }


# --------------------------------------------------------------------------- #
# EAA verification (deployed calibrated policy)                                #
# --------------------------------------------------------------------------- #
def ndcg_single(ranking: Sequence[str], target: str, k: int) -> float:
    for pos, item in enumerate(list(ranking)[:k]):
        if item == target:
            return 1.0 / math.log2(pos + 2)
    return 0.0


def load_eaa_ndcg(pdir: str, prefix: str, dataset: str, track: int,
                  condition: str, k: int) -> Dict[int, float]:
    path = os.path.join(pdir, f"{prefix}_{dataset}_track{track}_{condition}_predictions.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    out: Dict[int, float] = {}
    for rec in payload.get("records", []):
        gt = rec.get("groundtruth") or {}
        t = gt.get("ground truth") or gt.get("ground_truth")
        rk = rec.get("output")
        if t is None or not rk:
            continue
        out[int(rec["run_index"])] = ndcg_single([str(x) for x in rk], str(t), k)
    return out


def _recovered(eaa: Sequence[float], hm: Sequence[float],
               hmt: Sequence[float], oracle: Sequence[float]) -> dict:
    eaa_m = float(np.mean(eaa)); hm_m = float(np.mean(hm))
    hmt_m = float(np.mean(hmt)); oracle_m = float(np.mean(oracle))
    best_fixed_m = max(hm_m, hmt_m)               # always-HM vs always-HMT
    denom = oracle_m - best_fixed_m
    recovered = (eaa_m - best_fixed_m) / denom if denom > 1e-9 else float("nan")
    return {"eaa_ndcg": eaa_m, "best_fixed_ndcg": best_fixed_m,
            "oracle_ndcg": oracle_m, "recovered_headroom": recovered}


def eaa_recovered_headroom(bench_rows: Sequence[dict], pdir: str,
                           eaa_map: Dict[str, str], track: int, k: int,
                           scope: str) -> Optional[dict]:
    """Recovered headroom of the deployed calibrated EAA vs best-fixed / oracle.

    Returns the pooled number plus a per-scope breakdown so it can be compared
    to the shift ceiling at the *same* deployment scope.
    """
    by_cell: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in bench_rows:
        by_cell[(r["backbone"], r["dataset"], r["condition"])].append(r)

    pool = {"eaa": [], "hm": [], "hmt": [], "oracle": []}
    per_scope_acc: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {"eaa": [], "hm": [], "hmt": [], "oracle": []})
    matched = 0
    for (bb, ds, cond), rs in by_cell.items():
        prefix = eaa_map.get(bb)
        if not prefix:
            continue
        eaa_ndcg = load_eaa_ndcg(pdir, prefix, ds, track, cond, k)
        if not eaa_ndcg:
            continue
        sk = scope_key(rs[0], scope)
        for r in rs:
            e = eaa_ndcg.get(r["run_index"])
            if e is None or math.isnan(r["ndcg_hm"]) or math.isnan(r["ndcg_hmt"]):
                continue
            orc = max(r["ndcg_hm"], r["ndcg_hmt"])
            for tgt, val in ((pool, None), (per_scope_acc[sk], None)):
                tgt["eaa"].append(e); tgt["hm"].append(r["ndcg_hm"])
                tgt["hmt"].append(r["ndcg_hmt"]); tgt["oracle"].append(orc)
            matched += 1
    if matched < 30:
        return None
    out = _recovered(pool["eaa"], pool["hm"], pool["hmt"], pool["oracle"])
    out["n_matched"] = matched
    out["per_scope"] = {
        sk: {**_recovered(a["eaa"], a["hm"], a["hmt"], a["oracle"]), "n": len(a["eaa"])}
        for sk, a in per_scope_acc.items() if len(a["eaa"]) >= 20
    }
    return out


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    return f"{v:+.4f}" if isinstance(v, float) and not math.isnan(v) else "   n/a"


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bench = load_rows(args.benchmark_csv, args.signals)
    if not bench:
        raise SystemExit(f"No rows in {args.benchmark_csv}")
    print(f"[load] benchmark: {len(bench)} tasks from {args.benchmark_csv}")

    # per-scope ceilings
    scoped: Dict[str, List[dict]] = defaultdict(list)
    for r in bench:
        scoped[scope_key(r, args.scope)].append(r)
    scope_res: Dict[str, dict] = {}
    for key, rs in sorted(scoped.items()):
        if len(rs) < args.min_scope_n:
            continue
        res = ceiling_for_scope(rs, args.signals, args.folds, args.seed)
        if res:
            scope_res[key] = res
    if not scope_res:
        raise SystemExit("No scope had enough tasks; lower --min_scope_n.")
    agg = aggregate(scope_res)

    results: dict = {
        "scope": args.scope, "signals": list(args.signals),
        "per_scope": scope_res, "aggregate": agg,
    }

    # shift analysis
    shift: dict = {}
    cal: List[dict] = []
    cal_by: Dict[str, List[dict]] = defaultdict(list)
    if args.calibration_csv:
        cal = load_rows(args.calibration_csv, args.signals)
        print(f"[load] calibration: {len(cal)} tasks from {args.calibration_csv}")
        auc = covariate_shift_auc(cal, bench, args.signals, args.seed)
        shift["covariate_shift_auc"] = auc
        # scope-matched train-on-cal / apply-on-bench, aggregated
        for r in cal:
            cal_by[scope_key(r, args.scope)].append(r)
        parts, tot = [], 0
        shift_by_scope: Dict[str, dict] = {}
        for key, brs in scoped.items():
            crs = cal_by.get(key, [])
            sr = shift_realized(crs, brs, args.signals, args.seed) if crs else None
            if sr:
                parts.append((len(brs), sr)); tot += len(brs)
                shift_by_scope[key] = sr
        shift["per_scope"] = shift_by_scope
        if tot:
            hr = sum(n * s["oracle_headroom"] for n, s in parts) / tot
            rz = sum(n * s["realized_gain_over_bf"] for n, s in parts) / tot
            shift["shift_realized_gain_over_bf"] = rz
            shift["shift_oracle_headroom"] = hr
            shift["shift_recoverable_fraction"] = rz / hr if hr > 1e-9 else float("nan")
            shift["matched_scopes"] = sorted(shift_by_scope)
    elif args.shift_auc is not None:
        shift["covariate_shift_auc"] = args.shift_auc
    if shift:
        results["shift"] = shift

    # EAA verification
    if args.eaa_predictions_dir and args.eaa_map:
        eaa_map = {}
        for tok in args.eaa_map.split(","):
            tok = tok.strip()
            if tok:
                b, _, pfx = tok.partition("=")
                eaa_map[b.strip()] = pfx.strip()
        ver = eaa_recovered_headroom(bench, args.eaa_predictions_dir, eaa_map,
                                     args.track, args.k, args.scope)
        if ver:
            results["eaa_verification"] = ver

    # Stronger-router ablation (does not overwrite Ridge/HGB aggregates).
    if args.stronger_routers:
        results["stronger_routers"] = stronger_router_ablation(
            bench_rows=bench,
            cal_rows=cal or None,
            scoped_bench=dict(scoped),
            cal_by=dict(cal_by),
            signals=args.signals,
            folds=args.folds,
            seed=args.seed,
        )

    with open(os.path.join(args.output_dir, "voi_hardness_summary.json"),
              "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)

    # ---- console ----
    print("\n" + "=" * 78)
    print(f"INSTANCE-LEVEL HARDNESS  (scope = {args.scope})")
    print("=" * 78)
    print("  per-scope: oracle headroom H* vs realized signal gain G_hat (over best-fixed)")
    print(f"  {'scope':<20}{'n':>6}{'meanVoI':>9}{'H*':>9}{'G_hat':>9}{'frac':>7}{'gap':>9}")
    for key, s in sorted(scope_res.items()):
        print(f"  {key:<20}{s['n']:>6}{s['mean_voi']:>+9.4f}"
              f"{s['oracle_headroom']:>9.4f}{s['realized_gain_over_bf']:>+9.4f}"
              f"{s['recoverable_fraction']:>7.2f}{s['information_gap']:>9.4f}")
    print("-" * 78)
    print(f"  AGGREGATE (n-weighted):  H*={agg['oracle_headroom']:.4f}  "
          f"G_hat(in-dist)={agg['realized_gain_over_bf']:+.4f}  "
          f"recoverable={agg['recoverable_fraction']:.1%}  "
          f"info_gap={agg['information_gap']:.4f}")
    print("  Note: G_hat is cross-fit IN-DISTRIBUTION (calibrated on the very")
    print("  distribution it is deployed on) and the better of Ridge/HGB — an")
    print("  optimistic ceiling. info_gap is the bimodal-VoI variance no signal")
    print("  explains; the shift + EAA rows below show what actually transports.")

    if shift:
        print("-" * 78)
        if "covariate_shift_auc" in shift and shift["covariate_shift_auc"] is not None:
            print(f"  covariate-shift AUC (cal vs bench) = {shift['covariate_shift_auc']:.3f}  "
                  f"(0.5 = no shift, 1.0 = fully separable)")
        if "shift_recoverable_fraction" in shift:
            print(f"  train-on-cal / apply-on-bench:  G_hat_shift="
                  f"{shift['shift_realized_gain_over_bf']:+.4f}  "
                  f"recoverable={shift['shift_recoverable_fraction']:.1%}  "
                  f"(<= in-distribution ceiling)")

    if "eaa_verification" in results:
        v = results["eaa_verification"]
        print("-" * 78)
        print(f"  deployed calibrated-EAA (n={v['n_matched']}):  "
              f"NDCG={v['eaa_ndcg']:.4f}  best_fixed={v['best_fixed_ndcg']:.4f}  "
              f"oracle={v['oracle_ndcg']:.4f}")
        recov = v["recovered_headroom"]
        print(f"    EAA recovered headroom = {recov:.1%}")
        indist = agg["recoverable_fraction"]
        if not math.isnan(indist) and not math.isnan(recov):
            print(f"    transportability gap (in-dist ceiling {indist:.1%} - "
                  f"deployed {recov:.1%}) = {indist - recov:.1%} of H*")
        if recov < 0:
            print("    => deployed policy is BELOW always-HM: the in-distribution")
            print("       signal is real but does NOT transport. Shift, not signal")
            print("       poverty, is what defeats instance-level routing.")
        else:
            print("    => deployed policy sits below the in-distribution ceiling;")
            print("       the gap is the price of off-distribution calibration.")

    # ---- combined per-scope table: same-scope in-dist vs shift vs deployed ----
    shift_ps = shift.get("per_scope", {}) if shift else {}
    eaa_ps = results.get("eaa_verification", {}).get("per_scope", {})
    combined: List[dict] = []
    for key in sorted(scope_res):
        row = {
            "scope": key,
            "in_dist_frac": scope_res[key]["recoverable_fraction"],
            "shift_frac": shift_ps.get(key, {}).get("recoverable_fraction", float("nan")),
            "eaa_frac": eaa_ps.get(key, {}).get("recovered_headroom", float("nan")),
        }
        combined.append(row)
    if any(not math.isnan(r["shift_frac"]) or not math.isnan(r["eaa_frac"]) for r in combined):
        print("-" * 78)
        print("  SAME-SCOPE recovered headroom (fraction of H*):")
        print(f"  {'scope':<20}{'in-dist':>10}{'shifted':>10}{'deployed':>10}")
        for r in combined:
            def pf(v):
                return f"{v:>9.1%}" if isinstance(v, float) and not math.isnan(v) else "      n/a"
            print(f"  {r['scope']:<20}{pf(r['in_dist_frac']):>10}"
                  f"{pf(r['shift_frac']):>10}{pf(r['eaa_frac']):>10}")
        print("  'shifted' (retrained selector) and 'deployed' (calibrated cascade) are")
        print("  two distinct deployment mechanisms; BOTH collapsing far below 'in-dist'")
        print("  is the non-transportability result -- they need not equal each other.")
    csv_path = os.path.join(args.output_dir, "voi_hardness_by_scope.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "in_dist_frac", "shift_frac", "eaa_frac"])
        w.writeheader()
        w.writerows(combined)

    # Stronger-router console + CSV
    if "stronger_routers" in results:
        sr = results["stronger_routers"]
        print("-" * 78)
        print("  STRONGER ROUTERS (ablation; Ridge/HGB aggregates above unchanged)")
        print(f"  {'model':<10}{'in-dist frac':>14}{'shift frac':>12}{'admit@shift':>12}")
        for name in sr.get("models", []):
            ind = sr.get("in_distribution", {}).get(name, {})
            sh = sr.get("shift", {}).get(name, {})
            def pf(v):
                return f"{v:.1%}" if isinstance(v, float) and not math.isnan(v) else "n/a"
            print(
                f"  {name:<10}{pf(ind.get('recoverable_fraction', float('nan'))):>14}"
                f"{pf(sh.get('recoverable_fraction', float('nan'))):>12}"
                f"{pf(sh.get('admit_rate', float('nan'))):>12}"
            )
        rule = sr.get("two_feature_rule")
        if rule:
            b = rule.get("benchmark", {})
            c = rule.get("calibration", {})
            print(
                f"  two-feature rule: form={rule.get('form')}  "
                f"t_margin={rule.get('t_sid_top1_margin', float('nan')):.4f}  "
                f"t_std={rule.get('t_sid_score_std', float('nan')):.4f}"
            )
            print(
                f"    cal frac={c.get('recoverable_fraction', float('nan')):.1%}  "
                f"bench/shift frac={b.get('recoverable_fraction', float('nan')):.1%}  "
                f"bench admit={b.get('admit_rate', float('nan')):.1%}"
            )
        router_csv = os.path.join(args.output_dir, "voi_hardness_stronger_routers.csv")
        with open(router_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "router", "setting", "n", "recoverable_fraction",
                    "realized_gain_over_bf", "oracle_headroom", "admit_rate",
                    "form", "t_sid_top1_margin", "t_sid_score_std",
                ],
            )
            w.writeheader()
            for name, stats in sr.get("in_distribution", {}).items():
                w.writerow({
                    "router": name, "setting": "in_distribution",
                    "n": stats.get("n"),
                    "recoverable_fraction": stats.get("recoverable_fraction"),
                    "realized_gain_over_bf": stats.get("realized_gain_over_bf"),
                    "oracle_headroom": stats.get("oracle_headroom"),
                    "admit_rate": stats.get("admit_rate"),
                    "form": "", "t_sid_top1_margin": "", "t_sid_score_std": "",
                })
            for name, stats in sr.get("shift", {}).items():
                w.writerow({
                    "router": name, "setting": "shift",
                    "n": stats.get("n"),
                    "recoverable_fraction": stats.get("recoverable_fraction"),
                    "realized_gain_over_bf": stats.get("realized_gain_over_bf"),
                    "oracle_headroom": stats.get("oracle_headroom"),
                    "admit_rate": stats.get("admit_rate"),
                    "form": "", "t_sid_top1_margin": "", "t_sid_score_std": "",
                })
            if rule:
                for setting, stats in (
                    ("rule_calibration", rule.get("calibration", {})),
                    ("rule_benchmark", rule.get("benchmark", {})),
                ):
                    w.writerow({
                        "router": "two_feature_rule", "setting": setting,
                        "n": stats.get("n"),
                        "recoverable_fraction": stats.get("recoverable_fraction"),
                        "realized_gain_over_bf": stats.get("realized_gain_over_bf"),
                        "oracle_headroom": stats.get("oracle_headroom"),
                        "admit_rate": stats.get("admit_rate"),
                        "form": rule.get("form"),
                        "t_sid_top1_margin": rule.get("t_sid_top1_margin"),
                        "t_sid_score_std": rule.get("t_sid_score_std"),
                    })
        print(f"       {router_csv}")

    print("=" * 78)
    print(f"wrote: {os.path.join(args.output_dir, 'voi_hardness_summary.json')}")
    print(f"       {csv_path}")


if __name__ == "__main__":
    main()
