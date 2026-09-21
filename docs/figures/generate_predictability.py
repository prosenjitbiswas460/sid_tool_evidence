#!/usr/bin/env python3
"""Predictability and transport of SID-tool value.

Panel A — leave-one-dataset-out Ridge, predicted vs observed cell-mean Δ
Panel B — recovered fraction of hindsight headroom under shift

Usage (from repo root):
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 MPLBACKEND=Agg \\
    python docs/figures/generate_predictability.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

FEATS_CSV = ROOT / "results/voi_predictive/voi_features_per_task.csv"
PRED_JSON = ROOT / "results/voi_predictive/voi_predictive_summary.json"
HARD_JSON = ROOT / "results/voi_hardness/voi_hardness_summary.json"

FEATURES = [
    "cap",
    "cap_sq",
    "collision_rate",
    "log_mean_leaf",
    "level_util",
    "log_history_chars",
    "log_review_count",
    "metadata_richness",
    "sid_top1_margin",
    "sid_score_std",
    "candidate_sid_diversity",
]

DOMAIN_COLORS = {
    "amazon": "#1D4E89",
    "goodreads": "#C45C26",
    "yelp": "#2F6B4F",
}
DOMAIN_LABELS = {
    "amazon": "Amazon",
    "goodreads": "Goodreads",
    "yelp": "Yelp",
}
BB_MARKERS = {"1.5b": "o", "3b": "s", "7b": "D"}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).correlation)


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with FEATS_CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "backbone": r["backbone"],
                    "dataset": r["dataset"],
                    "condition": r["condition"],
                    "voi": float(r["voi"]),
                    **{k: float(r[k]) for k in FEATURES},
                }
            )
    return rows


def lodo_ridge_cells(rows: list[dict]) -> tuple[list[dict], float, float]:
    """Leave-one-dataset-out Ridge; return OOF cell means + metrics.

    Matches scripts/predict_voi.py deployable LODO Ridge
    (capability already stored in the features CSV under params mode).
    """
    cells: list[dict] = []
    pooled_true: list[float] = []
    pooled_pred: list[float] = []
    fold_sp: list[float] = []

    for held in ("amazon", "goodreads", "yelp"):
        train = [r for r in rows if r["dataset"] != held]
        test = [r for r in rows if r["dataset"] == held]
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        Xtr = np.array([[r[f] for f in FEATURES] for r in train], dtype=float)
        ytr = np.array([r["voi"] for r in train], dtype=float)
        Xte = np.array([[r[f] for f in FEATURES] for r in test], dtype=float)
        yte = np.array([r["voi"] for r in test], dtype=float)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)

        agg_p: dict[str, list[float]] = defaultdict(list)
        agg_t: dict[str, list[float]] = defaultdict(list)
        for r, p, t in zip(test, pred, yte):
            key = f"{r['backbone']}/{r['dataset']}/{r['condition']}"
            agg_p[key].append(float(p))
            agg_t[key].append(float(t))

        cp = np.array([float(np.mean(agg_p[k])) for k in sorted(agg_t)])
        ct = np.array([float(np.mean(agg_t[k])) for k in sorted(agg_t)])
        fold_sp.append(spearman(cp, ct))

        for key in sorted(agg_t):
            true_m = float(np.mean(agg_t[key]))
            pred_m = float(np.mean(agg_p[key]))
            bb, ds, cond = key.split("/")
            cells.append(
                {
                    "key": key,
                    "backbone": bb,
                    "dataset": ds,
                    "condition": cond,
                    "true": true_m,
                    "pred": pred_m,
                }
            )
            pooled_true.append(true_m)
            pooled_pred.append(pred_m)

    pooled_sp = spearman(np.array(pooled_pred), np.array(pooled_true))
    mean_fold_sp = float(np.mean(fold_sp))
    return cells, pooled_sp, mean_fold_sp


def plot() -> None:
    rows = load_rows()
    cells, pooled_sp, mean_fold_sp = lodo_ridge_cells(rows)

    # Sanity: match Table tab:predictive LODO Ridge within rounding.
    summary = json.load(PRED_JSON.open())
    expected = float(summary["lodo"]["linear"]["pooled_cell_spearman"])
    if abs(pooled_sp - expected) > 0.02:
        raise RuntimeError(
            f"LODO Ridge pooled Spearman {pooled_sp:.4f} diverges from "
            f"summary {expected:.4f}; check features / protocol."
        )

    hard = json.load(HARD_JSON.open())
    in_dist = float(hard["aggregate"]["recoverable_fraction"]) * 100.0
    shift_rec = float(hard["shift"]["shift_recoverable_fraction"]) * 100.0
    policy = float(hard["eaa_verification"]["recovered_headroom"]) * 100.0
    auc = float(hard["shift"]["covariate_shift_auc"])

    in_scopes = np.array(
        [float(v["recoverable_fraction"]) * 100.0 for v in hard["per_scope"].values()]
    )
    shift_scopes = np.array(
        [
            float(v["recoverable_fraction"]) * 100.0
            for v in hard["shift"]["per_scope"].values()
        ]
    )

    style()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(9.6, 4.45),
        gridspec_kw={"width_ratios": [1.05, 0.95]},
    )

    # ----- Panel A: LODO Ridge scatter (z-scored; matches Spearman claim) -----
    # Absolute LODO predictions are poorly magnitude-calibrated (Yelp OOF preds
    # ≈0.3 vs observed ≈0). Spearman concerns ranking, so plot standardized
    # observed/predicted cell means over the pooled LODO evaluation.
    xs_raw = np.array([c["true"] for c in cells], dtype=float)
    ys_raw = np.array([c["pred"] for c in cells], dtype=float)
    xs = (xs_raw - xs_raw.mean()) / xs_raw.std(ddof=0)
    ys = (ys_raw - ys_raw.mean()) / ys_raw.std(ddof=0)
    for c, xo, yo in zip(cells, xs, ys):
        c["true_z"] = float(xo)
        c["pred_z"] = float(yo)

    lim = float(max(np.abs(xs).max(), np.abs(ys).max()) + 0.35)
    ax_a.plot([-lim, lim], [-lim, lim], color="#94A3B8", lw=1.0, ls="--", zorder=1)
    ax_a.axhline(0.0, color="#CBD5E1", lw=0.7, zorder=0)
    ax_a.axvline(0.0, color="#CBD5E1", lw=0.7, zorder=0)

    for c in cells:
        ax_a.scatter(
            c["true_z"],
            c["pred_z"],
            s=46,
            color=DOMAIN_COLORS[c["dataset"]],
            marker=BB_MARKERS[c["backbone"]],
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            alpha=0.92,
        )

    ax_a.set_xlim(-lim, lim)
    ax_a.set_ylim(-lim, lim)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel(r"Observed mean $\Delta$NDCG@5, standardized")
    ax_a.set_ylabel(r"Predicted mean $\Delta$NDCG@5, standardized")
    ax_a.set_title("A. Predicting population-level $\\Delta$", fontweight="bold", pad=8)
    ax_a.text(
        0.03,
        0.97,
        f"LODO Ridge\npooled Spearman $= {pooled_sp:.2f}$\n"
        f"mean-fold Spearman $= {mean_fold_sp:.2f}$",
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#334155",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#F8FAFC",
            edgecolor="#E2E8F0",
            lw=0.6,
        ),
        zorder=5,
    )

    from matplotlib.lines import Line2D

    dom_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=DOMAIN_COLORS[d],
            markeredgecolor="white",
            markersize=7,
            label=DOMAIN_LABELS[d],
        )
        for d in ("amazon", "goodreads", "yelp")
    ]
    bb_handles = [
        Line2D(
            [0],
            [0],
            marker=BB_MARKERS[b],
            color="w",
            markerfacecolor="#64748B",
            markeredgecolor="white",
            markersize=7,
            label={"1.5b": "1.5B", "3b": "3B", "7b": "7B"}[b],
        )
        for b in ("1.5b", "3b", "7b")
    ]
    # Stack legends under Panel A (side-by-side collide: Yelp vs 1.5B).
    # Keep clear of the x-axis label.
    leg_dom = ax_a.legend(
        handles=dom_handles,
        title="Domain (color)",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    ax_a.add_artist(leg_dom)
    ax_a.legend(
        handles=bb_handles,
        title="Backbone (shape)",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42),
        ncol=3,
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    # ----- Panel B: recoverable headroom -----
    xs_b = np.array([0.0, 1.15, 2.3])
    vals = np.array([in_dist, shift_rec, policy])
    colors = ["#2563EB", "#DC2626", "#B91C1C"]
    labels = [
        "In-distribution\nrecovery",
        "Shifted\nrecovery",
        "Calibrated\npolicy",
    ]
    ax_b.axhline(0.0, color="#334155", lw=0.9, zorder=1)
    ax_b.bar(xs_b, vals, width=0.72, color=colors, edgecolor="none", zorder=3, alpha=0.93)

    # Whiskers = min–max across scopes (not bootstrap CIs).
    for i, scopes in enumerate((in_scopes, shift_scopes)):
        lo, hi = float(scopes.min()), float(scopes.max())
        ax_b.errorbar(
            xs_b[i],
            vals[i],
            yerr=[[vals[i] - lo], [hi - vals[i]]],
            fmt="none",
            ecolor="#0F172A",
            elinewidth=1.0,
            capsize=3.5,
            zorder=4,
        )

    for x, v in zip(xs_b, vals):
        ax_b.text(
            x,
            v + (2.2 if v >= 0 else -2.2),
            f"{v:+.1f}%",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
            fontweight="medium",
            color="#0F172A",
        )

    ax_b.set_xticks(xs_b)
    ax_b.set_xticklabels(labels, fontsize=9)
    ax_b.set_ylabel(r"Recovered fraction of $H^{*}$ (%)")
    ax_b.set_title(
        f"B. Recoverable headroom under shift\n(shift AUC $= {auc:.3f}$)",
        fontweight="bold",
        pad=8,
        fontsize=11,
    )
    ax_b.set_ylim(-35, 55)

    fig.suptitle(
        "Predictability and transport of SID-tool value",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.34, top=0.86, wspace=0.32)

    for ext in ("pdf", "png"):
        out = OUT / f"predictability.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")
    plt.close(fig)

    caption = r"""\caption{
Predictability and transport of SID-tool value.
Panel A compares observed and predicted backbone--dataset--condition mean
$\Delta\mathrm{NDCG}@5$ using deployable pre-admission features under
leave-one-dataset-out Ridge regression. Both axes are standardized; colors
denote domains and marker shapes denote backbone sizes. Panel B reports
the recovered fraction of hindsight headroom $H^*$ under in-distribution
selection, calibration-to-benchmark shift, and the deployed calibrated
admission policy. Error bars indicate min--max variation across
backbone--dataset scopes.
Although SID-tool value has some population-level structure, the recoverable
value does not transport reliably under shift.
}
\label{fig:predictability-transport}
"""
    (OUT / "predictability_caption.tex").write_text(caption)
    print(f"wrote {OUT / 'predictability_caption.tex'}")


if __name__ == "__main__":
    plot()
