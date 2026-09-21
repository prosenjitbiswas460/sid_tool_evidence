#!/usr/bin/env python3
"""Candidate-pool shortlisting: Recall@20 by domain and pool size.

Usage (from repo root):
  MPLBACKEND=Agg python docs/figures/generate_shortlist_recall.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "docs" / "tables" / "shortlist_coverage.csv"

# Domain display order.
DOMAINS = ["Amazon", "Goodreads", "Yelp"]
MS = [100, 200, 500]

C_RAND = "#94A3B8"  # slate
C_BM25 = "#64748B"  # cooler slate
C_EMB = "#0F766E"  # teal
C_SID = "#1D4E89"  # deep blue
C_SAS = "#C45C26"  # terracotta


def load_rows() -> list[dict]:
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def _has_text_baselines(rows: list[dict]) -> bool:
    for r in rows:
        if (r.get("bm25") or "").strip() and (r.get("embnn") or "").strip():
            return True
    return False


def main() -> None:
    rows = load_rows()
    by = {(r["domain"], int(r["M"])): r for r in rows}
    use_text = _has_text_baselines(rows)

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
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.2 if use_text else 9.2, 3.55), sharey=True)
    x = np.arange(len(MS))
    y_max = 0.0

    if use_text:
        # Five methods: Rand, BM25, EmbNN, SID, SASRec
        width = 0.15
        offsets = {
            "rand": -2 * width,
            "bm25": -width,
            "embnn": 0.0,
            "sid": width,
            "sasrec": 2 * width,
        }
        series = [
            ("rand", "Random", C_RAND),
            ("bm25", "BM25", C_BM25),
            ("embnn", "EmbNN", C_EMB),
            ("sid", "SID", C_SID),
            ("sasrec", "SASRec", C_SAS),
        ]
    else:
        width = 0.26
        offsets = {"rand": -width, "sid": 0.0, "sasrec": width}
        series = [
            ("rand", "Random", C_RAND),
            ("sid", "SID", C_SID),
            ("sasrec", "SASRec", C_SAS),
        ]

    for ax, dom in zip(axes, DOMAINS):
        vals = {}
        for key, _label, _color in series:
            vals[key] = [float(by[(dom, m)][key]) for m in MS]
        lifts = [float(by[(dom, m)]["sid_minus_rand"]) for m in MS]

        for key, label, color in series:
            ax.bar(
                x + offsets[key],
                vals[key],
                width,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=label,
                zorder=2,
            )

        # Lift labels sit above the tallest bar in each M group.
        pad = 0.025
        for i, lift in enumerate(lifts):
            top = max(vals[k][i] for k in vals)
            ax.text(
                x[i],
                top + pad,
                f"+{lift:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.0 if use_text else 7.5,
                color=C_SID,
                fontweight="medium",
                clip_on=False,
                zorder=4,
            )
            y_max = max(y_max, top + pad + 0.04)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [rf"$M_{{\mathrm{{pool}}}}={m}$" for m in MS],
            fontsize=8.5,
        )
        ax.set_title(dom, pad=6)
        ax.set_xlabel(r"Pool size $M_{\mathrm{pool}}$")

    for ax in axes:
        ax.set_ylim(0.0, max(0.45, y_max))

    axes[0].set_ylabel(r"Recall@$20$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(series),
        frameon=False,
        fontsize=8.5 if use_text else 9,
    )

    fig.suptitle(
        "Candidate-pool shortlisting: Recall@20",
        y=1.02,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))

    for ext in ("pdf", "png"):
        out = OUT / f"shortlist_recall.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")

    if use_text:
        caption = r"""\caption{Candidate-pool shortlisting coverage. Bars report Recall@$20$ for
random shortlisting, BM25, content-embedding nearest-neighbor (EmbNN; frozen
Flan-T5 item embeddings), SID-based shortlisting, and SASRec over identical
candidate universes with $M_{\mathrm{pool}}\in\{100,200,500\}$. Blue
annotations report the absolute Recall@$20$ lift of SID over random.
BM25 and EmbNN are training-free text baselines on the same $U_t$.}
\label{fig:shortlisting-coverage}
"""
    else:
        caption = r"""\caption{Candidate-pool shortlisting coverage. Bars report Recall@$20$ for random
shortlisting, SID-based shortlisting, and SASRec over identical candidate
universes with $M_{\mathrm{pool}}\in\{100,200,500\}$. Blue annotations above
SID bars report the absolute Recall@$20$ lift of SID over random shortlisting.
SID improves over random in every domain--pool-size cell, while comparison with
SASRec is domain-dependent.}
\label{fig:shortlisting-coverage}
"""
    path = OUT / "shortlist_recall_caption.tex"
    path.write_text(caption)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
