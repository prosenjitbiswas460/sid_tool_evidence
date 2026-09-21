#!/usr/bin/env python3
"""Delimitation of fixed-candidate SID-tool effects.

Whiskers are paired-bootstrap 95% CIs from the significance / VoI exports.

Usage (from repo root):
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 MPLBACKEND=Agg \\
    python docs/figures/generate_delimitation.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

RNG = np.random.default_rng(42)
N_BOOT = 5000

C_POINT = "#334155"      # slate (all points)
C_HURTS = "#9F1239"      # restrained dark red for hurts only
C_ZERO = "#64748B"
C_HEADER = "#0F172A"
C_ROW = "#334155"


def round3(x: float) -> float:
    import decimal

    return float(
        decimal.Decimal(str(x)).quantize(
            decimal.Decimal("0.001"), rounding=decimal.ROUND_HALF_UP
        )
    )


def bootstrap_ci(deltas: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    n = len(deltas)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = float(deltas[RNG.integers(0, n, size=n)].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi)


def load_sig_row(path: Path, *, dataset: str, comparison_substr: str, scope: str) -> dict:
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["metric"] != "ndcg@5":
                continue
            if r["dataset"] != dataset or r["scope"] != scope:
                continue
            if comparison_substr not in r["comparison"]:
                continue
            return {
                "delta": float(r["delta"]),
                "ci_lo": float(r["ci_lo"]),
                "ci_hi": float(r["ci_hi"]),
                "verdict": (r.get("verdict") or "ns").replace("*", ""),
                "n": int(r["n"]),
            }
    raise KeyError(f"No row in {path} for {dataset}/{comparison_substr}/{scope}")


def load_codebook_rows() -> tuple[dict, dict]:
    path = ROOT / "results/voi_codebook_contrast/voi_features_per_task.csv"
    by: dict[str, list[float]] = {"7b_collapsed": [], "7b_fixed": []}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["dataset"] != "yelp":
                continue
            if r["backbone"] in by:
                by[r["backbone"]].append(float(r["voi"]))
    out = {}
    for key, label in (("7b_collapsed", "precursor"), ("7b_fixed", "repaired")):
        arr = np.asarray(by[key], dtype=float)
        mean, lo, hi = bootstrap_ci(arr)
        verd = "hurts" if hi < 0 else ("helps" if lo > 0 else "ns")
        out[label] = {
            "delta": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "verdict": verd,
            "n": int(arr.size),
        }
    return out["precursor"], out["repaired"]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


def plot() -> None:
    nomerge = load_sig_row(
        ROOT / "results/hmt_prompt_only.csv",
        dataset="amazon",
        comparison_substr="nomerge7b_oracle_hmt-eaa_oracle_hm",
        scope="classic",
    )
    full = load_sig_row(
        ROOT / "results/hmt_significance.csv",
        dataset="amazon",
        comparison_substr="HMT-HM[7b]",
        scope="classic",
    )
    nr_am = load_sig_row(
        ROOT / "results/hmt_titles_only.csv",
        dataset="amazon",
        comparison_substr="HMT-HM[nr]",
        scope="POOLED",
    )
    nr_gr = load_sig_row(
        ROOT / "results/hmt_titles_only.csv",
        dataset="goodreads",
        comparison_substr="HMT-HM[nr]",
        scope="POOLED",
    )
    nr_ye = load_sig_row(
        ROOT / "results/hmt_titles_only.csv",
        dataset="yelp",
        comparison_substr="HMT-HM[nr]",
        scope="POOLED",
    )
    precursor, repaired = load_codebook_rows()

    # Top→bottom. Headers are empty-point spacer rows.
    # kind: "header" | "row"
    items: list[tuple[str, str | None, dict | None]] = [
        ("header", "Interface", None),
        ("row", "Full deployed HMT", full),
        ("row", "Prompt-only HMT", nomerge),
        ("header", "History ablation", None),
        ("row", "Titles only: Amazon", nr_am),
        ("row", "Titles only: Goodreads", nr_gr),
        ("row", "Titles only: Yelp", nr_ye),
        ("header", "Codebook quality", None),
        ("row", "Yelp high-collision", precursor),
        ("row", "Yelp repaired SID-v2", repaired),
    ]

    style()
    fig, ax = plt.subplots(figsize=(5.6, 3.35))

    # Assign y: headers and rows; top of figure = first item.
    y = 0.0
    coords: list[tuple[float, str, str | None, dict | None]] = []
    for kind, label, stats in items:
        gap = 0.35 if kind == "header" and coords else 0.0
        if gap:
            y += gap
        coords.append((y, kind, label, stats))
        y += 0.85 if kind == "header" else 1.0
    y_max = coords[-1][0]
    coords = [(y_max - yy, kind, lab, st) for yy, kind, lab, st in coords]

    ax.axvline(0.0, color=C_ZERO, lw=0.55, zorder=1)

    yticks, ylabels, ytick_colors, ytick_weights = [], [], [], []

    for yy, kind, label, stats in coords:
        if kind == "header":
            yticks.append(yy)
            ylabels.append(label)
            ytick_colors.append(C_HEADER)
            ytick_weights.append("bold")
            continue

        assert stats is not None and label is not None
        delta, lo, hi = stats["delta"], stats["ci_lo"], stats["ci_hi"]
        disp = round3(delta)
        hurts = hi < 0
        color = C_HURTS if hurts else C_POINT
        ax.errorbar(
            delta,
            yy,
            xerr=[[delta - lo], [hi - delta]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.4,
            markersize=4.8,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=3,
        )
        # Compact value labels next to CI tips (optional; keep small).
        ax.text(
            hi + 0.005,
            yy,
            f"{disp:+.3f}",
            va="center",
            ha="left",
            fontsize=6.5,
            color="#64748B",
        )
        yticks.append(yy)
        ylabels.append(f"   {label}")
        ytick_colors.append(C_ROW)
        ytick_weights.append("normal")

    # rebuild dump with groups in visual order
    dump2 = []
    cur_g = None
    for yy, kind, label, stats in sorted(coords, key=lambda r: -r[0]):
        if kind == "header":
            cur_g = label
            continue
        assert stats is not None and label is not None
        dump2.append(
            {
                "group": cur_g,
                "label": label,
                "delta": round3(stats["delta"]),
                "ci_lo": round3(stats["ci_lo"]),
                "ci_hi": round3(stats["ci_hi"]),
                "verdict": stats["verdict"],
                "n": stats["n"],
            }
        )

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    for tick, col, weight in zip(ax.get_yticklabels(), ytick_colors, ytick_weights):
        tick.set_color(col)
        tick.set_fontweight(weight)
        tick.set_fontsize(8 if weight == "bold" else 7.5)

    ax.set_xlabel(r"$\Delta\mathrm{NDCG}@5$ relative to HM", fontsize=8.5)
    ax.set_xlim(-0.145, 0.04)
    ax.set_ylim(-0.55, y_max + 0.55)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(False)

    fig.subplots_adjust(left=0.40, right=0.98, top=0.98, bottom=0.14)

    for ext in ("pdf", "png"):
        out = OUT / f"delimitation.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")
    plt.close(fig)

    (OUT / "delimitation.json").write_text(json.dumps(dump2, indent=2))
    print(f"wrote {OUT / 'delimitation.json'}")

    caption = r"""\caption{
Delimitation of fixed-candidate SID-tool effects. Points report mean
$\Delta\mathrm{NDCG}@5$ for interface, history, and codebook contrasts, with
negative values indicating that SID-tool admission hurts relative to HM ranking.
Whiskers indicate paired-bootstrap 95\% confidence intervals.
}
\label{fig:delimitation}
"""
    (OUT / "delimitation_caption.tex").write_text(caption)
    print(f"wrote {OUT / 'delimitation_caption.tex'}")


if __name__ == "__main__":
    plot()
