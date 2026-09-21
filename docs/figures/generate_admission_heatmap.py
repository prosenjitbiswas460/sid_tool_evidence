#!/usr/bin/env python3
"""SID-tool admission heatmap: pooled ΔNDCG@5 (HMT − HM) by backbone and domain.

Usage (from repo root):
  OMP_NUM_THREADS=1 MPLBACKEND=Agg python docs/figures/generate_admission_heatmap.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

TEXT_DARK = "#1E293B"


def round3(x: float) -> float:
    """Three-decimal display rounding, half-away-from-zero."""
    import decimal

    d = decimal.Decimal(str(x)).quantize(
        decimal.Decimal("0.001"), rounding=decimal.ROUND_HALF_UP
    )
    return float(d)


DOMAINS = ["amazon", "goodreads", "yelp"]
DOMAIN_LABELS = {"amazon": "Amazon", "goodreads": "Goodreads", "yelp": "Yelp"}
FAMILIES = ["Qwen", "Llama", "Gemma"]
SIZES = {
    "Qwen": ["1.5B", "3B", "7B"],
    "Llama": ["1B", "3B", "8B"],
    "Gemma": ["2B", "4B", "12B"],
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


def parse_comparison(comp: str) -> tuple[str | None, str | None]:
    m = re.search(r"\[([^\]]+)\]", comp)
    if not m:
        return None, None
    tag = m.group(1).lower().replace("_", "")
    if tag in ("1.5b", "3b", "7b"):
        return "Qwen", "1.5B" if tag == "1.5b" else tag.upper()
    if "llama" in tag:
        mm = re.search(r"(\d+)b", tag)
        if mm:
            return "Llama", f"{mm.group(1)}B"
    if "gemma" in tag:
        mm = re.search(r"(\d+)b", tag)
        if mm:
            return "Gemma", f"{mm.group(1)}B"
    return None, tag


def load_pooled_cells() -> dict:
    path = ROOT / "results/hmt_significance.csv"
    cells = {}
    for r in csv.DictReader(path.open()):
        if r["scope"] != "POOLED" or r["metric"] != "ndcg@5":
            continue
        fam, size = parse_comparison(r["comparison"])
        if fam is None:
            continue
        verd = (r.get("verdict") or "").strip() or "ns"
        raw = float(r["delta"])
        cells[(fam, size, r["dataset"])] = {
            "delta": round3(raw),
            "delta_raw": raw,
            "verdict": verd,
            "sig": "*" in verd,
        }
    return cells


def plot(cells: dict) -> None:
    row_keys = [(fam, sz) for fam in FAMILIES for sz in SIZES[fam]]
    # Diverging colour (magnitude) + * (CI excludes zero).
    fig, ax = plt.subplots(figsize=(6.6, 5.5))
    mat = np.full((len(row_keys), len(DOMAINS)), np.nan)
    sig_mat = np.zeros((len(row_keys), len(DOMAINS)), dtype=bool)
    for i, (fam, sz) in enumerate(row_keys):
        for j, ds in enumerate(DOMAINS):
            c = cells.get((fam, sz, ds))
            if not c:
                continue
            mat[i, j] = c["delta"]
            sig_mat[i, j] = bool(c.get("sig"))

    vmax = float(np.nanmax(np.abs(mat)))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    # Red = negative Δ, blue = positive Δ (diverging, centered at 0).
    cmap = matplotlib.colormaps["RdBu"]
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

    for i in range(len(row_keys)):
        for j in range(len(DOMAINS)):
            if np.isnan(mat[i, j]):
                continue
            # Dark text near zero; light text on saturated red/blue.
            rgba = cmap(norm(mat[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "white" if luminance < 0.55 else TEXT_DARK
            star = "*" if sig_mat[i, j] else ""
            ax.text(
                j,
                i,
                f"{mat[i, j]:+.3f}{star}",
                ha="center",
                va="center",
                color=tc,
                fontsize=9.5,
                fontweight="bold" if sig_mat[i, j] else "medium",
            )
            if sig_mat[i, j]:
                # Bold outline for cells whose CI excludes zero.
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#0F172A",
                    linewidth=1.35,
                    zorder=3,
                )
                ax.add_patch(rect)

    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([sz for _, sz in row_keys])
    ax.set_xlabel("Domain")
    ax.set_title("SID-tool admission effect", fontweight="bold", pad=10)

    for y in [2.5, 5.5]:
        ax.axhline(y, color="white", lw=3.5)
    fam_center = {"Qwen": 1, "Llama": 4, "Gemma": 7}
    for fam, ctr in fam_center.items():
        ax.text(
            -1.15,
            ctr,
            fam,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#334155",
            clip_on=False,
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.6)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta$NDCG@5 (HMT$-$HM)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "admission_delta.pdf")
    fig.savefig(OUT / "admission_delta.png")
    plt.close(fig)

    caption = (
        "SID-tool admission effect in fixed-candidate LLM reranking. Each cell "
        "reports pooled $\\Delta\\mathrm{NDCG}@5 = "
        "\\mathrm{NDCG}@5(\\mathrm{HMT})-\\mathrm{NDCG}@5(\\mathrm{HM})$ "
        "for a backbone--domain group. Colour encodes effect size, with "
        "negative values indicating that admitting SID-derived evidence hurts "
        "HM ranking and positive values indicating improvement. Asterisks "
        "indicate paired-bootstrap 95\\% confidence intervals excluding zero. "
        "The dominant pattern is negative, with one clear positive pooled cell."
    )
    (OUT / "admission_delta_caption.tex").write_text(
        "\\caption{" + caption + "}\n\\label{fig:admission-delta}\n"
    )


def main() -> None:
    style()
    cells = load_pooled_cells()
    assert len(cells) == 27, f"expected 27 pooled cells, got {len(cells)}"
    plot(cells)
    print("Wrote:")
    for p in sorted(OUT.glob("admission_delta.*")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
