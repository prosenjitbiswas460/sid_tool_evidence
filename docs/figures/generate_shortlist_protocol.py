#!/usr/bin/env python3
"""Shortlisting protocol: universe → shortlist → outcomes.

Usage (repo root):
  MPLBACKEND=Agg python docs/figures/generate_shortlist_protocol.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    FancyBboxPatch,
    Rectangle,
    FancyArrowPatch,
    Arc,
    PathPatch,
)
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parent

# DeepMind / Nature-adjacent palette
INK = "#1A1A1A"
MUTED = "#5F6368"
FAINT = "#9AA0A6"
PANEL = "#F5F7FA"
PANEL_EDGE = "#E8EAED"
TEAL = "#0D9488"
TEAL_DEEP = "#0F766E"
TEAL_SOFT = "#CCFBF1"
AMBER = "#D97706"
AMBER_SOFT = "#FEF3C7"
NAVY = "#1E3A5F"
WHITE = "#FFFFFF"
HAIR = "#D1D5DB"


def rounded(ax, x, y, w, h, fc=PANEL, ec=PANEL_EDGE, lw=0.8, r=0.12, z=1):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, *, size=9, color=INK, weight="regular", ha="left", va="center", **kw):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        fontfamily="sans-serif",
        zorder=5,
        **kw,
    )


# ----- Icons -----------------------------------------------------------------

def icon_universe(ax, cx, cy, scale=1.0):
    """Soft constellation + one GT highlight."""
    rng = np.random.default_rng(7)
    n = 48
    # elliptical cloud
    xs = cx + rng.normal(0, 0.38 * scale, n)
    ys = cy + rng.normal(0, 0.22 * scale, n)
    sizes = rng.uniform(8, 28, n) * scale
    alphas = rng.uniform(0.25, 0.7, n)
    for x, y, s, a in zip(xs, ys, sizes, alphas):
        ax.scatter([x], [y], s=s, c=NAVY, alpha=a, linewidths=0, zorder=3)
    # GT
    ax.scatter([cx + 0.12 * scale], [cy + 0.05 * scale], s=55 * scale, c=TEAL, zorder=4, linewidths=0)
    ax.scatter(
        [cx + 0.12 * scale],
        [cy + 0.05 * scale],
        s=95 * scale,
        facecolors="none",
        edgecolors=TEAL,
        linewidths=1.1,
        zorder=4,
    )
    txt(ax, cx + 0.28 * scale, cy + 0.18 * scale, "GT", size=7, color=TEAL_DEEP, weight="medium")


def icon_sid_tokens(ax, cx, cy, scale=1.0):
    """Three hierarchical SID code tokens."""
    w, h, gap = 0.22 * scale, 0.14 * scale, 0.06 * scale
    colors = ["#99F6E4", TEAL, TEAL_DEEP]
    labels = ["c₁", "c₂", "c₃"]
    total_w = 3 * w + 2 * gap
    x0 = cx - total_w / 2
    for i, (c, lab) in enumerate(zip(colors, labels)):
        x = x0 + i * (w + gap)
        rounded(ax, x, cy - h / 2, w, h, fc=c, ec=TEAL_DEEP, lw=0.7, r=0.04, z=4)
        txt(
            ax,
            x + w / 2,
            cy,
            lab,
            size=6.5,
            color=INK if i == 0 else WHITE,
            weight="medium",
            ha="center",
        )


def icon_random(ax, cx, cy, scale=1.0):
    """Shuffled dots."""
    rng = np.random.default_rng(3)
    for _ in range(12):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0.05, 0.22) * scale
        ax.scatter(
            [cx + rad * np.cos(ang)],
            [cy + rad * np.sin(ang)],
            s=rng.uniform(12, 26) * scale,
            c=MUTED,
            alpha=0.55,
            linewidths=0,
            zorder=4,
        )


def icon_sasrec(ax, cx, cy, scale=1.0):
    """Sequence of nodes with progression."""
    xs = np.linspace(cx - 0.28 * scale, cx + 0.18 * scale, 4)
    for i, x in enumerate(xs):
        ax.add_patch(
            Circle(
                (x, cy),
                0.055 * scale,
                facecolor=AMBER if i == 3 else "#FDE68A",
                edgecolor=AMBER,
                lw=0.8,
                zorder=4,
            )
        )
        if i < 3:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.06 * scale, cy),
                xytext=(x + 0.06 * scale, cy),
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.9),
                zorder=3,
            )


def icon_target(ax, cx, cy, scale=1.0):
    for r, lw in [(0.20 * scale, 1.0), (0.13 * scale, 1.0), (0.06 * scale, 0)]:
        ax.add_patch(
            Circle(
                (cx, cy),
                r,
                facecolor=TEAL if r < 0.07 * scale else "none",
                edgecolor=TEAL_DEEP,
                lw=lw,
                zorder=4,
            )
        )


def icon_ranked(ax, cx, cy, scale=1.0):
    heights = [0.28, 0.20, 0.14, 0.09]
    colors = [TEAL_DEEP, TEAL, "#5EEAD4", "#99F6E4"]
    x0 = cx - 0.22 * scale
    for i, (h, c) in enumerate(zip(heights, colors)):
        ax.add_patch(
            Rectangle(
                (x0 + i * 0.12 * scale, cy - 0.14 * scale),
                0.09 * scale,
                h * scale,
                facecolor=c,
                edgecolor="none",
                zorder=4,
            )
        )


def hairline(ax, x0, y0, x1, y1):
    """Thin elegant connector — not a flowchart arrow."""
    ax.plot([x0, x1], [y0, y1], color=HAIR, lw=0.9, solid_capstyle="round", zorder=0)
    # small diamond node
    ax.scatter([(x0 + x1) / 2], [(y0 + y1) / 2], s=12, c=FAINT, zorder=1, linewidths=0)


def main() -> None:
    fig_w, fig_h = 11.6, 4.15
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    # Title
    txt(
        ax,
        0.35,
        3.95,
        "CANDIDATE-POOL SHORTLISTING UNDER CONTEXT CONSTRAINTS",
        size=9.0,
        color=TEAL_DEEP,
        weight="bold",
    )

    # Three panels
    panels = [
        (0.25, 0.55, 3.55, 3.15),
        (4.15, 0.55, 3.7, 3.15),
        (8.2, 0.55, 3.55, 3.15),
    ]
    for x, y, w, h in panels:
        rounded(ax, x, y, w, h, fc=PANEL, ec=PANEL_EDGE, lw=0.9, r=0.14, z=1)

    # Subtle connectors between panels (hairlines, not arrows)
    hairline(ax, 3.8, 2.1, 4.15, 2.1)
    hairline(ax, 7.85, 2.1, 8.2, 2.1)

    # ========== Panel 01 Universe ==========
    x0, y0, w0, h0 = panels[0]
    txt(ax, x0 + 0.22, y0 + h0 - 0.28, "01", size=8, color=TEAL, weight="bold")
    txt(ax, x0 + 0.55, y0 + h0 - 0.28, "UNIVERSE", size=8, color=INK, weight="bold")
    icon_universe(ax, x0 + w0 / 2, y0 + h0 / 2 + 0.15, scale=1.15)
    txt(
        ax,
        x0 + w0 / 2,
        y0 + 0.85,
        r"$|U_t| = M_{\mathrm{pool}} \in \{100,200,500\}$",
        size=8.4,
        color=INK,
        ha="center",
        weight="medium",
    )
    txt(
        ax,
        x0 + w0 / 2,
        y0 + 0.52,
        "Held-out ground truth is in pool but not revealed to shortlisters",
        size=7.5,
        color=MUTED,
        ha="center",
    )

    # ========== Panel 02 Shortlist ==========
    x1, y1, w1, h1 = panels[1]
    txt(ax, x1 + 0.22, y1 + h1 - 0.28, "02", size=8, color=TEAL, weight="bold")
    txt(ax, x1 + 0.55, y1 + h1 - 0.28, "SHORTLIST", size=8, color=INK, weight="bold")
    txt(ax, x1 + w1 - 0.22, y1 + h1 - 0.28, r"$L=20$", size=8, color=MUTED, ha="right")

    # SID card (dominant)
    card_x = x1 + 0.2
    card_w = w1 - 0.4
    # SID
    rounded(ax, card_x, y1 + 2.00, card_w, 0.68, fc=WHITE, ec=TEAL, lw=1.35, r=0.1, z=2)
    ax.add_patch(Rectangle((card_x, y1 + 2.00), 0.08, 0.68, facecolor=TEAL, edgecolor="none", zorder=3))
    icon_sid_tokens(ax, card_x + 0.55, y1 + 2.34, scale=0.9)
    txt(ax, card_x + 1.15, y1 + 2.44, "SID scorer", size=9, color=INK, weight="bold")
    txt(ax, card_x + 1.15, y1 + 2.16, r"Training-free  ·  top-$L$ over $U_t$", size=7, color=MUTED)
    txt(ax, card_x + card_w - 0.15, y1 + 2.34, "PRIMARY", size=6.2, color=TEAL_DEEP, ha="right", weight="bold")

    # Random (quiet)
    rounded(ax, card_x, y1 + 1.18, card_w, 0.62, fc=WHITE, ec=PANEL_EDGE, lw=0.9, r=0.1, z=2)
    icon_random(ax, card_x + 0.55, y1 + 1.49, scale=0.9)
    txt(ax, card_x + 1.15, y1 + 1.58, "Random", size=8.5, color=INK, weight="medium")
    txt(ax, card_x + 1.15, y1 + 1.32, "Uniform sample · same universe", size=6.8, color=MUTED)

    # SASRec (two-line subtitle; padded so text never clips)
    rounded(ax, card_x, y1 + 0.22, card_w, 0.78, fc=WHITE, ec="#FDE68A", lw=0.95, r=0.1, z=2)
    ax.add_patch(
        Rectangle((card_x, y1 + 0.22), 0.08, 0.78, facecolor=AMBER, edgecolor="none", zorder=3, alpha=0.85)
    )
    icon_sasrec(ax, card_x + 0.55, y1 + 0.62, scale=0.8)
    txt(ax, card_x + 1.15, y1 + 0.78, "SASRec", size=8.5, color=INK, weight="medium")
    txt(ax, card_x + 1.15, y1 + 0.56, "Trained sequential baseline", size=6.8, color=MUTED)
    txt(ax, card_x + 1.15, y1 + 0.38, "GT excluded from history", size=6.8, color=MUTED)

    # ========== Panel 03 Outcomes ==========
    x2, y2, w2, h2 = panels[2]
    txt(ax, x2 + 0.22, y2 + h2 - 0.28, "03", size=8, color=TEAL, weight="bold")
    txt(ax, x2 + 0.55, y2 + h2 - 0.28, "OUTCOMES", size=8, color=INK, weight="bold")

    # Coverage tile
    rounded(ax, x2 + 0.22, y2 + 1.75, w2 - 0.44, 0.95, fc=WHITE, ec=PANEL_EDGE, lw=0.9, r=0.1, z=2)
    icon_target(ax, x2 + 0.7, y2 + 2.25, scale=0.85)
    txt(ax, x2 + 1.15, y2 + 2.42, "Coverage", size=9, color=INK, weight="bold")
    txt(ax, x2 + 1.15, y2 + 2.12, r"Recall@$L$  ·  GT reaches shortlist?", size=7.2, color=MUTED)

    # Conversion tile
    rounded(ax, x2 + 0.22, y2 + 0.65, w2 - 0.44, 0.95, fc=WHITE, ec=PANEL_EDGE, lw=0.9, r=0.1, z=2)
    icon_ranked(ax, x2 + 0.7, y2 + 1.12, scale=0.9)
    txt(ax, x2 + 1.15, y2 + 1.32, "Conversion", size=9, color=INK, weight="bold")
    txt(ax, x2 + 1.15, y2 + 1.02, "HR@5 · NDCG@5  ·  absent GT → 0", size=7.2, color=INK)

    txt(
        ax,
        x2 + w2 / 2,
        y2 + 0.32,
        "LLM ranks shortlist under HM (primary); HMT is a control",
        size=7.6,
        color=MUTED,
        ha="center",
    )

    # Footer (darker for print; ~12% larger for readability)
    rounded(ax, 0.25, 0.10, 11.5, 0.36, fc="#EEF2F6", ec=PANEL_EDGE, lw=0.7, r=0.08, z=1)
    txt(
        ax,
        0.45,
        0.28,
        "Identical universes within each domain–pool-size cell · "
        "Primary contrast: SID vs random · SASRec = trained sequential baseline",
        size=8.3,
        color=INK,
        weight="medium",
    )

    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        out = OUT / f"shortlist_protocol.{ext}"
        fig.savefig(out, dpi=350, bbox_inches="tight", facecolor=WHITE, pad_inches=0.08)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
