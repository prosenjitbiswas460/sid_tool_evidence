#!/usr/bin/env python3
"""Two deployment roles for the same SID scorer.

Usage:
  MPLBACKEND=Agg python docs/figures/generate_deployment_roles.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Arc,
    Circle,
    FancyBboxPatch,
    FancyArrowPatch,
    Polygon,
    Rectangle,
    Wedge,
)

OUT = Path(__file__).resolve().parent

INK = "#1A1A1A"
MUTED = "#5F6368"
FAINT = "#9AA0A6"
PANEL = "#F5F7FA"
PANEL_EDGE = "#E8EAED"
TEAL = "#0D9488"
TEAL_DEEP = "#0F766E"
TEAL_SOFT = "#CCFBF1"
AMBER = "#D97706"
NAVY = "#1E3A5F"
WHITE = "#FFFFFF"
HAIR = "#CBD5E1"
ROSE = "#BE123C"
ROSE_SOFT = "#FFF1F2"
BLUE_SOFT = "#DBEAFE"
BLUE = "#2563EB"


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
        zorder=6,
        **kw,
    )


# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------

def icon_sid_tokens(ax, cx, cy, scale=1.0):
    w, h, gap = 0.18 * scale, 0.12 * scale, 0.045 * scale
    colors = ["#99F6E4", TEAL, TEAL_DEEP]
    labels = ["c₁", "c₂", "c₃"]
    total = 3 * w + 2 * gap
    x0 = cx - total / 2
    for i, (c, lab) in enumerate(zip(colors, labels)):
        x = x0 + i * (w + gap)
        rounded(ax, x, cy - h / 2, w, h, fc=c, ec=TEAL_DEEP, lw=0.65, r=0.035, z=4)
        txt(
            ax,
            x + w / 2,
            cy,
            lab,
            size=5.8,
            color=INK if i == 0 else WHITE,
            weight="medium",
            ha="center",
        )


def icon_history(ax, cx, cy, scale=1.0):
    """User-history document / chat lines (H)."""
    rounded(
        ax,
        cx - 0.20 * scale,
        cy - 0.22 * scale,
        0.40 * scale,
        0.44 * scale,
        fc=BLUE_SOFT,
        ec=BLUE,
        lw=0.9,
        r=0.05,
        z=4,
    )
    for i, w in enumerate([0.26, 0.20, 0.16]):
        ax.add_patch(
            Rectangle(
                (cx - 0.13 * scale, cy + 0.10 * scale - i * 0.11 * scale),
                w * scale,
                0.04 * scale,
                facecolor=BLUE,
                alpha=0.55 + 0.15 * i,
                edgecolor="none",
                zorder=5,
            )
        )


def icon_metadata(ax, cx, cy, scale=1.0):
    """Item card / tag (M)."""
    rounded(
        ax,
        cx - 0.22 * scale,
        cy - 0.18 * scale,
        0.44 * scale,
        0.36 * scale,
        fc="#FEF3C7",
        ec=AMBER,
        lw=0.9,
        r=0.05,
        z=4,
    )
    # price/tag notch
    ax.add_patch(
        Circle((cx + 0.12 * scale, cy + 0.08 * scale), 0.035 * scale, facecolor=AMBER, zorder=5)
    )
    ax.plot(
        [cx - 0.12 * scale, cx + 0.05 * scale],
        [cy - 0.02 * scale, cy - 0.02 * scale],
        color=AMBER,
        lw=1.2,
        zorder=5,
    )


def icon_stool(ax, cx, cy, scale=1.0):
    """S_tool: SID tokens + score bars."""
    icon_sid_tokens(ax, cx - 0.02 * scale, cy + 0.08 * scale, scale=0.7 * scale)
    # mini score bars under tokens
    for i, h in enumerate([0.10, 0.16, 0.07]):
        ax.add_patch(
            Rectangle(
                (cx - 0.16 * scale + i * 0.12 * scale, cy - 0.18 * scale),
                0.08 * scale,
                h * scale,
                facecolor=TEAL if i == 1 else "#5EEAD4",
                edgecolor="none",
                zorder=5,
            )
        )


def icon_llm(ax, cx, cy, scale=1.0):
    """LLM brain / chip."""
    rounded(
        ax,
        cx - 0.26 * scale,
        cy - 0.18 * scale,
        0.52 * scale,
        0.36 * scale,
        fc=WHITE,
        ec=TEAL_DEEP,
        lw=1.15,
        r=0.07,
        z=4,
    )
    # neural nodes
    pts = [
        (cx - 0.12 * scale, cy + 0.05 * scale),
        (cx, cy - 0.02 * scale),
        (cx + 0.12 * scale, cy + 0.05 * scale),
        (cx - 0.06 * scale, cy - 0.08 * scale),
        (cx + 0.06 * scale, cy - 0.08 * scale),
    ]
    for a, b in [(0, 1), (1, 2), (0, 3), (2, 4), (1, 3), (1, 4)]:
        ax.plot(
            [pts[a][0], pts[b][0]],
            [pts[a][1], pts[b][1]],
            color=TEAL,
            lw=0.7,
            alpha=0.7,
            zorder=5,
        )
    for x, y in pts:
        ax.add_patch(Circle((x, y), 0.028 * scale, facecolor=TEAL_DEEP, zorder=6))


def icon_scores_merge_fallback(ax, cx, cy, scale=1.0):
    """Three mini glyphs: score · merge · fallback."""
    # score: ascending bars
    x0 = cx - 0.38 * scale
    for i, h in enumerate([0.08, 0.14, 0.20]):
        ax.add_patch(
            Rectangle(
                (x0 + i * 0.07 * scale, cy - 0.10 * scale),
                0.05 * scale,
                h * scale,
                facecolor=TEAL,
                alpha=0.55 + 0.15 * i,
                edgecolor="none",
                zorder=5,
            )
        )
    txt(ax, x0 + 0.1 * scale, cy - 0.24 * scale, "scores", size=6.0, color=MUTED, ha="center", weight="medium")

    # merge: two streams into one
    x1 = cx + 0.02 * scale
    ax.annotate(
        "",
        xy=(x1 + 0.08 * scale, cy),
        xytext=(x1 - 0.08 * scale, cy + 0.10 * scale),
        arrowprops=dict(arrowstyle="->", color=TEAL_DEEP, lw=0.9),
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(x1 + 0.08 * scale, cy),
        xytext=(x1 - 0.08 * scale, cy - 0.10 * scale),
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.9),
        zorder=5,
    )
    txt(ax, x1, cy - 0.24 * scale, "merge", size=6.0, color=MUTED, ha="center", weight="medium")

    # fallback: shield / safety
    x2 = cx + 0.32 * scale
    shield = Polygon(
        [
            (x2, cy + 0.14 * scale),
            (x2 + 0.12 * scale, cy + 0.06 * scale),
            (x2 + 0.12 * scale, cy - 0.06 * scale),
            (x2, cy - 0.14 * scale),
            (x2 - 0.12 * scale, cy - 0.06 * scale),
            (x2 - 0.12 * scale, cy + 0.06 * scale),
        ],
        closed=True,
        facecolor=TEAL_SOFT,
        edgecolor=TEAL_DEEP,
        lw=0.85,
        zorder=5,
    )
    ax.add_patch(shield)
    txt(ax, x2, cy - 0.24 * scale, "fallback", size=6.0, color=MUTED, ha="center", weight="medium")


def icon_list20(ax, cx, cy, scale=1.0):
    for i in range(5):
        ax.add_patch(
            Rectangle(
                (cx - 0.26 * scale, cy + 0.20 * scale - i * 0.10 * scale),
                0.52 * scale,
                0.065 * scale,
                facecolor=NAVY,
                alpha=0.30 + 0.12 * i,
                edgecolor="none",
                zorder=4,
            )
        )
    # lock badge = closed set
    ax.add_patch(
        Circle((cx + 0.28 * scale, cy + 0.18 * scale), 0.07 * scale, facecolor=WHITE, edgecolor=NAVY, lw=0.9, zorder=5)
    )
    ax.add_patch(
        Arc(
            (cx + 0.28 * scale, cy + 0.22 * scale),
            0.08 * scale,
            0.08 * scale,
            theta1=0,
            theta2=180,
            color=NAVY,
            lw=0.9,
            zorder=5,
        )
    )


def icon_universe(ax, cx, cy, scale=1.0):
    rng = np.random.default_rng(11)
    n = 34
    xs = cx + rng.normal(0, 0.30 * scale, n)
    ys = cy + rng.normal(0, 0.16 * scale, n)
    sizes = rng.uniform(6, 20, n) * scale
    for x, y, s in zip(xs, ys, sizes):
        ax.scatter([x], [y], s=s, c=NAVY, alpha=0.42, linewidths=0, zorder=3)
    ax.scatter([cx + 0.08 * scale], [cy + 0.02 * scale], s=40 * scale, c=TEAL, zorder=4, linewidths=0)
    ax.scatter(
        [cx + 0.08 * scale],
        [cy + 0.02 * scale],
        s=70 * scale,
        facecolors="none",
        edgecolors=TEAL,
        linewidths=1.0,
        zorder=4,
    )


def icon_shortlist(ax, cx, cy, scale=1.0):
    for i in range(4):
        ax.add_patch(
            Circle(
                (cx - 0.18 * scale + i * 0.12 * scale, cy + 0.04 * scale),
                0.048 * scale,
                facecolor=TEAL if i < 2 else "#99F6E4",
                edgecolor=TEAL_DEEP,
                lw=0.65,
                zorder=4,
            )
        )
    # funnel hint
    ax.plot(
        [cx - 0.28 * scale, cx - 0.10 * scale, cx + 0.10 * scale, cx + 0.28 * scale],
        [cy + 0.22 * scale, cy + 0.14 * scale, cy + 0.14 * scale, cy + 0.22 * scale],
        color=FAINT,
        lw=0.8,
        zorder=3,
    )


def icon_coverage(ax, cx, cy, scale=1.0):
    for r, fill in [(0.16 * scale, "none"), (0.10 * scale, "none"), (0.045 * scale, TEAL)]:
        ax.add_patch(
            Circle((cx, cy), r, facecolor=fill if fill != "none" else "none", edgecolor=TEAL_DEEP, lw=1.0, zorder=4)
        )


def icon_hurts(ax, cx, cy, scale=1.0):
    """Downward trend mark for 'usually hurts'."""
    ax.add_patch(
        Circle((cx, cy), 0.15 * scale, facecolor=ROSE_SOFT, edgecolor=ROSE, lw=1.1, zorder=4)
    )
    ax.annotate(
        "",
        xy=(cx + 0.06 * scale, cy - 0.07 * scale),
        xytext=(cx - 0.06 * scale, cy + 0.07 * scale),
        arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.4),
        zorder=5,
    )


def hairline(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color=HAIR, lw=1.0, solid_capstyle="round", zorder=0)
    ax.scatter([(x0 + x1) / 2], [(y0 + y1) / 2], s=14, c=FAINT, zorder=1, linewidths=0)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.7))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.85)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    # Title
    txt(ax, 0.3, 4.58, "TWO DEPLOYMENT ROLES", size=9, color=TEAL_DEEP, weight="bold")
    txt(
        ax,
        2.95,
        4.58,
        "·  Same non-generative SID scorer · different jobs",
        size=8.5,
        color=MUTED,
    )

    # Shared SID pill
    rounded(ax, 3.7, 3.88, 5.0, 0.52, fc=TEAL_SOFT, ec=TEAL, lw=1.25, r=0.22, z=2)
    icon_sid_tokens(ax, 4.45, 4.14, scale=0.95)
    txt(ax, 5.3, 4.22, "SID scorer", size=9.5, color=INK, weight="bold")
    txt(ax, 5.3, 3.98, r"Non-generative · emits $S_{\mathrm{tool}}$ over candidates", size=6.8, color=MUTED)
    txt(ax, 8.4, 4.14, "SHARED", size=6.8, color=TEAL_DEEP, weight="bold", ha="right")

    ax.plot([4.5, 2.5], [3.88, 3.62], color=TEAL, lw=1.0, alpha=0.5, zorder=0)
    ax.plot([7.9, 9.7], [3.88, 3.62], color=TEAL, lw=1.0, alpha=0.5, zorder=0)

    # ===== ROLE A =====
    lx, ly, lw, lh = 0.2, 0.72, 5.8, 2.8
    rounded(ax, lx, ly, lw, lh, fc=PANEL, ec=PANEL_EDGE, lw=1.0, r=0.14, z=1)
    txt(ax, lx + 0.2, ly + lh - 0.22, "ROLE A", size=7.8, color=TEAL, weight="bold")
    txt(ax, lx + 0.95, ly + lh - 0.22, "CLOSED-SET EVIDENCE ADMISSION", size=8.2, color=INK, weight="bold")

    # Fixed candidates column
    icon_list20(ax, lx + 0.85, ly + 1.95, scale=1.05)
    txt(ax, lx + 0.85, ly + 1.25, "Fixed list", size=7.6, color=INK, weight="bold", ha="center")
    txt(ax, lx + 0.85, ly + 1.02, r"$|C_t|=20$", size=7.0, color=MUTED, ha="center")
    txt(ax, lx + 0.85, ly + 0.82, "already in context", size=6.3, color=FAINT, ha="center")

    # HM card with H + M icons
    rounded(ax, lx + 1.7, ly + 1.85, 1.85, 1.15, fc=WHITE, ec=PANEL_EDGE, lw=1.0, r=0.1, z=2)
    txt(ax, lx + 2.62, ly + 2.78, "HM", size=9.5, color=INK, weight="bold", ha="center")
    icon_history(ax, lx + 2.15, ly + 2.35, scale=0.85)
    icon_metadata(ax, lx + 3.05, ly + 2.35, scale=0.85)
    txt(ax, lx + 2.15, ly + 2.05, "H", size=6.5, color=BLUE, weight="bold", ha="center")
    txt(ax, lx + 3.05, ly + 2.05, "M", size=6.5, color=AMBER, weight="bold", ha="center")
    txt(ax, lx + 2.62, ly + 1.95, "text only", size=6.3, color=FAINT, ha="center")

    # HMT card — heavier weight
    rounded(ax, lx + 3.75, ly + 1.85, 2.05, 1.15, fc=WHITE, ec=TEAL, lw=1.4, r=0.1, z=2)
    ax.add_patch(Rectangle((lx + 3.75, ly + 1.85), 0.08, 1.15, facecolor=TEAL, edgecolor="none", zorder=3))
    txt(ax, lx + 4.85, ly + 2.78, "HMT", size=9.5, color=INK, weight="bold", ha="center")
    # mini icons row: H M S_tool
    icon_history(ax, lx + 4.15, ly + 2.4, scale=0.65)
    icon_metadata(ax, lx + 4.7, ly + 2.4, scale=0.65)
    icon_stool(ax, lx + 5.35, ly + 2.4, scale=0.7)
    txt(ax, lx + 4.15, ly + 2.08, "H", size=5.8, color=BLUE, weight="bold", ha="center")
    txt(ax, lx + 4.7, ly + 2.08, "M", size=5.8, color=AMBER, weight="bold", ha="center")
    txt(ax, lx + 5.35, ly + 2.08, r"$S_{\mathrm{tool}}$", size=5.8, color=TEAL_DEEP, weight="bold", ha="center")
    txt(ax, lx + 4.85, ly + 1.92, "+ SID evidence", size=6.2, color=TEAL_DEEP, ha="center")

    # Deployed interface strip
    rounded(ax, lx + 1.7, ly + 1.05, 4.1, 0.65, fc=WHITE, ec=TEAL, lw=0.95, r=0.09, z=2)
    txt(ax, lx + 1.9, ly + 1.48, "Deployed interface", size=6.8, color=TEAL_DEEP, weight="bold")
    icon_scores_merge_fallback(ax, lx + 4.0, ly + 1.28, scale=1.05)

    # Estimand + LLM
    rounded(ax, lx + 1.7, ly + 0.2, 3.15, 0.7, fc=WHITE, ec=PANEL_EDGE, lw=0.9, r=0.09, z=2)
    txt(ax, lx + 1.9, ly + 0.7, "Paired estimand", size=6.6, color=MUTED, weight="medium")
    txt(
        ax,
        lx + 1.9,
        ly + 0.42,
        r"$\Delta=\mathrm{NDCG}@5(\mathrm{HMT})-\mathrm{NDCG}@5(\mathrm{HM})$",
        size=7.4,
        color=INK,
        weight="medium",
    )
    # LLM badge
    rounded(ax, lx + 5.0, ly + 0.2, 0.8, 0.7, fc=WHITE, ec=TEAL_DEEP, lw=1.1, r=0.09, z=2)
    icon_llm(ax, lx + 5.4, ly + 0.55, scale=0.75)
    txt(ax, lx + 5.4, ly + 0.28, "LLM", size=6.5, color=TEAL_DEEP, weight="bold", ha="center")

    # Verdict chip
    rounded(ax, lx + 0.25, ly + 0.2, 1.25, 0.7, fc=ROSE_SOFT, ec=ROSE, lw=0.9, r=0.09, z=2)
    icon_hurts(ax, lx + 0.48, ly + 0.55, scale=0.75)
    txt(ax, lx + 0.78, ly + 0.58, "usually", size=6.4, color=ROSE, weight="bold")
    txt(ax, lx + 0.78, ly + 0.35, "hurts / null", size=6.4, color=ROSE, weight="bold")

    # ===== ROLE B =====
    rx, ry, rw, rh = 6.35, 0.72, 5.8, 2.8
    rounded(ax, rx, ry, rw, rh, fc=PANEL, ec=PANEL_EDGE, lw=1.0, r=0.14, z=1)
    txt(ax, rx + 0.2, ry + rh - 0.22, "ROLE B", size=7.8, color=TEAL, weight="bold")
    txt(ax, rx + 0.95, ry + rh - 0.22, "POOL → SHORTLIST → RANK", size=8.2, color=INK, weight="bold")

    # Universe
    rounded(ax, rx + 0.22, ry + 1.55, 1.65, 1.35, fc=WHITE, ec=PANEL_EDGE, lw=1.0, r=0.1, z=2)
    icon_universe(ax, rx + 1.05, ry + 2.35, scale=0.95)
    txt(ax, rx + 1.05, ry + 1.85, "Universe", size=7.8, color=INK, weight="bold", ha="center")
    txt(ax, rx + 1.05, ry + 1.62, r"$M_{\mathrm{pool}}\!\gg\!20$", size=6.6, color=MUTED, ha="center")

    # SID filter (primary)
    rounded(ax, rx + 2.1, ry + 1.55, 1.85, 1.35, fc=WHITE, ec=TEAL, lw=1.4, r=0.1, z=2)
    ax.add_patch(Rectangle((rx + 2.1, ry + 1.55), 0.08, 1.35, facecolor=TEAL, edgecolor="none", zorder=3))
    icon_shortlist(ax, rx + 3.05, ry + 2.45, scale=1.05)
    icon_stool(ax, rx + 3.05, ry + 2.05, scale=0.65)
    txt(ax, rx + 3.05, ry + 1.78, "SID filter", size=7.8, color=INK, weight="bold", ha="center")
    txt(ax, rx + 3.05, ry + 1.58, r"top-$L{=}20$", size=6.5, color=MUTED, ha="center")

    # HM rank with LLM icon (LLM dominant)
    rounded(ax, rx + 4.2, ry + 1.55, 1.35, 1.35, fc=WHITE, ec=TEAL_DEEP, lw=1.15, r=0.1, z=2)
    icon_llm(ax, rx + 4.87, ry + 2.48, scale=1.05)
    txt(ax, rx + 4.87, ry + 2.05, "LLM", size=7.0, color=TEAL_DEEP, weight="bold", ha="center")
    icon_history(ax, rx + 4.55, ry + 1.78, scale=0.5)
    icon_metadata(ax, rx + 5.2, ry + 1.78, scale=0.5)
    txt(ax, rx + 4.87, ry + 1.58, "HM · text only", size=6.4, color=MUTED, ha="center")

    hairline(ax, rx + 1.87, ry + 2.2, rx + 2.1, ry + 2.2)
    hairline(ax, rx + 3.95, ry + 2.2, rx + 4.2, ry + 2.2)

    # Finding strip with coverage icon
    rounded(ax, rx + 0.22, ry + 0.2, 5.35, 1.15, fc=WHITE, ec=PANEL_EDGE, lw=0.95, r=0.1, z=2)
    icon_coverage(ax, rx + 0.7, ry + 0.85, scale=0.9)
    txt(ax, rx + 1.15, ry + 1.05, "Primary finding", size=7.0, color=MUTED, weight="medium")
    txt(
        ax,
        rx + 1.15,
        ry + 0.72,
        "SID beats random coverage  (~+0.06 Recall@20)",
        size=8.0,
        color=INK,
        weight="bold",
    )
    txt(
        ax,
        rx + 1.15,
        ry + 0.42,
        "Filter-then-text  ·  do not force-merge $S_{\\mathrm{tool}}$ after shortlisting",
        size=6.8,
        color=TEAL_DEEP,
    )

    # vs
    txt(ax, 6.15, 2.1, "vs", size=9, color=FAINT, ha="center", weight="bold")

    # ===== Footer (high visibility) =====
    rounded(ax, 0.15, 0.06, 12.1, 0.58, fc="#D1FAE5", ec=TEAL_DEEP, lw=1.35, r=0.1, z=2)
    txt(
        ax,
        0.4,
        0.35,
        "Role A — Do SID scores help after candidates are already visible?",
        size=8.4,
        color=INK,
        weight="bold",
    )
    txt(
        ax,
        6.5,
        0.35,
        "Role B — Does the same scorer help choose what enters context?",
        size=8.4,
        color=INK,
        weight="bold",
    )

    fig.tight_layout(pad=0.12)
    for ext in ("pdf", "png"):
        out = OUT / f"deployment_roles.{ext}"
        fig.savefig(out, dpi=350, bbox_inches="tight", facecolor=WHITE, pad_inches=0.06)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
