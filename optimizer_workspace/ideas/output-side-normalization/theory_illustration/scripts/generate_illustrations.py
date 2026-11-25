#!/usr/bin/env python3
"""Generate intuitive pedagogical vector illustrations for the Theory Illustration Primer.

Produces 5 Distill.pub / frontier-lab visual tutorial figures (6.5 inches wide, 1:1 with
LaTeX textwidth) combining 2D geometric diagrams, exact 2x2 worked matrices, and verified
130M/520M Transformer checkpoint measurements from `paper_v3/tables/rotation_mechanism_metrics.json`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

IDEA_ROOT = Path("/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization")
METRICS_JSON = IDEA_ROOT / "paper_v3" / "tables" / "rotation_mechanism_metrics.json"
OUT_DIR = IDEA_ROOT / "theory_illustration" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_TEX_FONTS = [
    "/opt/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre/texgyreheros-regular.otf",
    "/opt/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre/texgyreheros-bold.otf",
    "/opt/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre/texgyreheros-italic.otf",
    "/opt/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre/texgyreheros-bolditalic.otf",
    "/opt/texlive/2026/texmf-dist/fonts/opentype/public/lm/lmmono10-regular.otf",
    "/opt/texlive/2026/texmf-dist/fonts/opentype/public/lm/lmmonolt10-bold.otf",
]
for _fp in _TEX_FONTS:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)

COL = {
    "ink": "#18242C",
    "sub": "#334155",
    "muted": "#64748B",
    "spine": "#B0BCC7",
    "grid": "#ECEFF3",
    "panel": "#F6F8FB",
    "panel_edge": "#DCE3EC",
    "wt": "#1B4F9C",
    "wt_mid": "#3E70BA",
    "wt_soft": "#97B7E5",
    "wt_tint": "#EEF4FC",
    "od": "#0E7C7B",
    "od_soft": "#7CC0BE",
    "od_tint": "#EBF7F6",
    "muont": "#334155",
    "muon": "#7B8794",
    "ochre": "#C87D20",
    "ochre_tint": "#FAF3E6",
    "bad": "#B83B2A",
    "bad_soft": "#E08A7E",
    "bad_tint": "#FDF0EE",
    "green": "#1F7A54",
    "green_tint": "#ECF7F1",
}

HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
HALO_SM = [pe.withStroke(linewidth=1.6, foreground="white")]

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "font.monospace": ["Latin Modern Mono", "DejaVu Sans Mono"],
    "mathtext.fontset": "cm",
    "font.size": 7.6,
    "axes.titlesize": 8.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 7.5,
    "axes.labelcolor": COL["sub"],
    "axes.edgecolor": COL["spine"],
    "axes.linewidth": 0.55,
    "xtick.labelsize": 6.9,
    "ytick.labelsize": 6.9,
    "xtick.color": COL["muted"],
    "ytick.color": COL["muted"],
    "figure.dpi": 300,
})


def style_clean_2d(ax, lim: float = 2.4):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="#CBD5E1", lw=0.7, zorder=1)
    ax.axvline(0, color="#CBD5E1", lw=0.7, zorder=1)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def style_bar_ax(ax, grid_axis: str = "y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL["spine"])
    ax.spines["bottom"].set_color(COL["spine"])
    ax.grid(False)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=COL["grid"], lw=0.55, zorder=0)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=COL["grid"], lw=0.55, zorder=0)
    ax.set_axisbelow(True)


def draw_vec(ax, x, y, color, lw=2.0, ls="-", zorder=5, mutation=11, alpha=1.0):
    arr = FancyArrowPatch(
        (0, 0), (x, y),
        arrowstyle="-|>",
        mutation_scale=mutation,
        lw=lw,
        linestyle=ls,
        color=color,
        alpha=alpha,
        zorder=zorder,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# FIGURE 1: 5-Step Plain-English Logic Roadmap
# ---------------------------------------------------------------------------
def make_fig1_roadmap():
    fig, ax = plt.subplots(figsize=(6.5, 2.55))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.85)
    ax.axis("off")

    cards = [
        {
            "step": "STEP 1",
            "title": "Muon = Pure Axes",
            "body": "Muon resets all singular\nstretches to 1. Thus the\nupdate IS its choice of\nperpendicular axes (U, V).",
            "badge": "Axes = Everything",
            "bg": COL["panel"],
            "edge": COL["spine"],
            "accent": COL["muont"],
        },
        {
            "step": "STEP 2",
            "title": "Aligned vs. Tilted Ovals",
            "body": "MLP gate/up errors align\nwith neuron axes (80-91%),\nwhile residual o/down are\n>99% tilted across axes.",
            "badge": "Diag blind at 45°",
            "bg": COL["od_tint"],
            "edge": COL["od_soft"],
            "accent": COL["od"],
        },
        {
            "step": "STEP 3",
            "title": "Loud-Spike Hijacking",
            "body": "Loud output channels pull\nall vectors toward them &\nhijack 30-68% of Muon's\n90° orthogonality budget.",
            "badge": "15°-26° off course",
            "bg": COL["bad_tint"],
            "edge": COL["bad_soft"],
            "accent": COL["bad"],
        },
        {
            "step": "STEP 4",
            "title": "Inner -1/2 + Outer -1/8",
            "body": "Inner -1/2 restores fair\nvoting (hijack -> 3%);\nouter -1/8 shrinks steps\ninto steep canyon walls.",
            "badge": "+10% to +100% descent",
            "bg": COL["wt_tint"],
            "edge": COL["wt_soft"],
            "accent": COL["wt"],
        },
        {
            "step": "STEP 5",
            "title": "Calmer Momentum",
            "body": "Less wall-bouncing slows\nerror-ellipse spinning by\n11-40%, so 20-step\nmomentum adds up cleanly.",
            "badge": "Lower held-out loss",
            "bg": COL["green_tint"],
            "edge": "#86C7A8",
            "accent": COL["green"],
        },
    ]

    w = 1.82
    gap = 0.18
    x0 = 0.09
    y0 = 0.22
    h = 3.32

    for i, c in enumerate(cards):
        x = x0 + i * (w + gap)
        box = FancyBboxPatch(
            (x, y0), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.14",
            facecolor=c["bg"],
            edgecolor=c["edge"],
            linewidth=0.95,
            zorder=2,
        )
        ax.add_patch(box)

        # Header pill
        pill = FancyBboxPatch(
            (x + 0.10, y0 + h - 0.44), 0.70, 0.28,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=c["accent"],
            edgecolor="none",
            zorder=3,
        )
        ax.add_patch(pill)
        ax.text(
            x + 0.45, y0 + h - 0.30, c["step"],
            ha="center", va="center", fontsize=6.0, fontweight="bold", color="white", zorder=4
        )

        ax.text(
            x + 0.10, y0 + h - 0.72, c["title"],
            ha="left", va="top", fontsize=7.0, fontweight="bold", color=COL["ink"], zorder=4
        )
        ax.text(
            x + 0.10, y0 + h - 1.10, c["body"],
            ha="left", va="top", fontsize=6.1, color=COL["sub"], linespacing=1.28, zorder=4
        )

        # Bottom badge
        bb = FancyBboxPatch(
            (x + 0.09, y0 + 0.13), w - 0.18, 0.36,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="white",
            edgecolor=c["edge"],
            linewidth=0.7,
            zorder=3,
        )
        ax.add_patch(bb)
        ax.text(
            x + w / 2, y0 + 0.31, c["badge"],
            ha="center", va="center", fontsize=6.1, fontweight="bold", color=c["accent"], zorder=4
        )

        if i < len(cards) - 1:
            arr = FancyArrowPatch(
                (x + w + 0.02, y0 + h / 2),
                (x + w + gap - 0.02, y0 + h / 2),
                arrowstyle="-|>",
                mutation_scale=9,
                lw=1.2,
                color=COL["muted"],
                zorder=5,
            )
            ax.add_patch(arr)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.98, bottom=0.02)
    fig.savefig(OUT_DIR / "fig1_logic_roadmap.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 2: Visual Primer — Error Ellipses, SVD, and Muon's Polar Step
# ---------------------------------------------------------------------------
def make_fig2_primer():
    fig = plt.figure(figsize=(6.5, 2.38))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.96, 1.34], wspace=0.18)

    # Panel (a): Cloud of backward error vectors forming an ellipse B = E[delta delta^T]
    ax0 = fig.add_subplot(gs[0, 0])
    style_clean_2d(ax0, lim=2.45)
    ax0.set_title("(a)  Backward Errors Form an Oval", loc="left", fontsize=7.6, fontweight="bold", color=COL["ink"], pad=6)

    rng = np.random.default_rng(7)
    pts = rng.normal(size=(2, 160))
    theta = np.deg2rad(35)
    Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cloud = Q @ np.diag([1.65, 0.42]) @ pts
    ax0.scatter(cloud[0], cloud[1], s=5.0, color=COL["wt_soft"], alpha=0.55, edgecolors="none", zorder=2)

    ell = Ellipse((0, 0), width=2 * 1.85, height=2 * 0.52, angle=35,
                  facecolor=COL["wt_tint"], edgecolor=COL["wt"], lw=1.25, ls="--", alpha=0.55, zorder=1)
    ax0.add_patch(ell)

    q1 = Q[:, 0] * 1.85
    q2 = Q[:, 1] * 0.65
    draw_vec(ax0, q1[0], q1[1], COL["bad"], lw=2.0, mutation=10)
    draw_vec(ax0, q2[0], q2[1], COL["od"], lw=2.0, mutation=10)

    ax0.text(0.88, -0.58, r"Loud axis $q_1$" + "\n" + r"(width $\sqrt{\beta_1}$)",
             fontsize=6.3, fontweight="bold", color=COL["bad"], ha="center", path_effects=HALO_SM)
    ax0.text(-1.02, 0.86, r"Quiet axis $q_2$ ($\sqrt{\beta_2}$)",
             fontsize=6.3, fontweight="bold", color=COL["od"], ha="center", path_effects=HALO_SM)
    ax0.text(0, -2.25, r"Eigenvectors of $B = \mathbb{E}[\delta\delta^\top]$ = oval axes",
             ha="center", fontsize=6.3, color=COL["sub"])

    # Panel (b): Raw Gradient M vs Muon Polar(M)
    ax1 = fig.add_subplot(gs[0, 1])
    style_clean_2d(ax1, lim=2.45)
    ax1.set_title("(b)  What Muon's Polar Step Does", loc="left", fontsize=7.6, fontweight="bold", color=COL["ink"], pad=6)

    circ = Circle((0, 0), radius=1.28, facecolor=COL["wt_tint"], edgecolor=COL["wt"], lw=1.1, ls=":", alpha=0.65, zorder=1)
    ax1.add_patch(circ)
    raw_ell = Ellipse((0, 0), width=2 * 2.05, height=2 * 0.30, angle=30,
                      facecolor="none", edgecolor=COL["bad_soft"], lw=1.1, ls="--", zorder=2)
    ax1.add_patch(raw_ell)

    ang = np.deg2rad(30)
    u1 = np.array([np.cos(ang), np.sin(ang)])
    u2 = np.array([-np.sin(ang), np.cos(ang)])

    # Raw M vectors
    draw_vec(ax1, *(u1 * 2.05), COL["bad"], lw=1.6, ls="--", mutation=9, alpha=0.85)
    draw_vec(ax1, *(u2 * 0.30), COL["bad"], lw=1.6, ls="--", mutation=8, alpha=0.85)

    # Muon polar vectors (both length 1.28 = "1 unit")
    draw_vec(ax1, *(u1 * 1.28), COL["wt"], lw=2.2, mutation=11)
    draw_vec(ax1, *(u2 * 1.28), COL["wt"], lw=2.2, mutation=11)

    ax1.text(1.05, 1.48, r"Raw $\sigma_1 \gg \sigma_2$" + "\n(starves quiet mode)",
             fontsize=6.1, color=COL["bad"], ha="center", path_effects=HALO_SM)
    ax1.text(-1.18, 1.15, r"$\mathrm{polar}(M) = UV^\top$" + "\n(all lengths = 1!)",
             fontsize=6.2, fontweight="bold", color=COL["wt"], ha="center", path_effects=HALO_SM)
    ax1.text(0, -2.25, r"No stretches left $\Rightarrow$ axes $(U,V)$ ARE the step",
             ha="center", fontsize=6.3, color=COL["sub"])

    # Panel (c): Two sides of a linear layer y = W x
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_xlim(0, 10.4)
    ax2.set_ylim(0, 6.2)
    ax2.axis("off")
    ax2.set_title("(c)  Two Sides of Every Layer  y = W x", loc="left", fontsize=7.6, fontweight="bold", color=COL["ink"], pad=6)

    # Input box (right side of matrix G = delta x^T)
    b_in = FancyBboxPatch((0.15, 3.25), 2.95, 2.25, boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=COL["green_tint"], edgecolor="#86C7A8", lw=0.9)
    ax2.add_patch(b_in)
    ax2.text(1.62, 5.05, "Input x (Cols)", ha="center", fontsize=6.7, fontweight="bold", color=COL["green"])
    ax2.text(1.62, 4.08, "$C = \\mathbb{E}[xx^\\top]$\nFixed in MuonT\n(slow drift)",
             ha="center", va="center", fontsize=5.85, color=COL["sub"])

    # Weight box W
    b_w = FancyBboxPatch((3.80, 3.25), 2.80, 2.25, boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor=COL["panel"], edgecolor=COL["spine"], lw=1.0)
    ax2.add_patch(b_w)
    ax2.text(5.20, 5.05, "Weight W", ha="center", fontsize=6.8, fontweight="bold", color=COL["ink"])
    ax2.text(5.20, 4.08, "Gradient:\n$G = \\sum \\delta\\, x^\\top$\n(Rows $\\times$ Cols)",
             ha="center", va="center", fontsize=5.95, color=COL["ink"])

    # Output box (left side of matrix G)
    b_out = FancyBboxPatch((7.30, 3.25), 2.95, 2.25, boxstyle="round,pad=0.05,rounding_size=0.2",
                           facecolor=COL["wt_tint"], edgecolor=COL["wt_soft"], lw=1.0)
    ax2.add_patch(b_out)
    ax2.text(8.78, 5.05, r"Error $\delta$ (Rows)", ha="center", fontsize=6.7, fontweight="bold", color=COL["wt"])
    ax2.text(8.78, 4.08, "$B = \\mathbb{E}[\\delta\\delta^\\top]$\nUNFIXED in MuonT\n(fast-spinning!)",
             ha="center", va="center", fontsize=5.85, color=COL["wt"])

    # Forward & backward arrows
    ax2.add_patch(FancyArrowPatch((3.15, 4.45), (3.75, 4.45), arrowstyle="-|>", mutation_scale=8, lw=1.15, color=COL["green"]))
    ax2.text(3.45, 4.72, "fwd", ha="center", fontsize=5.5, fontweight="bold", color=COL["green"])
    ax2.add_patch(FancyArrowPatch((7.25, 4.45), (6.65, 4.45), arrowstyle="-|>", mutation_scale=8, lw=1.15, color=COL["wt"]))
    ax2.text(6.95, 4.72, "bwd", ha="center", fontsize=5.5, fontweight="bold", color=COL["wt"])

    # Bottom summary banner
    b_bot = FancyBboxPatch((0.15, 0.35), 10.1, 2.42, boxstyle="round,pad=0.05,rounding_size=0.2",
                           facecolor=COL["ochre_tint"], edgecolor="#E6C79C", lw=0.85)
    ax2.add_patch(b_bot)
    ax2.text(5.20, 2.22, "WT-MLP-Lag Update Formula in Plain Sight:",
             ha="center", fontsize=6.6, fontweight="bold", color=COL["ink"])
    ax2.text(5.20, 1.15,
             r"$D_t = \hat{L}(B^{-1/8})\,\mathrm{polar}\!\left(L(B^{-1/2})\,U_t\,R(C^{-1/2})\right)\hat{R}(C^{-1/8})$",
             ha="center", fontsize=6.35, color=COL["wt"])

    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.08)
    fig.savefig(OUT_DIR / "fig2_svd_muon_primer.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 3: Coordinate vs. Rotational Anisotropy & Residual Mixing-Bowl Law
# ---------------------------------------------------------------------------
def make_fig3_tilted_vs_aligned():
    with open(METRICS_JSON) as f:
        data = json.load(f)

    fig = plt.figure(figsize=(6.5, 2.62))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.98, 0.98, 1.25], wspace=0.30)

    # Panel (a): Axis-aligned ellipse (gate, up)
    ax0 = fig.add_subplot(gs[0, 0])
    style_clean_2d(ax0, lim=2.25)
    ax0.set_title("(a) Axis-Aligned Oval (gate, up)", loc="left", fontsize=7.5, fontweight="bold", color=COL["od"], pad=6)

    ell_a = Ellipse((0, 0), width=2 * 1.42, height=2 * 0.27, angle=0,
                    facecolor=COL["od_tint"], edgecolor=COL["od"], lw=1.6, zorder=3)
    ax0.add_patch(ell_a)

    # Shadows on x and y axes
    ax0.plot([-1.42, 1.42], [-1.72, -1.72], color=COL["od"], lw=3.0, solid_capstyle="round")
    ax0.text(0, -2.04, r"Neuron 1: $B_{11} = 1.96$ (wide!)", ha="center", fontsize=6.2, fontweight="bold", color=COL["od"])

    ax0.plot([-1.85, -1.85], [-0.27, 0.27], color=COL["od"], lw=3.0, solid_capstyle="round")
    ax0.text(-1.72, 0.62, r"$B_{22}=0.04$" + "\n(narrow!)", ha="left", fontsize=6.0, fontweight="bold", color=COL["od"], path_effects=HALO_SM)

    # Dashed projection lines
    for s in (-1, 1):
        ax0.plot([s * 1.42, s * 1.42], [0, -1.72], color=COL["od_soft"], lw=0.7, ls=":")
        ax0.plot([0, -1.85], [s * 0.27, s * 0.27], color=COL["od_soft"], lw=0.7, ls=":")

    ax0.text(0.15, 1.62, "B = [1.96, 0.00]\n      [0.00, 0.04]\nDiag sees 100%!",
             ha="center", va="center", fontsize=6.1, color=COL["ink"],
             bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=COL["od_soft"], lw=0.7))

    # Panel (b): 45-degree tilted ellipse (o, down)
    ax1 = fig.add_subplot(gs[0, 1])
    style_clean_2d(ax1, lim=2.25)
    ax1.set_title("(b) 45°-Tilted Oval (o, down)", loc="left", fontsize=7.5, fontweight="bold", color=COL["wt"], pad=6)

    ell_b = Ellipse((0, 0), width=2 * 1.42, height=2 * 0.27, angle=45,
                    facecolor=COL["wt_tint"], edgecolor=COL["wt"], lw=1.6, zorder=3)
    ax1.add_patch(ell_b)

    hw = np.sqrt((1.42**2 + 0.27**2) / 2.0)
    ax1.plot([-hw, hw], [-1.72, -1.72], color=COL["bad"], lw=3.0, solid_capstyle="round")
    ax1.text(0, -2.04, r"Neuron 1 shadow: $B_{11} = 1.00$", ha="center", fontsize=6.2, fontweight="bold", color=COL["bad"])

    ax1.plot([-1.85, -1.85], [-hw, hw], color=COL["bad"], lw=3.0, solid_capstyle="round")
    ax1.text(-2.12, 1.50, r"Neuron 2: $B_{22} = 1.00$" + "\n(SAME width!)",
             ha="left", fontsize=5.95, fontweight="bold", color=COL["bad"], path_effects=HALO_SM)

    for s in (-1, 1):
        ax1.plot([s * hw, s * hw], [-s * 0.96, -1.72], color=COL["bad_soft"], lw=0.7, ls=":")
        ax1.plot([-s * 0.96, -1.85], [s * hw, s * hw], color=COL["bad_soft"], lw=0.7, ls=":")

    ax1.text(1.02, -0.78, "B = [1.00, 0.96]\n      [0.96, 1.00]\nDiag 100% BLIND!",
             ha="center", va="center", fontsize=5.85, color=COL["ink"], zorder=6,
             bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=COL["bad_soft"], lw=0.7))

    # Panel (c): Real Transformer DiagAniso shrinking from 130M (d=512) to 520M (d=1024)
    ax2 = fig.add_subplot(gs[0, 2])
    style_bar_ax(ax2, grid_axis="x")
    ax2.set_title("(c) Real Layers: Diagonal Share (%)", loc="left", fontsize=7.5, fontweight="bold", color=COL["ink"], pad=6)

    groups = ["out:gate", "out:up", "out:q", "out:k", "out:v", "out:o", "out:down"]
    keys = ["out_gate", "out_up", "out_q", "out_k", "out_v", "out_o", "out_down"]
    v130 = [data["exp4_ipr_130m_by_kind"][k]["diag_aniso_pct"]["median"] for k in keys]
    v520 = [
        np.nan,
        np.nan,
        data["exp4_ipr_520m_by_kind"]["out_q"]["diag_aniso_pct"]["median"],
        data["exp4_ipr_520m_by_kind"]["out_k"]["diag_aniso_pct"]["median"],
        data["exp4_ipr_520m_by_kind"]["out_v"]["diag_aniso_pct"]["median"],
        data["exp4_ipr_520m_by_kind"]["out_o"]["diag_aniso_pct"]["median"],
        data["exp4_ipr_520m_by_kind"]["out_down"]["diag_aniso_pct"]["median"],
    ]

    y = np.arange(len(groups))[::-1]
    h = 0.34
    for idx, yi in enumerate(y):
        if np.isnan(v520[idx]):
            ax2.barh(yi, v130[idx], height=0.52, color=COL["od"], alpha=0.9, zorder=3)
            ax2.text(v130[idx] + 2.0, yi, f"{v130[idx]:.1f}% (Diag OK)", va="center", fontsize=5.95, fontweight="bold", color=COL["od"])
        else:
            c130 = COL["wt_soft"] if idx >= 5 else "#94A3B8"
            c520 = COL["wt"] if idx >= 5 else COL["muont"]
            ax2.barh(yi + h / 2, v130[idx], height=h, color=c130, label="130M (d=512)" if idx == 2 else None, zorder=3)
            ax2.barh(yi - h / 2, v520[idx], height=h, color=c520, label="520M (d=1024)" if idx == 2 else None, zorder=3)
            if idx == 4:  # out:v
                ax2.text(v130[idx] + 2.0, yi + 0.04, r"12.2% $\to$ 1.0%!", va="center", fontsize=5.95, fontweight="bold", color=COL["bad"])
            elif idx >= 5:
                ax2.text(v130[idx] + 2.0, yi, f"{v130[idx]:.1f}% " + r"$\to$" + f" {v520[idx]:.1f}%", va="center", fontsize=5.95, fontweight="bold", color=COL["wt"])
            else:
                ax2.text(max(v130[idx], v520[idx]) + 2.0, yi, f"{v520[idx]:.1f}%", va="center", fontsize=5.9, color=COL["sub"])

    ax2.set_yticks(y)
    ax2.set_yticklabels(groups, fontsize=6.5, fontfamily="monospace")
    ax2.set_xlim(0, 134)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("Diagonal Anisotropy Share (%)", fontsize=6.8)
    ax2.legend(loc="lower right", fontsize=5.8, handlelength=1.0, borderpad=0.2)

    fig.subplots_adjust(left=0.01, right=0.985, top=0.87, bottom=0.15)
    fig.savefig(OUT_DIR / "fig3_tilted_vs_aligned.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 4: The 2x2 "Loud-Channel Hijacking" Paradox Inside Polar
# ---------------------------------------------------------------------------
def make_fig4_hijacking():
    with open(METRICS_JSON) as f:
        data = json.load(f)

    fig = plt.figure(figsize=(6.5, 2.65))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.95, 0.95, 1.02, 1.28], wspace=0.22)

    # Panel (a): True clean features M_true at 45 deg and 135 deg
    ax0 = fig.add_subplot(gs[0, 0])
    style_clean_2d(ax0, lim=2.25)
    ax0.set_title(r"(a) True Signal $M_{\mathrm{true}}$", loc="left", fontsize=7.6, fontweight="bold", color=COL["green"], pad=6)

    circ0 = Circle((0, 0), radius=1.0, facecolor="none", edgecolor="#CBD5E1", lw=0.8, ls=":")
    ax0.add_patch(circ0)
    c1_true = np.array([2.0, 2.0]) / np.sqrt(2)
    c2_true = np.array([-0.5, 0.5]) / np.sqrt(2)
    draw_vec(ax0, *c1_true, COL["green"], lw=2.2, mutation=10)
    draw_vec(ax0, *(c2_true * 2.2), COL["od"], lw=2.0, mutation=9)  # visually scaled slightly with exact label

    ax0.text(0.92, 1.65, r"Feature 1" + "\n" + r"($45^\circ,\ \sigma_1\!=\!2.0$)",
             ha="center", fontsize=6.2, fontweight="bold", color=COL["green"], path_effects=HALO_SM)
    ax0.text(-1.12, 0.95, r"Feature 2" + "\n" + r"($135^\circ,\ \sigma_2\!=\!0.5$)",
             ha="center", fontsize=6.1, fontweight="bold", color=COL["od"], path_effects=HALO_SM)
    ax0.text(0, -1.98, "90° apart (clean axes)", ha="center", fontsize=6.3, color=COL["sub"])

    # Panel (b): Multiply row 1 by 10x error noise -> Z_0
    ax1 = fig.add_subplot(gs[0, 1])
    style_clean_2d(ax1, lim=2.25)
    ax1.set_title(r"(b) 10x Loud Axis 1 ($Z_0$)", loc="left", fontsize=7.6, fontweight="bold", color=COL["bad"], pad=6)

    ax1.axhspan(-0.28, 0.28, color=COL["bad_tint"], alpha=0.85, zorder=0)
    # Display normalized-to-fit vectors of Z_0 = 1/sqrt(2) [[20, -5], [2, 0.5]]
    z1 = np.array([20.0, 2.0]) / np.sqrt(2)
    z2 = np.array([-5.0, 0.5]) / np.sqrt(2)
    scale_vis = 2.05 / np.linalg.norm(z1)
    draw_vec(ax1, *(z1 * scale_vis), COL["bad"], lw=2.3, mutation=10)
    draw_vec(ax1, *(z2 * scale_vis * 2.1), COL["ochre"], lw=2.0, mutation=9)

    ax1.text(0.82, -0.98, "Col 1 (len 14.2)\npulled to 5.7°!", ha="center", fontsize=5.95, fontweight="bold", color=COL["bad"], path_effects=HALO_SM)
    ax1.text(-0.92, 0.78, "Col 2 (len 3.6)\npulled to 174.3°!", ha="center", fontsize=5.95, fontweight="bold", color=COL["ochre"], path_effects=HALO_SM)
    ax1.text(0, -1.98, "Both squashed onto loud x-axis!", ha="center", fontsize=6.1, color=COL["bad"])

    # Panel (c): Polar(Z_0) vs Polar(L Z_0)
    ax2 = fig.add_subplot(gs[0, 2])
    style_clean_2d(ax2, lim=2.25)
    ax2.set_title("(c) Polar Vote: 26.2° Error!", loc="left", fontsize=7.6, fontweight="bold", color=COL["wt"], pad=6)

    R_rad = 1.68
    circ2 = Circle((0, 0), radius=R_rad, facecolor=COL["wt_tint"], edgecolor=COL["spine"], lw=0.85, ls="--", alpha=0.5, zorder=1)
    ax2.add_patch(circ2)

    phi_true = np.deg2rad(45.0)
    phi_bad = np.deg2rad(18.85)

    # Ideal / WT axes (45 deg, 135 deg)
    draw_vec(ax2, R_rad * np.cos(phi_true), R_rad * np.sin(phi_true), COL["wt"], lw=2.2, mutation=10)
    draw_vec(ax2, -R_rad * np.sin(phi_true), R_rad * np.cos(phi_true), COL["wt"], lw=2.2, mutation=10)

    # Hijacked MuonT axes (18.85 deg, 108.85 deg)
    draw_vec(ax2, R_rad * np.cos(phi_bad), R_rad * np.sin(phi_bad), COL["bad"], lw=1.9, ls="--", mutation=9)
    draw_vec(ax2, -R_rad * np.sin(phi_bad), R_rad * np.cos(phi_bad), COL["bad"], lw=1.9, ls="--", mutation=9)

    arc1 = Arc((0, 0), 2 * 1.15, 2 * 1.15, angle=0, theta1=18.85, theta2=45.0, color=COL["bad"], lw=1.4)
    ax2.add_patch(arc1)
    ax2.text(1.48, 0.92, "26.2°\noff!", fontsize=6.2, fontweight="bold", color=COL["bad"], ha="center", path_effects=HALO_SM)
    ax2.text(-1.10, 1.62, r"$\mathrm{polar}(L Z_0)$" + "\n= exact 45°", fontsize=6.1, fontweight="bold", color=COL["wt"], ha="center", path_effects=HALO_SM)
    ax2.text(0, -1.98, r"$\tan\phi_0 = \frac{7}{20.5} \Rightarrow 18.85^\circ$", ha="center", fontsize=6.4, color=COL["ink"])

    # Panel (d): Real 130M Checkpoint De-Hijacking
    ax3 = fig.add_subplot(gs[0, 3])
    style_bar_ax(ax3, grid_axis="y")
    ax3.set_title("(d) Real 130M Hijacking (%)", loc="left", fontsize=7.5, fontweight="bold", color=COL["ink"], pad=6)

    projs = ["q", "down", "k", "o", "v"]
    hj_before = [data["exp123_130m_by_kind"][p]["hijack_Z0_top16"]["mean"] * 100 for p in projs]
    hj_after = [data["exp123_130m_by_kind"][p]["hijack_Zwt_top16"]["mean"] * 100 for p in projs]
    bot80_win = [data["exp123_130m_by_kind"][p]["procrustes_pct_bot80_improved"]["mean"] for p in projs]

    x = np.arange(len(projs))
    bw = 0.35
    ax3.bar(x - bw / 2, hj_before, width=bw, color=COL["bad_soft"], edgecolor=COL["bad"], lw=0.5, label=r"MuonT ($Z_0$)", zorder=3)
    ax3.bar(x + bw / 2, hj_after, width=bw, color=COL["wt"], label=r"WT ($L = B^{-1/2}$)", zorder=3)
    ax3.axhline(3.125, color=COL["green"], lw=0.95, ls="--", zorder=4)
    ax3.annotate(
        "Fair floor (3.1%)",
        xy=(3.55, 3.3), xytext=(3.42, 43.0),
        ha="center", fontsize=5.7, fontweight="bold", color=COL["green"],
        arrowprops=dict(arrowstyle="-|>", color=COL["green"], lw=0.8, mutation_scale=7),
        path_effects=HALO_SM, zorder=6,
    )

    for i, (b0, b1, win) in enumerate(zip(hj_before, hj_after, bot80_win)):
        ax3.text(i - bw / 2, b0 + 1.8, f"{b0:.0f}%", ha="center", fontsize=5.8, fontweight="bold", color=COL["bad"])
        ax3.text(i + bw / 2 + 0.03, b1 + 2.0, f"{b1:.1f}%", ha="center", fontsize=5.5, fontweight="bold", color=COL["wt"], path_effects=HALO_SM)

    ax3.set_xticks(x)
    ax3.set_xticklabels(projs, fontfamily="monospace", fontsize=6.9)
    ax3.set_ylim(0, 82)
    ax3.set_ylabel("Top-16 Error Hijacking (%)", fontsize=6.8)
    ax3.legend(loc="upper right", fontsize=5.7, handlelength=1.0, borderpad=0.2)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.14)
    fig.savefig(OUT_DIR / "fig4_hijacking_2x2.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 5: Tangent vs Normal Canyon Shaping, 520M Descent Gain, & Calm Drift
# ---------------------------------------------------------------------------
def make_fig5_canyon_payoff():
    with open(METRICS_JSON) as f:
        data = json.load(f)

    fig = plt.figure(figsize=(6.5, 2.65))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.05, 1.05], wspace=0.27)

    # Panel (a): 90-degree Orthogonal Roles — Tangent Inner L vs Radial Outer L_hat
    ax0 = fig.add_subplot(gs[0, 0])
    style_clean_2d(ax0, lim=2.40)
    ax0.set_title("(a) Two 90°-Orthogonal Jobs", loc="left", fontsize=7.6, fontweight="bold", color=COL["ink"], pad=6)

    # Steep canyon contours (y-axis is 40x steeper canyon wall, x-axis is gentle valley floor)
    xx, yy = np.meshgrid(np.linspace(-2.35, 2.35, 160), np.linspace(-2.35, 2.35, 160))
    zz = 0.22 * xx**2 + 2.4 * yy**2
    ax0.contour(xx, yy, zz, levels=[0.4, 1.1, 2.2, 3.8, 6.0], colors=["#E2E8F0"], linewidths=0.75, zorder=0)
    ax0.axhspan(1.48, 2.40, color=COL["bad_tint"], alpha=0.55, zorder=0)
    ax0.axhspan(-2.40, -1.48, color=COL["bad_tint"], alpha=0.55, zorder=0)
    ax0.text(0, 2.02, r"Steep Canyon Wall (large $\beta_1$)", ha="center", fontsize=5.9, fontweight="bold", color=COL["bad"])

    R_u = 1.32
    unit_c = Circle((0, 0), radius=R_u, facecolor="none", edgecolor=COL["muont"], lw=1.0, ls=":", zorder=2)
    ax0.add_patch(unit_c)

    a0 = np.deg2rad(66)
    a1 = np.deg2rad(28)
    p0 = R_u * np.array([np.cos(a0), np.sin(a0)])
    p1 = R_u * np.array([np.cos(a1), np.sin(a1)])
    p2 = np.array([p1[0] * 1.22, p1[1] * 0.62])

    ax0.scatter([p0[0]], [p0[1]], s=24, color=COL["bad"], zorder=6)
    ax0.scatter([p1[0]], [p1[1]], s=24, color=COL["wt_mid"], zorder=6)
    ax0.scatter([p2[0]], [p2[1]], s=30, color=COL["wt"], zorder=7)

    arc_tan = FancyArrowPatch(
        p0, p1,
        connectionstyle="arc3,rad=-0.24",
        arrowstyle="-|>",
        mutation_scale=9,
        lw=1.8,
        color=COL["wt_mid"],
        zorder=6,
    )
    ax0.add_patch(arc_tan)

    arr_norm = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        mutation_scale=9,
        lw=1.8,
        color=COL["od"],
        zorder=6,
    )
    ax0.add_patch(arr_norm)

    ax0.text(-0.62, 1.18, "MuonT\n(hits wall)", fontsize=5.8, fontweight="bold", color=COL["bad"], ha="center", path_effects=HALO_SM)
    ax0.text(1.56, 1.18, "(1) Inner -1/2\n(tangent turn)", fontsize=5.75, fontweight="bold", color=COL["wt_mid"], ha="center", path_effects=HALO_SM)
    ax0.text(1.36, -0.68, "(2) Outer -1/8\n(canyon cushion,\n90° to (1)!)", fontsize=5.75, fontweight="bold", color=COL["od"], ha="center", path_effects=HALO_SM)
    ax0.text(0, -2.14, r"Flat Valley Floor (gentle $\beta_2$)", ha="center", fontsize=5.95, fontweight="bold", color=COL["green"])

    # Panel (b): 520M Downhill Efficiency Gain Phi_BC vs MuonT
    ax1 = fig.add_subplot(gs[0, 1])
    style_bar_ax(ax1, grid_axis="y")
    ax1.set_title(r"(b) 520M Downhill Gain ($\Phi/\Phi_{\mathrm{MT}}$)", loc="left", fontsize=7.5, fontweight="bold", color=COL["ink"], pad=6)

    p520 = ["k", "q", "v", "down", "o"]
    phi_wt = [data["exp4_cf_520m_by_kind"][p]["Phi_wt_over_muont"]["mean"] for p in p520]
    phi_od = [data["exp4_cf_520m_by_kind"][p]["Phi_od_over_muont"]["mean"] for p in p520]

    x = np.arange(len(p520))
    bw = 0.36
    ax1.axhline(1.0, color=COL["muont"], lw=0.9, ls="--", zorder=2)
    ax1.bar(x - bw / 2, phi_od, width=bw, color=COL["od_soft"], edgecolor=COL["od"], lw=0.5, label="OD-All (Diag only)", zorder=3)
    ax1.bar(x + bw / 2, phi_wt, width=bw, color=COL["wt"], label="WT-MLP-Lag (Dense)", zorder=3)

    for i, (vo, vw) in enumerate(zip(phi_od, phi_wt)):
        ax1.text(i + bw / 2 + 0.03, vw + 0.04, f"{vw:.2f}x", ha="center", fontsize=5.8, fontweight="bold", color=COL["wt"])
        if i >= 2:
            ax1.text(i - bw / 2 - 0.05, vo + 0.035, f"{vo:.2f}x", ha="center", fontsize=5.35, color=COL["od"], path_effects=HALO_SM)

    ax1.set_xticks(x)
    ax1.set_xticklabels(p520, fontfamily="monospace", fontsize=6.9)
    ax1.set_ylim(0.85, 2.24)
    ax1.set_ylabel("Descent Efficiency vs. MuonT", fontsize=6.8)
    ax1.legend(loc="upper right", fontsize=5.7, handlelength=1.0, borderpad=0.2)

    # Panel (c): Calmer Subspace Drift across all 7 outputs (Table 1 of paper_v3)
    ax2 = fig.add_subplot(gs[0, 2])
    style_bar_ax(ax2, grid_axis="x")
    ax2.set_title(r"(c) Slower Ellipse Spin ($\mathrm{Drift}_{16}$)", loc="left", fontsize=7.6, fontweight="bold", color=COL["ink"], pad=6)

    # Verified Table 1 numbers from paper_v3/sections/02_diagnosis.tex
    g_names = ["up", "gate", "down", "o", "q", "k", "v"]
    d_mt = [0.215, 0.240, 0.107, 0.134, 0.140, 0.165, 0.662]
    d_wt = [0.130, 0.159, 0.072, 0.103, 0.110, 0.147, 0.586]
    pct_drop = [(wt - mt) / mt * 100 for mt, wt in zip(d_mt, d_wt)]

    y = np.arange(len(g_names))
    ax2.barh(y, [-p for p in pct_drop], height=0.56, color=COL["green"], alpha=0.88, zorder=3)
    for yi, p in zip(y, pct_drop):
        ax2.text(-p + 1.1, yi, f"{p:.0f}%", va="center", fontsize=6.1, fontweight="bold", color=COL["green"])

    ax2.set_yticks(y)
    ax2.set_yticklabels(g_names, fontfamily="monospace", fontsize=6.8)
    ax2.set_xlim(0, 50)
    ax2.set_xlabel("Reduction in 250-Step Spinning (%)", fontsize=6.8)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.15)
    fig.savefig(OUT_DIR / "fig5_canyon_payoff.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_fig1_roadmap()
    make_fig2_primer()
    make_fig3_tilted_vs_aligned()
    make_fig4_hijacking()
    make_fig5_canyon_payoff()
    print("Successfully generated all 5 pedagogical vector figures in:", OUT_DIR)
