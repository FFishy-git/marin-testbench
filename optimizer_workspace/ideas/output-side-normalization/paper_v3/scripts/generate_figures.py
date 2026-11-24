#!/usr/bin/env python3
"""Frontier-lab minimalist vector figure suite for Paper v3.

Generates all 8 publication-grade vector PDFs directly from `experiment_records/`
and verified checkpoint diagnostics (`mechanism_summary.json`, `measured_8gpu_timings.json`,
`lr_summary.json`). Sized at exact 1:1 physical LaTeX textwidth (6.5 inches),
harmonized with `paperstyle.sty` typography (TeX Gyre Heros / Helvetica + CM math)
and semantic color hierarchy.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.scale import FuncScale

IDEA_ROOT = Path("/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization")
RECORDS = IDEA_ROOT / "experiment_records"
RES_DIR = IDEA_ROOT / "paper" / "results"
OUT_DIR = IDEA_ROOT / "paper_v3" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Register TeX Live OpenType Helvetica (TeX Gyre Heros) & Latin Modern Mono
# to match pdflatex's `phv` headings/captions and `lmtt` identifiers 1:1.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Frontier-Lab Minimalist Semantic Color System (matched to main.tex)
# ---------------------------------------------------------------------------
COL = {
    "ink": "#18242C",        # Primary charcoal ink
    "sub": "#334155",        # Axis label slate
    "muted": "#64748B",      # Secondary / tick label slate
    "spine": "#B0BCC7",      # Subtle hairline axis spine
    "grid": "#ECEFF3",       # Whisper-light single-axis grid
    "panel": "#F6F8FB",      # Subtle card fill
    "panel_edge": "#DCE3EC", # Hairline card border
    # Method & structural semantics
    "wt": "#1B4F9C",         # Primary method: Deep Sapphire (colWT)
    "wt_mid": "#4A78BD",     # Secondary WT variant / paired metric
    "wt_soft": "#8FB1E0",    # Soft Sapphire bar fill
    "wt_tint": "#EEF4FC",    # Soft Sapphire background wash
    "od": "#0E7C7B",         # Diagonal geometry: Deep Mineral Teal (colOD)
    "od_soft": "#7CC0BE",    # Soft Teal bar fill
    "od_tint": "#EBF7F6",    # Soft Teal background wash
    "muont": "#1E293B",      # Primary baseline: Rich Slate
    "muon": "#7B8794",       # Unpreconditioned baseline: Cool Grey
    "dt": "#5B548A",         # Dense ceiling: Muted Indigo
    "null": "#C87D20",       # Matched-path control: Warm Ochre (colNull)
    "null_tint": "#FAF3E6",  # Gate band tint
    "bad": "#B83B2A",        # Failure / cliff / stall: Muted Vermilion
    "bad_soft": "#D97A6C",   # Soft terracotta for negative ablations
    "bad_tint": "#FDF0EE",   # Soft stall fill
    "neutral_bar": "#94A3B8",# Secondary neutral bars
    "neutral_lt": "#CBD5E1", # Dominated / background bars
    "rot_fill": "#DCE5F0",   # Subtle rotational complement in 100% stacked bar
    "green": "#1F7A54",      # Sampling-noise floor reference
}

HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
HALO_SM = [pe.withStroke(linewidth=1.6, foreground="white")]

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.monospace": ["Latin Modern Mono", "Nimbus Mono PS", "DejaVu Sans Mono"],
    "mathtext.fontset": "cm",
    "font.size": 7.5,
    "axes.titlesize": 8.1,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 7.4,
    "axes.labelcolor": COL["sub"],
    "axes.edgecolor": COL["spine"],
    "axes.linewidth": 0.55,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "xtick.color": COL["muted"],
    "ytick.color": COL["muted"],
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.major.pad": 2.4,
    "ytick.major.pad": 2.4,
    "legend.fontsize": 6.4,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def style_ax(ax, grid_axis: str = "y", hide_left: bool = False, hide_bottom: bool = False):
    """Apply clean frontier-lab minimalist spine and single-axis hairline styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    else:
        ax.spines["left"].set_color(COL["spine"])
        ax.spines["left"].set_linewidth(0.55)
    if hide_bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    else:
        ax.spines["bottom"].set_color(COL["spine"])
        ax.spines["bottom"].set_linewidth(0.55)

    ax.grid(False)
    if grid_axis in ("both", "y"):
        ax.yaxis.grid(True, color=COL["grid"], linewidth=0.5, zorder=0)
    if grid_axis in ("both", "x"):
        ax.xaxis.grid(True, color=COL["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_title(ax, tag: str, text: str, pad: float = 6.5):
    """Left-aligned frontier-lab panel title with dark ink hierarchy."""
    ax.set_title(f"{tag}  {text}", loc="left", pad=pad, fontsize=8.0, fontweight="bold", color=COL["ink"])


def clean_log_xaxis(ax, ticks: list[float], labels: list[str]):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def set_piecewise_ylim(ax, y_min: float, y_knee: float, y_max: float, compress: float = 0.22):
    """Piecewise linear scale: 1:1 resolution on [y_min, y_knee], compressed above y_knee."""
    def _fwd(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= y_knee, y, y_knee + (y - y_knee) * compress)

    def _inv(z):
        z = np.asarray(z, dtype=float)
        return np.where(z <= y_knee, z, y_knee + (z - y_knee) / compress)

    ax.set_yscale(FuncScale(ax.yaxis, (_fwd, _inv)))
    ax.set_ylim(y_min, y_max)


def load_jsonl_gz(rel_path: str) -> list[dict]:
    path = RECORDS / rel_path
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def eval_series(rows: list[dict], field: str = "selection_loss"):
    steps, toks, vals = [], [], []
    for r in rows:
        if field in r and r[field] is not None:
            steps.append(int(r["step"]))
            toks.append(float(r["tokens"]) / 1e9)
            vals.append(float(r[field]))
    return np.array(steps), np.array(toks), np.array(vals)


# ===========================================================================
# FIGURE 1: Hero 2-panel synthesis (11-moment geometry & cost, 520M Pareto)
# ===========================================================================
# Dense eigendecomposition cost shares across all 11 moments per layer (sum = 100%):
# a d_ff moment costs (4 d_m)^3 = 64 units against 1 unit per d_m moment, over
# (2*64 + 5*1) + (1*64 + 3*1) = 133 + 67 = 200 total units across both sides.
COST_TOTAL_UNITS = 2 * 64.0 + 5 * 1.0 + 1 * 64.0 + 3 * 1.0
COST_DFF, COST_DM = 64.0 / COST_TOTAL_UNITS, 1.0 / COST_TOTAL_UNITS


def build_fig1():
    mech = json.load(open(RES_DIR / "mechanism_summary.json"))
    m25 = mech["motivation_trajectories"]["muon"]["step_002500"]
    t520 = json.load(open(RES_DIR / "measured_520m_timings.json"))["final_ws32_mb2_c2"]

    fig = plt.figure(figsize=(6.5, 3.34))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.16, 0.30, 0.10, 1.10], wspace=0.11)
    ax1, axc, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])

    # --- Panel (a): where each moment's departure from identity lives ---
    outs = [
        ("out_gate", "out:gate", 2048, COST_DFF, True),
        ("out_up", "out:up", 2048, COST_DFF, True),
        ("out_q", "out:q", 512, COST_DM, False),
        ("out_k", "out:k", 512, COST_DM, False),
        ("out_v", "out:v", 512, COST_DM, False),
        ("out_down", "out:down", 512, COST_DM, False),
        ("out_o", "out:o", 512, COST_DM, False),
    ]
    ins = [
        ("in_down", "in:down", 2048, COST_DFF, False),
        ("in_o", "in:o", 512, COST_DM, False),
        ("in_attn", "in:attn", 512, COST_DM, False),
        ("in_mlp", "in:mlp", 512, COST_DM, False),
    ]
    rows = outs + ins
    y = np.concatenate([np.arange(len(outs))[::-1] + len(ins) + 1.3, np.arange(len(ins))[::-1]])
    coord = np.array([m25[k]["diag_aniso_pct"]["median"] for k, _, _, _, _ in rows])
    cost = np.array([c for _, _, _, c, _ in rows]) * 100.0
    diagonalized = [d for _, _, _, _, d in rows]
    div_y, top = len(ins) + 0.65, y[0] + 1.55

    style_ax(ax1, grid_axis="x", hide_left=True)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.75, top)
    bh = 0.6
    ax1.barh(y, coord, height=bh, color=COL["od"], zorder=3)
    ax1.barh(y, 100.0 - coord, left=coord, height=bh, color=COL["rot_fill"], zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{lbl} ({w})" for _, lbl, w, _, _ in rows], fontsize=6.1)
    ax1.tick_params(axis="y", length=0, pad=1.5)
    for tick, (_, _, w, _, _) in zip(ax1.get_yticklabels(), rows):
        tick.set_color(COL["ink"] if w == 2048 else COL["sub"])
        tick.set_fontweight("bold" if w == 2048 else "normal")

    for yi, c in zip(y, coord):
        inside = c > 14.0
        ax1.text(c - 1.6 if inside else c + 1.6, yi, f"{c:.1f}%", fontsize=5.7, fontweight="bold",
                 color="white" if inside else COL["od"], ha="right" if inside else "left",
                 va="center", zorder=5)

    ax1.axhline(div_y, color=COL["spine"], linestyle="--", lw=0.6, zorder=2)
    tr = ax1.get_yaxis_transform()
    ax1.text(-0.015, y[0] + 0.78, "Outputs $B_t$", transform=tr,
             fontsize=6.0, fontweight="bold", color=COL["ink"], ha="right", va="center")
    ax1.text(-0.015, y[len(outs)] + 0.78, "Inputs $C_t$", transform=tr,
             fontsize=6.0, fontweight="bold", color=COL["ink"], ha="right", va="center")

    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax1.set_xlabel("Share of $\\|\\tilde{A} - I\\|_F^2$ a diagonal can represent")
    ax1.text(1.0, top - 0.42, "coordinate", fontsize=6.0, fontweight="bold", color=COL["od"],
             ha="left", va="center")
    ax1.text(99.0, top - 0.42, "rotational", fontsize=6.0, fontweight="bold", color=COL["sub"],
             ha="right", va="center")
    panel_title(ax1, "(a)", "All 11 second moments: geometry and cost")

    # Companion column: share of total dense eigendecomposition cost (all 11 moments = 100%).
    style_ax(axc, grid_axis="x", hide_left=True)
    axc.set_xlim(0, 44.5)
    axc.set_ylim(-0.75, top)
    axc.barh(y, cost, height=bh, zorder=3,
             color=[COL["od"] if d else (COL["wt_soft"] if c > 10 else COL["neutral_lt"])
                    for c, d in zip(cost, diagonalized)])
    for yi, c, d in zip(y, cost, diagonalized):
        if c > 10:
            axc.text(c - 1.1, yi, f"{c:.1f}", fontsize=5.4, fontweight="bold", color="white",
                     ha="right", va="center", zorder=5)
        else:
            axc.text(c + 1.0, yi, f"{c:.1f}", fontsize=5.3, color=COL["muted"], ha="left", va="center")
    axc.axhline(div_y, color=COL["spine"], linestyle="--", lw=0.6, zorder=2)
    axc.set_xticks([])
    axc.set_yticks([])
    axc.set_xlabel("$O(d^3)$ cost\n(% of total)", fontsize=6.0, labelpad=3.0, linespacing=1.3)
    axc.plot([35.0, 37.8, 37.8, 35.0], [y[0] + 0.42, y[0] + 0.42, y[1] - 0.42, y[1] - 0.42],
             color=COL["od"], lw=0.7, clip_on=False)
    axc.text(39.2, 0.5 * (y[0] + y[1]), f"{cost[0] + cost[1]:.1f}%", fontsize=5.6, fontweight="bold",
             color=COL["od"], ha="left", va="center", rotation=90)

    # --- Panel (b): 520M quality-cost frontier, anchored at Muon ---
    def _gains(arm):
        rows = load_jsonl_gz(f"campaigns/confirmation-20261121/runs/wave4a/{arm}_520m_1x_seed0/metrics_from_000000.jsonl.gz")
        st, _, vals = eval_series(rows)
        pal = float(json.load(open(RECORDS / f"paloma_eval/{arm}_520m_1x_seed0.json"))["paloma_macro_mean"])
        return float(np.mean(vals[st >= 5001])), pal

    sel, pal = {}, {}
    for arm in ("muon", "muont", "od_all", "wt_mlp_lag"):
        sel[arm], pal[arm] = _gains(arm)
    g_sel = {k: (sel["muon"] - v) * 1e3 for k, v in sel.items()}
    g_pal = {k: (pal["muon"] - v) * 1e3 for k, v in pal.items()}
    ox = {k: (t520[k]["median_s"] / t520["muon"]["median_s"] - 1.0) * 100.0 for k in sel}

    style_ax(ax2, grid_axis="both")
    frontier = ["muon", "muont", "wt_mlp_lag"]
    ax2.plot([ox[k] for k in frontier], [g_sel[k] for k in frontier], "-", color=COL["wt"],
             lw=1.0, alpha=0.45, zorder=2)
    ax2.plot([ox[k] for k in frontier], [g_pal[k] for k in frontier], ":", color=COL["wt_mid"],
             lw=1.0, alpha=0.55, zorder=2)

    marks = {"muon": COL["muon"], "muont": COL["muont"], "od_all": COL["od"], "wt_mlp_lag": COL["wt"]}
    for k, c in marks.items():
        ax2.plot([ox[k], ox[k]], [g_sel[k], g_pal[k]], color=c, lw=0.6, alpha=0.5, zorder=3)
        ax2.scatter([ox[k]], [g_sel[k]], s=30, c=c, marker="o", edgecolors="white", linewidths=0.6, zorder=5)
        ax2.scatter([ox[k]], [g_pal[k]], s=30, facecolors="white", edgecolors=c, marker="D",
                    linewidths=0.9, zorder=5)

    ax2.text(ox["wt_mlp_lag"] - 0.35, g_pal["wt_mlp_lag"] + 0.15, "WT-MLP-Lag", fontsize=6.4,
             fontweight="bold", color=COL["wt"], ha="right", va="bottom")
    ax2.text(ox["muont"] - 0.3, g_pal["muont"] + 0.2, "MuonT", fontsize=6.3, fontweight="bold",
             color=COL["muont"], ha="right", va="bottom")
    ax2.text(ox["muon"] + 0.3, 0.25, "Muon (reference)", fontsize=6.3, fontweight="bold",
             color=COL["muon"], ha="left", va="bottom")
    ax2.annotate("OD-All: costlier than\nMuonT, gains less", xy=(ox["od_all"] + 0.1, g_sel["od_all"] - 0.05),
                 xytext=(ox["od_all"] + 0.85, 0.80), fontsize=5.6, fontweight="bold", color=COL["od"],
                 ha="left", va="center", linespacing=1.2,
                 arrowprops=dict(arrowstyle="->", color=COL["od"], lw=0.65, shrinkA=1, shrinkB=2))

    ax2.scatter([], [], s=28, c=COL["muted"], marker="o", edgecolors="white", linewidths=0.6,
                label="Held-out mixture")
    ax2.scatter([], [], s=28, facecolors="white", edgecolors=COL["muted"], marker="D", linewidths=0.9,
                label="Paloma, 16 domains")
    ax2.legend(loc="upper left", bbox_to_anchor=(-0.015, 1.015), fontsize=5.8,
               handletextpad=0.3, labelspacing=0.22)

    ax2.axhline(0.0, color=COL["spine"], linestyle=":", lw=0.6, zorder=1)
    ax2.axvline(0.0, color=COL["spine"], linestyle=":", lw=0.6, zorder=1)
    ax2.set_xlim(-1.0, 12.4)
    ax2.set_ylim(-1.2, 11.0)
    ax2.set_xlabel("Step-time overhead vs. Muon (%), $32\\times$A100")
    ax2.set_ylabel("Loss reduction vs. Muon ($\\times 10^{-3}$ nats, $\\uparrow$)")
    panel_title(ax2, "(b)", "520M quality--cost frontier ($d = 1024$)")

    fig.subplots_adjust(left=0.135, right=0.988, bottom=0.175, top=0.895)
    fig.savefig(OUT_DIR / "fig1_overview.pdf")
    plt.close(fig)


# ===========================================================================
# FIGURE 2: Structural Audit (Anisotropy Split, Subspace Drift, Root Fallbacks)
# ===========================================================================
def build_fig2():
    mech = json.load(open(RES_DIR / "mechanism_summary.json"))
    mt = mech["motivation_trajectories"]
    m25_muon = mt["muon"]["step_002500"]
    m25 = mech["motivation_null_all"]["step_002500"]
    m25_wt = mt["wt_mlp_lag"]["step_002500"]
    mdrift_muon = mt["muon"]["grassmann_drift_2250_to_2500"]
    mdrift = mech["motivation_null_all"]["grassmann_drift_250steps"]["step_002250_to_step_002500"]
    mdrift_wt = mt["wt_mlp_lag"]["grassmann_drift_2250_to_2500"]
    fb_steps = mech["motivation_null_all"]["fallback_totals_8rank"]

    order = [
        ("in_attn", "in:attn (512)"),
        ("in_o", "in:o (512)"),
        ("in_mlp", "in:mlp (512)"),
        ("in_down", "in:down (2048)"),
        ("out_o", "out:o (512)"),
        ("out_down", "out:down (512)"),
        ("out_v", "out:v (512)"),
        ("out_k", "out:k (512)"),
        ("out_q", "out:q (512)"),
        ("out_up", "out:up (2048)"),
        ("out_gate", "out:gate (2048)"),
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(6.5, 2.86), gridspec_kw={"width_ratios": [1.16, 0.94, 1.00], "wspace": 0.30}
    )

    y = np.arange(len(order))
    da_muon = np.array([m25_muon[k]["diag_aniso_pct"]["median"] for k, _ in order])
    da_muont = np.array([m25[k]["diag_aniso_pct"]["median"] for k, _ in order])
    da_wt = np.array([m25_wt[k]["diag_aniso_pct"]["median"] for k, _ in order])

    # --- Panel (a): 3-Trajectory Diagonal Anisotropy Bars ---
    style_ax(ax1, grid_axis="x", hide_left=True)
    ax1.axhspan(8.55, 10.45, color=COL["od_tint"], alpha=0.75, zorder=0)
    bh = 0.23
    ax1.barh(y + bh, da_muon, height=bh * 0.92, color=COL["muon"], zorder=3, label="Muon")
    ax1.barh(y, da_muont, height=bh * 0.92, color=COL["muont"], zorder=3, label="MuonT")
    ax1.barh(y - bh, da_wt, height=bh * 0.92, color=COL["wt"], zorder=3, label="WT-MLP-Lag")

    # Clean section divider between Inputs (0..3) and Outputs (4..10)
    ax1.axhline(3.5, color=COL["muted"], linestyle="--", linewidth=0.6, zorder=4)
    ax1.text(98.5, 3.05, "Inputs $C_t$", fontsize=5.6, fontweight="bold", color=COL["sub"], ha="right", va="top")
    ax1.text(98.5, 3.95, "Outputs $B_t$", fontsize=5.6, fontweight="bold", color=COL["sub"], ha="right", va="bottom")

    for idx in (3, 4, 5, 9, 10):
        vmax = max(da_muon[idx], da_muont[idx], da_wt[idx])
        vmin = min(da_muon[idx], da_muont[idx], da_wt[idx])
        if idx in (9, 10):
            ax1.text(vmin - 2.0, idx, f"{vmin:.1f}–{vmax:.1f}%", fontsize=5.1, color="white",
                     fontweight="bold", ha="right", va="center", zorder=5)
        elif idx in (4, 5):
            ax1.text(vmax + 1.8, idx, f"{vmin:.1f}–{vmax:.1f}%", fontsize=5.4, color=COL["wt"],
                     fontweight="bold", ha="left", va="center", zorder=5)
        else:
            ax1.text(vmax + 1.8, idx, f"{vmin:.0f}–{vmax:.0f}%", fontsize=5.4, color=COL["sub"],
                     ha="left", va="center", zorder=5)

    ax1.set_yticks(y)
    ax1.set_yticklabels([lbl for _, lbl in order], fontsize=6.4)
    for tick, (k, _) in zip(ax1.get_yticklabels(), order):
        if k in ("out_gate", "out_up"):
            tick.set_fontweight("bold")
            tick.set_color(COL["od"])
        elif k in ("out_o", "out_down"):
            tick.set_fontweight("bold")
            tick.set_color(COL["wt"])
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.65, len(order) - 0.15)
    ax1.set_xlabel("Diagonal anisotropy $\\mathrm{DiagAniso}$ (%)")
    panel_title(ax1, "(a)", "Coordinate vs. rotational", pad=13)
    ax1.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.005), ncol=3, fontsize=5.4,
               handlelength=0.85, handletextpad=0.28, columnspacing=0.65)

    # --- Panel (b): 3-Trajectory Subspace Grassmann Drift ---
    style_ax(ax2, grid_axis="x", hide_left=True)
    dr_m = np.array([mdrift_muon[k]["drift_top16"]["mean"] for k, _ in order])
    dr_mt = np.array([mdrift[k]["drift_top16"]["mean"] for k, _ in order])
    dr_wt = np.array([mdrift_wt[k]["drift_top16"]["mean"] for k, _ in order])

    ax2.axhspan(8.55, 10.45, color=COL["od_tint"], alpha=0.75, zorder=0)
    ax2.axhline(3.5, color=COL["muted"], linestyle="--", linewidth=0.6, zorder=1)
    ax2.axvline(0.0, color=COL["spine"], linewidth=0.55, zorder=1)
    for i in range(len(order)):
        ax2.plot([0, 0.78], [i, i], color="#F3F5F8", lw=0.45, zorder=0)
        ax2.plot([0, dr_m[i]], [i + bh, i + bh], color=COL["muon"], lw=0.85, zorder=2)
        ax2.plot([0, dr_mt[i]], [i, i], color=COL["muont"], lw=0.85, zorder=2)
        ax2.plot([0, dr_wt[i]], [i - bh, i - bh], color=COL["wt"], lw=0.85, zorder=2)
    ax2.scatter(dr_m, y + bh, s=11, c=COL["muon"], zorder=4, edgecolors="white", linewidths=0.35)
    ax2.scatter(dr_mt, y, s=11, c=COL["muont"], zorder=4, edgecolors="white", linewidths=0.35)
    ax2.scatter(dr_wt, y - bh, s=11, c=COL["wt"], zorder=4, edgecolors="white", linewidths=0.35)

    ax2.annotate(
        "16–32× faster\nthan in:attn",
        xy=(dr_mt[6], 6.0), xytext=(0.48, 7.65),
        fontsize=5.6, fontweight="bold", color=COL["wt"], ha="center", va="center", linespacing=1.15,
        arrowprops=dict(arrowstyle="->", color=COL["wt"], lw=0.6, shrinkA=1, shrinkB=3),
        path_effects=HALO
    )

    ax2.set_yticks(y)
    ax2.set_yticklabels(["" for _ in order])
    ax2.set_xlim(0, 0.78)
    ax2.set_ylim(-0.65, len(order) - 0.15)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax2.set_xlabel("250-step Grassmann $\\mathrm{Drift}_{16}$")
    panel_title(ax2, "(b)", "Subspace drift", pad=13)
    ax2.text(0.0, 1.02, "32-layer mean (2250→2500)", transform=ax2.transAxes, fontsize=5.4, color=COL["muted"])

    # --- Panel (c): All 10 Logged Checkpoints for Exact-Root Fallbacks ---
    style_ax(ax3, grid_axis="y")
    t_steps = [250, 500, 1000, 1750, 2250, 2500, 3500, 4250, 4500, 5120]
    toks_b = [fb_steps[f"step_{s:06d}"]["tokens_B"] for s in t_steps]
    in_fb = [fb_steps[f"step_{s:06d}"]["input_fallbacks_8rank"] / 1000.0 for s in t_steps]
    out_fb = [fb_steps[f"step_{s:06d}"]["output_fallbacks_8rank"] / 1000.0 for s in t_steps]

    ax3.axvspan(0, toks_b[0], color="#F1F5F9", alpha=0.9, zorder=0)
    ax3.fill_between(toks_b, in_fb, out_fb, color=COL["bad"], alpha=0.06, zorder=1)
    ax3.plot(toks_b, out_fb, "-o", color=COL["bad"], lw=1.6, markersize=2.8, zorder=4)
    ax3.plot(toks_b, in_fb, "-s", color=COL["muont"], lw=1.35, markersize=2.5, zorder=3)

    # Direct inline labeling (zero legend box clutter & zero curve crossing)
    ax3.text(0.18, 108.5, "Dense outputs $B_t$\n107.9k total (22.1 eigh/step)",
             fontsize=5.8, fontweight="bold", color=COL["bad"], ha="left", va="top", linespacing=1.18, path_effects=HALO)
    ax3.annotate(
        "3,714× post-warmup\nfallback rate",
        xy=(1.835, 79.3), xytext=(2.10, 44.0),
        fontsize=5.6, color=COL["bad"], fontweight="bold", ha="center", va="center", linespacing=1.15,
        arrowprops=dict(arrowstyle="->", color=COL["bad"], lw=0.6, shrinkA=2, shrinkB=3),
        path_effects=HALO
    )
    ax3.text(2.76, 5.5, "Dense inputs $C_t$: 369 (29 post-warmup)",
             fontsize=5.5, fontweight="bold", color=COL["muont"], ha="right", va="bottom", path_effects=HALO_SM)

    ax3.set_xlim(0, 2.82)
    ax3.set_ylim(-4, 117)
    ax3.set_xlabel("Training tokens (B)")
    ax3.set_ylabel("Cumulative 8-GPU fallbacks ($\\times 10^3$)")
    panel_title(ax3, "(c)", "Exact-root fallbacks", pad=13)
    ax3.text(0.0, 1.02, "Cached-eigenbasis tracker", transform=ax3.transAxes, fontsize=5.6, color=COL["muted"])

    fig.subplots_adjust(left=0.145, right=0.99, bottom=0.165, top=0.835, wspace=0.28)
    fig.savefig(OUT_DIR / "fig2_structural_audit.pdf")
    plt.close(fig)


# ===========================================================================
# FIGURE 3: Width-thresholded factor assignment & to-scale execution timeline
# ===========================================================================
FIG3_W, FIG3_H = 6.5, 3.95
FIG3_YMAX = 100.0 * FIG3_H / FIG3_W

# Measured 130M 8xA100 median step times (tab:bench19), seconds.
T_MUONT, T_SYNC, T_LAG = 1.0967, 1.2208, 1.1548


def _tokens_centered(fig, ax, xc, y, tokens, fontsize):
    """Lay out math tokens left to right, then center the row on xc.

    tokens: list of (text, color). Returns one (x0, x1) data-coordinate span
    per token, used to place the colored factor-slot highlights.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    handles, spans, x = [], [], 0.0
    for text, color in tokens:
        t = ax.text(x, y, text, fontsize=fontsize, color=color, ha="left", va="center", zorder=3)
        bb = t.get_window_extent(renderer=renderer)
        x0, _ = inv.transform((bb.x0, bb.y0))
        x1, _ = inv.transform((bb.x1, bb.y1))
        handles.append(t)
        spans.append([x0, x1])
        x = x1
    shift = xc - 0.5 * (spans[-1][1] - spans[0][0]) - spans[0][0]
    for t, s in zip(handles, spans):
        t.set_x(t.get_position()[0] + shift)
        s[0] += shift
        s[1] += shift
    return spans


def _slot(ax, span, y, color, tint):
    ax.add_patch(FancyBboxPatch((span[0] - 0.4, y - 2.2), span[1] - span[0] + 0.8, 4.4,
                                boxstyle="round,pad=0.0,rounding_size=0.4",
                                facecolor=tint, edgecolor=color, lw=0.5, zorder=2))


def _width_band(ax, x, w, y, h, color, tint, head, rule, chips, roots, cost):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.7",
                                facecolor=tint, edgecolor=color, lw=0.7))
    ax.text(x + 1.6, y + h - 2.4, head, fontsize=6.8, fontweight="bold", color=color, va="center")
    ax.text(x + 1.6, y + h - 5.4, rule, fontsize=6.2, color=COL["ink"], va="center")

    cw, ch = (w - 3.2 - 0.9 * (len(chips) - 1)) / len(chips), 5.2
    cy = y + h - 11.9
    for i, (name, aniso) in enumerate(chips):
        cx = x + 1.6 + i * (cw + 0.9)
        ax.add_patch(FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0.0,rounding_size=0.4",
                                    facecolor="white", edgecolor=color, lw=0.55))
        ax.text(cx + cw / 2, cy + ch * 0.68, name, fontsize=6.3, fontweight="bold",
                color=COL["ink"], ha="center", va="center")
        ax.text(cx + cw / 2, cy + ch * 0.24, aniso, fontsize=5.2, color=COL["muted"],
                ha="center", va="center")

    ax.text(x + 1.6, y + 3.9, roots, fontsize=6.0, color=COL["ink"], va="center")
    ax.text(x + 1.6, y + 1.5, cost, fontsize=5.6, color=COL["muted"], va="center")


def build_fig3():
    fig = plt.figure(figsize=(FIG3_W, FIG3_H))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 100)
    top = FIG3_YMAX

    # ---------------- (a) Which factor enters which slot ----------------
    ax.text(0.8, top - 2.2, "(a)  One update rule, two output-factor classes selected by projection width",
            fontsize=8.0, fontweight="bold", color=COL["ink"], va="center")

    eq_y = top - 8.6
    spans = _tokens_centered(fig, ax, 50.0, eq_y, [
        (r"$D_{p,t} \;=\;$", COL["ink"]),
        (r"$\,\hat{L}_{p,t-1}\,$", COL["wt"]),
        (r"$\;\mathcal{N}_{10}(\;$", COL["ink"]),
        (r"$\,L_{p,t-1}\,$", COL["wt"]),
        (r"$\;U_{p,t}\;$", COL["ink"]),
        (r"$\,R_{p,t}\,$", COL["muont"]),
        (r"$\;)\;\;$", COL["ink"]),
        (r"$\,\hat{R}_{p,t}\,$", COL["muont"]),
    ], fontsize=10.0)
    for i in (1, 3):
        _slot(ax, spans[i], eq_y, COL["wt"], COL["wt_tint"])
    for i in (5, 7):
        _slot(ax, spans[i], eq_y, "#8A96A4", "#EDF0F4")

    ax.text(49.4, eq_y - 5.0, "output (left) factors from $B_p = \\mathbb{E}[\\delta\\delta^{\\mathsf{T}}]$: set by width below",
            fontsize=6.1, color=COL["wt"], ha="right", va="center")
    ax.text(50.6, eq_y - 5.0, "input (right) factors from $C_{g(p)} = \\mathbb{E}[xx^{\\mathsf{T}}]$: dense throughout, as in MuonT",
            fontsize=6.1, color=COL["muont"], ha="left", va="center")

    band_y, band_h = top - 34.5, 18.6
    _width_band(
        ax, 0.8, 60.4, band_y, band_h, COL["wt"], COL["wt_tint"],
        "Outputs of width $d_{\\mathrm{model}}$ = 512 / 1024  ·  rotational",
        "$\\Rightarrow$  dense spectral inverse roots, cached-basis tracker, held one step",
        [("$W_q$", "18%"), ("$W_k$", "14%"), ("$W_v$", "12%"),
         ("$W_o$", "0.8%"), ("$W_{\\mathrm{down}}$", "0.8%")],
        "$L = A(B)^{-1/2}$,    $\\hat{L} = \\mathrm{unitRMS}(A(B)^{-1/8})$",
        "5 of 7 outputs per layer  ·  3.8% of dense output $O(d^3)$ work  ·  chips: DiagAniso (MuonT)",
    )
    _width_band(
        ax, 63.0, 36.2, band_y, band_h, COL["od"], COL["od_tint"],
        "Width $d_{\\mathrm{ff}}$ = 2048 / 4096  ·  coordinate-dominated",
        "$\\Rightarrow$  exact diagonal scaling, every step",
        [("$W_{\\mathrm{gate}}$", "82%"), ("$W_{\\mathrm{up}}$", "78%")],
        "$L = \\mathrm{Diag}(a(B)^{-1/2})$,  inline Triton",
        "2 of 7  ·  removes 96.2% of that work",
    )

    # ---------------- (b) To-scale execution timeline ----------------
    ax.text(0.8, 22.4, "(b)  One step at 130M on $8\\times$A100: the output pipeline on, and off, the critical path",
            fontsize=8.0, fontweight="bold", color=COL["ink"], va="center")

    x0, span_ms, lab_x = 20.0, 1500.0, 19.2
    sx = 79.0 / span_ms

    def tx(ms):
        return x0 + ms * sx

    def bar(ms_a, ms_b, y, h, txt, fc, ec, tc, fs=5.6, fw="bold", z=3):
        ax.add_patch(FancyBboxPatch((tx(ms_a), y), (ms_b - ms_a) * sx, h,
                                    boxstyle="round,pad=0.0,rounding_size=0.35",
                                    facecolor=fc, edgecolor=ec, lw=0.6, zorder=z))
        if txt:
            ax.text(tx(0.5 * (ms_a + ms_b)), y + h / 2, txt, fontsize=fs, fontweight=fw,
                    color=tc, ha="center", va="center", zorder=z + 1)

    def row_label(y, name, color, sub=None):
        ax.text(lab_x, y + 0.6, name, fontsize=6.3, fontweight="bold", color=color, ha="right", va="center")
        if sub:
            ax.text(lab_x, y - 1.8, sub, fontsize=5.4, color=COL["muted"], ha="right", va="center")

    rh = 2.9
    y_ref, y_sync, y_lag, y_side = 16.4, 11.0, 5.6, 0.5
    compute_fc, compute_ec = "#EDF1F6", "#9FADBC"

    # Row 1: MuonT reference.
    row_label(y_ref + rh / 2, "MuonT", COL["muont"], "input side only")
    bar(0, T_MUONT * 1000, y_ref, rh, "forward + backward + update, step $t$", compute_fc, compute_ec, COL["sub"], fw="normal")
    ax.text(tx(T_MUONT * 1000) + 1.2, y_ref + rh / 2, "1.097 s", fontsize=5.8, fontweight="bold",
            color=COL["muont"], ha="left", va="center")

    # Row 2: synchronous two-sided.
    row_label(y_sync + rh / 2, "WT-MLP", COL["bad"], "synchronous")
    bar(0, T_MUONT * 1000, y_sync, rh, "forward + backward, step $t$", compute_fc, compute_ec, COL["sub"], fw="normal")
    bar(T_MUONT * 1000, T_SYNC * 1000, y_sync, rh, "", COL["bad_tint"], COL["bad"], COL["bad"])
    ax.text(tx(T_SYNC * 1000) + 1.2, y_sync + rh / 2, "1.221 s  ($+11.3\\%$)", fontsize=5.8,
            fontweight="bold", color=COL["bad"], ha="left", va="center")
    ax.text(tx(T_SYNC * 1000) - 0.3, y_sync + rh + 1.6,
            "blocking syrk + reduce_scatter + fallback eigh:  $+124$ ms",
            fontsize=5.5, fontweight="bold", color=COL["bad"], ha="right", va="center")

    # Row 3: asynchronous one-step lag (compute stream + side stream).
    row_label(y_lag + rh / 2, "WT-MLP-Lag", COL["wt"], "$\\tau = 1$")
    bar(0, T_MUONT * 1000, y_lag, rh, "forward + backward, step $t$", compute_fc, compute_ec, COL["sub"], fw="normal")
    bar(T_MUONT * 1000, T_LAG * 1000, y_lag, rh, "", COL["wt_soft"], COL["wt"], COL["wt"])
    ax.text(tx(T_LAG * 1000) + 1.2, y_lag + rh / 2, "1.155 s  ($+5.3\\%$)", fontsize=5.8,
            fontweight="bold", color=COL["wt"], ha="left", va="center")
    ax.text(tx(T_LAG * 1000) - 0.3, y_lag + rh + 1.6,
            "grouped Triton capture + batched update:  $+58$ ms",
            fontsize=5.5, fontweight="bold", color=COL["wt"], ha="right", va="center")

    ax.text(lab_x, y_side + rh / 2, "side stream, priority 0", fontsize=5.4, color=COL["od"],
            ha="right", va="center")
    bar(0, 980, y_side, rh, "finishing $L_{t-1}$ (begun in step $t-1$): reduce_scatter + roots + fallback eigh",
        COL["od_tint"], COL["od"], COL["od"], fs=5.3)
    bar(T_MUONT * 1000, T_MUONT * 1000 + 42, y_side, rh, "", COL["od_soft"], COL["od"], "white")
    bar(T_MUONT * 1000 + 42, span_ms, y_side, rh, "$L_t$, hidden under step $t+1$",
        COL["od_tint"], COL["od"], COL["od"], fs=5.3)
    ax.text(tx(T_MUONT * 1000 + 21), y_side - 1.8, "syrk, 1.5 ms", fontsize=5.1, color=COL["od"],
            ha="center", va="center")

    # The step-t update consumes roots that the side stream finished long before.
    ax.annotate("", xy=(tx(T_MUONT * 1000 - 30), y_lag - 0.1), xytext=(tx(985), y_side + rh),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=4.5, color=COL["od"], lw=0.7,
                                connectionstyle="arc3,rad=-0.2"), zorder=6)
    ax.text(tx(975), y_side + rh + 1.2, "ready before the step-$t$ update needs it",
            fontsize=5.4, color=COL["od"], ha="right", va="center")

    # Time axis.
    ax.plot([tx(0), tx(span_ms)], [-2.2, -2.2], color=COL["spine"], lw=0.55)
    for ms in (0, 250, 500, 750, 1000, 1250, 1500):
        ax.plot([tx(ms), tx(ms)], [-2.2, -2.8], color=COL["spine"], lw=0.55)
        ax.text(tx(ms), -4.1, f"{ms / 1000:.2f}", fontsize=5.4, color=COL["muted"], ha="center", va="center")
    ax.text(tx(span_ms) + 1.2, -4.1, "s", fontsize=5.4, color=COL["muted"], ha="left", va="center")

    ax.set_ylim(-5.4, FIG3_YMAX)
    ax.axis("off")
    fig.savefig(OUT_DIR / "fig3_method_pipeline.pdf")
    plt.close(fig)


# ===========================================================================
# FIGURE 4: Eliminating Confounds (Broken-Axis Proxy Cliff, Full Sweep, Null Gate)
# ===========================================================================
def build_fig4():
    lr_sum = json.load(open(RES_DIR / "lr_summary.json"))

    fig = plt.figure(figsize=(6.5, 2.66))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.06], wspace=0.33,
                          left=0.078, right=0.99, bottom=0.175, top=0.875)

    # --- Panel (a): Broken Y-Axis for Short-Horizon Proxy vs Full-Budget Cliff ---
    gs_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], height_ratios=[1.0, 1.18], hspace=0.14)
    ax1_top = fig.add_subplot(gs_a[0])
    ax1_bot = fig.add_subplot(gs_a[1], sharex=ax1_top)

    style_ax(ax1_top, grid_axis="y", hide_bottom=True)
    style_ax(ax1_bot, grid_axis="y")

    mults = ["down", "base", "up", "2x", "2sqrt2x", "4x", "4sqrt2x"]
    lrs_short = [lr_sum[f"{m}/muont"]["lr"] for m in mults]
    mt_short = [lr_sum[f"{m}/muont"]["selection"] for m in mults]
    od_short = [lr_sum[f"{m}/inner_outer_leftdiag"]["selection"] for m in mults]

    # Top panel: 320-step proxy (6% budget)
    ax1_top.plot(lrs_short, mt_short, "--o", color=COL["muont"], lw=1.1, markersize=2.6,
                 markerfacecolor="white", markeredgewidth=0.8, label="MuonT")
    ax1_top.plot(lrs_short, od_short, "--^", color=COL["od"], lw=1.1, markersize=2.8,
                 markerfacecolor="white", markeredgewidth=0.8, label="OD-All")
    ax1_top.scatter([0.064, 0.09051], [4.2510, 4.2314], s=26, facecolors="none", edgecolors=COL["bad"], linewidths=0.85, zorder=5)
    ax1_top.axvline(0.064, color=COL["bad"], linestyle=":", lw=0.65, alpha=0.65)
    ax1_top.axvline(0.09051, color=COL["bad"], linestyle=":", lw=0.65, alpha=0.65)
    ax1_top.text(0.98, 0.90, "320-step proxy (0.17B)", transform=ax1_top.transAxes,
                 fontsize=5.5, fontweight="bold", color=COL["muted"], ha="right", va="top", path_effects=HALO)
    ax1_top.legend(loc="upper right", bbox_to_anchor=(1.0, 0.72), fontsize=5.5, handlelength=1.2, labelspacing=0.14)
    ax1_top.set_ylim(4.20, 4.63)
    ax1_top.set_yticks([4.3, 4.5])
    panel_title(ax1_top, "(a)", "Short-horizon proxy cliff", pad=6.5)

    # Bottom panel: 5,120-step full budget (2.68B)
    mt_full_lr = [0.016, 0.0226274, 0.032, 0.064]
    mt_full_sel = [3.23417, 3.23540, 3.23981, 3.27754]
    od_full_lr = [0.0226274, 0.032, 0.09051]
    od_full_sel = [3.23327, 3.24034, 3.31995]
    wt_full_lr = [0.016, 0.0226274, 0.032]
    wt_full_sel = [3.23033, 3.23111, 3.23558]

    ax1_bot.axvline(0.064, color=COL["bad"], linestyle=":", lw=0.65, alpha=0.65)
    ax1_bot.axvline(0.09051, color=COL["bad"], linestyle=":", lw=0.65, alpha=0.65)
    ax1_bot.plot(mt_full_lr, mt_full_sel, "-o", color=COL["muont"], lw=1.35, markersize=3.0, label="MuonT")
    ax1_bot.plot(od_full_lr, od_full_sel, "-^", color=COL["od"], lw=1.3, markersize=3.1, label="OD-All")
    ax1_bot.plot(wt_full_lr, wt_full_sel, "-s", color=COL["wt"], lw=1.65, markersize=3.1, label="WT-MLP-Lag")

    ax1_bot.annotate(
        "Proxy optima crash\nat 2.68B (+0.042–0.087)",
        xy=(0.064, 3.280), xytext=(0.0148, 3.306),
        fontsize=5.6, fontweight="bold", color=COL["bad"], ha="left", va="center", linespacing=1.12,
        arrowprops=dict(arrowstyle="->", color=COL["bad"], lw=0.65, shrinkA=1, shrinkB=2),
        path_effects=HALO
    )
    ax1_bot.text(0.015, 3.268, "5,120-step full", fontsize=5.5, fontweight="bold", color=COL["sub"])

    clean_log_xaxis(ax1_bot, [0.016, 0.032, 0.064, 0.128], ["0.016", "0.032", "0.064", "0.128"])
    ax1_top.set_xlim(0.0138, 0.145)
    ax1_bot.set_ylim(3.223, 3.332)
    ax1_bot.set_yticks([3.24, 3.28, 3.32])
    ax1_bot.set_xlabel("Matrix learning rate $\\eta$ (log)")
    fig.text(0.018, 0.525, "Held-out selection loss (nats)", rotation=90, va="center", ha="center",
             fontsize=7.4, color=COL["sub"])

    # Minimalist diagonal break marks on left spine
    d = 0.018
    ax1_top.plot([-d, +d], [-d, +d], transform=ax1_top.transAxes, color=COL["spine"], lw=0.65, clip_on=False)
    ax1_bot.plot([-d, +d], [1 - d, 1 + d], transform=ax1_bot.transAxes, color=COL["spine"], lw=0.65, clip_on=False)

    # --- Panel (b): Full-Budget LR Sweep ---
    ax2 = fig.add_subplot(gs[1])
    style_ax(ax2, grid_axis="y")

    ax2.plot([0.016, 0.0226274], [3.24610, 3.25236], "--o", color=COL["muon"], lw=1.2,
             markersize=3.4, markerfacecolor="white", markeredgewidth=0.9, label="Muon")
    ax2.plot([0.016, 0.0226274, 0.032], [3.23417, 3.23540, 3.23981], "-o", color=COL["muont"],
             lw=1.45, markersize=3.4, label="MuonT")
    ax2.plot([0.0226274, 0.032], [3.23327, 3.24034], "-^", color=COL["od"],
             lw=1.35, markersize=3.5, label="OD-All")
    ax2.plot([0.016, 0.0226274, 0.032], [3.23033, 3.23111, 3.23558], "-s", color=COL["wt"],
             lw=1.75, markersize=3.6, label="WT-MLP-Lag")
    ax2.scatter([0.0226274], [3.23510], s=48, c=COL["null"], marker="*", zorder=6, label="Null-WT (−0.0003)")

    # Vertical paired delta guides between MuonT and WT-MLP-Lag
    for lr_x, y_mt, y_wt, txt in [
        (0.016, 3.23417, 3.23033, "−3.8"),
        (0.0226274, 3.23540, 3.23111, "−4.3"),
        (0.032, 3.23981, 3.23558, "−4.2"),
    ]:
        ax2.plot([lr_x, lr_x], [y_wt + 0.0004, y_mt - 0.0004], color=COL["wt"], lw=0.6, linestyle=":", alpha=0.75)
        ax2.text(lr_x, y_wt - 0.0011, txt, fontsize=5.8, fontweight="bold", color=COL["wt"], ha="center", va="top", path_effects=HALO_SM)
    ax2.text(0.0226274, 3.2266, "($\\times 10^{-3}$ nats vs. MuonT)", fontsize=5.3, color=COL["muted"], ha="center")

    clean_log_xaxis(ax2, [0.016, 0.0226274, 0.032], ["0.016", "0.0226", "0.032"])
    ax2.set_xlim(0.0142, 0.0355)
    ax2.set_ylim(3.2255, 3.2555)
    ax2.set_xlabel("Full-budget matrix LR $\\eta$")
    ax2.set_ylabel("Final selection loss (nats)")
    panel_title(ax2, "(b)", "Full-budget LR sweep")
    ax2.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.01), fontsize=5.5,
               handlelength=1.15, handletextpad=0.35, labelspacing=0.16)

    # --- Panel (c): Null Falsification Gate & Late Checkpoints ---
    ax3 = fig.add_subplot(gs[2])
    style_ax(ax3, grid_axis="y")

    r_mt_226 = load_jsonl_gz("runs/wave_sym_fastlag/muont/metrics_from_000000.jsonl.gz")
    r_wt_226 = load_jsonl_gz("runs/wave_sym_fastlag/inner_outer_mlpdiag_lag/metrics_from_000000.jsonl.gz")
    r_null = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave1b/null_wt_130m_1x_seed0_lr0p0226/metrics_from_000000.jsonl.gz")
    r_od_226 = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave1b/od_all_130m_1x_seed0_lr0p0226/metrics_from_000000.jsonl.gz")
    r_mt_016 = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave1a/muont_130m_1x_seed0_lr0p016/metrics_from_000000.jsonl.gz")
    r_wt_016 = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016/metrics_from_000000.jsonl.gz")

    st_m, tk_m, v_mt226 = eval_series(r_mt_226)
    _, _, v_wt226 = eval_series(r_wt_226)
    _, _, v_null = eval_series(r_null)
    _, _, v_od226 = eval_series(r_od_226)
    _, _, v_mt016 = eval_series(r_mt_016)
    _, _, v_wt016 = eval_series(r_wt_016)

    mask = st_m >= 2751
    tk = tk_m[mask]
    ax3.axhspan(-0.5, 0.5, color=COL["null_tint"], alpha=0.95, zorder=0)
    ax3.axhline(0.0, color=COL["muont"], linestyle="--", lw=0.75, zorder=1)
    ax3.text(1.41, 0.65, "$\\pm 0.5$ null gate", fontsize=5.1, color=COL["null"], ha="left", va="bottom")

    ax3.plot(tk, (v_null[mask] - v_mt226[mask]) * 1e3, "-o", color=COL["null"], lw=1.15,
             markersize=2.4, markerfacecolor="white", markeredgewidth=0.8, label="Null-WT (.0226)")
    ax3.plot(tk, (v_od226[mask] - v_mt226[mask]) * 1e3, "-", color=COL["od"], lw=1.15, alpha=0.88, label="OD-All (.0226)")
    ax3.plot(tk, (v_wt016[mask] - v_mt016[mask]) * 1e3, "--", color=COL["wt_mid"], lw=1.25, label="WT-Lag (.016)")
    ax3.plot(tk, (v_wt226[mask] - v_mt226[mask]) * 1e3, "-o", color=COL["wt"], lw=1.65, markersize=2.7, label="WT-Lag (.0226)")

    ax3.text(2.14, -8.35, "WT-MLP-Lag: 10/10 late wins", fontsize=5.8, fontweight="bold",
             color=COL["wt"], ha="center", va="center", path_effects=HALO)
    ax3.set_xlim(1.38, 2.76)
    ax3.set_ylim(-9.3, 4.4)
    ax3.set_yticks([-8, -6, -4, -2, 0, 2, 4])
    ax3.set_xlabel("Training tokens (B)")
    ax3.set_ylabel("$\\Delta$ Sel. vs. MuonT ($\\times 10^{-3}$)")
    panel_title(ax3, "(c)", "Null gate & late checkpoints")
    ax3.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.03), fontsize=5.2, ncol=2,
               handlelength=1.05, handletextpad=0.28, columnspacing=0.65, labelspacing=0.16)

    fig.savefig(OUT_DIR / "fig4_confounds_lr_null.pdf")
    plt.close(fig)


# ===========================================================================
# FIGURE 5: Scaling trajectories, all anchored at Muon (paired 3-seed 130M,
# 4x 80-TPP overtraining, 520M width scaling)
# ===========================================================================
def build_fig5():
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(6.5, 2.72), gridspec_kw={"width_ratios": [1.02, 1.05, 1.05], "wspace": 0.33}
    )
    p2 = json.load(open(RES_DIR / "confirmation_campaign.json"))["phase2_3seed"]

    # --- Panel (a): paired 3-seed reductions against Muon ---
    style_ax(ax1, grid_axis="y")
    ax1.axhline(0.0, color=COL["muon"], linestyle="--", lw=0.75, zorder=2)
    ax1.text(2.42, 0.45, "Muon", fontsize=5.6, color=COL["muon"], ha="right", va="bottom")

    metrics = [("sel", "Held-out\nSelection"), ("c4", "Paloma\nC4-EN"), ("paloma", "Paloma\n16-Dom.")]
    arms = [("muont", "MuonT", COL["muont"]), ("od_all", "OD-All", COL["od_soft"]),
            ("wt_mlp_lag", "WT-MLP-Lag", COL["wt"])]
    xm, w = np.arange(3), 0.24
    for j, (arm, lbl, col) in enumerate(arms):
        off = (j - 1) * w
        seeds = [(np.array(p2["muon"][f"{m}_seeds"]) - np.array(p2[arm][f"{m}_seeds"])) * 1e3
                 for m, _ in metrics]
        ax1.bar(xm + off, [v.mean() for v in seeds], width=w * 0.92, color=col, alpha=0.9,
                zorder=2, label=lbl)
        for i, v in enumerate(seeds):
            ax1.scatter(np.full(3, xm[i] + off) + np.array([-0.045, 0.0, 0.045]), v, s=11,
                        c="white", edgecolors=col, linewidths=0.8, zorder=5)

    ax1.set_xticks(xm)
    ax1.set_xticklabels([l for _, l in metrics], fontsize=6.4)
    ax1.tick_params(axis="x", length=0)
    ax1.set_ylim(-1.0, 21.5)
    ax1.set_ylabel("Paired reduction vs. Muon ($\\times 10^{-3}$, $\\uparrow$)")
    panel_title(ax1, "(a)", "130M 1$\\times$ (seeds 1--3)")
    ax1.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), fontsize=5.6, ncol=3,
               handlelength=0.9, handletextpad=0.3, columnspacing=0.7)
    ax1.text(1.0, 18.6, "WT-MLP-Lag wins 3/3 seeds on all three benchmarks",
             fontsize=5.5, fontweight="bold", color=COL["wt"], ha="center", va="center",
             path_effects=HALO_SM)

    # --- Panel (b): 130M 4x Overtraining (Piecewise Unclipped Scale!) ---
    style_ax(ax2, grid_axis="y")
    r_mt4 = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave3a/muont_130m_4x_seed0/metrics_from_000000.jsonl.gz")
    r_mu4 = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave3a/muon_130m_4x_seed0/metrics_from_000000.jsonl.gz")
    r_wt4_hot = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave3a/wt_mlp_lag_130m_4x_seed0/metrics_from_000000.jsonl.gz")
    r_od4 = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave3b/od_all_130m_4x_seed0_lr0p01056/metrics_from_000000.jsonl.gz")
    r_wt4 = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave3b/wt_mlp_lag_130m_4x_seed0_lr0p01056/metrics_from_000000.jsonl.gz")

    st4, tk4, v_mt4 = eval_series(r_mt4)
    _, _, v_mu4 = eval_series(r_mu4)
    _, _, v_wt4_hot = eval_series(r_wt4_hot)
    _, _, v_od4 = eval_series(r_od4)
    _, _, v_wt4 = eval_series(r_wt4)

    m4 = st4 >= 3001
    set_piecewise_ylim(ax2, y_min=-9.8, y_knee=5.0, y_max=64.0, compress=0.13)
    ax2.set_yticks([-8, -4, 0, 4, 20, 40])
    ax2.axhline(5.0, color=COL["spine"], linestyle=":", lw=0.5, alpha=0.75, zorder=1)
    ax2.axhline(0.0, color=COL["muont"], linestyle="--", lw=0.75, zorder=2)
    ax2.axvspan(5.24, 10.9, color=COL["wt_tint"], alpha=0.55, zorder=0)

    ax2.plot(tk4[m4], (v_mu4[m4] - v_mt4[m4]) * 1e3, "--", color=COL["muon"], lw=1.1, label="Muon (+10.3)")
    ax2.plot(tk4[m4], (v_wt4_hot[m4] - v_mt4[m4]) * 1e3, ":", color=COL["bad_soft"], lw=1.3, label="WT (+41% LR)")
    ax2.plot(tk4[m4], (v_od4[m4] - v_mt4[m4]) * 1e3, "-", color=COL["od"], lw=1.25, label="OD-All (−1.9)")
    ax2.plot(tk4[m4], (v_wt4[m4] - v_mt4[m4]) * 1e3, "-o", color=COL["wt"], lw=1.65, markersize=2.3, label="WT-Lag (−4.0)")

    ax2.text(8.05, -8.55, "11/11 2nd-half wins (−4.44)", fontsize=5.6, fontweight="bold",
             color=COL["wt"], ha="center", va="center", path_effects=HALO)
    ax2.set_xlim(1.5, 11.0)
    ax2.set_xlabel("Training tokens (B) [80 TPP]")
    ax2.set_ylabel("$\\Delta$ Sel. vs. MuonT ($\\times 10^{-3}$)")
    panel_title(ax2, "(b)", "130M 4× overtraining")
    ax2.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.03), fontsize=5.15, ncol=2,
               handlelength=1.0, handletextpad=0.26, columnspacing=0.6, labelspacing=0.14)

    # --- Panel (c): 520M Width Scaling (Piecewise Unclipped Scale!) ---
    style_ax(ax3, grid_axis="y")
    r_mt5 = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave4a/muont_520m_1x_seed0/metrics_from_000000.jsonl.gz")
    r_mu5 = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave4a/muon_520m_1x_seed0/metrics_from_000000.jsonl.gz")
    r_od5 = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave4a/od_all_520m_1x_seed0/metrics_from_000000.jsonl.gz")
    r_wt5 = load_jsonl_gz("campaigns/confirmation-20261121/runs/wave4a/wt_mlp_lag_520m_1x_seed0/metrics_from_000000.jsonl.gz")

    st5, tk5, v_mt5 = eval_series(r_mt5)
    _, _, v_mu5 = eval_series(r_mu5)
    _, _, v_od5 = eval_series(r_od5)
    _, _, v_wt5 = eval_series(r_wt5)

    m5 = st5 >= 1501
    set_piecewise_ylim(ax3, y_min=-8.8, y_knee=4.0, y_max=25.5, compress=0.22)
    ax3.set_yticks([-8, -4, 0, 4, 15, 25])
    ax3.axhline(4.0, color=COL["spine"], linestyle=":", lw=0.5, alpha=0.75, zorder=1)
    ax3.axhline(0.0, color=COL["muont"], linestyle="--", lw=0.75, zorder=2)
    ax3.axvspan(5.24, 10.9, color=COL["wt_tint"], alpha=0.55, zorder=0)

    ax3.plot(tk5[m5], (v_mu5[m5] - v_mt5[m5]) * 1e3, "--", color=COL["muon"], lw=1.1, label="Muon (+6.5)")
    ax3.plot(tk5[m5], (v_od5[m5] - v_mt5[m5]) * 1e3, "-^", color=COL["od"], lw=1.25, markersize=2.3, label="OD-All (+0.6)")
    ax3.plot(tk5[m5], (v_wt5[m5] - v_mt5[m5]) * 1e3, "-o", color=COL["wt"], lw=1.65, markersize=2.4, label="WT-MLP-Lag (−2.8)")

    ax3.annotate(
        "OD-All falls behind\nMuonT (+0.58)",
        xy=(10.50, 0.72), xytext=(7.45, 2.95),
        fontsize=5.4, fontweight="bold", color=COL["bad"], ha="center", va="center", linespacing=1.1,
        arrowprops=dict(arrowstyle="->", color=COL["bad"], lw=0.6, shrinkA=1, shrinkB=2,
                        connectionstyle="arc3,rad=-0.10"),
        path_effects=HALO
    )
    ax3.text(8.05, -7.65, "11/11 2nd-half wins (−3.09)", fontsize=5.6, fontweight="bold",
             color=COL["wt"], ha="center", va="center", path_effects=HALO)
    ax3.set_xlim(1.5, 11.0)
    ax3.set_xlabel("Training tokens (B) [$d = 1024$]")
    ax3.set_ylabel("$\\Delta$ Sel. vs. MuonT ($\\times 10^{-3}$)")
    panel_title(ax3, "(c)", "520M width scaling")
    ax3.legend(loc="upper right", bbox_to_anchor=(1.02, 1.02), fontsize=5.3,
               handlelength=1.1, handletextpad=0.3, labelspacing=0.14)

    fig.subplots_adjust(left=0.076, right=0.99, bottom=0.175, top=0.875, wspace=0.34)
    fig.subplots_adjust(left=0.078, right=0.99, bottom=0.175, top=0.875, wspace=0.33)
    fig.savefig(OUT_DIR / "fig5_scaling_trajectories.pdf")
    plt.close(fig)


# ===========================================================================
# FIGURE 6: 16-Domain Paloma Out-of-Distribution Generalization Forest Plot
# ===========================================================================
def build_fig6():
    pal_dir = RECORDS / "paloma_eval"

    def load_pal(name: str) -> dict:
        return json.load(open(pal_dir / f"{name}.json"))

    def gloss(p: dict, dom: str) -> float:
        v = p["scores"][dom]
        return float(v["loss"] if isinstance(v, dict) else v)

    p_mt4 = load_pal("muont_130m_4x_seed0")
    p_od4 = load_pal("od_all_130m_4x_seed0_lr0p01056")
    p_wt4 = load_pal("wt_mlp_lag_130m_4x_seed0_lr0p01056")

    p_mt5 = load_pal("muont_520m_1x_seed0")
    p_od5 = load_pal("od_all_520m_1x_seed0")
    p_wt5 = load_pal("wt_mlp_lag_520m_1x_seed0")

    dom_map = [
        ("paloma/c4_en", "C4-EN"),
        ("paloma/c4_100_domains", "C4 (100 Domains)"),
        ("paloma/mc4", "mC4 (Multilingual)"),
        ("paloma/dolma-v1_5", "Dolma v1.5"),
        ("paloma/falcon-refinedweb", "Falcon-RefinedWeb"),
        ("paloma/redpajama", "RedPajama"),
        ("paloma/m2d2_wikipedia_unsplit", "M2D2 Wikipedia"),
        ("paloma/m2d2_s2orc_unsplit", "M2D2 S2ORC (Sci.)"),
        ("paloma/dolma_100_programing_languages", "Dolma 100-Code"),
        ("paloma/wikitext_103", "WikiText-103"),
        ("paloma/ptb", "Penn Treebank"),
        ("paloma/dolma_100_subreddits", "Dolma Subreddits"),
        ("paloma/twitterAAE_HELM_fixed", "TwitterAAE"),
        ("paloma/manosphere_meta_sep", "Manosphere"),
        ("paloma/4chan_meta_sep", "4chan"),
        ("paloma/gab", "Gab"),
    ]
    dom_rev = [("__macro__", "16-Domain Macro Mean")] + list(reversed(dom_map))

    od4_d, wt4_d, od5_d, wt5_d, labels = [], [], [], [], []
    for key, lbl in dom_rev:
        labels.append(lbl)
        if key == "__macro__":
            od4_d.append((p_od4["paloma_macro_mean"] - p_mt4["paloma_macro_mean"]) * 1e3)
            wt4_d.append((p_wt4["paloma_macro_mean"] - p_mt4["paloma_macro_mean"]) * 1e3)
            od5_d.append((p_od5["paloma_macro_mean"] - p_mt5["paloma_macro_mean"]) * 1e3)
            wt5_d.append((p_wt5["paloma_macro_mean"] - p_mt5["paloma_macro_mean"]) * 1e3)
        else:
            od4_d.append((gloss(p_od4, key) - gloss(p_mt4, key)) * 1e3)
            wt4_d.append((gloss(p_wt4, key) - gloss(p_mt4, key)) * 1e3)
            od5_d.append((gloss(p_od5, key) - gloss(p_mt5, key)) * 1e3)
            wt5_d.append((gloss(p_wt5, key) - gloss(p_mt5, key)) * 1e3)

    y = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.52), sharey=True, gridspec_kw={"wspace": 0.12})

    for ax, od_vals, wt_vals, tag, ttl, sub in [
        (ax1, od4_d, wt4_d, "(a)", "130M 4× Overtraining (80 TPP)", "WT-MLP-Lag: 16/16 wins (−5.66)   ·   OD-All: 15/16 (−2.59)"),
        (ax2, od5_d, wt5_d, "(b)", "520M Width Scaling ($d = 1024$)", "WT-MLP-Lag: 15/16 wins (−5.45)   ·   OD-All: 6/16 (+0.47)"),
    ]:
        style_ax(ax, grid_axis="x", hide_left=True)
        # Subtle highlight only on bottom macro-mean summary row (clean white above!)
        ax.axhspan(-0.55, 0.50, color=COL["wt_tint"], alpha=0.9, zorder=0)
        ax.axhline(0.50, color=COL["spine"], lw=0.65, linestyle="-", zorder=2)
        ax.axvline(0.0, color=COL["muont"], lw=0.8, linestyle="--", zorder=2)

        for i in range(len(labels)):
            if i > 0:
                ax.axhline(i, color="#F3F6F9", lw=0.45, zorder=0)
            seg_col = "#8FB1E0" if wt_vals[i] < od_vals[i] else "#CBD5E1"
            ax.plot([od_vals[i], wt_vals[i]], [i, i], color=seg_col, lw=1.15, zorder=3)

        ax.scatter(od_vals, y, s=26, c=COL["od"], marker="^", edgecolors="white", linewidths=0.45, zorder=4)
        ax.scatter(wt_vals, y, s=30, c=COL["wt"], marker="o", edgecolors="white", linewidths=0.45, zorder=5)

        panel_title(ax, tag, ttl, pad=13.5)
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=5.7, fontweight="bold", color=COL["wt"], ha="left", va="bottom")

    ax1.set_xlim(-21.8, max(5.5, max(od4_d) + 1.8))
    ax2.set_xlim(-44.0, 14.2)

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=6.6)
    ax1.get_yticklabels()[0].set_fontweight("bold")
    ax1.get_yticklabels()[0].set_color(COL["wt"])
    ax1.set_ylim(-0.65, len(labels) - 0.3)

    # Shared frameless top legend
    top_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="None", color=COL["wt"], markersize=5.2, label="WT-MLP-Lag (ours)"),
        mpl.lines.Line2D([], [], marker="^", linestyle="None", color=COL["od"], markersize=5.0, label="OD-All (purely diagonal)"),
        mpl.lines.Line2D([], [], linestyle="--", color=COL["muont"], lw=0.85, label="MuonT baseline (0)"),
    ]
    fig.legend(handles=top_handles, loc="upper center", bbox_to_anchor=(0.59, 0.995),
               ncol=3, fontsize=6.2, handletextpad=0.35, columnspacing=1.4)

    fig.supxlabel("$\\Delta$ Loss vs. MuonT ($\\times 10^{-3}$ nats; negative values indicate lower out-of-distribution loss)",
                  fontsize=7.3, color=COL["sub"], x=0.60, y=0.022)
    fig.subplots_adjust(left=0.23, right=0.99, bottom=0.115, top=0.835, wspace=0.12)
    fig.savefig(OUT_DIR / "fig6_paloma_domains.pdf")
    plt.close(fig)


# ===========================================================================
# FIGURE 7: Design-Space Ablations & Head-to-Head vs. GO-MUON / Mousse
# ===========================================================================
def build_fig7():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.5, 2.88), gridspec_kw={"width_ratios": [1.02, 1.08], "wspace": 0.31}
    )

    # --- Panel (a): Disciplined Semantic Color Hierarchy for 14 Ablations ---
    style_ax(ax1, grid_axis="x", hide_left=True)
    abl_rows = [
        ("WT-MLP (sync)", -6.58, COL["wt"]),
        ("DT-All (dense out)", -5.78, COL["dt"]),
        ("WT-MLP-Lag (ours)", -5.68, COL["wt"]),
        ("OD-All (diag out)", -4.50, COL["od"]),
        ("Abl-OuterDense", -3.98, COL["neutral_bar"]),
        ("Abl-No2048", -3.07, COL["neutral_bar"]),
        ("WT-Sym (sym. $d_{\\mathrm{ff}}$)", -2.50, COL["neutral_bar"]),
        ("Abl-DownOnly", -1.37, COL["neutral_bar"]),
        ("Null-WT (control)", -0.73, COL["null"]),
        ("MuonT (baseline)", 0.00, COL["muont"]),
        ("Abl-DownSwap", +2.51, COL["bad_soft"]),
        ("Muon (no factors)", +5.51, COL["muon"]),
        ("Abl-OutOnly", +8.71, COL["bad_soft"]),
        ("Abl-BothDiag", +18.01, COL["bad"]),
    ]
    abl_rev = list(reversed(abl_rows))
    ya = np.arange(len(abl_rev))
    vals_a = [r[1] for r in abl_rev]
    cols_a = [r[2] for r in abl_rev]

    ax1.axvline(0.0, color=COL["muont"], linestyle="--", lw=0.75, zorder=2)
    ax1.barh(ya, vals_a, height=0.64, color=cols_a, zorder=3)
    for idx, (lbl, v, c) in enumerate(abl_rev):
        is_key = "WT-MLP" in lbl or "OD-All" in lbl
        txt_c = c if (is_key or c in (COL["bad"], COL["null"])) else COL["sub"]
        if v < -0.1:
            ax1.text(v - 0.45, idx, f"{v:+.1f}", fontsize=5.6, color=txt_c,
                     fontweight="bold" if "WT-MLP" in lbl else "normal", ha="right", va="center", path_effects=HALO_SM)
        elif v > 0.1:
            ax1.text(v + 0.45, idx, f"{v:+.1f}", fontsize=5.6, color=txt_c, ha="left", va="center", path_effects=HALO_SM)
        else:
            ax1.text(0.45, idx, "0.0", fontsize=5.5, color=COL["muont"], fontweight="bold", ha="left", va="center")

    ax1.set_yticks(ya)
    ax1.set_yticklabels([r[0] for r in abl_rev], fontsize=6.2)
    for tick, (lbl, _, _) in zip(ax1.get_yticklabels(), abl_rev):
        if lbl.startswith("WT-MLP"):
            tick.set_fontweight("bold")
            tick.set_color(COL["wt"])
        elif lbl.startswith("MuonT"):
            tick.set_fontweight("bold")
            tick.set_color(COL["ink"])
    ax1.set_xlim(-11.2, 23.6)
    ax1.set_ylim(-0.65, len(abl_rev) - 0.35)
    ax1.set_xlabel("Late $\\Delta$ Sel. vs. MuonT ($\\times 10^{-3}$, $\\leftarrow$ better)")
    panel_title(ax1, "(a)", "Factor-placement ablations")

    # --- Panel (b): Multi-Seed Comparison vs. GO-MUON & Mousse (Zero Inset Collision!) ---
    style_ax(ax2, grid_axis="y")
    rel_base = "campaigns/related-selected-full-seed12-20261121/runs"

    def mean_2seed_series(arm_prefix: str):
        r1 = load_jsonl_gz(f"{rel_base}/{arm_prefix}_0p5m_seed1_lr2/metrics_from_000000.jsonl.gz")
        r2 = load_jsonl_gz(f"{rel_base}/{arm_prefix}_0p5m_seed2_lr2/metrics_from_000000.jsonl.gz")
        tr1 = np.array([float(r["train_loss"]) for r in r1])
        tr2 = np.array([float(r["train_loss"]) for r in r2])
        st, tk, s1 = eval_series(r1)
        _, _, s2 = eval_series(r2)
        return 0.5 * (tr1 + tr2), st, tk, 0.5 * (s1 + s2)

    tr_mu, st_e, tk_e, sel_mu = mean_2seed_series("muon_control")
    tr_wt, _, _, sel_wt = mean_2seed_series("inner_outer_mlpdiag")
    tr_mc, _, _, sel_mc = mean_2seed_series("mousse_control")
    tr_mp, _, _, sel_mp = mean_2seed_series("mousse_paper")
    tr_gc, _, _, sel_gc = mean_2seed_series("go_muon_control")
    tr_gp, _, _, sel_gp = mean_2seed_series("go_muon_paper")

    ax2.axhline(0.0, color=COL["muon"], linestyle="--", lw=0.75, label="Muon baseline (0)")
    idx_e = st_e >= 251
    ax2.plot(tk_e[idx_e], (sel_gp[idx_e] - sel_mu[idx_e]), "--", color=COL["bad_soft"], lw=1.2, label="GO-MUON paper (+0.032)")
    ax2.plot(tk_e[idx_e], (sel_gc[idx_e] - sel_mu[idx_e]), "-", color=COL["bad"], lw=1.35, label="GO-MUON ctrl (+0.031)")
    ax2.plot(tk_e[idx_e], (sel_mp[idx_e] - sel_mu[idx_e]), "--", color=COL["null"], lw=1.1, alpha=0.85, label="Mousse paper (+0.003)")
    ax2.plot(tk_e[idx_e], (sel_mc[idx_e] - sel_mu[idx_e]), "-", color=COL["null"], lw=1.25, label="Mousse ctrl (−0.001)")
    ax2.plot(tk_e[idx_e], (sel_wt[idx_e] - sel_mu[idx_e]), "-o", color=COL["wt"], lw=1.75, markersize=2.6, label="WT-MLP (−0.010)")

    # Clean inset in lower-right open area strictly below WT-MLP curve (-0.010)
    ins = ax2.inset_axes([0.44, 0.08, 0.53, 0.235])
    ins.set_facecolor("#FAFBFD")
    for sp in ("top", "right"):
        ins.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ins.spines[sp].set_color(COL["spine"])
        ins.spines[sp].set_linewidth(0.5)
    ins.axhline(0.0, color=COL["muon"], linestyle="--", lw=0.6)
    ker = np.ones(9) / 9.0
    xs_early = np.arange(1, 301)
    ins.plot(xs_early, np.convolve((tr_gc - tr_mu)[:300], ker, mode="same"), color=COL["bad"], lw=1.05)
    ins.plot(xs_early, np.convolve((tr_mc - tr_mu)[:300], ker, mode="same"), color=COL["null"], lw=0.95)
    ins.plot(xs_early, np.convolve((tr_wt - tr_mu)[:300], ker, mode="same"), color=COL["wt"], lw=1.15)
    ins.set_xlim(1, 300)
    ins.set_ylim(-0.62, 0.16)
    ins.set_yticks([-0.5, 0.0])
    ins.tick_params(labelsize=5.1, length=1.8, pad=1.8)
    ins.set_title("Early steps 1–300 ($\\Delta$ train)", loc="left", pad=2.0, fontsize=5.2, fontweight="bold", color=COL["sub"])
    ins.text(178, -0.50, "GO-MUON crosses >0 @ ~200", fontsize=4.8, fontweight="bold", color=COL["bad"], ha="center")

    ax2.set_xlim(0.05, 2.78)
    ax2.set_ylim(-0.155, 0.165)
    ax2.set_xlabel("Training tokens (B) [seeds 1 & 2 mean]")
    ax2.set_ylabel("Paired $\\Delta$ Sel. vs. Muon (nats)")
    panel_title(ax2, "(b)", "vs. GO-MUON & Mousse")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.02, 1.03), fontsize=5.3, ncol=1,
               handlelength=1.15, handletextpad=0.3, labelspacing=0.14)

    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.16, top=0.88, wspace=0.31)
    fig.savefig(OUT_DIR / "fig7_ablations_and_related.pdf")
    plt.close(fig)


# ===========================================================================
# FIGURE 8: Mechanistic Rotation & Multi-Scale (130M + 520M) Pareto Frontier
# ===========================================================================
def build_fig8():
    mech = json.load(open(RES_DIR / "mechanism_summary.json"))
    d_ak = mech["direction_step2500_by_arm_kind"]
    timings = json.load(open(RES_DIR / "measured_8gpu_timings.json"))["methods"]
    t520 = json.load(open(RES_DIR / "measured_520m_timings.json"))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(6.5, 4.62),
        gridspec_kw={"width_ratios": [1.0, 1.04], "height_ratios": [1.0, 1.06], "wspace": 0.30, "hspace": 0.38}
    )

    # --- Panel (a): Counterfactual Update Rotation + 32-Layer SD Whiskers ---
    style_ax(ax1, grid_axis="y")
    projs = ["q", "k", "v", "o", "down", "gate", "up"]
    xp = np.arange(len(projs))
    ang_wt = np.array([d_ak["wave2/inner_outer_mlpdiag"][p]["graft_angle_deg"]["mean"] for p in projs])
    std_wt = np.array([d_ak["wave2/inner_outer_mlpdiag"][p]["graft_angle_deg"]["std"] for p in projs])
    ang_od = np.array([d_ak["wave2/inner_outer_leftdiag"][p]["graft_angle_deg"]["mean"] for p in projs])
    std_od = np.array([d_ak["wave2/inner_outer_leftdiag"][p]["graft_angle_deg"]["std"] for p in projs])

    bw = 0.35
    ax1.axvspan(2.55, 4.45, color=COL["wt_tint"], alpha=0.8, zorder=0)
    ax1.bar(xp - bw / 2, ang_od, width=bw, color=COL["od_soft"], zorder=2, label="OD-All")
    ax1.bar(xp + bw / 2, ang_wt, width=bw, color=COL["wt"], zorder=2, label="WT-MLP")

    ax1.errorbar(xp - bw / 2, ang_od, yerr=std_od, fmt="none", ecolor=COL["od"], lw=0.6, capsize=1.5, capthick=0.6, zorder=3)
    ax1.errorbar(xp + bw / 2, ang_wt, yerr=std_wt, fmt="none", ecolor="#0B254D", lw=0.6, capsize=1.5, capthick=0.6, zorder=3)

    ax1.annotate(
        "Residual (o, down)\nWT: 17.7° / 17.0°\nOD: 4.3° / 5.4°",
        xy=(3.5, 19.8), xytext=(3.5, 28.0),
        fontsize=5.5, fontweight="bold", color=COL["wt"], ha="center", va="center", linespacing=1.15,
        arrowprops=dict(arrowstyle="->", color=COL["wt"], lw=0.6, shrinkA=1, shrinkB=2),
        path_effects=HALO
    )
    ax1.text(
        0.02, 0.98, "$\\|D_{\\mathrm{WT}}\\|_F / \\|D_{\\mathrm{MT}}\\|_F = 0.985$ ($-1.5\\%$)",
        transform=ax1.transAxes, fontsize=5.6, va="top", ha="left", color=COL["ink"], fontweight="bold"
    )

    ax1.set_xticks(xp)
    ax1.set_xticklabels(projs, fontsize=6.4)
    ax1.tick_params(axis="x", length=0)
    ax1.set_ylim(0, 41.0)
    ax1.set_xlabel("Projection $W_p$ (130M, step 2,500)")
    ax1.set_ylabel("Rotation $\\theta(D, D_{\\mathrm{MuonT}})$ (deg)")
    panel_title(ax1, "(a)", "Counterfactual subspace rotation")
    ax1.legend(loc="upper left", bbox_to_anchor=(-0.01, 0.89), fontsize=5.7, ncol=2,
               handlelength=0.95, handletextpad=0.3, columnspacing=0.7)

    # --- Panel (b): Disciplined Minimalist 8xA100 Overhead Hierarchy (130M) ---
    style_ax(ax2, grid_axis="x", hide_left=True)
    sys_items = [
        ("Muon", timings["muon"]["measured_benchmark"]["median_overhead_vs_muont_pct"], COL["muon"], "19.0G"),
        ("MuonT", 0.0, COL["muont"], "20.5G"),
        ("OD-All", timings["inner_outer_leftdiag"]["measured_benchmark"]["median_overhead_vs_muont_pct"], COL["od"], "20.5G"),
        ("WT-Sym-Lag", timings["inner_outer_sym_mlpdiag_lag"]["measured_benchmark"]["median_overhead_vs_muont_pct"], COL["wt_mid"], "20.8G"),
        ("WT-MLP-Lag", timings["inner_outer_mlpdiag_lag"]["measured_benchmark"]["median_overhead_vs_muont_pct"], COL["wt"], "21.6G"),
        ("WT-MLP", timings["inner_outer_mlpdiag"]["measured_benchmark"]["median_overhead_vs_muont_pct"], COL["neutral_lt"], "21.5G"),
        ("DT-All-Lag", timings["inner_outer_all_lag"]["measured_benchmark"]["median_overhead_vs_muont_pct"], COL["neutral_lt"], "24.8G"),
        ("DT-All", timings["inner_outer_all"]["measured_benchmark"]["median_overhead_vs_muont_pct"], COL["neutral_lt"], "24.2G"),
        ("Null-Dense", timings["null_all"]["measured_benchmark"]["median_overhead_vs_muont_pct"], COL["neutral_lt"], "24.2G"),
    ]
    sys_rev = list(reversed(sys_items))
    yb = np.arange(len(sys_rev))
    vb = [r[1] for r in sys_rev]
    cb = [r[2] for r in sys_rev]

    # Re-express MuonT-referenced benchmark overheads against Muon (1.0967 / 1.0810 s).
    MUONT_OVER_MUON = 1.0967 / 1.0810
    vb = [((1.0 + v / 100.0) * MUONT_OVER_MUON - 1.0) * 100.0 for v in vb]
    sys_rev = [(lbl, ((1.0 + val / 100.0) * MUONT_OVER_MUON - 1.0) * 100.0, col, mem)
               for lbl, val, col, mem in sys_rev]
    ax2.axvline(0.0, color=COL["muon"], linestyle="--", lw=0.75, zorder=2)
    ax2.barh(yb, vb, height=0.62, color=cb, zorder=3)
    for idx, (lbl, val, col, mem) in enumerate(sys_rev):
        xpos = max(val, 0.0) + 0.65
        is_hi = lbl in ("WT-MLP-Lag", "WT-Sym-Lag")
        ax2.text(xpos, idx, f"{val:+.1f}% ({mem})", fontsize=5.5,
                 color=COL["wt"] if lbl == "WT-MLP-Lag" else (COL["sub"] if is_hi else COL["muted"]),
                 fontweight="bold" if is_hi else "normal", va="center", path_effects=HALO_SM)

    ax2.set_yticks(yb)
    ax2.set_yticklabels([r[0] for r in sys_rev], fontsize=6.3)
    for tick, (lbl, _, _, _) in zip(ax2.get_yticklabels(), sys_rev):
        if lbl == "WT-MLP-Lag":
            tick.set_fontweight("bold")
            tick.set_color(COL["wt"])
        elif lbl == "MuonT":
            tick.set_fontweight("bold")
            tick.set_color(COL["ink"])
    ax2.set_xlim(-1.5, 41.5)
    ax2.set_ylim(-0.6, len(sys_rev) - 0.35)
    ax2.set_xlabel("Step overhead vs. Muon (%) [8×A100]")
    panel_title(ax2, "(b)", "130M step-time & memory profile")

    # --- Panel (c): 130M Design-Space Pareto Frontier (d = 512) ---
    style_ax(ax3, grid_axis="both")
    # Stored as (overhead, gain) against MuonT; shifted onto Muon for plotting.
    MUONT_OVER_MUON, MUONT_GAIN = 1.0967 / 1.0810, 5.51

    def _to_muon(ox_mt, gy_mt):
        return ((1.0 + ox_mt / 100.0) * MUONT_OVER_MUON - 1.0) * 100.0, gy_mt + MUONT_GAIN

    fx, fy = zip(*[_to_muon(x, y) for x, y in
                   zip([-1.43, 0.00, 2.22, 5.29, 11.31, 29.0],
                       [-5.51, 0.00, 4.50, 5.68, 6.58, 6.58])])
    fx, fy = list(fx), list(fy)
    ax3.fill_between(fx, -1.7, fy, color=COL["wt_tint"], alpha=0.45, zorder=0)
    ax3.plot(fx, fy, "--", color=COL["wt"], lw=1.05, alpha=0.80, zorder=2)
    ax3.axhline(0.0, color=COL["muon"], linestyle=":", lw=0.65, zorder=1)
    ax3.axvline(0.0, color=COL["spine"], linestyle=":", lw=0.55, zorder=1)

    pareto_pts = [
        (-1.43, -5.51, "Muon", COL["muon"], "o", True, (0.6, -5.51), "left"),
        (0.00, 0.00, "MuonT", COL["muont"], "D", True, (1.2, -0.75), "left"),
        (2.22, 4.50, "OD-All", COL["od"], "^", True, (0.9, 5.55), "center"),
        (5.29, 5.68, "WT-MLP-Lag", COL["wt"], "o", True, (5.3, 7.25), "center"),
        (11.31, 6.58, "WT-MLP", COL["wt"], "s", True, (11.6, 7.70), "center"),
        (3.32, 2.50, "WT-Sym", COL["wt_mid"], "d", False, (4.6, 2.35), "left"),
        (12.05, 3.07, "No2048", COL["muted"], "o", False, (12.1, 1.75), "center"),
        (16.55, 5.89, "DT-All-Lag", COL["dt"], "v", False, (16.5, 4.50), "center"),
        (21.57, 5.78, "DT-All", COL["dt"], "P", False, (21.6, 4.40), "center"),
        (26.38, 3.98, "OuterDense", COL["muted"], "s", False, (24.8, 2.55), "center"),
    ]

    pareto_pts = [(*_to_muon(ox, gy), lbl, col, mk, is_front, _to_muon(tx, ty), ha)
                  for ox, gy, lbl, col, mk, is_front, (tx, ty), ha in pareto_pts]

    for ox, gy, lbl, col, mk, is_front, (tx, ty), ha in pareto_pts:
        if is_front:
            sz = 46 if "WT-MLP" in lbl else 34
            ax3.scatter([ox], [gy], s=sz, c=col, marker=mk, zorder=5, edgecolors="white", linewidths=0.65)
        else:
            ax3.scatter([ox], [gy], s=21, facecolors="white", edgecolors=col, marker=mk, linewidths=0.85, zorder=4)

        if abs(tx - ox) > 1.2 or abs(ty - gy) > 1.0:
            ax3.plot([ox, tx], [gy, ty - (0.35 if ty > gy else -0.35)], color=col, lw=0.45, alpha=0.55, zorder=3)

        ax3.text(tx, ty, lbl, fontsize=5.6,
                 fontweight="bold" if lbl in ("WT-MLP-Lag", "WT-MLP", "MuonT") else "normal",
                 color=col if is_front else COL["muted"], ha=ha, va="center", path_effects=HALO_SM, zorder=6)

    ax3.set_xlim(-1.2, 30.5)
    ax3.set_ylim(-1.5, 14.2)
    ax3.set_xlabel("Step overhead vs. Muon (%) [8×A100]")
    ax3.set_ylabel("Quality gain vs. Muon ($\\times 10^{-3}$, $\\uparrow$)")
    panel_title(ax3, "(c)", "130M Pareto frontier ($d = 512$)")

    # --- Panel (d): 520M Pareto and systems progression, anchored at Muon ---
    style_ax(ax4, grid_axis="both")

    def _load_520m_gains(arm: str):
        rows = load_jsonl_gz(f"campaigns/confirmation-20261121/runs/wave4a/{arm}_520m_1x_seed0/metrics_from_000000.jsonl.gz")
        st, _, vals = eval_series(rows)
        late_mean = float(np.mean(vals[st >= 5120]))
        step_med = float(np.median([float(r["step_seconds"]) for r in rows if r.get("step_seconds") is not None]))
        pal = float(json.load(open(RECORDS / f"paloma_eval/{arm}_520m_1x_seed0.json"))["paloma_macro_mean"])
        return late_mean, pal, step_med

    l5, p5, s5 = {}, {}, {}
    for arm in ("muon", "muont", "od_all", "wt_mlp_lag"):
        l5[arm], p5[arm], s5[arm] = _load_520m_gains(arm)
    g_sel = {k: (l5["muon"] - v) * 1e3 for k, v in l5.items()}
    g_pal = {k: (p5["muon"] - v) * 1e3 for k, v in p5.items()}

    ws32 = t520["final_ws32_mb2_c2"]
    ox = {k: (ws32[k]["median_s"] / ws32["muon"]["median_s"] - 1.0) * 100.0 for k in l5}
    ws8 = t520["final_ws8_mb2_c2"]
    ox_wt8_opt = (5.2019 / 4.6423 - 1.0) * 100.0  # matched-checkpoint 8xA100 medians (tab:bench520m)
    ox_wt8_unchunk = (s5["wt_mlp_lag"] / s5["muon"] - 1.0) * 100.0

    y_wt = g_sel["wt_mlp_lag"]
    ax4.fill_between([0.0, ox["muont"], ox["wt_mlp_lag"], 17.0], -1.0,
                     [0.0, g_sel["muont"], y_wt, y_wt], color=COL["wt_tint"], alpha=0.5, zorder=0)
    ax4.plot([0.0, ox["muont"], ox["wt_mlp_lag"]], [0.0, g_sel["muont"], y_wt], "--",
             color=COL["wt"], lw=1.1, alpha=0.85, zorder=2, label="Held-out ($-\\Delta\\mathcal{L}_{\\mathrm{sel}}$)")
    ax4.plot([0.0, ox["muont"], ox["wt_mlp_lag"]], [0.0, g_pal["muont"], g_pal["wt_mlp_lag"]], ":",
             color=COL["wt_mid"], lw=1.05, alpha=0.85, zorder=2, label="Paloma 16-Dom ($-\\Delta\\mathcal{L}_{\\mathrm{Pal}}$)")

    # Systems progression at fixed quality: 8x mb=1 -> 8x chunk=2 -> 32x RDMA.
    ax4.annotate("", xy=(ox_wt8_opt + 0.42, y_wt), xytext=(ox_wt8_unchunk - 0.42, y_wt),
                 arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.18", color=COL["wt_mid"], lw=0.8), zorder=3)
    ax4.annotate("", xy=(ox["wt_mlp_lag"] + 0.48, y_wt), xytext=(ox_wt8_opt - 0.42, y_wt),
                 arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.18", color=COL["wt"], lw=0.85), zorder=3)
    ax4.scatter([ox_wt8_unchunk], [y_wt], s=24, facecolors="white", edgecolors=COL["muted"], marker="o", linewidths=0.85, zorder=4)
    ax4.scatter([ox_wt8_opt], [y_wt], s=26, c=COL["wt_soft"], marker="o", edgecolors="white", linewidths=0.55, zorder=4)
    ax4.text(ox_wt8_unchunk - 0.25, y_wt - 0.45, f"+{ox_wt8_unchunk:.1f}%  (8$\\times$ mb=1)", fontsize=4.85,
             color=COL["muted"], ha="right", va="top", path_effects=HALO_SM)
    ax4.text(ox_wt8_opt - 0.3, y_wt + 0.40, f"+{ox_wt8_opt:.1f}%  (8$\\times$ c=2)", fontsize=4.85,
             color=COL["wt_mid"], ha="center", va="bottom", path_effects=HALO_SM)

    for k, col, mk in (("muont", COL["muont"], "D"), ("od_all", COL["od"], "^"), ("wt_mlp_lag", COL["wt"], "o")):
        ax4.plot([ox[k], ox[k]], [g_sel[k], g_pal[k]], color=col, lw=0.6, alpha=0.5, zorder=3)
        ax4.scatter([ox[k]], [g_pal[k]], s=30, facecolors="white", edgecolors=col, marker="D", linewidths=0.9, zorder=5)
    ax4.scatter([0.0], [0.0], s=30, c=COL["muon"], marker="o", edgecolors="white", linewidths=0.6, zorder=5)
    ax4.scatter([ox["muont"]], [g_sel["muont"]], s=32, c=COL["muont"], marker="o", edgecolors="white", linewidths=0.6, zorder=5)
    ax4.scatter([ox["od_all"]], [g_sel["od_all"]], s=30, c=COL["od"], marker="^", edgecolors="white", linewidths=0.6, zorder=5)
    ax4.scatter([ox["wt_mlp_lag"]], [y_wt], s=46, c=COL["wt"], marker="o", edgecolors="white", linewidths=0.7, zorder=6)

    ax4.text(0.35, -0.15, "Muon", fontsize=5.5, color=COL["muon"], ha="left", va="top")
    ax4.text(ox["muont"] - 0.3, g_pal["muont"] + 0.25, "MuonT", fontsize=5.6, fontweight="bold",
             color=COL["muont"], ha="right", va="bottom", path_effects=HALO_SM)
    ax4.text(ox["wt_mlp_lag"] - 0.3, g_pal["wt_mlp_lag"] + 0.25, "WT-MLP-Lag", fontsize=5.5, fontweight="bold",
             color=COL["wt"], ha="right", va="bottom", path_effects=HALO_SM)
    ax4.annotate("OD-All: off the frontier\n(costlier than MuonT, lower gain)",
                 xy=(ox["od_all"] + 0.1, g_sel["od_all"]), xytext=(ox["od_all"] + 1.2, 0.55),
                 fontsize=5.2, fontweight="bold", color=COL["od"], ha="left", va="center", linespacing=1.12,
                 arrowprops=dict(arrowstyle="->", color=COL["od"], lw=0.6, shrinkA=1, shrinkB=2),
                 path_effects=HALO)

    ax4.axhline(0.0, color=COL["spine"], linestyle=":", lw=0.6, zorder=1)
    ax4.axvline(0.0, color=COL["spine"], linestyle=":", lw=0.6, zorder=1)
    ax4.set_xlim(-1.2, 17.5)
    ax4.set_ylim(-1.0, 11.0)
    ax4.set_xlabel("Step overhead vs. Muon (%) [32$\\times$ / 8$\\times$A100]")
    ax4.set_ylabel("Quality gain vs. Muon ($\\times 10^{-3}$, $\\uparrow$)")
    panel_title(ax4, "(d)", "520M Pareto frontier ($d = 1024$)")
    ax4.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.01), fontsize=5.15, ncol=1,
               handlelength=1.15, handletextpad=0.3, labelspacing=0.14)

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.095, top=0.935, wspace=0.30, hspace=0.38)
    fig.savefig(OUT_DIR / "fig8_mechanism_and_pareto.pdf")
    plt.close(fig)


def main():
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    build_fig5()
    build_fig6()
    build_fig7()
    build_fig8()
    print("Successfully generated all 8 frontier-lab minimalist vector PDF figures in", OUT_DIR)


if __name__ == "__main__":
    main()
