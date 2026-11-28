# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python vector SVG generator for 130M loss-vs-training-time and step-time waterfall plots.

Matches the exact visual layout and styling of paper_v4/figs/training_loss_vs_time_130m_seed0.svg.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

TASK = Path(__file__).resolve().parents[1]
PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
CAMPAIGN_ROOT = PVC_ROOT / "campaigns" / "efficiency-20261128"
MUONH_ROOT = PVC_ROOT / "campaigns" / "muonh-20261127"
LR_BRACKET_ROOT = PVC_ROOT / "campaigns" / "lr-bracket-20261125"
CONFIRM_ROOT = PVC_ROOT / "campaigns" / "confirmation-20261121"


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def load_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    plain = sorted(run_dir.glob("metrics_from_*.jsonl"))
    if plain:
        for mf in plain:
            raw = mf.read_bytes().replace(b"\x00", b"")
            complete = raw[: raw.rfind(b"\n") + 1]
            for line in complete.splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    else:
        for mf in sorted(run_dir.glob("metrics_from_*.jsonl.gz")):
            with gzip.open(mf, "rt") as fh:
                for line in fh:
                    if line.strip():
                        rows.append(json.loads(line))
    by_step = {int(r["step"]): r for r in rows}
    return [by_step[s] for s in sorted(by_step)]


def extract_trajectories(run_dir: Path, ma_win: int = 51) -> dict:
    rows = load_rows(run_dir)
    pure_sec = np.array(
        [
            max(
                0.0,
                float(r.get("step_seconds", 0.0))
                - float(r.get("eval_seconds", 0.0))
                - float(r.get("checkpoint_seconds", 0.0)),
            )
            for r in rows
        ],
        dtype=np.float64,
    )
    cum_min = np.cumsum(pure_sec) / 60.0
    train_loss = np.array([float(r["train_loss"]) for r in rows], dtype=np.float64)
    csum = np.cumsum(np.insert(train_loss, 0, 0.0))
    ma = np.empty_like(train_loss)
    half = ma_win // 2
    for i in range(len(train_loss)):
        lo = max(0, i - half)
        hi = min(len(train_loss), i + half + 1)
        ma[i] = (csum[hi] - csum[lo]) / (hi - lo)

    eval_t, eval_sel, eval_c4, eval_step = [], [], [], []
    for idx, r in enumerate(rows):
        if "selection_loss" in r and r["selection_loss"] is not None:
            eval_step.append(int(r["step"]))
            eval_t.append(float(cum_min[idx]))
            eval_sel.append(float(r["selection_loss"]))
            eval_c4.append(float(r["c4_loss"]))
    return {
        "cum_min": cum_min,
        "train_ma": ma,
        "eval_step": np.array(eval_step),
        "eval_min": np.array(eval_t),
        "eval_sel": np.array(eval_sel),
        "eval_c4": np.array(eval_c4),
        "final_min": float(cum_min[-1]),
        "final_sel": float(eval_sel[-1]),
        "final_c4": float(eval_c4[-1]),
    }


def _draw_axes(
    lines: list[str],
    *,
    clip_id: str,
    px: float,
    py: float,
    pw: float,
    ph: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xticks: list[float],
    yticks: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    yfmt: str = "{:.2f}",
    xfmt: str = "{:.0f}",
):
    def sx(v: float) -> float:
        return px + (v - xmin) / (xmax - xmin) * pw

    def sy(v: float) -> float:
        return py + ph - (v - ymin) / (ymax - ymin) * ph

    lines.append(f'<defs><clipPath id="{clip_id}"><rect x="{px}" y="{py}" width="{pw}" height="{ph}"/></clipPath></defs>')
    for xt in xticks:
        x = sx(xt)
        lines.append(f'<line x1="{x:.1f}" y1="{py}" x2="{x:.1f}" y2="{py + ph}" stroke="#e5e7eb" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{x:.1f}" y="{py + ph + 18}" text-anchor="middle" font-size="11" fill="#374151">{esc(xfmt.format(xt))}</text>')
    for yt in yticks:
        y = sy(yt)
        lines.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px + pw}" y2="{y:.1f}" stroke="#e5e7eb" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{px - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#374151">{esc(yfmt.format(yt))}</text>')
    lines.append(f'<line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" stroke="#111827" stroke-width="1.2"/>')
    lines.append(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" stroke="#111827" stroke-width="1.2"/>')
    lines.append(f'<text x="{px + pw / 2:.1f}" y="{py - 10}" text-anchor="middle" font-size="13" font-weight="bold" fill="#111827">{esc(title)}</text>')
    lines.append(f'<text x="{px + pw / 2:.1f}" y="{py + ph + 40}" text-anchor="middle" font-size="12" fill="#111827">{esc(xlabel)}</text>')
    cy = py + ph / 2
    cx = px - 48
    lines.append(f'<text x="{cx}" y="{cy:.1f}" text-anchor="middle" font-size="12" fill="#111827" transform="rotate(-90 {cx} {cy:.1f})">{esc(ylabel)}</text>')
    return sx, sy


def _polyline(xs, ys, sx, sy, *, color: str, width: float = 2.2, dash: str = "", opacity: float = 1.0) -> str:
    pts = " ".join(f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in zip(xs, ys, strict=True))
    d_attr = f' stroke-dasharray="{dash}"' if dash else ""
    o_attr = f' opacity="{opacity:.2f}"' if opacity < 1.0 else ""
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}"{d_attr}{o_attr} points="{pts}"/>'


def plot_seed0_svg(out_path: Path, extra_runs: list[tuple[str, Path, str, str, float]]) -> None:
    base_series = [
        ("Muon", PVC_ROOT / "runs" / "wave0" / "muon", "#555555", "6,3", 2.0),
        ("MuonH", MUONH_ROOT / "runs" / "muonh_130m_1x_seed0_lr0p0113", "#D97706", "5,2.5", 2.1),
        ("MuonT", CONFIRM_ROOT / "runs" / "wave1a" / "muont_130m_1x_seed0_lr0p016", "#1B4F9C", "8,3,2,3", 2.0),
        ("Muon2-ExpDiag (v4)", CONFIRM_ROOT / "runs" / "wave1a" / "wt_mlp_lag_130m_1x_seed0_lr0p016", "#0E7C7B", "2.5,2.5", 2.1),
    ]
    loaded = []
    for label, path, color, dash, lw in [*base_series, *extra_runs]:
        if path.is_dir():
            loaded.append((label, extract_trajectories(path), color, dash, lw))

    W, H = 1320, 500
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="background:#ffffff;font-family:DejaVu Sans,Arial,sans-serif;">',
        f'<text x="{W / 2:.0f}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#111111">130M Seed 0 (20 TPP, 2.68B tokens): Training &amp; Held-Out Loss vs. Pure Training Time (8×A100)</text>',
    ]

    # Panel (a): Full run
    sx1, sy1 = _draw_axes(
        svg,
        clip_id="clip_a",
        px=78,
        py=68,
        pw=535,
        ph=365,
        xmin=0.0,
        xmax=104.0,
        ymin=3.10,
        ymax=5.10,
        xticks=[0, 20, 40, 60, 80, 100],
        yticks=[3.10, 3.50, 3.90, 4.30, 4.70, 5.10],
        title="(a) Full Run: Training Loss (51-step MA) vs. Time",
        xlabel="Cumulative Pure Training Time (minutes)",
        ylabel="Cross-Entropy Loss (nats)",
    )
    svg.append('<g clip-path="url(#clip_a)">')
    for _, tr, color, dash, lw in loaded:
        idx = np.unique(np.append(np.arange(5, len(tr["cum_min"]), 8), len(tr["cum_min"]) - 1))
        svg.append(_polyline(tr["cum_min"][idx], tr["train_ma"][idx], sx1, sy1, color=color, width=lw, dash=dash))
    svg.append("</g>")

    # Legend (a)
    lh1 = 16 + 20 * len(loaded)
    svg.append(f'<rect x="308" y="78" width="295" height="{lh1}" rx="4" fill="#ffffff" fill-opacity="0.93" stroke="#d1d5db"/>')
    for i, (label, tr, color, dash, lw) in enumerate(loaded):
        ly = 94 + i * 20
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="318" y1="{ly}" x2="344" y2="{ly}" stroke="{color}" stroke-width="{lw}"{d_attr}/>')
        fw = ' font-weight="bold"' if "Ours" in label else ""
        svg.append(f'<text x="352" y="{ly + 4}" font-size="10.0"{fw} fill="#111827">{esc(f"{label} ({tr["final_min"]:.1f}m)")}</text>')

    # Panel (b): Second-half zoom
    sx2, sy2 = _draw_axes(
        svg,
        clip_id="clip_b",
        px=735,
        py=68,
        pw=555,
        ph=365,
        xmin=45.0,
        xmax=103.0,
        ymin=3.12,
        ymax=3.58,
        xticks=[50, 60, 70, 80, 90, 100],
        yticks=[3.12, 3.20, 3.28, 3.36, 3.44, 3.52, 3.58],
        title="(b) Second-Half Zoom: Train Loss & Held-Out Selection vs. Time",
        xlabel="Cumulative Pure Training Time (minutes)",
        ylabel="Cross-Entropy Loss (nats)",
    )
    svg.append('<g clip-path="url(#clip_b)">')
    muonh_tr = next((tr for lbl, tr, *_ in loaded if lbl == "MuonH"), None)
    if muonh_tr is not None:
        xmh = sx2(muonh_tr["final_min"])
        ymh = sy2(muonh_tr["final_sel"])
        svg.append(f'<line x1="{xmh:.1f}" y1="68" x2="{xmh:.1f}" y2="433" stroke="#D97706" stroke-width="1.1" stroke-dasharray="3,3" opacity="0.7"/>')
        svg.append(f'<line x1="735" y1="{ymh:.1f}" x2="1290" y2="{ymh:.1f}" stroke="#D97706" stroke-width="1.0" stroke-dasharray="2,3" opacity="0.6"/>')

    for _, tr, color, dash, lw in loaded:
        mz = np.where(tr["cum_min"] >= 44.0)[0][::4]
        if len(mz) == 0 or mz[-1] != len(tr["cum_min"]) - 1:
            mz = np.append(mz, len(tr["cum_min"]) - 1)
        svg.append(_polyline(tr["cum_min"][mz], tr["train_ma"][mz], sx2, sy2, color=color, width=lw, dash=dash, opacity=0.75))
        em = tr["eval_min"] >= 44.0
        svg.append(_polyline(tr["eval_min"][em], tr["eval_sel"][em], sx2, sy2, color=color, width=1.3 if dash else 2.0, dash="2,2" if dash else "", opacity=0.92))
        for ex, ey in zip(tr["eval_min"][em], tr["eval_sel"][em], strict=True):
            r = 3.8 if not dash else 3.1
            svg.append(f'<circle cx="{sx2(float(ex)):.1f}" cy="{sy2(float(ey)):.1f}" r="{r}" fill="{color}" stroke="#ffffff" stroke-width="0.7"/>')
    svg.append("</g>")

    # Legend (b)
    lh2 = 34 + 20 * len(loaded)
    svg.append(f'<rect x="936" y="78" width="344" height="{lh2}" rx="4" fill="#ffffff" fill-opacity="0.93" stroke="#d1d5db"/>')
    for i, (label, tr, color, dash, lw) in enumerate(loaded):
        ly = 94 + i * 20
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="946" y1="{ly}" x2="972" y2="{ly}" stroke="{color}" stroke-width="{lw}"{d_attr}/>')
        svg.append(f'<circle cx="959" cy="{ly}" r="3.2" fill="{color}" stroke="#ffffff" stroke-width="0.6"/>')
        fw = ' font-weight="bold"' if "Ours" in label else ""
        svg.append(f'<text x="980" y="{ly + 4}" font-size="9.8"{fw} fill="#111827">{esc(f"{label} (sel {tr["final_sel"]:.4f}, {tr["final_min"]:.1f}m)")}</text>')
    svg.append(f'<text x="946" y="{94 + len(loaded) * 20 + 4}" font-size="9.5" fill="#4b5563">Lines: 51-step train MA | Dots: held-out loss</text>')
    svg.append("</svg>\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg))


def plot_multiseed_svg(out_path: Path, winner_specs: list[tuple[str, str, str, str, str]]) -> None:
    """Plot Seeds 1-3 mean held-out loss & paired delta vs Muon over wall-clock time."""
    groups = [
        ("Muon", [CONFIRM_ROOT / "runs" / f"wave2{w}" / f"muon_130m_1x_seed{s}" for s, w in ((1, "a"), (2, "b"), (3, "c"))], "#555555", "6,3", 1.9),
        ("MuonH", [MUONH_ROOT / "runs" / f"muonh_130m_1x_seed{s}_lr0p0113" for s in (1, 2, 3)], "#D97706", "5,2.5", 2.0),
        ("MuonT", [CONFIRM_ROOT / "runs" / f"wave2{w}" / f"muont_130m_1x_seed{s}" for s, w in ((1, "a"), (2, "b"), (3, "c"))], "#1B4F9C", "8,3,2,3", 1.9),
        ("Muon2-ExpDiag (v4)", [LR_BRACKET_ROOT / "runs" / f"wt_mlp_lag_130m_1x_seed{s}_lr0p016" for s in (1, 2, 3)], "#0E7C7B", "2.5,2.5", 2.0),
    ]
    for prefix, lr_tag, label, color, dash in winner_specs:
        paths = [CAMPAIGN_ROOT / "runs" / f"{prefix}_s{s}_lr{lr_tag}" for s in (1, 2, 3)]
        if all(p.is_dir() and (p / "run.json").is_file() and json.loads((p / "run.json").read_text()).get("status") == "complete" for p in paths):
            groups.append((label, paths, color, dash, 2.4 if not dash else 2.0))

    W, H = 1320, 500
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="background:#ffffff;font-family:DejaVu Sans,Arial,sans-serif;">',
        f'<text x="{W / 2:.0f}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#111111">130M Paired Fresh Seeds 1–3 (20 TPP, 2.68B tokens): Held-Out Loss vs. Pure Training Time (8×A100)</text>',
    ]
    sx1, sy1 = _draw_axes(
        svg,
        clip_id="clip_ms_a",
        px=78,
        py=68,
        pw=535,
        ph=365,
        xmin=40.0,
        xmax=103.0,
        ymin=3.17,
        ymax=3.49,
        xticks=[40, 55, 70, 85, 100],
        yticks=[3.18, 3.24, 3.30, 3.36, 3.42, 3.48],
        title="(a) Seeds 1–3 Mean Held-Out Selection Loss vs. Time",
        xlabel="Mean Cumulative Pure Training Time (minutes)",
        ylabel="Held-Out Selection Loss (nats)",
    )
    sx2, sy2 = _draw_axes(
        svg,
        clip_id="clip_ms_b",
        px=735,
        py=68,
        pw=555,
        ph=365,
        xmin=40.0,
        xmax=103.0,
        ymin=-52.0,
        ymax=6.0,
        xticks=[40, 55, 70, 85, 100],
        yticks=[-50, -40, -30, -20, -10, 0, 5],
        title="(b) Paired Selection Δ vs. Same-Step Muon (×10⁻³ nats)",
        xlabel="Mean Cumulative Pure Training Time (minutes)",
        ylabel="Paired Δ Loss vs. Muon (10⁻³ nats, lower is better)",
        yfmt="{:+.0f}",
    )

    loaded_ms = []
    muon_sels = None
    for label, paths, color, dash, lw in groups:
        if not all(p.is_dir() for p in paths):
            continue
        trs = [extract_trajectories(p) for p in paths]
        mean_min = np.mean([t["eval_min"] for t in trs], axis=0)
        sels = np.stack([t["eval_sel"] for t in trs], axis=0)
        if label == "Muon":
            muon_sels = sels
        loaded_ms.append((label, mean_min, sels, color, dash, lw))

    svg.append('<g clip-path="url(#clip_ms_a)">')
    for _, mean_min, sels, color, dash, lw in loaded_ms:
        m = mean_min >= 39.0
        mean_sel = np.mean(sels, axis=0)
        svg.append(_polyline(mean_min[m], mean_sel[m], sx1, sy1, color=color, width=lw, dash=dash))
        for ex, ey in zip(mean_min[m], mean_sel[m], strict=True):
            svg.append(f'<circle cx="{sx1(float(ex)):.1f}" cy="{sy1(float(ey)):.1f}" r="3.3" fill="{color}" stroke="#ffffff" stroke-width="0.6"/>')
    svg.append("</g>")

    lh1 = 16 + 21 * len(loaded_ms)
    svg.append(f'<rect x="293" y="78" width="310" height="{lh1}" rx="4" fill="#ffffff" fill-opacity="0.93" stroke="#d1d5db"/>')
    for i, (label, mean_min, sels, color, dash, lw) in enumerate(loaded_ms):
        ly = 95 + i * 21
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="303" y1="{ly}" x2="329" y2="{ly}" stroke="{color}" stroke-width="{lw}"{d_attr}/>')
        fw = ' font-weight="bold"' if "Ours" in label else ""
        msel = float(np.mean(sels[:, -1]))
        svg.append(f'<text x="337" y="{ly + 4}" font-size="10.0"{fw} fill="#111827">{esc(f"{label}: {msel:.5f} ({mean_min[-1]:.1f}m)")}</text>')

    svg.append('<g clip-path="url(#clip_ms_b)">')
    y0 = sy2(0.0)
    svg.append(f'<line x1="735" y1="{y0:.1f}" x2="1290" y2="{y0:.1f}" stroke="#555555" stroke-width="1.1" stroke-dasharray="4,3"/>')
    non_muon = []
    for label, mean_min, sels, color, dash, lw in loaded_ms:
        if label == "Muon" or muon_sels is None:
            continue
        diffs = (sels - muon_sels) * 1000.0
        md = np.mean(diffs, axis=0)
        sd = np.std(diffs, axis=0, ddof=1)
        m = mean_min >= 39.0
        svg.append(_polyline(mean_min[m], md[m], sx2, sy2, color=color, width=lw, dash=dash))
        for ex, ey in zip(mean_min[m], md[m], strict=True):
            svg.append(f'<circle cx="{sx2(float(ex)):.1f}" cy="{sy2(float(ey)):.1f}" r="3.3" fill="{color}" stroke="#ffffff" stroke-width="0.6"/>')
        non_muon.append((label, mean_min, md, sd, color, dash, lw))
    svg.append("</g>")

    lh2 = 16 + 21 * len(non_muon)
    svg.append(f'<rect x="896" y="78" width="388" height="{lh2}" rx="4" fill="#ffffff" fill-opacity="0.93" stroke="#d1d5db"/>')
    for i, (label, mean_min, md, sd, color, dash, lw) in enumerate(non_muon):
        ly = 95 + i * 21
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="906" y1="{ly}" x2="932" y2="{ly}" stroke="{color}" stroke-width="{lw}"{d_attr}/>')
        fw = ' font-weight="bold"' if "Ours" in label else ""
        svg.append(
            f'<text x="940" y="{ly + 4}" font-size="9.7"{fw} fill="#111827">{esc(f"{label}: {md[-1]:+.2f} ± {sd[-1]:.2f} mnat ({mean_min[-1]:.1f}m)")}</text>'
        )

    svg.append("</svg>\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg))


def plot_waterfall_svg(out_path: Path, eff_json: Path) -> None:
    if not eff_json.is_file():
        return
    eff = json.loads(eff_json.read_text())
    b = eff["baselines"]
    c = eff["campaign_runs"]
    selected = [
        ("Muon2-ExpDiag (paper_v4)", b["expdiag_orig_seed0"], "#B91C1C"),
        ("MuonT Baseline (Input-Only)", b["muont_seed0"], "#2563EB"),
        ("Muon Baseline (1st-Order)", b["muon_seed0"], "#6B7280"),
        ("MuonH Baseline (Hyperball)", b["muonh_seed0"], "#D97706"),
        ("Muon2-ExpDiag-Lazy8 (ctrl)", c["expdiag_lazy8_ctrl_s0_lr0p016"], "#7C3AED"),
        ("Muon2-HeadDiag-Lazy12 (ctrl)", c["headdiag_qk_lazy12_ctrl_s0_lr0p016"], "#059669"),
        ("Muon2-HeadDiag-MetricHB-Lazy12 (5120)", c["headdiag_qk_metrichb_wd0p15_lazy12_ctrl_s0_lr0p0135"], "#DC2626"),
        ("Muon2-HeadDiag-MetricHB-Lazy12 (Iso-Time)", c["headdiag_qk_metrichb_wd0p15_lazy12_iso5030_ctrl_s0_lr0p0135"], "#991B1B"),
    ]

    W, H = 1040, 410
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="background:#ffffff;font-family:DejaVu Sans,Arial,sans-serif;">',
        f'<text x="{W / 2:.0f}" y="28" text-anchor="middle" font-size="15.5" font-weight="bold" fill="#111111">130M 8×A100 Control-Aligned Full-Run Step-Time &amp; Total Pure Training Time (compile_ns = False)</text>',
    ]
    px, py, pw, ph = 330, 58, 430, 290
    xmin, xmax = 88.0, 102.0

    def sx(v: float) -> float:
        return px + (v - xmin) / (xmax - xmin) * pw

    for xt in [90, 92, 94, 96, 98, 100]:
        x = sx(xt)
        svg.append(f'<line x1="{x:.1f}" y1="{py}" x2="{x:.1f}" y2="{py + ph}" stroke="#e5e7eb" stroke-dasharray="3,3"/>')
        svg.append(f'<text x="{x:.1f}" y="{py + ph + 18}" text-anchor="middle" font-size="11" fill="#374151">{xt}m</text>')

    xm = sx(b["muonh_seed0"]["pure_train_minutes"])
    svg.append(f'<line x1="{xm:.1f}" y1="{py}" x2="{xm:.1f}" y2="{py + ph}" stroke="#D97706" stroke-width="1.3" stroke-dasharray="4,3"/>')

    row_h = ph / len(selected)
    for idx, (lbl, rec, col) in enumerate(selected):
        cy = py + (idx + 0.5) * row_h
        bh = row_h * 0.58
        w_bar = sx(rec["pure_train_minutes"]) - px
        is_bold = "MetricHB" in lbl or "MuonH" in lbl
        svg.append(f'<text x="{px - 10}" y="{cy + 4:.1f}" text-anchor="end" font-size="11.0" font-weight="{"bold" if is_bold else "normal"}" fill="#111827">{esc(lbl)}</text>')
        svg.append(f'<rect x="{px}" y="{cy - bh / 2:.1f}" width="{w_bar:.1f}" height="{bh:.1f}" rx="3" fill="{col}" fill-opacity="0.82"/>')
        txt = f"{rec['pure_train_minutes']:.2f}m | med {rec['steady_median_step_s']:.4f}s | sel {rec['selection_loss']:.5f}"
        svg.append(f'<text x="{sx(rec["pure_train_minutes"]) + 8:.1f}" y="{cy + 4:.1f}" font-size="10.2" font-weight="{"bold" if "MetricHB" in lbl else "normal"}" fill="#111827">{esc(txt)}</text>')

    svg.append(f'<line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" stroke="#111827" stroke-width="1.2"/>')
    svg.append(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" stroke="#111827" stroke-width="1.2"/>')
    svg.append(f'<text x="{px + pw / 2:.1f}" y="{py + ph + 40}" text-anchor="middle" font-size="11.5" fill="#111827">Total Pure Training Time Across Full Run (minutes, 8×A100, Identical Compilation Controls)</text>')
    svg.append("</svg>\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg))


def render_svg_to_png(svg_path: Path, png_path: Path, *, scale: int = 2, supersample: int = 2) -> None:
    """High-DPI supersampled PIL rasterizer for the exact subset of SVG primitives emitted by this module."""
    font_reg = "/usr/share/fonts/dejavu/DejaVuSans.ttf"
    font_bold = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
    s = scale * supersample
    tree = ET.parse(svg_path)
    root = tree.getroot()
    W = int(float(root.attrib["width"]))
    H = int(float(root.attrib["height"]))
    canvas = Image.new("RGBA", (W * s, H * s), (255, 255, 255, 255))

    clips: dict[str, tuple[int, int, int, int]] = {}
    font_cache: dict[tuple[bool, int], ImageFont.FreeTypeFont] = {}

    def get_font(bold: bool, px_size: float) -> ImageFont.FreeTypeFont:
        key = (bold, max(8, int(round(px_size * s))))
        if key not in font_cache:
            font_cache[key] = ImageFont.truetype(font_bold if bold else font_reg, key[1])
        return font_cache[key]

    def rgba(hex_col: str | None, alpha_mult: float = 1.0) -> tuple[int, int, int, int] | None:
        if not hex_col or hex_col == "none":
            return None
        rgb = ImageColor.getrgb(hex_col)
        return (rgb[0], rgb[1], rgb[2], int(round(255 * max(0.0, min(1.0, alpha_mult)))))

    def draw_dashed_polyline(
        draw: ImageDraw.ImageDraw,
        pts: list[tuple[float, float]],
        color: tuple[int, int, int, int],
        width_px: int,
        dash_str: str | None,
    ) -> None:
        if len(pts) < 2:
            return
        if not dash_str:
            draw.line(pts, fill=color, width=width_px, joint="curve")
            return
        pattern = [float(x) * s for x in dash_str.split(",") if x.strip()]
        if not pattern:
            draw.line(pts, fill=color, width=width_px, joint="curve")
            return
        pat_idx = 0
        pat_rem = pattern[0]
        pen_down = True
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:], strict=True):
            dx, dy = x1 - x0, y1 - y0
            seg_len = math.hypot(dx, dy)
            if seg_len <= 1e-6:
                continue
            ux, uy = dx / seg_len, dy / seg_len
            pos = 0.0
            while pos < seg_len:
                step = min(pat_rem, seg_len - pos)
                if pen_down and step > 0:
                    ax, ay = x0 + ux * pos, y0 + uy * pos
                    bx, by = x0 + ux * (pos + step), y0 + uy * (pos + step)
                    draw.line([(ax, ay), (bx, by)], fill=color, width=width_px)
                pos += step
                pat_rem -= step
                if pat_rem <= 1e-6:
                    pat_idx = (pat_idx + 1) % len(pattern)
                    pat_rem = pattern[pat_idx]
                    pen_down = not pen_down

    def render_elem(el: ET.Element, target: Image.Image) -> None:
        tag = el.tag.split("}")[-1]
        if tag == "defs":
            for cp in el:
                if cp.tag.split("}")[-1] == "clipPath":
                    cid = cp.attrib["id"]
                    r = cp[0]
                    rx, ry = int(round(float(r.attrib["x"]) * s)), int(round(float(r.attrib["y"]) * s))
                    rw, rh = int(round(float(r.attrib["width"]) * s)), int(round(float(r.attrib["height"]) * s))
                    clips[cid] = (rx, ry, rx + rw, ry + rh)
            return

        if tag == "g":
            cp_attr = el.attrib.get("clip-path", "")
            m = re.match(r"url\(#([^)]+)\)", cp_attr)
            layer = Image.new("RGBA", target.size, (0, 0, 0, 0))
            for child in el:
                render_elem(child, layer)
            if m and m.group(1) in clips:
                box = clips[m.group(1)]
                cropped = Image.new("RGBA", target.size, (0, 0, 0, 0))
                cropped.paste(layer.crop(box), (box[0], box[1]))
                target.alpha_composite(cropped)
            else:
                target.alpha_composite(layer)
            return

        opacity = float(el.attrib.get("opacity", "1.0"))
        fill_op = float(el.attrib.get("fill-opacity", "1.0"))
        needs_alpha = opacity < 0.999 or fill_op < 0.999
        layer = Image.new("RGBA", target.size, (0, 0, 0, 0)) if needs_alpha else target
        draw = ImageDraw.Draw(layer)

        if tag == "line":
            x1, y1 = float(el.attrib["x1"]) * s, float(el.attrib["y1"]) * s
            x2, y2 = float(el.attrib["x2"]) * s, float(el.attrib["y2"]) * s
            col = rgba(el.attrib.get("stroke", "#000000"), opacity)
            lw = max(1, int(round(float(el.attrib.get("stroke-width", "1.0")) * s)))
            if col:
                draw_dashed_polyline(draw, [(x1, y1), (x2, y2)], col, lw, el.attrib.get("stroke-dasharray"))
            if needs_alpha:
                target.alpha_composite(layer)

        elif tag == "polyline":
            raw_pts = el.attrib.get("points", "").split()
            pts = []
            for tok in raw_pts:
                px_s, py_s = tok.split(",")
                pts.append((float(px_s) * s, float(py_s) * s))
            col = rgba(el.attrib.get("stroke", "#000000"), opacity)
            lw = max(1, int(round(float(el.attrib.get("stroke-width", "1.0")) * s)))
            if col and pts:
                draw_dashed_polyline(draw, pts, col, lw, el.attrib.get("stroke-dasharray"))
            if needs_alpha:
                target.alpha_composite(layer)

        elif tag == "rect":
            rx = float(el.attrib["x"]) * s
            ry = float(el.attrib["y"]) * s
            rw = float(el.attrib["width"]) * s
            rh = float(el.attrib["height"]) * s
            rad = float(el.attrib.get("rx", "0")) * s
            fcol = rgba(el.attrib.get("fill"), fill_op * opacity)
            scol = rgba(el.attrib.get("stroke"), opacity)
            lw = max(1, int(round(float(el.attrib.get("stroke-width", "1.0")) * s))) if scol else 0
            draw.rounded_rectangle([rx, ry, rx + rw, ry + rh], radius=rad, fill=fcol, outline=scol, width=lw)
            if needs_alpha:
                target.alpha_composite(layer)

        elif tag == "circle":
            cx, cy = float(el.attrib["cx"]) * s, float(el.attrib["cy"]) * s
            r = float(el.attrib["r"]) * s
            fcol = rgba(el.attrib.get("fill"), opacity)
            scol = rgba(el.attrib.get("stroke"), opacity)
            lw = max(1, int(round(float(el.attrib.get("stroke-width", "1.0")) * s))) if scol else 0
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fcol, outline=scol, width=lw)
            if needs_alpha:
                target.alpha_composite(layer)

        elif tag == "text":
            txt = "".join(el.itertext())
            if not txt:
                return
            tx = float(el.attrib["x"]) * s
            ty = float(el.attrib["y"]) * s
            fsize = float(el.attrib.get("font-size", "11"))
            bold = el.attrib.get("font-weight") == "bold"
            font = get_font(bold, fsize)
            fcol = rgba(el.attrib.get("fill", "#111827"), opacity)
            anchor_map = {"start": "ls", "middle": "ms", "end": "rs"}
            pil_anchor = anchor_map.get(el.attrib.get("text-anchor", "start"), "ls")
            tr_attr = el.attrib.get("transform", "")
            rot_m = re.match(r"rotate\((-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\)", tr_attr)
            if rot_m:
                angle = -float(rot_m.group(1))
                bbox = font.getbbox(txt)
                tw = (bbox[2] - bbox[0]) + 40 * s
                th = (bbox[3] - bbox[1]) + 40 * s
                patch = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
                pdraw = ImageDraw.Draw(patch)
                pdraw.text((tw / 2, th / 2), txt, font=font, fill=fcol, anchor="mm")
                rot_patch = patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
                ox = int(round(tx - rot_patch.width / 2))
                oy = int(round(ty - rot_patch.height / 2))
                target.alpha_composite(rot_patch, (ox, oy))
            else:
                draw.text((tx, ty), txt, font=font, fill=fcol, anchor=pil_anchor)
                if needs_alpha:
                    target.alpha_composite(layer)

    for child in root:
        render_elem(child, canvas)

    out_img = canvas.resize((W * scale, H * scale), resample=Image.Resampling.LANCZOS).convert("RGB")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(png_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=TASK / "results" / "figs")
    args = parser.parse_args()

    runs_root = CAMPAIGN_ROOT / "runs"
    extra = []
    # Strictly plot ONLY control-aligned (compile_ns=False) runs against Muon, MuonH, MuonT, and Muon2-ExpDiag (v4)
    palette = [
        ("headdiag_qk_metrichb_wd0p15_lazy12_ctrl_s0_lr0p0135", "HeadQK-MetricHB-Lazy12 (Ours)", "#DC2626", "", 2.5),
        ("headdiag_qk_metrichb_wd0p15_lazy12_iso5030_ctrl_s0_lr0p0135", "HeadQK-MetricHB-Lazy12 (Iso-Time)", "#991B1B", "4,2", 2.2),
        ("expdiag_lazy8_ctrl_s0_lr0p016", "ExpDiag-Lazy8", "#7C3AED", "", 2.1),
        ("headdiag_qk_lazy12_ctrl_s0_lr0p016", "HeadQK-Lazy12", "#059669", "6,2", 2.0),
    ]
    for stem, lbl, col, dash, lw in palette:
        p = runs_root / stem
        if p.is_dir() and (p / "run.json").is_file():
            meta = json.loads((p / "run.json").read_text())
            if meta.get("status") == "complete" and not meta.get("config", {}).get("optimizer", {}).get("compile_ns", False):
                extra.append((lbl, p, col, dash, lw))

    plot_seed0_svg(args.out_dir / "training_loss_vs_time_130m_seed0_eff.svg", extra)
    plot_waterfall_svg(args.out_dir / "step_time_waterfall_130m_eff.svg", TASK / "results" / "eff_summary.json")
    plot_multiseed_svg(
        args.out_dir / "training_loss_vs_time_130m_seeds123_eff.svg",
        [
            ("headdiag_qk_metrichb_wd0p15_lazy12_ctrl", "0p0135", "HeadQK-MetricHB-Lazy12 (Ours)", "#DC2626", ""),
            ("expdiag_lazy8_ctrl", "0p016", "ExpDiag-Lazy8", "#7C3AED", ""),
            ("headdiag_qk_lazy12_ctrl", "0p016", "HeadQK-Lazy12", "#059669", "6,2"),
        ],
    )
    for stem in (
        "training_loss_vs_time_130m_seed0_eff",
        "step_time_waterfall_130m_eff",
        "training_loss_vs_time_130m_seeds123_eff",
    ):
        svg_p = args.out_dir / f"{stem}.svg"
        if svg_p.is_file():
            render_svg_to_png(svg_p, args.out_dir / f"{stem}.png", scale=2, supersample=2)
    print(f"Generated SVG and PNG plots in {args.out_dir}")


if __name__ == "__main__":
    main()
