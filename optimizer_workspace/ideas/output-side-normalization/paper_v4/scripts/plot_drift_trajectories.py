#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render three-method output-drift trajectories with Muon as the baseline."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
KINDS = ("q", "k", "v", "o", "down", "gate", "up")
COLORS = {"Muon": "#67717E", "MuonT": "#D18B3A", "WT": "#2459A6"}
LABELS = {"Muon": "Muon", "MuonT": "MuonT", "WT": "Muon2-ExpDiag"}
SCALES = (("130M", 250, 5120), ("520M", 2560, 10240))


def series(ax, records, scale, kinds, total_steps, interval, title, replicates=False):
    subset = [
        r for r in records if r["scale"] == scale and r["seed"] == 0 and r["kind"] in kinds and r["interval"] == interval
    ]
    for arm in ("Muon", "MuonT", "WT"):
        rows = [r for r in subset if r["arm"] == arm]
        if not rows:
            continue
        steps = sorted(set(r["step2"] for r in rows))
        values = [np.mean([r["drift16"] for r in rows if r["step2"] == step]) for step in steps]
        x = np.array(steps) / total_steps
        ax.plot(x, values, color=COLORS[arm], lw=1.5, marker="o" if len(steps) < 5 else None, ms=3, label=LABELS[arm])
        if replicates:
            repvalues = np.array(
                [
                    [
                        np.mean([r["drift16"] for r in rows if r["step2"] == step and r["replicate"] == rep])
                        for step in steps
                    ]
                    for rep in (0, 1)
                ]
            )
            ax.fill_between(x, repvalues.min(axis=0), repvalues.max(axis=0), color=COLORS[arm], alpha=0.12, linewidth=0)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
    ax.set_xlabel("Training fraction at interval end")
    ax.set_ylabel("Mean top-16 drift")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E4E9EF", lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="best")


def lookup(data, scale, interval, arm, seed=0):
    return {
        r["kind"]: r
        for r in data["probe_comparisons"]
        if r["scale"] == scale and r["seed"] == seed and r["interval"] == interval and r["k"] == 16 and r["arm"] == arm
    }


def table(data):
    records = data["probe_records"]
    rows = [
        r"\begin{table}[tbp]\centering\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{\textbf{Output-subspace drift with Muon as the baseline.} Fixed-data top-16 drift, "
        r"averaged equally over 32 layers, two probe sets, and 19 intervals of 250 steps at 130M or "
        r"three intervals of 2,560 steps at 520M. At 520M and in fresh 130M seeds 1--3 (one "
        r"2,560--5,120 interval per seed), all three methods share $\eta=0.016$; the densely saved "
        r"130M seed-0 trajectories use $0.016$ for Muon and $0.02263$ for MuonT and ExpDiag, so that column "
        r"does not isolate optimizer effects at a common rate. Reductions are $100(1-\overline D_{\rm method}/"
        r"\overline D_{\rm Muon})$; positive values mean lower drift than Muon. Seed summaries are "
        r"mean $\pm$ sample SD across three seeds. The all-role row averages drift before taking "
        r"ratios; absolute drift is not compared across model sizes. ExpDiag denotes Muon2-ExpDiag.}"
        r"\label{tab:output_drift}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X*{6}{>{\raggedleft\arraybackslash}X}@{}}\toprule",
        r"& \multicolumn{3}{c}{130M: absolute drift} & \multicolumn{3}{c}{520M: absolute drift} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Output & Muon & MuonT & ExpDiag & Muon & MuonT & ExpDiag \\\midrule",
    ]
    for kind in (*KINDS, "All roles"):
        row = [rf"\texttt{{{kind}}}" if kind in KINDS else kind]
        for scale, interval, _ in SCALES:
            for arm in ("Muon", "MuonT", "WT"):
                values = [
                    r["drift16"]
                    for r in records
                    if r["scale"] == scale
                    and r["seed"] == 0
                    and r["interval"] == interval
                    and r["arm"] == arm
                    and (kind == "All roles" or r["kind"] == kind)
                ]
                row.append(f"{np.mean(values):.3f}")
        if kind == "All roles":
            rows.append(r"\midrule")
        rows.append(" & ".join(row) + r" \\")
    rows.extend(
        [
            r"\bottomrule\end{tabularx}",
            r"\par\vspace{6pt}",
            r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xrrrrrr@{}}\toprule",
            r"& \multicolumn{2}{c}{130M: reduction (\%)} & \multicolumn{2}{c}{520M: reduction (\%)} "
            r"& \multicolumn{2}{c}{130M seeds 1--3: reduction (\%)} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"Output & MuonT & ExpDiag & MuonT & ExpDiag & MuonT & ExpDiag \\\midrule",
        ]
    )
    for kind in (*KINDS, "All roles"):
        row = [rf"\texttt{{{kind}}}" if kind in KINDS else kind]
        for scale, interval, _ in SCALES:
            for arm in ("MuonT", "WT"):
                selected = lookup(data, scale, interval, arm)
                vals = list(selected.values()) if kind == "All roles" else [selected[kind]]
                reduction = 100 * (
                    1 - np.mean([r["arm_mean"] for r in vals]) / np.mean([r["baseline_mean"] for r in vals])
                )
                row.append("0.0" if abs(reduction) < 0.05 else f"{reduction:+.1f}")
        for arm in ("MuonT", "WT"):
            reductions = []
            for seed in (1, 2, 3):
                selected = lookup(data, "130M", 2560, arm, seed)
                vals = list(selected.values()) if kind == "All roles" else [selected[kind]]
                reductions.append(
                    100 * (1 - np.mean([r["arm_mean"] for r in vals]) / np.mean([r["baseline_mean"] for r in vals]))
                )
            row.append(rf"${np.mean(reductions):+.1f}\pm{np.std(reductions, ddof=1):.1f}$")
        if kind == "All roles":
            rows.append(r"\midrule")
        rows.append(" & ".join(row) + r" \\")
    rows.extend([r"\bottomrule\end{tabularx}", r"\end{table}"])
    (PAPER / "tables/output_drift.tex").write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(gzip.decompress(args.summary.read_bytes()))
    assert not data["missing"] and data["baseline"] == "Muon"
    for style in ("regular", "bold"):
        path = Path(f"/opt/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre/texgyreheros-{style}.otf")
        if path.exists():
            fm.fontManager.addfont(path)
    mpl.rcParams.update(
        {
            "font.family": "TeX Gyre Heros",
            "font.size": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#263445",
            "text.color": "#263445",
            "axes.edgecolor": "#A6B2C0",
            "xtick.color": "#586777",
            "ytick.color": "#586777",
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.1), layout="constrained")
    series(axs[0, 0], data["probe_records"], "130M", KINDS, 5120, 250, "(a) 130M: fixed data, 250-step intervals", True)
    series(
        axs[0, 1], data["probe_records"], "520M", KINDS, 10240, 2560, "(b) 520M: fixed data, 2,560-step intervals", True
    )
    for ax, (scale, interval, _), panel in zip(axs[1], SCALES, ("c", "d"), strict=True):
        for j, arm in enumerate(("MuonT", "WT")):
            selected = lookup(data, scale, interval, arm)
            values = [selected[k]["reduction_pct"] for k in KINDS]
            ax.bar(np.arange(7) + (j - 0.5) * 0.36, values, width=0.34, color=COLORS[arm], label=LABELS[arm])
        ax.axhline(0, color=COLORS["Muon"], lw=0.8)
        ax.set_xticks(range(7), KINDS)
        ax.set_ylabel("Reduction from Muon (%)")
        ax.set_title(f"({panel}) {scale}: reduction from Muon", loc="left", fontsize=9, fontweight="bold")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(axis="y", color="#E4E9EF", lw=0.6)
        ax.set_axisbelow(True)
    fig.savefig(PAPER / "figs/fig9_output_drift.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figs/fig9_output_drift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(6.5, 3.8), layout="constrained")
    extent = [0.0]
    for ax, kind in zip(axs.flat, KINDS, strict=False):
        for scale, interval, total in SCALES:
            rows = [
                r
                for r in data["probe_records"]
                if r["scale"] == scale and r["seed"] == 0 and r["interval"] == interval and r["kind"] == kind
            ]
            steps = sorted({r["step2"] for r in rows})
            for arm in ("MuonT", "WT"):
                values = []
                for step in steps:
                    means = {
                        method: np.mean([r["drift16"] for r in rows if r["step2"] == step and r["arm"] == method])
                        for method in ("Muon", arm)
                    }
                    values.append(100 * (1 - means[arm] / means["Muon"]))
                extent.extend(values)
                ax.plot(
                    np.array(steps) / total,
                    values,
                    color=COLORS[arm],
                    marker="o" if scale == "520M" else None,
                    ls="--" if scale == "130M" else "-",
                    ms=2.5,
                    lw=1.2,
                    label=f"{scale} {'ExpDiag' if arm == 'WT' else arm}",
                )
        ax.axhline(0, color=COLORS["Muon"], lw=0.7)
        ax.set_title(kind, loc="left", fontweight="bold")
        ax.set_xlim(0, 1.02)
        ax.grid(axis="y", color="#E4E9EF", lw=0.6)
        ax.set_xlabel("Training fraction")
        ax.set_ylabel("Reduction from Muon (%)")
    for ax in list(axs.flat)[:7]:
        ax.set_ylim(10 * np.floor(min(extent) / 10) - 5, 10 * np.ceil(max(extent) / 10) + 5)
    axs.flat[-1].axis("off")
    handles, labels = axs.flat[0].get_legend_handles_labels()
    axs.flat[-1].legend(handles, labels, frameon=False, loc="upper left", fontsize=7)
    axs.flat[-1].text(
        0.05,
        0.45,
        "Zero = Muon\nPositive = lower drift\n\n130M: 250-step intervals\n520M: 2,560-step intervals\nx = interval end",
        va="top",
        fontsize=7,
    )
    fig.savefig(PAPER / "figs/fig10_drift_by_role.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figs/fig10_drift_by_role.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    table(data)


if __name__ == "__main__":
    main()
