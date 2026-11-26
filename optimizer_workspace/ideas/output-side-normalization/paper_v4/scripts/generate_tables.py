#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build manuscript tables from the recorded experiments."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
IDEA = PAPER.parent
RESULTS = IDEA / "paper" / "results"
RECORDS = IDEA / "experiment_records"
TABLES = PAPER / "tables"


def read(name: str):
    return json.loads((RESULTS / name).read_text())


def record(rel: str):
    with gzip.open(RECORDS / rel, "rt") as f:
        return [json.loads(line) for line in f]


CAMPAIGN = read("confirmation_campaign.json")
LR_BRACKET = read("lr_bracket_summary.json")
MECH = read("mechanism_summary.json")
ROT = json.loads((TABLES / "rotation_mechanism_metrics.json").read_text())
TIMING130 = read("measured_8gpu_timings.json")["methods"]
TIMING520 = read("measured_520m_timings.json")
SUMMARY = read("summary.json")


ORDER = [
    ("in_attn", "Input: attention", "input"),
    ("in_o", "Input: o", "input"),
    ("in_mlp", "Input: MLP", "input"),
    ("in_down", "Input: down", "input"),
    ("out_q", "Output: q", "output"),
    ("out_k", "Output: k", "output"),
    ("out_v", "Output: v", "output"),
    ("out_o", "Output: o (res.)", "output"),
    ("out_down", "Output: down (res.)", "output"),
    ("out_gate", "Output: gate (exp.)", "output"),
    ("out_up", "Output: up (exp.)", "output"),
]


DOMAINS = [
    ("c4_en", "C4-EN"),
    ("c4_100_domains", "C4 domains"),
    ("mc4", "mC4"),
    ("dolma-v1_5", "Dolma"),
    ("falcon-refinedweb", "RefinedWeb"),
    ("redpajama", "RedPajama"),
    ("m2d2_wikipedia_unsplit", "Wikipedia"),
    ("m2d2_s2orc_unsplit", "S2ORC"),
    ("dolma_100_programing_languages", "Programming"),
    ("wikitext_103", "WikiText-103"),
    ("ptb", "Penn Treebank"),
    ("dolma_100_subreddits", "Subreddits"),
    ("twitterAAE_HELM_fixed", "TwitterAAE"),
    ("manosphere_meta_sep", "Manosphere"),
    ("4chan_meta_sep", "4chan"),
    ("gab", "Gab"),
]


ABLATION_GROUPS = [
    (
        "Reference baselines and execution-path control",
        [
            ("Muon", "$I$ / $I$", "$I$ / $I$", "---", "wave0/muon", "muon", True),
            (
                "MuonT",
                "Dense / Dense",
                "$I$ / $I$",
                "---",
                "wave0/muont",
                "muont",
                True,
            ),
            (
                "Null-WT",
                "Dense / Dense",
                "$I$ / $I$ (solved)",
                "0",
                "wave3/null_mlpdiag",
                "null_mlpdiag",
                True,
            ),
            (
                "Null-Dense",
                "Dense / Dense",
                "$I$ / $I$ (solved)",
                "0",
                "wave1/null_all",
                "null_all",
                False,
            ),
            (
                "Null-OD",
                "Dense / Dense",
                "$I$ / $I$ (solved)",
                "0",
                "wave10/null_leftdiag_upstream_fused",
                "null_leftdiag_upstream_fused",
                False,
            ),
        ],
    ),
    (
        "Replacing or weakening input whitening",
        [
            (
                "Abl-BothDiag",
                "Diag / Diag",
                "Diag / Diag",
                "0",
                "wave1/inner_outer_all_diag",
                "inner_outer_all_diag",
                True,
            ),
            (
                "Abl-OutOnly",
                "$I$ / $I$",
                "Dense / Dense",
                "0",
                "wave1/left_all",
                "left_all",
                True,
            ),
            (
                "Abl-DownSwap",
                "Dense / $I$",
                r"$I$ (\texttt{down} Dense)",
                "0",
                "wave0/down_hybrid",
                "down_hybrid",
                False,
            ),
            (
                "WT-Sym",
                "Dense / Diag",
                "Dense / Diag",
                "0",
                "wave_sym_fastlag/inner_outer_sym_mlpdiag",
                "inner_outer_sym_mlpdiag",
                False,
            ),
            (
                "WT-Sym-Lag",
                "Dense / Diag",
                "Dense / Diag",
                "1",
                "wave_sym_fastlag/inner_outer_sym_mlpdiag_lag",
                "inner_outer_sym_mlpdiag_lag",
                True,
            ),
        ],
    ),
    (
        "Output projection role and inner/outer exponent placement",
        [
            (
                "Abl-DownOnly",
                "Dense / Dense",
                r"$\hat L_{\rm down}$ only",
                "0",
                "wave0/outer_left_down",
                "outer_left_down",
                False,
            ),
            (
                "OD-Outer",
                "Dense / Dense",
                r"$\hat L$ Diag / $\hat L$ Diag",
                "0",
                "wave10/outer_leftdiag_upstream_fused",
                "outer_leftdiag_upstream_fused",
                False,
            ),
            (
                "Abl-No2048",
                "Dense / Dense",
                "Dense / $I$",
                "0",
                "wave2/inner_outer_hidden",
                "inner_outer_hidden",
                True,
            ),
            (
                "Abl-OuterDense",
                "Dense / Dense",
                r"$\hat L$ Dense ($L=I$)",
                "0",
                "wave2/outer_left_all",
                "outer_left_all",
                True,
            ),
            (
                "OD-All",
                "Dense / Dense",
                "Diag / Diag",
                "0",
                "wave2/inner_outer_leftdiag",
                "inner_outer_leftdiag",
                True,
            ),
        ],
    ),
    (
        "Dense two-sided ceiling versus width thresholding and one-step lag",
        [
            (
                "DT-All",
                "Dense / Dense",
                "Dense / Dense",
                "0",
                "wave1/inner_outer_all",
                "inner_outer_all",
                True,
            ),
            (
                "DT-All-Lag",
                "Dense / Dense",
                "Dense / Dense",
                "1",
                "wave3/inner_outer_all_lag",
                "inner_outer_all_lag",
                True,
            ),
            (
                "WT-MLP",
                "Dense / Dense",
                "Dense / Diag",
                "0",
                "wave2/inner_outer_mlpdiag",
                "inner_outer_mlpdiag",
                True,
            ),
            (
                "WT-MLP-Lag",
                "Dense / Dense",
                "Dense / Diag",
                "1",
                "wave4/inner_outer_mlpdiag_lag",
                "inner_outer_mlpdiag_lag",
                True,
            ),
        ],
    ),
]


def audit_sets():
    a = MECH["motivation_trajectories"]
    b = MECH["motivation_null_all"]
    return [
        (a["muon"]["step_002500"], a["muon"]["grassmann_drift_2250_to_2500"], "Muon"),
        (
            b["step_002500"],
            b["grassmann_drift_250steps"]["step_002250_to_step_002500"],
            "MuonT",
        ),
        (
            a["wt_mlp_lag"]["step_002500"],
            a["wt_mlp_lag"]["grassmann_drift_2250_to_2500"],
            "WT-MLP-Lag",
        ),
    ]


def pal(name: str):
    return json.loads((RECORDS / "paloma_eval" / f"{name}.json").read_text())


def pal_loss(d: dict, key: str) -> float:
    v = d["scores"]["paloma/" + key]
    return float(v["loss"] if isinstance(v, dict) else v)


CAMPAIGN_RUNS = "campaigns/confirmation-20261121/runs"
LR520_RUNS = "campaigns/lr-520m-20261125/runs"


def run_curves(run: str) -> dict[str, np.ndarray]:
    """Held-out and C4 evaluation curves for one archived run, ordered by step."""
    if run.startswith("lr520m/"):
        rel_dir = f"{LR520_RUNS}/{run.split('/', 1)[1]}"
    else:
        rel_dir = f"{CAMPAIGN_RUNS}/{run}"
    rows = record(f"{rel_dir}/metrics_from_000000.jsonl.gz")
    curves = {}
    for field in ("selection_loss", "c4_loss"):
        pts = sorted((int(r["step"]), float(r[field])) for r in rows if r.get(field) is not None)
        curves[field] = np.array([v for _, v in pts])
    return curves


def curve_deltas(base_run: str, arm_run: str, n_late: int = 11) -> dict[str, float]:
    """Final and late-window differences (arm minus base) in nats, from the archived metrics."""
    base, arm = run_curves(base_run), run_curves(arm_run)
    sb, sa = base["selection_loss"], arm["selection_loss"]
    n = min(n_late, len(sb), len(sa))
    return {
        "d_sel": float(sa[-1] - sb[-1]),
        "d_late11": float(np.mean(sa[-n:] - sb[-n:])),
        "d_c4": float(arm["c4_loss"][-1] - base["c4_loss"][-1]),
    }


def pal_deltas(base: str, arm: str) -> list[float]:
    a, b = pal(arm), pal(base)
    return [(pal_loss(a, k) - pal_loss(b, k)) * 1000 for k, _ in DOMAINS] + [
        (a["paloma_macro_mean"] - b["paloma_macro_mean"]) * 1000
    ]


def write_table(
    name: str,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str] | str],
    spec: str | None = None,
    preamble_rows: list[str] | None = None,
    size: str = r"\small",
    colsep: str = "5.0pt",
    pos: str = "tbp",
):
    spec = spec or ("X" + "r" * (len(headers) - 1))
    spec_expanded = spec.replace("X", r">{\raggedright\arraybackslash}X")
    head = " & ".join(headers) + r" \\"
    start = [
        rf"\begin{{table}}[{pos}]\centering{size}",
        rf"\setlength{{\tabcolsep}}{{{colsep}}}",
        rf"\caption{{{caption}}}\label{{{label}}}",
        rf"\begin{{tabularx}}{{\textwidth}}{{@{{}}{spec_expanded}@{{}}}}",
        r"\toprule",
    ]
    if preamble_rows:
        start.extend(preamble_rows)
    start.append(head)
    start.append(r"\midrule")
    end = [r"\bottomrule\end{tabularx}", r"\end{table}"]
    body = [(" & ".join(row) + r" \\") if isinstance(row, list) else row for row in rows]
    (TABLES / f"{name}.tex").write_text("\n".join(start + body + end) + "\n")


def group(text: str, n: int, space: str = "4pt") -> str:
    return rf"\addlinespace[{space}]\multicolumn{{{n}}}{{@{{}}l}}{{\textit{{{text}}}}}\\"


def num(v: float, dec: int = 2, bold: bool = False) -> str:
    s = f"{v:+.{dec}f}"
    return rf"$\mathbf{{{s}}}$" if bold else f"${s}$"


def tint(row: list[str]) -> list[str]:
    out = list(row)
    out[0] = r"\rowcolor{rowtint}[0pt][0pt]" + out[0]
    return out


def build_diagnosis_tables():
    sets = {label: (mom, drift) for mom, drift, label in audit_sets()}

    rows: list[list[str] | str] = []
    current_side = None
    for k, disp, side in ORDER:
        if side != current_side:
            title = (
                r"Input activation moments $C_t=\E[xx^{\Tr}]$ (4 per block)"
                if side == "input"
                else r"Output cotangent moments $B_t=\E[\delta\delta^{\Tr}]$ (7 per block)"
            )
            rows.append(group(title, 8, space="2pt" if current_side is None else "5pt"))
            current_side = side
        elif k in {"out_o", "out_gate"}:
            rows.append(r"\addlinespace[2.5pt]")

        mt_m, mt_d = sets["MuonT"][0][k], sets["MuonT"][1][k]
        muon_d = sets["Muon"][1][k]
        wt_d = sets["WT-MLP-Lag"][1][k]
        da_all = [sets[lbl][0][k]["diag_aniso_pct"]["median"] for lbl in ["Muon", "MuonT", "WT-MLP-Lag"]]
        w = int(mt_m["width"])

        lo, hi = min(da_all), max(da_all)
        rng_s = f"{lo:.1f}--{hi:.1f}" if hi < 10.0 else f"{lo:.0f}--{hi:.0f}"
        row = [
            disp.replace("Input: ", "").replace("Output: ", ""),
            f"{w:,}",
            f"{mt_m['offdiag_corr_rms']['median']:.3f}",
            f"{mt_m['diag_aniso_pct']['median']:.1f} ({rng_s})",
            f"{mt_m['exp_var_L_pct']['median']:.1f}",
            f"{muon_d['drift_top16']['mean']:.3f}",
            f"{mt_d['drift_top16']['mean']:.3f}",
            f"{wt_d['drift_top16']['mean']:.3f}",
        ]
        if k in {"out_gate", "out_up"}:
            row = tint(row)
        rows.append(row)

    write_table(
        "diagnosis_summary",
        (
            r"\textbf{Moment geometry and subspace drift by projection role at 130M.} Static columns report "
            r"32-layer medians on the MuonT trajectory at step 2,500. For diagonal anisotropy share (DA), "
            r"parentheses give the range of these medians across Muon, MuonT, and WT-MLP-Lag. $\rho_{\rm rms}$ "
            r"is the RMS of off-diagonal uncentered correlations; root $R^2$ compares diagonal and dense "
            r"inverse roots. Drift compares the subspaces at steps 2,250 and 2,500, averaged over 32 layers "
            r"for each trajectory (WT denotes WT-MLP-Lag). WT uses a resumed trajectory; "
            r"uninterrupted-checkpoint comparisons are reported in \Cref{sec:output_drift}. "
            r"Tinted expansion outputs account for 96.2\% "
            r"of nominal dense output eigendecomposition work."
        ),
        "tab:diagnosis_summary",
        [
            "Moment",
            "$d$",
            r"$\rho_{\rm rms}$",
            "DA (\\%, 3-traj)",
            "Root $R^2$ (\\%)",
            "Muon",
            "MuonT",
            "WT",
        ],
        rows,
        spec="Xrrrrrrr",
        preamble_rows=[
            r"& & & \multicolumn{2}{c}{Diagonal fidelity} & \multicolumn{3}{c}{250-step $\mathrm{Drift}_{16}$} \\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
        ],
        size=r"\footnotesize",
        colsep="5.8pt",
        pos="tbp",
    )

    full_rows: list[list[str] | str] = []
    for t_idx, (mom, drift, label) in enumerate(audit_sets()):
        full_rows.append(group(label, 8, space="2pt" if t_idx == 0 else "4pt"))
        for k, disp, _ in ORDER:
            d = mom[k]
            clean_disp = disp.replace(" (res.)", "").replace(" (exp.)", "")
            full_rows.append(
                [
                    clean_disp,
                    f"{int(d['width']):,}",
                    f"{d['erank_pct']['median']:.1f}",
                    f"{d['diag_cv']['median']:.2f}",
                    f"{d['offdiag_corr_rms']['median']:.4f}",
                    f"{d['diag_aniso_pct']['median']:.1f}",
                    f"{d['exp_var_L_pct']['median']:.1f}",
                    f"{drift[k]['drift_top16']['mean']:.3f}",
                ]
            )
    write_table(
        "audit",
        (
            r"\textbf{Complete 352-matrix moment audit across three 130M trajectories.} Effective rank is a "
            r"percentage of width $d$; CV is diagonal coefficient of variation; $\rho_{\rm rms}$ is "
            r"off-diagonal correlation RMS; DA is diagonal anisotropy share (\%); root $R^2$ is diagonal "
            r"inverse-root explained variance (\%). Static metrics are 32-layer medians at step 2,500; drift "
            r"is the 32-layer mean over steps 2,250 to 2,500."
        ),
        "tab:diagnosis",
        [
            "Moment",
            "$d$",
            "Rank (\\%)",
            "CV",
            r"$\rho_{\rm rms}$",
            "DA (\\%)",
            "Root $R^2$",
            "Drift",
        ],
        full_rows,
        spec="Xrrrrrrr",
        size=r"\footnotesize",
        colsep="8.0pt",
    )


def build_method_and_protocol_tables():
    tax_rows: list[list[str] | str] = [
        ["Muon", "$I$", "$I$", "$I$", "$I$", "---", "0\\%"],
        ["MuonT", "Dense", "Dense", "$I$", "$I$", "---", "0\\%"],
        ["OD-All", "Dense", "Dense", "Diag", "Diag", "0", "0\\%"],
        ["WT-Sym-Lag", "Dense", "Diag", "Dense", "Diag", "1", "3.8\\%"],
        ["WT-MLP", "Dense", "Dense", "Dense", "Diag", "0", "3.8\\%"],
        tint(
            [
                "\\textbf{WT-MLP-Lag}",
                "\\textbf{Dense}",
                "\\textbf{Dense}",
                "\\textbf{Dense}",
                "\\textbf{Diag}",
                "\\textbf{1}",
                "\\textbf{3.8\\%}",
            ]
        ),
        ["DT-All", "Dense", "Dense", "Dense", "Dense", "0", "100\\%"],
        [
            "Null-WT",
            "Dense",
            "Dense",
            "$I$ (after solve)",
            "$I$ (after solve)",
            "0",
            "3.8\\%",
        ],
    ]
    write_table(
        "taxonomy",
        (
            r"\textbf{Factor assignment and nominal output $O(d^3)$ share across core methods.} Dense and "
            r"Diag denote the $(-1/2,-1/8)$ factor pairs in \Cref{eq:dense_roots,eq:diag_roots}. Splitting "
            r"both sides by width ($\dm$ versus $\dff=4\dm$) highlights the input--output asymmetry: "
            r"WT-MLP-Lag retains dense factors on the $\dff$-wide \texttt{down} input while diagonalizing "
            r"both $\dff$-wide expansion outputs. Fine-grained placement ablations appear in \Cref{tab:ablations}."
        ),
        "tab:taxonomy",
        [
            "Method",
            r"$\dm$ (\texttt{attn,o,mlp})",
            r"$\dff$ (\texttt{down})",
            r"$\dm$ (\texttt{q,k,v,o,down})",
            r"$\dff$ (\texttt{gate,up})",
            r"Lag $\tau$",
            r"Out $O(d^3)$",
        ],
        tax_rows,
        spec="Xcccccr",
        preamble_rows=[
            r"& \multicolumn{2}{c}{Input factors ($R,\hat R$)} & \multicolumn{2}{c}{Output factors ($L,\hat L$)} & & \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        ],
        colsep="4.5pt",
        pos="!htb",
    )

    reg_rows: list[list[str] | str] = [
        [
            "130M controls \\& LR bracket",
            "134M",
            "512 / 2,048",
            "2.68B (20)",
            "0.5M / 5,120",
            "8$\\times$A100",
            "0",
        ],
        [
            "130M confirmation",
            "134M",
            "512 / 2,048",
            "2.68B (20)",
            "0.5M / 5,120",
            "8$\\times$A100",
            "1, 2, 3",
        ],
        [
            "130M longer training",
            "134M",
            "512 / 2,048",
            "10.74B (80)",
            "0.5M / 20,480",
            "8$\\times$A100",
            "0",
        ],
        [
            "520M width transfer",
            "537M",
            "1,024 / 4,096",
            "10.74B (20)",
            "1.0M / 10,240",
            "32$\\times$A100",
            "0",
        ],
    ]
    write_table(
        "regimes",
        (
            r"\textbf{Pretraining regimes.} Params counts non-embedding parameters; TPP is tokens per "
            r"non-embedding parameter. Fresh-seed confirmation uses seeds 1--3 after learning-rate "
            r"selection and execution-path checks on seed 0."
        ),
        "tab:regimes",
        [
            "Regime",
            "Params",
            r"$\dm$ / $\dff$",
            "Tokens (TPP)",
            "Batch / Steps",
            "GPUs",
            "Seeds",
        ],
        reg_rows,
        spec="Xrrrrrc",
        colsep="5.8pt",
        pos="tbp",
    )

    runs = CAMPAIGN["runs"]
    s0 = LR_BRACKET["seed0_grid"]
    mt_ref = s0["muont"]["0p0226"]["selection_loss"]
    null_0226 = runs["wave1b/null_wt_130m_1x_seed0_lr0p0226"]["final_sel"]
    mt_short_full = SUMMARY["wave12/muont"]["sel_final"]
    od_short_full = SUMMARY["wave12/inner_outer_leftdiag"]["sel_final"]

    def _s0_cells(arm_key: str) -> list[str]:
        vals = [s0[arm_key][t]["selection_loss"] for t in ("0p0113", "0p016", "0p0226", "0p032")]
        best = int(np.argmin(vals))
        return [rf"\textbf{{{v:.4f}}}" if i == best else f"{v:.4f}" for i, v in enumerate(vals)]

    ctrl_rows: list[list[str] | str] = [
        [
            "Muon",
            *_s0_cells("muon"),
            num((s0["muon"]["0p0226"]["selection_loss"] - mt_ref) * 1000),
            "---",
        ],
        [
            "MuonT",
            *_s0_cells("muont"),
            "$+0.00$",
            rf"$0.0640 \to {mt_short_full:.4f}$",
        ],
        [
            "Null-WT (ctrl)",
            "---",
            "---",
            f"{null_0226:.4f}",
            "---",
            num((null_0226 - mt_ref) * 1000),
            "---",
        ],
        [
            "OD-All",
            *_s0_cells("od_all"),
            num((s0["od_all"]["0p0226"]["selection_loss"] - mt_ref) * 1000),
            rf"$0.0905 \to {od_short_full:.4f}$",
        ],
        tint(
            [
                "\\textbf{WT-MLP-Lag}",
                *_s0_cells("wt_mlp_lag"),
                num(
                    (s0["wt_mlp_lag"]["0p0226"]["selection_loss"] - mt_ref) * 1000,
                    bold=True,
                ),
                "---",
            ]
        ),
    ]
    write_table(
        "controls",
        (
            r"\textbf{Seed-0 learning-rate bracket, execution-path control, and short-schedule transfer at "
            r"130M (2.68B tokens).} Entries are final held-out losses in nats across the complete four-point "
            r"bracket (bold marks each row's full-budget minimum, $\eta^*=0.0160$ for every method); "
            r"$\Delta_{0.0226}$ is the paired difference from MuonT at $\eta=0.02263$ in millinats where "
            r"Null-WT is evaluated. The rightmost column transfers each checked arm's 320-step argmax "
            r"learning rate (\Cref{tab:lr_cliff}) to the full 5,120-step schedule."
        ),
        "tab:controls",
        [
            "Method",
            r"$0.0113$",
            r"$0.0160$",
            r"$0.0226$",
            r"$0.0320$",
            r"$\Delta_{0.0226}$ (mnats)",
            r"320-step $\eta^* \to$ full",
        ],
        ctrl_rows,
        spec="Xrrrrrr",
        preamble_rows=[
            r"& \multicolumn{4}{c}{Full-budget held-out loss $\Lsel$ (nats)} & & \\",
            r"\cmidrule(lr){2-5}",
        ],
        colsep="5.2pt",
        pos="tbp",
    )


def build_main_scaling_table():
    rows: list[list[str] | str] = []
    s123 = LR_BRACKET["seeds123_summaries"]
    muon123 = s123["muon@0p016"]

    rows.append(
        group(
            "130M confirmation: 20 TPP, 2.68B tokens, fresh seeds 1--3 "
            "(matched $\\eta^*=0.0160$, paired mean $\\pm$ SD)",
            8,
            "2pt",
        )
    )
    for arm, label, pal_suf in [
        ("muon", "Muon", ""),
        ("muont", "MuonT", ""),
        ("od_all", "OD-All", "_lr0p016"),
        ("wt_mlp_lag", "WT-MLP-Lag", "_lr0p016"),
    ]:
        d = s123[f"{arm}@0p016"]
        lr_s = f"{d['lr']:.5f}"
        sel_s = rf"\textbf{{{d['sel_mean']:.4f}}}" if arm == "wt_mlp_lag" else f"{d['sel_mean']:.4f}"
        if arm == "muon":
            vals = ["$+0.00$", "$+0.00$", "$+0.00$", "$+0.00$", "---"]
        else:
            is_wt = arm == "wt_mlp_lag"
            vals = []
            for key_s in ["sel_seeds", "late11_sel_seeds", "c4_seeds", "paloma_seeds"]:
                diffs = (np.array(d[key_s]) - np.array(muon123[key_s])) * 1000.0
                cell = f"{float(np.mean(diffs)):+.2f}\\pm{float(np.std(diffs, ddof=1)):.2f}"
                vals.append(rf"$\mathbf{{{cell}}}$" if is_wt else f"${cell}$")
            mean_dom = np.mean(
                [pal_deltas(f"muon_130m_1x_seed{s}", f"{arm}_130m_1x_seed{s}{pal_suf}") for s in [1, 2, 3]],
                axis=0,
            )
            wins = int(np.sum(mean_dom[:16] < 0))
            vals.append(rf"\textbf{{{wins}/16}}" if is_wt else f"{wins}/16")
        row = [
            f"\\textbf{{{label}}}" if arm == "wt_mlp_lag" else label,
            lr_s,
            sel_s,
            *vals,
        ]
        rows.append(tint(row) if arm == "wt_mlp_lag" else row)

    for kind, desc in [
        ("p3", "130M longer training: 80 TPP, 10.74B tokens, seed 0 (matched rate)"),
        (
            "p4",
            "520M width transfer: 20 TPP, 10.74B tokens, $\\dm=1{,}024$, seed 0 (matched $\\eta^*=0.0160$)",
        ),
    ]:
        rows.append(group(desc, 8, "5pt"))

        def run_key_for(arm: str, _kind: str = kind) -> str:
            if _kind == "p3":
                if arm in {"muon", "muont"}:
                    return f"wave3a/{arm}_130m_4x_seed0"
                return f"wave3b/{arm}_130m_4x_seed0_lr0p01056"
            if arm == "muon":
                return "wave4a/muon_520m_1x_seed0"
            return f"lr520m/{arm}_520m_1x_seed0_lr0p016"

        base_run = run_key_for("muon")
        for arm, label in [
            ("muon", "Muon"),
            ("muont", "MuonT"),
            ("od_all", "OD-All"),
            ("wt_mlp_lag", "WT-MLP-Lag"),
        ]:
            run_key = run_key_for(arm)
            r = CAMPAIGN["runs"][run_key]
            is_wt = arm == "wt_mlp_lag"
            sel_s = rf"\textbf{{{r['final_sel']:.4f}}}" if is_wt else f"{r['final_sel']:.4f}"
            if arm == "muon":
                vals = ["$+0.00$", "$+0.00$", "$+0.00$", "$+0.00$", "---"]
            else:
                cd = curve_deltas(base_run, run_key)
                dom = pal_deltas(base_run.rsplit("/", 1)[-1], run_key.rsplit("/", 1)[-1])
                wins = int(np.sum(np.array(dom[:16]) < 0))
                vals = [
                    num(cd["d_sel"] * 1000, bold=is_wt),
                    num(cd["d_late11"] * 1000, bold=is_wt),
                    num(cd["d_c4"] * 1000, bold=is_wt),
                    num(dom[16], bold=is_wt),
                    rf"\textbf{{{wins}/16}}" if is_wt else f"{wins}/16",
                ]
            row = [
                f"\\textbf{{{label}}}" if is_wt else label,
                f"{r['lr']:.5f}",
                sel_s,
                *vals,
            ]
            rows.append(tint(row) if is_wt else row)

    write_table(
        "main_results",
        (
            r"\textbf{Confirmatory and scaling results across fresh seeds, longer horizon, and greater width.} "
            r"Held-out reports final cross-entropy $\Lsel$ in nats; all $\Delta$ columns report differences "
            r"from same-regime Muon in millinats (negative favors the row method). MuonT applies input "
            r"preconditioning only, so the gap between the MuonT and WT-MLP-Lag rows isolates the output-side "
            r"contribution. Late-11 averages the 11 second-half checkpoints; Wins counts the 16 Paloma domains "
            r"improved over Muon. All four methods share their verified seed-0 optimum within each regime "
            r"($\eta^*=0.01600$ at 130M and 520M 20 TPP, and $0.01600\times4^{-0.3}=0.01056$ at 130M 80 TPP). "
            r"Complete absolute losses and $\eta=0.02263$ replicates appear in "
            r"\Cref{tab:absolute_results,tab:seeds}."
        ),
        "tab:main_scaling",
        [
            "Method",
            r"LR $\eta$",
            "Held-out",
            r"$\Delta$ Held-out",
            r"Late-11 $\Delta$",
            r"$\Delta$ C4",
            r"$\Delta$ Paloma",
            "Wins",
        ],
        rows,
        spec="Xrrrrrrr",
        preamble_rows=[
            r"& & (nats) & \multicolumn{4}{c}{Differences from Muon (millinats, $\downarrow$)} & Paloma \\",
            r"\cmidrule(lr){4-7}",
        ],
        colsep="2.2pt",
        pos="!htb",
    )


def build_analysis_tables():
    # 1. Main-body question-grouped ablation table + Appendix full provenance ledger
    main_rows: list[list[str] | str] = []
    full_rows: list[list[str] | str] = []
    for g_idx, (g_title, items) in enumerate(ABLATION_GROUPS):
        main_rows.append(group(g_title, 7, "2pt" if g_idx == 0 else "4pt"))
        full_rows.append(group(g_title, 9, "2pt" if g_idx == 0 else "4pt"))
        for label, in_s, out_s, lag_s, key, timing_key, in_main in items:
            d = SUMMARY[key]
            t = TIMING130[timing_key]["measured_benchmark"]
            is_wt_lag = label == "WT-MLP-Lag"
            is_wt_sync = label == "WT-MLP"
            wave_map = {
                "wave0": "W0",
                "wave1": "W1",
                "wave2": "W2",
                "wave3": "W3",
                "wave4": "W4",
                "wave10": "W10",
                "wave_sym_fastlag": "FastLag",
            }
            raw_wave = key.split("/")[0]
            wave_s = wave_map.get(raw_wave, raw_wave)
            disp = f"\\textbf{{{label}}}" if is_wt_lag else label

            if in_main:
                r_main = [
                    disp,
                    in_s,
                    out_s,
                    lag_s,
                    num(d["d_sel_late"] * 1000, bold=is_wt_lag or is_wt_sync),
                    num(d["d_c4_late"] * 1000, bold=is_wt_lag),
                    rf"${t['median_overhead_vs_muont_pct']:+.1f}\%$",
                ]
                main_rows.append(tint(r_main) if is_wt_lag else r_main)

            r_full = [
                disp,
                wave_s,
                in_s,
                out_s,
                num(d["d_sel_final"] * 1000, bold=is_wt_lag or is_wt_sync),
                num(d["d_sel_late"] * 1000, bold=is_wt_lag or is_wt_sync),
                num(d["d_c4_late"] * 1000, bold=is_wt_lag),
                num(d["d_train"] * 1000, bold=is_wt_lag),
                rf"${t['median_overhead_vs_muont_pct']:+.1f}\%$",
            ]
            full_rows.append(tint(r_full) if is_wt_lag else r_full)

    write_table(
        "ablations",
        (
            r"\textbf{Factor-placement, exponent, and scheduling ablations at 130M (seed 0, 2.68B tokens).} "
            r"Inputs and outputs list the $(\dm / \dff)$ operator assignment under the $(-1/2,-1/8)$ split "
            r"unless noted; Muon uses $\eta=0.016$ and all other arms use $\eta=0.02263$. Loss columns report "
            r"11-checkpoint second-half mean differences (millinats) relative to each run's same-wave MuonT "
            r"reference; Step $\Delta t$ is median step-time overhead in the unified 8-GPU benchmark "
            r"(\Cref{tab:bench19}). All 19 exploratory arms and wave references appear in "
            r"\Cref{tab:ablations_full}."
        ),
        "tab:ablations",
        [
            "Method",
            r"Inputs ($\dm / \dff$)",
            r"Outputs ($\dm / \dff$)",
            "Lag",
            r"Late $\Delta\Lsel$",
            r"Late $\Delta\Lc$",
            r"Step $\Delta t$",
        ],
        main_rows,
        spec="Xllcrrr",
        colsep="5.2pt",
        pos="!htb",
    )

    write_table(
        "ablations_full",
        (
            r"\textbf{Complete 19-arm placement and operator ledger at 130M (seed 0, 2.68B tokens).} "
            r"Cohort identifies the implementation wave providing each arm's paired MuonT reference (W0--W4, "
            r"W10, or FastLag); loss columns are millinats from that reference (Late averages 11 second-half "
            r"checkpoints; Train averages the final 537M tokens). Step $\Delta t$ is median overhead from the "
            r"unified 8-GPU benchmark. Because earlier waves precede the unified confirmation runner, "
            r"cross-wave differences are exploratory."
        ),
        "tab:ablations_full",
        [
            "Method",
            "Cohort",
            "Inputs",
            "Outputs",
            "Final",
            "Late",
            "Late C4",
            "Train",
            r"Step $\Delta t$",
        ],
        full_rows,
        spec="Xlllrrrrr",
        preamble_rows=[
            r"& & ($\dm/\dff$) & ($\dm/\dff$) & \multicolumn{4}{c}{$\Delta$ vs.\ cohort MuonT (mnats)} & \\",
            r"\cmidrule(lr){5-8}",
        ],
        size=r"\footnotesize",
        colsep="3.6pt",
    )

    # 2. Related polar baselines table
    rel_base = "campaigns/related-selected-full-seed12-20261121/runs"
    rel_methods = [
        ("muon_control", "Muon", "Reference", r"None; $\Nten(U)$"),
        (
            "mousse_paper",
            "Mousse",
            "Paper",
            r"$GG^{\Tr}, G^{\Tr}G$; $(-1/8,-1/8)$ + graft",
        ),
        (
            "mousse_control",
            "Mousse",
            "Controlled",
            r"$GG^{\Tr}, G^{\Tr}G$; $(-1/8,-1/8)$ + graft",
        ),
        ("go_muon_control", "GO-MUON", "Controlled", r"$C, B$; $(-1/4,-1/4)$ + graft"),
        ("go_muon_paper", "GO-MUON", "Paper", r"OAS $C, B$; SVD + graft"),
        (
            "inner_outer_mlpdiag",
            "WT-MLP",
            "Width-thresh.",
            r"$C, B$ ($\dff$ diag); $(-1/2,-1/8)$",
        ),
    ]
    rel_stats = {}
    for arm, _, _, _ in rel_methods:
        finals_sel, lates_sel, finals_c4, med_steps = [], [], [], []
        for s in [1, 2]:
            recs = record(f"{rel_base}/{arm}_0p5m_seed{s}_lr2/metrics_from_000000.jsonl.gz")
            evals = [r for r in recs if r.get("selection_loss") is not None]
            finals_sel.append(float(evals[-1]["selection_loss"]))
            lates_sel.append(float(np.mean([e["selection_loss"] for e in evals[-11:]])))
            finals_c4.append(float(evals[-1]["c4_loss"]))
            steps = [float(r["step_seconds"]) for r in recs if r.get("step", 0) > 20 and r.get("step_seconds")]
            med_steps.append(float(np.median(steps)))
        rel_stats[arm] = {
            "sel": np.array(finals_sel),
            "late": np.array(lates_sel),
            "c4": np.array(finals_c4),
            "step": float(np.mean(med_steps)),
        }

    ref_s = rel_stats["muon_control"]
    rel_rows: list[list[str] | str] = []
    for arm, label, profile, spec_txt in rel_methods:
        st = rel_stats[arm]
        is_wt = label == "WT-MLP"
        row = [
            f"\\textbf{{{label}}}" if is_wt else label,
            profile,
            spec_txt,
            (rf"\textbf{{{st['sel'].mean():.4f}}}" if is_wt else f"{st['sel'].mean():.4f}"),
            num((st["sel"] - ref_s["sel"]).mean() * 1000, bold=is_wt),
            num((st["late"] - ref_s["late"]).mean() * 1000, bold=is_wt),
            num((st["c4"] - ref_s["c4"]).mean() * 1000, bold=is_wt),
            f"{st['step']:.2f}",
        ]
        rel_rows.append(tint(row) if is_wt else row)

    write_table(
        "related_results",
        (
            r"\textbf{Two-sided polar baselines under a shared isotropic-decay recipe (130M, 2.68B tokens).} "
            r"Entries average fresh seeds 1 and 2 at $\eta=0.032$ and $\lambda=0.1$ on the standalone "
            r"synchronous harness. Held-out is final $\Lsel$ in nats; $\Delta$ columns are millinats from "
            r"Muon (Late-11 averages the second-half checkpoints); Step is median seconds per update."
        ),
        "tab:related",
        [
            "Method",
            "Profile",
            "Moments \\& Exponents",
            "Held-out",
            r"$\Delta\Lsel$",
            r"Late-11 $\Delta$",
            r"$\Delta\Lc$",
            "Step (s)",
        ],
        rel_rows,
        spec="llXrrrrr",
        colsep="4.2pt",
        pos="!htb",
    )

    # 3. Main-body counterfactual rotation + 520M delocalization table & Appendix inner/outer detail
    rot_rows: list[list[str] | str] = []
    d130 = ROT["exp123_130m_by_kind"]
    d520 = ROT["exp4_cf_520m_by_kind"]
    ipr520 = ROT["exp4_ipr_520m_by_kind"]

    for p in ["q", "k", "v", "o", "down", "gate", "up"]:
        if p in {"o", "gate"}:
            rot_rows.append(r"\addlinespace[2.5pt]")
        a1 = d130[p]
        a5 = d520[p]
        da5_key = f"out_{p}"
        da5 = f"{ipr520[da5_key]['diag_aniso_pct']['median']:.1f}\\%" if da5_key in ipr520 else "---"
        phi5_wt = f"{a5['Phi_wt_over_muont']['mean']:.3f}" if "Phi_wt_over_muont" in a5 else "---"
        phi5_od = f"{a5['Phi_od_over_muont']['mean']:.3f}" if "Phi_od_over_muont" in a5 else "---"
        hijack = f"{a1['hijack_Z0_top16']['mean'] * 100:.1f}\\to{a1['hijack_Zwt_top16']['mean'] * 100:.1f}\\%"
        is_res = p in {"o", "down"}
        disp_p = rf"\textbf{{{p}}} (res.)" if is_res else p
        row = [
            disp_p,
            f"${hijack}$",
            rf"${a1['deg_wt_ns10']['mean']:.1f}^\circ$",
            rf"${a1['deg_od_ns10']['mean']:.1f}^\circ$",
            f"{a1['Phi_wt_over_muont']['mean']:.3f}",
            da5,
            (
                rf"$\mathbf{{{a5['deg_wt_vs_muont']['mean']:.1f}^\circ}}$"
                if is_res
                else rf"${a5['deg_wt_vs_muont']['mean']:.1f}^\circ$"
            ),
            rf"${a5['deg_od_vs_muont']['mean']:.1f}^\circ$",
            rf"\textbf{{{phi5_wt}}}" if is_res else phi5_wt,
            phi5_od,
        ]
        rot_rows.append(tint(row) if is_res else row)

    norm_520_mean = sum(d520[p]["norm_wt_over_muont"]["mean"] for p in ["q", "k", "v", "o", "down", "gate", "up"]) / 7.0
    res_ipr_520 = 0.5 * (ipr520["out_o"]["ipr_all_median"]["median"] + ipr520["out_down"]["ipr_all_median"]["median"])
    write_table(
        "rotation",
        (
            r"\textbf{Counterfactual update rotation, top-mode de-hijacking, and width scaling across 130M "
            r"(32-layer means) and 520M (16-layer means).} Holding checkpoint momentum and input factors "
            rf"fixed, $\theta_{{\rm WT}}$ and $\theta_{{\rm OD}}$ measure update angles relative to input-only "
            rf"updates ($\|D_{{\rm WT}}\|_F/\|D_{{\rm MT}}\|_F = 0.985$ at 130M and ${norm_520_mean:.3f}$ at 520M). "
            r"Hijack reports top-16 overlap between pre-polar momentum singular vectors and $B_t$ eigenvectors "
            r"before and after inner whitening; DA is 520M median diagonal anisotropy share (residual "
            rf"$\mathrm{{IPR}}={res_ipr_520:.2f}$); $\Phi$ ratios are directional quadratic proxies relative to MuonT, "
            r"not measured loss gains. At 520M, \texttt{gate,up} track 1D diagonals online ($\text{WT}=\text{OD}$)."
        ),
        "tab:mechanism_rotation",
        [
            "Output",
            "Hijack",
            r"$\theta_{\rm WT}$",
            r"$\theta_{\rm OD}$",
            r"$\Phi_{\rm WT}/\Phi_{\rm MT}$",
            "DA",
            r"$\theta_{\rm WT}$",
            r"$\theta_{\rm OD}$",
            r"$\Phi_{\rm WT}/\Phi_{\rm MT}$",
            r"$\Phi_{\rm OD}/\Phi_{\rm MT}$",
        ],
        rot_rows,
        spec="Xrrrrrrrrr",
        preamble_rows=[
            (
                r"& \multicolumn{4}{c}{130M ($\dm=512$, step 2,500)} "
                r"& \multicolumn{5}{c}{520M ($\dm=1{,}024$, step 5,120)} \\"
            ),
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-10}",
        ],
        colsep="4.5pt",
        pos="!htb",
    )

    det_rows: list[list[str] | str] = []
    for p in ["q", "k", "v", "o", "down", "gate", "up"]:
        a = d130[p]
        det_rows.append(
            [
                p,
                rf"${a['deg_inner_svd']['mean']:.1f}^\circ$",
                rf"${a['deg_inner_ns10']['mean']:.1f}^\circ$",
                rf"${a['deg_outer_ns10']['mean']:.1f}^\circ$",
                f"${a['cos_inner_vs_outer_delta']['mean']:+.3f}$",
                f"${a['hijack_Z0_top16']['mean'] * 100:.1f}\\to{a['hijack_Zwt_top16']['mean'] * 100:.1f}\\%$",
                f"{a['procrustes_pct_bot80_improved']['mean']:.1f}\\%",
                f"${a['erank_Z0_pct']['mean']:.1f}\\to{a['erank_Zwt_pct']['mean']:.1f}\\%$",
                f"{a['Q_BC_wt_over_muont']['mean']:.3f}",
                f"{a['Phi_od_over_muont']['mean']:.3f}",
            ]
        )
    write_table(
        "rotation_detail",
        (
            r"\textbf{Complete 130M inner/outer decomposition and spectral alignment probes (32-layer means, "
            r"step 2,500).} Inner SVD and Inner $\Nten$ measure left-factor rotation under exact polar "
            r"decomposition versus ten-step Newton--Schulz; Outer $\Nten$ isolates the post-polar $-1/8$ "
            r"factor; Cosine compares the inner and outer update deltas (near zero on $m\le n$ projections, "
            r"confirming orthogonal tangent/normal action). \texttt{bot80} win is the fraction of trailing "
            r"80\% $B_t$ modes gaining row alignment; $\mathrm{erank}$ compares pre-polar effective rank "
            r"($Z_0\to L Z_0$); $\mathcal Q_{\rm WT}/\mathcal Q_{\rm MT}$ and $\Phi_{\rm OD}/\Phi_{\rm MT}$ "
            r"complete the quadratic surrogate ratios."
        ),
        "tab:rotation_detail",
        [
            "Output",
            "Inner SVD",
            r"Inner $\Nten$",
            r"Outer $\Nten$",
            "Cosine",
            "Hijack",
            r"\texttt{bot80} win",
            r"$\mathrm{erank}$",
            r"$\mathcal Q_{\rm WT}/\mathcal Q_{\rm MT}$",
            r"$\Phi_{\rm OD}/\Phi_{\rm MT}$",
        ],
        det_rows,
        spec="Xrrrrrrrrr",
        size=r"\footnotesize",
        colsep="3.2pt",
    )

    # 4. Systems timing summary table
    t_rows: list[list[str] | str] = [
        group(
            "130M co-design progression: 8 GPUs, 0.5M batch, matched checkpoint (steps 37--96)",
            6,
            "2pt",
        )
    ]
    mt130 = TIMING130["muont"]["measured_benchmark"]
    for key, label in [
        ("muon", "Muon"),
        ("muont", "MuonT (reference)"),
        ("inner_outer_leftdiag", "OD-All (all outputs diagonal)"),
        ("inner_outer_all", "DT-All (all outputs dense, sync)"),
        ("inner_outer_mlpdiag", "WT-MLP (width threshold, sync)"),
        ("inner_outer_mlpdiag_lag", "WT-MLP-Lag (+ one-step lag)"),
    ]:
        b = TIMING130[key]["measured_benchmark"]
        med_d = (b["median_step_seconds"] / mt130["median_step_seconds"] - 1) * 100
        mean_d = (b["mean_step_seconds"] / mt130["mean_step_seconds"] - 1) * 100
        is_wt = key == "inner_outer_mlpdiag_lag"
        row = [
            f"\\textbf{{{label}}}" if is_wt else label,
            (rf"\textbf{{{b['median_step_seconds']:.3f}}}" if is_wt else f"{b['median_step_seconds']:.3f}"),
            rf"$\mathbf{{{med_d:+.1f}\%}}$" if is_wt else rf"${med_d:+.1f}\%$",
            (rf"\textbf{{{b['mean_step_seconds']:.3f}}}" if is_wt else f"{b['mean_step_seconds']:.3f}"),
            rf"$\mathbf{{{mean_d:+.1f}\%}}$" if is_wt else rf"${mean_d:+.1f}\%$",
            (rf"\textbf{{{b['peak_allocated_gib']:.2f}}}" if is_wt else f"{b['peak_allocated_gib']:.2f}"),
        ]
        t_rows.append(tint(row) if is_wt else row)

    t_rows.append(
        group(
            "520M width and cluster scaling: 32 GPUs, 1.0M batch, chunked buffers",
            6,
            "5pt",
        )
    )
    g32 = TIMING520["final_ws32_mb2_c2"]
    mt32 = g32["muont"]
    for key, label in [
        ("muon", "Muon (matched ckpt)"),
        ("muont", "MuonT (matched ckpt, ref.)"),
        ("od_all", "OD-All (matched ckpt)"),
        ("wt_mlp_lag", "WT-MLP-Lag (matched ckpt)"),
    ]:
        b = g32[key]
        med_d = (b["median_s"] / mt32["median_s"] - 1) * 100
        mean_d = (b["mean_s"] / mt32["mean_s"] - 1) * 100
        is_wt = key == "wt_mlp_lag"
        row = [
            f"\\textbf{{{label}}}" if is_wt else label,
            rf"\textbf{{{b['median_s']:.3f}}}" if is_wt else f"{b['median_s']:.3f}",
            rf"$\mathbf{{{med_d:+.1f}\%}}$" if is_wt else rf"${med_d:+.1f}\%$",
            rf"\textbf{{{b['mean_s']:.3f}}}" if is_wt else f"{b['mean_s']:.3f}",
            rf"$\mathbf{{{mean_d:+.1f}\%}}$" if is_wt else rf"${mean_d:+.1f}\%$",
            (rf"\textbf{{{b['peak_alloc_gib']:.2f}}}" if is_wt else f"{b['peak_alloc_gib']:.2f}"),
        ]
        t_rows.append(tint(row) if is_wt else row)

    w32 = TIMING520["warm_ws32_mb2_c2"]
    bw, mtw = w32["wt_mlp_lag"], w32["muont"]
    t_rows.append(
        [
            "WT-MLP-Lag (own warm state vs.\\ warm MuonT)",
            f"{bw['median_s']:.3f}",
            rf"${(bw['median_s'] / mtw['median_s'] - 1) * 100:+.1f}\%$",
            f"{bw['mean_s']:.3f}",
            rf"${(bw['mean_s'] / mtw['mean_s'] - 1) * 100:+.1f}\%$",
            f"{bw['peak_alloc_gib']:.2f}",
        ]
    )

    write_table(
        "timing_summary",
        (
            r"\textbf{Step-time co-design waterfall and multi-node scaling (both median and mean estimators).} "
            r"Overheads ($\Delta$) are relative to same-group MuonT using the matching estimator; Peak GiB "
            r"is per-GPU peak allocated memory. Width thresholding and one-step lag cut median overhead 4$\times$ "
            r"relative to synchronous dense outputs (DT-All), while retaining a slower mean tail from periodic "
            r"basis repairs. Complete 130M and 520M configurations appear in \Cref{tab:bench19,tab:bench520m}."
        ),
        "tab:timing_summary",
        [
            "Configuration / Method",
            "Median (s)",
            r"Median $\Delta$",
            "Mean (s)",
            r"Mean $\Delta$",
            "Peak GiB",
        ],
        t_rows,
        spec="Xrrrrr",
        colsep="6.5pt",
        pos="!htb",
    )


def build_remaining_appendix_tables():
    # 1. LR cliff table (combining 320-step short sweep and 5,120-step full-budget runs across 8 LRs)
    lr_data = read("lr_summary.json")
    s0 = LR_BRACKET["seed0_grid"]
    rates_num: list[float] = []
    lr_rows: list[list[str] | str] = [group("Short-schedule proxy sweep: 320 steps (0.17B tokens, seed 0)", 9, "2pt")]
    for arm, label in [
        ("muont", "MuonT"),
        ("inner_outer_leftdiag", "OD-All"),
        ("outer_leftdiag", "OD-Outer"),
        ("outer_leftdiag_fused", "OD-Outer-F"),
    ]:
        values = sorted(
            [x for k, x in lr_data.items() if k.split("/")[1] == arm],
            key=lambda x: x["lr"],
        )
        rates_num = [float(x["lr"]) for x in values]
        sels = [float(x["selection"]) for x in values]
        best_i = int(np.argmin(sels))
        cells = [
            "---",
            *[rf"\textbf{{{v:.4f}}}" if i == best_i else f"{v:.4f}" for i, v in enumerate(sels)],
        ]
        lr_rows.append([label, *cells])

    all_rates = [s0["muont"]["0p0113"]["lr"], *rates_num]
    runs = CAMPAIGN["runs"]
    full_grid = [
        (
            "Muon",
            {
                0: s0["muon"]["0p0113"]["selection_loss"],
                1: s0["muon"]["0p016"]["selection_loss"],
                2: s0["muon"]["0p0226"]["selection_loss"],
                3: s0["muon"]["0p032"]["selection_loss"],
            },
        ),
        (
            "MuonT",
            {
                0: s0["muont"]["0p0113"]["selection_loss"],
                1: s0["muont"]["0p016"]["selection_loss"],
                2: s0["muont"]["0p0226"]["selection_loss"],
                3: s0["muont"]["0p032"]["selection_loss"],
                5: SUMMARY["wave12/muont"]["sel_final"],
            },
        ),
        (
            "OD-All",
            {
                0: s0["od_all"]["0p0113"]["selection_loss"],
                1: s0["od_all"]["0p016"]["selection_loss"],
                2: s0["od_all"]["0p0226"]["selection_loss"],
                3: s0["od_all"]["0p032"]["selection_loss"],
                6: SUMMARY["wave12/inner_outer_leftdiag"]["sel_final"],
            },
        ),
        (
            "OD-Outer-F",
            {
                2: SUMMARY["wave10/outer_leftdiag_fused"]["sel_final"],
                5: SUMMARY["wave12/outer_leftdiag_fused"]["sel_final"],
            },
        ),
        (
            "WT-MLP-Lag",
            {
                0: s0["wt_mlp_lag"]["0p0113"]["selection_loss"],
                1: s0["wt_mlp_lag"]["0p016"]["selection_loss"],
                2: s0["wt_mlp_lag"]["0p0226"]["selection_loss"],
                3: s0["wt_mlp_lag"]["0p032"]["selection_loss"],
            },
        ),
    ]
    lr_rows.append(group("Full-budget verification: 5,120 steps (2.68B tokens, seed 0)", 9, "5pt"))
    for label, mapping in full_grid:
        best_col = min(mapping, key=mapping.__getitem__)
        cells = []
        for idx in range(8):
            if idx in mapping:
                val = mapping[idx]
                cells.append(rf"\textbf{{{val:.4f}}}" if idx == best_col else f"{val:.4f}")
            else:
                cells.append("---")
        row = [f"\\textbf{{{label}}}" if label == "WT-MLP-Lag" else label, *cells]
        lr_rows.append(tint(row) if label == "WT-MLP-Lag" else row)

    write_table(
        "lr_short",
        (
            r"\textbf{Short-schedule (320-step) versus full-budget (5,120-step) matrix learning-rate sweeps at "
            r"130M (seed 0).} Entries are final held-out losses $\Lsel$ in nats; bold marks each row's minimum "
            r"within that horizon. All four core full-budget curves achieve an interior minimum at $0.0160$, "
            r"whereas the compressed 320-step schedule shifts apparent optima $4\text{--}6\times$ higher."
        ),
        "tab:lr_cliff",
        ["Method", *[f"{r:.4f}" for r in all_rates]],
        lr_rows,
        spec="Xrrrrrrrr",
        size=r"\footnotesize",
        colsep="5.2pt",
    )

    # 2. Complete absolute losses: seed-0 bracket, Null-WT, and both fresh-seed learning rates.
    abs_rows: list[list[str] | str] = []
    abs_rows.append(
        group(
            "130M seed-0 controls: 20 TPP, 2.68B tokens (complete learning-rate bracket)",
            6,
            "2pt",
        )
    )
    for lr_tag in ("0p0113", "0p016", "0p0226", "0p032"):
        for arm_k, label in [
            ("muon", "Muon"),
            ("muont", "MuonT"),
            ("od_all", "OD-All"),
            ("wt_mlp_lag", "WT-MLP-Lag"),
        ]:
            if lr_tag == "0p0226" and arm_k == "od_all":
                rn = runs["wave1b/null_wt_130m_1x_seed0_lr0p0226"]
                abs_rows.append(
                    [
                        "Null-WT",
                        f"{rn['lr']:.5f}",
                        f"{rn['final_sel']:.4f}",
                        f"{rn['late11_sel']:.4f}",
                        f"{rn['final_c4']:.4f}",
                        "---",
                    ]
                )
            r_s0 = s0[arm_k][lr_tag]
            pal_p = RECORDS / "paloma_eval" / f"{arm_k}_130m_1x_seed0_lr{lr_tag}.json"
            pal_cell = f"{json.loads(pal_p.read_text())['paloma_macro_mean']:.4f}" if pal_p.is_file() else "---"
            abs_rows.append(
                [
                    label,
                    f"{r_s0['lr']:.5f}",
                    f"{r_s0['selection_loss']:.4f}",
                    f"{r_s0['late11_sel']:.4f}",
                    f"{r_s0['c4_loss']:.4f}",
                    pal_cell,
                ]
            )

    s123 = LR_BRACKET["seeds123_summaries"]
    abs_rows.append(
        group(
            "130M confirmation: 20 TPP, 2.68B tokens (mean of fresh seeds 1--3; matched $\\eta^*$ \\& hotter LR)",
            6,
            "4pt",
        )
    )
    for key_123, label, is_wt_tint in [
        ("muon@0p016", "Muon", False),
        ("muont@0p016", "MuonT", False),
        ("od_all@0p016", "OD-All", False),
        ("wt_mlp_lag@0p016", "WT-MLP-Lag", True),
        ("od_all@0p0226", "OD-All (hotter LR)", False),
        ("wt_mlp_lag@0p0226", "WT-MLP-Lag (hotter LR)", False),
    ]:
        d = s123[key_123]
        row = [
            label,
            f"{d['lr']:.5f}",
            f"{d['sel_mean']:.4f}",
            f"{d['late11_sel_mean']:.4f}",
            f"{d['c4_mean']:.4f}",
            f"{d['paloma_mean']:.4f}",
        ]
        abs_rows.append(tint(row) if is_wt_tint else row)

    abs_rows.append(
        group(
            "130M longer training: 80 TPP, 10.74B tokens, seed 0 (matched \\& hotter rates)",
            6,
            "4pt",
        )
    )
    for label, rk, pal_name in [
        ("Muon", "wave3a/muon_130m_4x_seed0", "muon_130m_4x_seed0"),
        ("MuonT", "wave3a/muont_130m_4x_seed0", "muont_130m_4x_seed0"),
        (
            "OD-All",
            "wave3b/od_all_130m_4x_seed0_lr0p01056",
            "od_all_130m_4x_seed0_lr0p01056",
        ),
        (
            "WT-MLP-Lag",
            "wave3b/wt_mlp_lag_130m_4x_seed0_lr0p01056",
            "wt_mlp_lag_130m_4x_seed0_lr0p01056",
        ),
        (
            "WT-MLP-Lag (hotter LR)",
            "wave3a/wt_mlp_lag_130m_4x_seed0",
            "wt_mlp_lag_130m_4x_seed0",
        ),
    ]:
        r = runs[rk]
        pa = pal(pal_name)
        row = [
            label,
            f"{r['lr']:.5f}",
            f"{r['final_sel']:.4f}",
            f"{r['late11_sel']:.4f}",
            f"{r['final_c4']:.4f}",
            f"{pa['paloma_macro_mean']:.4f}",
        ]
        abs_rows.append(tint(row) if label == "WT-MLP-Lag" else row)

    abs_rows.append(
        group(
            "520M width transfer: 20 TPP, 10.74B tokens, seed 0 (matched $\\eta^*$ \\& hotter rates)",
            6,
            "4pt",
        )
    )
    for label, rk, pal_name, is_wt_tint in [
        ("Muon", "wave4a/muon_520m_1x_seed0", "muon_520m_1x_seed0", False),
        (
            "MuonT",
            "lr520m/muont_520m_1x_seed0_lr0p016",
            "muont_520m_1x_seed0_lr0p016",
            False,
        ),
        (
            "OD-All",
            "lr520m/od_all_520m_1x_seed0_lr0p016",
            "od_all_520m_1x_seed0_lr0p016",
            False,
        ),
        (
            "WT-MLP-Lag",
            "lr520m/wt_mlp_lag_520m_1x_seed0_lr0p016",
            "wt_mlp_lag_520m_1x_seed0_lr0p016",
            True,
        ),
        (
            "Muon (hotter LR)",
            "lr520m/muon_520m_1x_seed0_lr0p0226",
            "muon_520m_1x_seed0_lr0p0226",
            False,
        ),
        (
            "MuonT (hotter LR)",
            "wave4a/muont_520m_1x_seed0",
            "muont_520m_1x_seed0",
            False,
        ),
        (
            "OD-All (hotter LR)",
            "wave4a/od_all_520m_1x_seed0",
            "od_all_520m_1x_seed0",
            False,
        ),
        (
            "WT-MLP-Lag (hotter LR)",
            "wave4a/wt_mlp_lag_520m_1x_seed0",
            "wt_mlp_lag_520m_1x_seed0",
            False,
        ),
    ]:
        r = runs[rk]
        pa = pal(pal_name)
        row = [
            label,
            f"{r['lr']:.5f}",
            f"{r['final_sel']:.4f}",
            f"{r['late11_sel']:.4f}",
            f"{r['final_c4']:.4f}",
            f"{pa['paloma_macro_mean']:.4f}",
        ]
        abs_rows.append(tint(row) if is_wt_tint else row)

    write_table(
        "absolute_results",
        (
            r"\textbf{Complete absolute losses and learning rates across all confirmatory regimes.} All "
            r"losses are cross-entropy in nats; Late-11 Held-out averages the 11 second-half evaluation "
            r"checkpoints. Hotter-LR rows ($\eta=0.02263$ at 130M and 520M 20 TPP; $\eta=0.01493$ at 130M 80 TPP) "
            r"document the two-rate checks and learning-rate sensitivity discussed in "
            r"\Cref{sec:phase2,sec:phase3,sec:phase4}."
        ),
        "tab:absolute_results",
        [
            "Method",
            r"Matrix $\eta$",
            "Final Held-out",
            "Late-11 Held-out",
            "Final C4",
            "Paloma Macro",
        ],
        abs_rows,
        spec="Xrrrrr",
        size=r"\scriptsize",
        colsep="8.5pt",
    )

    # 3. Per-seed fresh replication table (both matched 0.01600 and hotter 0.02263 replicates)
    mt123 = s123["muont@0p016"]
    seed_rows: list[list[str] | str] = []
    for s_idx, seed in enumerate([1, 2, 3]):
        seed_rows.append(
            group(
                rf"Seed {seed}: MuonT reference ($\Lsel={mt123['sel_seeds'][s_idx]:.4f}$, "
                rf"Late-11 $={mt123['late11_sel_seeds'][s_idx]:.4f}$, "
                rf"$\Lc={mt123['c4_seeds'][s_idx]:.4f}$, "
                rf"$\Lpal={mt123['paloma_seeds'][s_idx]:.4f}$ nats)",
                6,
                "2pt" if s_idx == 0 else "4pt",
            )
        )
        for key_123, label, is_wt, show_l11 in [
            ("muon@0p016", "Muon", False, True),
            ("od_all@0p016", "OD-All", False, True),
            ("wt_mlp_lag@0p016", "WT-MLP-Lag", True, True),
            ("od_all@0p0226", "OD-All (hotter LR)", False, False),
            ("wt_mlp_lag@0p0226", "WT-MLP-Lag (hotter LR)", False, False),
        ]:
            d = s123[key_123]
            l11_cell = (
                num(
                    (d["late11_sel_seeds"][s_idx] - mt123["late11_sel_seeds"][s_idx]) * 1000,
                    bold=is_wt,
                )
                if show_l11
                else "---"
            )
            row = [
                f"\\textbf{{{label}}}" if is_wt else label,
                f"{d['lr']:.5f}",
                num(
                    (d["sel_seeds"][s_idx] - mt123["sel_seeds"][s_idx]) * 1000,
                    bold=is_wt,
                ),
                l11_cell,
                num((d["c4_seeds"][s_idx] - mt123["c4_seeds"][s_idx]) * 1000, bold=is_wt),
                num(
                    (d["paloma_seeds"][s_idx] - mt123["paloma_seeds"][s_idx]) * 1000,
                    bold=is_wt,
                ),
            ]
            seed_rows.append(tint(row) if is_wt else row)

    write_table(
        "seeds",
        (
            r"\textbf{Individual fresh-seed replication at 130M (20 TPP, 2.68B tokens).} Subheaders report "
            r"each seed's absolute MuonT losses ($\eta=0.01600$); rows report paired differences from "
            r"same-seed MuonT in millinats (negative favors the row method; Late-11 is omitted for "
            r"$\eta=0.02263$ rows because intermediate decay schedules differ). WT-MLP-Lag improves over "
            r"both Muon and MuonT on every seed across all evaluation suites at both learning rates."
        ),
        "tab:seeds",
        [
            "Method",
            r"Matrix $\eta$",
            r"$\Delta$ Held-out",
            r"Late-11 $\Delta$",
            r"$\Delta$ C4",
            r"$\Delta$ Paloma",
        ],
        seed_rows,
        spec="Xrrrrr",
        size=r"\footnotesize",
        colsep="7.5pt",
    )

    # 4. Complete 130M 19-arm timing table (sorted by median latency, with both Median % and Mean % vs MuonT)
    mt130 = TIMING130["muont"]["measured_benchmark"]
    sorted_130 = sorted(TIMING130.values(), key=lambda x: x["measured_benchmark"]["median_step_seconds"])
    t130_rows: list[list[str] | str] = []
    for item in sorted_130:
        t = item["measured_benchmark"]
        med_pct = (t["median_step_seconds"] / mt130["median_step_seconds"] - 1) * 100
        mean_pct = (t["mean_step_seconds"] / mt130["mean_step_seconds"] - 1) * 100
        is_wt = item["tax_id"] == "WT-MLP-Lag"
        row = [
            f"\\textbf{{{item['tax_id']}}}" if is_wt else item["tax_id"],
            f"{t['median_step_seconds']:.4f}",
            rf"${med_pct:+.1f}\%$",
            f"{t['mean_step_seconds']:.4f}",
            rf"${mean_pct:+.1f}\%$",
            f"{t['peak_allocated_gib']:.2f}",
        ]
        t130_rows.append(tint(row) if is_wt else row)

    write_table(
        "timing130",
        (
            r"\textbf{Complete 19-arm 130M timing and memory benchmark on 8$\times$A100.} Sorted by median "
            r"step duration over steps 37--96 (after 36 warmup steps) from identical step-2,500 weights and "
            r"fresh optimizer/moment state. Both median and mean overheads ($\Delta$) are relative to MuonT."
        ),
        "tab:bench19",
        [
            "Method",
            "Median (s)",
            r"Median $\Delta$",
            "Mean (s)",
            r"Mean $\Delta$",
            "Peak GiB",
        ],
        t130_rows,
        spec="Xrrrrr",
        size=r"\footnotesize",
        colsep="9.5pt",
    )

    # 5. Complete 520M timing table (both Median % and Mean % vs MuonT across all 4 recorded configurations)
    descriptions = {
        "final_ws8_mb2_c2": "8 GPUs, matched checkpoint, steps 37--96",
        "final_ws32_mb2_c2": "32 GPUs, matched checkpoint, steps 37--96",
        "warm_ws32_mb2_c2": "32 GPUs, own optimizer states, steps 17--76",
        "warm_numa_ws8_mb2_c2": "8 GPUs, own optimizer states, steps 17--76",
    }
    t520_rows: list[list[str] | str] = []
    for idx, (k, d) in enumerate(TIMING520.items()):
        t520_rows.append(group(descriptions[k], 6, "2pt" if idx == 0 else "4pt"))
        mt = d["muont"]
        for arm, label in [
            ("muon", "Muon"),
            ("muont", "MuonT"),
            ("od_all", "OD-All"),
            ("wt_mlp_lag", "WT-MLP-Lag"),
        ]:
            a = d[arm]
            med_over = (a["median_s"] / mt["median_s"] - 1) * 100
            mean_over = (a["mean_s"] / mt["mean_s"] - 1) * 100
            is_wt = arm == "wt_mlp_lag"
            row = [
                f"\\textbf{{{label}}}" if is_wt else label,
                f"{a['median_s']:.4f}",
                rf"${med_over:+.1f}\%$",
                f"{a['mean_s']:.4f}",
                rf"${mean_over:+.1f}\%$",
                f"{a['peak_alloc_gib']:.2f}",
            ]
            t520_rows.append(tint(row) if is_wt else row)

    write_table(
        "timing520",
        (
            r"\textbf{Complete 520M timing and memory records across 8 and 32 A100 GPUs (1.0M batch, chunked "
            r"buffers).} All rows report the complete recorded measurement window without outlier removal; "
            r"own optimizer states restore each arm's converged step-5,120 moments and eigenbases. Overheads "
            r"($\Delta$) are relative to within-group MuonT."
        ),
        "tab:bench520m",
        [
            "Method",
            "Median (s)",
            r"Median $\Delta$",
            "Mean (s)",
            r"Mean $\Delta$",
            "Peak GiB",
        ],
        t520_rows,
        spec="Xrrrrr",
        size=r"\footnotesize",
        colsep="9.5pt",
    )

    # 6. Complete 16-domain Paloma table with grouped headers, bold per-regime winners, and win counts
    pal_groups = [
        (
            "muont_130m_4x_seed0",
            "od_all_130m_4x_seed0_lr0p01056",
            "wt_mlp_lag_130m_4x_seed0_lr0p01056",
        ),
        (
            "muont_520m_1x_seed0_lr0p016",
            "od_all_520m_1x_seed0_lr0p016",
            "wt_mlp_lag_520m_1x_seed0_lr0p016",
        ),
    ]
    columns = []
    for arm in ["od_all", "wt_mlp_lag"]:
        columns.append(
            np.mean(
                [pal_deltas(f"muont_130m_1x_seed{seed}", f"{arm}_130m_1x_seed{seed}_lr0p016") for seed in [1, 2, 3]],
                axis=0,
            )
        )
    for mt, od, wt in pal_groups:
        columns.extend([pal_deltas(mt, od), pal_deltas(mt, wt)])

    pal_rows: list[list[str] | str] = []
    for i, (_, label) in enumerate([*DOMAINS, ("", "16-domain macro mean")]):
        is_macro = i == len(DOMAINS)
        if is_macro:
            pal_rows.append(r"\midrule")
        cells = [r"\textbf{16-domain macro mean}" if is_macro else label]
        for pair_idx in range(3):
            v_od = float(columns[2 * pair_idx][i])
            v_wt = float(columns[2 * pair_idx + 1][i])
            cells.append(num(v_od, bold=v_od < v_wt))
            cells.append(num(v_wt, bold=v_wt <= v_od))
        pal_rows.append(tint(cells) if is_macro else cells)

    win_cells = [r"\textbf{Domains won vs.\ MuonT}"]
    for col_idx, col in enumerate(columns):
        wins = int(np.sum(np.array(col[:16]) < 0))
        is_wt = col_idx % 2 == 1
        win_cells.append(rf"\textbf{{{wins}/16}}" if is_wt else f"{wins}/16")
    pal_rows.append(win_cells)

    write_table(
        "paloma",
        (
            r"\textbf{Per-domain Paloma loss differences from MuonT across all 16 domains.} Entries are "
            r"millinats ($\text{Method}-\text{MuonT}$; negative favors the method, bold marks the stronger "
            r"output approximation within each regime). The 130M 20-TPP columns average paired differences "
            r"across fresh seeds 1--3; 130M 80-TPP and 520M 20-TPP use single-seed matched-rate runs."
        ),
        "tab:paloma_full",
        ["Paloma domain", "OD-All", "WT-Lag", "OD-All", "WT-Lag", "OD-All", "WT-Lag"],
        pal_rows,
        spec="Xrrrrrr",
        preamble_rows=[
            (
                r"& \multicolumn{2}{c}{130M, 20 TPP (3 seeds)} "
                r"& \multicolumn{2}{c}{130M, 80 TPP} "
                r"& \multicolumn{2}{c}{520M, 20 TPP} \\"
            ),
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        ],
        colsep="7.0pt",
    )


def main():
    TABLES.mkdir(exist_ok=True)
    build_diagnosis_tables()
    build_method_and_protocol_tables()
    build_main_scaling_table()
    build_analysis_tables()
    build_remaining_appendix_tables()
    print("Generated manuscript tables in", TABLES)


if __name__ == "__main__":
    main()
