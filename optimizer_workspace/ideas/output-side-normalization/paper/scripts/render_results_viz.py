"""Systematic taxonomy, structured tables, and vector visualization generator.

Reads ONLY the materialized JSON artifacts in paper/results/:
  - results/summary.json
  - results/lr_summary.json
  - results/mechanism_summary.json
  - results/systems_summary.json

Emits:
  - results/table_main.tex       Taxonomic master comparison with 5-slot badges,
                                 Raw vs. Net-of-Null selection delta, and
                                 Legacy -> Fast-Lag step overhead
  - results/table_windows.tex    Taxonomically grouped 5-window paired evolution
                                 table with temporal regime classification
  - figs/method_arch.tex         Pure-TikZ 2-panel architecture & fast_lag
                                 asynchronous timeline diagram
  - figs/plane_plots.tex         Zero-collision Cost-Quality Pareto plane with
                                 fast_lag frontier shift arrows & starred points
  - figs/attribution.dat         Machine-readable causal attribution table
  - figs/attribution_plots.tex   Panel (b) multi-metric causal attribution
                                 forest plot (Null share + Net gain + C4 + Train)
  - figs/lr_horizon.dat          Full-budget 5,120-step horizon-shift series
  - figs/transfer_bars.tex       Wave-12 transferred-rate delta bars

Run from paper/:
    python -I scripts/render_results_viz.py
"""

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(name):
    path = os.path.join(HERE, "results", name)
    with open(path) as handle:
        return json.load(handle)


def write_file(rel_path, lines):
    path = os.path.join(HERE, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {rel_path}")


def main():
    runs = load_json("summary.json")
    lr_data = load_json("lr_summary.json")
    mech = load_json("mechanism_summary.json")
    sys_data = load_json("systems_summary.json")

    sys_tot = sys_data["totals_8gpu"]
    sys_cal = sys_data["calibrated_8gpu_ms"]
    sys_mem = sys_data["memory_32l_isolated"]

    # Matched-null selection deltas for causal net-of-null attribution.
    d_null_wt = runs["wave3/null_mlpdiag"]["d_sel_late"]
    d_null_dense = runs["wave1/null_all"]["d_sel_late"]
    d_null_od = runs["wave10/null_leftdiag_upstream_fused"]["d_sel_late"]

    # Fast-lag calibrated step overheads vs. MuonT and vs. Muon, plus 8-GPU GiB.
    fast_wt_muont = sys_tot["fast_mlpdiag_lag_vs_muont_pct_mid"]
    fast_wt_muont_lo = sys_tot["fast_mlpdiag_lag_vs_muont_pct_lo"]
    fast_wt_muont_hi = sys_tot["fast_mlpdiag_lag_vs_muont_pct_hi"]
    fast_dt_muont = sys_tot["fast_all_lag_vs_muont_pct_mid"]
    fast_dt_muont_lo = sys_tot["fast_all_lag_vs_muont_pct_lo"]
    fast_dt_muont_hi = sys_tot["fast_all_lag_vs_muont_pct_hi"]
    fast_wt_gib = sys_mem["fast_mlpdiag_8gpu_est_gib"]
    fast_dt_gib = sys_mem["fast_all_8gpu_est_gib"]

    # Modular discovery for symmetric width-thresholded runs (WT-Sym / WT-Sym-Lag)
    # so they automatically slot into Family II as soon as executed.
    sym_entries = []
    for key in sorted(runs):
        rec = runs[key]
        if rec.get("seed") != 0 or rec.get("updates") != 5120:
            continue
        short = key.split("/")[-1]
        if "sym" in short and "mlpdiag" in short and "null" not in short:
            is_lag = short.endswith("_lag")
            tax = "WT-Sym-Lag" if is_lag else "WT-Sym"
            tau = "1" if is_lag else "0"
            code = short.replace("_", r"\_")
            sym_entries.append((
                key,
                rf"\textbf{{\taxid{{{tax}}}}} {{\tiny(\texttt{{{code}}})}}",
                rf"\tcfg{{\fFull}}{{\fDiag}}{{\fFull}}{{\fDiag}}{{{tau}}}",
                d_null_wt,
                "",
                True,
                None,
                None,
            ))

    # ------------------------------------------------------------------
    # 1. Master Results Table: results/table_main.tex
    # ------------------------------------------------------------------
    # Tuple schema per row:
    # (run_key, arm_cell, tuple_cell, null_ref, net_suffix, pareto_shade,
    #  fast_step_vs_muont, fast_gib)
    families = [
        (
            r"I.\ Baselines (\texttt{Ref})",
            "famRef",
            [
                (
                    "wave0/muont",
                    r"\taxid{MuonT} {\tiny(\muont{})}",
                    r"\tcfg{\fFull}{\fFull}{\fNone}{\fNone}{0}",
                    0.0,
                    "",
                    False,
                    None,
                    None,
                ),
                (
                    "wave0/muon",
                    r"\taxid{Muon} {\tiny(\texttt{muon})}",
                    r"\tcfg{\fNone}{\fNone}{\fNone}{\fNone}{\text{---}}",
                    None,
                    "",
                    False,
                    None,
                    None,
                ),
            ],
        ),
        (
            r"II.\ Width-Thresholded Two-Sided (\texttt{WT} --- Proposed Core Family)",
            "famWT",
            [
                (
                    "wave2/inner_outer_mlpdiag",
                    r"\textbf{\taxid{WT-MLP}} {\tiny(\texttt{mlpdiag})}",
                    r"\tcfg{\fFull}{\fFull}{\fFull}{\fDiag}{0}",
                    d_null_wt,
                    "",
                    True,
                    None,
                    None,
                ),
                (
                    "wave4/inner_outer_mlpdiag_lag",
                    r"\textbf{\taxid{WT-MLP-Lag}} {\tiny(\texttt{mlpdiag\_lag})}",
                    r"\tcfg{\fFull}{\fFull}{\fFull}{\fDiag}{1}",
                    d_null_wt,
                    "",
                    True,
                    fast_wt_muont,
                    fast_wt_gib,
                ),
            ]
            + sym_entries,
        ),
        (
            r"III.\ Pure-Diagonal Output (\texttt{OD} --- Zero-Eigendecomposition Limit)",
            "famOD",
            [
                (
                    "wave2/inner_outer_leftdiag",
                    r"\textbf{\taxid{OD-All}} {\tiny(\texttt{leftdiag})}",
                    r"\tcfg{\fFull}{\fFull}{\fDiag}{\fDiag}{0}",
                    d_null_dense,
                    r"^{\S}",
                    True,
                    None,
                    None,
                ),
                (
                    "wave10/outer_leftdiag_upstream_fused",
                    r"\taxid{OD-Outer}$^{\dagger}$ {\tiny(\texttt{outer\_leftdiag\_uf})}",
                    r"\tcfg{\fFull}{\fFull}{\fOuterDiag}{\fOuterDiag}{0}",
                    d_null_od,
                    "",
                    False,
                    None,
                    None,
                ),
            ],
        ),
        (
            r"IV.\ Dense Two-Sided Ceiling (\texttt{DT})",
            "famDT",
            [
                (
                    "wave3/inner_outer_all_lag",
                    r"\taxid{DT-All-Lag} {\tiny(\texttt{all\_lag})}",
                    r"\tcfg{\fFull}{\fFull}{\fFull}{\fFull}{1}",
                    d_null_dense,
                    "",
                    False,
                    fast_dt_muont,
                    fast_dt_gib,
                ),
                (
                    "wave1/inner_outer_all",
                    r"\taxid{DT-All} {\tiny(\texttt{inner\_outer\_all})}",
                    r"\tcfg{\fFull}{\fFull}{\fFull}{\fFull}{0}",
                    d_null_dense,
                    "",
                    False,
                    None,
                    None,
                ),
            ],
        ),
        (
            r"V.\ Matched Execution-Path Nulls (\texttt{Null} --- Identity Left Factors)",
            "famNull",
            [
                (
                    "wave3/null_mlpdiag",
                    r"\taxid{Null-WT} {\tiny(\texttt{null\_mlpdiag})}",
                    r"\tcfg{\fFull}{\fFull}{\fNullFull}{\fNullDiag}{0}",
                    "SELF",
                    "",
                    False,
                    None,
                    None,
                ),
                (
                    "wave1/null_all",
                    r"\taxid{Null-Dense} {\tiny(\texttt{null\_all})}",
                    r"\tcfg{\fFull}{\fFull}{\fNullFull}{\fNullFull}{0}",
                    "SELF",
                    "",
                    False,
                    None,
                    None,
                ),
                (
                    "wave10/null_leftdiag_upstream_fused",
                    r"\taxid{Null-OD}$^{\dagger}$ {\tiny(\texttt{null\_leftdiag\_uf})}",
                    r"\tcfg{\fFull}{\fFull}{\fNullDiag}{\fNullDiag}{0}",
                    "SELF",
                    "",
                    False,
                    None,
                    None,
                ),
            ],
        ),
        (
            r"VI.\ Structural \& Input-Degradation Ablations (\texttt{Abl})",
            "famAbl",
            [
                (
                    "wave2/outer_left_all",
                    r"\taxid{Abl-OuterDense} {\tiny(\texttt{outer\_left\_all})}",
                    r"\tcfg{\fFull}{\fFull}{\fOuterFull}{\fOuterFull}{0}",
                    d_null_dense,
                    "",
                    False,
                    None,
                    None,
                ),
                (
                    "wave2/inner_outer_hidden",
                    r"\taxid{Abl-No2048} {\tiny(\texttt{hidden})}",
                    r"\tcfg{\fFull}{\fFull}{\fFull}{\fNone}{0}",
                    d_null_wt,
                    "",
                    False,
                    None,
                    None,
                ),
                (
                    "wave0/outer_left_down",
                    r"\taxid{Abl-DownOnly} {\tiny(\texttt{outer\_left\_dn})}",
                    r"\tcfg{\fFull}{\fFull}{\fOuterFull^{\text{dn}}}{\fNone}{0}",
                    d_null_dense,
                    r"^{\S}",
                    False,
                    None,
                    None,
                ),
                (
                    "wave0/down_hybrid",
                    r"\taxid{Abl-DownSwap} {\tiny(\texttt{down\_hybrid})}",
                    r"\tcfg{\fFull}{\fNone}{\fFull^{\text{dn}}}{\fNone}{0}",
                    None,
                    "",
                    False,
                    None,
                    None,
                ),
                (
                    "wave1/left_all",
                    r"\taxid{Abl-OutOnly} {\tiny(\texttt{left\_all})}",
                    r"\tcfg{\fNone}{\fNone}{\fFull}{\fFull}{0}",
                    None,
                    "",
                    False,
                    None,
                    None,
                ),
                (
                    "wave1/inner_outer_all_diag",
                    r"\taxid{Abl-BothDiag} {\tiny(\texttt{all\_diag})}",
                    r"\tcfg{\fDiag}{\fDiag}{\fDiag}{\fDiag}{0}",
                    None,
                    "",
                    False,
                    None,
                    None,
                ),
            ],
        ),
    ]

    tmain = [
        "% Generated by scripts/render_results_viz.py from results/*_summary.json. Do not edit.",
        r"\begin{tabular}{@{}lcrrrrrr@{}}",
        r"\toprule",
        r"Arm (\texttt{key}) & $\mathcal{A}(R_{512},R_{2048}\mid L_{512},L_{2048}\mid\tau)$ & "
        r"Raw $\Delta$Sel & Net $\Delta$Sel & $\Delta$C4 & $\Delta$Train $\pm$ SE & "
        r"Step \% & GiB \\",
        r"\midrule",
    ]

    for fam_idx, (fam_title, fam_col, rows) in enumerate(families):
        if fam_idx > 0:
            tmain.append(r"\addlinespace[2.5pt]")
        tmain.append(
            rf"\multicolumn{{8}}{{@{{}}l}}{{\textcolor{{{fam_col}}}{{\pagesfsb {fam_title}}}}} \\"
        )
        for (
            run_key,
            arm_cell,
            tuple_cell,
            null_ref,
            net_suf,
            pareto,
            fast_step,
            fast_gib,
        ) in rows:
            rec = runs[run_key]
            prefix = r"\rowcolor{pagepanel} " if pareto else ""
            if run_key == "wave0/muont":
                tmain.append(
                    f"{prefix}{arm_cell} & {tuple_cell} & "
                    f"$0.0000$ & $0.0000$ & $0.0000$ & $0.0000 \\pm \\text{{---}}$ & "
                    f"$0.0$ & ${rec['peak_gib']:.1f}$ \\\\"
                )
                continue

            raw_s = f"{rec['d_sel_late']:+.4f}"
            if run_key == "wave2/inner_outer_mlpdiag":
                raw_cell = rf"$\mathbf{{{raw_s}}}$"
            else:
                raw_cell = f"${raw_s}$"

            if null_ref == "SELF":
                net_cell = "$0.0000$"
            elif null_ref is None:
                net_cell = "---"
            else:
                net_val = rec["d_sel_late"] - null_ref
                if run_key in (
                    "wave2/inner_outer_mlpdiag",
                    "wave4/inner_outer_mlpdiag_lag",
                ):
                    net_cell = rf"$\mathbf{{{net_val:+.4f}}}{net_suf}$"
                else:
                    net_cell = f"${net_val:+.4f}{net_suf}$"

            c4_cell = f"${rec['d_c4_late']:+.4f}$"
            se_str = f"{rec['d_train_se']:.5f}"[1:]  # .00004 compact form
            train_cell = f"${rec['d_train']:+.4f} \\pm {se_str}$"

            leg_step = f"{rec['step_overhead']:+.1f}"
            if fast_step is not None:
                step_cell = rf"${leg_step} \!\to\! \mathbf{{{fast_step:+.1f}}}$"
            elif run_key == "wave2/inner_outer_leftdiag":
                step_cell = rf"$\mathbf{{{leg_step}}}$"
            else:
                step_cell = f"${leg_step}$"

            if fast_gib is not None and abs(rec["peak_gib"] - fast_gib) >= 0.15:
                gib_cell = rf"${rec['peak_gib']:.1f} \!\to\! \mathbf{{{fast_gib:.1f}}}$"
            else:
                gib_cell = f"${rec['peak_gib']:.1f}$"

            tmain.append(
                f"{prefix}{arm_cell} & {tuple_cell} & {raw_cell} & {net_cell} & "
                f"{c4_cell} & {train_cell} & {step_cell} & {gib_cell} \\\\"
            )

    tmain += [r"\bottomrule", r"\end{tabular}"]
    write_file("results/table_main.tex", tmain)

    # ------------------------------------------------------------------
    # 2. Window Evolution Table: results/table_windows.tex
    # ------------------------------------------------------------------
    win_groups = [
        (
            r"II.\ Width-Thresholded Two-Sided (\texttt{WT}, Seed 0 \& Cross-Seed)",
            "famWT",
            [
                ("wave2/inner_outer_mlpdiag", r"\textbf{\taxid{WT-MLP}} (\texttt{s0})", r"Front-loaded $\to$ stable", True),
                ("wave4/inner_outer_mlpdiag_lag", r"\textbf{\taxid{WT-MLP-Lag}} (\texttt{s0})", r"Front-loaded $\to$ stable", True),
                ("wave3/inner_outer_mlpdiag_seed1", r"\taxid{WT-MLP} (\texttt{s1})", r"Front-loaded $\to$ stable", False),
                ("wave4/inner_outer_mlpdiag_seed2", r"\taxid{WT-MLP} (\texttt{s2})", r"Front-loaded $\to$ stable", False),
            ],
        ),
        (
            r"III.\ Pure-Diagonal Output (\texttt{OD})",
            "famOD",
            [
                ("wave2/inner_outer_leftdiag", r"\textbf{\taxid{OD-All}}", r"Flat persistent", True),
                ("wave10/outer_leftdiag_upstream_fused", r"\taxid{OD-Outer}$^{\dagger}$", r"Mid-run peak $\to$ mild", False),
            ],
        ),
        (
            r"IV.\ Dense Two-Sided Ceiling (\texttt{DT})",
            "famDT",
            [
                ("wave3/inner_outer_all_lag", r"\taxid{DT-All-Lag}", r"Front-loaded $\to$ stable", False),
                ("wave1/inner_outer_all", r"\taxid{DT-All}", r"Front-loaded $\to$ stable", False),
            ],
        ),
        (
            r"V.\ Matched Execution-Path Nulls (\texttt{Null})",
            "famNull",
            [
                ("wave3/null_mlpdiag", r"\taxid{Null-WT}", r"Accumulating path drift", False),
                ("wave1/null_all", r"\taxid{Null-Dense}", r"Accumulating path drift", False),
                ("wave10/null_leftdiag_upstream_fused", r"\taxid{Null-OD}$^{\dagger}$", r"Accumulating path drift", False),
            ],
        ),
        (
            r"VI.\ Structural \& Input-Degradation Ablations + \texttt{Muon}",
            "famAbl",
            [
                ("wave2/outer_left_all", r"\taxid{Abl-OuterDense}", r"Flat persistent", False),
                ("wave2/inner_outer_hidden", r"\taxid{Abl-No2048}", r"Delayed steady offset", False),
                ("wave0/outer_left_down", r"\taxid{Abl-DownOnly}", r"Within null floor", False),
                ("wave0/down_hybrid", r"\taxid{Abl-DownSwap}", r"Persistent deficit", False),
                ("wave0/muon", r"\taxid{Muon} (\texttt{Ref})", r"Unwhitened deficit", False),
                ("wave1/left_all", r"\taxid{Abl-OutOnly}", r"Divergent (no $R,\hat R$)", False),
                ("wave1/inner_outer_all_diag", r"\taxid{Abl-BothDiag}", r"Divergent (diag $R,\hat R$)", False),
            ],
        ),
    ]

    tw = [
        "% Generated by scripts/render_results_viz.py from results/summary.json. Do not edit.",
        r"\begin{tabular}{@{}lrrrrrl@{}}",
        r"\toprule",
        r"Arm & Win.\ 1 & Win.\ 2 & Win.\ 3 & Win.\ 4 & Win.\ 5 & Temporal regime \\",
        r"\midrule",
    ]
    for g_idx, (g_title, g_col, g_rows) in enumerate(win_groups):
        if g_idx > 0:
            tw.append(r"\addlinespace[2pt]")
        tw.append(
            rf"\multicolumn{{7}}{{@{{}}l}}{{\textcolor{{{g_col}}}{{\pagesfsb {g_title}}}}} \\"
        )
        for run_key, wlabel, regime, shade in g_rows:
            wins = runs[run_key]["windows"]
            wcells = " & ".join(f"${w:+.4f}$" for w in wins)
            prefix = r"\rowcolor{pagepanel} " if shade else ""
            tw.append(f"{prefix}{wlabel} & {wcells} & {{\\scriptsize {regime}}} \\\\")
    tw += [r"\bottomrule", r"\end{tabular}"]
    write_file("results/table_windows.tex", tw)

    # ------------------------------------------------------------------
    # 3. Cost-Quality Pareto Plane (Panel a): figs/plane_plots.tex
    # ------------------------------------------------------------------
    # All 15 Wave 0..4 seed-0 runs sharing the exact wave0/muont reference frame,
    # plus the 2 calibrated fast_lag frontier points (WT-MLP-Lag & DT-All-Lag).
    y_wt_lag = runs["wave4/inner_outer_mlpdiag_lag"]["d_sel_late"]
    x_wt_lag_leg = runs["wave4/inner_outer_mlpdiag_lag"]["step_overhead"]
    y_dt_lag = runs["wave3/inner_outer_all_lag"]["d_sel_late"]
    x_dt_lag_leg = runs["wave3/inner_outer_all_lag"]["step_overhead"]

    fam_plot_specs = [
        (
            "WT",
            "mark=*, mark size=2.3pt, color=famWT",
            r"Width-thresholded (\texttt{WT})",
            [
                ("wave2/inner_outer_mlpdiag", r"\textbf{\taxid{WT-MLP}}", "anchor=north, yshift=-2.8pt"),
                ("wave4/inner_outer_mlpdiag_lag", r"\taxid{WT-MLP-Lag}", "anchor=south, yshift=2.5pt"),
            ],
        ),
        (
            "OD",
            "mark=diamond*, mark size=2.6pt, color=famOD",
            r"Pure-diagonal (\texttt{OD})",
            [
                ("wave2/inner_outer_leftdiag", r"\textbf{\taxid{OD-All}}", "anchor=west, xshift=3.5pt"),
            ],
        ),
        (
            "DT",
            "mark=pentagon*, mark size=2.4pt, color=famDT",
            r"Dense two-sided (\texttt{DT})",
            [
                ("wave1/inner_outer_all", r"\taxid{DT-All}", "anchor=north, yshift=-2.8pt"),
                ("wave3/inner_outer_all_lag", r"\taxid{DT-All-Lag}", "anchor=south west, xshift=2pt, yshift=1.5pt"),
            ],
        ),
        (
            "Null",
            "mark=square*, mark size=2.1pt, color=famNull",
            r"Matched nulls (\texttt{Null})",
            [
                ("wave3/null_mlpdiag", r"\taxid{Null-WT}", "anchor=south, yshift=2.5pt"),
                ("wave1/null_all", r"\taxid{Null-Dense}", "anchor=south, yshift=2.5pt"),
            ],
        ),
        (
            "Abl",
            "mark=triangle*, mark size=2.3pt, color=famAbl",
            r"Ablations (\texttt{Abl})",
            [
                ("wave0/outer_left_down", r"\taxid{Abl-DownOnly}", "anchor=north west, xshift=2pt, yshift=-1.5pt"),
                ("wave2/inner_outer_hidden", r"\taxid{Abl-No2048}", "anchor=south west, xshift=2.5pt, yshift=1.5pt"),
                ("wave2/outer_left_all", r"\taxid{Abl-OuterDense}", "anchor=west, xshift=3pt"),
                ("wave0/down_hybrid", r"\taxid{Abl-DownSwap}", "anchor=west, xshift=3pt"),
                ("wave1/left_all", r"\taxid{Abl-OutOnly}", "anchor=east, xshift=-3pt"),
            ],
        ),
        (
            "Ref",
            "mark=oplus*, mark size=2.3pt, color=famRef",
            r"Baselines (\texttt{Ref})",
            [
                ("wave0/muon", r"\taxid{Muon}", "anchor=west, xshift=3pt"),
                ("wave0/muont", r"\taxid{MuonT}", "anchor=south west, xshift=1.5pt, yshift=2pt"),
            ],
        ),
    ]

    p_lines = [
        "% Generated by scripts/render_results_viz.py. Do not edit.",
        f"% Shaded net-of-null improvement zone (below Null-Dense = {d_null_dense:.6f})",
        f"\\fill[famWT!6] (axis cs:-5.5,-0.0082) rectangle (axis cs:44.5,{d_null_dense:.6f});",
        r"\draw[pagemuted!70, dashed, line width=0.55pt] (axis cs:-5.5,0) -- (axis cs:44.5,0);",
        r"\draw[pagemuted!45, dotted, line width=0.5pt] (axis cs:0,-0.0082) -- (axis cs:0,0.0108);",
        f"\\draw[famNull!85, densely dashed, line width=0.7pt] (axis cs:-5.5,{d_null_dense:.6f}) -- (axis cs:44.5,{d_null_dense:.6f});",
        f"\\node[font=\\tiny, text=famNull!90!black, anchor=north] at (axis cs:30.5,{d_null_dense - 0.00008:.6f}) {{null floor ($-0.0013$)}};",
    ]

    # Plot family scatter groups (legend entries) + individual zero-collision nodes
    for fam_key, style, leg_text, items in fam_plot_specs:
        coords = " ".join(
            f"({runs[rk]['step_overhead']:.3f},{runs[rk]['d_sel_late']:.6f})"
            for rk, _, _ in items
        )
        p_lines.append(f"\\addplot[only marks, {style}] coordinates {{{coords}}};")
        p_lines.append(f"\\addlegendentry{{{leg_text}}}")

    # Fast-lag frontier starred points + horizontal calibration intervals
    p_lines.append(
        f"\\addplot[only marks, mark=star, mark size=3.5pt, line width=0.9pt, color=famWT] "
        f"coordinates {{({fast_wt_muont:.3f},{y_wt_lag:.6f}) ({fast_dt_muont:.3f},{y_dt_lag:.6f})}};"
    )
    p_lines.append(r"\addlegendentry{\texttt{fast\_lag} frontier}")

    # Calibration range whiskers and Pareto shift arrows
    p_lines += [
        f"\\draw[famWT, line width=0.9pt] (axis cs:{fast_wt_muont_lo:.3f},{y_wt_lag:.6f}) -- (axis cs:{fast_wt_muont_hi:.3f},{y_wt_lag:.6f});",
        f"\\draw[famDT, line width=0.9pt] (axis cs:{fast_dt_muont_lo:.3f},{y_dt_lag:.6f}) -- (axis cs:{fast_dt_muont_hi:.3f},{y_dt_lag:.6f});",
        f"\\draw[-Latex, famWT, densely dashed, line width=0.85pt] "
        f"(axis cs:{x_wt_lag_leg - 0.55:.3f},{y_wt_lag:.6f}) -- (axis cs:{fast_wt_muont + 0.65:.3f},{y_wt_lag:.6f});",
        f"\\draw[-Latex, famDT, densely dotted, line width=0.8pt] "
        f"(axis cs:{x_dt_lag_leg - 0.55:.3f},{y_dt_lag:.6f}) to[bend left=10] (axis cs:{fast_dt_muont + 0.70:.3f},{y_dt_lag:.6f});",
    ]

    # Explicit per-point zero-collision labels
    color_by_fam = {
        "WT": "famWT",
        "OD": "famOD",
        "DT": "famDT!90!black",
        "Null": "famNull!90!black",
        "Abl": "famAbl!90!black",
        "Ref": "famRef",
    }
    for fam_key, _, _, items in fam_plot_specs:
        col = color_by_fam[fam_key]
        for rk, lbl, placement in items:
            xv = runs[rk]["step_overhead"]
            yv = runs[rk]["d_sel_late"]
            p_lines.append(
                f"\\node[font=\\tiny, text={col}, {placement}, inner sep=1pt] "
                f"at (axis cs:{xv:.3f},{yv:.6f}) {{{lbl}}};"
            )

    # Leader labels for the two starred fast_lag frontier points
    p_lines += [
        f"\\draw[famWT!75, line width=0.45pt] (axis cs:{fast_wt_muont:.3f},{y_wt_lag + 0.00024:.6f}) -- (axis cs:0.4,-0.00365);",
        r"\node[font=\tiny, text=famWT, anchor=west, fill=white, fill opacity=0.90, text opacity=1, inner sep=1pt] "
        f"at (axis cs:-5.1,-0.00335) {{\\textbf{{\\taxid{{WT-MLP-Lag}}}} (\\texttt{{fast}} $+{fast_wt_muont:.1f}\\%$)}};",
        f"\\draw[famDT!80, line width=0.45pt] (axis cs:{fast_dt_muont:.3f},{y_dt_lag - 0.00024:.6f}) -- (axis cs:1.5,-0.00720);",
        r"\node[font=\tiny, text=famDT!90!black, anchor=west, inner sep=1pt] "
        f"at (axis cs:-5.1,-0.00755) {{\\textbf{{\\taxid{{DT-All-Lag}}}} (\\texttt{{fast}} $+{fast_dt_muont:.1f}\\%$)}};",
    ]

    # Off-scale indicator for Abl-BothDiag (+0.0180 at +16.45% step)
    x_bd = runs["wave1/inner_outer_all_diag"]["step_overhead"]
    y_bd = runs["wave1/inner_outer_all_diag"]["d_sel_late"]
    p_lines += [
        f"\\draw[-Latex, famAbl, line width=0.85pt] (axis cs:{x_bd:.3f},0.0093) -- (axis cs:{x_bd:.3f},0.01055);",
        f"\\node[font=\\tiny, text=famAbl!90!black, anchor=west, inner sep=1.5pt] "
        f"at (axis cs:{x_bd + 0.6:.3f},0.00995) {{\\taxid{{Abl-BothDiag}} (${y_bd:+.4f}$)}};",
    ]

    write_file("figs/plane_plots.tex", p_lines)

    # ------------------------------------------------------------------
    # 4. Multi-Metric Causal Attribution Forest Plot (Panel b):
    #    figs/attribution.dat & figs/attribution_plots.tex
    # ------------------------------------------------------------------
    attr_spec = [
        # y_idx, run_key, y_label, null_val, color, is_positive_addition
        (12, "wave2/inner_outer_mlpdiag", r"\textbf{\taxid{WT-MLP}}", d_null_wt, "famWT", True),
        (11, "wave4/inner_outer_mlpdiag_lag", r"\textbf{\taxid{WT-MLP-Lag}}", d_null_wt, "famWT", True),
        (10, "wave3/inner_outer_all_lag", r"\taxid{DT-All-Lag}", d_null_dense, "famDT", True),
        (9, "wave1/inner_outer_all", r"\taxid{DT-All}", d_null_dense, "famDT", True),
        (8, "wave2/inner_outer_leftdiag", r"\textbf{\taxid{OD-All}}", d_null_dense, "famOD", True),
        (7, "wave2/outer_left_all", r"\taxid{Abl-OuterDense}", d_null_dense, "famAbl", True),
        (6, "wave2/inner_outer_hidden", r"\taxid{Abl-No2048}", d_null_wt, "famAbl", True),
        (5, "wave10/outer_leftdiag_upstream_fused", r"\taxid{OD-Outer}$^{\dagger}$", d_null_od, "famOD", True),
        (4, "wave0/outer_left_down", r"\taxid{Abl-DownOnly}", d_null_dense, "famAbl", True),
        (3, "wave0/down_hybrid", r"\taxid{Abl-DownSwap}", 0.0, "famAbl", False),
        (2, "wave0/muon", r"\taxid{Muon}", 0.0, "famRef", False),
        (1, "wave1/left_all", r"\taxid{Abl-OutOnly}", 0.0, "famAbl", False),
        (0, "wave1/inner_outer_all_diag", r"\taxid{Abl-BothDiag}", 0.0, "famAbl", False),
    ]

    attr_dat = [
        "y_idx arm d_sel_raw d_null d_sel_net d_c4 d_train d_train_2se"
    ]
    a_lines = [
        "% Generated by scripts/render_results_viz.py. Do not edit.",
        r"\fill[famWT!5] (axis cs:-0.0078,3.52) rectangle (axis cs:0.0,12.65);",
        r"\fill[famAbl!4] (axis cs:0.0,-0.65) rectangle (axis cs:0.0118,3.52);",
        r"\draw[pagemuted!75, line width=0.6pt] (axis cs:0,-0.65) -- (axis cs:0,12.65);",
        r"\draw[pagemuted!65, densely dashed, line width=0.55pt] (axis cs:-0.0078,3.52) -- (axis cs:0.0118,3.52);",
        r"\node[font=\tiny\itshape, text=pagemuted!95!black, anchor=south east] at (axis cs:0.0115,3.70) {Two-sided additions ($\uparrow$)};",
        r"\node[font=\tiny\itshape, text=famAbl!85!black, anchor=north west, align=left] at (axis cs:-0.0075,3.34) {Input removed /\\ degraded ($\downarrow$)};",
    ]

    # Legend proxy plots for Panel (b)
    a_lines += [
        r"\addplot[area legend, fill=famNull!42, draw=famNull!85] coordinates {(1,-10) (1,-10)};",
        r"\addlegendentry{Null path share}",
        r"\addplot[area legend, fill=famWT!75, draw=famWT!95] coordinates {(1,-10) (1,-10)};",
        r"\addlegendentry{Net geometry $\Delta$Sel}",
        r"\addplot[only marks, mark=diamond*, mark size=2.1pt, mark options={fill=white, draw=pageink, line width=0.65pt}] coordinates {(1,-10)};",
        r"\addlegendentry{$\Delta$C4}",
        r"\addplot[only marks, mark=*, mark size=1.4pt, color=pageink] coordinates {(1,-10)};",
        r"\addlegendentry{Paired $\Delta$Train ($\pm 2\,\mathrm{SE}$)}",
    ]

    c4_coords = []
    tr_coords = []
    bh = 0.24
    xmax_cap = 0.0113

    for y_idx, rk, _, null_v, col, is_add in attr_spec:
        rec = runs[rk]
        raw_v = rec["d_sel_late"]
        net_v = raw_v - null_v
        c4_v = rec["d_c4_late"]
        tr_v = rec["d_train"]
        se2 = 2.0 * rec["d_train_se"]
        short_id = rk.split("/")[-1]
        attr_dat.append(
            f"{y_idx} {short_id} {raw_v:.6f} {null_v:.6f} {net_v:.6f} "
            f"{c4_v:.6f} {tr_v:.6f} {se2:.6f}"
        )

        if is_add:
            # Segment 1: 0 -> null_v (Execution-path null share)
            a_lines.append(
                f"\\filldraw[fill=famNull!42, draw=famNull!85, line width=0.45pt] "
                f"(axis cs:0,{y_idx - bh:.2f}) rectangle (axis cs:{null_v:.6f},{y_idx + bh:.2f});"
            )
            # Segment 2: null_v -> raw_v (Net geometry gain)
            a_lines.append(
                f"\\filldraw[fill={col}!76, draw={col}!95, line width=0.5pt] "
                f"(axis cs:{null_v:.6f},{y_idx - bh:.2f}) rectangle (axis cs:{raw_v:.6f},{y_idx + bh:.2f});"
            )
        else:
            disp_end = min(raw_v, xmax_cap)
            a_lines.append(
                f"\\filldraw[fill={col}!48, draw={col}!88, line width=0.5pt] "
                f"(axis cs:0,{y_idx - bh:.2f}) rectangle (axis cs:{disp_end:.6f},{y_idx + bh:.2f});"
            )
            if raw_v > xmax_cap:
                a_lines.append(
                    f"\\draw[-Latex, {col}!95!black, line width=0.75pt] "
                    f"(axis cs:0.0104,{y_idx}) -- (axis cs:0.01165,{y_idx});"
                )
                a_lines.append(
                    f"\\node[font=\\tiny, text={col}!95!black, fill=white, fill opacity=0.88, text opacity=1, "
                    f"anchor=east, inner sep=1pt] at (axis cs:0.0103,{y_idx}) {{${raw_v:+.4f}$}};"
                )

        if c4_v <= xmax_cap:
            c4_coords.append(f"({c4_v:.6f},{y_idx})")
        if tr_v + se2 <= xmax_cap:
            tr_coords.append(f"({tr_v:.6f},{y_idx})")
            a_lines.append(
                f"\\draw[pageink, line width=0.65pt] "
                f"(axis cs:{tr_v - se2:.6f},{y_idx}) -- (axis cs:{tr_v + se2:.6f},{y_idx});"
            )
            a_lines.append(
                f"\\draw[pageink, line width=0.55pt] "
                f"(axis cs:{tr_v - se2:.6f},{y_idx - 0.11:.2f}) -- (axis cs:{tr_v - se2:.6f},{y_idx + 0.11:.2f});"
            )
            a_lines.append(
                f"\\draw[pageink, line width=0.55pt] "
                f"(axis cs:{tr_v + se2:.6f},{y_idx - 0.11:.2f}) -- (axis cs:{tr_v + se2:.6f},{y_idx + 0.11:.2f});"
            )

    a_lines.append(
        "\\addplot[only marks, forget plot, mark=diamond*, mark size=2.1pt, "
        "mark options={fill=white, draw=pageink, line width=0.65pt}] "
        f"coordinates {{{' '.join(c4_coords)}}};"
    )
    a_lines.append(
        "\\addplot[only marks, forget plot, mark=*, mark size=1.35pt, color=pageink] "
        f"coordinates {{{' '.join(tr_coords)}}};"
    )

    write_file("figs/attribution.dat", attr_dat)
    write_file("figs/attribution_plots.tex", a_lines)

    # ------------------------------------------------------------------
    # 5. Full-Budget Horizon Shift Data & Transfer Bars:
    #    figs/lr_horizon.dat & figs/transfer_bars.tex
    # ------------------------------------------------------------------
    lr_rec = 0.02262741696
    lr_2s2 = 0.064
    lr_4x = 0.09050966784

    hor_lines = [
        "lr muont wt_mlp od_all od_outer_fused od_outer_unfused",
        (
            f"{lr_rec:.9f} "
            f"{runs['wave10/muont']['sel_late']:.6f} "
            f"{runs['wave2/inner_outer_mlpdiag']['sel_late']:.6f} "
            f"{runs['wave2/inner_outer_leftdiag']['sel_late']:.6f} "
            f"{runs['wave10/outer_leftdiag_fused']['sel_late']:.6f} "
            f"nan"
        ),
        (
            f"{lr_2s2:.9f} "
            f"{runs['wave12/muont']['sel_late']:.6f} "
            f"nan "
            f"nan "
            f"{runs['wave12/outer_leftdiag_fused']['sel_late']:.6f} "
            f"nan"
        ),
        (
            f"{lr_4x:.9f} "
            f"nan "
            f"nan "
            f"{runs['wave12/inner_outer_leftdiag']['sel_late']:.6f} "
            f"nan "
            f"{runs['wave12/outer_leftdiag']['sel_late']:.6f}"
        ),
    ]
    write_file("figs/lr_horizon.dat", hor_lines)

    # Wave-12 transfer bars separating matched-LR (0.064) win from mismatched-LR (0.0905) cliff
    d_w12_fused = runs["wave12/outer_leftdiag_fused"]["d_sel_late"]
    d_w12_alldiag = runs["wave12/inner_outer_leftdiag"]["d_sel_late"]
    d_w12_outer = runs["wave12/outer_leftdiag"]["d_sel_late"]
    tbars = [
        "% Generated by scripts/render_results_viz.py. Do not edit.",
        f"\\addplot[ybar, bar width=13pt, bar shift=0pt, fill=famOD!75, draw=famOD!95, "
        f"nodes near coords={{\\pgfmathprintnumber[fixed, precision=3, print sign]{{\\pgfplotspointmeta}}}}, "
        f"every node near coord/.append style={{font=\\tiny, text=famOD!95!black, anchor=north, yshift=-1pt}}] "
        f"coordinates {{(OD-Outer-F,{d_w12_fused:.6f})}};",
        f"\\addplot[ybar, bar width=13pt, bar shift=0pt, fill=famAbl!70, draw=famAbl!92, "
        f"nodes near coords={{\\pgfmathprintnumber[fixed, precision=3, print sign]{{\\pgfplotspointmeta}}}}, "
        f"every node near coord/.append style={{font=\\tiny, text=famAbl!95!black, anchor=south, yshift=1pt}}] "
        f"coordinates {{(OD-All,{d_w12_alldiag:.6f}) (OD-Outer,{d_w12_outer:.6f})}};",
    ]
    write_file("figs/transfer_bars.tex", tbars)

    # ------------------------------------------------------------------
    # 6. Pure-TikZ Spatial + Temporal Architecture Figure:
    #    figs/method_arch.tex
    # ------------------------------------------------------------------
    cap_ms = sys_cal["capture_ovh_fast_mlpdiag"]
    upd_ms = sys_cal["upd_ovh_fast_mlpdiag"]
    saved_gib = sys_mem["fast_mlpdiag_saved_gib"]
    wt_muon_lo = sys_tot["fast_mlpdiag_lag_vs_muon_pct_lo"]
    wt_muon_hi = sys_tot["fast_mlpdiag_lag_vs_muon_pct_hi"]
    dt_muon_lo = sys_tot["fast_all_lag_vs_muon_pct_lo"]
    dt_muon_hi = sys_tot["fast_all_lag_vs_muon_pct_hi"]

    arch = [
        "% Generated by scripts/render_results_viz.py from systems_summary.json & mechanism_summary.json. Do not edit.",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tikzpicture}[x=1cm, y=1cm, >=Latex]",
        r"  % ======================================================================",
        r"  % PANEL (a): Spatial Width-Thresholding in Transformer Layer \ell",
        r"  % ======================================================================",
        r"  \filldraw[fill=white, draw=pagerule, rounded corners=4pt, line width=0.6pt] (0.0,5.52) rectangle (15.40,10.88);",
        r"  \fill[pagepanel, rounded corners=4pt] (0.04,10.30) rectangle (15.36,10.84);",
        r"  \node[anchor=west, font=\pagesfsb\footnotesize\color{pageink}] at (0.24,10.57)",
        r"    {(a) Spatial Width-Thresholding in Transformer Block $\ell$ (\taxid{WT-MLP} / \mlpdiag{})};",
        r"",
        r"  % --- Left Sub-block: Multi-Head Attention (d = 512) ---",
        r"  \filldraw[fill=famWT!4, draw=famWT!30, rounded corners=3pt, line width=0.5pt] (0.20,6.22) rectangle (7.48,10.14);",
        r"  \node[anchor=north west, font=\pagesfsb\scriptsize\color{famWT}] at (0.34,10.06)",
        r"    {Attention Sub-Block ($d = 512$)};",
        r"",
        r"  % Attention Row 1: RMSNorm -> Q,K,V <- Dense 512",
        r"  \filldraw[fill=white, draw=famWT!65, rounded corners=2pt, line width=0.5pt] (0.34,8.50) rectangle (2.18,9.46);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (1.26,8.98)",
        r"    {\texttt{RMSNorm}$(h_\ell)$\\ $\fFull$ \texttt{in:attn} ($512$)};",
        r"  \draw[->, famWT!80, line width=0.6pt] (2.18,8.98) -- (2.48,8.98);",
        r"  \filldraw[fill=white, draw=famWT!80, rounded corners=2pt, line width=0.6pt] (2.48,8.50) rectangle (4.54,9.46);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (3.51,8.98)",
        r"    {$W_q,\,W_k,\,W_v$\\ {\color{pagemuted}$3 \times (512\times 512)$}};",
        r"  \draw[->, famWT, densely dashed, line width=0.65pt] (4.86,8.98) -- (4.54,8.98);",
        r"  \filldraw[fill=famWT!12, draw=famWT!85, rounded corners=2pt, line width=0.6pt] (4.86,8.50) rectangle (7.34,9.46);",
        r"  \node[align=center, font=\tiny\color{famWT!95!black}] at (6.10,8.98)",
        r"    {$\fFull$ \textbf{Dense $512\times 512$}\\ $\delta_q,\delta_k,\delta_v$ ($R^2\!=\!31\text{--}72\%$)};",
        r"",
        r"  % Attention mixing -> Row 2: W_o",
        r"  \draw[->, pageink!70, line width=0.6pt] (3.51,8.50) -- node[right, font=\tiny\color{pagemuted}] {RoPE + SDPA} (3.51,7.56);",
        r"  \filldraw[fill=white, draw=famWT!65, rounded corners=2pt, line width=0.5pt] (0.34,6.44) rectangle (2.18,7.50);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (1.26,6.97)",
        r"    {attn context\\ $\fFull$ \texttt{in:o} ($512$)};",
        r"  \draw[->, famWT!80, line width=0.6pt] (2.18,6.97) -- (2.48,6.97);",
        r"  \filldraw[fill=white, draw=famWT!80, rounded corners=2pt, line width=0.6pt] (2.48,6.44) rectangle (4.54,7.50);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (3.51,6.97)",
        r"    {$W_o \to h'_\ell$\\ {\color{pagemuted}$512\times 512$ (residual)}};",
        r"  \draw[->, famWT, densely dashed, line width=0.65pt] (4.86,6.97) -- (4.54,6.97);",
        r"  \filldraw[fill=famWT!14, draw=famWT!90, rounded corners=2pt, line width=0.65pt] (4.86,6.44) rectangle (7.34,7.50);",
        r"  \node[align=center, font=\tiny\color{famWT!95!black}] at (6.10,6.97)",
        r"    {$\fFull$ \textbf{Dense $512\times 512$}\\ $\delta_o$ ($>99\%$ rotational)};",
        r"",
        r"  % --- Right Sub-block: SwiGLU MLP (d = 512 <-> 2048) ---",
        r"  \filldraw[fill=famOD!4, draw=famOD!32, rounded corners=3pt, line width=0.5pt] (7.74,6.22) rectangle (15.20,10.14);",
        r"  \node[anchor=north west, font=\pagesfsb\scriptsize\color{famOD}] at (7.88,10.06)",
        r"    {SwiGLU Feed-Forward Sub-Block ($512 \leftrightarrow 2048$)};",
        r"",
        r"  % MLP Row 1: RMSNorm -> W_gate, W_up <- 2D-Tiled Diag 2048",
        r"  \filldraw[fill=white, draw=famWT!65, rounded corners=2pt, line width=0.5pt] (7.88,8.50) rectangle (9.70,9.46);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (8.79,8.98)",
        r"    {\texttt{RMSNorm}$(h'_\ell)$\\ $\fFull$ \texttt{in:mlp} ($512$)};",
        r"  \draw[->, famWT!80, line width=0.6pt] (9.70,8.98) -- (10.00,8.98);",
        r"  \filldraw[fill=white, draw=famOD!90, rounded corners=2pt, line width=0.75pt] (10.00,8.50) rectangle (12.10,9.46);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (11.05,8.98)",
        r"    {$W_{\text{gate}},\,W_{\text{up}}$ ($s\!=\!2$)\\ {\color{famOD!90!black}$2 \times (\mathbf{2048}\times 512)$}};",
        r"  \draw[->, famOD, densely dashed, line width=0.75pt] (12.42,8.98) -- (12.10,8.98);",
        r"  \filldraw[fill=famOD!15, draw=famOD, rounded corners=2.5pt, line width=0.85pt] (12.42,8.38) rectangle (15.06,9.56);",
        r"  \node[align=center, font=\tiny\color{famOD!95!black}] at (13.74,8.97)",
        r"    {$\fDiag$ \textbf{2D-Tiled Diag $2048$}\\ $\delta_{\text{gate}},\delta_{\text{up}}$ ($\rho_{\mathrm{rms}}\!\approx\!N_t^{-1/2}$)\\ "
        f"$82\\%/78\\%$ diag; $-{saved_gib:.2f}$\\,GiB}};",
        r"",
        r"  % Hadamard SwiGLU gate -> Row 2: W_down",
        r"  \draw[->, famOD!85, line width=0.6pt] (11.05,8.50) -- (11.05,8.22);",
        r"  \filldraw[fill=famOD!13, draw=famOD!75, rounded corners=6pt, line width=0.55pt] (9.96,7.80) rectangle (12.14,8.20);",
        r"  \node[font=\tiny\color{famOD!95!black}] at (11.05,8.00) {$\operatorname{SiLU}(g)\odot u$ (Hadamard)};",
        r"  \draw[->, pageink!70, line width=0.6pt] (11.05,7.80) -- (11.05,7.50);",
        r"",
        r"  \filldraw[fill=white, draw=famWT!65, rounded corners=2pt, line width=0.5pt] (7.88,6.44) rectangle (9.70,7.50);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (8.79,6.97)",
        r"    {$\fFull$ \texttt{in:down} ($2048$)\\ {\color{pagemuted}dense ($\mu\mu^{\Tr}\!>\!0$)}};",
        r"  \draw[->, famWT!80, line width=0.6pt] (9.70,6.97) -- (10.00,6.97);",
        r"  \filldraw[fill=white, draw=famWT!80, rounded corners=2pt, line width=0.6pt] (10.00,6.44) rectangle (12.10,7.50);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (11.05,6.97)",
        r"    {$W_{\text{down}} \to h_{\ell+1}$\\ {\color{pagemuted}$512\times 2048$ (residual)}};",
        r"  \draw[->, famWT, densely dashed, line width=0.65pt] (12.42,6.97) -- (12.10,6.97);",
        r"  \filldraw[fill=famWT!14, draw=famWT!90, rounded corners=2pt, line width=0.65pt] (12.42,6.44) rectangle (15.06,7.50);",
        r"  \node[align=center, font=\tiny\color{famWT!95!black}] at (13.74,6.97)",
        r"    {$\fFull$ \textbf{Dense $512\times 512$}\\ $\delta_{\text{down}}$ ($>99\%$ rotational)};",
        r"",
        r"  % Panel (a) Full-Width Legend Strip",
        r"  \fill[pagepanel, rounded corners=2pt] (0.20,5.64) rectangle (15.20,6.08);",
        r"  \node[font=\tiny\color{pageink}] at (7.70,5.86)",
        r"    {$\fFull$ \textbf{Dense spectral roots:} $4$ right-inputs ($C_t$) + $5$ width-$512$ left-cotangents \qquad$\mid$\qquad $\fDiag$ \textbf{Age-$0$ inline diagonal:} $2$ width-$2048$ pre-SwiGLU cotangents ($-96.2\%$ output $O(d^3)$ work)};",
        r"",
        r"  % ======================================================================",
        r"  % PANEL (b): Asynchronous 1-Step Lagged Pipeline (fast_lag, \tau = 1)",
        r"  % ======================================================================",
        r"  \filldraw[fill=white, draw=pagerule, rounded corners=4pt, line width=0.6pt] (0.0,0.0) rectangle (15.40,5.24);",
        r"  \fill[pagepanel, rounded corners=4pt] (0.04,4.66) rectangle (15.36,5.20);",
        r"  \node[anchor=west, font=\pagesfsb\footnotesize\color{pageink}] at (0.24,4.93)",
        r"    {(b) Asynchronous 1-Step Lagged Pipeline (\taxid{WT-MLP-Lag}, \texttt{fast\_lag}, $\tau = 1$)};",
        r"",
        r"  % Step timeline banner",
        r"  \draw[->, pagemuted, line width=0.55pt] (2.56,4.34) -- (15.18,4.34);",
        r"  \draw[pagemuted, line width=0.65pt] (7.68,4.22) -- (7.68,4.46);",
        r"  \node[font=\pagesfsb\tiny\color{pageink}, anchor=south] at (5.12,4.35) {Step $t$};",
        r"  \node[font=\pagesfsb\tiny\color{famWT}, anchor=south] at (11.43,4.35) {Step $t+1$ ($\approx 900$\,ms Fwd+Bwd overlap window)};",
        r"",
        r"  % --- Lane 1: Main CUDA Stream ---",
        r"  \node[align=left, anchor=west, font=\pagesfsb\scriptsize\color{pageink}] at (0.24,3.70)",
        r"    {Main Stream\\ {\tiny\color{pagemuted}(Compute-Bound)}};",
        f"  \\filldraw[fill=pagepanel, draw=pagemuted!80, rounded corners=2pt, line width=0.55pt] (2.56,3.24) rectangle (5.36,4.16);",
        f"  \\node[align=center, font=\\tiny\\color{{pageink}}] at (3.96,3.70)",
        f"    {{\\textbf{{Fwd+Bwd MB 0..3 $(t)$}}\\\\ 4 block hooks $(+{cap_ms:.1f}\\,\\text{{ms}})$}};",
        r"",
        f"  \\filldraw[fill=famWT!15, draw=famWT, rounded corners=2pt, line width=0.8pt] (5.48,3.24) rectangle (7.58,4.16);",
        f"  \\node[align=center, font=\\tiny\\color{{famWT!95!black}}] at (6.53,3.70)",
        f"    {{\\textbf{{Ozaki Update$(t)$}}\\\\ $L^{{(t-1)}},\\hat L^{{(t-1)}}$ $(+{upd_ms:.1f}\\,\\text{{ms}})$}};",
        r"",
        r"  \filldraw[fill=pagepanel, draw=pagemuted!80, rounded corners=2pt, line width=0.55pt] (7.78,3.24) rectangle (10.18,4.16);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (8.98,3.70)",
        r"    {\textbf{MB 0 $(t+1)$}\\ $\approx 235$\,ms (waits \texttt{ev\_rows})};",
        r"",
        r"  \filldraw[fill=pagepanel, draw=pagemuted!80, rounded corners=2pt, line width=0.55pt] (10.30,3.24) rectangle (13.48,4.16);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (11.89,3.70)",
        r"    {\textbf{MB 1..3 $(t+1)$}\\ $\approx 665$\,ms Fwd+Bwd (polls readiness)};",
        r"",
        r"  \filldraw[fill=famWT!15, draw=famWT, rounded corners=2pt, line width=0.8pt] (13.60,3.24) rectangle (15.18,4.16);",
        r"  \node[align=center, font=\tiny\color{famWT!95!black}] at (14.39,3.70)",
        r"    {\textbf{Update$(t\!+\!1)$}\\ uses $L^{(t)},\hat L^{(t)}$};",
        r"",
        r"  % --- Lane 2: Priority-0 Output Side Stream ---",
        r"  \node[align=left, anchor=west, font=\pagesfsb\scriptsize\color{famOD}] at (0.24,2.14)",
        r"    {Output Stream\\ {\tiny\color{pagemuted}(Priority-0 + Host)}};",
        r"  \filldraw[fill=famOD!15, draw=famOD, rounded corners=2pt, line width=0.65pt] (5.36,1.68) rectangle (6.84,2.60);",
        r"  \node[align=center, font=\tiny\color{famOD!95!black}] at (6.10,2.14)",
        r"    {\textbf{\texttt{syrk}}\\ packed ($1.4$\,ms)};",
        r"",
        r"  \filldraw[fill=famDT!16, draw=famDT!95!black, rounded corners=2pt, line width=0.65pt] (6.96,1.68) rectangle (10.36,2.60);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (8.66,2.14)",
        r"    {\textbf{\texttt{reduce\_scatter} + Graph Roots}\\ {\color{pagemuted}(overlaps Update$(t)$ + MB 0)}};",
        r"",
        r"  \filldraw[fill=famNull!15, draw=famNull!90!black, rounded corners=2pt, line width=0.65pt] (10.48,1.68) rectangle (13.56,2.60);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (12.02,2.14)",
        r"    {\textbf{Mid-Step Repair / \texttt{eigh}}\\ {\color{pagemuted}+\,\texttt{unitRMS} + FP16 Split ($\sim 700$\,ms early)}};",
        r"",
        r"  % Dedicated Cross-Stream Event Band (y = 2.60 .. 3.24, zero text clutter)",
        r"  \draw[->, famOD!90!black, densely dashed, line width=0.7pt] (5.16,3.24) -- (5.56,2.60);",
        r"  \draw[->, famOD!90!black, line width=0.75pt] (6.72,2.60) -- node[pos=0.50, fill=white, inner sep=1pt, font=\tiny\color{famOD!95!black}] {\texttt{ev\_rows} ($0$\,ms stall)} (8.04,3.24);",
        r"  \draw[->, famNull!90!black, densely dotted, line width=0.7pt] (10.66,3.24) -- (10.92,2.60);",
        r"  \draw[->, famWT, line width=0.85pt] (13.38,2.60) -- node[pos=0.50, right, font=\tiny\color{famWT}] {$L^{(t)},\hat L^{(t)}$} (13.86,3.24);",
        r"",
        r"  % Bottom Split: Eliminated Bottlenecks (left) + Calibrated 8-A100 Summary (right)",
        r"  \filldraw[fill=famAbl!5, draw=famAbl!45, densely dashed, rounded corners=2.5pt, line width=0.5pt] (0.20,0.16) rectangle (9.96,1.46);",
        r"  \node[align=left, anchor=west, font=\tiny\color{pageink}] at (0.34,0.81)",
        r"    {{\color{famAbl!95!black}\pagesfsb Eliminated legacy bottlenecks:} "
        r"(1) Stream-local timer drops per-step \texttt{cuda.synchronize()}.\\"
        r"     (2) Mid-step \texttt{select\_microbatch(1..3)} hides \texttt{eigh} fallbacks ($+89\text{--}230\,\text{ms} \to +6.6\text{--}21\,\text{ms}$).\\"
        r"     (3) 4 block-level hooks + Ozaki Tensor-Core GEMMs replace 896 custom-op wrappers \& SIMT FP32.};",
        r"",
        r"  \filldraw[fill=pagepanel, draw=famWT!45, rounded corners=2.5pt, line width=0.55pt] (10.12,0.16) rectangle (15.20,1.46);",
        r"  \node[align=left, anchor=west, font=\tiny\color{pageink}] at (10.26,0.81)",
        f"    {{{{\\color{{famWT}}\\pagesfsb Calibrated 8-A100 \\texttt{{fast\\_lag}}:}}\\\\"
        f"     \\textbf{{\\taxid{{WT-MLP-Lag}}}}: $\\mathbf{{+{wt_muon_lo:.1f}\\%\\text{{--}}{wt_muon_hi:.1f}\\%}}$ vs.\\ Muon\\\\"
        f"     ($+{fast_wt_muont_lo:.1f}\\%\\text{{--}}{fast_wt_muont_hi:.1f}\\%$ vs.\\ MuonT; ${fast_wt_gib:.2f}$\\,GiB)\\\\"
        f"     \\taxid{{DT-All-Lag}}: $\\mathbf{{+{dt_muon_lo:.1f}\\%\\text{{--}}{dt_muon_hi:.1f}\\%}}$ vs.\\ Muon}};",
        r"\end{tikzpicture}%",
        r"}",
    ]
    write_file("figs/method_arch.tex", arch)


if __name__ == "__main__":
    main()
