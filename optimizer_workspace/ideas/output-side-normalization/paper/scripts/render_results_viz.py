"""Systematic taxonomy, structured tables, and vector visualization generator.

Reads ONLY the materialized JSON artifacts in paper/results/:
  - results/summary.json
  - results/lr_summary.json
  - results/mechanism_summary.json
  - results/systems_summary.json
  - results/measured_8gpu_timings.json (matched Table 5 and Figure 3 costs)

Emits:
  - results/table_main.tex       Taxonomic master comparison with 5-slot badges,
                                 Raw vs. Net-of-Null selection delta, and
                                 executed 8xA100 fast_lag & WT-Sym / WT-Sym-Lag rows
  - results/table_windows.tex    Taxonomically grouped 5-window paired evolution
                                 table with temporal regime classification
  - figs/method_arch.tex         Pure-TikZ 2-panel architecture & fast_lag
                                 asynchronous timeline diagram
  - figs/plane_plots.tex         Cost-quality plane with matched eight-A100
                                 mean timings and full-budget quality results
  - figs/attribution.dat         Machine-readable causal attribution table
  - figs/attribution_plots.tex   Panel (b) multi-metric causal attribution
                                 forest plot (Null share + Net gain + C4 + Train)
  - figs/lr_horizon.dat          Full-budget 5,120-step horizon-shift series
  - figs/transfer_bars.tex       Wave-12 transferred-rate delta bars

Run from paper/:
    python -I scripts/render_results_viz.py
"""

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--table-only", action="store_true", help="Regenerate only the master comparison table.")
    mode.add_argument("--results-only", action="store_true", help="Regenerate only Table 5 and Figure 3.")
    args = parser.parse_args()
    runs = load_json("summary.json")
    lr_data = load_json("lr_summary.json")
    mech = load_json("mechanism_summary.json")
    sys_data = load_json("systems_summary.json")
    measured = load_json("measured_8gpu_timings.json")
    if measured["missing_methods"] or measured["covered_methods"] != measured["total_methods"]:
        raise ValueError("Table 5 and Figure 3 require measured eight-A100 costs for every method")
    protocol = measured["protocol"]
    if protocol["world_size"] != 8 or protocol["benchmark_window"] != [37, 96]:
        raise ValueError("Table 5 and Figure 3 require the matched eight-A100 benchmark protocol")
    bench = {
        arm: entry["measured_benchmark"]
        for arm, entry in measured["methods"].items()
    }
    baseline_mean = bench["muont"]["mean_step_seconds"]
    mean_overhead = {}
    for arm, costs in bench.items():
        if costs["window"] != protocol["benchmark_window"] or costs["measured_updates"] != 60:
            raise ValueError(f"{arm}: costs do not use the shared 60-update window")
        mean_overhead[arm] = 100 * (costs["mean_step_seconds"] / baseline_mean - 1)

    sys_cal = sys_data["calibrated_8gpu_ms"]
    sys_mem = sys_data["memory_32l_isolated"]
    ex_sys = sys_data["executed_8gpu_wave_sym_fastlag"]

    # Matched-null selection deltas for causal net-of-null attribution.
    d_null_wt = runs["wave3/null_mlpdiag"]["d_sel_late"]
    d_null_dense = runs["wave1/null_all"]["d_sel_late"]
    d_null_od = runs["wave10/null_leftdiag_upstream_fused"]["d_sel_late"]

    # Full-budget execution measurements remain in the architecture diagram.
    exec_wt_lag_muont = ex_sys["wt_mlp_lag"]["vs_muont_pct"]
    exec_wt_lag_gib = ex_sys["wt_mlp_lag"]["peak_gib"]
    exec_sym_lag_muont = ex_sys["wt_sym_lag"]["vs_muont_pct"]
    exec_sym_lag_gib = ex_sys["wt_sym_lag"]["peak_gib"]

    # ------------------------------------------------------------------
    # 1. Master Results Table: results/table_main.tex
    # ------------------------------------------------------------------
    # Tuple schema per row:
    # (run_key, arm_cell, tuple_cell, null_ref, net_suffix, pareto_shade)
    families = [
        (
            r"I.\ Baselines (\texttt{Ref})",
            "famRef",
            [
                (
                    "wave0/muont",
                    r"\taxid{MuonT} {\scriptsize(\muont{})}",
                    r"\tcfg{\fFull}{\fFull}{\fNone}{\fNone}{0}",
                    0.0,
                    "",
                    False,
                ),
                (
                    "wave0/muon",
                    r"\taxid{Muon} {\scriptsize(\texttt{muon})}",
                    r"\tcfg{\fNone}{\fNone}{\fNone}{\fNone}{\text{---}}",
                    None,
                    "",
                    False,
                ),
            ],
        ),
        (
            r"II.\ Width-Thresholded Two-Sided (\texttt{WT} --- Asymmetric \& Symmetric $2\times 2$ Factorial)",
            "famWT",
            [
                (
                    "wave2/inner_outer_mlpdiag",
                    r"\textbf{\taxid{WT-MLP}} {\scriptsize(\texttt{mlpdiag})}",
                    r"\tcfg{\fFull}{\fFull}{\fFull}{\fDiag}{0}",
                    d_null_wt,
                    "",
                    True,
                ),
                (
                    "wave4/inner_outer_mlpdiag_lag",
                    r"\textbf{\taxid{WT-MLP-Lag}} {\scriptsize(\texttt{mlpdiag\_lag}, W4)}",
                    r"\tcfg{\fFull}{\fFull}{\fFull}{\fDiag}{1}",
                    d_null_wt,
                    "",
                    True,
                ),
                (
                    "wave_sym_fastlag/inner_outer_mlpdiag_lag",
                    r"\textbf{\taxid{WT-MLP-Lag}}$^{\ddagger}$ {\scriptsize(\texttt{fast\_lag})}",
                    r"\tcfg{\fFull}{\fFull}{\fFull}{\fDiag}{1}",
                    d_null_wt,
                    r"^{\ddagger}",
                    True,
                ),
                (
                    "wave_sym_fastlag/inner_outer_sym_mlpdiag",
                    r"\textbf{\taxid{WT-Sym}}$^{\ddagger}$ {\scriptsize(\texttt{sym\_mlpdiag})}",
                    r"\tcfg{\fFull}{\fDiag}{\fFull}{\fDiag}{0}",
                    d_null_wt,
                    r"^{\ddagger}",
                    True,
                ),
                (
                    "wave_sym_fastlag/inner_outer_sym_mlpdiag_lag",
                    r"\textbf{\taxid{WT-Sym-Lag}}$^{\ddagger}$ {\scriptsize(\texttt{sym\_mlpdiag\_lag})}",
                    r"\tcfg{\fFull}{\fDiag}{\fFull}{\fDiag}{1}",
                    d_null_wt,
                    r"^{\ddagger}",
                    True,
                ),
            ],
        ),
        (
            r"III.\ Pure-Diagonal Output (\texttt{OD} --- Zero-Eigendecomposition Limit)",
            "famOD",
            [
                (
                    "wave2/inner_outer_leftdiag",
                    r"\textbf{\taxid{OD-All}} {\scriptsize(\texttt{leftdiag})}",
                    r"\tcfg{\fFull}{\fFull}{\fDiag}{\fDiag}{0}",
                    d_null_dense,
                    r"^{\S}",
                    True,
                ),
                (
                    "wave10/outer_leftdiag_upstream_fused",
                    r"\taxid{OD-Outer}$^{\dagger}$ {\scriptsize(\texttt{outer\_leftdiag\_uf})}",
                    r"\tcfg{\fFull}{\fFull}{\fOuterDiag}{\fOuterDiag}{0}",
                    d_null_od,
                    "",
                    False,
                ),
            ],
        ),
        (
            r"IV.\ Dense Two-Sided Ceiling (\texttt{DT})",
            "famDT",
            [
                (
                    "wave3/inner_outer_all_lag",
                    r"\taxid{DT-All-Lag} {\scriptsize(\texttt{all\_lag})}",
                    r"\tcfg{\fFull}{\fFull}{\fFull}{\fFull}{1}",
                    d_null_dense,
                    "",
                    False,
                ),
                (
                    "wave1/inner_outer_all",
                    r"\taxid{DT-All} {\scriptsize(\texttt{inner\_outer\_all})}",
                    r"\tcfg{\fFull}{\fFull}{\fFull}{\fFull}{0}",
                    d_null_dense,
                    "",
                    False,
                ),
            ],
        ),
        (
            r"V.\ Matched Execution-Path Nulls (\texttt{Null} --- Identity Left Factors)",
            "famNull",
            [
                (
                    "wave3/null_mlpdiag",
                    r"\taxid{Null-WT} {\scriptsize(\texttt{null\_mlpdiag})}",
                    r"\tcfg{\fFull}{\fFull}{\fNullFull}{\fNullDiag}{0}",
                    "SELF",
                    "",
                    False,
                ),
                (
                    "wave1/null_all",
                    r"\taxid{Null-Dense} {\scriptsize(\texttt{null\_all})}",
                    r"\tcfg{\fFull}{\fFull}{\fNullFull}{\fNullFull}{0}",
                    "SELF",
                    "",
                    False,
                ),
                (
                    "wave10/null_leftdiag_upstream_fused",
                    r"\taxid{Null-OD}$^{\dagger}$ {\scriptsize(\texttt{null\_leftdiag\_uf})}",
                    r"\tcfg{\fFull}{\fFull}{\fNullDiag}{\fNullDiag}{0}",
                    "SELF",
                    "",
                    False,
                ),
            ],
        ),
        (
            r"VI.\ Structural \& Input-Degradation Ablations (\texttt{Abl})",
            "famAbl",
            [
                (
                    "wave2/outer_left_all",
                    r"\taxid{Abl-OuterDense} {\scriptsize(\texttt{outer\_left\_all})}",
                    r"\tcfg{\fFull}{\fFull}{\fOuterFull}{\fOuterFull}{0}",
                    d_null_dense,
                    "",
                    False,
                ),
                (
                    "wave2/inner_outer_hidden",
                    r"\taxid{Abl-No2048} {\scriptsize(\texttt{hidden})}",
                    r"\tcfg{\fFull}{\fFull}{\fFull}{\fNone}{0}",
                    d_null_wt,
                    "",
                    False,
                ),
                (
                    "wave0/outer_left_down",
                    r"\taxid{Abl-DownOnly} {\scriptsize(\texttt{outer\_left\_dn})}",
                    r"\tcfg{\fFull}{\fFull}{\fOuterFull^{\text{dn}}}{\fNone}{0}",
                    d_null_dense,
                    r"^{\S}",
                    False,
                ),
                (
                    "wave0/down_hybrid",
                    r"\taxid{Abl-DownSwap} {\scriptsize(\texttt{down\_hybrid})}",
                    r"\tcfg{\fFull}{\fNone}{\fFull^{\text{dn}}}{\fNone}{0}",
                    None,
                    "",
                    False,
                ),
                (
                    "wave1/left_all",
                    r"\taxid{Abl-OutOnly} {\scriptsize(\texttt{left\_all})}",
                    r"\tcfg{\fNone}{\fNone}{\fFull}{\fFull}{0}",
                    None,
                    "",
                    False,
                ),
                (
                    "wave1/inner_outer_all_diag",
                    r"\taxid{Abl-BothDiag} {\scriptsize(\texttt{all\_diag})}",
                    r"\tcfg{\fDiag}{\fDiag}{\fDiag}{\fDiag}{0}",
                    None,
                    "",
                    False,
                ),
            ],
        ),
    ]

    tmain = [
        "% Generated by scripts/render_results_viz.py. Costs: results/measured_8gpu_timings.json. Do not edit.",
        r"\begin{tabular}{@{}lcrrrrrr@{}}",
        r"\toprule",
        r"Arm (\texttt{key}) & $\mathcal{A}(R_{512},R_{2048}\mid L_{512},L_{2048}\mid\tau)$ & "
        r"Raw $\Delta$Sel & Net $\Delta$Sel & $\Delta$C4 & $\Delta$Train $\pm$ SE & "
        r"Mean Step \% & Peak GiB \\",
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
        ) in rows:
            rec = runs[run_key]
            arm = run_key.split("/", 1)[1]
            costs = bench[arm]
            step_pct = mean_overhead[arm]
            step_cell = f"${step_pct:+.1f}$" if arm != "muont" else "$0.0$"
            gib_cell = f"${costs['peak_allocated_gib']:.1f}$"
            prefix = r"\rowcolor{pagepanel} " if pareto else ""
            if run_key == "wave0/muont":
                tmain.append(
                    f"{prefix}{arm_cell} & {tuple_cell} & "
                    f"$0.0000$ & $0.0000$ & $0.0000$ & $0.0000 \\pm \\text{{---}}$ & "
                    f"{step_cell} & {gib_cell} \\\\"
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
                    "wave_sym_fastlag/inner_outer_mlpdiag_lag",
                ):
                    net_cell = rf"$\mathbf{{{net_val:+.4f}}}{net_suf}$"
                else:
                    net_cell = f"${net_val:+.4f}{net_suf}$"

            c4_cell = f"${rec['d_c4_late']:+.4f}$"
            se_str = f"{rec['d_train_se']:.5f}"[1:]
            train_cell = f"${rec['d_train']:+.4f} \\pm {se_str}$"

            tmain.append(
                f"{prefix}{arm_cell} & {tuple_cell} & {raw_cell} & {net_cell} & "
                f"{c4_cell} & {train_cell} & {step_cell} & {gib_cell} \\\\"
            )

    tmain += [r"\bottomrule", r"\end{tabular}"]
    write_file("results/table_main.tex", tmain)
    if args.table_only:
        return

    # ------------------------------------------------------------------
    # 2. Window Evolution Table: results/table_windows.tex
    # ------------------------------------------------------------------
    win_groups = [
        (
            r"II.\ Width-Thresholded Two-Sided (\texttt{WT}, Seed 0 \& Cross-Seed)",
            "famWT",
            [
                ("wave2/inner_outer_mlpdiag", r"\textbf{\taxid{WT-MLP}} (\texttt{s0})", r"Front-loaded $\to$ stable", True),
                ("wave4/inner_outer_mlpdiag_lag", r"\textbf{\taxid{WT-MLP-Lag}} (W4, \texttt{s0})", r"Front-loaded $\to$ stable", True),
                ("wave_sym_fastlag/inner_outer_mlpdiag_lag", r"\textbf{\taxid{WT-MLP-Lag}}$^{\ddagger}$ (\texttt{fast\_lag})", r"Front-loaded $\to$ stable", True),
                ("wave_sym_fastlag/inner_outer_sym_mlpdiag", r"\textbf{\taxid{WT-Sym}}$^{\ddagger}$ (no-lag)", r"Win-1 \texttt{in:down} cost $\to$ decay recovery", True),
                ("wave_sym_fastlag/inner_outer_sym_mlpdiag_lag", r"\textbf{\taxid{WT-Sym-Lag}}$^{\ddagger}$ (\texttt{fast\_lag})", r"Win-1 \texttt{in:down} cost $\to$ decay recovery", True),
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
    if not args.results_only:
        write_file("results/table_windows.tex", tw)

    # ------------------------------------------------------------------
    # 3. Matched Cost-Quality Plane (Panel a): figs/plane_plots.tex
    # ------------------------------------------------------------------
    y_wt_lag_exec = runs["wave_sym_fastlag/inner_outer_mlpdiag_lag"]["d_sel_late"]
    x_wt_lag_exec = mean_overhead["inner_outer_mlpdiag_lag"]
    y_sym_lag_exec = runs["wave_sym_fastlag/inner_outer_sym_mlpdiag_lag"]["d_sel_late"]
    x_sym_lag_exec = mean_overhead["inner_outer_sym_mlpdiag_lag"]

    fam_plot_specs = [
        (
            "WT",
            "mark=*, mark size=2.2pt, color=famWT",
            r"Width-thresholded (\texttt{WT})",
            [
                ("wave2/inner_outer_mlpdiag", r"\textbf{\taxid{WT-MLP}}", "anchor=north, yshift=-3pt"),
                ("wave4/inner_outer_mlpdiag_lag", r"\taxid{WT-MLP-Lag} (W4)", "anchor=east, xshift=-3pt"),
                (
                    "wave_sym_fastlag/inner_outer_sym_mlpdiag",
                    r"\taxid{WT-Sym}$^{\ddagger}$",
                    "anchor=south west, xshift=2pt, yshift=3pt",
                ),
            ],
        ),
        (
            "OD",
            "mark=diamond*, mark size=2.5pt, color=famOD",
            r"Pure-diagonal (\texttt{OD})",
            [
                ("wave2/inner_outer_leftdiag", r"\textbf{\taxid{OD-All}}", None),
                ("wave10/outer_leftdiag_upstream_fused", r"\taxid{OD-Outer}$^{\dagger}$", None),
            ],
        ),
        (
            "DT",
            "mark=pentagon*, mark size=2.3pt, color=famDT",
            r"Dense two-sided (\texttt{DT})",
            [
                ("wave1/inner_outer_all", r"\taxid{DT-All}", "anchor=north east, xshift=-1pt, yshift=-3pt"),
                ("wave3/inner_outer_all_lag", r"\taxid{DT-All-Lag}", "anchor=south west, xshift=3pt, yshift=2pt"),
            ],
        ),
        (
            "Null",
            "mark=square*, mark size=2.0pt, color=famNull",
            r"Matched nulls (\texttt{Null})",
            [
                ("wave3/null_mlpdiag", r"\taxid{Null-WT}", "anchor=south, yshift=2.5pt"),
                ("wave1/null_all", r"\taxid{Null-Dense}", "anchor=west, xshift=3pt"),
                ("wave10/null_leftdiag_upstream_fused", r"\taxid{Null-OD}$^{\dagger}$", None),
            ],
        ),
        (
            "Abl",
            "mark=triangle*, mark size=2.2pt, color=famAbl",
            r"Ablations (\texttt{Abl})",
            [
                ("wave0/outer_left_down", r"\taxid{Abl-DownOnly}", None),
                ("wave2/inner_outer_hidden", r"\taxid{Abl-No2048}", "anchor=south west, xshift=2pt, yshift=1.5pt"),
                ("wave2/outer_left_all", r"\taxid{Abl-OuterDense}", "anchor=west, xshift=3pt"),
                ("wave0/down_hybrid", r"\taxid{Abl-DownSwap}", "anchor=south west, xshift=3pt, yshift=2pt"),
                ("wave1/left_all", r"\taxid{Abl-OutOnly}", "anchor=east, xshift=-3pt"),
            ],
        ),
        (
            "Ref",
            "mark=oplus*, mark size=2.2pt, color=famRef",
            r"Baselines (\texttt{Ref})",
            [
                ("wave0/muon", r"\taxid{Muon}", "anchor=west, xshift=3pt"),
                ("wave0/muont", r"\taxid{MuonT}", "anchor=south east, xshift=-2pt, yshift=2pt"),
            ],
        ),
    ]

    p_lines = [
        "% Generated by scripts/render_results_viz.py. Costs: results/measured_8gpu_timings.json. Do not edit.",
        f"% Shaded net-of-null improvement zone (below Null-Dense = {d_null_dense:.6f})",
        f"\\fill[famWT!6] (axis cs:-25.5,-0.0086) rectangle (axis cs:47.5,{d_null_dense:.6f});",
        r"\draw[pagemuted!70, dashed, line width=0.55pt] (axis cs:-25.5,0) -- (axis cs:47.5,0);",
        r"\draw[pagemuted!45, dotted, line width=0.5pt] (axis cs:0,-0.0086) -- (axis cs:0,0.0108);",
        f"\\draw[famNull!85, densely dashed, line width=0.7pt] "
        f"(axis cs:-25.5,{d_null_dense:.6f}) -- (axis cs:47.5,{d_null_dense:.6f});",
        f"\\node[font=\\tiny, text=famNull!90!black, anchor=south east] "
        f"at (axis cs:46.8,{d_null_dense + 0.00005:.6f}) {{null floor ($-0.0013$)}};",
    ]

    for _fam_key, style, leg_text, items in fam_plot_specs:
        coords = " ".join(
            f"({mean_overhead[rk.split('/', 1)[1]]:.3f},{runs[rk]['d_sel_late']:.6f})"
            for rk, _, _ in items
        )
        p_lines.append(f"\\addplot[only marks, {style}] coordinates {{{coords}}};")
        p_lines.append(f"\\addlegendentry{{{leg_text}}}")

    # Stars identify full-budget fast_lag quality rows; costs share the benchmark.
    p_lines.append(
        f"\\addplot[only marks, mark=star, mark size=3.6pt, line width=0.95pt, color=famWT] "
        f"coordinates {{({x_wt_lag_exec:.3f},{y_wt_lag_exec:.6f}) ({x_sym_lag_exec:.3f},{y_sym_lag_exec:.6f})}};"
    )
    p_lines.append(r"\addlegendentry{\texttt{fast\_lag} quality (${}^{\ddagger}$)}")

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
            if placement is None:
                continue
            xv = mean_overhead[rk.split("/", 1)[1]]
            yv = runs[rk]["d_sel_late"]
            p_lines.append(
                f"\\node[font=\\tiny, text={col}, {placement}, inner sep=1pt] "
                f"at (axis cs:{xv:.3f},{yv:.6f}) {{{lbl}}};"
            )

    # Separate labels for the cluster of low-overhead methods.
    leader_specs = [
        ("wave_sym_fastlag/inner_outer_sym_mlpdiag_lag",
         r"\textbf{\taxid{WT-Sym-Lag}}$^{\ddagger}$", "famWT", -0.8, -0.00075, "east"),
        ("wave_sym_fastlag/inner_outer_mlpdiag_lag",
         r"\textbf{\taxid{WT-MLP-Lag}}$^{\ddagger}$", "famWT", -0.8, -0.00765, "east"),
        ("wave2/inner_outer_leftdiag", r"\textbf{\taxid{OD-All}}", "famOD", -0.8, -0.00375, "east"),
        ("wave10/outer_leftdiag_upstream_fused",
         r"\taxid{OD-Outer}$^{\dagger}$", "famOD", -0.8, -0.00275, "east"),
        ("wave0/outer_left_down", r"\taxid{Abl-DownOnly}", "famAbl!90!black", -0.8, -0.00205, "east"),
        ("wave10/null_leftdiag_upstream_fused",
         r"\taxid{Null-OD}$^{\dagger}$", "famNull!90!black", 8.0, -0.0018, "west"),
    ]
    for rk, label, color, label_x, label_y, anchor in leader_specs:
        xv = mean_overhead[rk.split("/", 1)[1]]
        yv = runs[rk]["d_sel_late"]
        p_lines.append(
            f"\\draw[{color}, line width=0.45pt] "
            f"(axis cs:{xv:.3f},{yv:.6f}) -- (axis cs:{label_x:.3f},{label_y:.6f});"
        )
        p_lines.append(
            f"\\node[font=\\tiny, text={color}, anchor={anchor}, fill=white, "
            f"fill opacity=0.92, text opacity=1, inner sep=1pt] "
            f"at (axis cs:{label_x:.3f},{label_y:.6f}) {{{label}}};"
        )

    x_bd = mean_overhead["inner_outer_all_diag"]
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
        (15, "wave2/inner_outer_mlpdiag", r"\textbf{\taxid{WT-MLP}}", d_null_wt, "famWT", True),
        (14, "wave4/inner_outer_mlpdiag_lag", r"\taxid{WT-MLP-Lag} (W4)", d_null_wt, "famWT", True),
        (13, "wave_sym_fastlag/inner_outer_mlpdiag_lag", r"\textbf{\taxid{WT-MLP-Lag}}$^{\ddagger}$", d_null_wt, "famWT", True),
        (12, "wave3/inner_outer_all_lag", r"\taxid{DT-All-Lag}", d_null_dense, "famDT", True),
        (11, "wave1/inner_outer_all", r"\taxid{DT-All}", d_null_dense, "famDT", True),
        (10, "wave2/inner_outer_leftdiag", r"\textbf{\taxid{OD-All}}", d_null_dense, "famOD", True),
        (9, "wave2/outer_left_all", r"\taxid{Abl-OuterDense}", d_null_dense, "famAbl", True),
        (8, "wave2/inner_outer_hidden", r"\taxid{Abl-No2048}", d_null_wt, "famAbl", True),
        (7, "wave10/outer_leftdiag_upstream_fused", r"\taxid{OD-Outer}$^{\dagger}$", d_null_od, "famOD", True),
        (6, "wave_sym_fastlag/inner_outer_sym_mlpdiag", r"\textbf{\taxid{WT-Sym}}$^{\ddagger}$", d_null_wt, "famWT", True),
        (5, "wave_sym_fastlag/inner_outer_sym_mlpdiag_lag", r"\textbf{\taxid{WT-Sym-Lag}}$^{\ddagger}$", d_null_wt, "famWT", True),
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
        r"\fill[famWT!5] (axis cs:-0.0078,3.52) rectangle (axis cs:0.0,15.65);",
        r"\fill[famAbl!4] (axis cs:0.0,-0.65) rectangle (axis cs:0.0118,3.52);",
        r"\draw[pagemuted!75, line width=0.6pt] (axis cs:0,-0.65) -- (axis cs:0,15.65);",
        r"\draw[pagemuted!65, densely dashed, line width=0.55pt] (axis cs:-0.0078,3.52) -- (axis cs:0.0118,3.52);",
        r"\node[font=\tiny\itshape, text=pagemuted!95!black, anchor=south east] at (axis cs:0.0115,3.72) {Two-sided additions ($\uparrow$)};",
        r"\node[font=\tiny\itshape, text=famAbl!85!black, anchor=north west, align=left] at (axis cs:-0.0075,3.34) {Input removed /\\ degraded ($\downarrow$)};",
    ]

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
        short_id = rk.replace("/", "_")
        attr_dat.append(
            f"{y_idx} {short_id} {raw_v:.6f} {null_v:.6f} {net_v:.6f} "
            f"{c4_v:.6f} {tr_v:.6f} {se2:.6f}"
        )

        if is_add:
            a_lines.append(
                f"\\filldraw[fill=famNull!42, draw=famNull!85, line width=0.45pt] "
                f"(axis cs:0,{y_idx - bh:.2f}) rectangle (axis cs:{null_v:.6f},{y_idx + bh:.2f});"
            )
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
        "\\addplot[only marks, forget plot, mark=diamond*, mark size=2.0pt, "
        "mark options={fill=white, draw=pageink, line width=0.65pt}] "
        f"coordinates {{{' '.join(c4_coords)}}};"
    )
    a_lines.append(
        "\\addplot[only marks, forget plot, mark=*, mark size=1.3pt, color=pageink] "
        f"coordinates {{{' '.join(tr_coords)}}};"
    )

    write_file("figs/attribution.dat", attr_dat)
    write_file("figs/attribution_plots.tex", a_lines)
    if args.results_only:
        return

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

    arch = [
        "% Generated by scripts/render_results_viz.py from systems_summary.json & mechanism_summary.json. Do not edit.",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tikzpicture}[x=1cm, y=1cm, >=Latex]",
        r"  % ======================================================================",
        r"  % PANEL (a): Spatial Width-Thresholding in Transformer Layer \ell",
        r"  % ======================================================================",
        r"  \filldraw[fill=white, draw=pagerule, rounded corners=4pt, line width=0.6pt] (0.0,5.80) rectangle (20.40,11.50);",
        r"  \fill[pagepanel, rounded corners=4pt] (0.04,10.88) rectangle (20.36,11.46);",
        r"  \node[anchor=west, font=\pagesfsb\footnotesize\color{pageink}] at (0.28,11.17)",
        r"    {(a) Spatial Width-Thresholding in Transformer Block $\ell$ (\taxid{WT-MLP} vs.\ \taxid{WT-Sym})};",
        r"",
        r"  % --- Left Sub-block: Multi-Head Attention (d = 512) ---",
        r"  \filldraw[fill=famWT!4, draw=famWT!30, rounded corners=3pt, line width=0.5pt] (0.24,6.56) rectangle (9.86,10.70);",
        r"  \node[anchor=north west, font=\pagesfsb\scriptsize\color{famWT}] at (0.42,10.62)",
        r"    {Attention Sub-Block ($d = 512$)};",
        r"",
        r"  % Attention Row 1: RMSNorm -> Q,K,V <- Dense 512",
        r"  \filldraw[fill=white, draw=famWT!65, rounded corners=2pt, line width=0.5pt] (0.42,8.92) rectangle (2.92,9.96);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (1.67,9.44)",
        r"    {\texttt{RMSNorm}$(h_\ell)$\\ $\fFull$ \texttt{in:attn} ($512$)};",
        r"  \draw[->, famWT!80, line width=0.6pt] (2.92,9.44) -- (3.30,9.44);",
        r"  \filldraw[fill=white, draw=famWT!80, rounded corners=2pt, line width=0.6pt] (3.30,8.92) rectangle (6.06,9.96);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (4.68,9.44)",
        r"    {$W_q,\,W_k,\,W_v$\\ {\color{pagemuted}$3 \times (512\times 512)$}};",
        r"  \draw[->, famWT, densely dashed, line width=0.65pt] (6.46,9.44) -- (6.06,9.44);",
        r"  \filldraw[fill=famWT!12, draw=famWT!85, rounded corners=2pt, line width=0.6pt] (6.46,8.92) rectangle (9.68,9.96);",
        r"  \node[align=center, font=\tiny\color{famWT!95!black}] at (8.07,9.44)",
        r"    {$\fFull$ \textbf{Dense $512\times 512$}\\ $\delta_q,\delta_k,\delta_v$ ($R^2\!=\!31\text{--}72\%$)};",
        r"",
        r"  % Attention mixing -> Row 2: W_o",
        r"  \draw[->, pageink!70, line width=0.6pt] (4.68,8.92) -- node[right, font=\tiny\color{pagemuted}] {RoPE + SDPA} (4.68,7.92);",
        r"  \filldraw[fill=white, draw=famWT!65, rounded corners=2pt, line width=0.5pt] (0.42,6.78) rectangle (2.92,7.88);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (1.67,7.33)",
        r"    {attn context\\ $\fFull$ \texttt{in:o} ($512$)};",
        r"  \draw[->, famWT!80, line width=0.6pt] (2.92,7.33) -- (3.30,7.33);",
        r"  \filldraw[fill=white, draw=famWT!80, rounded corners=2pt, line width=0.6pt] (3.30,6.78) rectangle (6.06,7.88);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (4.68,7.33)",
        r"    {$W_o \to h'_\ell$\\ {\color{pagemuted}$512\times 512$ (residual)}};",
        r"  \draw[->, famWT, densely dashed, line width=0.65pt] (6.46,7.33) -- (6.06,7.33);",
        r"  \filldraw[fill=famWT!14, draw=famWT!90, rounded corners=2pt, line width=0.65pt] (6.46,6.78) rectangle (9.68,7.88);",
        r"  \node[align=center, font=\tiny\color{famWT!95!black}] at (8.07,7.33)",
        r"    {$\fFull$ \textbf{Dense $512\times 512$}\\ $\delta_o$ ($>99\%$ rotational)};",
        r"",
        r"  % --- Right Sub-block: SwiGLU MLP (d = 512 <-> 2048) ---",
        r"  \filldraw[fill=famOD!4, draw=famOD!32, rounded corners=3pt, line width=0.5pt] (10.16,6.56) rectangle (20.16,10.70);",
        r"  \node[anchor=north west, font=\pagesfsb\scriptsize\color{famOD}] at (10.34,10.62)",
        r"    {SwiGLU Feed-Forward Sub-Block ($512 \leftrightarrow 2048$)};",
        r"",
        r"  % MLP Row 1: RMSNorm -> W_gate, W_up <- 2D-Tiled Diag 2048",
        r"  \filldraw[fill=white, draw=famWT!65, rounded corners=2pt, line width=0.5pt] (10.34,8.92) rectangle (13.16,9.96);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (11.75,9.44)",
        r"    {\texttt{RMSNorm}$(h'_\ell)$\\ $\fFull$ \texttt{in:mlp} ($512$)};",
        r"  \draw[->, famWT!80, line width=0.6pt] (13.16,9.44) -- (13.52,9.44);",
        r"  \filldraw[fill=white, draw=famOD!90, rounded corners=2pt, line width=0.75pt] (13.52,8.92) rectangle (16.32,9.96);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (14.92,9.44)",
        r"    {$W_{\text{gate}},\,W_{\text{up}}$ ($s\!=\!2$)\\ {\color{famOD!90!black}$2 \times (\mathbf{2048}\times 512)$}};",
        r"  \draw[->, famOD, densely dashed, line width=0.75pt] (16.72,9.44) -- (16.32,9.44);",
        r"  \filldraw[fill=famOD!15, draw=famOD, rounded corners=2.5pt, line width=0.85pt] (16.72,8.80) rectangle (19.98,10.06);",
        r"  \node[align=center, font=\tiny\color{famOD!95!black}] at (18.35,9.43)",
        r"    {$\fDiag$ \textbf{2D-Tiled Diag $2048$}\\ $\delta_{\text{gate}},\delta_{\text{up}}$ ($\rho_{\mathrm{rms}}\!\approx\!N_t^{-1/2}$)\\ "
        f"$82\\%/78\\%$ diag; $-{saved_gib:.2f}$\\,GiB}};",
        r"",
        r"  % Hadamard SwiGLU gate -> Row 2: W_down",
        r"  \draw[->, famOD!85, line width=0.6pt] (14.92,8.92) -- (14.92,8.60);",
        r"  \filldraw[fill=famOD!13, draw=famOD!75, rounded corners=6pt, line width=0.55pt] (13.38,8.16) rectangle (16.46,8.58);",
        r"  \node[font=\tiny\color{famOD!95!black}] at (14.92,8.37) {$\operatorname{SiLU}(g)\odot u$ (Hadamard)};",
        r"  \draw[->, pageink!70, line width=0.6pt] (14.92,8.16) -- (14.92,7.88);",
        r"",
        r"  \filldraw[fill=white, draw=famWT!65, rounded corners=2pt, line width=0.5pt] (10.34,6.78) rectangle (13.16,7.88);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (11.75,7.33)",
        r"    {\texttt{in:down} ($2048$, $\mu\mu^{\Tr}\!>\!0$)\\ $\fFull$ \texttt{WT-MLP} / $\fDiag$ \texttt{WT-Sym}};",
        r"  \draw[->, famWT!80, line width=0.6pt] (13.16,7.33) -- (13.52,7.33);",
        r"  \filldraw[fill=white, draw=famWT!80, rounded corners=2pt, line width=0.6pt] (13.52,6.78) rectangle (16.32,7.88);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (14.92,7.33)",
        r"    {$W_{\text{down}} \to h_{\ell+1}$\\ {\color{pagemuted}$512\times 2048$ (residual)}};",
        r"  \draw[->, famWT, densely dashed, line width=0.65pt] (16.72,7.33) -- (16.32,7.33);",
        r"  \filldraw[fill=famWT!14, draw=famWT!90, rounded corners=2pt, line width=0.65pt] (16.72,6.78) rectangle (19.98,7.88);",
        r"  \node[align=center, font=\tiny\color{famWT!95!black}] at (18.35,7.33)",
        r"    {$\fFull$ \textbf{Dense $512\times 512$}\\ $\delta_{\text{down}}$ ($>99\%$ rotational)};",
        r"",
        r"  % Panel (a) Full-Width Legend Strip",
        r"  \fill[pagepanel, rounded corners=2pt] (0.24,5.94) rectangle (20.16,6.40);",
        r"  \node[font=\tiny\color{pageink}] at (10.20,6.17)",
        r"    {$\fFull$ \textbf{Dense spectral roots:} width-$512$ inputs \& cotangents (+ \texttt{in:down} in \taxid{WT-MLP}) \quad$\mid$\quad $\fDiag$ \textbf{Age-$0$ inline diagonal:} width-$2048$ \texttt{out:gate,up} (+ \texttt{in:down} in \taxid{WT-Sym})};",
        r"",
        r"  % ======================================================================",
        r"  % PANEL (b): Asynchronous 1-Step Lagged Pipeline (fast_lag, \tau = 1)",
        r"  % ======================================================================",
        r"  \filldraw[fill=white, draw=pagerule, rounded corners=4pt, line width=0.6pt] (0.0,0.0) rectangle (20.40,5.52);",
        r"  \fill[pagepanel, rounded corners=4pt] (0.04,4.90) rectangle (20.36,5.48);",
        r"  \node[anchor=west, font=\pagesfsb\footnotesize\color{pageink}] at (0.28,5.19)",
        r"    {(b) Asynchronous 1-Step Lagged Pipeline (\taxid{WT-MLP-Lag} \& \taxid{WT-Sym-Lag}, \texttt{fast\_lag}, $\tau = 1$)};",
        r"",
        r"  % Step timeline banner",
        r"  \draw[->, pagemuted, line width=0.55pt] (3.00,4.54) -- (20.14,4.54);",
        r"  \draw[pagemuted, line width=0.65pt] (10.12,4.42) -- (10.12,4.66);",
        r"  \node[font=\pagesfsb\tiny\color{pageink}, anchor=south] at (6.56,4.56) {Step $t$};",
        r"  \node[font=\pagesfsb\tiny\color{famWT}, anchor=south] at (15.13,4.56) {Step $t+1$ ($\approx 900$\,ms Fwd+Bwd overlap window)};",
        r"",
        r"  % --- Lane 1: Main CUDA Stream ---",
        r"  \node[align=left, anchor=west, font=\pagesfsb\scriptsize\color{pageink}] at (0.26,3.88)",
        r"    {Main Stream\\ {\tiny\color{pagemuted}(Compute-Bound)}};",
        f"  \\filldraw[fill=pagepanel, draw=pagemuted!80, rounded corners=2pt, line width=0.55pt] (3.00,3.40) rectangle (6.74,4.36);",
        f"  \\node[align=center, font=\\tiny\\color{{pageink}}] at (4.87,3.88)",
        f"    {{\\textbf{{Fwd+Bwd MB 0..3 $(t)$}}\\\\ 4 block hooks $(+{cap_ms:.1f}\\,\\text{{ms}})$}};",
        r"",
        f"  \\filldraw[fill=famWT!15, draw=famWT, rounded corners=2pt, line width=0.8pt] (6.90,3.40) rectangle (10.00,4.36);",
        f"  \\node[align=center, font=\\tiny\\color{{famWT!95!black}}] at (8.45,3.88)",
        f"    {{\\textbf{{Ozaki Update$(t)$}}\\\\ $L^{{(t-1)}},\\hat L^{{(t-1)}}$ $(+{upd_ms:.1f}\\,\\text{{ms}})$}};",
        r"",
        r"  \filldraw[fill=pagepanel, draw=pagemuted!80, rounded corners=2pt, line width=0.55pt] (10.24,3.40) rectangle (13.44,4.36);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (11.84,3.88)",
        r"    {\textbf{MB 0 $(t+1)$}\\ $\approx 235$\,ms (waits \texttt{ev\_rows})};",
        r"",
        r"  \filldraw[fill=pagepanel, draw=pagemuted!80, rounded corners=2pt, line width=0.55pt] (13.60,3.40) rectangle (17.88,4.36);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (15.74,3.88)",
        r"    {\textbf{MB 1..3 $(t+1)$}\\ $\approx 665$\,ms Fwd+Bwd (polls readiness)};",
        r"",
        r"  \filldraw[fill=famWT!15, draw=famWT, rounded corners=2pt, line width=0.8pt] (18.04,3.40) rectangle (20.14,4.36);",
        r"  \node[align=center, font=\tiny\color{famWT!95!black}] at (19.09,3.88)",
        r"    {\textbf{Update$(t\!+\!1)$}\\ uses $L^{(t)},\hat L^{(t)}$};",
        r"",
        r"  % --- Lane 2: Priority-0 Output Side Stream ---",
        r"  \node[align=left, anchor=west, font=\pagesfsb\scriptsize\color{famOD}] at (0.26,2.26)",
        r"    {Output Stream\\ {\tiny\color{pagemuted}(Priority-0 + Host)}};",
        r"  \filldraw[fill=famOD!15, draw=famOD, rounded corners=2pt, line width=0.65pt] (6.74,1.78) rectangle (8.86,2.74);",
        r"  \node[align=center, font=\tiny\color{famOD!95!black}] at (7.80,2.26)",
        r"    {\textbf{\texttt{syrk}}\\ packed ($1.4$\,ms)};",
        r"",
        r"  \filldraw[fill=famDT!16, draw=famDT!95!black, rounded corners=2pt, line width=0.65pt] (9.02,1.78) rectangle (13.48,2.74);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (11.25,2.26)",
        r"    {\textbf{\texttt{reduce\_scatter} + Graph Roots}\\ {\color{pagemuted}(overlaps Update$(t)$ + MB 0)}};",
        r"",
        r"  \filldraw[fill=famNull!15, draw=famNull!90!black, rounded corners=2pt, line width=0.65pt] (13.64,1.78) rectangle (18.08,2.74);",
        r"  \node[align=center, font=\tiny\color{pageink}] at (15.86,2.26)",
        r"    {\textbf{Mid-Step Repair / \texttt{eigh}}\\ {\color{pagemuted}+\,\texttt{unitRMS} + FP16 Split ($\sim 700$\,ms early)}};",
        r"",
        r"  % Dedicated Cross-Stream Event Band",
        r"  \draw[->, famOD!90!black, densely dashed, line width=0.7pt] (6.52,3.40) -- (7.02,2.74);",
        r"  \draw[->, famOD!90!black, line width=0.75pt] (8.68,2.74) -- node[pos=0.50, fill=white, inner sep=1pt, font=\tiny\color{famOD!95!black}] {\texttt{ev\_rows} ($0$\,ms stall)} (10.52,3.40);",
        r"  \draw[->, famNull!90!black, densely dotted, line width=0.7pt] (14.04,3.40) -- (14.34,2.74);",
        r"  \draw[->, famWT, line width=0.85pt] (17.84,2.74) -- node[pos=0.50, right, font=\tiny\color{famWT}] {$L^{(t)},\hat L^{(t)}$} (18.36,3.40);",
        r"",
        r"  % Bottom Split: Eliminated Bottlenecks (left) + Executed 8-A100 Summary (right)",
        r"  \filldraw[fill=famAbl!5, draw=famAbl!45, densely dashed, rounded corners=2.5pt, line width=0.5pt] (0.24,0.16) rectangle (13.34,1.54);",
        r"  \node[align=left, anchor=west, font=\tiny\color{pageink}] at (0.42,0.85)",
        r"    {{\color{famAbl!95!black}\pagesfsb Eliminated legacy bottlenecks:} "
        r"(1) Stream-local timer drops per-step \texttt{cuda.synchronize()}.\\"
        r"     (2) Mid-step \texttt{select\_microbatch(1..3)} hides \texttt{eigh} fallbacks ($+89\text{--}230\,\text{ms} \to +6.6\text{--}21\,\text{ms}$).\\"
        r"     (3) Per-site block hooks + Ozaki Tensor-Core GEMMs avoid DDP bucket conflicts \& SIMT FP32.};",
        r"",
        r"  \filldraw[fill=pagepanel, draw=famWT!45, rounded corners=2.5pt, line width=0.55pt] (13.54,0.16) rectangle (20.16,1.54);",
        r"  \node[align=left, anchor=west, font=\tiny\color{pageink}] at (13.72,0.85)",
        f"    {{{{\\color{{famWT}}\\pagesfsb Executed 8$\\times$A100 \\texttt{{fast\\_lag}}:}}\\\\"
        f"     \\textbf{{\\taxid{{WT-Sym-Lag}}}}: $\\mathbf{{+{exec_sym_lag_muont:.1f}\\%}}$ vs.\\ MuonT (${exec_sym_lag_gib:.2f}$\\,GiB)\\\\"
        f"     \\textbf{{\\taxid{{WT-MLP-Lag}}}}: $\\mathbf{{+{exec_wt_lag_muont:.1f}\\%}}$ vs.\\ MuonT (${exec_wt_lag_gib:.2f}$\\,GiB)\\\\"
        f"     {{\\color{{pagemuted}}Beats MuonT \\& Muon to $\\text{{C4}}\\le 3.52$ ($5{{,}}459$\\,s)}}}};",
        r"\end{tikzpicture}%",
        r"}",
    ]
    write_file("figs/method_arch.tex", arch)


if __name__ == "__main__":
    main()
