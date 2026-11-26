# Paper v4: Width-Thresholded Output Preconditioning for Muon Optimizers

`main.pdf` is the compiled optimizer-first manuscript, with all 130M and 520M
scaling experiments and checkpoint diagnostics evaluated at their verified
full-budget learning-rate optima.

## Narrative Organization

1. **Section 1 (`sections/01_intro.tex`) — Introduction**: Motivates two-sided
   preconditioning on top of `Muon` and `MuonT`, immediately introduces
   `WT-MLP-Lag` (dense $(-1/2,-1/8)$ output roots at hidden width $\dm$, inline
   exact diagonal scaling at MLP expansion width $\dff=4\dm$, and one-step
   asynchronous side-stream overlap $\tau=1$), summarizes why this matches
   Transformer geometry (`fig1_overview.pdf`), and states main scaling gains and
   runtime scope.
2. **Section 2 (`sections/02_method.tex`) — The `WT-MLP-Lag` optimizer**:
   Presents the complete mathematical update rule, moment definitions,
   width-thresholded factor assignment (`tab:taxonomy`), 96.2% cubic-work
   reduction, and the one-step-lagged GPU execution pipeline
   (`fig3_method_pipeline.pdf`).
3. **Section 3 (`sections/03_diagnosis.tex`) — Why width thresholding and
   overlap fit Transformer geometry**: Tests `WT-MLP-Lag`'s two design
   assumptions across 352 input and output moments on `Muon`, `MuonT`, and
   `WT-MLP-Lag` trajectories (`fig2_structural_audit.pdf`,
   `tab:diagnosis_summary`).
4. **Section 4 (`sections/04_experiments.tex`) — Pretraining experiments and
   scaling**: Concise experimental setup (`tab:regimes`) followed directly by
   main pretraining results across fresh 130M seeds, 4x token horizon (80 TPP),
   and 520M width transfer (`tab:main_scaling`, `fig5_scaling_trajectories.pdf`,
   `fig6_paloma_domains.pdf`).
5. **Section 5 (`sections/05_ablations.tex`) — Controls, ablations, and
   two-sided baselines**: `Null-WT` execution-path control and full-budget vs.
   320-step learning-rate bracket (`fig4_confounds_lr_null.pdf`, `tab:controls`),
   factor-placement/exponent/lag ablations (`tab:ablations`,
   `fig7_ablations_and_related.pdf`), and long-horizon `GO-MUON` / `Mousse`
   comparisons (`tab:related`).
6. **Section 6 (`sections/06_mechanism_cost.tex`) — Optimization mechanism and
   systems cost**: Counterfactual update rotation and Haar delocalization
   (`tab:mechanism_rotation`, `fig8_mechanism_and_pareto.pdf`), trajectory-level
   output-subspace drift reduction relative to `Muon` (`tab:output_drift`,
   `fig9_output_drift.pdf`), and median/mean step-time waterfalls across 8 and
   32 A100 GPUs (`tab:timing_summary`).
7. **Sections 7--8 (`sections/07_related_conclusion.tex`) — Related work and
   Conclusion & scope**.
8. **Appendices A--E (`sections/99_appendix.tex`)**: Mixed-precision
   Newton--Schulz, FP16-split arithmetic, and one-sided ($W\hat R^2$) vs.
   two-sided ($\hat L^2 W\hat R^2$) metric-shaped weight-decay analysis, full
   diagnostic derivations and 352-matrix audit (`tab:diagnosis`), inner/outer
   rotation details (`tab:rotation_detail`), complete fixed-data and
   uninterrupted-EMA subspace-drift protocol and per-role trajectories
   (`fig10_drift_by_role.pdf`), short-vs.-full LR sweeps and per-seed losses
   (`tab:lr_cliff`, `tab:absolute_results`, `tab:seeds`), 19-arm ablation ledger
   (`tab:ablations_full`), and complete 130M/520M timing and Paloma tables
   (`tab:bench19`, `tab:bench520m`, `tab:paloma_full`).

## Rebuild

From this directory:

```bash
uv run --offline --no-project --with numpy python scripts/generate_tables.py
uv run --offline --no-project --with matplotlib python scripts/generate_figures.py --only 1 5 6 8
uv run --offline --no-project --with matplotlib python scripts/plot_drift_trajectories.py --summary tables/output_drift_metrics.json.gz
PATH="/opt/texlive/2026/bin/x86_64-linux:$PATH" latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The state immediately prior to unifying 520M comparisons and checkpoint probes
at `lr=0.016` (`lr-520m-20261125`) is preserved at
`/root/tmp_paper_tools/paper_v4_before_520m_lr0p016_20261126/`.
