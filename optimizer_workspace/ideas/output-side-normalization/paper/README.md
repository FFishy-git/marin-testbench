# Width-Thresholded Output-Cotangent Normalization — Paper Source

Scoped empirical, mechanistic, and systems manuscript on output-cotangent moment
normalization in Muon-type optimizers (**`WT-MLP`** / `inner_outer_mlpdiag`,
**`WT-MLP-Lag`** / `inner_outer_mlpdiag_lag`, **`OD-All`** / `inner_outer_leftdiag`,
and the `fast_lag` low-overhead lagged architecture), built from the executed runs in
`marin-testbench/optimizer_workspace/ideas/output-side-normalization/`.

See [REPORT.md](REPORT.md) for the standalone executive summary of the
systematic 5-slot taxonomy, motivating 6+1 covariance/drift audit on `wave1/null_all`,
post-training norm-grafting check, Wave-12 horizon-shift correction, and
`fast_lag` A100 latency/memory benchmarks.

## Build

```bash
latexmk -pdf main.tex        # pdfLaTeX, as Overleaf and arXiv compile
```

## Systematic 5-Slot Taxonomy (`sections/method.tex`)

Every arm is indexed by $\mathcal{A}(R_{512}, R_{2048} \mid L_{512}, L_{2048} \mid \tau)$
across six colorblind-safe families defined in `main.tex`:

| Family | Tag | Arms & Implementation Keys |
| :--- | :--- | :--- |
| **I. Baselines** | `Ref` | `Muon` (`muon`), `MuonT` (`muont`) |
| **II. Width-Thresholded Two-Sided** *(Proposed)* | `WT` | `WT-MLP` (`inner_outer_mlpdiag`), `WT-MLP-Lag` (`inner_outer_mlpdiag_lag`), modular `WT-Sym` |
| **III. Pure-Diagonal Output** *(Zero-`eigh` Limit)* | `OD` | `OD-All` (`inner_outer_leftdiag`), `OD-Outer` (`outer_leftdiag` / `outer_leftdiag_fused`) |
| **IV. Dense Two-Sided Ceiling** | `DT` | `DT-All` (`inner_outer_all`), `DT-All-Lag` (`inner_outer_all_lag`) |
| **V. Matched Execution-Path Nulls** | `Null` | `Null-WT` (`null_mlpdiag`), `Null-Dense` (`null_all`), `Null-OD` (`null_leftdiag_upstream_fused`) |
| **VI. Structural & Input-Degradation Ablations** | `Abl` | `Abl-OuterDense` (`outer_left_all`), `Abl-No2048` (`inner_outer_hidden`), `Abl-DownOnly` (`outer_left_down`), `Abl-DownSwap` (`down_hybrid`), `Abl-OutOnly` (`left_all`), `Abl-BothDiag` (`inner_outer_all_diag`) |

## Regenerating every number and figure

The manuscript reads only generated files for anything derived from a run,
checkpoint audit, or hardware benchmark, so prose cannot drift from the data.
Run from this directory, in order:

```bash
python -I scripts/analyze.py             # full-budget runs -> results/summary.json
python -I scripts/analyze_lr.py          # 28 learning-rate screening trials -> results/lr_summary.json
python -I scripts/check_duplicates.py    # distinct-trajectory audit
python -I scripts/derive.py              # cross-study & Wave-12 horizon comparisons
python -I scripts/analyze_mechanism.py   # full-trajectory motivating 6+1 audit & step-norm check
python -I scripts/analyze_systems.py     # fast_lag A100 stage latency & parity tables
python -I scripts/render_results_viz.py  # taxonomic tables, TikZ method arch, 2-panel Pareto/attribution & 3-panel LR horizon
```

(`render_results_viz.py` reads ONLY the materialized JSON files inside `results/`
and never touches `$PVC` or GPUs; `analyze_mechanism.py --recompute` recomputes
`results/mechanism_summary.json` from saved `$PVC` checkpoints on CPU.)

Static and numerical verification:

```bash
python -I scripts/check_tex.py      # \input targets, refs, citations, brace/env balance
python -I scripts/check_numbers.py  # every prose numeral against the computed JSON values
```

## Outputs

| Path | Contents |
| --- | --- |
| `results/run_numbers.tex` | per-run macros (deltas, step times, memory, fallbacks) |
| `results/lr_numbers.tex` | screening-trial macros |
| `results/derived_numbers.tex` | cross-study & Wave-12 horizon macros |
| `results/evidence_numbers.tex` | run-directory and distinct-trajectory counts |
| `results/motivation_numbers.tex`, `results/table_motivation.tex` | Section 3 motivating 6+1 covariance & subspace-drift audit (`wave1/null_all`) |
| `results/mechanism_numbers.tex` | Section 6.6 post-training norm-grafting & active-run invariance macros |
| `results/systems_numbers.tex`, `results/table_systems.tex` | Section 7 `fast_lag` A100 latency & memory breakdown |
| `results/table_main.tex` | Taxonomically grouped master seed-0 table (5-slot badges, Raw vs. Net-of-Null $\Delta$Sel, Legacy $\to$ `fast_lag` Step %) |
| `results/table_windows.tex` | Taxonomically grouped five-window paired deltas & temporal regimes |
| `results/table_duplicates.tex` | bitwise-identical run groups |
| `figs/method_arch.tex` | Section 4 pure-TikZ 2-panel spatial width-thresholding & `fast_lag` async pipeline diagram |
| `figs/plane_plots.tex`, `figs/attribution.dat`, `figs/attribution_plots.tex` | Section 6 2-panel Cost–Quality Pareto plane (`fast_lag` frontier shift) & multi-metric causal attribution forest plot |
| `figs/lr_sweep.dat`, `figs/lr_horizon.dat`, `figs/transfer_bars.tex` | Section 8 3-panel 320-step screening vs. 5,120-step horizon-shift visualization |
| `figs/motivation_bars.dat`, `figs/motivation_traj.dat` | Section 3 11-moment & 8-checkpoint trajectory pgfplots series |
| `figs/mechanism_bars.dat`, `figs/systems_bars.dat` | Section 6.6 step-rotation and Section 7 latency pgfplots series |

## Status of each section

| Section | Status |
| --- | --- |
| Abstract, 1 Introduction | Updated around motivating 6+1 baseline audit, width-thresholded scope, and `fast_lag` |
| 2 Background | Complete |
| 3 Motivating Analysis (`sections/motivation.tex`) | **New (before Method)** — full-trajectory 6+1 covariance & eigenspace-drift audit on baseline `MuonT` (`wave1/null_all`, 8+2 checkpoints × 32 layers) deriving the 3 design choices |
| 4 Method, 5 Protocol | Complete; smoothly linked to Section 3's three design consequences |
| 6 Results (`sections/results.tex`) | Complete; includes Subsection 6.6 post-training norm-grafting (`0.9854` norm ratio, `22.4°` rotation) & active-run invariance check |
| 7 Fast-Lag Systems Architecture | **New** — `3.6%–4.1%` (`mlpdiag_lag`) and `4.8%–5.4%` (`all_lag`) over Muon |
| 8 Tuning Confound | Corrected — exposes `+0.100`-nat Wave-12 baseline regression at `LR=0.064` |
| 9 Final Confirmation Program | Focused around `LR=0.0226` (`{0.016, 0.0226, 0.032, 0.045}`) on `fast_lag` |
| 10 Limitations, 11 Related Work, 12 Conclusion | Complete |
| Appendices A, B, C | Complete |
