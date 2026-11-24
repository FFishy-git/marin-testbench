# Width-Thresholded Output-Cotangent Normalization — Paper Source

Scoped empirical, mechanistic, and systems manuscript on output-cotangent moment
normalization in Muon-type optimizers (**`WT-MLP`** / `inner_outer_mlpdiag`,
**`WT-MLP-Lag`** / `inner_outer_mlpdiag_lag`, **`WT-Sym`** / `inner_outer_sym_mlpdiag`,
**`WT-Sym-Lag`** / `inner_outer_sym_mlpdiag_lag`, **`OD-All`** / `inner_outer_leftdiag`,
and the executed 8xA100 `fast_lag` asynchronous architecture), formatted in clean
full-page `article` style (`\documentclass[11pt]{article}` + `\usepackage{fullpage}`).

See [REPORT.md](REPORT.md) for the standalone executive summary of all `19`
configurations across `37` distinct full-budget trajectories (`45` completed run
directories), motivating 6+1 covariance/drift audit, $2\times 2$ asymmetric-versus-symmetric
width-2048 validation, post-training norm-grafting check, Wave-12 horizon-shift
correction, and executed 8xA100 `fast_lag` wall-clock crossings.

## Build

```bash
latexmk -pdf main.tex        # pdfLaTeX, full-page standard article style
```

## Systematic 5-Slot Taxonomy (`sections/method.tex`)

Every arm is indexed by $\mathcal{A}(R_{512}, R_{2048} \mid L_{512}, L_{2048} \mid \tau)$
across six colorblind-safe families (`19` configurations):

| Family | Tag | Arms & Implementation Keys |
| :--- | :--- | :--- |
| **I. Baselines** | `Ref` | `Muon` (`muon`), `MuonT` (`muont`) |
| **II. Width-Thresholded Two-Sided** *($2\times 2$ Core)* | `WT` | `WT-MLP` (`inner_outer_mlpdiag`), `WT-MLP-Lag` (`inner_outer_mlpdiag_lag`), `WT-Sym` (`inner_outer_sym_mlpdiag`), `WT-Sym-Lag` (`inner_outer_sym_mlpdiag_lag`) |
| **III. Pure-Diagonal Output** *(Zero-`eigh` Limit)* | `OD` | `OD-All` (`inner_outer_leftdiag`), `OD-Outer` (`outer_leftdiag` / `outer_leftdiag_fused`) |
| **IV. Dense Two-Sided Ceiling** | `DT` | `DT-All` (`inner_outer_all`), `DT-All-Lag` (`inner_outer_all_lag`) |
| **V. Matched Execution-Path Nulls** | `Null` | `Null-WT` (`null_mlpdiag`), `Null-Dense` (`null_all`), `Null-OD` (`null_leftdiag_upstream_fused`) |
| **VI. Structural & Input-Degradation Ablations** | `Abl` | `Abl-OuterDense` (`outer_left_all`), `Abl-No2048` (`inner_outer_hidden`), `Abl-DownOnly` (`outer_left_down`), `Abl-DownSwap` (`down_hybrid`), `Abl-OutOnly` (`left_all`), `Abl-BothDiag` (`inner_outer_all_diag`) |

## Regenerating every number and figure (local JSON pipeline)

The manuscript reads only generated files in `results/` and `figs/`, so prose
cannot drift from the computed data:

```bash
python -I scripts/integrate_sym_wave.py  # merge results/sym_wave_raw.json -> summary.json, run_numbers.tex, evidence_numbers.tex, systems_summary.json
python -I scripts/derive.py              # cross-study, Wave-12 horizon & wave_sym_fastlag macros -> results/derived_numbers.tex
python -I scripts/analyze_systems.py     # stage & executed 8xA100 fast_lag latency & C4 crossings -> results/table_systems.tex, results/table_crossings.tex
python -I scripts/render_results_viz.py  # taxonomic tables, TikZ method arch, Pareto/attribution & 3-panel LR horizon
```

Table 5's Mean Step % and Peak GiB and Figure 3(a)'s x-coordinates come only
from `results/measured_8gpu_timings.json`. Every method uses the same optimized
eight-A100 campaign: arithmetic
mean over updates 37-96 and peak Rank-0 CUDA allocation over all 96 updates,
including warmup. Quality columns retain their original full-budget waves.
Regenerate this table alone with:

```bash
python -I scripts/render_results_viz.py --table-only
```

Regenerate Table 5 and both Figure 3 panels without changing other figures:

```bash
python -I scripts/render_results_viz.py --results-only
```

Figure 3(a) contains no stage-calibrated estimates or historical speedup arrows.
Stars identify full-budget `fast_lag` quality rows; their timings use the same
benchmark as the other methods. Figure 3(b)'s quality metrics are unchanged.

### Exact WT Lagged Training Curves

Section 6.4 plots all 5,120 per-update training losses for same-wave
`WT-MLP-Lag` and `WT-Sym-Lag`, an early-training zoom, and their exact paired
difference. There is no smoothing, binning, or downsampling. The discussion
separates diagonal-root fidelity from optimizer-direction fidelity and states
that the comparison changes inner and outer input factors, shaped decay, and
execution together.

Export from the original completed JSONL histories and render the figure:

```bash
python -I scripts/plot_wt_lag_training.py
```

Rebuild offline from the checked, full-precision exported data:

```bash
python -I scripts/plot_wt_lag_training.py --from-data
```

The script requires Matplotlib. `results/wt_lag_training.json` records the run
commands, matching configuration, source and metric hashes, row counts, and
paired window means. `figs/wt_lag_training.dat` preserves every logged loss
with round-trip float precision.

### Paired Related-Baseline Comparison (Mousse & GO-MUON)

Section 6.8 documents the 18-trial seed-0 short LR screen and 12-trial seed-1/2
full-budget paired comparison (`muon_control`, `WT-MLP`, `mousse_control`,
`mousse_paper`, `go_muon_control`, `go_muon_paper`) with a dedicated comparison
table (`results/table_related.tex`) and a three-panel exact training-behavior
and LR-censoring figure (`figs/related_baselines_training.pdf`).

```bash
python -I scripts/plot_related_baselines.py              # live export + table + macros + figure
python -I scripts/plot_related_baselines.py --from-data  # offline rebuild from checked export
python -m pytest scripts/test_related_baselines.py       # exact export & tamper-detection test
```

### Confirmation & Scaling Campaign (`confirmation-20261121`, 30 Runs + 22 Paloma Evals)

Section 6.9 documents the 30-run `fast_lag` confirmation and scaling campaign
(Phase 1 full-budget LR bracket + `fast_lag` `Null-WT`, Phase 2 fresh 3-seed
paired confirmation, Phase 3A/3B `130M 4x` `80 TPP` overtraining, Phase 4
`520M 1x` scaling, and Phase 5 16-domain Paloma evaluation) via
`results/confirmation_campaign.json`, `results/confirmation_numbers.tex`,
`results/table_confirmation.tex`, and TikZ/pgfplots data files
`figs/conf_{lr_grid,traj_delta,paloma}.dat`:

```bash
python -I scripts/integrate_confirmation_campaign.py              # live export + macros + table + .dat
python -I scripts/integrate_confirmation_campaign.py --from-data  # offline rebuild from checked JSON
```

Static and numerical verification:

```bash
python -m pytest scripts/test_wt_lag_training.py  # exact export and offline rebuild
python -I scripts/check_tex.py      # \input targets, refs, citations, macros, brace/env balance (exits 0)
python -I scripts/check_numbers.py  # every prose numeral against computed JSON values (exits 0)
```

## Outputs

| Path | Contents |
| --- | --- |
| `results/sym_wave_raw.json` | Committed raw metrics, 5-window series, head-to-head pairs & C4 crossings for `wave_sym_fastlag` |
| `results/wt_lag_training.json`, `figs/wt_lag_training.dat`, `figs/wt_lag_training.pdf` | Provenance, exact 5,120-update paired training series, and Section 6.4 comparison figure |
| `results/measured_8gpu_timings.json` | Matched eight-A100 Table 5/Figure 3 costs and separately labeled full-budget timing audit |
| `results/run_numbers.tex`, `results/evidence_numbers.tex` | Per-run macros (`37` distinct trajectories / `45` dirs / `19` arms) |
| `results/derived_numbers.tex` | Cross-study, Wave-12 horizon & $2\times 2$ `WT` comparison macros |
| `results/motivation_trajectories.json`, `results/motivation_numbers.tex`, `results/table_motivation.tex` | Section 3 3-trajectory motivating 6+1 covariance & subspace-drift audit (`Muon`, `MuonT`, `WT-MLP-Lag`) |
| `results/mechanism_numbers.tex` | Section 6.7 post-training norm-grafting & active-run invariance macros |
| `results/systems_numbers.tex`, `results/table_systems.tex`, `results/table_crossings.tex` | Section 7 stage + executed 8xA100 `fast_lag` latency, memory & uncensored C4 crossing tables |
| `results/table_main.tex`, `results/table_windows.tex`, `results/table_duplicates.tex` | Master seed-0 table, 5-window paired evolution & duplicate-group tables |
| `figs/method_arch.tex`, `figs/plane_plots.tex`, `figs/attribution_plots.tex`, `figs/transfer_bars.tex` | Pure-TikZ architecture, Cost–Quality Pareto & attribution forest plot, and LR horizon panels |
