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

Static and numerical verification:

```bash
python -I scripts/check_tex.py      # \input targets, refs, citations, macros, brace/env balance (exits 0)
python -I scripts/check_numbers.py  # every prose numeral against computed JSON values (exits 0)
```

## Outputs

| Path | Contents |
| --- | --- |
| `results/sym_wave_raw.json` | Committed raw metrics, 5-window series, head-to-head pairs & C4 crossings for `wave_sym_fastlag` |
| `results/run_numbers.tex`, `results/evidence_numbers.tex` | Per-run macros (`37` distinct trajectories / `45` dirs / `19` arms) |
| `results/derived_numbers.tex` | Cross-study, Wave-12 horizon & $2\times 2$ `WT` comparison macros |
| `results/motivation_numbers.tex`, `results/table_motivation.tex` | Section 3 motivating 6+1 covariance & subspace-drift audit (`wave1/null_all`) |
| `results/mechanism_numbers.tex` | Section 6.7 post-training norm-grafting & active-run invariance macros |
| `results/systems_numbers.tex`, `results/table_systems.tex`, `results/table_crossings.tex` | Section 7 stage + executed 8xA100 `fast_lag` latency, memory & uncensored C4 crossing tables |
| `results/table_main.tex`, `results/table_windows.tex`, `results/table_duplicates.tex` | Master seed-0 table, 5-window paired evolution & duplicate-group tables |
| `figs/method_arch.tex`, `figs/plane_plots.tex`, `figs/attribution_plots.tex`, `figs/transfer_bars.tex` | Pure-TikZ architecture, Cost–Quality Pareto & attribution forest plot, and LR horizon panels |
