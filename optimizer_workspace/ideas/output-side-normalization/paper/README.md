# Width-Thresholded Output-Cotangent Normalization — Paper Source

Scoped empirical, mechanistic, and systems manuscript on output-cotangent moment
normalization in Muon-type optimizers (`inner_outer_mlpdiag`, `inner_outer_leftdiag`,
and the `fast_lag` low-overhead lagged architecture), built from the executed runs in
`marin-testbench/optimizer_workspace/ideas/output-side-normalization/`.

See [REPORT.md](REPORT.md) for the standalone executive summary of the settled
research scope, checkpoint mechanism audit, Wave-12 horizon-shift correction,
and `fast_lag` A100 latency/memory benchmarks.

## Build

```bash
latexmk -pdf main.tex        # pdfLaTeX, as Overleaf and arXiv compile
```

## Regenerating every number and figure

The manuscript reads only generated files for anything derived from a run,
checkpoint audit, or hardware benchmark, so prose cannot drift from the data.
Run from this directory, in order:

```bash
python -I scripts/analyze.py            # full-budget runs
python -I scripts/analyze_lr.py         # 28 learning-rate screening trials
python -I scripts/check_duplicates.py   # distinct-trajectory audit
python -I scripts/derive.py             # cross-study & Wave-12 horizon comparisons
python -I scripts/analyze_mechanism.py  # 32-layer checkpoint covariance & step-norm audit
python -I scripts/analyze_systems.py    # fast_lag A100 stage latency & parity tables
```

(`analyze_mechanism.py --recompute` recomputes `results/mechanism_summary.json`
directly from the saved `$PVC` checkpoints on CPU.)

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
| `results/mechanism_numbers.tex`, `results/table_mechanism.tex` | Section 6 checkpoint covariance & norm-grafting audit |
| `results/systems_numbers.tex`, `results/table_systems.tex` | Section 7 `fast_lag` A100 latency & memory breakdown |
| `results/table_main.tex` | Table 3, main arm comparison |
| `results/table_windows.tex` | five-window paired deltas |
| `results/table_duplicates.tex` | bitwise-identical run groups |
| `figs/mechanism_bars.dat`, `figs/systems_bars.dat` | Section 6 and Section 7 pgfplots series |

## Status of each section

| Section | Status |
| --- | --- |
| Abstract, 1 Introduction | Updated around width-thresholded scope, mechanism audit, and `fast_lag` |
| 2 Background, 3 Method, 4 Protocol | Complete |
| 5 Results | Complete; linked to mechanism and systems sections |
| 6 Mechanism & Step Geometry | **New** — 32-layer checkpoint audit (`352` moments) + norm-grafting proof |
| 7 Fast-Lag Systems Architecture | **New** — `3.6%–4.1%` (`mlpdiag_lag`) and `4.8%–5.4%` (`all_lag`) over Muon |
| 8 Tuning Confound | Corrected — exposes `+0.100`-nat Wave-12 baseline regression at `LR=0.064` |
| 9 Final Confirmation Program | Focused around `LR=0.0226` (`{0.016, 0.0226, 0.032, 0.045}`) on `fast_lag` |
| 10 Limitations, 11 Related Work, 12 Conclusion | Complete |
| Appendices A, B, C | Complete |
