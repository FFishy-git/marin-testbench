# Whitening the Other Side — paper source

Draft manuscript on output-cotangent moment normalization in Muon-type
optimizers, built from the executed runs recorded in
`marin-testbench/optimizer_workspace/ideas/output-side-normalization/`.

## Build

```bash
latexmk -pdf main.tex        # pdfLaTeX, as Overleaf and arXiv compile
```

Builds clean with TeX Live 2026 (pdfLaTeX): 46 pages, no errors, no unresolved
references or citations. TeX Live was installed from upstream via the corporate
proxy (`star-proxy.oa.com:3128`); the distribution's own `texlive` packages are
from 2018 and are too old for this source, which needs `pgfplots` at
`compat=1.18`, `xurl`, and kernel `\NewDocumentCommand`. The
template's `smile.sty` and `ims.bst` were not present in
`/root/dl/runtime/script/template`, so `main.tex` loads the standard packages
they would have provided and uses `plainnat` for the bibliography;
`macro.tex` and `paperstyle.sty` are the template's own files, copied
unchanged. Because `smile.sty` is absent, `main.tex` must load `caption`,
`hyperref` and `cleveref` *before* `\input{macro.tex}`: `macro.tex` calls
`\captionsetup` at its top and `\crefname` in its finding box.

## Regenerating every number and figure

The manuscript reads only generated files for anything derived from a run, so
prose cannot drift from the histories. Run from this directory, in order:

```bash
python -I scripts/analyze.py           # full-budget runs
python -I scripts/analyze_lr.py        # 28 learning-rate screening trials
python -I scripts/check_duplicates.py  # distinct-trajectory audit
python -I scripts/derive.py            # cross-study comparisons (needs the first two)
```

Two checks substitute for a trial compile, since no TeX is installed here:

```bash
python -I scripts/check_tex.py      # \input targets, refs, citations, brace/env balance
python -I scripts/check_numbers.py  # every prose numeral against the computed values
```

`check_numbers.py` extracts decimal literals from the section sources, skipping
plot bodies, and verifies each against the values computed from the run
histories or against an explicit list of configuration and externally sourced
constants. Both checks currently pass.

Inputs are the per-update `metrics_from_000000.jsonl` files under
`$PVC/runs/` and `$PVC/campaigns/`, where

```
PVC=/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization
```

Outputs:

| Path | Contents |
| --- | --- |
| `results/run_numbers.tex` | per-run macros (deltas, step times, memory, fallbacks) |
| `results/lr_numbers.tex` | screening-trial macros |
| `results/derived_numbers.tex` | cross-study comparisons quoted in the prose |
| `results/evidence_numbers.tex` | run-directory and distinct-trajectory counts |
| `results/table_main.tex` | Table 3, the main arm comparison |
| `results/table_windows.tex` | Table 6, five-window paired deltas |
| `results/table_duplicates.tex` | Table 5, bitwise-identical run groups |
| `results/summary.json`, `results/lr_summary.json` | machine-readable dumps for cross-checking |
| `figs/plane_plots.tex` | Figure 1 scatter plots, labels included |
| `figs/train_delta.dat` | Figure 2 paired-delta series |
| `figs/freshness.dat` | Figure 3 basis age and fallback series |
| `figs/seeds.dat` | Figure 4 seed replication |
| `figs/lr_sweep.dat`, `figs/transfer_bars.tex` | Figure 5 tuning panels |
| `figs/windows.dat`, `figs/transfer.dat` | supporting series |

All figures are pgfplots drawn from these files, so there are no external
image dependencies.

## Status of each section

| Section | Status |
| --- | --- |
| Abstract, 1 Introduction | Drafted; numbers generated |
| 2 Background | Drafted. Needs a citation for the pinned preconditioned-MuonT implementation (marked `\todo`) |
| 3 Method | Drafted |
| 4 Protocol | Drafted |
| 5 Results | Drafted; all tables and figures generated |
| 6 Tuning confound | Drafted; generated from the 28 screening trials |
| 7 Program | Drafted — the prioritized experiment list |
| 8 Limitations, 9 Related work | Drafted |
| 10 Conclusion | Drafted |
| A Arm definitions | Drafted |
| B Run details | Drafted |
| C Finite Newton--Schulz | Drafted |
| Author list, affiliations | `\todo` placeholders |

## Evidence policy

- Any number derived from a run comes from a generated file, never typed by
  hand.
- Deltas are always against the **seed-matched** baseline; selection loss is
  compared only within a seed, C4 across seeds.
- Geometry claims are stated as candidate minus **matched null**, not candidate
  minus baseline.
- Bitwise-identical reruns are counted once: 41 run directories hold 33
  distinct trajectories, audited by `scripts/check_duplicates.py`.
- Retrospective loss-target crossings are labelled exploratory wherever they
  appear; they are not launch-anchored time-to-target measurements.

## Known gaps the draft states rather than hides

No full-budget tuned comparison; no matched null at the selected learning rate;
richness and freshness confounded in the diagonal arms; no mechanistic
measurement of why the output moment helps; no seed reserved from exploration;
no launch-anchored efficiency measurement. Section 7 lists these as nine runs
plus checkpoint-only probes.
