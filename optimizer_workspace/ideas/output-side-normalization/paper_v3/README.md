# Paper v3: Where Output Preconditioning Matters in Muon Optimizers

`main.pdf` is the compiled manuscript. `main.tex`, `paperstyle.sty`, and
`sections/` contain the editable text and layout. The paper uses a single-column
sans-serif design with ten vector figures. Tables use full-width `tabularx`
layouts at fixed type sizes without `resizebox`.

The main text pairs each empirical claim with a focused anchor table: projection
geometry and drift (`tab:diagnosis_summary`), width-split factor assignment
(`tab:taxonomy`), pretraining regimes and seed-0 execution/learning-rate
controls (`tab:regimes`, `tab:controls`), confirmatory scaling across seeds,
horizon, and width (`tab:main_scaling`), question-grouped placement ablations
and polar baselines (`tab:ablations`, `tab:related`), counterfactual rotation
and width delocalization (`tab:mechanism_rotation`), and median/mean step-time
scaling (`tab:timing_summary`), and output-subspace trajectories
(`tab:output_drift`, Figures 9--10). Complete 352-matrix audits, inner/outer spectral
decompositions, full learning-rate sweeps, absolute and per-seed losses, the
19-arm multi-wave ablation ledger, complete 130M/520M timing windows, and
16-domain Paloma scores are in the appendix.

## Rebuild

From this directory:

```bash
uv run --offline --no-project --with numpy python scripts/generate_tables.py
PATH="/opt/texlive/2026/bin/x86_64-linux:$PATH" latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The table script generates the LaTeX files in `tables/` from existing results
and logs; it does not launch training. Regenerate tables from those records
rather than editing their rounded values manually.

Figures 1--3 use explicit diagonal and off-diagonal anisotropy labels, with both
shares defined in equation (3).
A manuscript rebuild uses the PDFs directly. To regenerate all figures from
the source records, run:

```bash
uv run --offline --no-project --with matplotlib python scripts/generate_figures.py
```

Figures 9--10 and their table use the separate drift analysis below.

## Data provenance

- `../paper/results/confirmation_campaign.json`: full-budget controls, fresh
  paired seeds, final and late losses, and long-horizon/width comparisons.
- `../paper/results/summary.json`: exploratory placement ablations. Synchronous
  WT-MLP and the earlier lag ablation share the original MuonT reference; the
  later fast-lag confirmation uses its own same-path reference. These results
  must not be combined as if they were one implementation-matched experiment.
- `../paper/results/mechanism_summary.json`: three-trajectory moment audit.
- `tables/rotation_mechanism_metrics.json`: saved-checkpoint spectral probes,
  produced by `scripts/analyze_rotation_mechanism.py`. The 520M probes use a
  WT-trained state with counterfactual output factors.
- `tables/output_drift_metrics.json.gz`: output-subspace trajectories from
  uninterrupted training moments and 93 fixed-data checkpoint probes across
  Muon, MuonT, and WT. It contains checkpoint receipts, per-interval layer means, paired comparisons,
  boundary-gap sensitivity, and variation between two fixed probe sets.
- `../experiment_records/paloma_eval/`: domain-level scores, including the
  positive OD-All TwitterAAE delta at 130M/80 TPP.
- `../paper/results/measured_8gpu_timings.json` and
  `../paper/results/measured_520m_timings.json`: median and mean step times,
  including the complete windows and restored optimizer states.
- `../paper/results/sym_wave_raw.json`: recorded target-loss crossings and
  elapsed times. The 5.3% token saving at C4 3.52 does not yield a time saving
  in those recorded runs.

The main timing comparison reports both estimators: median overhead over MuonT
is 5.3% at 130M and 3.3% at 520M; mean overhead is 8.9% and 11.4%. The manuscript
claims lower loss at a fixed token budget, not a demonstrated training speedup.
Exact polar identities, empirical moment patterns, and directional proxies are
identified separately. Appendix B.1 defines each diagnostic, derives subspace
drift from orthogonal projection, and cites Björck and Golub (1973), Yu et al.
(2015), and Roy and Vetterli (2007) for principal angles, eigenvalue-gap
sensitivity, and effective rank. Root R² is identified as the paper's relative
squared-error reduction against identity.

The replay checks have different scopes: Muon reproduces its recorded trajectory
bitwise; WT-MLP-Lag's passive and active accumulators agree within its resumed
trajectory, which differs from the uninterrupted run. B.1 gives both the
within-replay error and the mismatch against the original WT checkpoint.

## Reproduce the output-drift analysis

The shared diagnostic directory is:

```bash
drift_root=/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization
drift_diag="$drift_root/diagnostics/output-drift-trajectories-20261125"
```

`probe_manifest_all_methods.json` records all 93 source checkpoints;
`probe_manifest_muon.json` selects the 31 added Muon checkpoints. The original
62-checkpoint manifest is preserved as `probe_manifest.json`. `ema_drift.json`
contains the uninterrupted 130M training-moment comparisons, with cached
bases in `ema_basis_cache/`. `probes/probe_*.npz` contains two sets of leading
64 eigenvectors, gaps at ranks 8/16/32/64, numerical residuals, checkpoint
configuration, and source/data hashes. `drift_summary_muon.json` retains all layer
observations; the smaller JSON in `tables/` supports the manuscript figure
and table without rereading model weights.

Regenerate the CPU summary, Figures 9--10, and their table:

```bash
OPENBLAS_NUM_THREADS=1 uv run --offline --no-project --python /opt/conda/envs/torch-base/bin/python python scripts/summarize_drift_trajectories.py \
  --diagnostics "$drift_diag" --manifest "$drift_diag/probe_manifest_all_methods.json" \
  --out "$drift_diag/drift_summary_muon.json" \
  --compact tables/output_drift_metrics.json.gz
uv run --offline --no-project --python /opt/conda/envs/torch-base/bin/python --with matplotlib python scripts/plot_drift_trajectories.py \
  --summary tables/output_drift_metrics.json.gz
```

`scripts/analyze_drift_trajectories.py --root "$drift_root" --out "$drift_diag"`
rebuilds the EMA analysis. The GPU probe source is
`scripts/probe_drift_checkpoints.py`; it imports the frozen model/capture path
from `candidates/sym-mlpdiag-fastlag-20261120`. The executed allocation command
was `python "$drift_diag/run_probe_allocation.py" --manifest
"$drift_diag/probe_manifest.json" --workers 8 --name full`, using the shared
node lock and keeper restoration procedure. Muon uses
`run_muon_probe_allocation.py --manifest probe_manifest_muon.json --workers 8
--name muon` under the same directory and allocation procedure, with its
probe source frozen in `source_muon/`. These commands reuse completed
probe files. Original worker logs and allocation status remain in that
directory. Frozen script copies are under `source/`; per-checkpoint metadata
records the probe-script hash observed during execution. The initial cached
pilot, `130M_MuonT_000250`, predates the model/capture hash fields; the other
records from the original allocation have matching model/capture hashes.
The Muon allocation runs the frozen probe source throughout. The pilot uses
the same frozen candidate path. Probe-script metadata/formatting changed
during the original allocation,
so these recorded file hashes are not all process-start snapshots.

The comparison covers all saved checkpoints, with different temporal
resolution: 130M seed 0 has 19 equal 250-step intervals from steps 250 to
5,000 plus the separate 120-step tail; 520M seed 0 has three 2,560-step
intervals from steps 2,560 to 10,240. The latter omits the first quarter and
cannot establish 250-step behavior. Three fresh 130M seeds supply one
2,560-step interval per method, with all three methods at learning rate 0.016. The densely saved seed-0
trajectories use learning rate 0.016 for Muon and 0.02263 for MuonT/WT.
The seed-0 comparison therefore includes the learning-rate difference.
No additional training is performed by these probes.

Each checkpoint uses the same two disjoint sets of 16 token windows and the
same compiled BF16 path. A full uncentered moment uses 4,576 sampled rows per
probe set. Probe-set drift scores are averaged after computing their separate
moments. The sampling windows are held out for seed 0; fresh training seeds
use different partitions. The saved probe-token SHA-256 is
`9faa42473155fd0dabc6c5bc088b9ab05bb09451070d15240eb7ed0f831673dd`.
Reported percentage reduction is `100 * (1 - mean(method) / mean(Muon))`
for paired observations, with separate MuonT and WT results. The original
WT-versus-MuonT comparison is retained as `incremental_comparisons` in the
summary. Muon has no saved output training EMA; the EMA audit remains a
separate MuonT-passive versus WT comparison. Layers and intervals are
correlated; they are not independent seed replications or confidence intervals.

With Muon as the baseline, mean fixed-data top-16 drift falls by 16.7% at
130M and 18.9% at 520M under WT. MuonT changes it by +2.9% and -1.9%,
respectively. WT has lower average drift in six roles at both sizes; value
projection drift is higher by 4.6% at 130M and 2.3% at 520M. These signs
persist at ranks 8/16/32/64 in each probe set, and under the 1% boundary-gap
filter. At the common learning rate 0.016, WT has lower drift in all seven
roles in all three fresh 130M seeds, with all-role reduction 18.8% ± 0.6%
(mean ± sample SD). MuonT's corresponding reduction is 3.3% ± 2.9%.

The incremental WT-versus-MuonT results are unchanged: 19.0% and 17.3%
all-role reductions at 130M and 520M. The training-EMA comparison is less
uniform, including reversals for some roles at larger subspace sizes.
The paper distinguishes these descriptive patterns from causal claims about
loss, population eigenspaces, or root refresh frequency.

## Restoration and validation

The current labels in Figures 1--3 distinguish diagonal from off-diagonal
anisotropy; the captions and equation (3) explain that these are relative
shares. Table generation is separate from figure generation so rebuilding
tables preserves the figure PDFs.

The figure-restoration build had 24 pages, no LaTeX warnings or overfull/underfull
boxes, and 62 resolved cross-reference labels. Its rendered figure pages were
checked for layout, and regenerating tables left their contents unchanged.

The anisotropy-label revision builds to 28 pages, matching the immediately
preceding PDF, and preserves equation number (3). All 11 displayed diagonal
shares in Figure 1 and the numeric text in Figures 2--3 are unchanged. The
revised equation and figures were rendered and checked for label clipping.
The redesigned `tabularx` table build compiles cleanly with zero LaTeX warnings
and zero overfull or underfull boxes across all 18 tables (`ablations_full.tex`,
`paloma.tex`, `rotation_detail.tex`, and all main-body tables verified). The
repository lint entry point passes on `scripts/generate_tables.py` and `README.md`
(existing issues in `scripts/generate_figures.py` are unchanged).

The B.1 revision expands the diagnostic explanations and references while
preserving equation numbers (3), (12), and (13). The rebuilt PDF has 32 pages;
LaTeX reports no warnings or overfull/underfull boxes, and the new citations
and Figure 2(b) cross-references resolve. Projection, principal-angle, and
projector forms of drift were checked numerically, together with the diagonal
decomposition, CV identity, and effective-rank examples. The edited text,
bibliography, and README pass the repository lint entry point.

The original two-method output-drift revision built to 36 pages, with Section 7 beginning on
page 16, Table 10 on page 19, and Figures 9--10 on pages 20--21. Equations
(3), (12), and (13) retain their numbers. The build has no LaTeX warnings or
overfull/underfull boxes. The new table and figure pages were rendered and
checked; a separate cold-reader review clarified averaging, matching, and
probe-data scope. `analysis_validation.json` in the diagnostic directory
records completeness, key uniqueness, matching checkpoint mtimes, and
projection-form checks. The maximum eigensolve relative residual is
2.54e-15; sampled projection-form agreement after FP32 basis storage is
within 4.9e-9. The GPU allocation completed and restored the keeper.
The scoped manuscript/script/data checks pass through `infra/pre-commit.py`.
Including the appended ENDEAVOR/DETAILS logs passes the content checks but
triggers the file-size check; both logs already exceeded that limit before
this diagnostic entry.

The Muon-baseline extension adds 31 checkpoint probes, for 93 total across
15 training runs. `muon_analysis_validation.json` verifies 34,944 fixed-data
observations with no duplicate keys or missing probes, unchanged checkpoint
mtimes, the reduction formula, matched probe data and positions, and the
executed Muon probe-source hash. All 23,296 original fixed-data observations
are exactly unchanged. The maximum FP64 relative eigensolve residual is
2.531e-15. The 6,400 EMA observations retain their MuonT-passive reference
because Muon did not save those moments. `muon_allocation.json` records
completion and successful keeper restoration at 16:27:41 Asia/Shanghai on
November 25, 2026. `muon_source_receipt.json` records the frozen probe source
and final analysis/plot source hashes. The original two-method compact
summary is preserved as `output_drift_metrics_before_muon.json` in the
diagnostic directory.

The Muon-baseline PDF has 36 pages: Section 7 begins on page 16, Table 10
is on page 19, and Figures 9--10 are on pages 20--21. The final build has
no LaTeX warnings or overfull/underfull boxes, and equations (3), (12), and
(13) retain their numbers. The section, table, and figures were rendered
and checked. A contextless reader review led to explicit cotangent/loss
definitions, exact token positions, interval scope, identification of the
separate passive EMA control, and weighting after gap filtering. Filtered
means weight retained layer--interval--probe observations equally, so the
two sets can contribute unequally when their retention counts differ.
`muon_manuscript_validation.json` records the final PDF hash, page labels,
preservation of Figures 1--8, and scoped lint result.

The original PDF and sources are preserved at
`/root/tmp_paper_tools/paper_v3_before_editorial_20261125_031307/`.
The revised state immediately before figure restoration is preserved at
`/root/tmp_paper_tools/paper_v3_before_figure_restore_20261125/`.
The complete paper immediately before adding Muon to the drift comparison
is preserved at `/root/tmp_paper_tools/paper_v3_before_muon_drift_20261125/`.
