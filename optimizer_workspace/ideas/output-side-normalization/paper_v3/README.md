# Paper v3: Output-Cotangent Preconditioning in Muon-Type Optimizers

Ground-up refined manuscript (`paper_v3`), independent of `paper_v2`.

## Structure & Deductive Flow
- `main.tex` / `paperstyle.sty` / `refs.bib`: Master LaTeX build, clean typography, and verified primary-artifact bibliography.
- `sections/01_intro.tex`: Research question, three core obstacles (theoretical polar ambiguity, $O(d^3)$ & critical-path bottleneck, milli-nat benchmarking confounds), hero synthesis (`fig1_overview.pdf`), and contributions.
- `sections/02_diagnosis.tex`: Three-trajectory structural and temporal audit (`Muon`, `MuonT`, `WT-MLP-Lag`) across all 11 Transformer block moments (`tab:diagnosis`, `fig2_structural_audit.pdf`), establishing the $\dff$ coordinate-dominated vs. $\dm$ rotation-dominated dichotomy and the $16\times$–$32\times$ cotangent subspace drift bottleneck.
- `sections/03_method.tex`: Unified two-sided update equations, width-thresholded factorization (`WT-MLP`), and 3-pillar asynchronous `fast_lag` systems architecture (`WT-MLP-Lag`, `fig3_method_pipeline.pdf`), including chunked row buffers and non-blocking batched root fallbacks.
- `sections/04_protocol.tex`: Pretraining testbed (`130M` and `520M`, `25.39B` corpus) and two mandatory falsification gates (`Null-WT` matched-path control and full-budget LR bracketing vs. the `320`-step short-horizon proxy cliff, `fig4_confounds_lr_null.pdf`).
- `sections/05_scaling.tex`: Multi-seed confirmation (`130M 1x`, seeds 1–3), `4x` Chinchilla overtraining (`80` TPP, `10.74B` tokens, resolving the $\sqrt{2}$ LR confound), `520M` width scaling (`d=1024`, where `OD-All` collapses and `WT-MLP-Lag` scales with `+3.3%` steady-state overhead on `32xA100`), and 16-domain Paloma evaluation (`fig5_scaling_trajectories.pdf`, `fig6_paloma_domains.pdf`).
- `sections/06_analysis.tex`: Controlled 19-arm design-space ablations, multi-seed comparison with `GO-MUON` and `Mousse` (`fig7_ablations_and_related.pdf`), counterfactual direction-grafting mechanistic verification, and $8\text{--}32\times\text{A100}$ quality–overhead Pareto frontier & `520M` scaling timings (`fig8_mechanism_and_pareto.pdf`).
- `sections/07_related_conclusion.tex`: Related work, conclusion, and explicit limitations/future directions.
- `sections/99_appendix.tex`: Exact finite Newton–Schulz and Ozaki FP16-split arithmetic, short-horizon LR sweep table, active-trajectory covariance cross-check, full 19-arm `130M` timing table, `520M` `8x/32xA100` scaling timing table (`tab:bench520m`), and 16-domain Paloma breakdown.

## Rebuilding Figures & PDF
```bash
uv run --offline --no-project --with matplotlib python scripts/generate_figures.py
PATH="/opt/texlive/2026/bin/x86_64-linux:$PATH" latexmk -pdf -interaction=nonstopmode main.tex
```
