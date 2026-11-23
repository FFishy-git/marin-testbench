# Output-Side Normalization Confirmation & Scaling Campaign

Self-contained confirmation and scaling campaign for `output-side-normalization` (see [PLAN.md](PLAN.md)).

## Layout
- `PLAN.md` — Explicit experimental plan, run matrix (28 runs across 4 phases), hyperparameters, gates, and wall-clock budget.
- `experiment/` — Isolated copy of the optimized `fast_lag` runner (`bench_8gpu_opt/_code/experiment`), pointing read-only at the pinned Toner snapshot.
- `campaign/` — Wait script (`wait_related.sh`), config generator (`configure_campaign.py`), node GPU allocation/keeper wrapper (`allocation.py`), multi-wave campaign driver (`run_campaign.py`), and 16-domain Paloma evaluator (`eval_paloma.py`).
- `configs/` — Explicit generated JSON configs for Phases 1–4.
- `results/` — Machine-readable summaries and `RESULTS.md` written as each phase completes.

## Isolation Guarantee
This directory is completely independent of `../output-side-normalization/`. Nothing in `../output-side-normalization/` is modified while `related-selected-full-seed12-20261121` or this campaign runs.
