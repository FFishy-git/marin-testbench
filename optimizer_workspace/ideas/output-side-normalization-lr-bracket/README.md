# Output-Side Normalization — Full-Budget LR-Bracket & Matched-Optimum Campaign

Self-contained campaign closing the left-boundary learning-rate bracket (`lr <= 0.016`) in Figure 4b / Table 6 and completing matched-optimal 3-seed confirmation (`seeds 1, 2, 3` + 16-domain Paloma) in Table 3 / Figure 5a.

## Layout
- `PLAN.md` — Scientific rationale, run matrix (Stage 1 `seed 0` full-budget LR bracket + Stage 2 `seeds 1–3` matched-optimal replication + Stage 3 16-domain Paloma & SwanLab), and adaptive left-boundary gate.
- `experiment/` — Bitwise-identical clone of `$PVC/runs/bench_8gpu_opt/_code/experiment/` (100% SHA256 match with `confirmation-20261121` Waves 1–4).
- `campaign/configure_lr_campaign.py` — Pre-generates all `130M 1x` JSON configs across `{0.0057, 0.0080, 0.0113, 0.0160, 0.0226, 0.0320}` and `seeds 0–3`.
- `campaign/deploy_lr_campaign.py` — Tar-streams `experiment/`, `campaign/`, `configs/`, and `artifacts/` to all 4 A100 nodes (`28.216.17.{9,80,144,139}`).
- `campaign/run_lr_campaign.py` — Dynamic 4-node worker-pool orchestrator with 8-step bitwise parity smoke gate, adaptive left-boundary LR bracketing, automatic `seeds 1–3` optimal/matched-LR scheduling, 16-domain Paloma evaluation, and SwanLab upload.
