# Output-Side Normalization — Weight-Decay Operator Ablation (`wd-ablation-20261127`)

Controlled 130M full-budget (`20 TPP = 2.68B tokens`, `5,120 steps`) factorial comparing isotropic weight decay ($Q = W$), right-shaped decay ($Q = W\hat R^2$, current `WT-MLP-Lag`), and two-sided shaped decay ($Q = \hat L_{t-1}^2 W \hat R^2$) on the frozen `WT-MLP-Lag` runner across 4×8 NVIDIA A100 nodes (32 GPUs).

## Layout
- `PLAN.md` — Experimental design, mathematical specification, and 32-GPU wave schedule.
- `experiment/` — Cloned from `output-side-normalization-lr-bracket/experiment/` with `decay_mode ∈ {"default", "isotropic", "two_sided"}` in `experiment/block_toner.py` (`"default"` path preserved byte-identical).
- `campaign/configure_wd_campaign.py` — Generates `130M 1x` JSON configs across `decay_mode`, `η ∈ {0.01600, 0.02263}`, and `seeds 0–3`.
- `campaign/verify_decay_modes_gpu.py` — A100 GPU verification of `"default"` bitwise parity and `"isotropic"` / `"two_sided"` Ozaki Tensor-Core accuracy against FP64 references.
- `campaign/deploy_wd_campaign.py` — Deploys the self-contained workspace across `28.216.17.{9,80,144,139}`.
- `campaign/run_wd_campaign.py` — Dynamic 4-node worker-pool orchestrator with pre-flight gates, live status (`progress.json`), 16-domain Paloma evaluation, and guaranteed keeper restoration.
