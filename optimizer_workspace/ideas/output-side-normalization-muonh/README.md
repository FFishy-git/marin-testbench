# Output-Side Normalization — Hyperball (`MuonH`, `MuonTH`, `WT-MLP-Lag + MuonH`) Campaign (`muonh-20261127`)

Self-contained 130M full-budget (`20 TPP = 2.68B tokens`, `5,120 steps`) experimental workspace evaluating Frobenius Hyperball optimization (`MuonH`, `MuonTH`, and `WT-MLP-Lag + MuonH`) under both the Toner-matched WSD control and Marin's exact `1pct-noclip + AdamH + quintic-5` recipe across 4×8 NVIDIA A100 nodes (32 GPUs).

## Layout
- `PLAN.md` — Mathematical specification, LR scaling derivation, and 32-GPU staged campaign schedule.
- `experiment/block_toner.py` — Unified 3-pillar block optimizer adding `none_all` (`MuonH`), `input_only` (`MuonTH`), `DecayMode.HYPERBALL` (`_block_out_hyperball_raw` + `hyperball_delta`), and Marin's 5-step `quintic` Newton–Schulz (`QUINTIC_NS_COEFFS`), while preserving `0.0` bitwise parity on `default`.
- `experiment/optimizer.py` — Adds PyTorch `AdamH` (`scale_by_adamh` + Frobenius Hyperball) and `head_mode ∈ {"adamw", "adamh"}`.
- `experiment/capture.py` — Marks `none_all` as `NO_INPUT_CAPTURE` so `MuonH` incurs zero forward/backward capture overhead.
- `campaign/configure_muonh_campaign.py` — Generates `130M 1x` JSON configs across Toner-matched (`muonh`, `muonth`, `wt_mlp_lag_muonh`) and Marin-exact (`muonh_marin`, `muonth_marin`, `wt_mlp_lag_muonh_marin`) arms, $\sqrt{2}$ LR bracket, and `seeds 0–3`.
- `campaign/verify_hyperball_gpu.py` — A100 GPU verification of `default` bitwise parity, FP64 Levanter `norm_preserving_update` accuracy, 50-step Frobenius norm conservation, and scale invariance.
- `campaign/deploy_muonh_campaign.py` — Deploys the workspace across `28.216.17.{9,80,144,139}`.
- `campaign/run_muonh_campaign.py` — Dynamic 4-node orchestrator (`gen2`) with non-intrusive per-node v1 drain handover, 8-step Marin-exact 4+4 resume gate, adaptive LR bracket + `Seeds 1–3` scheduler, 16-domain Paloma + norm-conservation audit, and automatic keeper restoration.
