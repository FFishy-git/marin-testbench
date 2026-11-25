# Full-Budget LR-Bracket & Matched-Optimum Campaign Plan (`lr-bracket-20261125`)

- **Workspace:** `/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization-lr-bracket`
- **PVC Campaign Root:** `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/campaigns/lr-bracket-20261125`
- **Cluster:** 4 nodes × 8 NVIDIA A100-SXM4-40GB (`28.216.17.9`, `28.216.17.80`, `28.216.17.144`, `28.216.17.139`)
- **Codebase Parity:** Exact SHA256 clone of `$PVC/runs/bench_8gpu_opt/_code/experiment/` (identical to `confirmation-20261121` Waves 1–4; zero code-path drift).

---

## 1. Motivation & Gap Analysis

In `confirmation-20261121` Phase 1 (`130M 1×` Chinchilla = `2.68B` tokens, `5,120` steps, `0.5M` batch, `seed 0`), the full-budget matrix LR sweep only covered `η ∈ {0.016, 0.02263, 0.032}`:
- Every tested curve (`Muon`, `MuonT`, `WT-MLP-Lag`) decreased monotonically down to `η = 0.016` (left edge of Figure 4b), leaving `η < 0.016` unbracketed.
- `OD-All` lacked `η = 0.016` and `0.01131`; `Muon` lacked `η = 0.01131` and `0.032`.
- On `seed 0`, `WT-MLP-Lag` improved from `0.02263` (`sel=3.23111, c4=3.44905`) to `0.016` (`sel=3.23033, c4=3.44720`), yet Phase 2 (`seeds 1–3`) ran `MuonT` at `0.016` while keeping `OD-All` and `WT-MLP-Lag` at `0.02263` due to the `0.0010`-nat hysteresis rule—both handicapping `WT-MLP-Lag` and preventing apples-to-apples second-half trajectory averaging in Table 3.

---

## 2. Campaign Stages & Run Matrix

### Pre-Flight Gate — 8-Step Bitwise Reproducibility Check (`28.216.17.9`)
Run `wt_mlp_lag_130m_1x_seed0_lr0p016` for 8 steps and verify `max |Δ train_loss| < 1e-6` against archived `confirmation-20261121/runs/wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016`.

### Stage 1 — Complete Full-Budget LR Bracket on `seed 0` (`130M 1×`, `5,120` steps, `auxiliary_lr = 0.0048`)
Completes the symmetric 4-arm × 4-LR matrix `η ∈ {0.01131, 0.01600, 0.02263, 0.03200}` across `Muon`, `MuonT`, `OD-All`, and `WT-MLP-Lag`:

| Priority | Run Name | Arm | Seed | Matrix LR `η` | Purpose |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | `muon_130m_1x_seed0_lr0p0113` | `muon` | 0 | `0.0113137` | Left bracket below `0.016` for `Muon` |
| 2 | `muont_130m_1x_seed0_lr0p0113` | `muont` | 0 | `0.0113137` | Left bracket below `0.016` for `MuonT` |
| 3 | `od_all_130m_1x_seed0_lr0p016` | `inner_outer_leftdiag` | 0 | `0.0160000` | Missing `0.016` point for `OD-All` |
| 4 | `wt_mlp_lag_130m_1x_seed0_lr0p0113` | `inner_outer_mlpdiag_lag` | 0 | `0.0113137` | Left bracket below `0.016` for `WT-MLP-Lag` |
| 5 | `od_all_130m_1x_seed0_lr0p0113` | `inner_outer_leftdiag` | 0 | `0.0113137` | Left bracket below `0.016` for `OD-All` |
| 6 | `muon_130m_1x_seed0_lr0p032` | `muon` | 0 | `0.0320000` | Complete 4-point curve for `Muon` |
| *(Adaptive)* | `<arm>_130m_1x_seed0_lr0p008` | any arm with `sel(0.01131) <= sel(0.016)` | 0 | `0.0080000` | Automatic further left bracket (`0.016 / 2`) until strict U-turn |

### Stage 2 — Unconstrained Per-Arm Optimal & Matched-LR 3-Seed Confirmation (`seeds 1, 2, 3`)
Once Stage 1 brackets the true unconstrained `seed 0` minimum `η*(arm) = argmin_η L_sel(arm, η)` for each arm:
1. Ensure all 4 core arms (`Muon`, `MuonT`, `OD-All`, `WT-MLP-Lag`) have complete `seeds 1, 2, 3` runs at their own true `η*(arm)` (reusing `confirmation-20261121` `wave2a/b/c` where `Muon` & `MuonT` already ran at `0.016` and `OD-All` & `WT-MLP-Lag` ran at `0.02263`).
2. If `η*(WT-MLP-Lag)` or `η*(OD-All)` differs from `η*(MuonT)`, also ensure `seeds 1, 2, 3` exist at `η*(MuonT)` so Table 3 reports both per-arm optimal LRs and strictly matched-LR 3-seed trajectories.
- Under the expected `η* = 0.016` outcome, Stage 2 runs the **6** missing runs:
  - `wt_mlp_lag_130m_1x_seed{1,2,3}_lr0p016` (3 runs)
  - `od_all_130m_1x_seed{1,2,3}_lr0p016` (3 runs)
  Total = **12 runs** (`6` Stage 1 + `6` Stage 2) = **3 waves across 4×8 A100 nodes (~5.0 hours)**.

### Stage 3 — 16-Domain Paloma Evaluation, Consolidated Summary & SwanLab Upload
- Evaluate all newly generated final checkpoints across all **16 Paloma domains** (`campaign/eval_paloma.py`).
- Write `results/lr_bracket_summary.json` and `results/LR_BRACKET_RESULTS.md`.
- Upload all newly completed runs to SwanLab (`output-side-normalization-confirmation`).
