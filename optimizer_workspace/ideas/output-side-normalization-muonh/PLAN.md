# Hyperball (`MuonH`, `MuonTH`, and `WT-MLP-Lag + MuonH`) Campaign (`muonh-20261127`)

- **Workspace:** `/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization-muonh`
- **PVC Campaign Root:** `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/campaigns/muonh-20261127`
- **Queued Behind:** `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/campaigns/wd-ablation-20261127` (`PID 3851349`)
- **Cluster:** 4 nodes × 8 NVIDIA A100-SXM4-40GB (`28.216.17.9`, `28.216.17.80`, `28.216.17.144`, `28.216.17.139` — 32 GPUs total)
- **Model & Horizon:** `130M` (`d_model=512, d_ff=2048`, 32 layers, `seq_len=4096`), `1x` Chinchilla (`20 TPP = 2.68B tokens`, `5,120 steps` at `0.5M` global batch)

---

## 1. Scientific Motivation & Exact Mathematical Formulation

Hyperball Optimization (`MuonH` / `AdamH`; Wen et al., 2026; canonical implementation in `lib/levanter/src/levanter/optim/muonh.py` and `experiments/grug/fast_track/optimizer.py`) replaces decoupled weight decay and Euclidean step addition on 2D weight matrices $W_{p,t} \in \mathbb{R}^{m_p \times n_p}$ with a scale-invariant retraction onto the initial Frobenius sphere $\|W_{p,t}\|_F = \|W_{p,0}\|_F$:

$$r_{p,t} = \|W_{p,t}\|_F, \qquad \hat D_{p,t} = \frac{D_{p,t}}{\max(\|D_{p,t}\|_F, 10^{-10})}, \qquad \widetilde W_{p,t+1} = W_{p,t} - \eta_t \, r_{p,t} \, \hat D_{p,t},$$
$$W_{p,t+1} = \frac{\widetilde W_{p,t+1}}{\max(\|\widetilde W_{p,t+1}\|_F, 10^{-10})} \, r_{p,t}, \qquad \Delta W_{p,t} = W_{p,t+1} - W_{p,t},$$

with all Frobenius norms reduced in FP32. Evaluating output preconditioning on the Hyperball provides **two major scientific benefits**:
1. **Eliminates weight-decay confounds ($\lambda = 0$)**: Neither $W$, $W \hat R^2$, nor $\hat L_{t-1}^2 W \hat R^2$ enters the update because $\|W_{p,t}\|_F$ is conserved on every step.
2. **Eliminates global step-norm and shape-multiplier confounds**: Because $\hat D_{p,t} = D_{p,t} / \|D_{p,t}\|_F$, any positive scalar multiplier on $D_{p,t}$ (including $s_{m,n} = \sqrt{\max(1, m/n)}$ and per-layer norm shifts from $\hat R$ or $\hat L$) cancels identically. Any loss improvement from `MuonH -> MuonTH -> WT-MLP-Lag + MuonH` is therefore **100% attributable to update direction** on the Frobenius sphere.

### Core 1×3 Hyperball Factorial (Matched Auxiliary Control, `head_mode = "adamw"`)
All three primary arms run through the unified distributed `BlockToner` engine (`decay_mode = "hyperball"`, `weight_decay = 0.0` on all 224 block matrices) while keeping `emb`, `RMSNorm`, and `lm_head` on the exact pinned auxiliary `AdamW` configuration (`lr = 0.0048`, `head_wd = 0.1`, `other_wd = 0.0`, `betas = (0.8, 0.98)`, `eps = 1e-15`, `clip_norm = 1.0`) used by `Muon`, `MuonT`, `OD-All`, and `WT-MLP-Lag`:
1. **`MuonH` (`muonh`, `arm = "none_all"`)**:
   $$D_{p,t} = \mathcal{N}_{10}(U_{p,t})$$
   Unpreconditioned polar direction + Frobenius Hyperball (zero input/output capture overhead).
2. **`MuonTH` (`muonth`, `arm = "input_only"`)**:
   $$D_{p,t} = \mathcal{N}_{10}(U_{p,t} R_{p,t}) \, \hat R_{p,t}$$
   Input-whitened `MuonT` direction + Frobenius Hyperball (clean intermediate bridge isolating input vs. output geometry).
3. **`WT-MLP-Lag + MuonH` (`wt_mlp_lag_muonh`, `arm = "inner_outer_mlpdiag_lag"`)**:
   $$D_{p,t} = \hat L_{p,t-1} \, \mathcal{N}_{10}(L_{p,t-1} U_{p,t} R_{p,t}) \, \hat R_{p,t}$$
   Two-sided width-thresholded lagged direction (dense $512 \times 512$ on `q,k,v,o,down`, diagonal $2048$ on `gate,up`, $\tau=1$ lag) + Frobenius Hyperball.

### Secondary Full-Hyperball Triad (`head_mode = "adamh"`, Canonical Marin Recipe Check)
In Marin's `MuonHConfig` (`levanter/optim/muonh.py`), `lm_head` (`model.head.weight`) also uses `AdamH` (bias-corrected Adam direction projected onto the Frobenius sphere at `head_lr = matrix_lr`), leaving **zero weight decay anywhere in the model**. Once Stage 1 identifies the optimal Hyperball matrix LR $\eta^*$, Stage 2b runs the matched Seed-0 triad `{muonh_adamh, muonth_adamh, wt_mlp_lag_muonh_adamh}` at $\eta^*$ to verify that the directional gain also holds under full `MuonH + AdamH`.

---

## 2. Learning-Rate Scaling Analysis & Full-Budget Bracket

Under `MuonW` at $\eta_{\text{MuonW}} = 0.016, \lambda = 0.1$, the initial relative step size $\| \Delta W_0 \|_F / \|W_0\|_F \approx 0.87 \eta_{\text{MuonW}} \approx 0.014$, whereas the weight-decay equilibrium angular step size scales as $\sqrt{\eta_{\text{MuonW}} \lambda} = \sqrt{0.016 \times 0.1} = 0.040$ (`levanter/optim/muonh.py` line 39; compare Marin `MoeHeuristic` at `2.68B` tokens: `0.0174`). Because short-horizon sweeps do not transfer reliably, Stage 1 sweeps a full-budget (`5,120-step`) $\sqrt{2}$-spaced bracket covering both regimes:

$$\eta \in \{0.01600,\; 0.02263,\; 0.03200,\; 0.04525\},$$

with automatic adaptive boundary guards at `0.01131` ($0.016/\sqrt{2}$) and `0.06400` ($0.032 \times 2$) if any arm's Seed-0 minimum falls on the lower (`0.01600`) or upper (`0.04525`) edge, guaranteeing a verified interior optimum $\eta^*_a$ for every arm before multi-seed replication.

---

## 3. Campaign Stages & Queue Handoff (`32 × A100`)

### Stage 0 — Automatic Queue Barrier & Pre-Flight Gates
1. **Queue Barrier**: Detached orchestrator `campaign/run_muonh_campaign.py` polls `wd-ablation-20261127/completion.json` and `PID 3851349` every 60s without touching any GPUs or node locks until `wd-ablation-20261127` completes Stage 3 Paloma and restores all keepers.
2. **Gate 1 (`campaign/verify_hyperball_gpu.py`)**: Verifies on A100 SM80 that `decay_mode="default"` retains `0.0` bitwise parity with `_compiled_block_out`, that `decay_mode="hyperball"` matches FP64 Levanter `norm_preserving_update` to `< 1e-5` relative error across all 130M shapes, that exact Frobenius norm conservation (`| ||W_T||_F / ||W_0||_F - 1 | < 1e-6`) holds over 50 steps for `MuonH`, `MuonTH`, `WT-MLP-Lag-H`, and `AdamH`, and that positive rescaling of $D_{p,t}$ leaves $\Delta W_{p,t}$ invariant.
3. **Gate 2 (`8-Step Distributed Parity & Hyperball Resume Smoke`)**: Confirms `0.000e+00` 8-step parity on `wt_mlp_lag` vs archived `wave1a`, and verifies `4+4` checkpoint save/resume equivalence on `8×A100` for `wt_mlp_lag_muonh_adamh`.

### Stage 1 — Seed 0 Full-Budget LR Bracket (Waves 1–3, `12` core runs, ~4.8h)
- Wave 1–2 (Priorities 1–6): `{muonh, muonth, wt_mlp_lag_muonh} × {0.02263, 0.03200}` on `Seed 0`.
- Wave 2–3 (Priorities 7–12): `{muonh, muonth, wt_mlp_lag_muonh} × {0.01600, 0.04525}` on `Seed 0`.
- Adaptive Boundary Extension (Priorities 13–18, `0` runs if interior): `0.01131` if `0.01600` wins; `0.06400` if `0.04525` wins.

### Stage 2 — Fresh Paired `Seeds 1, 2, 3` Confirmation (`9–12` runs, ~3.7–4.8h)
- Runs `Seeds 1, 2, 3` for each of `muonh`, `muonth`, and `wt_mlp_lag_muonh` at its Stage-1 full-budget optimal $\eta^*_a$.
- If $\eta^*(\text{wt\_mlp\_lag\_muonh}) \neq \eta^*(\text{muonth})$, also runs `wt_mlp_lag_muonh` on `Seeds 1, 2, 3` at $\eta^*(\text{muonth})$ so both tuned-optimum and matched-LR paired multi-seed comparisons are reported.

### Stage 2b — Canonical Marin `MuonH + AdamH` Head Triad (`3` runs on `Seed 0`, ~1.3h)
- Runs `{muonh_adamh, muonth_adamh, wt_mlp_lag_muonh_adamh}` on `Seed 0` at $\eta^*(\text{muonth})$ (`head_mode = "adamh"`, `head_lr = η*`).

### Stage 3 — 16-Domain Paloma Evaluation & Keeper Restoration (~8 min)
- Evaluates all completed campaign runs across all 16 tokenized Paloma domains on 4×8 A100s (also recording exact final Frobenius sphere drift $\|W_{5120}\|_F / \|W_0\|_F - 1$ across all 224 block matrices), writes `results/MUONH_RESULTS.md` and `results/muonh_summary.json`, restores `gpu_matmul.py` keepers on all 4 nodes, and writes `completion.json`.
