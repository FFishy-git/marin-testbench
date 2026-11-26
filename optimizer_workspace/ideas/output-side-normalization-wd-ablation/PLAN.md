# Weight-Decay Operator Ablation Campaign (`wd-ablation-20261127`)

- **Workspace:** `/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization-wd-ablation`
- **PVC Campaign Root:** `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/campaigns/wd-ablation-20261127`
- **Cluster:** 4 nodes × 8 NVIDIA A100-SXM4-40GB (`28.216.17.9`, `28.216.17.80`, `28.216.17.144`, `28.216.17.139` — 32 GPUs total)
- **Model & Horizon:** `130M` (`d_model=512, d_ff=2048`, 32 layers, `seq_len=4096`), `1x` Chinchilla (`20 TPP = 2.68B tokens`, `5,120 steps` at `0.5M` global batch)

---

## 1. Scientific Question & 1×3 Weight-Decay Factorial

In `WT-MLP-Lag`, each matrix parameter $W_{p,t} \in \mathbb{R}^{m \times n}$ takes the decoupled update
$$W_{p,t+1} = W_{p,t} - \eta_t \left( s_p \, \hat L_{p,t-1} \, \mathcal{N}_{10}(L_{p,t-1} U_{p,t} R_{p,t}) \, \hat R_{p,t} + \lambda \, Q_{p,t} \right), \qquad \lambda = 0.1,$$
where both $\hat L_{p,t-1}$ and $\hat R_{p,t}$ are unit-RMS normalized ($\frac{1}{m}\operatorname{tr}(\hat L_{p,t-1}^2) = 1$, $\frac{1}{n}\operatorname{tr}(\hat R_{p,t}^2) = 1$). Holding the preconditioned descent direction $D_{p,t}$ and global average decay strength $\lambda = 0.1$ strictly fixed, we compare all three canonical weight-decay operators:

1. **Isotropic (`wt_mlp_lag_wd_iso`, `decay_mode = "isotropic"`)**:
   $$Q_{p,t} = W_{p,t}$$
   Pure Euclidean Frobenius regularization $\frac{\lambda}{2}\|W\|_F^2$ directly in $W$-space.
2. **Right-Shaped (`wt_mlp_lag`, `decay_mode = "default"`, archived baseline)**:
   $$Q_{p,t} = W_{p,t} \hat R_{p,t}^2$$
   Inherited from `MuonT` ($W = \widetilde W \hat R$), matching `MuonT`'s regularizer to isolate $D_{p,t}$.
3. **Two-Sided Shaped (`wt_mlp_lag_wd_lr2`, `decay_mode = "two_sided"`)**:
   $$Q_{p,t} = \hat L_{p,t-1}^2 W_{p,t} \hat R_{p,t}^2$$
   Induced by Euclidean $\ell_2$ decay in the two-sided post-polar coordinates $W = \hat L \widehat W \hat R$ using the already-ready $\tau=1$ lagged factor $\hat L_{p,t-1}$ (dense $512 \times 512$ on `q,k,v,o,down`, diagonal $2048$ on `gate,up`, identity on step 1; zero added synchronization stall).

---

## 2. Campaign Stages & 32-GPU Schedule

### Pre-Flight Gates (`28.216.17.9`)
1. **GPU Decay-Operator Math Gate (`campaign/verify_decay_modes_gpu.py`)**: Verifies exact bitwise agreement (`0.0` max abs diff) between `decay_mode="default"` and the unmodified `WT-MLP-Lag` kernel, and `< 1e-6` relative agreement against independent FP64 reference equations for `"isotropic"` and `"two_sided"` across `(512, 512)`, `(512, 2048)`, and `(2048, 512)` (dense $\hat L$, diagonal $\hat L$, and `None` step-1 $\hat L$).
2. **8-Step Bitwise Parity Gate**: Runs `wt_mlp_lag_130m_1x_seed0_lr0p016` (`decay_mode="default"`) on 8×A100 for 8 steps and requires `max |Δ train_loss| <= 1e-5` against archived `confirmation-20261121/runs/wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016`.

### Stage 1 — Seed 0 Two-LR Screen (Wave 1 across 4×8 A100s, ~1.7h)
Simultaneously tests both top learning rates `η ∈ {0.01600, 0.02263}` on `seed 0` to check both fixed-LR performance and any LR-optimum shift:
- Priority 1: `wt_mlp_lag_wd_iso_130m_1x_seed0_lr0p016` (`Q = W`, `η = 0.01600`)
- Priority 2: `wt_mlp_lag_wd_lr2_130m_1x_seed0_lr0p016` (`Q = L^2 W R^2`, `η = 0.01600`)
- Priority 3: `wt_mlp_lag_wd_iso_130m_1x_seed0_lr0p0226` (`Q = W`, `η = 0.02263`)
- Priority 4: `wt_mlp_lag_wd_lr2_130m_1x_seed0_lr0p0226` (`Q = L^2 W R^2`, `η = 0.02263`)

### Stage 2 — Fresh Paired `Seeds 1, 2, 3` Confirmation (Waves 2–3, ~3.4h)
- Runs `seeds 1, 2, 3` at the shared confirmation optimum `η* = 0.01600` (`6` runs: `{wd_iso, wd_lr2} × {seed1, seed2, seed3}`) for direct paired comparison against archived `WT-MLP-Lag (W R^2)` (`3.18813 ± 0.03061`), `MuonT` (`3.19163 ± 0.03024`), `OD-All` (`3.19192 ± 0.02996`), and `Muon` (`3.20354 ± 0.02888`).
- If either new decay mode achieves strictly lower `seed 0` held-out loss at `0.02263` than at `0.01600` by `> 0.0005` nats, automatically also schedules `seeds 1, 2, 3` at `0.02263` (paired against archived `wt_mlp_lag@0p0226`).

### Stage 3 — 16-Domain Paloma Evaluation & Keeper Restoration
- Evaluates all 16 Paloma domains on all newly completed runs plus archived `wt_mlp_lag` `seed 0` (`0.01600` and `0.02263`) checkpoints.
- Writes consolidated reports to `results/wd_ablation_summary.json` and `results/WD_ABLATION_RESULTS.md`.
- Every node allocation automatically restores its `gpu_matmul.py` keeper upon completion.
