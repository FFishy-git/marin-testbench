# Improving Training Efficiency & Wall-Clock Loss Pareto of `Muon2-ExpDiag` (`130M`, `1x Chinchilla = 2.68B Tokens`)

## 1. Executive Summary

In `paper_v4`, `Muon2-ExpDiag` (`WT-MLP-Lag`) achieved strong per-step quality (`3.18813` vs. `Muon` `3.20354` on Seeds 1–3), but suffered an **`+8.0%` to `+9.4%` wall-clock training-time penalty** (`99.8–100.2 minutes` vs. `92.0 minutes` for `Muon` and `MuonH` on `8 × NVIDIA A100-SXM4-40GB`). Because `MuonH` accelerates mid-training optimization (`Late-11 = -21.2 mnat` vs. `Muon`) at zero step-time overhead, `paper_v4`'s `Muon2-ExpDiag` lagged behind `MuonH` by **`+23.1 mnat`** at `MuonH`'s wall-clock finish time (`t = 92.0m`).

Under **strictly aligned compilation controls** (`compile_ns = False`, identical `torch.compile(model.body, ...)` and `PR._EXACT` block kernels across all baselines and proposed arms), we introduce **`Muon2-HeadDiag-MetricHB-Lazy12`** (`arm = "headdiag_qk_mlpdiag_lag"`, `decay_mode = "metric_hyperball"`, `K = 12` single-microbatch lazy refresh, `async_host_both + lag_input`, `lr = 0.01345, wd = 0.15`), combining three synergistic innovations:

1. **Lazy Single-Microbatch Capture + Zero-cuSOLVER Async-Host Roots on Both Sides (`K = 12, mb = 1`)**:
   - Compile-time gated `FlexBody` (`capture_in=False, capture_out=False` on `47/48` microbatches = `97.9%` of forward/backward passes) eliminates SwiGLU activation retention, row-sampling kernels, and custom autograd observer nodes on non-refresh microbatches.
   - `BatchExactDenseEngine(async_host=True)` with 1-step lagged roots on **both** inputs (`lag_input = True`) and outputs (`lag_left = True`) offloads exact FP32 eigendecompositions (`[4, 2048, 2048]` + `512×512` shards, scheduled longest-job-first on 4 CPU workers) completely behind the next step's `~1.08s` GPU forward+backward pass, with **zero GPU `cuSOLVER` stalls** and pre-split Ozaki FP16 factors cached across the 12-step lazy interval.
   - Cuts steady-state median step-time overhead vs. `Muon` from **`+6.2%` (`1.1412s` vs. `1.0744s`)** down to **`+0.43%` (`1.0790s` vs. `1.0744s`; `93.14m` vs. `100.00m` across Seeds 1–3)** — and **`+0.10%` (`1.0755s`, `92.79m`)** for `Muon2-HeadDiag-Lazy12` (`Q = W\hat{R}^2`).
2. **Structure-Aware Per-Head Block-Diagonal Output Preconditioning on `(q, k)` (`Side.HEAD_BLOCK`)**:
   - A 32-layer checkpoint moment audit (`step_002560`) revealed that `(q, k)` output second moments (`512×512`) concentrate **`99.39%` and `99.15%` of their total Frobenius anisotropy** inside the **`8 × 64×64` per-head diagonal blocks** (`99.82%` and `98.70%` inverse-root $R^2$), whereas residual writers `(o, down)` are Haar-mixed (`13.1%` vs. random `12.5%`) and `v` is mixed across heads by $W_o^\top$ (`35.4%`).
   - Replacing `512×512` dense roots on `(q, k)` with `8 × 64×64` per-head exact roots + Ozaki FP16-split batched Tensor-Core left multiplication removes `87.5%` (`56/64` blocks) of cross-head finite-sample estimation noise while retaining dense `512×512` roots on `(v, o, down)` and exact `2048` coordinate-diagonal roots on `(gate, up)`.
3. **Right-Metric Tangential Hyperball (`DecayMode.METRIC_HYPERBALL`)**:
   - Standard Euclidean Hyperball (`MuonH`, `Muon2-ExpDiagH`) accelerates mid-run optimization by fixing $\|W\|_F = r_0$, but forfeits `-2.4` to `-5.6 mnat` at cooldown because Euclidean sphere retraction discards right-metric column regularization $W\hat{R}_t^2$.
   - `metric_hyperball_delta` projects the right-metric decay vector $\lambda W \hat{R}_t^2$ onto the tangent space of the Frobenius sphere before retracting (`max_block_rel_norm_drift = 0.57%` across 5,120 BF16 steps), fusing **Hyperball's `-20` to `-37 mnat` mid-run lead** with **`Q = W\hat{R}_t^2`'s late-cooldown convergence**.

---

## 2. Headline Results (`130M`, `1x Chinchilla = 2.68B Tokens`, Strictly Aligned `compile_ns = False`)

### 2.1 Paired Fresh Confirmation (`Seeds 1, 2, 3`, `5,120` Steps, `8 × A100`)

| Method (`compile_ns = False`) | Pure Time (m) | Med Step (s) | Held-Out Sel. Loss | Paired $\Delta$ vs. `Muon` | Paired $\Delta$ vs. `MuonH` | C4 Loss | Late-11 Sel. | Paloma (16-Dom.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Muon` (`lr=0.016`) | `92.01` | `1.0744` | `3.20354 ± 0.0289` | `0.00` | `+0.99` | `3.46404` | `3.33753` | `3.41641` |
| `MuonH` (`lr=0.01131`) | `92.04` | `1.0747` | `3.20255 ± 0.0291` | `-0.99 ± 0.72` | `0.00` | `3.46262` | `3.31630` | `3.41587` |
| `MuonT` (`lr=0.016`) | `94.35` | `1.0957` | `3.19163 ± 0.0302` | `-11.91 ± 1.56` | `-10.92` | `3.45225` | `3.31427` | `3.40491` |
| `MuonTH` (`lr=0.01131`) | `94.80` | `1.1005` | `3.19227 ± 0.0291` | `-11.27 ± 0.42` | `-10.28` | `3.45419` | `3.30471` | `3.40416` |
| `Muon2-ExpDiag` (`paper_v4`) | `100.00` | `1.1412` | `3.18813 ± 0.0306` | `-15.41 ± 1.88` | `-14.42` | `3.44840` | `3.31074` | `3.40050` |
| `Muon2-ExpDiagH` (`orig`) | `99.25` | `1.1372` | `3.19050 ± 0.0282` | `-13.04 ± 0.77` | `-12.05` | `3.45137` | `3.30161` | `3.40293` |
| **`Muon2-HeadDiag-MetricHB-Lazy12` (Ours)** | **`93.14`** | **`1.0790`** | **`3.18714 ± 0.0279`** | **`-16.40 ± 1.29` (3/3)** | **`-15.41 ± 1.23` (3/3)** | **`3.44741`** | **`3.30029`** | **`3.40225`** (**19/19** vs. `Muon`/`MuonH`) |

### 2.2 Mean Held-Out Selection Loss at Matched Wall-Clock Training Time (`Seeds 1–3`, `x = Minutes`)

| Pure Training Time `x` | `Muon` | `MuonH` | `MuonT` | `Muon2-ExpDiag` (`paper_v4`) | **`Muon2-HeadDiag-MetricHB-Lazy12` (Ours)** | Lead vs. `MuonH` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `40.00m` | `3.55619` | `3.54100` | `3.52904` | `3.54164` (`+0.64 mnat`) | **`3.52463`** | **`-16.37 mnat`** |
| `50.00m` | `3.48721` | `3.46263` | `3.45972` | `3.47668` (`+14.05 mnat`) | **`3.45090`** | **`-11.73 mnat`** |
| `60.00m` | `3.42056` | `3.39011` | `3.40097` | `3.41860` (`+28.49 mnat`) | **`3.37939`** | **`-10.72 mnat`** |
| `70.00m` | `3.35102` | `3.32260` | `3.33837` | `3.36222` (`+39.62 mnat`) | **`3.31301`** | **`-9.59 mnat`** |
| `80.00m` | `3.27940` | `3.25856` | `3.27528` | `3.30147` (`+42.91 mnat`) | **`3.24978`** | **`-8.78 mnat`** |
| `85.00m` | `3.24281` | `3.22987` | `3.24072` | `3.27086` (`+40.99 mnat`) | **`3.22021`** | **`-9.66 mnat`** |
| `90.00m` | `3.21014` | `3.20694` | `3.21010` | `3.23796` (`+31.02 mnat`) | **`3.19583`** | **`-11.11 mnat`** |
| **`92.01m` (`Muon`/`MuonH` finish)** | `3.20354` | `3.20255` | `3.19867` | `3.22560` (`+23.05 mnat`) | **`3.18927`** | **`-13.28 mnat`** (`-14.27` vs. `Muon`) |
| `Final (5,120 steps)` | `3.20354` | `3.20255` | `3.19163` | `3.18813` (`@ 100.00m`) | **`3.18714`** (`@ 93.14m`) | **`-15.41 mnat`** (`-16.40` vs. `Muon`) |

### 2.3 Seed-0 Control-Aligned Ablations & Strict Iso-Time (`5,030` Steps) Verification

| Seed-0 Run (`compile_ns = False`) | Steps | Pure Time (m) | Med Step (s) | Held-Out Sel. | C4 Loss | Late-11 Sel. | Paloma (16-Dom.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Muon` | `5,120` | `92.23` | `1.0744` | `3.24610` | `3.46181` | `3.38243` | — |
| `MuonH` | `5,120` | `92.00` | `1.0731` | `3.24420` | `3.46145` | `3.36025` | `3.41605` |
| `MuonT` | `5,120` | `94.65` | `1.0987` | `3.23417` | `3.45106` | `3.35907` | `3.40407` |
| `Muon2-ExpDiag` (`paper_v4`) | `5,120` | `99.84` | `1.1405` | `3.23033` | `3.44720` | `3.35437` | `3.39854` |
| `expdiag_lazy8_ctrl` (`Q = W\hat{R}^2, K=8`) | `5,120` | `93.06` | `1.0774` | `3.23048` | `3.44683` | `3.35465` | `3.39887` |
| `headdiag_qk_lazy12_ctrl` (`Q = W\hat{R}^2, K=12`) | `5,120` | `92.79` | `1.0755` | `3.23102` | `3.44745` | `3.35506` | `3.39899` |
| **`headdiag_qk_metrichb_wd0p15_lazy12_iso5030_ctrl`** (**Strict Iso-Time**) | **`5,030`** | **`91.86`** | `1.0827` | **`3.23301`** | **`3.44968`** | **`3.34083`** | **`3.40415`** |
| **`headdiag_qk_metrichb_wd0p15_lazy12_ctrl`** (**Full 5,120**) | **`5,120`** | **`93.46`** | `1.0823` | **`3.22994`** | **`3.44601`** | **`3.34515`** | **`3.39816`** |

Even when truncated to a **strictly shorter `5,030`-step WSD schedule (`91.86m` pure training time < `92.00m` `MuonH` and `< 92.23m` `Muon`)**, `Muon2-HeadDiag-MetricHB-Lazy12` completes cooldown to `lr = 0` **`-11.19 mnat` below `MuonH` (`3.23301` vs. `3.24420`), `-13.09 mnat` below `Muon` (`3.24610`), and `-11.90 mnat` lower on 16-domain Paloma (`3.40415` vs. `3.41605`)**.

---

## 3. Generated Figures & Artifacts

- Seed-0 Loss vs. Pure Training Time (SVG & 2× PNG): `results/figs/training_loss_vs_time_130m_seed0_eff.svg`, `results/figs/training_loss_vs_time_130m_seed0_eff.png`
- Paired Fresh Seeds 1–3 Loss vs. Pure Training Time (SVG & 2× PNG): `results/figs/training_loss_vs_time_130m_seeds123_eff.svg`, `results/figs/training_loss_vs_time_130m_seeds123_eff.png`
- Control-Aligned Step-Time & Pure Training Time Waterfall (SVG & 2× PNG): `results/figs/step_time_waterfall_130m_eff.svg`, `results/figs/step_time_waterfall_130m_eff.png`
- Full machine-readable summary: `results/eff_summary.json`
- PVC checkpoints & Paloma JSONs: `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/campaigns/efficiency-20261128/`
