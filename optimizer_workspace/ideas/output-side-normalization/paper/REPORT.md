# Output-Side Normalization in MuonT — Systematic Taxonomy, Motivating Audit & Fast-Lag Report

This report summarizes the systematic 5-slot operator taxonomy, full-trajectory motivating covariance and subspace-drift audit on baseline `MuonT` (`wave1/null_all`), causal net-of-null evaluation, post-training step-geometry verification, Wave-12 horizon-shift audit, and SM80 A100 `fast_lag` systems optimization integrated into `main.tex` via `scripts/render_results_viz.py`.

---

## 0. Systematic 5-Slot Operator Taxonomy (`sections/method.tex`, Section 4)

Every configuration is specified by $\mathcal{A}(R_{512}, R_{2048} \mid L_{512}, L_{2048} \mid \tau)$ (`•` full spectral, `⊕` exact age-0 diagonal, `▷` outer `-1/8` only, `---` uncollected identity, `∅` withheld matched null) across six colorblind-safe families:
- **I. Baselines (`Ref`):** `Muon` `(---,--- | ---,--- | ---)`, `MuonT` `(•,• | ---,--- | 0)`
- **II. Width-Thresholded Two-Sided (`WT` — Proposed Core Family):** `WT-MLP` (`inner_outer_mlpdiag`, `(•,• | •,⊕ | 0)`), `WT-MLP-Lag` (`inner_outer_mlpdiag_lag`, `(•,• | •,⊕ | 1)`), plus modular support for symmetric `WT-Sym` `(•,⊕ | •,⊕ | τ)`
- **III. Pure-Diagonal Output (`OD` — Zero-`eigh` Limit):** `OD-All` (`inner_outer_leftdiag`, `(•,• | ⊕,⊕ | 0)`), `OD-Outer` (`outer_leftdiag` / `outer_leftdiag_fused`, `(•,• | ▷_⊕,▷_⊕ | 0)`)
- **IV. Dense Two-Sided Ceiling (`DT`):** `DT-All` (`inner_outer_all`, `(•,• | •,• | 0)`), `DT-All-Lag` (`inner_outer_all_lag`, `(•,• | •,• | 1)`)
- **V. Matched Execution-Path Nulls (`Null`):** `Null-WT` (`null_mlpdiag`), `Null-Dense` (`null_all`), `Null-OD` (`null_leftdiag_upstream_fused`)
- **VI. Structural & Input-Degradation Ablations (`Abl`):** `Abl-OuterDense` (`outer_left_all`), `Abl-No2048` (`inner_outer_hidden`), `Abl-DownOnly` (`outer_left_down`), `Abl-DownSwap` (`down_hybrid`), `Abl-OutOnly` (`left_all`), `Abl-BothDiag` (`inner_outer_all_diag`)

---

## 1. Full-Trajectory Motivating Analysis Before Method (`sections/motivation.tex`, Section 3)

Auditing all **32 layers (8 ranks, 352 FP32 EMA matrices per checkpoint)** across **10 checkpoints (`steps 250..5120`, `0.13B–2.68B` tokens)** of `wave1/null_all` ($L = \hat L = I$, pure input-side `MuonT` updates while passively capturing all 224 left-cotangent moments $B_t = \mathbb{E}[\delta\delta^\top]$ and 128 right-input moments $C_t = \mathbb{E}[xx^\top]$) establishes the design deductively before any candidate arm is introduced:

### A. 6+1 Motivating Metrics on Pure Baseline `MuonT` (`wave1/null_all`, 32-Layer Medians at Step 2500)

| Moment | $d$ | $\mathrm{erank}/d$ (%) | $\mathrm{CV}_{\text{diag}}$ | $\rho_{\text{rms}}$ | $\mathrm{DiagE}$ (%) | $\mathrm{DiagAniso}$ (%) | $R^2(A^{-1/2})$ (%) | $\mathrm{Drift}_{16}$ (`2250→2500`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `in:attn` ($q,k,v$) | 512 | `17.9` | `0.84` | `0.2031` | `3.5` | `1.0` | `10.6` | `0.021` |
| `in:o` | 512 | `20.8` | `0.75` | `0.2120` | `5.3` | `1.8` | `22.5` | `0.094` |
| `in:mlp` (`gate,up`) | 512 | `21.6` | `0.64` | `0.1959` | `3.7` | `0.8` | `6.0` | `0.021` |
| `in:down` (post-SwiGLU) | 2048 | `29.1` | `2.92` | `0.0346` | `47.3` | `43.4` | `79.4` | `0.106` |
| `out:q` | 512 | `38.5` | `0.97` | `0.0641` | `30.0` | `18.0` | `62.4` | `0.140` |
| `out:k` | 512 | `23.7` | `1.50` | `0.0774` | `20.5` | `14.4` | `71.8` | `0.165` |
| `out:v` | 512 | `50.4` | `0.58` | `0.0560` | `41.4` | `12.2` | `31.2` | `0.662` |
| `out:o` (residual) | 512 | `85.3` | `0.06` | `0.0301` | `68.0` | **`0.8`** | **`1.2`** | `0.134` |
| `out:down` (residual) | 512 | `84.6` | `0.07` | `0.0312` | `66.3` | **`0.8`** | **`1.2`** | `0.107` |
| **`out:gate` (pre-SwiGLU)** | **2048** | **`87.1`** | **`0.59`** (`1.49` mean) | **`0.0051`** | **`94.9`** | **`82.0`** | **`82.2`** | **`0.240`** |
| **`out:up` (pre-SwiGLU)** | **2048** | **`82.5`** | **`0.75`** (`1.17` mean) | **`0.0073`** | **`91.2`** | **`78.1`** | **`75.9`** | **`0.215`** |

- **Pythagorean Anisotropy Split (`DiagE%` vs. `DiagAniso%`):** Subtracting the isotropic component $I_d$ ($\|\tilde M - I_d\|_F^2 = \|\operatorname{diag}(\tilde M)-\mathbf{1}\|_2^2 + \sum_{i\neq j}\tilde M_{ij}^2$) exposes why raw `DiagE%` alone is insufficient: residual-stream `out:o` and `out:down` have `68.0%` and `66.3%` raw diagonal energy only because $\tilde M \approx I_d$, whereas **`99.2%` of their anisotropy is off-diagonal** (`DiagAniso% = 0.8%`, $R^2(A^{-1/2}) = 1.2\%$). By contrast, pre-SwiGLU `out:gate` and `out:up` sit on the finite-sample floor $N_t^{-1/2} = 0.0052$ across the entire `0.13B–2.68B` token trajectory (`0.0051` and `0.0073` median $\rho_{\text{rms}}$; `84.0%` and `81.5%` bulk-layer median `DiagAniso%`, `82.2%` and `75.9%` $R^2(A^{-1/2})$) alongside up to `281x` neuron variance spread (`51.3x` and `32.0x` 32-layer mean).
- **Temporal Subspace Drift & `3,714x` Post-Warmup Fallback Gap:** Cotangent eigenspaces rotate **`5x–32x` faster** than RMSNormed inputs (`Drift_16 = 0.662` on `out:v` and `0.240/0.215` on `out:gate/up` vs. `0.021` on `in:attn/mlp`). Across all 8 GPUs over 5,120 steps, dense output moments trigger **`338,375` repair rounds and `107,911` exact `cuSOLVER eigh` fallbacks** (`21.2/step`, `9.4%` of evaluations) vs. **`369` input fallbacks** (`292x` raw, `167x` per statistic) — and after the initial `0.13B`-token (`250`-step) transient, **`107,701` output fallbacks vs. just `29` input fallbacks (`3,714x` raw, `2,122x` per statistic; `32,838` vs. `10` on Rank 0)**.
- **Three Deductive Design Consequences:** (1) Retain dense input whitening ($R, \hat R$); (2) Width-threshold output whitening (`inner_outer_mlpdiag`, dense at $d=512$ and exact diagonal at $d=2048$, with `inner_outer_leftdiag` as the zero-`eigh` limit); (3) Lag output roots by 1 update (`inner_outer_mlpdiag_lag`) on an asynchronous stream.

---

## 2. Empirical Results & Post-Training Step-Geometry Verification (`sections/results.tex`, Section 6)

1. **Primary Method — `inner_outer_mlpdiag` (and `inner_outer_mlpdiag_lag`):**
   - Achieves **`-0.0066` nats** held-out selection loss (`-0.0059` net of matched execution-path null) and **`-0.0061` nats** late C4 (`-0.0040` net of null), winning across all 5 training windows, all 3 seeds (`-0.0061`, `-0.0042`, `-0.0040` late C4), and 1M batch size (`-0.0035` late C4), while `inner_outer_leftdiag` retains ~68% of the gain (`-0.0045` selection) with zero output eigendecompositions and `left_all` regresses by `+0.0087` nats.
2. **Ruling Out Implicit Step-Size Inflation & Active-Run Invariance (Subsection 6.6):**
   - Instantaneous direction-norm ratio $\|D\|_F / \|D_{\text{in}}\|_F$ across all 224 matrices is **`0.9854` (`-1.46%`)** for `mlpdiag`, **`0.9943` (`-0.57%`)** for `leftdiag`, and **`0.9833` (`-1.67%`)** for `inner_outer_all` (`0.9809` at step 5,120), while rotating updates by **`22.4°`** (`4.3°` and `5.4°` on `o`/`down` in `leftdiag` vs. `17.7°` and `17.0°` in `mlpdiag`), and equilibrium weight norm $\|W\|_F$ is **`-3.00%`** smaller than `muont` (`null_mlpdiag`: `-0.11%`).
   - Cross-checking `wave1/inner_outer_all` at step 2,500 confirms all covariance regimes persist under active two-sided preconditioning (`out:gate`/`up` median $\rho_{\text{rms}} = 0.0050, 0.0092$, $R^2(L) = 93.2\%, 85.4\%$ mean / $95.1\%, 90.1\%$ median; `out:o`/`down` $R^2(L) = 3.0\%, 2.9\%$).

---

## 3. Closing the Systems Gap: `fast_lag` at `3.6%–5.4%` Over Vanilla Muon (`sections/systems.tex`, Section 7)

| Configuration | Bwd Capture | Update GEMMs | Async Roots/RS | Steady `s/step` | Overhead vs. `Muon` (`1.074s`) | Overhead vs. `MuonT` (`1.095s`) | Peak GiB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Muon` (Wave 0) | `-1.8 ms` | — | `0.0 ms` | `1.074 s` | `0.0%` | `-1.9%` | `18.97` |
| `MuonT` (baseline) | `0.0 ms` | `0.0 ms` | `0.0 ms` | `1.095 s` | `+2.0%` | `0.0%` | `20.53` |
| Legacy `all_lag` (Wave 3) | `+92.9 ms` | `+19.8 ms` | `+230 ms` | `1.438 s` | `+33.9%` | `+31.3%` | `24.49` |
| Legacy `mlpdiag_lag` (Wave 4) | `+92.9 ms` | `+15.0 ms` | `+89.1 ms` | `1.292 s` | `+20.3%` | `+18.0%` | `22.68` |
| Wave-10 `null` (0 dense out) | `+96.0 ms` | `0.0 ms` | `0.0 ms` | `1.195 s` | `+11.3%` | `+9.1%` | `22.39` |
| **`fast_lag` `mlpdiag_lag`** | **`+11.1 ms`** | **`+1.7 ms`** | **`+5.2–8.0 ms`** | **`1.113–1.118 s`** | **`+3.6%–4.1%`** | **`+1.6%–2.1%`** | **`21.58` (`-1.11 GiB`)** |
| **`fast_lag` `all_lag`** | **`+10.4 ms`** | **`+2.3 ms`** | **`+18–24 ms`** | **`1.126–1.132 s`** | **`+4.8%–5.4%`** | **`+2.8%–3.4%`** | **`24.55`** |

---

## 4. Corrected Wave-12 Tuning Audit & Final Confirmation Program (`sections/tuning.tex`, `sections/program.tex`)

- Carrying `MuonT` from recipe `LR = 0.0226` to its 320-step screening optimum `LR = 0.064` **degraded full-budget `MuonT` itself by `+0.1001` nats** (`3.4755` vs. `3.3754` late selection; `+0.0402` final C4), and noisy short-schedule tie-breaking split equivalent outer-diagonal equations across `0.0905` (`+0.050` deficit) and `0.064` (`-0.0029` gain).
- Remaining full-budget confirmation brackets `0.0226` (`{0.016, 0.0226, 0.032, 0.045}`) on `fast_lag` `inner_outer_mlpdiag_lag`, `muont`, and `inner_outer_leftdiag`, paired with a same-wave `fast_lag` `null_mlpdiag`.
