# Output-Side Normalization in MuonT — Systematic Taxonomy, Motivating Audit & 8xA100 Fast-Lag Report

This report summarizes the systematic 5-slot operator taxonomy (`19` configurations across `37` distinct full-budget trajectories / `45` completed run directories), full-trajectory motivating covariance and subspace-drift audit on baseline `MuonT` (`wave1/null_all`), $2\times 2$ asymmetric-versus-symmetric width-thresholded evaluation (`WT-MLP`, `WT-MLP-Lag`, `WT-Sym`, `WT-Sym-Lag`), post-training step-geometry verification, Wave-12 horizon-shift audit, and executed 4-node uncontended 8xA100 `fast_lag` results (`wave_sym_fastlag`) integrated into `main.tex`.

---

## 0. Systematic 5-Slot Operator Taxonomy (`sections/method.tex`, Section 4)

Every configuration is specified by $\mathcal{A}(R_{512}, R_{2048} \mid L_{512}, L_{2048} \mid \tau)$ (`•` full spectral, `⊕` exact age-0 diagonal, `▷` outer `-1/8` only, `---` uncollected identity, `∅` withheld matched null) across six colorblind-safe families (`19` configurations total):
- **I. Baselines (`Ref`):** `Muon` `(---,--- | ---,--- | ---)`, `MuonT` `(•,• | ---,--- | 0)`
- **II. Width-Thresholded Two-Sided (`WT` — $2\times 2$ Proposed Core Family):**
  - `WT-MLP` (`inner_outer_mlpdiag`, `(•,• | •,⊕ | 0)`) — Asymmetric output width-thresholding, synchronous
  - `WT-MLP-Lag` (`inner_outer_mlpdiag_lag`, `(•,• | •,⊕ | 1)`) — Asymmetric output width-thresholding, 1-step `fast_lag`
  - `WT-Sym` (`inner_outer_sym_mlpdiag`, `(•,⊕ | •,⊕ | 0)`) — Symmetric MLP width-thresholding (`in:down` & `out:gate,up` inline diagonal), synchronous
  - `WT-Sym-Lag` (`inner_outer_sym_mlpdiag_lag`, `(•,⊕ | •,⊕ | 1)`) — Symmetric MLP width-thresholding, 1-step `fast_lag`
- **III. Pure-Diagonal Output (`OD` — Zero-`eigh` Limit):** `OD-All` (`inner_outer_leftdiag`, `(•,• | ⊕,⊕ | 0)`), `OD-Outer` (`outer_leftdiag` / `outer_leftdiag_fused`, `(•,• | ▷_⊕,▷_⊕ | 0)`)
- **IV. Dense Two-Sided Ceiling (`DT`):** `DT-All` (`inner_outer_all`, `(•,• | •,• | 0)`), `DT-All-Lag` (`inner_outer_all_lag`, `(•,• | •,• | 1)`)
- **V. Matched Execution-Path Nulls (`Null`):** `Null-WT` (`null_mlpdiag`), `Null-Dense` (`null_all`), `Null-OD` (`null_leftdiag_upstream_fused`)
- **VI. Structural & Input-Degradation Ablations (`Abl`):** `Abl-OuterDense` (`outer_left_all`), `Abl-No2048` (`inner_outer_hidden`), `Abl-DownOnly` (`outer_left_down`), `Abl-DownSwap` (`down_hybrid`), `Abl-OutOnly` (`left_all`), `Abl-BothDiag` (`inner_outer_all_diag`)

---

## 1. Full-Trajectory Motivating Analysis Before Method (`sections/motivation.tex`, Section 3)

Auditing all **32 layers (8 ranks, 352 FP32 EMA matrices per checkpoint)** across **10 checkpoints (`steps 250..5120`, `0.13B–2.68B` tokens)** of `wave1/null_all` ($L = \hat L = I$, pure input-side `MuonT` updates) establishes the design deductively:

| Moment | $d$ | $\mathrm{erank}/d$ (%) | $\mathrm{CV}_{\text{diag}}$ | $\rho_{\text{rms}}$ | $\mathrm{DiagE}$ (%) | $\mathrm{DiagAniso}$ (%) | $R^2(A^{-1/2})$ (%) | $\mathrm{Drift}_{16}$ (`2250→2500`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `in:attn` ($q,k,v$) | 512 | `17.9` | `0.84` | `0.2031` | `3.5` | `1.0` | `10.6` | `0.021` |
| `in:o` | 512 | `20.8` | `0.75` | `0.2120` | `5.3` | `1.8` | `22.5` | `0.094` |
| `in:mlp` (`gate,up`) | 512 | `21.6` | `0.64` | `0.1959` | `3.7` | `0.8` | `6.0` | `0.021` |
| `in:down` (post-SwiGLU) | 2048 | `29.1` | `2.92` | `0.0346` | `47.3` | **`43.4`** (`56.6%` off-diag) | `79.4` | `0.106` |
| `out:q` | 512 | `38.5` | `0.97` | `0.0641` | `30.0` | `18.0` | `62.4` | `0.140` |
| `out:k` | 512 | `23.7` | `1.50` | `0.0774` | `20.5` | `14.4` | `71.8` | `0.165` |
| `out:v` | 512 | `50.4` | `0.58` | `0.0560` | `41.4` | `12.2` | `31.2` | `0.662` |
| `out:o` (residual) | 512 | `85.3` | `0.06` | `0.0301` | `68.0` | **`0.8`** | **`1.2`** | `0.134` |
| `out:down` (residual) | 512 | `84.6` | `0.07` | `0.0312` | `66.3` | **`0.8`** | **`1.2`** | `0.107` |
| **`out:gate` (pre-SwiGLU)** | **2048** | **`87.1`** | **`0.59`** (`1.49` mean) | **`0.0051`** | **`94.9`** | **`82.0`** | **`82.2`** | **`0.240`** |
| **`out:up` (pre-SwiGLU)** | **2048** | **`82.5`** | **`0.75`** (`1.17` mean) | **`0.0073`** | **`91.2`** | **`78.1`** | **`75.9`** | **`0.215`** |

---

## 2. Empirical Results, $2\times 2$ `WT` Causal Validation & Step-Geometry Audit (`sections/results.tex`, Section 6)

1. **Asymmetric Width-Thresholding (`WT-MLP` & `WT-MLP-Lag`):**
   - `WT-MLP` achieves **`-0.0066` nats** held-out selection loss (`-0.0059` net of `Null-WT`) and **`-0.0061` nats** late C4 (`-0.0040` net), winning across all 5 windows, all 3 seeds (`-0.0061`, `-0.0042`, `-0.0040` late C4), and 1M batch (`-0.0035` late C4), at matched step norm (`0.9854`, `22.4°` rotation).
   - Uncontended 8xA100 `fast_lag` `WT-MLP-Lag` achieves **`-0.0047` nats** late selection vs. same-wave `MuonT` (**`-0.0053`** vs. `wave0/muont`), cutting legacy Wave 4 step overhead by >3x (`+17.6%` $\to$ **`+5.8%`** vs. `MuonT`, `21.59 GiB`, `-685.4 s` E2E).
2. **Symmetric Width-Thresholding (`WT-Sym` & `WT-Sym-Lag`) Causally Validates `Finding 3.1`:**
   - Diagonalizing post-SwiGLU `in:down` (`56.6%` off-diagonal from $\mu\mu^\top > 0$) alongside `out:gate,up` incurs a `+0.0181 ± 0.0024` paired training-loss penalty vs. `WT-MLP-Lag` in Window 1 (`+0.0065` / `+0.0071` vs. `MuonT`), before recovering monotonically across LR decay (`-0.0013` / `-0.0015` vs. `MuonT` in Window 5) to **tie/edge `WT-MLP-Lag` on final C4 (`3.4482` vs. `3.4491`, `-0.0029` vs. same-wave `MuonT`)** while saving `21.0 ms/step` and `0.78 GiB` (`20.81 GiB` peak).

---

## 3. Executed 8xA100 `fast_lag` Latency & Wall-Clock Crossings (`sections/systems.tex`, Section 7)

| Arm (`0p5m`, `seed=0`, 8xA100) | Step Med (s) | vs. `Muon` / `MuonT` | E2E `s` (vs. `MuonT`) | Peak GiB | `C4 <= 3.59` | `C4 <= 3.52` | `C4 <= 3.46` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wave0/muon` (`Ref`) | `1.074 s` | `0.0%` / `-1.9%` | `5,889.7 s` | `18.97` | `4,880.0 s` (`4251`) | `5,467.0 s` (`4751`) | **Censored** (`3.4618`) |
| `wave0/muont` (`Ref`) | `1.095 s` | `+2.0%` / `0.0%` | `5,984.6 s` | `20.53` | `4,721.3 s` (`4001`) | `5,566.3 s` (`4751`) | `5,848.2 s` (`5001`) |
| `wave_sym_fastlag/muont` (`Ref`) | `1.099 s` | `+2.3%` / `0.0%` | `5,891.3 s` (`+0.0%`) | `20.53` | `4,619.6 s` (`4001`) | `5,468.4 s` (`4751`) | `5,754.4 s` (`5001`) |
| **`WT-Sym-Lag` (`fast_lag` 8xA100)** | **`1.141 s`** | **`+6.2%` / `+3.8%`** | **`6,179.3 s` (`+4.9%`)** | **`20.81`** | **`4,869.6 s` (`4001`)** | **`5,458.9 s` (`4501`, #1)** | **`6,040.3 s` (`5001`)** |
| **`WT-MLP-Lag` (`fast_lag` 8xA100)** | **`1.162 s`** | **`+8.2%` / `+5.8%`** | **`6,300.3 s` (`+6.9%`)** | **`21.59`** | `4,966.4 s` (`4001`) | **`5,564.6 s` (`4501`)** | `6,157.2 s` (`5001`) |
| `WT-Sym` (no-lag 8xA100) | `1.234 s` | `+14.9%` / `+12.4%` | `6,593.9 s` (`+11.9%`) | `20.74` | `5,187.7 s` (`4001`) | `5,822.6 s` (`4501`) | `6,446.1 s` (`5001`) |
| Legacy `WT-MLP-Lag` (Wave 4) | `1.292 s` | `+20.3%` / `+18.0%` | `6,985.7 s` (`+16.7%`) | `22.68` | `5,500.6 s` (`4001`) | `6,164.4 s` (`4501`) | `6,824.2 s` (`5001`) |

---

## 4. Corrected Wave-12 Tuning Audit & Remaining Confirmation (`sections/tuning.tex`, `sections/program.tex`)

- Carrying `MuonT` from recipe `LR = 0.0226` to its 320-step screening optimum `LR = 0.064` **degraded full-budget `MuonT` itself by `+0.1001` nats** (`3.4755` vs. `3.3754` late selection; `+0.0402` final C4).
- With uncontended 8xA100 `fast_lag` anchors at `LR = 0.0226` and the symmetric width-2048 ablation now completed in `wave_sym_fastlag`, remaining confirmation comprises the `{0.016, 0.032, 0.045}` full-budget LR curve around `0.0226` (Item 1), a same-wave `fast_lag` `Null-WT` run (Item 2), live norm-grafting / 4-step held-diagonal checks (Item 3), and reserved seeds `3/4/5` (Item 4).
