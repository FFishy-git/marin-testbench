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

## 2. Empirical Results, $2\times 2$ `WT` Recipe Comparison & Step-Geometry Audit (`sections/results.tex`, Section 6)

1. **Asymmetric Width-Thresholding (`WT-MLP` & `WT-MLP-Lag`):**
   - `WT-MLP` achieves **`-0.0066` nats** held-out selection loss (`-0.0059` net of `Null-WT`) and **`-0.0061` nats** late C4 (`-0.0040` net), winning across all 5 windows, all 3 seeds (`-0.0061`, `-0.0042`, `-0.0040` late C4), and 1M batch (`-0.0035` late C4), at matched step norm (`0.9854`, `22.4°` rotation).
   - Uncontended 8xA100 `fast_lag` `WT-MLP-Lag` achieves **`-0.0047` nats** late selection vs. same-wave `MuonT` (**`-0.0053`** vs. `wave0/muont`), cutting legacy Wave 4 step overhead by >3x (`+17.6%` $\to$ **`+5.8%`** vs. `MuonT`, `21.59 GiB`, `-685.4 s` E2E).
2. **Symmetric Width-Thresholding (`WT-Sym` & `WT-Sym-Lag`) Has an Early Recipe Penalty:**
   - Diagonalizing post-SwiGLU `in:down` (`56.6%` off-diagonal anisotropy) alongside `out:gate,up` incurs a `+0.0181 ± 0.0024` paired training-loss penalty vs. `WT-MLP-Lag` in Window 1 (`+0.0065` / `+0.0071` vs. `MuonT`), before recovering across LR decay (`-0.0013` / `-0.0015` vs. `MuonT` in Window 5) to **tie/edge `WT-MLP-Lag` on final C4 (`3.4482` vs. `3.4491`, `-0.0029` vs. same-wave `MuonT`)** while saving `21.0 ms/step` and `0.78 GiB` (`20.81 GiB` peak).
   - Section 6.4 now includes [exact training curves](figs/wt_lag_training.pdf): all `5,120` logged losses per arm, an early zoom, and the exact paired difference, without smoothing, binning, or downsampling. The five window means of symmetric-minus-asymmetric loss are `+0.01807148`, `+0.00412952`, `+0.00256797`, `+0.00103564`, and `+0.00051875` nats. Late and final mixture selection remain worse even though final C4 is slightly better.
   - The diagonalized groups have the highest inner-root fidelity (`out:gate` `82.2%`, `in:down` `79.4%`, `out:up` `75.9%`). This Frobenius score does not measure momentum-weighted optimizer-direction error. `in:down` also has lower outer-root fidelity (`66.7%` at update `2,500`). The recipe comparison changes inner whitening, outer transport, input-shaped decay, and execution together; it does not isolate missing off-diagonal inner conditioning. Channel-mean contributions have not been measured.
   - [Full-precision data](figs/wt_lag_training.dat) and a [provenance receipt](results/wt_lag_training.json) preserve the original commands, source/configuration identities, SHA-256 hashes, and row counts. Run `python -I scripts/plot_wt_lag_training.py` from this paper directory to export the original same-wave logs, or add `--from-data` to rebuild offline. A checkpoint replay separating inner, outer, and decay factors is proposed but has not run.

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

## 4. Corrected Wave-12 Tuning Audit & Completed 30-Run Confirmation & Scaling Campaign (`sections/tuning.tex`, `sections/results.tex` §6.9, `sections/program.tex`)

- Carrying `MuonT` from recipe `LR = 0.0226` to its 320-step screening optimum `LR = 0.064` **degraded full-budget `MuonT` itself by `+0.1001` nats** (`3.4755` vs. `3.3754` late selection; `+0.0402` final C4).
- The isolated **30-run `fast_lag` confirmation & scaling campaign** (`confirmation-20261121`, 22 post-hoc 16-domain Paloma evaluations) is now **complete** and integrated via `scripts/integrate_confirmation_campaign.py` into Section 6.9 (`Table 7`, `Figure 9`):
  - **Phase 1 (`130M 1×` full-budget LR bracket + `fast_lag` `Null-WT`):** `Null-WT` lands within `-0.0003` final (`-0.0008` late-11) selection nats of `MuonT`; `WT-MLP-Lag` beats `MuonT` on `11/11` second-half evaluations at both matched `0.016` (`-0.0047` late-11, `-0.0038` final) and `0.0226` (`-0.0047` late-11, `-0.0039` net of `Null-WT`).
  - **Phase 2 (`130M 1×` fresh paired seeds `1, 2, 3`):** `WT-MLP-Lag` (`0.0226`) beats Gate-1-tuned `MuonT` (`0.016`) on `3/3` fresh seeds (`-0.0034 ± 0.0014` selection, `-0.0035 ± 0.0008` C4, `-0.0040 ± 0.0027` across 16 Paloma domains, `+5.1%` median step time).
  - **Phase 3A/3B (`130M 4×` = `80 TPP`, `10.74B` tokens):** Matching `OD-All` and `WT-MLP-Lag` to `MuonT`'s exact `0.0105561` LR resolves Wave 3A's `+41%` LR confound; `WT-MLP-Lag` beats `MuonT` on `11/11` second-half evaluations (`-0.0045` late-11, `-0.0040` final, `-0.0041 ± 0.00002` across `10,240` second-half training updates) and wins `16/16` Paloma domains (`-0.0057` macro mean).
  - **Phase 4 (`520M 1×` = `20 TPP`, `799.6M` total params, `10.74B` tokens, pinned matched LR `0.0226274`):** `WT-MLP-Lag` beats `MuonT` on `11/11` second-half evaluations (`-0.0030` late-11, `-0.0028` final, `-0.0024 ± 0.00002` across `5,120` second-half updates) and wins `15/16` Paloma domains (`-0.0054` macro mean) at `+10.5%` step time (`31.6 GiB`), whereas diagonal-only `OD-All` regresses slightly (`+0.0006` selection, `+0.0005` Paloma).

---

<!-- measured-8gpu-audit:start -->
## Measured Eight-A100 Timing Audit

Measured real-data training records cover 19/19 paper methods. No stage estimates are used below. These records completed on November 20, 2026.
Each frozen optimized-runner benchmark starts from MuonT model weights at update 2,500, with fresh optimizer/statistics and data/LR offset 2,500. It executes 96 training updates on eight A100-SXM4-40GB GPUs; the first 36 are discarded and all updates 37-96 are averaged.
The model uses 32 layers, width 512, MLP width 2048, sequence length 4096, global batch 128 sequences, microbatch 4 and four accumulation passes. Eight GPUs share one node. BF16 model execution, FP32 moments, TF32 disabled, compiled model/update paths, cached-root CUDA graphs, and NCCL DDP are retained. Exact roots run for the initial 20 optimizer updates; the configured host refresh interval is four updates. Evaluation and checkpoint IO are disabled during these short timing runs.
Timing is Rank-0 host elapsed time for data loading/transfers, all four forward/backward accumulation passes, clipping, optimizer/statistics work and DDP communication, ending after main-stream synchronization. It is not the maximum elapsed time across ranks; background work may overlap the next update.

| Method | Short Mean (s) | Short Median (s) | Step SD (s) | Trajectory Mean (s) | Trajectory Runner |
| --- | ---: | ---: | ---: | ---: | --- |
| MuonT | 1.1482 | 1.1007 | 0.1867 | 1.1021 | optimized |
| Muon | 1.0799 | 1.0816 | 0.0319 | 1.0921 | historical |
| WT-MLP | 1.3167 | 1.2489 | 0.1671 | 1.3674 | historical |
| WT-MLP-Lag | 1.2709 | 1.1700 | 0.2369 | 1.1887 | optimized |
| WT-Sym | 1.2464 | 1.2033 | 0.1322 | 1.2670 | optimized |
| WT-Sym-Lag | 1.1843 | 1.1288 | 0.1669 | 1.1695 | optimized |
| OD-All | 1.1629 | 1.1204 | 0.1768 | 1.2104 | historical |
| OD-Outer | 1.1639 | 1.1166 | 0.1854 | 1.1946 | historical |
| DT-All | 1.4468 | 1.3085 | 0.2528 | 1.6398 | historical |
| DT-All-Lag | 1.4641 | 1.2493 | 0.3933 | 1.5045 | historical |
| Null-WT | 1.3205 | 1.2531 | 0.1790 | 1.3758 | historical |
| Null-Dense | 1.5195 | 1.3963 | 0.2847 | 1.6568 | historical |
| Null-OD | 1.1672 | 1.1214 | 0.2329 | 1.1988 | historical |
| Abl-OuterDense | 1.4416 | 1.3145 | 0.2552 | 1.5440 | historical |
| Abl-No2048 | 1.3146 | 1.2352 | 0.1934 | 1.3316 | historical |
| Abl-DownOnly | 1.1661 | 1.1357 | 0.1284 | 1.1508 | historical |
| Abl-DownSwap | 1.0993 | 1.0972 | 0.0361 | 1.1332 | historical |
| Abl-OutOnly | 1.5194 | 1.4115 | 0.2914 | 1.5932 | historical |
| Abl-BothDiag | 1.0948 | 1.0951 | 0.0319 | 1.2763 | historical |

Means retain all slow steps and timing outliers, whose causes are not separated in these logs. Standard deviations describe step variability; they are not confidence intervals. A single 60-update window with fresh optimizer state is insufficient for precise percent-level speed rankings or a full-budget wall-clock claim. The 36-update exclusion exceeds the configured 20-update exact-root startup, but mature statistics and long-run fallback frequencies have not been established. Historical occupancy and independent repeat measurements were not recorded in this audit.

Trajectory means use updates 2049-4096 (2048 measured updates) of completed 5120-update training trajectories at matrix LR 0.02262741696 (Muon: 0.016). The selected record is the optimized wave when available, otherwise the first eligible historical directory in lexicographic order. Other full-budget records use historical implementations and cannot be compared directly with the optimized short means. Cross-method trajectory-mean comparisons also mix runner versions. The JSON audit retains every eligible full-budget record, including rate variants, without pooling runner versions.

Regenerate the audit and report from existing records, without launching training:

```bash
uv run --no-project --python /opt/conda/envs/torch-base/bin/python \
  python tracking/audit_8gpu_timings.py \
  --out paper/results/measured_8gpu_timings.json \
  --report paper/REPORT.md
```

Raw benchmarks: `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/runs/bench_8gpu_opt`.
The [JSON audit](results/measured_8gpu_timings.json) records exact commands, source/configuration identities, metric file hashes, measurement windows, half-window means, and full-budget record paths. The current workspace source differs from this frozen optimized runner; these results do not measure later implementation changes or newly prepared related-method adapters.
<!-- measured-8gpu-audit:end -->
