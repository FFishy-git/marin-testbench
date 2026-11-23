# Learning Rate Choices Across the Confirmation & Scaling Campaign

This document explains the exact provenance, mathematical rationale, and empirical selection rules behind every matrix learning rate (`lr`) and auxiliary AdamW learning rate (`auxiliary_lr`) used in the `output-side-normalization-confirmation` campaign.

---

## 1. Direct Answer: Why `MuonT / OD-All / WT-MLP-Lag` Use `lr = 0.0226274` ($\approx 0.016\sqrt{2}$) and `auxiliary_lr = 0.0048`

There are **three distinct pieces** behind those two numbers:

### 1.1 Where `0.016` and `0.0048` come from (Marin's base `Muon` recipe)
Marin's optimizer-scaling sweeps search matrix learning rates on a **$\sqrt{2}$-spaced logarithmic grid**:
$$\dots,\ 0.008,\ 0.0113137,\ \mathbf{0.016},\ \mathbf{0.0226274},\ 0.032,\ 0.0452548,\ \dots$$
and couple the AdamW learning rate for non-matrix parameters (`embed_tokens`, `lm_head`, and `RMSNorm` gains) to the base `Muon` learning rate via the fixed ratio **`muon_to_adam_lr = 0.3`** (recorded explicitly in upstream `Toner/reproduce/marin/configs/520m/muon_0p5m.json`, line 74):

- At **`130M`, `0.5M` batch (`5,120` steps)**:
  - Base `Muon` optimal matrix LR: $\eta_{\text{Muon}} = 0.016$
  - Auxiliary AdamW LR: $\eta_{\text{aux}} = 0.3 \times 0.016 = \mathbf{0.0048}$
- At **`520M`, `0.5M` batch (`20,480` steps)**:
  - Base `Muon` optimal matrix LR: $\eta_{\text{Muon}} = 0.008$
  - Auxiliary AdamW LR: $\eta_{\text{aux}} = 0.3 \times 0.008 = \mathbf{0.0024}$

### 1.2 Why `MuonT` shifts `lr` up by $\sqrt{2}$ (`0.016 → 0.0226274`) while keeping `auxiliary_lr` unchanged (`0.0048`)
In the pinned upstream Toner repository (`Toner-dab7f3777c75e740536df822f50ffa1d929a340c/reproduce/marin/configs/`), Toner tuned `MuonT` independently on the same $\sqrt{2}$ grid and found a consistent **$\sqrt{2}\times$ upward shift in the optimal matrix LR** relative to raw `Muon` across every published model size and batch size:

| Upstream Pinned Config (`Toner/reproduce/marin/configs/`) | Model | Batch / Steps | `Muon` `lr` | `MuonT` `lr` | Ratio `MuonT / Muon` | Shared `auxiliary_lr` |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `130m/{muon,muont}_0p5m.json` | `130M` | `0.5M` / `5,120` | `0.016` | **`0.02262741696`** | $\mathbf{\sqrt{2}}$ (`1.4142×`) | **`0.0048`** |
| `130m/{muon,muont}_1m.json` | `130M` | `1.0M` / `2,560` | `0.016` | **`0.02262741696`** | $\mathbf{\sqrt{2}}$ (`1.4142×`) | `0.006788224` (`0.0048√2`) |
| `520m/{muon,muont}_0p5m.json` | `520M` | `0.5M` / `20,480` | `0.008` | **`0.01131370848`** | $\mathbf{\sqrt{2}}$ (`1.4142×`) | `0.0024` |
| `520m/{muon,muont}_1m.json` | `520M` | `1.0M` / `10,240` | `0.016` | **`0.02262741696`** | $\mathbf{\sqrt{2}}$ (`1.4142×`) | **`0.0024`** |

Why **`auxiliary_lr` is NOT multiplied by $\sqrt{2}$**:
- `MuonT` (and our output-side variants) only modify the preconditioning of the **2D transformer block matrices** (`q, k, v, o, gate, up, down`).
- Embeddings, `lm_head`, and `RMSNorm` parameters still use the exact same AdamW optimizer; their optimal step size does not shift when the 2D block preconditioner changes. Therefore `auxiliary_lr` stays pinned at `0.0048` (`130M 0.5M`) and `0.0024` (`520M 1M`) across all arms.
- (Note: in `520m/{muon,muont}_1m.json`, upstream Toner doubled the `520M` matrix LR from `0.008/0.0113` to `0.016/0.0226` when doubling batch size from `0.5M` to `1M` and halving steps from `20,480` to `10,240`, while keeping `auxiliary_lr = 0.0024` unchanged.)

### 1.3 Why `OD-All` and `WT-MLP-Lag` inherit `MuonT`'s `0.0226274` / `0.0048` as their starting anchor
1. **Norm-preserving outer factors:** In `experiment/factors.py` (`inverse_factors`, line 33) and `experiment/block_toner.py`, every outer root factor $R_{\text{out}} \in \mathbb{R}^{n\times n}$ (whether dense or diagonal) is normalized to unit RMS:
   $$R_{\text{out}} \leftarrow R_{\text{out}} \cdot \frac{\sqrt{n}}{\|R_{\text{out}}\|_F} \implies \frac{1}{n}\|R_{\text{out}}\|_F^2 = 1.$$
   Combined with the polar factor $\operatorname{polar}(\cdot)$ inside `MuonT`, adding output-side factors changes update **direction geometry** without inflating the effective step-size norm (verified empirically via the direction-norm grafting diagnostic in the paper).
2. **Isolating geometry from step size, followed by independent tuning:** Starting `OD-All` and `WT-MLP-Lag` at `MuonT`'s exact pinned `0.0226274` / `0.0048` gives a clean, zero-extra-tuning apples-to-apples geometry comparison. However, as emphasized in `output-side-normalization/paper/sections/tuning.tex`, a fixed-LR comparison alone is not sufficient to rule out LR-optimum shifts — which is exactly why **Phase 1** ran an independent full-budget 3-point LR sweep around `0.0226274`.

---

## 2. Why We Did *Not* Use Short-Horizon (320-Step) LR Screening

Before this confirmation campaign, we tested whether a cheap 6%-budget (`320`-step, `0.17B`-token) LR sweep could predict full-budget (`5,120`-step, `2.68B`-token) optimal LRs (documented in `output-side-normalization/paper/sections/tuning.tex`). It failed cleanly for two structural reasons:

1. **Severe horizon shift (`+0.1001`-nat over-stepping cliff):** On a 320-step schedule, all arms favored LRs $2.8\times$–$4\times$ higher (`0.064`–`0.0905`). Carrying `MuonT`'s own 320-step optimum (`0.064`) to the full 5,120-step horizon **degraded `MuonT` itself by `+0.1001` nats** (`3.4755` vs `3.3754` at `0.0226`).
2. **Noise-induced grid split on equivalent arms:** On the flat top of the 320-step curve, evaluation noise split two mathematically identical arms (`OD-Outer` vs `OD-Outer-F`, differing only by $1.4\times 10^{-7}$ FP32 fusion) onto `0.0905` vs `0.064`, causing a fake `+0.050`-nat full-budget regression past the stability cliff (`0.0905`) while the matched `0.064` arm still beat `MuonT` by `-0.0029` nats.

**Takeaway:** Full-budget LR bracketing around the recipe anchor (`{0.016, 0.0226274, 0.032}`) is mandatory; short-schedule proxies cannot be trusted.

---

## 3. Phase-by-Phase LR Decisions in This Campaign

### 3.1 Phase 1 (`130M`, `1×` Chinchilla = `20` TPP, `5,120` steps, `seed 0`): Full-Budget LR Check & Gate 1
In Phase 1 (`wave1a` + `wave1b`), we bracketed the pinned `130M 0.5M` LRs at **full `5,120`-step budget** on the unified `fast_lag` codebase, keeping `auxiliary_lr = 0.0048` fixed across all arms:

| Arm | `lr = 0.016` (`sel` / `c4`) | `lr = 0.0226274` (`sel` / `c4`, Pinned Toner) | `lr = 0.032` (`sel` / `c4`) | Best `sel` LR | Δ `sel` (`best - pinned`) | Gate 1 Locked `lr*` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`Muon`** | **`3.24610` / `3.46181`** (pinned ref) | `3.25236` / `3.46746` (`+0.00626`) | — | `0.016` | `0.00000` | **`0.016`** |
| **`MuonT`** | **`3.23417` / `3.45106`** (`-0.00123`) | `3.23540` / `3.45113` (pinned ref) | `3.23981` / `3.45630` (`+0.00441`) | `0.016` | **`-0.00123`** | **`0.016`** (switched) |
| **`OD-All`** | *(not swept in 1A/1B)* | **`3.23327` / `3.44894`** (pinned) | `3.24034` / `3.45629` (`+0.00706`) | `0.0226274` | `0.00000` | **`0.0226274`** (kept) |
| **`WT-MLP-Lag`** | **`3.23033` / `3.44720`** (`-0.00078`) | `3.23111` / `3.44905` (pinned ref) | `3.23558` / `3.45326` (`+0.00447`) | `0.016` | `-0.00078` | **`0.0226274`** (kept) |

*(Source: `results/gate1.json` and `results/summary.json`. Also passed `Null-WT` falsification gate at `0.0226274`: `sel = 3.23510`, `c4 = 3.45057` vs `MuonT` `sel = 3.23540`, `c4 = 3.45113`.)*

#### Why Phase 1 `MuonT` edged lower at `0.016` instead of Toner's pinned `0.0226274` (`0.016√2`)
This is a real and instructive nuance, explained by three facts in the table above:
1. **Flat valley between `0.016` and `0.02263` on C4 (Toner's own tuning metric):** Upstream Toner's `reference_validation_loss` (`3.45058`) is evaluated on `paloma/c4_en`. On `paloma/c4_en`, our `MuonT` at `0.016` (`3.45106`) and `0.0226274` (`3.45113`) are **virtually tied (`Δ = -0.00006` nats)**. The true continuous optimum for `MuonT` sits in the flat bottom *between* `0.016` and `0.02263` (roughly `~0.018–0.020`), so on a coarse $\sqrt{2}$ grid (`41%` spacing), slight differences in evaluation metric, data shard order, or seed can flip the discrete argmax between `0.016` and `0.02263`.
2. **In-distribution mixture metric (`selection_loss`) slightly favors lower LR:** Our Gate 1 selected on `selection_loss` (DCLM + StarCoder + Proof-Pile-2). On `seed 0`, `0.016` improved StarCoder (`-0.00146`) and Proof-Pile-2 (`-0.00271`) slightly more than C4 (`-0.00006`), producing the `-0.00123` `selection_loss` edge that crossed Gate 1's `0.0010` threshold.
3. **Toner's rightward shift relative to `Muon` is still clearly real:** Pushing raw **`Muon`** up to `0.0226274` immediately hurts it by **`+0.00626` nats** (`sel`) and **`+0.00565` nats** (`c4`), whereas `MuonT`, `OD-All`, and `WT-MLP-Lag` are all flat/near-optimal at `0.0226274` and only degrade at `0.032` (`+0.0044` to `+0.0071`). Thus Toner whitening genuinely shifts the optimal LR basin rightward relative to `Muon` — just slightly less than a full `1.414×` step on our `130M` mixture.
4. **`WT-MLP-Lag` wins at *both* matched LRs:** At `130M 1×` (`seed 0`), `WT-MLP-Lag` beats `MuonT` by **`-0.00384` nats** (`c4`: `-0.00387`) at matched `0.016` *and* by **`-0.00429` nats** (`c4`: `-0.00207`) at matched `0.0226274`.

**Exact Gate 1 Rule (pre-registered in `PLAN.md` §3):**
> Default to Marin's pinned LR (`0.016` for `Muon`; `0.02262741696` for `MuonT` / `OD-All` / `WT-MLP-Lag`) **unless** another grid point improves `selection_loss` by **`> 0.0010` nats** cleanly without instability.

**What happened at Gate 1:**
- `0.032` clearly degraded every arm (`+0.0044` to `+0.0071` nats), ruling out higher LRs.
- `0.016` improved both `MuonT` (`-0.00123` nats) and `WT-MLP-Lag` (`-0.00078` nats) slightly over `0.0226274`.
- Because `-0.00123` crossed the pre-registered `0.0010` threshold while `-0.00078` fell just shy of it, Gate 1 automatically locked:
  - `Muon` = **`0.016`**
  - `MuonT` = **`0.016`**
  - `OD-All` = **`0.02262741696`**
  - `WT-MLP-Lag` = **`0.02262741696`**

---

### 3.2 Phase 2 (`130M`, `1×` Chinchilla, Fresh Paired `seeds 1, 2, 3`): Conservative Against Our Method
Phase 2 (`wave2a`, `wave2b`, `wave2c`) ran all 4 arms on 3 fresh post-selection seeds using Gate 1's locked LRs (`auxiliary_lr = 0.0048` for all):

- Notice that Gate 1's asymmetry was **strictly conservative against `WT-MLP-Lag`**: `MuonT` received its tuned-down `0.016` LR, whereas `WT-MLP-Lag` stayed at the slightly suboptimal `0.0226274` (even though `WT-MLP-Lag` had also been `0.00078` nats better at `0.016` on `seed 0`).
- **Result (3/3 seeds won by `WT-MLP-Lag`):**
  - `Muon (0.016)`: mean `3.20354 ± 0.02888`
  - `MuonT (0.016)`: mean `3.19163 ± 0.03024`
  - `OD-All (0.02263)`: mean `3.19108 ± 0.03001` (`-0.00054` vs tuned `MuonT`)
  - **`WT-MLP-Lag (0.02263)`**: mean **`3.18826 ± 0.02882`** (**`-0.00337` vs tuned `MuonT`**, **`-0.01528` vs `Muon`**; beats `MuonT` on all 3/3 seeds and beats `MuonT` by `-0.00384` even at matched `0.016` on `seed 0`).

---

### 3.3 Phase 3 (`130M`, `4×` Chinchilla = `80` TPP, `20,480` steps, `10.74B` tokens): Token-Horizon Scaling & Wave 3B Matched-LR Follow-Up

#### Marin's $T^{-0.3}$ token-horizon LR scaling
When scaling token horizon by $4\times$ at fixed model size (`130M`) and fixed batch size (`0.5M`, scaling steps from `5,120` to `20,480` while preserving the 20% stable / 80% linear-decay schedule), we apply Marin's token-horizon LR power law:
$$\eta(4T_0) = \eta^*(T_0) \cdot 4^{-0.3} \approx 0.6597539554 \cdot \eta^*(T_0)$$
applied identically to both `lr` and `auxiliary_lr` (`0.0048 × 4^{-0.3} = 0.0031668190`) so the `auxiliary_lr / lr` ratio is preserved.

#### Why Wave 3A showed an LR confound — and why we added Wave 3B
Because `generate_phase3()` scaled Gate 1's locked LRs directly by $4^{-0.3}$, **Wave 3A** inherited Gate 1's `0.016` vs `0.0226274` split as a **41% ($\sqrt{2}\times$) LR mismatch** in the overtrained `80`-TPP regime:

| Wave | Arm | `4×` Matrix LR | `4×` Aux LR | `selection_loss` | `c4_loss` | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **Wave 3A** | `Muon` | `0.0105561` (`0.016 × 4^-0.3`) | `0.0031668` | `3.07783` | `3.29887` | Lower LR |
| **Wave 3A** | `MuonT` | **`0.0105561`** (`0.016 × 4^-0.3`) | `0.0031668` | **`3.06754`** | **`3.28861`** | Lower LR |
| **Wave 3A** | `OD-All` | `0.0149285` (`0.02263 × 4^-0.3`) | `0.0031668` | `3.07089` | `3.29293` | **41% higher LR** (`+0.00335` vs MuonT) |
| **Wave 3A** | `WT-MLP-Lag` | `0.0149285` (`0.02263 × 4^-0.3`) | `0.0031668` | `3.06955` | `3.28997` | **41% higher LR** (`+0.00201` vs MuonT) |
| **Wave 3B** *(queued)* | `OD-All` | **`0.0105561`** *(matched to MuonT)* | `0.0031668` | *(pending)* | *(pending)* | Clean matched-LR `4×` test |
| **Wave 3B** *(queued)* | `WT-MLP-Lag` | **`0.0105561`** *(matched to MuonT)* | `0.0031668` | *(pending)* | *(pending)* | Clean matched-LR `4×` test |

At `4×` Chinchilla (`20,480` steps), optimal matrix LR shifts **downward**, so running `OD-All` and `WT-MLP-Lag` 41% hotter than `MuonT` penalized them — consistent with Phase 1 already showing `WT-MLP-Lag` preferred `0.016` over `0.02263`. **Wave 3B** (`campaign/run_wave3b_matched_lr.py`) resolves this confound cleanly by running `OD-All` and `WT-MLP-Lag` at the exact matched `0.010556063286183155` (`auxiliary_lr = 0.003166818985854946`) immediately after `520M` completes.

---

### 3.4 Phase 4 (`520M`, `1×` Chinchilla = `20` TPP, `1M` batch, `10,240` steps, `10.74B` tokens): Zero-Retuning Official Pinned Recipe
For `520M`, running a multi-point LR sweep per arm would cost ~100+ node-hours. Instead, we use **Toner's exact official pinned PyTorch `520m` `1M`-batch configs** (`Toner/reproduce/marin/configs/520m/{muon,muont}_1m.json`) with **zero per-arm retuning** and **strictly matched LR across all three Toner-family arms**:

- `Muon`: `lr = 0.016`, `auxiliary_lr = 0.0024` (exact `520m/muon_1m.json`)
- `MuonT`: `lr = 0.02262741696`, `auxiliary_lr = 0.0024` (exact `520m/muont_1m.json`)
- `OD-All`: `lr = 0.02262741696`, `auxiliary_lr = 0.0024` (exact match to `MuonT`)
- `WT-MLP-Lag`: `lr = 0.02262741696`, `auxiliary_lr = 0.0024` (exact match to `MuonT`)

Because `MuonT`, `OD-All`, and `WT-MLP-Lag` share the exact same `lr = 0.02262741696`, `auxiliary_lr = 0.0024`, `weight_decay = 0.1`, `betas = (0.8, 0.9604)`, `clip_norm = 2.0`, and `10,240`-step linear decay schedule (`stable_steps = 0`), there is **zero LR mismatch** at `520M`.

---

## 4. Master Summary Table of All Learning Rates in This Campaign

| Phase / Wave | Regime | Steps / Batch | Arm | Matrix `lr` | `auxiliary_lr` | Basis / Rationale |
| --- | --- | --- | --- | ---: | ---: | --- |
| **Phase 1** (`1A, 1B`) | `130M 1×` (`20` TPP, `seed 0`) | `5,120` / `0.5M` | `Muon` | `0.016`, `0.0226274` | `0.0048` | Pinned `0.016` + $\sqrt{2}$ check |
| **Phase 1** (`1A, 1B`) | `130M 1×` (`20` TPP, `seed 0`) | `5,120` / `0.5M` | `MuonT`, `WT-MLP-Lag` | `0.016`, `0.0226274`, `0.032` | `0.0048` | Full-budget 3-pt bracket around pinned `0.02263` |
| **Phase 1** (`1B`) | `130M 1×` (`20` TPP, `seed 0`) | `5,120` / `0.5M` | `OD-All` | `0.0226274`, `0.032` | `0.0048` | Pinned `0.02263` + upward check |
| **Phase 1** (`1B`) | `130M 1×` (`20` TPP, `seed 0`) | `5,120` / `0.5M` | `Null-WT` | `0.0226274` | `0.0048` | Same-wave falsification control vs `MuonT` ref |
| **Phase 2** (`2A–2C`) | `130M 1×` (`20` TPP, `seeds 1–3`) | `5,120` / `0.5M` | `Muon`, `MuonT` | `0.016` | `0.0048` | Gate 1 locked (`MuonT` crossed `0.0010` threshold) |
| **Phase 2** (`2A–2C`) | `130M 1×` (`20` TPP, `seeds 1–3`) | `5,120` / `0.5M` | `OD-All`, `WT-MLP-Lag` | `0.0226274` | `0.0048` | Gate 1 locked (pinned anchor retained) |
| **Phase 3A** (`3A`) | `130M 4×` (`80` TPP, `seed 0`) | `20,480` / `0.5M` | `Muon`, `MuonT` | `0.0105561` | `0.0031668` | Gate 1 `0.016 × 4^-0.3`; aux `0.0048 × 4^-0.3` |
| **Phase 3A** (`3A`) | `130M 4×` (`80` TPP, `seed 0`) | `20,480` / `0.5M` | `OD-All`, `WT-MLP-Lag` | `0.0149285` | `0.0031668` | Gate 1 `0.02263 × 4^-0.3` (41% hotter than MuonT) |
| **Phase 3B** (`3B`) | `130M 4×` (`80` TPP, `seed 0`) | `20,480` / `0.5M` | `OD-All`, `WT-MLP-Lag` | **`0.0105561`** | `0.0031668` | **Matched-LR follow-up** (`= MuonT` in Wave 3A) |
| **Phase 4** (`4A`) | `520M 1×` (`20` TPP, `seed 0`) | `10,240` / `1.0M` | `Muon` | `0.016` | `0.0024` | Exact pinned `520m/muon_1m.json` |
| **Phase 4** (`4A`) | `520M 1×` (`20` TPP, `seed 0`) | `10,240` / `1.0M` | `MuonT`, `OD-All`, `WT-MLP-Lag` | **`0.0226274`** | `0.0024` | Exact pinned `520m/muont_1m.json` (matched across arms) |

---

## 5. Note on Predecessor Campaign (`related-selected-full-seed12-20261121`)

For readers comparing logs across folders: the separate predecessor campaign `related-selected-full-seed12-20261121` in `../output-side-normalization/` tested an **isotropic weight-decay** variant (`decay_mode=isotropic`, without Toner's shaped $W R_{\text{out}}^2$ decay) and selected `lr = 0.032` for that variant. All runs in **this** confirmation & scaling campaign use Toner's standard **shaped weight decay** recipe and the exact pinned Marin/Toner LRs documented above.

---

## 6. Reference Files

- Upstream pinned configs:
  - `artifacts/external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/reproduce/marin/configs/130m/muon_0p5m.json`
  - `artifacts/external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/reproduce/marin/configs/130m/muont_0p5m.json`
  - `artifacts/external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/reproduce/marin/configs/520m/muon_0p5m.json` (contains `"muon_to_adam_lr": 0.3`)
  - `artifacts/external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/reproduce/marin/configs/520m/muon_1m.json`
  - `artifacts/external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/reproduce/marin/configs/520m/muont_1m.json`
- Campaign config & gate files:
  - `PLAN.md`
  - `campaign/configure_campaign.py`
  - `campaign/run_wave3b_matched_lr.py`
  - `results/gate1.json`
  - `results/RESULTS.md`
- Norm-preserving outer-root normalization:
  - `experiment/factors.py` (`inverse_factors`, line 33)
- Historical short-horizon vs full-budget tuning analysis:
  - `../output-side-normalization/paper/sections/tuning.tex`
