# Output-Side Normalization Confirmation & Scaling Campaign Plan

- **Workspace Folder:** `/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization-confirmation`
- **PVC Campaign Output Root:** `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/campaigns/confirmation-20261121`
- **Predecessor Campaign (Wait Dependency):** `related-selected-full-seed12-20261121` (12 runs on `28.216.17.{9,80,144,139}`; do **not** modify `/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization/` while it runs).
- **Date Planned:** 2026-11-21

---

## 1. Objectives & Gap Closure

This campaign closes the remaining experimental gaps identified in the current `output-side-normalization` paper draft before final publication:

1. **Full-budget LR verification & same-wave `Null-WT` control (Phase 1):**
   - Verify full-budget (`2.68B` tokens) LR optimality around `0.0226` (`{0.016, 0.0226, 0.032}`) for `Muon`, `MuonT`, `OD-All`, and `WT-MLP-Lag` on the unified optimized `fast_lag` runner.
   - Run same-wave `fast_lag` `Null-WT` (`null_mlpdiag`) to close the legacy `+0.0005` cross-wave discrepancy.
2. **Matched $\ge 3$-seed replication across all 4 core arms (Phase 2):**
   - Previous main-recipe inventory had only **1 seed** (`seed 0`) for `Muon` and `OD-All`, and **1 seed** on the optimized `fast_lag` runner for `WT-MLP-Lag` (legacy 3-seed `WT-MLP` included selection `seed 0` and ran on the `+22%` unoptimized runner).
   - Phase 2 runs **3 fresh, matched, post-selection seeds (`seeds 1, 2, 3`)** for all four core arms (`Muon`, `MuonT`, `OD-All`, `WT-MLP-Lag`) on the identical `fast_lag` codebase and locked per-arm `lr*`, giving **4 total seeds (`0, 1, 2, 3`)** and **3 strictly post-selection paired seeds (`1, 2, 3`)** for every arm.
3. **Short vs. Large Tokens-per-Parameter (TPP) sweep (Phase 3):**
   - Compare our anchor **`1×` Chinchilla (`20.0` TPP non-emb = `2.68B` tokens)** against a large-TPP **`4×` Chinchilla (`80.0` TPP non-emb = `10.74B` tokens)** overtraining regime on `130M` for all 4 core arms.
   - Fits comfortably inside the **`25.39B` tokenized training tokens** already prepared and verified on PVC (`experiment_receipt.json` SHA256 `e6ff883cb816c34cc502061dbc6a8239fc19dfc803588786758bb1020e1afe10`).
4. **Model-size scaling to `520M` (NO `300M` to save compute) (Phase 4):**
   - Scale directly from `130M` (`d=512, mlp=2048, L=32`, `134.3M` non-emb / `265.6M` total params) to **`520M`** (`d=1024, mlp=4096, L=32, H=16`, `536.9M` non-emb / `799.6M` total params) at `1×` Chinchilla (`10.74B` tokens = `20.0` TPP non-emb) across all 4 core arms using Toner's pinned official PyTorch `520m` `1M`-batch recipe.
   - **`300M` is intentionally omitted** to conserve compute while still testing a `4×` non-embedding parameter jump (`2×` width jump).
5. **Multi-domain Paloma evaluation (Phase 5):**
   - Evaluate final checkpoints from Phases 2, 3, and 4 across all **16 Paloma subsets** already tokenized at `$PVC/data/paloma/*`.

---

## 2. Shared Experimental Protocol

| Item | Specification |
| --- | --- |
| **Cluster** | 4 nodes × 8 NVIDIA A100-SXM4-40GB (`28.216.17.9`, `28.216.17.80`, `28.216.17.144`, `28.216.17.139`), 1 independent 8-rank run per node |
| **Python & Runtime** | `/opt/conda/envs/torch-base/bin/python` (PyTorch 2.10.0, CUDA 12.9, FA2, deterministic algorithms, `allow_tf32=False`) |
| **Mandatory Env Vars** | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, `NCCL_INSPECTOR_ENABLE=0`, `PYTHONFAULTHANDLER=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` |
| **Codebase** | Frozen optimized `fast_lag` runner cloned from `$PVC/runs/bench_8gpu_opt/_code/experiment/` into this folder's `experiment/` (zero edits to `../output-side-normalization/`) |
| **Pinned Upstream Toner** | `Toner-dab7f3777c75e740536df822f50ffa1d929a340c` (read-only) |
| **Prepared Dataset** | `$PVC/data` (`25.39B` train tokens: DCLM `3.8`, StarCoder `0.25`, Proof-Pile-2 `0.055`, receipt SHA256 `e6ff883cb816c34cc502061dbc6a8239fc19dfc803588786758bb1020e1afe10`) |
| **Core 4 Arms** | `Muon` (`muon`), `MuonT` (`muont`), `OD-All` (`inner_outer_leftdiag`), `WT-MLP-Lag` (`inner_outer_mlpdiag_lag`) |
| **Falsification Control** | `Null-WT` (`null_mlpdiag`, Phase 1 same-wave control) |
| **Primary Metric** | Held-out Marin mixture `selection_loss` (64 windows per source, mixture-weighted); `paloma/c4_en` logged during training; all 16 Paloma subsets evaluated post-hoc |
| **GPU Keeper Policy** | Every node-level GPU command runs through `campaign/allocation.py`, which acquires the node lock, releases the active 8-rank GPU keeper via `pidfd`, executes training/validation, and **always restores** `/root/dl/runtime/script/hy_infuse_moe/gpu_resource_manager.sh start` on completion or error |

---

## 3. Detailed Phase-by-Phase Run Matrix (28 Runs, 7 Waves)

### Phase 1 — `130M`, `1×` Chinchilla (`20` TPP, `2.68B` tok): Full-Budget LR Confirmation & Same-Wave `Null-WT`
- **Model:** `130m` (`hidden=512, mlp=2048, heads=8, layers=32, seq=4096, vocab=128256, params=265,585,152`)
- **Budget:** `0.5M` batch (`global_batch_sequences=128`, `micro_batch_size=8`), `5,120` steps (`2,684,354,560` tokens), schedule `1,024` stable + `4,096` linear decay (`min_lr_ratio=0.0`), `weight_decay=0.1`, `head_weight_decay=0.1`, `momentum=0.95`, `betas=(0.8, 0.98)`, `eps=1e-15`, `clip_norm=1.0`, `eval_every=250`, `checkpoint_every=5120` (save final + mid-check `2560`).
- **Existing `seed 0` baselines reused for the 3-point LR grid (`{0.016, 0.0226, 0.032}`):**
  - `Muon` at `lr=0.016` (`wave0/muon`, `sel=3.29818`)
  - `MuonT` at `lr=0.02262741696` (`wave_sym_fastlag/muont`, `sel=3.26835`)
  - `WT-MLP-Lag` at `lr=0.02262741696` (`wave_sym_fastlag/inner_outer_mlpdiag_lag`, `sel=3.26105`)

| Wave | Node | Run Name | Arm | Seed | Matrix LR | Aux LR | Est. Wall Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1A** | `28.216.17.9` | `muont_130m_1x_seed0_lr0p016` | `muont` | 0 | `0.016` | `0.0048` | ~1.7 h |
| **1A** | `28.216.17.80` | `muont_130m_1x_seed0_lr0p032` | `muont` | 0 | `0.032` | `0.0048` | ~1.7 h |
| **1A** | `28.216.17.144` | `wt_mlp_lag_130m_1x_seed0_lr0p016` | `inner_outer_mlpdiag_lag` | 0 | `0.016` | `0.0048` | ~1.8 h |
| **1A** | `28.216.17.139` | `wt_mlp_lag_130m_1x_seed0_lr0p032` | `inner_outer_mlpdiag_lag` | 0 | `0.032` | `0.0048` | ~1.8 h |
| **1B** | `28.216.17.9` | `od_all_130m_1x_seed0_lr0p0226` | `inner_outer_leftdiag` | 0 | `0.02262741696` | `0.0048` | ~1.8 h |
| **1B** | `28.216.17.80` | `od_all_130m_1x_seed0_lr0p032` | `inner_outer_leftdiag` | 0 | `0.032` | `0.0048` | ~1.8 h |
| **1B** | `28.216.17.144` | `muon_130m_1x_seed0_lr0p0226` | `muon` | 0 | `0.02262741696` | `0.0048` | ~1.5 h |
| **1B** | `28.216.17.139` | `null_wt_130m_1x_seed0_lr0p0226` | `null_mlpdiag` | 0 | `0.02262741696` | `0.0048` | ~1.8 h |

- **Gate G1 (after Wave 1B):**
  - Verify `|selection_loss(Null-WT) - selection_loss(MuonT)| <= 0.0005`.
  - Lock per-arm optimal full-budget LR `lr*` from `{0.016, 0.02262741696, 0.032}` for each of `Muon`, `MuonT`, `OD-All`, `WT-MLP-Lag` (defaulting to Marin pinned `0.016` for `Muon` and `0.02262741696` for `MuonT`/`OD-All`/`WT-MLP-Lag` unless another grid point improves `selection_loss` by `> 0.0010` cleanly without instability).

---

### Phase 2 — `130M`, `1×` Chinchilla (`20` TPP, `2.68B` tok): Matched 3-Seed Replication (`seeds 1, 2, 3`)
- **Model & Budget:** Identical to Phase 1 (`130m`, `0.5M` batch, `5,120` steps, `2.68B` tokens), using each arm's locked `lr*` from Gate G1.
- **Resulting Seed Inventory:** `3` strictly post-selection matched seeds (`seeds 1, 2, 3`) on the unified `fast_lag` runner for all 4 core arms (and `4` total seeds including `seed 0`).

| Wave | Node | Run Name | Arm | Seed | Matrix LR | Est. Wall Time |
| --- | --- | --- | --- | --- | --- | --- |
| **2A** | `28.216.17.9` | `muon_130m_1x_seed1` | `muon` | 1 | `lr*(muon)` | ~1.5 h |
| **2A** | `28.216.17.80` | `muont_130m_1x_seed1` | `muont` | 1 | `lr*(muont)` | ~1.7 h |
| **2A** | `28.216.17.144` | `od_all_130m_1x_seed1` | `inner_outer_leftdiag` | 1 | `lr*(od_all)` | ~1.8 h |
| **2A** | `28.216.17.139` | `wt_mlp_lag_130m_1x_seed1` | `inner_outer_mlpdiag_lag` | 1 | `lr*(wt_mlp_lag)` | ~1.8 h |
| **2B** | `28.216.17.9` | `muon_130m_1x_seed2` | `muon` | 2 | `lr*(muon)` | ~1.5 h |
| **2B** | `28.216.17.80` | `muont_130m_1x_seed2` | `muont` | 2 | `lr*(muont)` | ~1.7 h |
| **2B** | `28.216.17.144` | `od_all_130m_1x_seed2` | `inner_outer_leftdiag` | 2 | `lr*(od_all)` | ~1.8 h |
| **2B** | `28.216.17.139` | `wt_mlp_lag_130m_1x_seed2` | `inner_outer_mlpdiag_lag` | 2 | `lr*(wt_mlp_lag)` | ~1.8 h |
| **2C** | `28.216.17.9` | `muon_130m_1x_seed3` | `muon` | 3 | `lr*(muon)` | ~1.5 h |
| **2C** | `28.216.17.80` | `muont_130m_1x_seed3` | `muont` | 3 | `lr*(muont)` | ~1.7 h |
| **2C** | `28.216.17.144` | `od_all_130m_1x_seed3` | `inner_outer_leftdiag` | 3 | `lr*(od_all)` | ~1.8 h |
| **2C** | `28.216.17.139` | `wt_mlp_lag_130m_1x_seed3` | `inner_outer_mlpdiag_lag` | 3 | `lr*(wt_mlp_lag)` | ~1.8 h |

- **Gate G2 (after Wave 2C):**
  - Report per-seed paired deltas `WT-MLP-Lag - MuonT`, `OD-All - MuonT`, and `WT-MLP-Lag - Muon` across `seeds {1, 2, 3}` (mean ± std, min/max) and unified `fast_lag` step-time overhead.

---

### Phase 3 — `130M`, Large TPP = `4×` Chinchilla (`80` TPP non-emb, `10.74B` tok): Overtraining Horizon Test
- **Model:** `130m` (`hidden=512, mlp=2048, heads=8, layers=32, seq=4096, vocab=128256, params=265,585,152`)
- **Budget:** `0.5M` batch (`global_batch_sequences=128`, `micro_batch_size=8`), **`20,480` steps (`10,737,418,240` tokens = `4×` Chinchilla = `80.0` TPP non-emb)**.
- **Schedule:** Exact same 20% stable / 80% linear decay shape as `1×` Chinchilla (`stable_steps=4096`, `decay_steps=16384`, `min_lr_ratio=0.0`).
- **LR Scaling:** Marin token-horizon power law $\eta(4T_0) = \eta^*(T_0) \cdot 4^{-0.3} \approx 0.659754 \cdot \eta^*(T_0)$ (applied identically to both `lr` and `auxiliary_lr` so `aux/matrix` ratio is preserved across all 4 arms).
- **Eval / Checkpoint cadence:** `eval_every=1000`, `checkpoint_every=5120` (saves `5120`, `10240`, `15360`, `20480`).

| Wave | Node | Run Name | Arm | Seed | Matrix LR | Aux LR | Est. Wall Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **3A** | `28.216.17.9` | `muon_130m_4x_seed0` | `muon` | 0 | `lr*(muon) * 4^-0.3` | `0.0048 * 4^-0.3` | ~5.8 h |
| **3A** | `28.216.17.80` | `muont_130m_4x_seed0` | `muont` | 0 | `lr*(muont) * 4^-0.3` | `0.0048 * 4^-0.3` | ~6.5 h |
| **3A** | `28.216.17.144` | `od_all_130m_4x_seed0` | `inner_outer_leftdiag` | 0 | `lr*(od_all) * 4^-0.3` | `0.0048 * 4^-0.3` | ~6.8 h |
| **3A** | `28.216.17.139` | `wt_mlp_lag_130m_4x_seed0` | `inner_outer_mlpdiag_lag` | 0 | `lr*(wt_mlp_lag) * 4^-0.3` | `0.0048 * 4^-0.3` | ~6.9 h |

- **Gate G3 (after Wave 3A):**
  - Compare `1×` (`20` TPP) vs `4×` (`80` TPP) final `selection_loss`, C4 loss, and trajectory crossovers (`step 5120`, `10240`, `15360`, `20480`).
- **Wave 3B (Matched-LR `0.0105561` `4×` Chinchilla Follow-Up — queued immediately after Phase 4 `520M` completes):**
  - Because Gate 1 locked `MuonT` to `0.016` (`4×`-scaled `0.0105561`) while leaving `OD-All` and `WT-MLP-Lag` at `0.0226` (`4×`-scaled `0.0149285`), Wave 3B runs `od_all_130m_4x_seed0_lr0p01056` (`28.216.17.144`) and `wt_mlp_lag_130m_4x_seed0_lr0p01056` (`28.216.17.139`) at the exact matched LR `0.010556063286183155` and `auxiliary_lr = 0.003166818985854946` (~6.5 h, plus 16-domain Paloma evaluation via `campaign/run_wave3b_matched_lr.py`).

---

### Phase 4 — `520M` Model Scaling (NO `300M`, `1×` Chinchilla = `20` TPP non-emb, `10.74B` tok)
- **Model:** `520m` (`hidden=1024, mlp=4096, heads=16, layers=32, seq=4096, vocab=128256, non_emb=536,936,448, expected_model_parameters=799,605,760`).
- **Base Recipe:** Pinned official PyTorch `Toner/reproduce/marin/configs/520m/muon_1m.json` (ref val `3.07648`) and `520m/muont_1m.json` (ref val `3.07071`).
- **Budget:** `1M` batch (`global_batch_sequences=256`), **`10,240` steps (`10,737,418,240` tokens = `1×` Chinchilla = `20.0` TPP non-emb)**. Using the official PyTorch `1M` recipe halves optimizer-step frequency (`10,240` vs `20,480`) relative to `0.5M` while retaining exact PyTorch reference baselines for both `Muon` and `MuonT`.
- **Hyperparameters (exact pinned `520m/1m`):**
  - `Muon`: `lr=0.016`, `auxiliary_lr=0.0024`
  - `MuonT` / `OD-All` / `WT-MLP-Lag`: `lr=0.02262741696`, `auxiliary_lr=0.0024` (if Gate G1 reveals a different `lr*(arm)/lr*(muont)` ratio, note and keep `MuonT` at pinned `0.02262741696` and apply the same ratio or pinned value consistently; default is pinned `0.02262741696` matching Toner's `520m/muont_1m.json`).
  - Shared: `weight_decay=0.1`, `head_weight_decay=0.1`, `betas=(0.8, 0.9604)`, `eps=1e-25`, `clip_norm=2.0`, `momentum=0.95`, `stable_steps=0`, `decay_steps=10240`, `kind="linear"`, `min_lr_ratio=0.0`.
  - A100-40GB adaptation: `attention="FA2"`, `deterministic=True`, `root.chunk=0`, `eval_every=500`, `checkpoint_every=2560` (saves `2560`, `5120`, `7680`, `10240`).
- **Pre-Launch Gate G4-pre (`520M` GPU Smoke + Memory + Continuation Check):**
  - Before launching Wave 4A, run a 4-step GPU memory & throughput probe on 1 node (`28.216.17.9`) to pick the largest `micro_batch_size` dividing `32` (`4`, `2`, or `1`) that stays cleanly `< 36 GiB` peak memory across all 4 arms (especially `WT-MLP-Lag`), and pass a 4-step exact checkpoint-continuation check (`max |Δloss| == 0.0` or `< 1e-6`).

| Wave | Node | Run Name | Arm | Seed | Matrix LR | Aux LR | Est. Wall Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **4A** | `28.216.17.9` | `muon_520m_1x_seed0` | `muon` | 0 | `0.016` | `0.0024` | ~18–22 h |
| **4A** | `28.216.17.80` | `muont_520m_1x_seed0` | `muont` | 0 | `0.02262741696` | `0.0024` | ~20–24 h |
| **4A** | `28.216.17.144` | `od_all_520m_1x_seed0` | `inner_outer_leftdiag` | 0 | `0.02262741696` | `0.0024` | ~21–25 h |
| **4A** | `28.216.17.139` | `wt_mlp_lag_520m_1x_seed0` | `inner_outer_mlpdiag_lag` | 0 | `0.02262741696` | `0.0024` | ~22–26 h |

---

### Phase 5 — Post-Hoc 16-Domain Paloma Evaluation & Paper Artifact Generation
- Run `campaign/eval_paloma.py` across all 16 Paloma subsets (`$PVC/data/paloma/*`) on final checkpoints from:
  - Phase 2 (`130M`, `1×`, `seeds 1, 2, 3`, 12 checkpoints)
  - Phase 3 (`130M`, `4×`, `seed 0`, 4 checkpoints)
  - Phase 4 (`520M`, `1×`, `seed 0`, 4 checkpoints)
- Write consolidated machine-readable JSON (`results/summary.json`) and human-readable `RESULTS.md` in this folder.

---

## 4. Wall-Clock & Compute Budget Summary

| Phase | Regime | Runs | Waves (4 nodes parallel) | Est. Wall Time |
| --- | --- | --- | --- | --- |
| **Pre-flight** | Wait for `related-selected-full-seed12-20261121` + deploy + `130M` smoke | — | — | ~2.5–3.0 h (wait) + 0.2 h |
| **Phase 1** | `130M`, `1×` (20 TPP) LR grid + `Null-WT` | 8 | 2 | ~3.6 h |
| **Phase 2** | `130M`, `1×` (20 TPP) matched `seeds 1, 2, 3` | 12 | 3 | ~5.4 h |
| **Phase 3** | `130M`, `4×` (80 TPP, `10.74B` tok) `seed 0` | 4 | 1 | ~6.9 h |
| **Phase 4** | `520M`, `1×` (20 TPP, `10.74B` tok) `seed 0` (NO `300M`) | 4 | 1 | ~20–26 h |
| **Phase 5** | 16-domain Paloma final checkpoint eval + report | — | — | ~0.5 h |
| **Total (Active Training)** | — | **28 runs** | **7 waves** | **~36–43 h** |

---

## 5. Execution & Handoff Protocol

1. **Wait Stage (Subagent 1 — lightweight model):**
   - Execute `bash campaign/wait_related.sh` until `related-selected-full-seed12-20261121` reports `completed_trials == 12`, `status == "complete"`, and all 4 cluster nodes have zero active `experiment.train` / `torch.distributed.run` processes.
2. **Launch & Execution Stage (Subagent 2 — campaign runner):**
   - Deploy `output-side-normalization-confirmation/` to all 4 nodes (isolated directory; never touches `../output-side-normalization/`).
   - Run `130M` preflight & validation gate, then execute **Phase 1 → Gate G1 → Phase 2 → Gate G2 → Phase 3 → Gate G3 → `520M` Gate G4-pre → Phase 4 → Phase 5**.
   - Restore GPU keepers on all 4 nodes after every wave and on completion.
