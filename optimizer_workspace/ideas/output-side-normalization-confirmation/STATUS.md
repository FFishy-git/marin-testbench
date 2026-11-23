# Confirmation & Scaling Campaign — Final Status

**Last Updated:** `2026-11-23 18:00 +08:00` (`2026-11-23T10:00Z`)  
**Status:** **COMPLETE** — **`30 / 30` total runs complete**, **`22 / 22` Paloma 16-domain evaluations complete**, **`30 / 30` runs uploaded & verified on SwanLab**, GPU keepers (`gpu_matmul`) verified healthy on all 4 nodes, and enhanced multi-checkpoint late-stage matched-LR analysis added to `results/RESULTS.md`.

---

## 1. Key Paths, Endpoints & Verification

| Resource | Path / Value | Status |
| --- | --- | --- |
| **Isolated Workspace** | `/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization-confirmation` | Complete (predecessor `../output-side-normalization/` untouched) |
| **PVC Campaign Output Root** | `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/campaigns/confirmation-20261121` | `30 / 30` runs complete |
| **Paloma 16-Domain Evals** | `$CAMPAIGN_ROOT/paloma_eval/*.json` | `22 / 22` JSONs verified |
| **SwanLab Project (Private)** | `https://swanlab.cn/@siyuc3141/output-side-normalization-confirmation` | `30 / 30` verified (`swanlab_receipt.json`) |
| **Cluster GPU Keepers** | `28.216.17.{9,80,144,139}` | `0` training/alloc procs; `gpu_matmul` healthy (8/8 GPUs @ 100%) on all 4 nodes |
| **Final Report & Data** | `results/RESULTS.md`, `results/summary.json`, `results/gate1.json`, `LR_CHOICES.md`, `PLAN.md` | Up to date |

---

## 2. Campaign Checklist (`30 / 30` Runs Complete)

| Wave | Phase / Regime | Runs | Status | SwanLab |
| --- | --- | ---: | --- | --- |
| `smoke-130m` | 130M 4-step 5-arm smoke (`mb=4`) | — | **DONE** | N/A |
| `wave1a` | Phase 1: `130M 1×` LR grid (`seed 0`, `5,120` steps) | 4/4 | **DONE** | **4/4 verified** |
| `wave1b` | Phase 1: `130M 1×` LR grid + `Null-WT` (`seed 0`, `5,120` steps) | 4/4 | **DONE** (`Null-WT` gate **PASS**, `gate1.json` written) | **4/4 verified** |
| `wave2a` | Phase 2: `130M 1×` fresh `seed 1` (`5,120` steps) | 4/4 | **DONE** | **4/4 verified** |
| `wave2b` | Phase 2: `130M 1×` fresh `seed 2` (`5,120` steps) | 4/4 | **DONE** | **4/4 verified** |
| `wave2c` | Phase 2: `130M 1×` fresh `seed 3` (`5,120` steps) | 4/4 | **DONE** | **4/4 verified** |
| `wave3a` | Phase 3A: `130M 4×` (`80 TPP`, `20,480` steps, `seed 0`, Gate-1 LRs) | 4/4 | **DONE** | **4/4 verified** |
| `smoke-520m` | 520M memory probe + 4-arm smoke (`mb=1`, `29.73 GiB`) | — | **DONE** | N/A |
| `wave4a` | Phase 4: `520M 1×` (`10.74B` tok, `10,240` steps, `1M` batch, **matched LR**) | 4/4 | **DONE** | **4/4 verified** |
| `eval-paloma` (pass 1) | Phase 5: 16-domain Paloma on Phases 2, 3A, 4 (`20` ckpts) | 20/20 | **DONE** | N/A |
| `wave3b` | Phase 3B: `130M 4×` **matched LR** (`0.0105561`, `aux=0.0031668`) for `OD-All` & `WT-MLP-Lag` | 2/2 | **DONE** (completed `2026-11-23 17:54 +08:00`) | **2/2 verified** |
| `eval-paloma` (pass 2) | Phase 5: 16-domain Paloma on `wave3b` (`2` ckpts) | 2/2 | **DONE** (completed `2026-11-23 17:55 +08:00`) | N/A |
| Final Upload & Report | `30/30` SwanLab verified + Multi-Checkpoint Late-Stage Matched-LR Analysis in `results/RESULTS.md` | — | **DONE** | **30/30 verified** |

---

## 3. Final Headline Results (`WT-MLP-Lag` vs `MuonT` & `Muon`)

- **Phase 1 (`130M 1×`, `seed 0`, matched LRs + `Null-WT`):**
  - `Null-WT` control **PASS**: `3.23510` vs `MuonT` ref `3.23540` (`Δ = -0.00030`).
  - Matched `0.016`: `WT-MLP-Lag` beats `MuonT` by **`-0.00384`** final (`-0.00466` across **10/10** late checkpoints `3001..5120`).
  - Matched `0.02263`: `WT-MLP-Lag` beats `MuonT` by **`-0.00429`** final (`-0.00457` across **10/10** late checkpoints `3001..5120`).
- **Phase 2 (`130M 1×`, fresh paired `seeds 1, 2, 3`):**
  - `WT-MLP-Lag (0.02263)` beats tuned `MuonT (0.016)` on **3/3 seeds**: `sel = -0.00337 ± 0.00145`, `c4 = -0.00350 ± 0.00083`, 16-domain Paloma = **`-0.00401 ± 0.00270`** (`-0.01528 sel` / `-0.01551 Paloma` vs `Muon`), at `+5.1%` step time.
- **Phase 3A/3B (`130M 4×` = `80 TPP`, `10.74B` tok, `seed 0`, matched `lr = 0.0105561`):**
  - `Muon`: `sel = 3.07783`, `c4 = 3.29887`, Paloma = `3.24793`
  - `MuonT`: `sel = 3.06754`, `c4 = 3.28861`, Paloma = `3.23682`
  - `OD-All` (matched `3B`): `sel = 3.06563` (`-0.00191`), `c4 = 3.28605` (`-0.00256`), Paloma = `3.23423` (`-0.00259`, `15/16` domains)
  - **`WT-MLP-Lag` (matched `3B`)**: **`sel = 3.06359`** (**`-0.00395` vs `MuonT`**, **`-0.01424` vs `Muon`**), **`c4 = 3.28480`** (**`-0.00381`**), **Paloma = `3.23115`** (**`-0.00566`**, wins **16/16 domains**!). Beats `MuonT` on **12/12** second-half checkpoints (`10001..20480`, mean `Δsel = -0.00444`, `Δc4 = -0.00430`).
- **Phase 4 (`520M 1×` = `20 TPP`, `10.74B` tok, `1M` batch, `seed 0`, matched `lr = 0.0226274`):**
  - `Muon`: `sel = 2.85279`, `c4 = 3.07694`, Paloma = `3.02914`
  - `MuonT`: `sel = 2.84633`, `c4 = 3.07044`, Paloma = `3.02565`
  - `OD-All`: `sel = 2.84691` (`+0.00058`), `c4 = 3.07085` (`+0.00041`), Paloma = `3.02612` (`+0.00047`)
  - **`WT-MLP-Lag`**: **`sel = 2.84354`** (**`-0.00280` vs `MuonT`**, **`-0.00925` vs `Muon`**), **`c4 = 3.06823`** (**`-0.00221`**), **Paloma = `3.02020`** (**`-0.00545`**, wins **15/16 domains**!), `+10.5%` step time (`31.6 GiB`). Beats `MuonT` on **12/12** second-half checkpoints (`5001..10240`, mean `Δsel = -0.00309`, `Δc4 = -0.00252` `11/12`).
