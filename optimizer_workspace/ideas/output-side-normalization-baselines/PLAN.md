# 130M 1× Chinchilla Muon-Family Baselines Campaign (`baselines-20261128`)

- **Workspace:** `/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization-baselines`
- **PVC Campaign Root:** `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/campaigns/baselines-20261128`
- **Cluster:** 4 nodes × 8 NVIDIA A100-SXM4-40GB (`28.216.17.9`, `28.216.17.80`, `28.216.17.144`, `28.216.17.139`)
- **Model & Horizon:** `130M` (`d_model=512, d_ff=2048, heads=8, layers=32, seq_len=4096, vocab=128256`, `265,585,152` params), `1×` Chinchilla (`2.68B` tokens, `5,120` steps at `0.5M` global batch)
- **Controlled Constants:** Identical Marin data mixture & seed, WSD schedule (`0` warmup / `1024` stable / `4096` linear decay to `0`), pinned auxiliary `AdamW` on `emb`, `RMSNorm`, `lm_head` (`lr=0.0048`, `head_wd=0.1`, `other_wd=0.0`, `betas=(0.8, 0.98)`, `eps=1e-15`, `clip_norm=1.0`), decoupled `wd=0.1`, `momentum=0.95`, `nesterov=True`, `ns_steps=10`, `ns_coeffs="simple"`, `ns_eps=1e-5`, shape scale $\sqrt{\max(1, m/n)}$.

---

## 1. The Five Investigated Baselines

| Arm Key | Paper & Official Code | Mechanism | Extra State | Input Capture? |
| --- | --- | --- | --- | --- |
| `muown` | **Muown** (`arXiv:2605.10797`, `kcc-lion/muown`; also adopted in **MiMo-V2.6**, `arXiv:2610.11959v1`) | Implicit weight normalization $W = g \odot (v / \|v\|_{\text{row}})$ inside optimizer: Muon on direction $v$, bias-corrected Adam on row norms $g \in \mathbb{R}^{m \times 1}$ (`g_lr = 0.30 * lr`, matching `auxiliary_lr / lr`), decoupled WD + resync `g`. | $4m$ floats (`g, v_norm, m_g, v_g`) | No |
| `normuon` | **NorMuon** (`arXiv:2510.05491`, `zichongli5/NorMuon`) | Post-NS row-wise (neuron-wise) second-moment EMA (`beta2=0.95`, `eps=1e-10`) with exact Frobenius norm restoration: $O' = O / (\sqrt{v_t} + \epsilon)$, $O_{\text{final}} = O' \cdot \|O\|_F / (\|O'\|_F + \epsilon)$. | $m$ floats (`second_momentum`) | No |
| `muon2` (`muon2_f`) | **Muon²** / **Muon²-F** (`arXiv:2604.09967`, Algs. 1 & 2) | Pre-NS Adam-style element-wise second-moment preconditioning: $\tilde M_t = M_t / (\sqrt{V_t} + \epsilon)$ (`beta2=0.95`, `eps=1e-8`) before NS. `muon2_f` uses rank-1 factorized $V_t \approx r_t c_t^\top / \sum r_t$. | $mn$ floats (`muon2`) or $m+n$ (`muon2_f`) | No |
| `muoneq_r` (`muoneq_rc`) | **MuonEq** (`arXiv:2603.28254`, `MaeChd/muon-eq`) | Stateless pre-NS equilibration of Nesterov momentum $U$: `muoneq_r` (paper default for LLM hidden weights) rescales rows by $(\sum_j U_{ij}^2 + \epsilon)^{-1/2}$; `muoneq_rc` rescales both rows and columns. | $0$ (stateless) | No |
| `newton_muon` | **Newton-Muon** (`arXiv:2604.01472`, `zhehangdu/Newton-Muon`) | Input-side activation covariance right-preconditioner *before* momentum + NS: $G_{\text{pre}} = G (ZZ^\top + \delta I)^{-1}$ with EWMA `0.95`, refresh every `32` steps, ridge `0.2 * (tr/d) + 1e-8`, Cholesky inverse, and 4× block-diag $d \times d$ for `down`. | $d^2$ per group (+ $4d^2$ for `down`) | Yes (reuses Toner input capture) |

---

## 2. Campaign Stages

### Stage 1 — Seed 0 Full-Budget LR Bracket (`15` core runs, ~5h on 4×8 A100)
- Runs `{muown, normuon, muon2, muoneq_r, newton_muon}` × `LR ∈ {0.0113, 0.016, 0.0226}` on `Seed 0`.
- Automatic adaptive boundary guards at `0.008` (if `0.0113` wins) or `0.032` (if `0.0226` wins) to guarantee an interior optimum $\eta^*_a$ for every baseline.

### Stage 2 — Fresh Paired `Seeds 1, 2, 3` Confirmation (`15` runs, ~5h on 4×8 A100)
- Runs `Seeds 1, 2, 3` for each baseline at its verified Stage-1 optimal $\eta^*_a$, directly comparable to archived `Muon` (`3.20354`), `MuonH` (`3.20255`), `MuonT` (`3.19163`), and `WT-MLP-Lag` (`3.18813`).

---

## 3. Quick Commands

```bash
cd /root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization-baselines

# 1. Inspect planned Stage 1 runs (does not touch GPUs)
/opt/conda/envs/torch-base/bin/python campaign/run_baselines_campaign.py --wave stage1 --dry-run

# 2. Queue behind the active efficiency-20261128 campaign (recommended)
/opt/conda/envs/torch-base/bin/python campaign/run_baselines_campaign.py \
  --wave all \
  --wait-for /apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/campaigns/efficiency-20261128/completion.json

# 3. Generate/refresh comparison table at any time
/opt/conda/envs/torch-base/bin/python campaign/run_baselines_campaign.py --wave report
```
