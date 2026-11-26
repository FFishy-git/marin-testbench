"""Fast, exact CPU-only auto-research on the "rotation" & eigenvector-system mechanism
in Output-Side Normalization (WT-MLP-Lag vs MuonT / OD-All / Muon, 130M & 520M).

Reuses precomputed eigendecompositions (lam, V) across all powers (-1/2, -1/8, -1/4, +1/2, -1)
and uses float32 eigh/svd (verified <1e-7 vs fp64) with float64 inner-product accumulations.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import concurrent.futures
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

PVC = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
RUNS_130 = PVC / "runs"
RUNS_520 = PVC / "campaigns" / "confirmation-20261121" / "runs" / "wave4a"
RUNS_520_LR016 = PVC / "campaigns" / "lr-520m-20261125" / "runs"
OUT_PATH = Path(__file__).resolve().parent.parent / "tables" / "rotation_mechanism_metrics.json"

EPS = 1e-3
NS_STEPS = 10
NS_EPS = 1e-5
ABC = (3.4445, -4.7750, 2.0315)

BLOCK_KINDS = ("q", "k", "v", "o", "gate", "up", "down")
INPUT_KIND_MAP = {"q": "attn", "k": "attn", "v": "attn", "o": "o", "gate": "mlp", "up": "mlp", "down": "down"}
INPUT_D_KINDS = ("attn", "o", "mlp")
HIDDEN_OUT_KINDS = ("q", "k", "v", "o", "down")
MLP_OUT_KINDS = ("gate", "up")
FT_SQUARE_TO_BT = [2, 0, 3, 1]  # ("k", "o", "q", "v") -> ("q", "k", "v", "o")


def _bmm_tpu_cpu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.bmm(x.to(torch.bfloat16).float(), y.to(torch.bfloat16).float())


def newton_schulz_single(u: torch.Tensor, steps: int = NS_STEPS, eps: float = NS_EPS, mode: str = "tpu") -> torch.Tensor:
    a, b, c = ABC
    x = u.float().unsqueeze(0)
    x = x / (torch.linalg.matrix_norm(x, dim=(-2, -1), keepdim=True) + eps)
    tr = x.size(-2) > x.size(-1)
    if tr:
        x = x.transpose(-2, -1)
    mm = _bmm_tpu_cpu if mode == "tpu" else torch.bmm
    for _ in range(steps):
        A = mm(x, x.transpose(-2, -1))
        x = a * x + mm(b * A + c * mm(A, A), x)
    return (x.transpose(-2, -1) if tr else x)[0].float()


def exact_svd_polar(u: torch.Tensor) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(u.float(), full_matrices=False)
    mask = (S > (S.max() * 1e-7)).float()
    return (U * mask.unsqueeze(-2)) @ Vh


def spectral_pack(F: torch.Tensor, eps: float = EPS) -> dict:
    """Single float32 eigh per moment matrix, deriving all needed spectral operators in O(d^2-d^3) GEMMs."""
    d = F.shape[-1]
    F_sym = 0.5 * (F.float() + F.float().mT)
    tr = float(F_sym.diagonal().double().sum().item())
    F_tilde = F_sym * (d / max(tr, 1e-30))
    lam_asc, V_asc = torch.linalg.eigh(F_tilde)
    lam = lam_asc.flip(0).clamp_min(0.0)
    V = V_asc.flip(1)  # descending eigenvalues
    lam = lam / max(float(lam.mean().item()), 1e-30)

    def _sym_op(weights: torch.Tensor, unit_rms_norm: bool = False) -> torch.Tensor:
        M = (V * weights.unsqueeze(0)) @ V.mT
        M = 0.5 * (M + M.mT)
        if unit_rms_norm:
            nrm = float(M.double().norm().item())
            M = M * (math.sqrt(d) / max(nrm, 1e-30))
        return M

    return {
        "F_tilde": F_tilde,
        "lam": lam,
        "V": V,
        "in_half": _sym_op((lam + eps).pow(-0.5)),
        "out_eighth": _sym_op((lam + eps).pow(-0.125), unit_rms_norm=True),
        "q4": _sym_op((lam + eps).pow(-0.25)),
        "pos_half": _sym_op(lam.sqrt()),
        "inv_damped": _sym_op((lam + eps).reciprocal()),
    }


def diag_roots(diag_vec: torch.Tensor, inner_p: float = 0.5, outer_p: float = 0.125, eps: float = EPS):
    d = diag_vec.numel()
    m = diag_vec.double().clamp_min(0.0)
    tr = float(m.sum().item())
    a = m * (d / max(tr, 1e-30)) + eps
    f_in = a.pow(-inner_p)
    f_out = a.pow(-outer_p)
    f_out_hat = f_out * (math.sqrt(d) / max(float(f_out.norm().item()), 1e-30))
    return f_in.float(), f_out_hat.float()


def apply_l(fac, mat: torch.Tensor) -> torch.Tensor:
    if fac is None:
        return mat
    return fac[:, None] * mat if fac.ndim == 1 else fac @ mat


def apply_r(mat: torch.Tensor, fac) -> torch.Tensor:
    if fac is None:
        return mat
    return mat * fac[None, :] if fac.ndim == 1 else mat @ fac


def frob_cos(A: torch.Tensor, B: torch.Tensor) -> float:
    a, b = A.double(), B.double()
    den = float(a.norm().item() * b.norm().item())
    if den <= 0:
        return 0.0
    return max(-1.0, min(1.0, float((a * b).sum().item() / den)))


def deg(c: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def top_sv_fast(M: torch.Tensor, iters: int = 8) -> float:
    """Fast power iteration for spectral norm ||M||_op (accurate to <0.1%)."""
    x = M.float()
    v = x[0].clone()
    nrm = float(v.norm().item())
    if nrm <= 0:
        v = torch.ones(x.shape[1], dtype=torch.float32)
    v = v / v.norm().clamp_min(1e-30)
    for _ in range(iters):
        u = x @ v
        un = u.norm().clamp_min(1e-30)
        u = u / un
        v = x.mT @ u
        vn = v.norm().clamp_min(1e-30)
        v = v / vn
    return float((x @ v).norm().item())


def eig_erank_pct(lam: torch.Tensor) -> float:
    x = lam.double().clamp_min(0.0)
    s = float(x.sum().item())
    if s <= 0:
        return 0.0
    p = (x / s).clamp_min(1e-30)
    return float(torch.exp(-(p * p.log()).sum()).item()) / x.numel() * 100.0


def ipr_stats(V: torch.Tensor) -> dict:
    d = V.shape[0]
    v4 = (V.double() ** 4).sum(dim=0) * d
    return {
        "all_mean": float(v4.mean().item()),
        "all_median": float(v4.median().item()),
        "top16_mean": float(v4[:16].mean().item()),
        "bot64_mean": float(v4[-64:].mean().item()),
    }


def analyze_130m_rank(rank: int) -> tuple[list[dict], list[dict]]:
    p_null = RUNS_130 / "wave1/null_all/checkpoints/step_002500" / f"rank_{rank}.pt"
    ck_null = torch.load(p_null, map_location="cpu", weights_only=False)
    fast_null = ck_null["fast"]

    layers = list(range(4 * rank, 4 * rank + 4))
    m512 = fast_null["momentum"][(512, 512)].float()
    m_mlp = fast_null["momentum"][(2048, 512)].float()
    m_down = fast_null["momentum"][(512, 2048)].float()

    C_d = fast_null["input"]["groups"]["d"]["C"].float()
    C_f = fast_null["input"]["groups"]["f"]["C"].float()
    B_hid = fast_null["output"]["groups"]["out_hidden"]["C"].float()
    B_mlp = fast_null["output"]["groups"]["out_mlp"]["C"].float()

    in_map, out_map = {}, {}
    idx = 0
    for layer in layers:
        for ik in INPUT_D_KINDS:
            in_map[(layer, ik)] = spectral_pack(C_d[idx])
            idx += 1
    for li, layer in enumerate(layers):
        in_map[(layer, "down")] = spectral_pack(C_f[li])

    idx = 0
    for layer in layers:
        for k in HIDDEN_OUT_KINDS:
            out_map[(layer, k)] = spectral_pack(B_hid[idx])
            idx += 1
    idx = 0
    for layer in layers:
        for k in MLP_OUT_KINDS:
            out_map[(layer, k)] = spectral_pack(B_mlp[idx])
            idx += 1

    mech_records, ipr_records = [], []

    for side_prefix, mp in (("in", in_map), ("out", out_map)):
        for (layer, kname), sp in mp.items():
            Ft, lam, V = sp["F_tilde"], sp["lam"], sp["V"]
            d = Ft.shape[0]
            dg = Ft.diagonal().double()
            off_sq = float((Ft.double() - torch.diag(dg)).norm().item() ** 2)
            dg_sq = float((dg - 1.0).norm().item() ** 2)
            ipr_records.append({
                "scale": "130M",
                "layer": layer,
                "kind": f"{side_prefix}_{kname}",
                "width": d,
                "erank_pct": eig_erank_pct(lam),
                "diag_cv": float(dg.std(unbiased=False).item()),
                "diag_aniso_pct": 100.0 * dg_sq / max(dg_sq + off_sq, 1e-30),
                "haar_pred_diag_aniso_pct": 100.0 * 2.0 / (d + 2.0),
                **{f"ipr_{k2}": v2 for k2, v2 in ipr_stats(V).items()},
            })

    for li, layer in enumerate(layers):
        mom_layer = {
            "q": m512[4 * li + 0], "k": m512[4 * li + 1], "v": m512[4 * li + 2], "o": m512[4 * li + 3],
            "gate": m_mlp[2 * li + 0], "up": m_mlp[2 * li + 1], "down": m_down[li],
        }
        for k in BLOCK_KINDS:
            M = mom_layer[k]
            spC = in_map[(layer, INPUT_KIND_MAP[k])]
            spB = out_map[(layer, k)]

            R_in, R_out_hat, R_q4, C_half = spC["in_half"], spC["out_eighth"], spC["q4"], spC["pos_half"]
            B_tilde, V_B, L_in_dense, L_out_dense, L_q4, B_half, B_inv = (
                spB["F_tilde"], spB["V"], spB["in_half"], spB["out_eighth"], spB["q4"], spB["pos_half"], spB["inv_damped"]
            )
            L_in_diag, L_out_diag = diag_roots(B_tilde.diagonal())
            L_in_wt, L_out_wt = (L_in_dense, L_out_dense) if k in HIDDEN_OUT_KINDS else (L_in_diag, L_out_diag)

            Z0 = apply_r(M, R_in)
            Z_dense = apply_l(L_in_dense, Z0)
            Z_wt = apply_l(L_in_wt, Z0)
            Z_diag = apply_l(L_in_diag, Z0)

            # Single SVD per Z0 / Z_wt / Z_dense gives exact polar AND singular values AND left singular vectors!
            U_Z0, S_Z0, Vh_Z0 = torch.linalg.svd(Z0, full_matrices=False)
            U_Zwt, S_Zwt, Vh_Zwt = torch.linalg.svd(Z_wt, full_matrices=False)
            P0_svd = U_Z0 @ Vh_Z0
            Pwt_svd = U_Zwt @ Vh_Zwt
            Pdense_svd = Pwt_svd if k in HIDDEN_OUT_KINDS else exact_svd_polar(Z_dense)

            D_muont_svd = apply_r(P0_svd, R_out_hat)
            D_inner_svd = apply_r(Pwt_svd, R_out_hat)
            D_outer_svd = apply_l(L_out_wt, D_muont_svd)
            D_wt_svd = apply_l(L_out_wt, D_inner_svd)

            # Actual NS10 directions
            P0_ns = newton_schulz_single(Z0)
            Pwt_ns = newton_schulz_single(Z_wt)
            Pdiag_ns = Pwt_ns if k in MLP_OUT_KINDS else newton_schulz_single(Z_diag)
            Pdense_ns = Pwt_ns if k in HIDDEN_OUT_KINDS else newton_schulz_single(Z_dense)
            Pmuon_ns = newton_schulz_single(M)

            D_muon = Pmuon_ns
            D_muont = apply_r(P0_ns, R_out_hat)
            D_inner_only = apply_r(Pwt_ns, R_out_hat)
            D_outer_only = apply_l(L_out_wt, D_muont)
            D_wt = apply_l(L_out_wt, D_inner_only)
            D_od = apply_l(L_out_diag, apply_r(Pdiag_ns, R_out_hat))
            D_dt = apply_l(L_out_dense, apply_r(Pdense_ns, R_out_hat))
            D_gomuon = apply_l(L_q4, apply_r(newton_schulz_single(apply_l(L_q4, apply_r(M, R_q4))), R_q4))

            so_m_err = 0.0
            commutator_rel = 0.0
            if M.shape[0] <= M.shape[1]:
                S0 = (U_Z0 * S_Z0.unsqueeze(0)) @ U_Z0.mT
                Omega = exact_svd_polar(L_in_dense @ S0)
                so_m_err = float((Pdense_svd - Omega @ P0_svd).double().norm().item() / Pdense_svd.double().norm().item())
                comm = L_in_dense @ S0 - S0 @ L_in_dense
                commutator_rel = float(comm.double().norm().item() / (L_in_dense.double().norm().item() * S0.double().norm().item()))

            # Exp 2: SV spectrum & Hijacking
            erank_Z0 = eig_erank_pct(S_Z0)
            erank_Zwt = eig_erank_pct(S_Zwt)
            top1_Z0 = float((S_Z0[0].double() ** 2).item() / (S_Z0.double() ** 2).sum().item() * 100.0)
            top1_Zwt = float((S_Zwt[0].double() ** 2).item() / (S_Zwt.double() ** 2).sum().item() * 100.0)

            k_sub = 16
            hijack_Z0_top16 = float(((U_Z0[:, :k_sub].mT @ V_B[:, :k_sub]).double() ** 2).sum().item() / k_sub)
            hijack_Zwt_top16 = float(((U_Zwt[:, :k_sub].mT @ V_B[:, :k_sub]).double() ** 2).sum().item() / k_sub)
            hijack_top1_Z0 = float(((U_Z0[:, :1].mT @ V_B[:, :8]).double() ** 2).sum().item())
            hijack_top1_Zwt = float(((U_Zwt[:, :1].mT @ V_B[:, :8]).double() ** 2).sum().item())

            # Dual Procrustes alignment along every eigenvector u_i of B
            G_rows = (V_B.mT @ Z0).double()
            O0_rows = (V_B.mT @ P0_ns).double()
            Owt_rows = (V_B.mT @ Pwt_ns).double()

            g_nrm = G_rows.norm(dim=1).clamp_min(1e-30)
            cos_mode_muont = (G_rows * O0_rows).sum(dim=1) / (g_nrm * O0_rows.norm(dim=1).clamp_min(1e-30))
            cos_mode_wt = (G_rows * Owt_rows).sum(dim=1) / (g_nrm * Owt_rows.norm(dim=1).clamp_min(1e-30))

            m_dim = B_tilde.shape[0]
            n_top10 = max(1, m_dim // 10)
            n_bot80_start = max(1, m_dim // 5)

            # Exp 3: Curvature-quadratic shock & Taylor descent potential
            M64 = M.double()
            M_nrm = float(M64.norm().item())

            def eval_dir(Dc: torch.Tensor) -> dict:
                dn = float(Dc.double().norm().item())
                Dh = Dc / max(dn, 1e-30)
                lin = float((M64 * Dh.double()).sum().item() / max(M_nrm, 1e-30))
                W_shock = (B_half @ Dh) @ C_half
                q_bc = float((W_shock.double() ** 2).sum().item())
                op_bc = top_sv_fast(W_shock)
                q_b = float(((B_half @ Dh).double() ** 2).sum().item())
                inv_b_noise = float((Dh.double() * (B_inv @ Dh).double()).sum().item())
                phi = (lin * lin) / max(q_bc, 1e-30)
                return {"lin_cos_M": lin, "Q_BC": q_bc, "op_BC": op_bc, "Q_B": q_b, "invB_noise": inv_b_noise, "Phi_BC": phi}

            ev_muon = eval_dir(D_muon)
            ev_muont = eval_dir(D_muont)
            ev_inner = eval_dir(D_inner_only)
            ev_outer = eval_dir(D_outer_only)
            ev_od = eval_dir(D_od)
            ev_wt = eval_dir(D_wt)
            ev_gomuon = eval_dir(D_gomuon)

            def top10_out_energy_share(Dc: torch.Tensor) -> float:
                Dh = Dc / max(float(Dc.double().norm().item()), 1e-30)
                proj = (V_B.mT @ Dh) @ C_half
                e_per_mode = (proj.double() ** 2).sum(dim=1)
                return float(e_per_mode[:n_top10].sum().item() / max(float(e_per_mode.sum().item()), 1e-30) * 100.0)

            mech_records.append({
                "layer": layer,
                "kind": k,
                "deg_inner_svd": deg(frob_cos(D_inner_svd, D_muont_svd)),
                "deg_outer_svd": deg(frob_cos(D_outer_svd, D_muont_svd)),
                "deg_total_svd": deg(frob_cos(D_wt_svd, D_muont_svd)),
                "deg_inner_ns10": deg(frob_cos(D_inner_only, D_muont)),
                "deg_outer_ns10": deg(frob_cos(D_outer_only, D_muont)),
                "deg_wt_ns10": deg(frob_cos(D_wt, D_muont)),
                "deg_od_ns10": deg(frob_cos(D_od, D_muont)),
                "deg_dt_ns10": deg(frob_cos(D_dt, D_muont)),
                "so_m_exact_err": so_m_err,
                "commutator_rel": commutator_rel,
                "cos_inner_vs_outer_delta": frob_cos(D_inner_only - D_muont, D_outer_only - D_muont),
                "erank_Z0_pct": erank_Z0,
                "erank_Zwt_pct": erank_Zwt,
                "top1_Z0_pct": top1_Z0,
                "top1_Zwt_pct": top1_Zwt,
                "hijack_Z0_top16": hijack_Z0_top16,
                "hijack_Zwt_top16": hijack_Zwt_top16,
                "hijack_random_chance": k_sub / m_dim,
                "hijack_top1_Z0_in_Btop8": hijack_top1_Z0,
                "hijack_top1_Zwt_in_Btop8": hijack_top1_Zwt,
                "procrustes_cos_top10_muont": float(cos_mode_muont[:n_top10].mean().item()),
                "procrustes_cos_top10_wt": float(cos_mode_wt[:n_top10].mean().item()),
                "procrustes_cos_bot80_muont": float(cos_mode_muont[n_bot80_start:].mean().item()),
                "procrustes_cos_bot80_wt": float(cos_mode_wt[n_bot80_start:].mean().item()),
                "procrustes_cos_all_muont": float(cos_mode_muont.mean().item()),
                "procrustes_cos_all_wt": float(cos_mode_wt.mean().item()),
                "procrustes_pct_modes_improved": float((cos_mode_wt > cos_mode_muont).double().mean().item() * 100.0),
                "procrustes_pct_bot80_improved": float((cos_mode_wt[n_bot80_start:] > cos_mode_muont[n_bot80_start:]).double().mean().item() * 100.0),
                "lin_cos_M_muont": ev_muont["lin_cos_M"],
                "lin_cos_M_wt": ev_wt["lin_cos_M"],
                "Q_BC_wt_over_muont": ev_wt["Q_BC"] / ev_muont["Q_BC"],
                "Q_BC_od_over_muont": ev_od["Q_BC"] / ev_muont["Q_BC"],
                "op_BC_wt_over_muont": ev_wt["op_BC"] / ev_muont["op_BC"],
                "op_BC_od_over_muont": ev_od["op_BC"] / ev_muont["op_BC"],
                "Phi_wt_over_muont": ev_wt["Phi_BC"] / ev_muont["Phi_BC"],
                "Phi_od_over_muont": ev_od["Phi_BC"] / ev_muont["Phi_BC"],
                "Phi_inner_over_muont": ev_inner["Phi_BC"] / ev_muont["Phi_BC"],
                "Phi_outer_over_muont": ev_outer["Phi_BC"] / ev_muont["Phi_BC"],
                "Phi_gomuon_over_muont": ev_gomuon["Phi_BC"] / ev_muont["Phi_BC"],
                "invB_noise_muont": ev_muont["invB_noise"],
                "invB_noise_wt": ev_wt["invB_noise"],
                "invB_noise_gomuon": ev_gomuon["invB_noise"],
                "top10B_out_share_muont_pct": top10_out_energy_share(D_muont),
                "top10B_out_share_od_pct": top10_out_energy_share(D_od),
                "top10B_out_share_wt_pct": top10_out_energy_share(D_wt),
            })

    del ck_null
    return mech_records, ipr_records


def analyze_520m_rank(rank: int) -> tuple[list[dict], list[dict]]:
    p_wt = RUNS_520_LR016 / "wt_mlp_lag_520m_1x_seed0_lr0p016/checkpoints/step_005120" / f"rank_{rank}.pt"
    p_mt = RUNS_520_LR016 / "muont_520m_1x_seed0_lr0p016/checkpoints/step_005120" / f"rank_{rank}.pt"
    ck_wt = torch.load(p_wt, map_location="cpu", weights_only=False)
    ck_mt = torch.load(p_mt, map_location="cpu", weights_only=False)

    layers = list(range(4 * rank, 4 * rank + 4))
    fast_wt = ck_wt["fast"]
    fast_mt = ck_mt["fast"]

    mt_m1024 = fast_mt["momentum"][(1024, 1024)]["m"].float()
    mt_mmlp = fast_mt["momentum"][(4096, 1024)]["m"].float()
    mt_mdown = fast_mt["momentum"][(1024, 4096)]["m"].float()
    mt_Cd = fast_mt["groups"]["d"]["C"].float()
    mt_Cf = fast_mt["groups"]["f"]["C"].float()

    wt_Cd = fast_wt["input"]["groups"]["d"]["C"].float()
    wt_Bhid = fast_wt["output"]["groups"]["out_hidden"]["C"].float()
    wt_Bmlp_diag = fast_wt["output_diagonal"]["moment"][4096][8 * rank : 8 * (rank + 1)].float()

    in_mt = {}
    idx = 0
    for layer in layers:
        for ik in INPUT_D_KINDS:
            in_mt[(layer, ik)] = spectral_pack(mt_Cd[idx])
            idx += 1
    for li, layer in enumerate(layers):
        in_mt[(layer, "down")] = spectral_pack(mt_Cf[li])

    out_wt, ipr_520 = {}, []

    idx = 0
    for layer in layers:
        for ik in INPUT_D_KINDS:
            sp = spectral_pack(wt_Cd[idx])
            Ft, lam, V = sp["F_tilde"], sp["lam"], sp["V"]
            dg = Ft.diagonal().double()
            off_sq = float((Ft.double() - torch.diag(dg)).norm().item() ** 2)
            dg_sq = float((dg - 1.0).norm().item() ** 2)
            ipr_520.append({
                "scale": "520M", "layer": layer, "kind": f"in_{ik}", "width": 1024,
                "erank_pct": eig_erank_pct(lam),
                "diag_cv": float(dg.std(unbiased=False).item()),
                "diag_aniso_pct": 100.0 * dg_sq / max(dg_sq + off_sq, 1e-30),
                "haar_pred_diag_aniso_pct": 100.0 * 2.0 / 1026.0,
                **{f"ipr_{k2}": v2 for k2, v2 in ipr_stats(V).items()},
            })
            idx += 1

    I_1024 = torch.eye(1024)
    idx = 0
    for layer in layers:
        for ok in HIDDEN_OUT_KINDS:
            sp = spectral_pack(wt_Bhid[idx])
            out_wt[(layer, ok)] = sp
            Ft, lam, V, Lin_d = sp["F_tilde"], sp["lam"], sp["V"], sp["in_half"]
            dg = Ft.diagonal().double()
            off_sq = float((Ft.double() - torch.diag(dg)).norm().item() ** 2)
            dg_sq = float((dg - 1.0).norm().item() ** 2)
            Lin_dg, _ = diag_roots(Ft.diagonal())
            num_sq = float((Lin_d - torch.diag(Lin_dg)).double().norm().item() ** 2)
            den_sq = float((Lin_d - I_1024).double().norm().item() ** 2)
            ipr_520.append({
                "scale": "520M", "layer": layer, "kind": f"out_{ok}", "width": 1024,
                "erank_pct": eig_erank_pct(lam),
                "diag_cv": float(dg.std(unbiased=False).item()),
                "diag_aniso_pct": 100.0 * dg_sq / max(dg_sq + off_sq, 1e-30),
                "haar_pred_diag_aniso_pct": 100.0 * 2.0 / 1026.0,
                "r2_L_pct": 100.0 * (1.0 - num_sq / max(den_sq, 1e-30)),
                "eig_max_over_mean": float(lam[0].item()),
                **{f"ipr_{k2}": v2 for k2, v2 in ipr_stats(V).items()},
            })
            idx += 1

    idx = 0
    for layer in layers:
        for ok in MLP_OUT_KINDS:
            dvec = wt_Bmlp_diag[idx]
            Lin_dg, Lout_dg = diag_roots(dvec)
            out_wt[(layer, ok)] = {"diag_vec": dvec, "in_half": Lin_dg, "out_eighth": Lout_dg}
            idx += 1

    cf_520 = []
    for li, layer in enumerate(layers):
        sub1024 = mt_m1024[4 * li : 4 * (li + 1)][FT_SQUARE_TO_BT]
        mom_mt = {
            "q": sub1024[0], "k": sub1024[1], "v": sub1024[2], "o": sub1024[3],
            "gate": mt_mmlp[2 * li + 0], "up": mt_mmlp[2 * li + 1], "down": mt_mdown[li],
        }
        for k in BLOCK_KINDS:
            M = mom_mt[k]
            spC = in_mt[(layer, INPUT_KIND_MAP[k])]
            spB = out_wt[(layer, k)]
            R_in, R_out_hat, C_half = spC["in_half"], spC["out_eighth"], spC["pos_half"]
            L_in_wt, L_out_wt = spB["in_half"], spB["out_eighth"]

            if k in HIDDEN_OUT_KINDS:
                L_in_od, L_out_od = diag_roots(spB["F_tilde"].diagonal())
            else:
                L_in_od, L_out_od = L_in_wt, L_out_wt

            Z0 = apply_r(M, R_in)
            D_muont = apply_r(newton_schulz_single(Z0), R_out_hat)
            D_wt = apply_l(L_out_wt, apply_r(newton_schulz_single(apply_l(L_in_wt, Z0)), R_out_hat))
            D_od = apply_l(L_out_od, apply_r(newton_schulz_single(apply_l(L_in_od, Z0)), R_out_hat))

            rec = {
                "layer": layer,
                "kind": k,
                "deg_wt_vs_muont": deg(frob_cos(D_wt, D_muont)),
                "deg_od_vs_muont": deg(frob_cos(D_od, D_muont)),
                "norm_wt_over_muont": float(D_wt.double().norm().item() / D_muont.double().norm().item()),
                "norm_od_over_muont": float(D_od.double().norm().item() / D_muont.double().norm().item()),
            }
            if k in HIDDEN_OUT_KINDS:
                B_half = spB["pos_half"]
                M64 = M.double()
                M_nrm = float(M64.norm().item())

                def _phi_q(Dc):
                    Dh = Dc / float(Dc.double().norm().item())
                    lin = float((M64 * Dh.double()).sum().item() / M_nrm)
                    Ws = (B_half @ Dh) @ C_half
                    q = float((Ws.double() ** 2).sum().item())
                    return q, (lin * lin) / q

                q_mt, phi_mt = _phi_q(D_muont)
                q_od, phi_od = _phi_q(D_od)
                q_wt, phi_wt = _phi_q(D_wt)
                rec["Q_BC_wt_over_muont"] = q_wt / q_mt
                rec["Q_BC_od_over_muont"] = q_od / q_mt
                rec["Phi_wt_over_muont"] = phi_wt / phi_mt
                rec["Phi_od_over_muont"] = phi_od / phi_mt
            cf_520.append(rec)

    del ck_wt, ck_mt
    return cf_520, ipr_520


def analyze_weights_and_isospectral() -> dict:
    out = {}
    arms_130 = {
        "Muon": RUNS_130 / "wave0/muon/checkpoints/step_005120/rank_0.pt",
        "MuonT": RUNS_130 / "wave_sym_fastlag/muont/checkpoints/step_005120/rank_0.pt",
        "OD-All": RUNS_130 / "wave2/inner_outer_leftdiag/checkpoints/step_005120/rank_0.pt",
        "WT-MLP-Lag": RUNS_130 / "wave_sym_fastlag/inner_outer_mlpdiag_lag/checkpoints/step_005120/rank_0.pt",
    }
    arms_520 = {
        "Muon": RUNS_520 / "muon_520m_1x_seed0/checkpoints/step_010240/rank_0.pt",
        "MuonT": RUNS_520_LR016 / "muont_520m_1x_seed0_lr0p016/checkpoints/step_010240/rank_0.pt",
        "OD-All": RUNS_520_LR016 / "od_all_520m_1x_seed0_lr0p016/checkpoints/step_010240/rank_0.pt",
        "WT-MLP-Lag": RUNS_520_LR016 / "wt_mlp_lag_520m_1x_seed0_lr0p016/checkpoints/step_010240/rank_0.pt",
    }

    for scale_label, arm_paths, sample_layers in (
        ("130M_step5120", arms_130, list(range(0, 32, 4))),
        ("520M_step10240", arms_520, list(range(0, 32, 8))),
    ):
        scale_res = {}
        for arm_name, p in arm_paths.items():
            ck = torch.load(p, map_location="cpu", weights_only=False)
            sd = ck["model"]
            by_k = {}
            for k in BLOCK_KINDS:
                eranks, s_ranks, top1_ratios = [], [], []
                for l in sample_layers:
                    W = sd[f"body.blocks.{l}.{k}.weight"].float()
                    s = torch.linalg.svdvals(W).double()
                    r = s.numel()
                    p_s = (s / s.sum()).clamp_min(1e-30)
                    eranks.append(float(torch.exp(-(p_s * p_s.log()).sum()).item() / r * 100.0))
                    s_ranks.append(float((s * s).sum().item() / (s[0].item() ** 2) / r * 100.0))
                    top1_ratios.append(float(s[0].item() / s.mean().item()))
                by_k[k] = {
                    "erank_s1_pct": float(np.mean(eranks)),
                    "stable_rank_pct": float(np.mean(s_ranks)),
                    "sv_max_over_mean": float(np.mean(top1_ratios)),
                }
            scale_res[arm_name] = by_k
            del ck, sd
        out[scale_label] = scale_res

    ck_w = torch.load(RUNS_130 / "wave1/null_all/checkpoints/step_002500/rank_0.pt", map_location="cpu", weights_only=False)["model"]
    iso_by_kind = defaultdict(lambda: defaultdict(list))
    for rank in (0, 4, 7):
        ck = torch.load(RUNS_130 / f"wave1/null_all/checkpoints/step_002500/rank_{rank}.pt", map_location="cpu", weights_only=False)
        fast = ck["fast"]
        layers = list(range(4 * rank, 4 * rank + 4))
        m512 = fast["momentum"][(512, 512)].float()
        m_mlp = fast["momentum"][(2048, 512)].float()
        m_down = fast["momentum"][(512, 2048)].float()
        C_d = fast["input"]["groups"]["d"]["C"].float()
        C_f = fast["input"]["groups"]["f"]["C"].float()
        B_hid = fast["output"]["groups"]["out_hidden"]["C"].float()
        B_mlp = fast["output"]["groups"]["out_mlp"]["C"].float()

        in_m, out_m = {}, {}
        idx = 0
        for layer in layers:
            for ik in INPUT_D_KINDS:
                in_m[(layer, ik)] = spectral_pack(C_d[idx])
                idx += 1
        for li, layer in enumerate(layers):
            in_m[(layer, "down")] = spectral_pack(C_f[li])
        idx = 0
        for layer in layers:
            for ok in HIDDEN_OUT_KINDS:
                out_m[(layer, ok)] = spectral_pack(B_hid[idx])
                idx += 1
        idx = 0
        for layer in layers:
            for ok in MLP_OUT_KINDS:
                out_m[(layer, ok)] = spectral_pack(B_mlp[idx])
                idx += 1

        for li, layer in enumerate(layers):
            moms = {
                "q": m512[4 * li + 0], "k": m512[4 * li + 1], "v": m512[4 * li + 2], "o": m512[4 * li + 3],
                "gate": m_mlp[2 * li + 0], "up": m_mlp[2 * li + 1], "down": m_down[li],
            }
            for k in BLOCK_KINDS:
                W = ck_w[f"body.blocks.{layer}.{k}.weight"].float()
                Uw, _, Vwh = torch.linalg.svd(W, full_matrices=False)
                M = moms[k]
                spC = in_m[(layer, INPUT_KIND_MAP[k])]
                spB = out_m[(layer, k)]
                R_in, R_out = spC["in_half"], spC["out_eighth"]
                if k in HIDDEN_OUT_KINDS:
                    L_in_wt, L_out_wt = spB["in_half"], spB["out_eighth"]
                else:
                    L_in_wt, L_out_wt = diag_roots(spB["F_tilde"].diagonal())

                Z0 = apply_r(M, R_in)
                D_raw = M
                D_muon = newton_schulz_single(M)
                D_muont = apply_r(newton_schulz_single(Z0), R_out)
                D_wt = apply_l(L_out_wt, apply_r(newton_schulz_single(apply_l(L_in_wt, Z0)), R_out))

                for label, Dc in (("Raw_M", D_raw), ("Muon", D_muon), ("MuonT", D_muont), ("WT-MLP", D_wt)):
                    diag_proj = (Uw.mT * (Dc @ Vwh.mT).mT).sum(dim=0).double()
                    rho = 1.0 - float((diag_proj * diag_proj).sum().item() / max(float((Dc.double() ** 2).sum().item()), 1e-30))
                    iso_by_kind[k][label].append(rho * 100.0)
        del ck

    out["isospectral_rho_pct_130m_step2500"] = {
        k: {lbl: float(np.mean(vals)) for lbl, vals in d.items()} for k, d in iso_by_kind.items()
    }
    return out


def agg_by_kind(records: list[dict]) -> dict:
    by_k = defaultdict(list)
    for r in records:
        by_k[r["kind"]].append(r)
    out = {}
    for k, recs in by_k.items():
        flds = [f for f, v in recs[0].items() if isinstance(v, (int, float)) and f not in ("layer", "width")]
        out[k] = {
            f: {
                "mean": float(np.mean([x[f] for x in recs])),
                "median": float(np.median([x[f] for x in recs])),
            }
            for f in flds
        }
    return out


def main():
    t0 = time.perf_counter()
    print("=== [1/3] Running 130M all-32-layer mechanistic rotation & Procrustes experiments ===", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        res_130 = list(pool.map(analyze_130m_rank, range(8)))
    mech_130 = [x for r, _ in res_130 for x in r]
    ipr_130 = [x for _, ip in res_130 for x in ip]
    print(f"    Finished 130M ({len(mech_130)} matrices, {len(ipr_130)} moments) in {time.perf_counter() - t0:.2f}s", flush=True)

    t1 = time.perf_counter()
    print("=== [2/3] Running 520M (d=1024) scaling, IPR, & counterfactual rotation experiments (ranks 0,2,4,6 = 16 layers) ===", flush=True)
    ranks_520 = (0, 2, 4, 6)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        res_520 = list(pool.map(analyze_520m_rank, ranks_520))
    cf_520 = [x for r, _ in res_520 for x in r]
    ipr_520 = [x for _, ip in res_520 for x in ip]
    print(f"    Finished 520M ({len(cf_520)} matrices, {len(ipr_520)} moments) in {time.perf_counter() - t1:.2f}s", flush=True)

    t2 = time.perf_counter()
    print("=== [3/3] Running Weight Spectrum & Isospectral Tangent Rotation audit (130M & 520M) ===", flush=True)
    w_res = analyze_weights_and_isospectral()
    print(f"    Finished Weight & Isospectral audit in {time.perf_counter() - t2:.2f}s", flush=True)

    summary = {
        "meta": {"total_seconds": round(time.perf_counter() - t0, 2)},
        "exp123_130m_by_kind": agg_by_kind(mech_130),
        "exp4_ipr_130m_by_kind": agg_by_kind(ipr_130),
        "exp4_ipr_520m_by_kind": agg_by_kind(ipr_520),
        "exp4_cf_520m_by_kind": agg_by_kind(cf_520),
        "exp5_weights_and_isospectral": w_res,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    s130 = summary["exp123_130m_by_kind"]
    print("\n--- EXP 1: ROTATION DECOMPOSITION (130M, 32-layer MEAN angles vs MuonT) ---")
    print(f"{'kind':<6} {'in_SVD':>8} {'in_NS10':>8} {'out_NS10':>9} {'WT_NS10':>8} {'OD_NS10':>8} {'SO(m)err':>9} {'commRel':>8} {'cos(dI,dO)':>11}")
    for k in BLOCK_KINDS:
        x = s130[k]
        print(
            f"{k:<6} "
            f"{x['deg_inner_svd']['mean']:>7.2f}° "
            f"{x['deg_inner_ns10']['mean']:>7.2f}° "
            f"{x['deg_outer_ns10']['mean']:>8.2f}° "
            f"{x['deg_wt_ns10']['mean']:>7.2f}° "
            f"{x['deg_od_ns10']['mean']:>7.2f}° "
            f"{x['so_m_exact_err']['mean']:>9.2e} "
            f"{x['commutator_rel']['mean']:>8.4f} "
            f"{x['cos_inner_vs_outer_delta']['mean']:>+11.4f}"
        )

    print("\n--- EXP 2: SINGULAR-VECTOR HIJACKING & DUAL PROCRUSTES EQUITY (130M, 32-layer MEAN) ---")
    print(f"{'kind':<6} {'erZ0%':>6} {'erZwt%':>7} {'hijZ0_16':>9} {'hijZwt16':>9} {'rand16':>7} {'top10_MT':>9} {'top10_WT':>9} {'bot80_MT':>9} {'bot80_WT':>9} {'%bot80Win':>10}")
    for k in BLOCK_KINDS:
        x = s130[k]
        print(
            f"{k:<6} "
            f"{x['erank_Z0_pct']['mean']:>6.1f} "
            f"{x['erank_Zwt_pct']['mean']:>7.1f} "
            f"{x['hijack_Z0_top16']['mean']:>9.4f} "
            f"{x['hijack_Zwt_top16']['mean']:>9.4f} "
            f"{x['hijack_random_chance']['mean']:>7.4f} "
            f"{x['procrustes_cos_top10_muont']['mean']:>9.4f} "
            f"{x['procrustes_cos_top10_wt']['mean']:>9.4f} "
            f"{x['procrustes_cos_bot80_muont']['mean']:>9.4f} "
            f"{x['procrustes_cos_bot80_wt']['mean']:>9.4f} "
            f"{x['procrustes_pct_bot80_improved']['mean']:>9.1f}%"
        )

    print("\n--- EXP 3: CURVATURE-QUADRATIC SHOCK Q_{B,C} & TAYLOR POTENTIAL Phi_{B,C} (130M, 32-layer MEAN, unit ||D||_F) ---")
    print(f"{'kind':<6} {'Q_WT/MT':>8} {'Q_OD/MT':>8} {'opWT/MT':>8} {'Phi_in/MT':>10} {'Phi_out/MT':>11} {'Phi_OD/MT':>10} {'Phi_WT/MT':>10} {'top10E_MT':>10} {'top10E_WT':>10} {'invBNs_GM/WT':>13}")
    for k in BLOCK_KINDS:
        x = s130[k]
        gm_over_wt_noise = x["invB_noise_gomuon"]["mean"] / x["invB_noise_wt"]["mean"]
        print(
            f"{k:<6} "
            f"{x['Q_BC_wt_over_muont']['mean']:>8.4f} "
            f"{x['Q_BC_od_over_muont']['mean']:>8.4f} "
            f"{x['op_BC_wt_over_muont']['mean']:>8.4f} "
            f"{x['Phi_inner_over_muont']['mean']:>10.4f} "
            f"{x['Phi_outer_over_muont']['mean']:>11.4f} "
            f"{x['Phi_od_over_muont']['mean']:>10.4f} "
            f"{x['Phi_wt_over_muont']['mean']:>10.4f} "
            f"{x['top10B_out_share_muont_pct']['mean']:>9.2f}% "
            f"{x['top10B_out_share_wt_pct']['mean']:>9.2f}% "
            f"{gm_over_wt_noise:>12.2f}x"
        )

    print("\n--- EXP 4A: EIGENVECTOR IPR & O(1/d) HAAR SCALING (130M d=512 vs 520M d=1024 MEDIANS) ---")
    ip130 = summary["exp4_ipr_130m_by_kind"]
    ip520 = summary["exp4_ipr_520m_by_kind"]
    print(f"{'kind':<10} {'IPR_130':>9} {'IPR_520':>9} {'DgAni130%':>10} {'DgAni520%':>10} {'Haar130%':>9} {'Haar520%':>9}")
    for k in ("out_o", "out_down", "out_v", "out_q", "out_k", "out_gate", "out_up", "in_attn", "in_mlp", "in_o", "in_down"):
        a = ip130[k]
        b = ip520.get(k)
        ip_b = f"{b['ipr_all_median']['median']:>9.2f}" if b else f"{'---':>9}"
        da_b = f"{b['diag_aniso_pct']['median']:>9.3f}%" if b else f"{'---':>10}"
        hp_b = f"{b['haar_pred_diag_aniso_pct']['median']:>8.3f}%" if b else f"{'---':>9}"
        print(
            f"{k:<10} "
            f"{a['ipr_all_median']['median']:>9.2f} "
            f"{ip_b} "
            f"{a['diag_aniso_pct']['median']:>9.3f}% "
            f"{da_b} "
            f"{a['haar_pred_diag_aniso_pct']['median']:>8.3f}% "
            f"{hp_b}"
        )

    print("\n--- EXP 4B: FIRST-EVER 520M (d=1024) COUNTERFACTUAL ROTATION & DESCENT POTENTIAL ---")
    c520 = summary["exp4_cf_520m_by_kind"]
    print(f"{'kind':<6} {'deg_WT_520':>11} {'deg_OD_520':>11} {'Q_WT/MT':>9} {'Q_OD/MT':>9} {'Phi_WT/MT':>10} {'Phi_OD/MT':>10}")
    for k in BLOCK_KINDS:
        x = c520[k]
        qw = f"{x['Q_BC_wt_over_muont']['mean']:>9.4f}" if "Q_BC_wt_over_muont" in x else f"{'---':>9}"
        qo = f"{x['Q_BC_od_over_muont']['mean']:>9.4f}" if "Q_BC_od_over_muont" in x else f"{'---':>9}"
        pw = f"{x['Phi_wt_over_muont']['mean']:>10.4f}" if "Phi_wt_over_muont" in x else f"{'---':>10}"
        po = f"{x['Phi_od_over_muont']['mean']:>10.4f}" if "Phi_od_over_muont" in x else f"{'---':>10}"
        print(
            f"{k:<6} "
            f"{x['deg_wt_vs_muont']['mean']:>10.2f}° "
            f"{x['deg_od_vs_muont']['mean']:>10.2f}° "
            f"{qw} {qo} {pw} {po}"
        )

    print("\n--- EXP 5: ISOSPECTRAL TANGENT ROTATION FRACTION rho(D; W) (%) AT 130M STEP 2500 ---")
    iso = w_res["isospectral_rho_pct_130m_step2500"]
    print(f"{'kind':<6} {'Raw_M':>9} {'Muon':>9} {'MuonT':>9} {'WT-MLP':>9}")
    for k in BLOCK_KINDS:
        print(f"{k:<6} {iso[k]['Raw_M']:>8.2f}% {iso[k]['Muon']:>8.2f}% {iso[k]['MuonT']:>8.2f}% {iso[k]['WT-MLP']:>8.2f}%")


if __name__ == "__main__":
    main()
