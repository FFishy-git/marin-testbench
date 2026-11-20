"""Build mechanistic moment-geometry and step-size artifacts for the paper.

By default, reads the compact checkpoint-derived summary in
`results/mechanism_summary.json` and emits:

  results/mechanism_numbers.tex   LaTeX macros for numbers quoted in prose
  results/table_mechanism.tex     Tabular comparing all 11 output/input moments
  figs/mechanism_bars.dat         pgfplots data table for mechanism figures

Pass `--recompute` to recompute `results/mechanism_summary.json` on CPU
(`CUDA_VISIBLE_DEVICES=""`) directly from the saved distributed checkpoints on
the PVC share, or `--from-scratch-dir DIR` to assemble `mechanism_summary.json`
from precomputed `moment_diagnostics.json` and `step_size_diagnostics.json`.

    python -I scripts/analyze_mechanism.py
"""

import argparse
import concurrent.futures
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent.parent
PVC = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
RUNS = PVC / "runs"

EPS = 1e-3
NS_STEPS = 10
NS_EPS = 1e-5
WD = 0.1
ABC = (3.4445, -4.7750, 2.0315)

BLOCK_KINDS = ("q", "k", "v", "o", "gate", "up", "down")
INPUT_KIND_MAP = {"q": "attn", "k": "attn", "v": "attn", "o": "o", "gate": "mlp", "up": "mlp", "down": "down"}
INPUT_D_KINDS = ("attn", "o", "mlp")
HIDDEN_OUT_KINDS = ("q", "k", "v", "o", "down")
MLP_OUT_KINDS = ("gate", "up")
FT_512_TO_BT = [2, 0, 3, 1]

MOMENT_ORDER = (
    "out_gate",
    "out_up",
    "out_q",
    "out_k",
    "out_v",
    "out_o",
    "out_down",
    "in_down",
    "in_attn",
    "in_o",
    "in_mlp",
)

MOMENT_KEY_FIELDS = (
    "diag_energy_frac",
    "offdiag_corr_rms",
    "offdiag_corr_energy_per_dim",
    "diag_cv",
    "diag_max_over_mean",
    "eig_max_over_mean",
    "corr_eig_max_over_mean",
    "eig_B_top1_share",
    "eig_B_erank_frac",
    "corr_erank_frac",
    "corr_eig_min",
    "cos_L",
    "rel_err_L",
    "rel_err_drop_L",
    "diag_over_full_depart_norm_L",
    "explained_depart_var_L",
    "true_diag_share_depart_var_L",
    "cos_Lhat",
    "rel_err_Lhat",
    "rel_err_drop_Lhat",
    "diag_over_full_depart_norm_Lhat",
    "explained_depart_var_Lhat",
    "true_diag_share_depart_var_Lhat",
    "aniso_per_dim_none",
    "aniso_per_dim_diag",
)

DIRECTION_KEY_FIELDS = (
    "graft_norm_ratio",
    "inner_only_norm_ratio",
    "outer_only_norm_ratio",
    "graft_cos",
    "graft_angle_deg",
    "commonU_norm_ratio",
    "commonU_cos",
    "commonU_angle_deg",
    "realized_D_ratio_vs_muont",
    "dir_over_W_ratio_vs_muont",
    "step_over_W_ratio_vs_muont",
)


def _compact_stat(stat_dict):
    return {k: round(float(stat_dict[k]), 8) for k in ("mean", "median", "std", "min", "max") if k in stat_dict}


def build_summary_from_raw(moment_raw: dict, step_raw: dict) -> dict:
    """Assemble compact mechanism_summary.json from raw moment & step-size diagnostics."""
    s2500 = moment_raw["summary_step_002500_all32"]
    recs_2500 = moment_raw["per_matrix_step_002500"]

    import numpy as np

    moment_geom = OrderedDict()
    for kind in MOMENT_ORDER:
        src = s2500[kind]
        entry = OrderedDict()
        entry["width"] = int(src["width"])
        entry["side"] = "output" if kind.startswith("out_") else "input"
        for f in MOMENT_KEY_FIELDS:
            entry[f] = _compact_stat(src[f])
        # Bulk layers 4..31 mean (excluding L00..L03 token-embedding boundary transient)
        sub_bulk = [r for r in recs_2500 if r["kind"] == kind and r["layer"] >= 4]
        entry["layers_04_31_mean"] = {
            f: round(float(np.mean([r[f] for r in sub_bulk])), 8) for f in MOMENT_KEY_FIELDS
        }
        # Depth quartiles
        sq = moment_raw["summary_step_002500_by_depth_quartile"]
        entry["depth_quartiles_mean"] = {
            q: {
                f: round(float(sq[f"{kind}|{q}"][f]["mean"]), 8)
                for f in (
                    "diag_energy_frac",
                    "offdiag_corr_rms",
                    "offdiag_corr_energy_per_dim",
                    "diag_cv",
                    "corr_eig_max_over_mean",
                    "cos_L",
                    "rel_err_L",
                    "explained_depart_var_L",
                    "rel_err_Lhat",
                    "explained_depart_var_Lhat",
                )
            }
            for q in ("L00_07", "L08_15", "L16_23", "L24_31")
        }
        moment_geom[kind] = entry

    temporal = OrderedDict()
    for step_name in ("step_000500", "step_002500", "step_005120"):
        tsrc = moment_raw["temporal_matched_12layers"][step_name]
        temporal[step_name] = {
            kind: {
                f: {
                    "mean": round(float(tsrc[kind][f]["mean"]), 8),
                    "median": round(float(tsrc[kind][f]["median"]), 8),
                }
                for f in (
                    "diag_energy_frac",
                    "offdiag_corr_rms",
                    "offdiag_corr_energy_per_dim",
                    "diag_cv",
                    "corr_eig_max_over_mean",
                    "cos_L",
                    "rel_err_L",
                    "explained_depart_var_L",
                    "rel_err_Lhat",
                    "explained_depart_var_Lhat",
                )
            }
            for kind in MOMENT_ORDER
        }

    engine_fb = {}
    for step_name, em in moment_raw["meta"]["engine_meta"].items():
        engine_fb[step_name] = {
            "num_ranks": len(em),
            "input_fallbacks": sum(v["input"]["fallbacks"] for v in em.values()),
            "input_extra_roots": sum(v["input"]["extra"] for v in em.values()),
            "output_fallbacks": sum(v["output"]["fallbacks"] for v in em.values()),
            "output_extra_roots": sum(v["output"]["extra"] for v in em.values()),
        }

    dir_by_arm_2500 = OrderedDict()
    for arm, s in step_raw["summary_step_002500_by_arm"].items():
        dir_by_arm_2500[arm] = {f: _compact_stat(s[f]) for f in DIRECTION_KEY_FIELDS}

    dir_by_arm_kind_2500 = OrderedDict()
    for arm in ("wave0/muont", "wave2/inner_outer_mlpdiag", "wave2/inner_outer_leftdiag", "wave1/inner_outer_all"):
        dir_by_arm_kind_2500[arm] = OrderedDict()
        for k in BLOCK_KINDS:
            s = step_raw["summary_step_002500_by_arm_kind"][f"{arm}|{k}"]
            dir_by_arm_kind_2500[arm][k] = {f: _compact_stat(s[f]) for f in DIRECTION_KEY_FIELDS}

    dir_by_arm_5120 = OrderedDict()
    for arm, s in step_raw["summary_step_005120_spot_by_arm"].items():
        dir_by_arm_5120[arm] = {f: _compact_stat(s[f]) for f in DIRECTION_KEY_FIELDS}

    cf_2500 = OrderedDict()
    for k in BLOCK_KINDS:
        s = step_raw["counterfactual_step_002500_by_kind"][k]
        cf_2500[k] = {
            f: _compact_stat(s[f])
            for f in (
                "cos_D_diag_vs_dense",
                "angle_D_diag_vs_dense_deg",
                "rel_err_D_diag_vs_dense",
                "norm_ratio_D_diag_over_dense",
                "cos_D_none_vs_dense",
                "angle_D_none_vs_dense_deg",
                "rel_err_D_none_vs_dense",
                "norm_ratio_D_none_over_dense",
            )
        }

    return OrderedDict([
        ("meta", OrderedDict([
            ("pvc_root", str(PVC)),
            ("moment_run", "wave1/inner_outer_all"),
            ("step_size_arms", [
                "wave0/muont",
                "wave2/inner_outer_mlpdiag",
                "wave2/inner_outer_leftdiag",
                "wave1/inner_outer_all",
            ]),
            ("damping", EPS),
            ("inner_strength", 0.5),
            ("outer_strength", 0.125),
            ("ns_steps", NS_STEPS),
            ("ns_eps", NS_EPS),
            ("ns_mode", "tpu"),
            ("weight_decay", WD),
            ("fp64_eigh_max_abs_diff", max(x["max_abs_diff"] for x in moment_raw["meta"]["fp64_eigh_verification"])),
            ("engine_fallbacks_wave1_dense", engine_fb),
        ])),
        ("moment_geometry_step2500", moment_geom),
        ("moment_temporal_12layers", temporal),
        ("weight_norms", step_raw["weight_norms"]),
        ("direction_step2500_by_arm", dir_by_arm_2500),
        ("direction_step2500_by_arm_kind", dir_by_arm_kind_2500),
        ("direction_step5120_spot_by_arm", dir_by_arm_5120),
        ("counterfactual_post_ns_step2500", cf_2500),
    ])


def recompute_from_pvc() -> dict:
    """Standalone CPU-only recomputation from $PVC checkpoints."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    import numpy as np
    import torch

    torch.set_num_threads(4)
    ckpt_w1 = RUNS / "wave1" / "inner_outer_all" / "checkpoints"

    def _load_moment_rank(step_name: str, rank: int):
        st = torch.load(ckpt_w1 / step_name / f"rank_{rank}.pt", map_location="cpu", weights_only=False)
        fast = st["fast"]
        layers = list(range(4 * rank, 4 * rank + 4))
        meta = {
            side: {
                "fallbacks": int(fast[side]["fallbacks"]),
                "extra": int(fast[side]["extra"]),
            }
            for side in ("input", "output")
        }
        items = []
        C_d = fast["input"]["groups"]["d"]["C"].clone()
        idx = 0
        for layer in layers:
            for k in INPUT_D_KINDS:
                items.append({"layer": layer, "rank": rank, "kind": f"in_{k}", "width": 512, "C": C_d[idx]})
                idx += 1
        C_f = fast["input"]["groups"]["f"]["C"].clone()
        for idx, layer in enumerate(layers):
            items.append({"layer": layer, "rank": rank, "kind": "in_down", "width": 2048, "C": C_f[idx]})
        C_oh = fast["output"]["groups"]["out_hidden"]["C"].clone()
        idx = 0
        for layer in layers:
            for k in HIDDEN_OUT_KINDS:
                items.append({"layer": layer, "rank": rank, "kind": f"out_{k}", "width": 512, "C": C_oh[idx]})
                idx += 1
        C_om = fast["output"]["groups"]["out_mlp"]["C"].clone()
        idx = 0
        for layer in layers:
            for k in MLP_OUT_KINDS:
                items.append({"layer": layer, "rank": rank, "kind": f"out_{k}", "width": 2048, "C": C_om[idx]})
                idx += 1
        del st, fast
        return meta, items

    def _analyze_one(item: dict, fp64: bool = False) -> dict:
        B = item["C"]
        d = B.shape[0]
        B64 = 0.5 * (B.double() + B.double().mT)
        b_diag = B64.diagonal()
        tr_B = float(b_diag.sum().item())
        diag_mean = tr_B / d
        B_frob_sq = float(B64.square().sum().item())
        diag_frob_sq = float(b_diag.square().sum().item())
        diag_energy_frac = diag_frob_sq / B_frob_sq

        inv_sqrt_d = b_diag.clamp_min(1e-300).rsqrt()
        R64 = 0.5 * ((B64 * inv_sqrt_d[:, None] * inv_sqrt_d[None, :]) + (B64 * inv_sqrt_d[:, None] * inv_sqrt_d[None, :]).mT)
        R_off = R64.clone()
        R_off.diagonal().zero_()
        offdiag_sq_sum = float(R_off.square().sum().item())
        offdiag_corr_rms = math.sqrt(offdiag_sq_sum / (d * (d - 1)))
        offdiag_corr_energy_per_dim = offdiag_sq_sum / d

        diag_cv = float(b_diag.std(correction=0).item() / diag_mean)
        diag_max_over_mean = float(b_diag.max().item() / diag_mean)
        B_norm64 = B64 * (d / tr_B)
        if fp64:
            lam_B, V_mat = torch.linalg.eigh(B_norm64)
            lam_B = lam_B.clamp_min(0.0)
            lam_R = torch.linalg.eigvalsh(R64).clamp_min(0.0)
        else:
            lam_B32, V_mat = torch.linalg.eigh(B_norm64.float())
            lam_B = lam_B32.double().clamp_min(0.0)
            lam_R = torch.linalg.eigvalsh(R64.float()).double().clamp_min(0.0)

        eig_max_over_mean = float((lam_B.max() / lam_B.mean()).item())
        eig_B_top1_share = float((lam_B[-1] / lam_B.sum()).item())
        p_B = lam_B / lam_B.sum()
        eig_B_erank_frac = float(torch.exp(-(p_B * p_B.clamp_min(1e-300).log()).sum()).item() / d)
        corr_eig_max_over_mean = float((lam_R.max() / lam_R.mean()).item())
        corr_eig_min = float(lam_R.min().item())
        p_R = lam_R / lam_R.sum()
        corr_erank_frac = float(torch.exp(-(p_R * p_R.clamp_min(1e-300).log()).sum()).item() / d)

        lam_A = (lam_B / lam_B.mean()) + EPS
        a_diag = B_norm64.diagonal().clamp_min(0.0) + EPS
        f_L = lam_A.pow(-0.5)
        l_diag = a_diag.pow(-0.5)
        L_mat = ((V_mat * f_L[None, :]) @ V_mat.mT) if fp64 else ((V_mat * f_L.float()[None, :]) @ V_mat.mT).double()
        L_mat = 0.5 * (L_mat + L_mat.mT)
        norm_L = float(L_mat.norm().item())
        cos_L = float((L_mat.diagonal() * l_diag).sum().item() / (norm_L * float(l_diag.norm().item())))
        L_diff = L_mat.clone()
        L_diff.diagonal().sub_(l_diag)
        norm_diff_L = float(L_diff.norm().item())
        rel_err_L = norm_diff_L / norm_L
        L_minus_I = L_mat.clone()
        L_minus_I.diagonal().sub_(1.0)
        norm_L_minus_I = float(L_minus_I.norm().item())
        rel_err_drop_L = norm_L_minus_I / norm_L
        diag_over_full_depart_norm_L = float((l_diag - 1.0).norm().item()) / max(norm_L_minus_I, 1e-30)
        explained_depart_var_L = 1.0 - (norm_diff_L ** 2) / max(norm_L_minus_I ** 2, 1e-30)
        true_diag_share_depart_var_L = float((L_mat.diagonal() - 1.0).square().sum().item()) / max(norm_L_minus_I ** 2, 1e-30)

        f_raw = lam_A.pow(-0.125)
        f_Lhat = f_raw * (math.sqrt(d) / float(f_raw.norm().item()))
        lh_raw = a_diag.pow(-0.125)
        lhat_diag = lh_raw * (math.sqrt(d) / float(lh_raw.norm().item()))
        Lhat_mat = ((V_mat * f_Lhat[None, :]) @ V_mat.mT) if fp64 else ((V_mat * f_Lhat.float()[None, :]) @ V_mat.mT).double()
        Lhat_mat = 0.5 * (Lhat_mat + Lhat_mat.mT)
        norm_Lhat = float(Lhat_mat.norm().item())
        cos_Lhat = float((Lhat_mat.diagonal() * lhat_diag).sum().item() / (norm_Lhat * float(lhat_diag.norm().item())))
        Lh_diff = Lhat_mat.clone()
        Lh_diff.diagonal().sub_(lhat_diag)
        norm_diff_Lh = float(Lh_diff.norm().item())
        rel_err_Lhat = norm_diff_Lh / norm_Lhat
        Lh_minus_I = Lhat_mat.clone()
        Lh_minus_I.diagonal().sub_(1.0)
        norm_Lh_minus_I = float(Lh_minus_I.norm().item())
        rel_err_drop_Lhat = norm_Lh_minus_I / norm_Lhat
        diag_over_full_depart_norm_Lhat = float((lhat_diag - 1.0).norm().item()) / max(norm_Lh_minus_I, 1e-30)
        explained_depart_var_Lhat = 1.0 - (norm_diff_Lh ** 2) / max(norm_Lh_minus_I ** 2, 1e-30)
        true_diag_share_depart_var_Lhat = float((Lhat_mat.diagonal() - 1.0).square().sum().item()) / max(norm_Lh_minus_I ** 2, 1e-30)

        A64 = B_norm64.clone()
        A64.diagonal().add_(EPS)
        Au = A64 * (d / float(A64.diagonal().sum().item()))
        Au.diagonal().sub_(1.0)
        aniso_per_dim_none = float(Au.square().sum().item() / d)
        M64 = A64 * l_diag[:, None] * l_diag[None, :]
        Mu = M64 * (d / float(M64.diagonal().sum().item()))
        Mu.diagonal().sub_(1.0)
        aniso_per_dim_diag = float(Mu.square().sum().item() / d)

        return {
            "layer": item["layer"],
            "rank": item["rank"],
            "kind": item["kind"],
            "width": d,
            "depth_quartile": f"L{8 * (item['layer'] // 8):02d}_{8 * (item['layer'] // 8) + 7:02d}",
            "diag_energy_frac": diag_energy_frac,
            "offdiag_corr_rms": offdiag_corr_rms,
            "offdiag_corr_energy_per_dim": offdiag_corr_energy_per_dim,
            "diag_cv": diag_cv,
            "diag_max_over_mean": diag_max_over_mean,
            "eig_max_over_mean": eig_max_over_mean,
            "corr_eig_max_over_mean": corr_eig_max_over_mean,
            "eig_B_top1_share": eig_B_top1_share,
            "eig_B_erank_frac": eig_B_erank_frac,
            "corr_erank_frac": corr_erank_frac,
            "corr_eig_min": corr_eig_min,
            "cos_L": cos_L,
            "rel_err_L": rel_err_L,
            "rel_err_drop_L": rel_err_drop_L,
            "diag_over_full_depart_norm_L": diag_over_full_depart_norm_L,
            "explained_depart_var_L": explained_depart_var_L,
            "true_diag_share_depart_var_L": true_diag_share_depart_var_L,
            "cos_Lhat": cos_Lhat,
            "rel_err_Lhat": rel_err_Lhat,
            "rel_err_drop_Lhat": rel_err_drop_Lhat,
            "diag_over_full_depart_norm_Lhat": diag_over_full_depart_norm_Lhat,
            "explained_depart_var_Lhat": explained_depart_var_Lhat,
            "true_diag_share_depart_var_Lhat": true_diag_share_depart_var_Lhat,
            "aniso_per_dim_none": aniso_per_dim_none,
            "aniso_per_dim_diag": aniso_per_dim_diag,
        }

    def _summarize(records, keys):
        buckets = {}
        for r in records:
            buckets.setdefault("|".join(str(r[k]) for k in keys), []).append(r)
        skip = {"layer", "rank", "kind", "width", "depth_quartile", "arm", "step"}
        fields = [k for k, v in records[0].items() if isinstance(v, (int, float)) and k not in skip]
        out = {}
        for kstr, recs in buckets.items():
            s = {"count": len(recs), "width": recs[0].get("width", 0)}
            for f in fields:
                vals = np.array([x[f] for x in recs], dtype=np.float64)
                s[f] = {"mean": float(vals.mean()), "std": float(vals.std()), "median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max())}
            out[kstr] = s
        return out

    def _run_moment_step(step_name, ranks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            loaded = list(pool.map(lambda r: (r, *_load_moment_rank(step_name, r)), ranks))
        em = {f"rank_{r}": m for r, m, _ in loaded}
        items = [it for _, _, its in loaded for it in its]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            recs = list(pool.map(_analyze_one, items))
        fp64_chk = []
        if step_name == "step_002500":
            for it in items:
                if it["layer"] == 16 and it["width"] == 2048:
                    r64 = _analyze_one(it, fp64=True)
                    r32 = next(x for x in recs if x["layer"] == 16 and x["kind"] == it["kind"])
                    fp64_chk.append({"max_abs_diff": max(abs(r64[f] - r32[f]) for f in ("cos_L", "rel_err_L", "cos_Lhat", "rel_err_Lhat", "eig_max_over_mean"))})
        return em, recs, fp64_chk

    em_2500, m_recs_2500, fp64_chk = _run_moment_step("step_002500", range(8))
    spot = (0, 4, 7)
    em_0500, m_recs_0500, _ = _run_moment_step("step_000500", spot)
    em_5120, m_recs_5120, _ = _run_moment_step("step_005120", spot)
    moment_raw = {
        "meta": {"fp64_eigh_verification": fp64_chk, "engine_meta": {"step_000500": em_0500, "step_002500": em_2500, "step_005120": em_5120}},
        "summary_step_002500_all32": _summarize(m_recs_2500, ("kind",)),
        "summary_step_002500_by_depth_quartile": _summarize(m_recs_2500, ("kind", "depth_quartile")),
        "temporal_matched_12layers": {
            "step_000500": _summarize(m_recs_0500, ("kind",)),
            "step_002500": _summarize([x for x in m_recs_2500 if x["rank"] in spot], ("kind",)),
            "step_005120": _summarize(m_recs_5120, ("kind",)),
        },
        "per_matrix_step_002500": m_recs_2500,
    }

    # Step-size & direction rotation recomputation
    def _ns_cpu(u):
        a, b, c = ABC
        x = u.float().unsqueeze(0)
        x = x / (torch.linalg.matrix_norm(x, dim=(-2, -1), keepdim=True) + NS_EPS)
        tr = x.size(-2) > x.size(-1)
        if tr:
            x = x.transpose(-2, -1)
        for _ in range(NS_STEPS):
            xb = x.to(torch.bfloat16).float()
            A = torch.bmm(xb, xb.transpose(-2, -1))
            Ab = A.to(torch.bfloat16).float()
            AA = torch.bmm(Ab, Ab)
            M = (b * A + c * AA).to(torch.bfloat16).float()
            x = a * x + torch.bmm(M, x.to(torch.bfloat16).float())
        return (x.transpose(-2, -1) if tr else x)[0].float()

    def _dense_roots(C):
        d = C.shape[-1]
        Cs = 0.5 * (C.float() + C.float().mT)
        tr = Cs.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).clamp_min(1e-30)
        lam, V = torch.linalg.eigh(Cs * (d / tr.unsqueeze(-1)))
        x = lam.clamp_min(0.0)
        x = x / x.mean(-1, keepdim=True).clamp_min(1e-30)
        Ri = 0.5 * (((V * (x + EPS).pow(-0.5).unsqueeze(-2)) @ V.mT) + ((V * (x + EPS).pow(-0.5).unsqueeze(-2)) @ V.mT).mT)
        Ro = 0.5 * (((V * (x + EPS).pow(-0.125).unsqueeze(-2)) @ V.mT) + ((V * (x + EPS).pow(-0.125).unsqueeze(-2)) @ V.mT).mT)
        return Ri, Ro * (math.sqrt(d) / Ro.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30))

    def _diag_roots(m):
        d = m.shape[-1]
        mf = m.float()
        normed = (mf * (d / mf.sum(-1, keepdim=True).clamp_min(1e-30))).clamp_min(0.0) + EPS
        inner = normed.pow(-0.5)
        outer = normed.pow(-0.125)
        return inner, outer * (math.sqrt(d) / outer.norm(dim=-1, keepdim=True).clamp_min(1e-30))

    def _al(f, mat):
        return mat if f is None else (f[:, None] * mat if f.ndim == 1 else f @ mat)

    def _ar(mat, f):
        return mat if f is None else (mat * f[None, :] if f.ndim == 1 else mat @ f)

    def _fcos(A, B):
        a64, b64 = A.double(), B.double()
        return max(-1.0, min(1.0, float((a64 * b64).sum().item() / (a64.norm().item() * b64.norm().item()))))

    weight_arms = ("wave0/muont", "wave2/inner_outer_mlpdiag", "wave2/inner_outer_leftdiag", "wave1/inner_outer_all", "wave2/inner_outer_hidden", "wave3/null_mlpdiag", "wave1/null_all")
    weight_norms = {}
    for st_name in ("step_000500", "step_002500", "step_005120"):
        ms = {r: {k: v.float().clone() for k, v in torch.load(RUNS / r / "checkpoints" / st_name / "rank_0.pt", map_location="cpu", weights_only=False)["model"].items()} for r in weight_arms}
        bsd = ms["wave0/muont"]
        sr = {}
        for r, sd in ms.items():
            tsq = bsq = dsq = 0.0
            bk = {}
            for k in BLOCK_KINDS:
                ns, bns = [], []
                for l in range(32):
                    w, wb = sd[f"body.blocks.{l}.{k}.weight"].double(), bsd[f"body.blocks.{l}.{k}.weight"].double()
                    wn, wbn = float(w.norm().item()), float(wb.norm().item())
                    tsq += wn * wn
                    bsq += wbn * wbn
                    dsq += float((w - wb).square().sum().item())
                    ns.append(wn)
                    bns.append(wbn)
                bk[k] = {"mean_frob": float(np.mean(ns)), "pct_diff_vs_muont": float(100.0 * (np.mean(ns) / np.mean(bns) - 1.0))}
            sr[r] = {"total_blocks_frob": math.sqrt(tsq), "total_blocks_pct_diff_vs_muont": 100.0 * (math.sqrt(tsq) / math.sqrt(bsq) - 1.0), "total_blocks_rel_l2_dist_vs_muont": math.sqrt(dsq / bsq), "by_kind": bk}
        weight_norms[st_name] = sr

    arms_4 = ("wave0/muont", "wave2/inner_outer_mlpdiag", "wave2/inner_outer_leftdiag", "wave1/inner_outer_all")

    def _dir_step(st_name, ranks):
        ms = {r: {k: v.float().clone() for k, v in torch.load(RUNS / r / "checkpoints" / st_name / "rank_0.pt", map_location="cpu", weights_only=False)["model"].items()} for r in arms_4}
        recs, cfs = [], []
        for rank in ranks:
            layers = list(range(4 * rank, 4 * rank + 4))
            sts = {}
            for r in arms_4:
                ck = torch.load(RUNS / r / "checkpoints" / st_name / f"rank_{rank}.pt", map_location="cpu", weights_only=False)["fast"]
                mom, right, left, raw_c = {}, {}, {}, {}
                if r == "wave0/muont":
                    m512, mmlp, mdn = ck["momentum"][(512, 512)]["m"].float(), ck["momentum"][(2048, 512)]["m"].float(), ck["momentum"][(512, 2048)]["m"].float()
                    for li, l in enumerate(layers):
                        s5 = m512[4 * li : 4 * (li + 1)][FT_512_TO_BT]
                        for ki, k in enumerate(("q", "k", "v", "o")): mom[(l, k)] = s5[ki].clone()
                        for ki, k in enumerate(("gate", "up")): mom[(l, k)] = mmlp[2 * li + ki].clone()
                        mom[(l, "down")] = mdn[li].clone()
                    ie = ck
                else:
                    m512, mmlp, mdn = ck["momentum"][(512, 512)].float(), ck["momentum"][(2048, 512)].float(), ck["momentum"][(512, 2048)].float()
                    for li, l in enumerate(layers):
                        for ki, k in enumerate(("q", "k", "v", "o")): mom[(l, k)] = m512[4 * li + ki].clone()
                        for ki, k in enumerate(("gate", "up")): mom[(l, k)] = mmlp[2 * li + ki].clone()
                        mom[(l, "down")] = mdn[li].clone()
                    ie = ck["input"]
                    if ck["output"] is not None:
                        for gn, ks in (("out_hidden", HIDDEN_OUT_KINDS), ("out_mlp", MLP_OUT_KINDS)):
                            if gn in ck["output"]["groups"]:
                                Co = ck["output"]["groups"][gn]["C"].float().clone()
                                Li, Lo = _dense_roots(Co)
                                idx = 0
                                for l in layers:
                                    for k in ks:
                                        left[(l, k)] = (Li[idx], Lo[idx])
                                        raw_c[(l, k)] = Co[idx]
                                        idx += 1
                    if ck["output_diagonal"] is not None:
                        om = ck["output_diagonal"]["moment"]
                        if 512 in om:
                            Li, Lo = _diag_roots(om[512][20 * rank : 20 * (rank + 1)])
                            idx = 0
                            for l in layers:
                                for k in HIDDEN_OUT_KINDS:
                                    left[(l, k)] = (Li[idx], Lo[idx]); idx += 1
                        if 2048 in om:
                            Li, Lo = _diag_roots(om[2048][8 * rank : 8 * (rank + 1)])
                            idx = 0
                            for l in layers:
                                for k in MLP_OUT_KINDS:
                                    left[(l, k)] = (Li[idx], Lo[idx]); idx += 1
                Rid, Rod = _dense_roots(ie["groups"]["d"]["C"])
                Rif, Rof = _dense_roots(ie["groups"]["f"]["C"])
                idx = 0
                for l in layers:
                    for ik in INPUT_D_KINDS:
                        right[(l, ik)] = (Rid[idx], Rod[idx]); idx += 1
                for li, l in enumerate(layers):
                    right[(l, "down")] = (Rif[li], Rof[li])
                sts[r] = {"mom": mom, "right": right, "left": left, "raw_c": raw_c}
            bst = sts["wave0/muont"]
            for l in layers:
                for k in BLOCK_KINDS:
                    ik = INPUT_KIND_MAP[k]
                    Ub = bst["mom"][(l, k)]
                    Rib, Rob = bst["right"][(l, ik)]
                    Db = _ar(_ns_cpu(_ar(Ub, Rib)), Rob)
                    nDb = float(Db.double().norm().item())
                    Wb = ms["wave0/muont"][f"body.blocks.{l}.{k}.weight"]
                    sc = max(1.0, Wb.shape[0] / Wb.shape[1]) ** 0.5
                    nWb = float(Wb.double().norm().item())
                    dWb = sc * nDb / nWb
                    sWb = float((sc * Db + WD * _ar(_ar(Wb, Rob), Rob)).double().norm().item()) / nWb
                    for r in arms_4:
                        st = sts[r]
                        Ua = st["mom"][(l, k)]
                        Ria, Roa = st["right"][(l, ik)]
                        Lia, Loa = st["left"].get((l, k), (None, None))
                        Wa = ms[r][f"body.blocks.{l}.{k}.weight"]
                        nWa = float(Wa.double().norm().item())
                        Din = _ar(_ns_cpu(_ar(Ua, Ria)), Roa)
                        nDin = float(Din.double().norm().item())
                        Da = Din if r == "wave0/muont" else _al(Loa, _ar(_ns_cpu(_al(Lia, _ar(Ua, Ria))), Roa))
                        Dinn = Din if r == "wave0/muont" else _ar(_ns_cpu(_al(Lia, _ar(Ua, Ria))), Roa)
                        Dout = Din if r == "wave0/muont" else _al(Loa, _ar(_ns_cpu(_ar(Ua, Ria)), Roa))
                        Dcu = Db if r == "wave0/muont" else _al(Loa, _ar(_ns_cpu(_al(Lia, _ar(Ub, Rib))), Rob))
                        nDa = float(Da.double().norm().item())
                        gc = _fcos(Da, Din)
                        cc = _fcos(Dcu, Db)
                        recs.append({
                            "arm": r, "kind": k, "layer": l, "rank": rank,
                            "graft_norm_ratio": nDa / nDin,
                            "inner_only_norm_ratio": float(Dinn.double().norm().item()) / nDin,
                            "outer_only_norm_ratio": float(Dout.double().norm().item()) / nDin,
                            "graft_cos": gc, "graft_angle_deg": math.degrees(math.acos(gc)),
                            "commonU_norm_ratio": float(Dcu.double().norm().item()) / nDb,
                            "commonU_cos": cc, "commonU_angle_deg": math.degrees(math.acos(cc)),
                            "realized_D_ratio_vs_muont": nDa / nDb,
                            "dir_over_W_ratio_vs_muont": (sc * nDa / nWa) / dWb,
                            "step_over_W_ratio_vs_muont": (float((sc * Da + WD * _ar(_ar(Wa, Roa), Roa)).double().norm().item()) / nWa) / sWb,
                        })
                    if st_name == "step_002500":
                        sa = sts["wave1/inner_outer_all"]
                        Ua = sa["mom"][(l, k)]
                        Ria, Roa = sa["right"][(l, ik)]
                        Lid, Lod = sa["left"][(l, k)]
                        Ldg, Lodg = _diag_roots(sa["raw_c"][(l, k)].diagonal().unsqueeze(0))
                        Ddns = _al(Lod, _ar(_ns_cpu(_al(Lid, _ar(Ua, Ria))), Roa))
                        Ddg = _al(Lodg[0], _ar(_ns_cpu(_al(Ldg[0], _ar(Ua, Ria))), Roa))
                        Dn = _ar(_ns_cpu(_ar(Ua, Ria)), Roa)
                        ndns = float(Ddns.double().norm().item())
                        cd = _fcos(Ddg, Ddns)
                        cn = _fcos(Dn, Ddns)
                        cfs.append({
                            "kind": k, "layer": l, "rank": rank,
                            "cos_D_diag_vs_dense": cd, "angle_D_diag_vs_dense_deg": math.degrees(math.acos(cd)),
                            "rel_err_D_diag_vs_dense": float((Ddg - Ddns).double().norm().item()) / ndns,
                            "norm_ratio_D_diag_over_dense": float(Ddg.double().norm().item()) / ndns,
                            "cos_D_none_vs_dense": cn, "angle_D_none_vs_dense_deg": math.degrees(math.acos(cn)),
                            "rel_err_D_none_vs_dense": float((Dn - Ddns).double().norm().item()) / ndns,
                            "norm_ratio_D_none_over_dense": float(Dn.double().norm().item()) / ndns,
                        })
        return recs, cfs

    d_recs_2500, cf_2500 = _dir_step("step_002500", range(8))
    d_recs_5120, _ = _dir_step("step_005120", spot)
    step_raw = {
        "weight_norms": weight_norms,
        "summary_step_002500_by_arm": _summarize(d_recs_2500, ("arm",)),
        "summary_step_002500_by_arm_kind": _summarize(d_recs_2500, ("arm", "kind")),
        "summary_step_005120_spot_by_arm": _summarize(d_recs_5120, ("arm",)),
        "counterfactual_step_002500_by_kind": _summarize(cf_2500, ("kind",)),
    }
    return build_summary_from_raw(moment_raw, step_raw)


def write_outputs(summary: dict):
    mg = summary["moment_geometry_step2500"]
    d_arm = summary["direction_step2500_by_arm"]
    d_ak = summary["direction_step2500_by_arm_kind"]
    d_5120 = summary["direction_step5120_spot_by_arm"]
    wn = summary["weight_norms"]
    cf = summary["counterfactual_post_ns_step2500"]

    # 1. Write results/mechanism_numbers.tex (strictly letters-only TeX macro names)
    macros = ["% Generated by scripts/analyze_mechanism.py. Do not edit.\n"]

    def mac(name: str, val: str):
        assert name.isalpha(), f"Macro name must contain letters only: {name}"
        macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # Width-2048 output moments (gate, up) vs input & hidden moments
    for key, tag in (
        ("out_gate", "OutGate"),
        ("out_up", "OutUp"),
        ("out_q", "OutQ"),
        ("out_k", "OutK"),
        ("out_v", "OutV"),
        ("out_o", "OutO"),
        ("out_down", "OutDown"),
        ("in_down", "InDown"),
        ("in_attn", "InAttn"),
        ("in_o", "InO"),
        ("in_mlp", "InMlp"),
    ):
        s = mg[key]
        mac(f"mech{tag}DiagEnergyPct", f"{100.0 * s['diag_energy_frac']['mean']:.1f}")
        mac(f"mech{tag}DiagEnergyMedPct", f"{100.0 * s['diag_energy_frac']['median']:.1f}")
        mac(f"mech{tag}CorrRms", f"{s['offdiag_corr_rms']['mean']:.4f}")
        mac(f"mech{tag}CorrRmsMed", f"{s['offdiag_corr_rms']['median']:.4f}")
        mac(f"mech{tag}CorrEnergy", f"{s['offdiag_corr_energy_per_dim']['mean']:.2f}")
        mac(f"mech{tag}CorrEnergyMed", f"{s['offdiag_corr_energy_per_dim']['median']:.3f}")
        mac(f"mech{tag}DiagCv", f"{s['diag_cv']['mean']:.2f}")
        mac(f"mech{tag}DiagMaxMean", f"{s['diag_max_over_mean']['mean']:.1f}")
        mac(f"mech{tag}EigMaxMean", f"{s['eig_max_over_mean']['mean']:.1f}")
        mac(f"mech{tag}CorrEigMax", f"{s['corr_eig_max_over_mean']['mean']:.1f}")
        mac(f"mech{tag}CorrEigMaxMed", f"{s['corr_eig_max_over_mean']['median']:.2f}")
        mac(f"mech{tag}CosL", f"{s['cos_L']['mean']:.4f}")
        mac(f"mech{tag}CosLMed", f"{s['cos_L']['median']:.4f}")
        mac(f"mech{tag}RelErrL", f"{s['rel_err_L']['mean']:.3f}")
        mac(f"mech{tag}DropErrL", f"{s['rel_err_drop_L']['mean']:.3f}")
        mac(f"mech{tag}ExpVarLPct", f"{100.0 * s['explained_depart_var_L']['mean']:.1f}")
        mac(f"mech{tag}ExpVarLMedPct", f"{100.0 * s['explained_depart_var_L']['median']:.1f}")
        mac(f"mech{tag}CosLhat", f"{s['cos_Lhat']['mean']:.4f}")
        mac(f"mech{tag}RelErrLhat", f"{s['rel_err_Lhat']['mean']:.3f}")
        mac(f"mech{tag}ExpVarLhatPct", f"{100.0 * s['explained_depart_var_Lhat']['mean']:.1f}")
        mac(f"mech{tag}ExpVarLhatMedPct", f"{100.0 * s['explained_depart_var_Lhat']['median']:.1f}")

    # Bulk Layers 4..31 means for out_gate & out_up
    mac("mechOutGateBulkCorrRms", f"{mg['out_gate']['layers_04_31_mean']['offdiag_corr_rms']:.4f}")
    mac("mechOutUpBulkCorrRms", f"{mg['out_up']['layers_04_31_mean']['offdiag_corr_rms']:.4f}")
    mac("mechOutGateBulkExpVarLPct", f"{100.0 * mg['out_gate']['layers_04_31_mean']['explained_depart_var_L']:.1f}")
    mac("mechOutUpBulkExpVarLPct", f"{100.0 * mg['out_up']['layers_04_31_mean']['explained_depart_var_L']:.1f}")

    # Step-size & direction-rotation macros
    for run, tag in (
        ("wave2/inner_outer_mlpdiag", "Mlpdiag"),
        ("wave2/inner_outer_leftdiag", "Leftdiag"),
        ("wave1/inner_outer_all", "Dense"),
    ):
        s = d_arm[run]
        s5 = d_5120[run]
        mac(f"mech{tag}GraftNorm", f"{s['graft_norm_ratio']['mean']:.4f}")
        mac(f"mech{tag}GraftNormPct", f"{100.0 * (s['graft_norm_ratio']['mean'] - 1.0):+.2f}")
        mac(f"mech{tag}GraftCos", f"{s['graft_cos']['mean']:.4f}")
        mac(f"mech{tag}GraftDeg", f"{s['graft_angle_deg']['mean']:.1f}")
        mac(f"mech{tag}RealDirNorm", f"{s['realized_D_ratio_vs_muont']['mean']:.4f}")
        mac(f"mech{tag}StepOverWRatio", f"{s['step_over_W_ratio_vs_muont']['mean']:.4f}")
        mac(f"mech{tag}EndGraftNorm", f"{s5['graft_norm_ratio']['mean']:.4f}")
        mac(f"mech{tag}EndGraftDeg", f"{s5['graft_angle_deg']['mean']:.1f}")
        mac(f"mech{tag}WeightDiffMidPct", f"{wn['step_002500'][run]['total_blocks_pct_diff_vs_muont']:+.2f}")
        mac(f"mech{tag}WeightDiffEndPct", f"{wn['step_005120'][run]['total_blocks_pct_diff_vs_muont']:+.2f}")

    mac("mechNullMlpWeightDiffEndPct", f"{wn['step_005120']['wave3/null_mlpdiag']['total_blocks_pct_diff_vs_muont']:+.2f}")
    mac("mechLeftdiagOutODeg", f"{d_ak['wave2/inner_outer_leftdiag']['o']['graft_angle_deg']['mean']:.1f}")
    mac("mechLeftdiagOutDownDeg", f"{d_ak['wave2/inner_outer_leftdiag']['down']['graft_angle_deg']['mean']:.1f}")
    mac("mechMlpdiagOutODeg", f"{d_ak['wave2/inner_outer_mlpdiag']['o']['graft_angle_deg']['mean']:.1f}")
    mac("mechMlpdiagOutDownDeg", f"{d_ak['wave2/inner_outer_mlpdiag']['down']['graft_angle_deg']['mean']:.1f}")
    mac("mechCfGateCosDiagDense", f"{cf['gate']['cos_D_diag_vs_dense']['mean']:.4f}")
    mac("mechCfGateCosDiagDenseMed", f"{cf['gate']['cos_D_diag_vs_dense']['median']:.4f}")
    mac("mechCfUpCosDiagDense", f"{cf['up']['cos_D_diag_vs_dense']['mean']:.4f}")
    mac("mechCfUpCosDiagDenseMed", f"{cf['up']['cos_D_diag_vs_dense']['median']:.4f}")

    (HERE / "results" / "mechanism_numbers.tex").write_text("\n".join(macros) + "\n")

    # 2. Write results/table_mechanism.tex (complete tabular environment)
    TABLE_LABELS = OrderedDict([
        ("out_gate", r"$\Out$: \texttt{gate}"),
        ("out_up", r"$\Out$: \texttt{up}"),
        ("__sep1__", None),
        ("out_q", r"$\Out$: \texttt{q}"),
        ("out_k", r"$\Out$: \texttt{k}"),
        ("out_v", r"$\Out$: \texttt{v}"),
        ("out_o", r"$\Out$: \texttt{o}"),
        ("out_down", r"$\Out$: \texttt{down}"),
        ("__sep2__", None),
        ("in_down", r"$\In$: \texttt{down}"),
        ("in_attn", r"$\In$: \texttt{attn}"),
        ("in_o", r"$\In$: \texttt{o}"),
        ("in_mlp", r"$\In$: \texttt{mlp}"),
    ])

    tlines = [
        "% Generated by scripts/analyze_mechanism.py. Do not edit.",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Moment & $d$ & Diag.\ \% & $\mathrm{CV}(D)$ & $\lambda_{\max}/\bar\lambda$ & "
        r"$\lambda_{\max}(R)$ & $\cos(L,L_{\diag})$ & $\mathrm{err}(L_{\diag})$ & "
        r"$\mathrm{err}(I)$ & $R^2(L)$ \% & $R^2(\hat L)$ \% \\",
        r"\midrule",
    ]
    for key, label in TABLE_LABELS.items():
        if label is None:
            tlines.append(r"\addlinespace")
            continue
        s = mg[key]
        bold = key in ("out_gate", "out_up")
        exp_l = f"{100.0 * s['explained_depart_var_L']['mean']:.1f}"
        exp_lh = f"{100.0 * s['explained_depart_var_Lhat']['mean']:.1f}"
        exp_l_cell = rf"\textbf{{${exp_l}$}}" if bold else f"${exp_l}$"
        exp_lh_cell = rf"\textbf{{${exp_lh}$}}" if bold else f"${exp_lh}$"
        tlines.append(
            f"{label} & {s['width']} & ${100.0 * s['diag_energy_frac']['mean']:.1f}$ & "
            f"${s['diag_cv']['mean']:.2f}$ & ${s['eig_max_over_mean']['mean']:.0f}$ & "
            f"${s['corr_eig_max_over_mean']['mean']:.1f}$ & ${s['cos_L']['mean']:.4f}$ & "
            f"${s['rel_err_L']['mean']:.3f}$ & ${s['rel_err_drop_L']['mean']:.3f}$ & "
            f"{exp_l_cell} & {exp_lh_cell} \\\\"
        )
    tlines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "results" / "table_mechanism.tex").write_text("\n".join(tlines) + "\n")

    # 3. Write figs/mechanism_bars.dat
    header = [
        "idx",
        "kind",
        "side",
        "width",
        "diag_energy_pct",
        "diag_energy_med_pct",
        "corr_rms_med",
        "corr_energy_med",
        "diag_cv",
        "eig_max_mean",
        "corr_eig_max",
        "corr_eig_max_med",
        "cos_L",
        "rel_err_L",
        "drop_err_L",
        "exp_var_L_pct",
        "exp_var_Lhat_pct",
        "graft_norm_mlpdiag",
        "graft_deg_mlpdiag",
    ]
    dat_lines = [" ".join(header)]
    for idx, kind in enumerate(MOMENT_ORDER):
        s = mg[kind]
        proj = kind.split("_", 1)[1]
        if kind.startswith("out_") and proj in d_ak["wave2/inner_outer_mlpdiag"]:
            gn = f"{d_ak['wave2/inner_outer_mlpdiag'][proj]['graft_norm_ratio']['mean']:.6f}"
            gd = f"{d_ak['wave2/inner_outer_mlpdiag'][proj]['graft_angle_deg']['mean']:.4f}"
        else:
            gn, gd = "nan", "nan"
        row = [
            str(idx),
            kind,
            s["side"],
            str(s["width"]),
            f"{100.0 * s['diag_energy_frac']['mean']:.4f}",
            f"{100.0 * s['diag_energy_frac']['median']:.4f}",
            f"{s['offdiag_corr_rms']['median']:.6f}",
            f"{s['offdiag_corr_energy_per_dim']['median']:.6f}",
            f"{s['diag_cv']['mean']:.4f}",
            f"{s['eig_max_over_mean']['mean']:.4f}",
            f"{s['corr_eig_max_over_mean']['mean']:.4f}",
            f"{s['corr_eig_max_over_mean']['median']:.4f}",
            f"{s['cos_L']['mean']:.6f}",
            f"{s['rel_err_L']['mean']:.6f}",
            f"{s['rel_err_drop_L']['mean']:.6f}",
            f"{100.0 * s['explained_depart_var_L']['mean']:.4f}",
            f"{100.0 * s['explained_depart_var_Lhat']['mean']:.4f}",
            gn,
            gd,
        ]
        dat_lines.append(" ".join(row))
    (HERE / "figs" / "mechanism_bars.dat").write_text("\n".join(dat_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="Recompute summary from PVC checkpoints on CPU.")
    parser.add_argument("--from-scratch-dir", type=Path, help="Build mechanism_summary.json from raw JSONs in DIR.")
    args = parser.parse_args()

    summary_path = HERE / "results" / "mechanism_summary.json"

    if args.from_scratch_dir is not None:
        moment_raw = json.loads((args.from_scratch_dir / "moment_diagnostics.json").read_text())
        step_raw = json.loads((args.from_scratch_dir / "step_size_diagnostics.json").read_text())
        summary = build_summary_from_raw(moment_raw, step_raw)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    elif args.recompute:
        summary = recompute_from_pvc()
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    else:
        summary = json.loads(summary_path.read_text())

    write_outputs(summary)

    # Console report matching scripts/analyze.py style
    print(f"{'moment':12s} {'d':>5s} {'diagE%':>7s} {'diagCV':>7s} {'cMaxR':>7s} "
          f"{'cosL':>7s} {'errL':>7s} {'errI':>7s} {'R2_L%':>7s} {'R2_Lh%':>8s}")
    for k in MOMENT_ORDER:
        s = summary["moment_geometry_step2500"][k]
        print(
            f"{k:12s} {s['width']:5d} {100*s['diag_energy_frac']['mean']:7.2f} "
            f"{s['diag_cv']['mean']:7.2f} {s['corr_eig_max_over_mean']['mean']:7.2f} "
            f"{s['cos_L']['mean']:7.4f} {s['rel_err_L']['mean']:7.3f} "
            f"{s['rel_err_drop_L']['mean']:7.3f} "
            f"{100*s['explained_depart_var_L']['mean']:7.2f} "
            f"{100*s['explained_depart_var_Lhat']['mean']:8.2f}"
        )
    print("\nStep-size & rotation summary (step 2500, all 224 block matrices):")
    for arm, s in summary["direction_step2500_by_arm"].items():
        print(
            f"  {arm:28s} graft_norm={s['graft_norm_ratio']['mean']:.4f} "
            f"graft_deg={s['graft_angle_deg']['mean']:5.2f}deg "
            f"real_D_ratio={s['realized_D_ratio_vs_muont']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
