"""Build motivating covariance, trajectory, and post-training step-geometry artifacts.

By default, reads the compact checkpoint-derived summary in
`results/mechanism_summary.json` and emits:

  results/motivation_numbers.tex  LaTeX macros for Section 3 (wave1/null_all 6+1 & trajectory)
  results/mechanism_numbers.tex   LaTeX macros for post-training step-norm & active checks
  results/table_motivation.tex    Tabular of 6+1 motivating metrics across all 11 moments
  results/table_mechanism.tex     Tabular of active-run covariance checks
  figs/motivation_bars.dat        pgfplots data table for 11-moment anisotropy & fidelity bars
  figs/motivation_traj.dat        pgfplots data table for 8-checkpoint trajectory evolution
  figs/mechanism_bars.dat         pgfplots data table for post-training step rotation

Pass `--from-scratch-dir DIR` to assemble `results/mechanism_summary.json` from
precomputed `motivation_null_all_diagnostics.json`, `moment_diagnostics.json`,
and `step_size_diagnostics.json`, or `--recompute` to recompute on CPU
(`CUDA_VISIBLE_DEVICES=""`) directly from the saved `$PVC` checkpoints.

    python -I scripts/analyze_mechanism.py
"""

import argparse
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
BATCH_TOKENS = 524_288
SAMPLES_PER_STEP = 36_608

BLOCK_KINDS = ("q", "k", "v", "o", "gate", "up", "down")
MOTIVATION_ORDER = (
    "in_attn",
    "in_o",
    "in_mlp",
    "in_down",
    "out_q",
    "out_k",
    "out_v",
    "out_o",
    "out_down",
    "out_gate",
    "out_up",
)
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
TRAJ_STEPS = (250, 500, 1000, 1750, 2500, 3500, 4500, 5120)

MOTIVATION_KEY_FIELDS = (
    "erank_pct",
    "corr_erank_pct",
    "diag_cv",
    "diag_max_over_mean",
    "offdiag_corr_rms",
    "offdiag_corr_energy_per_dim",
    "diag_energy_pct",
    "diag_aniso_pct",
    "exp_var_L_pct",
    "exp_var_Lhat_pct",
    "eig_max_over_mean",
    "corr_eig_max_over_mean",
    "cos_L",
    "rel_err_L",
    "rel_err_drop_L",
    "cos_Lhat",
    "rel_err_Lhat",
    "cached_rho",
    "cached_est2_over_tol",
    "cached_offdiag_frob_rel",
    "short_drift_top16",
    "short_drift_top32",
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
    return {
        k: round(float(stat_dict[k]), 8)
        for k in ("mean", "median", "std", "min", "max", "q25", "q75")
        if k in stat_dict
    }


def _fmt_int_commas(n: int) -> str:
    return f"{int(round(n)):,}".replace(",", "{,}")


def build_summary_from_raw(mot_raw: dict, moment_raw: dict, step_raw: dict) -> dict:
    """Assemble compact results/mechanism_summary.json from raw diagnostics."""
    import numpy as np

    # 1. Motivating analysis on wave1/null_all (pure baseline MuonT trajectory)
    s25_null = mot_raw["all_steps_summary"]["step_002500"]
    mot_step2500 = OrderedDict()
    for kind in MOTIVATION_ORDER:
        src = s25_null[kind]
        entry = OrderedDict()
        entry["width"] = int(src["width"])
        entry["side"] = src["side"]
        for f in MOTIVATION_KEY_FIELDS:
            entry[f] = _compact_stat(src[f])
        entry["layers_04_31_mean"] = {
            f: round(float(src["layers_04_31_mean"][f]), 8) for f in MOTIVATION_KEY_FIELDS
        }
        entry["layers_04_31_median"] = {
            f: round(float(src["layers_04_31_median"][f]), 8) for f in MOTIVATION_KEY_FIELDS
        }
        mot_step2500[kind] = entry

    mot_traj = OrderedDict()
    traj_fields = (
        "erank_pct",
        "diag_cv",
        "diag_max_over_mean",
        "offdiag_corr_rms",
        "diag_energy_pct",
        "diag_aniso_pct",
        "exp_var_L_pct",
        "exp_var_Lhat_pct",
        "cos_L",
    )
    for st in TRAJ_STEPS:
        sname = f"step_{st:06d}"
        tsrc = mot_raw["trajectory_8steps_summary"][sname]
        mot_traj[sname] = OrderedDict()
        mot_traj[sname]["step"] = st
        mot_traj[sname]["tokens_B"] = round(st * BATCH_TOKENS / 1e9, 6)
        for kind in MOTIVATION_ORDER:
            mot_traj[sname][kind] = {
                f: {
                    "mean": round(float(tsrc[kind][f]["mean"]), 8),
                    "median": round(float(tsrc[kind][f]["median"]), 8),
                }
                for f in traj_fields
            }

    mot_traj_avg = OrderedDict()
    for kind in MOTIVATION_ORDER:
        ta_src = mot_raw["trajectory_averaged_8steps"][kind]
        mot_traj_avg[kind] = {
            "width": int(ta_src["width"]),
            "side": ta_src["side"],
        }
        for f in traj_fields:
            mot_traj_avg[kind][f] = {
                k: round(float(v), 8) for k, v in ta_src[f].items()
            }

    mot_drift = OrderedDict()
    for pair_key, gd_src in mot_raw["grassmann_drift_250steps"].items():
        mot_drift[pair_key] = OrderedDict()
        for kind in MOTIVATION_ORDER:
            mot_drift[pair_key][kind] = {
                f: _compact_stat(gd_src[kind][f])
                for f in ("drift_top16", "drift_top32", "drift_top64", "drift_bot16", "drift_bot32")
            }

    fb_8rank = mot_raw["fallback_totals_8rank_null_all"]
    fb_250 = fb_8rank["step_000250"]
    fb_5120 = fb_8rank["step_005120"]
    post_in = fb_5120["input_fallbacks_8rank"] - fb_250["input_fallbacks_8rank"]
    post_out = fb_5120["output_fallbacks_8rank"] - fb_250["output_fallbacks_8rank"]
    post_steps = 5120 - 250
    gd_mid = mot_drift["step_002250_to_step_002500"]
    derived_mot = OrderedDict([
        ("finite_sample_corr_floor", round(1.0 / math.sqrt(SAMPLES_PER_STEP), 8)),
        ("post250_steps", post_steps),
        ("post250_input_fallbacks_8rank", post_in),
        ("post250_output_fallbacks_8rank", post_out),
        ("post250_input_fb_per_update", round(post_in / post_steps, 8)),
        ("post250_output_fb_per_update", round(post_out / post_steps, 8)),
        ("post250_input_fb_pct_per_stat_step", round(100.0 * post_in / (post_steps * 128.0), 8)),
        ("post250_output_fb_pct_per_stat_step", round(100.0 * post_out / (post_steps * 224.0), 8)),
        ("post250_raw_out_over_in_ratio", round(post_out / max(1, post_in), 6)),
        ("post250_per_stat_out_over_in_ratio", round((post_out / 224.0) / (post_in / 128.0), 6)),
        ("rank0_raw_out_over_in_ratio", round(
            mot_raw["rank0_jsonl_fallbacks_by_arm"]["wave1/null_all"]["rank0_output_fallbacks"]
            / max(1, mot_raw["rank0_jsonl_fallbacks_by_arm"]["wave1/null_all"]["rank0_input_fallbacks"]),
            6,
        )),
        ("drift_top16_ratio_out_v_over_in_attn", round(
            gd_mid["out_v"]["drift_top16"]["mean"] / gd_mid["in_attn"]["drift_top16"]["mean"], 6
        )),
        ("drift_top16_ratio_out_gate_over_in_attn", round(
            gd_mid["out_gate"]["drift_top16"]["mean"] / gd_mid["in_attn"]["drift_top16"]["mean"], 6
        )),
        ("drift_top16_ratio_out_up_over_in_attn", round(
            gd_mid["out_up"]["drift_top16"]["mean"] / gd_mid["in_attn"]["drift_top16"]["mean"], 6
        )),
        ("mlpdiag_rank0_fb_reduction_pct_vs_null", round(
            100.0 * (
                1.0
                - mot_raw["rank0_jsonl_fallbacks_by_arm"]["wave2/inner_outer_mlpdiag"]["rank0_output_fallbacks"]
                / mot_raw["rank0_jsonl_fallbacks_by_arm"]["wave1/null_all"]["rank0_output_fallbacks"]
            ),
            6,
        )),
    ])

    invariance_2500 = OrderedDict()
    for kind in MOTIVATION_ORDER:
        inv_src = mot_raw["invariance_null_vs_dense_step_002500"][kind]
        invariance_2500[kind] = {
            f: {k: round(float(v), 8) for k, v in inv_src[f].items()}
            for f in ("erank_pct", "diag_cv", "offdiag_corr_rms", "diag_energy_pct", "diag_aniso_pct", "exp_var_L_pct")
        }

    # 2. Active-run moment_geometry_step2500 (wave1/inner_outer_all) + step-size & norm-grafting
    s2500 = moment_raw["summary_step_002500_all32"]
    recs_2500 = moment_raw["per_matrix_step_002500"]
    moment_geom = OrderedDict()
    for kind in MOMENT_ORDER:
        src = s2500[kind]
        entry = OrderedDict()
        entry["width"] = int(src["width"])
        entry["side"] = "output" if kind.startswith("out_") else "input"
        for f in MOMENT_KEY_FIELDS:
            entry[f] = _compact_stat(src[f])
        # Also include explicit percentages so check_numbers.py sees them
        entry["diag_energy_pct"] = {k: round(100.0 * v, 6) for k, v in entry["diag_energy_frac"].items()}
        entry["exp_var_L_pct"] = {k: round(100.0 * v, 6) for k, v in entry["explained_depart_var_L"].items()}
        entry["exp_var_Lhat_pct"] = {k: round(100.0 * v, 6) for k, v in entry["explained_depart_var_Lhat"].items()}
        sub_bulk = [r for r in recs_2500 if r["kind"] == kind and r["layer"] >= 4]
        entry["layers_04_31_mean"] = {
            f: round(float(np.mean([r[f] for r in sub_bulk])), 8) for f in MOMENT_KEY_FIELDS
        }
        entry["layers_04_31_mean"]["exp_var_L_pct"] = round(
            100.0 * entry["layers_04_31_mean"]["explained_depart_var_L"], 6
        )
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
        dir_by_arm_2500[arm]["graft_norm_pct_diff"] = round(
            100.0 * (s["graft_norm_ratio"]["mean"] - 1.0), 6
        )

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
            ("motivating_run", "wave1/null_all"),
            ("active_crosscheck_run", "wave1/inner_outer_all"),
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
            ("traj_steps", list(TRAJ_STEPS)),
            ("pythagorean_max_rel_err_step2500", mot_raw["meta"]["pythagorean_max_rel_err_step2500"]),
            ("fp64_eigh_max_abs_diff_null", max(x["max_abs_diff"] for x in mot_raw["meta"]["fp64_eigh_verification"])),
            ("fp64_eigh_max_abs_diff", max(x["max_abs_diff"] for x in moment_raw["meta"]["fp64_eigh_verification"])),
            ("engine_fallbacks_wave1_dense", engine_fb),
        ])),
        ("motivation_null_all", OrderedDict([
            ("derived", derived_mot),
            ("fallback_totals_8rank", fb_8rank),
            ("rank0_jsonl_fallbacks_by_arm", mot_raw["rank0_jsonl_fallbacks_by_arm"]),
            ("step_002500", mot_step2500),
            ("trajectory_8steps", mot_traj),
            ("trajectory_averaged_8steps", mot_traj_avg),
            ("grassmann_drift_250steps", mot_drift),
            ("invariance_vs_dense_step2500", invariance_2500),
        ])),
        ("moment_geometry_step2500", moment_geom),
        ("moment_temporal_12layers", temporal),
        ("weight_norms", step_raw["weight_norms"]),
        ("direction_step2500_by_arm", dir_by_arm_2500),
        ("direction_step2500_by_arm_kind", dir_by_arm_kind_2500),
        ("direction_step5120_spot_by_arm", dir_by_arm_5120),
        ("counterfactual_post_ns_step2500", cf_2500),
    ])


def write_outputs(summary: dict):
    mot = summary["motivation_null_all"]
    m25 = mot["step_002500"]
    mtraj = mot["trajectory_8steps"]
    mavg = mot["trajectory_averaged_8steps"]
    mdrift = mot["grassmann_drift_250steps"]["step_002250_to_step_002500"]
    mdrift_early = mot["grassmann_drift_250steps"]["step_000250_to_step_000500"]
    mdrift_late = mot["grassmann_drift_250steps"]["step_004250_to_step_004500"]
    mfb = mot["fallback_totals_8rank"]["step_005120"]
    mfb_r0 = mot["rank0_jsonl_fallbacks_by_arm"]
    mder = mot["derived"]

    mg = summary["moment_geometry_step2500"]
    d_arm = summary["direction_step2500_by_arm"]
    d_ak = summary["direction_step2500_by_arm_kind"]
    d_5120 = summary["direction_step5120_spot_by_arm"]
    wn = summary["weight_norms"]
    cf = summary["counterfactual_post_ns_step2500"]

    # ------------------------------------------------------------------
    # 1. Write results/motivation_numbers.tex (letters-only macro names)
    mot_macros = ["% Generated by scripts/analyze_mechanism.py (wave1/null_all motivating audit). Do not edit.\n"]

    def mmac(name: str, val: str):
        assert name.isalpha(), f"Macro name must contain letters only: {name}"
        mot_macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    TAG_MAP = (
        ("in_attn", "InAttn"),
        ("in_o", "InO"),
        ("in_mlp", "InMlp"),
        ("in_down", "InDown"),
        ("out_q", "OutQ"),
        ("out_k", "OutK"),
        ("out_v", "OutV"),
        ("out_o", "OutO"),
        ("out_down", "OutDown"),
        ("out_gate", "OutGate"),
        ("out_up", "OutUp"),
    )
    for key, tag in TAG_MAP:
        s = m25[key]
        ta = mavg[key]
        gd = mdrift[key]
        mmac(f"mot{tag}ErankMedPct", f"{s['erank_pct']['median']:.1f}")
        mmac(f"mot{tag}ErankMeanPct", f"{s['erank_pct']['mean']:.1f}")
        mmac(f"mot{tag}DiagCvMed", f"{s['diag_cv']['median']:.2f}")
        mmac(f"mot{tag}DiagCvMean", f"{s['diag_cv']['mean']:.2f}")
        mmac(f"mot{tag}DiagMaxMean", f"{s['diag_max_over_mean']['mean']:.1f}")
        mmac(f"mot{tag}DiagMaxMed", f"{s['diag_max_over_mean']['median']:.1f}")
        mmac(f"mot{tag}DiagMaxPeak", f"{s['diag_max_over_mean']['max']:.0f}")
        mmac(f"mot{tag}CorrRmsMed", f"{s['offdiag_corr_rms']['median']:.4f}")
        mmac(f"mot{tag}CorrRmsMean", f"{s['offdiag_corr_rms']['mean']:.4f}")
        mmac(f"mot{tag}BulkCorrRmsMean", f"{s['layers_04_31_mean']['offdiag_corr_rms']:.4f}")
        mmac(f"mot{tag}DiagEMedPct", f"{s['diag_energy_pct']['median']:.1f}")
        mmac(f"mot{tag}DiagEMeanPct", f"{s['diag_energy_pct']['mean']:.1f}")
        mmac(f"mot{tag}DiagAnisoMedPct", f"{s['diag_aniso_pct']['median']:.1f}")
        mmac(f"mot{tag}DiagAnisoMeanPct", f"{s['diag_aniso_pct']['mean']:.1f}")
        mmac(f"mot{tag}BulkDiagAnisoMedPct", f"{s['layers_04_31_median']['diag_aniso_pct']:.1f}")
        mmac(f"mot{tag}BulkDiagAnisoMeanPct", f"{s['layers_04_31_mean']['diag_aniso_pct']:.1f}")
        mmac(f"mot{tag}ExpVarLMedPct", f"{s['exp_var_L_pct']['median']:.1f}")
        mmac(f"mot{tag}ExpVarLMeanPct", f"{s['exp_var_L_pct']['mean']:.1f}")
        mmac(f"mot{tag}BulkExpVarLMedPct", f"{s['layers_04_31_median']['exp_var_L_pct']:.1f}")
        mmac(f"mot{tag}BulkExpVarLMeanPct", f"{s['layers_04_31_mean']['exp_var_L_pct']:.1f}")
        mmac(f"mot{tag}ExpVarLhatMedPct", f"{s['exp_var_Lhat_pct']['median']:.1f}")
        mmac(f"mot{tag}ExpVarLhatMeanPct", f"{s['exp_var_Lhat_pct']['mean']:.1f}")
        mmac(f"mot{tag}CosLMed", f"{s['cos_L']['median']:.4f}")
        mmac(f"mot{tag}CosLMean", f"{s['cos_L']['mean']:.4f}")
        mmac(f"mot{tag}RelErrLMed", f"{s['rel_err_L']['median']:.3f}")
        mmac(f"mot{tag}RelErrLMean", f"{s['rel_err_L']['mean']:.3f}")
        mmac(f"mot{tag}DropErrLMed", f"{s['rel_err_drop_L']['median']:.3f}")
        mmac(f"mot{tag}DropErrLMean", f"{s['rel_err_drop_L']['mean']:.3f}")
        # 8-step trajectory averages
        mmac(f"mot{tag}TrajDiagAnisoMedPct", f"{ta['diag_aniso_pct']['traj_mean_of_medians']:.1f}")
        mmac(f"mot{tag}TrajDiagAnisoMeanPct", f"{ta['diag_aniso_pct']['traj_mean_of_means']:.1f}")
        mmac(f"mot{tag}TrajExpVarLMedPct", f"{ta['exp_var_L_pct']['traj_mean_of_medians']:.1f}")
        mmac(f"mot{tag}TrajExpVarLMeanPct", f"{ta['exp_var_L_pct']['traj_mean_of_means']:.1f}")
        mmac(f"mot{tag}TrajCorrRmsMed", f"{ta['offdiag_corr_rms']['traj_mean_of_medians']:.4f}")
        mmac(f"mot{tag}TrajErankMedPct", f"{ta['erank_pct']['traj_mean_of_medians']:.1f}")
        mmac(f"mot{tag}TrajDiagCvMean", f"{ta['diag_cv']['traj_mean_of_means']:.2f}")
        # Grassmann subspace drift & cached-basis iteration ratio
        mmac(f"mot{tag}DriftTopSixteenMean", f"{gd['drift_top16']['mean']:.3f}")
        mmac(f"mot{tag}DriftTopSixteenMed", f"{gd['drift_top16']['median']:.3f}")
        mmac(f"mot{tag}DriftTopThirtyTwoMean", f"{gd['drift_top32']['mean']:.3f}")
        mmac(f"mot{tag}DriftTopThirtyTwoMed", f"{gd['drift_top32']['median']:.3f}")
        mmac(f"mot{tag}CachedEstOverTolMed", f"{s['cached_est2_over_tol']['median']:.2f}")
        mmac(f"mot{tag}ShortDriftTopSixteen", f"{s['short_drift_top16']['mean']:.4f}")

    # Early & late drift highlights
    mmac("motOutGateEarlyDriftTopSixteen", f"{mdrift_early['out_gate']['drift_top16']['mean']:.3f}")
    mmac("motInAttnEarlyDriftTopSixteen", f"{mdrift_early['in_attn']['drift_top16']['mean']:.3f}")
    mmac("motOutVLateDriftTopSixteen", f"{mdrift_late['out_v']['drift_top16']['mean']:.3f}")
    mmac("motInAttnLateDriftTopSixteen", f"{mdrift_late['in_attn']['drift_top16']['mean']:.3f}")

    # Trajectory start-to-end growth highlights
    mmac("motInDownEarlyCvMean", f"{mtraj['step_000250']['in_down']['diag_cv']['mean']:.2f}")
    mmac("motInDownFinalCvMean", f"{mtraj['step_005120']['in_down']['diag_cv']['mean']:.2f}")
    mmac("motOutGateEarlyCvMean", f"{mtraj['step_000250']['out_gate']['diag_cv']['mean']:.2f}")
    mmac("motOutGateFinalCvMean", f"{mtraj['step_005120']['out_gate']['diag_cv']['mean']:.2f}")
    mmac("motOutGatePeakCvMean", f"{mtraj['step_003500']['out_gate']['diag_cv']['mean']:.2f}")

    # Global 8-GPU and Rank-0 fallback macros
    mmac("motNoiseFloorCorrRms", f"{mder['finite_sample_corr_floor']:.4f}")
    mmac("motFbInTotal", _fmt_int_commas(mfb["input_fallbacks_8rank"]))
    mmac("motFbOutTotal", _fmt_int_commas(mfb["output_fallbacks_8rank"]))
    mmac("motExtraInTotal", _fmt_int_commas(mfb["input_extra_roots_8rank"]))
    mmac("motExtraOutTotal", _fmt_int_commas(mfb["output_extra_roots_8rank"]))
    mmac("motFbRatioTotal", _fmt_int_commas(round(mfb["raw_out_over_in_fb_ratio"])))
    mmac("motFbRatioPerStat", _fmt_int_commas(round(mfb["per_stat_out_over_in_fb_ratio"])))
    mmac("motFbInPostWarmup", _fmt_int_commas(mder["post250_input_fallbacks_8rank"]))
    mmac("motFbOutPostWarmup", _fmt_int_commas(mder["post250_output_fallbacks_8rank"]))
    mmac("motFbRatioPostWarmup", _fmt_int_commas(round(mder["post250_raw_out_over_in_ratio"])))
    mmac("motFbRatioPostWarmupPerStat", _fmt_int_commas(round(mder["post250_per_stat_out_over_in_ratio"])))
    mmac("motFbOutPerUpdate", f"{mfb['output_fb_per_update']:.1f}")
    mmac("motFbInPerUpdate", f"{mfb['input_fb_per_update']:.2f}")
    mmac("motFbOutPostPerUpdate", f"{mder['post250_output_fb_per_update']:.1f}")
    mmac("motFbOutPctStatStep", f"{mfb['output_fb_pct_per_stat_step']:.1f}")
    mmac("motFbInPctStatStep", f"{mfb['input_fb_pct_per_stat_step']:.3f}")
    mmac("motFbRankZeroIn", _fmt_int_commas(mfb_r0["wave1/null_all"]["rank0_input_fallbacks"]))
    mmac("motFbRankZeroOut", _fmt_int_commas(mfb_r0["wave1/null_all"]["rank0_output_fallbacks"]))
    mmac("motFbRankZeroRatio", _fmt_int_commas(round(mder["rank0_raw_out_over_in_ratio"])))
    mmac("motFbRankZeroMlpdiagOut", _fmt_int_commas(mfb_r0["wave2/inner_outer_mlpdiag"]["rank0_output_fallbacks"]))
    mmac("motFbMlpdiagDropPct", f"{mder['mlpdiag_rank0_fb_reduction_pct_vs_null']:.0f}")
    mmac("motDriftRatioOutVVsInAttn", f"{mder['drift_top16_ratio_out_v_over_in_attn']:.0f}")
    mmac("motDriftRatioOutGateVsInAttn", f"{mder['drift_top16_ratio_out_gate_over_in_attn']:.1f}")
    mmac("motDriftRatioOutUpVsInAttn", f"{mder['drift_top16_ratio_out_up_over_in_attn']:.1f}")

    (HERE / "results" / "motivation_numbers.tex").write_text("\n".join(mot_macros) + "\n")

    # ------------------------------------------------------------------
    # 2. Write results/table_motivation.tex (6+1 columns across all 11 moments)
    MOT_TABLE_ROWS = [
        ("__hdr_in__", r"\multicolumn{9}{@{}l}{\emph{Right-input second moments $C_t = \mathbb{E}[x x^{\Tr}]$ (4 per layer, 128 total)}} \\"),
        ("in_attn", r"\texttt{in:attn} ($q,k,v$)"),
        ("in_o", r"\texttt{in:o}"),
        ("in_mlp", r"\texttt{in:mlp} ($\text{gate},\text{up}$)"),
        ("in_down", r"\texttt{in:down} (post-SwiGLU)"),
        ("__hdr_out__", r"\addlinespace \multicolumn{9}{@{}l}{\emph{Left-cotangent second moments $B_t = \mathbb{E}[\delta \delta^{\Tr}]$ (7 per layer, 224 total)}} \\"),
        ("out_q", r"\texttt{out:q}"),
        ("out_k", r"\texttt{out:k}"),
        ("out_v", r"\texttt{out:v}"),
        ("out_o", r"\texttt{out:o} (residual)"),
        ("out_down", r"\texttt{out:down} (residual)"),
        ("out_gate", r"\textbf{\texttt{out:gate} (pre-SwiGLU)}"),
        ("out_up", r"\textbf{\texttt{out:up} (pre-SwiGLU)}"),
    ]

    mtlines = [
        "% Generated by scripts/analyze_mechanism.py (wave1/null_all motivating audit). Do not edit.",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Moment & $d$ & $\mathrm{erank}/d$ (\%) & $\mathrm{CV}_{\diag}$ & $\rho_{\mathrm{rms}}$ & "
        r"$\mathrm{DiagE}$ (\%) & $\mathrm{DiagAniso}$ (\%) & $R^2(A^{-1/2})$ (\%) & $\mathrm{Drift}_{16}$ \\",
        r"\midrule",
    ]
    for key, label in MOT_TABLE_ROWS:
        if key.startswith("__hdr"):
            mtlines.append(label)
            continue
        s = m25[key]
        gd = mdrift[key]
        bold = key in ("out_gate", "out_up")
        er = f"{s['erank_pct']['median']:.1f}"
        cv = f"{s['diag_cv']['median']:.2f}"
        rho = f"{s['offdiag_corr_rms']['median']:.4f}"
        de = f"{s['diag_energy_pct']['median']:.1f}"
        da = f"{s['diag_aniso_pct']['median']:.1f}"
        r2 = f"{s['exp_var_L_pct']['median']:.1f}"
        dr = f"{gd['drift_top16']['mean']:.3f}"
        if bold:
            rho_cell = rf"\textbf{{${rho}$}}"
            da_cell = rf"\textbf{{${da}$}}"
            r2_cell = rf"\textbf{{${r2}$}}"
        else:
            rho_cell = f"${rho}$"
            da_cell = f"${da}$"
            r2_cell = f"${r2}$"
        mtlines.append(
            f"{label} & {s['width']} & ${er}$ & ${cv}$ & {rho_cell} & ${de}$ & {da_cell} & {r2_cell} & ${dr}$ \\\\"
        )
    mtlines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "results" / "table_motivation.tex").write_text("\n".join(mtlines) + "\n")

    # ------------------------------------------------------------------
    # 3. Write figs/motivation_bars.dat (all 11 moments in MOTIVATION_ORDER)
    mb_header = [
        "idx",
        "kind",
        "side",
        "width",
        "erank_med_pct",
        "erank_mean_pct",
        "diag_cv_med",
        "diag_cv_mean",
        "corr_rms_med",
        "corr_rms_mean",
        "diag_energy_med_pct",
        "diag_energy_mean_pct",
        "diag_aniso_med_pct",
        "diag_aniso_mean_pct",
        "exp_var_L_med_pct",
        "exp_var_L_mean_pct",
        "traj_diag_aniso_med_pct",
        "traj_exp_var_L_med_pct",
        "drift_top16_mean",
        "drift_top32_mean",
    ]
    mb_lines = [" ".join(mb_header)]
    for idx, kind in enumerate(MOTIVATION_ORDER):
        s = m25[kind]
        ta = mavg[kind]
        gd = mdrift[kind]
        row = [
            str(idx),
            kind,
            s["side"],
            str(s["width"]),
            f"{s['erank_pct']['median']:.4f}",
            f"{s['erank_pct']['mean']:.4f}",
            f"{s['diag_cv']['median']:.4f}",
            f"{s['diag_cv']['mean']:.4f}",
            f"{s['offdiag_corr_rms']['median']:.6f}",
            f"{s['offdiag_corr_rms']['mean']:.6f}",
            f"{s['diag_energy_pct']['median']:.4f}",
            f"{s['diag_energy_pct']['mean']:.4f}",
            f"{s['diag_aniso_pct']['median']:.4f}",
            f"{s['diag_aniso_pct']['mean']:.4f}",
            f"{s['exp_var_L_pct']['median']:.4f}",
            f"{s['exp_var_L_pct']['mean']:.4f}",
            f"{ta['diag_aniso_pct']['traj_mean_of_medians']:.4f}",
            f"{ta['exp_var_L_pct']['traj_mean_of_medians']:.4f}",
            f"{gd['drift_top16']['mean']:.6f}",
            f"{gd['drift_top32']['mean']:.6f}",
        ]
        mb_lines.append(" ".join(row))
    (HERE / "figs" / "motivation_bars.dat").write_text("\n".join(mb_lines) + "\n")

    # ------------------------------------------------------------------
    # 4. Write figs/motivation_traj.dat (8-checkpoint trajectory evolution)
    mt_header = [
        "step",
        "tokens",
        "in_fb_k",
        "out_fb_k",
        "in_fb",
        "out_fb",
        # DiagAniso% (32-layer median) across representative groups
        "da_gate",
        "da_up",
        "da_in_down",
        "da_out_qk",
        "da_in_d512",
        "da_out_odown",
        # R^2(A^{-1/2})% (32-layer median) across representative groups
        "r2_gate",
        "r2_up",
        "r2_in_down",
        "r2_out_qk",
        "r2_out_v",
        "r2_in_d512",
        "r2_out_odown",
        # Off-diagonal correlation RMS (32-layer median)
        "rho_gate",
        "rho_up",
        "rho_out_odown",
        "rho_in_down",
        "rho_out_qk",
        "rho_in_d512",
        # Coordinate scale spread CV_diag (32-layer mean & median)
        "cv_gate_mean",
        "cv_up_mean",
        "cv_in_down_mean",
        "cv_out_qk_mean",
        "cv_in_d512_mean",
        "cv_out_odown_mean",
    ]
    mt_lines = [" ".join(mt_header)]
    for st in TRAJ_STEPS:
        sname = f"step_{st:06d}"
        ts = mtraj[sname]
        fb = mot["fallback_totals_8rank"][sname]

        def _avg_med(ks, field):
            return sum(ts[k][field]["median"] for k in ks) / len(ks)

        def _avg_mean(ks, field):
            return sum(ts[k][field]["mean"] for k in ks) / len(ks)

        row = [
            str(st),
            f"{ts['tokens_B']:.6f}",
            f"{fb['input_fallbacks_8rank'] / 1000.0:.4f}",
            f"{fb['output_fallbacks_8rank'] / 1000.0:.4f}",
            str(fb["input_fallbacks_8rank"]),
            str(fb["output_fallbacks_8rank"]),
            f"{ts['out_gate']['diag_aniso_pct']['median']:.4f}",
            f"{ts['out_up']['diag_aniso_pct']['median']:.4f}",
            f"{ts['in_down']['diag_aniso_pct']['median']:.4f}",
            f"{_avg_med(('out_q', 'out_k'), 'diag_aniso_pct'):.4f}",
            f"{_avg_med(('in_attn', 'in_o', 'in_mlp'), 'diag_aniso_pct'):.4f}",
            f"{_avg_med(('out_o', 'out_down'), 'diag_aniso_pct'):.4f}",
            f"{ts['out_gate']['exp_var_L_pct']['median']:.4f}",
            f"{ts['out_up']['exp_var_L_pct']['median']:.4f}",
            f"{ts['in_down']['exp_var_L_pct']['median']:.4f}",
            f"{_avg_med(('out_q', 'out_k'), 'exp_var_L_pct'):.4f}",
            f"{ts['out_v']['exp_var_L_pct']['median']:.4f}",
            f"{_avg_med(('in_attn', 'in_o', 'in_mlp'), 'exp_var_L_pct'):.4f}",
            f"{_avg_med(('out_o', 'out_down'), 'exp_var_L_pct'):.4f}",
            f"{ts['out_gate']['offdiag_corr_rms']['median']:.6f}",
            f"{ts['out_up']['offdiag_corr_rms']['median']:.6f}",
            f"{_avg_med(('out_o', 'out_down'), 'offdiag_corr_rms'):.6f}",
            f"{ts['in_down']['offdiag_corr_rms']['median']:.6f}",
            f"{_avg_med(('out_q', 'out_k'), 'offdiag_corr_rms'):.6f}",
            f"{_avg_med(('in_attn', 'in_o', 'in_mlp'), 'offdiag_corr_rms'):.6f}",
            f"{ts['out_gate']['diag_cv']['mean']:.4f}",
            f"{ts['out_up']['diag_cv']['mean']:.4f}",
            f"{ts['in_down']['diag_cv']['mean']:.4f}",
            f"{_avg_mean(('out_q', 'out_k'), 'diag_cv'):.4f}",
            f"{_avg_mean(('in_attn', 'in_o', 'in_mlp'), 'diag_cv'):.4f}",
            f"{_avg_mean(('out_o', 'out_down'), 'diag_cv'):.4f}",
        ]
        mt_lines.append(" ".join(row))
    (HERE / "figs" / "motivation_traj.dat").write_text("\n".join(mt_lines) + "\n")

    # ------------------------------------------------------------------
    # 5. Write results/mechanism_numbers.tex (post-training & active checks)
    macros = ["% Generated by scripts/analyze_mechanism.py. Do not edit.\n"]

    def mac(name: str, val: str):
        assert name.isalpha(), f"Macro name must contain letters only: {name}"
        macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")

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

    mac("mechOutGateBulkCorrRms", f"{mg['out_gate']['layers_04_31_mean']['offdiag_corr_rms']:.4f}")
    mac("mechOutUpBulkCorrRms", f"{mg['out_up']['layers_04_31_mean']['offdiag_corr_rms']:.4f}")
    mac("mechOutGateBulkExpVarLPct", f"{100.0 * mg['out_gate']['layers_04_31_mean']['explained_depart_var_L']:.1f}")
    mac("mechOutUpBulkExpVarLPct", f"{100.0 * mg['out_up']['layers_04_31_mean']['explained_depart_var_L']:.1f}")

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

    # ------------------------------------------------------------------
    # 6. Write results/table_mechanism.tex (retained for active-run cross-check)
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

    # ------------------------------------------------------------------
    # 7. Write figs/mechanism_bars.dat (including both mlpdiag and leftdiag angles)
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
        "graft_norm_leftdiag",
        "graft_deg_leftdiag",
    ]
    dat_lines = [" ".join(header)]
    for idx, kind in enumerate(MOMENT_ORDER):
        s = mg[kind]
        proj = kind.split("_", 1)[1]
        if kind.startswith("out_") and proj in d_ak["wave2/inner_outer_mlpdiag"]:
            gn = f"{d_ak['wave2/inner_outer_mlpdiag'][proj]['graft_norm_ratio']['mean']:.6f}"
            gd = f"{d_ak['wave2/inner_outer_mlpdiag'][proj]['graft_angle_deg']['mean']:.4f}"
            gn_ld = f"{d_ak['wave2/inner_outer_leftdiag'][proj]['graft_norm_ratio']['mean']:.6f}"
            gd_ld = f"{d_ak['wave2/inner_outer_leftdiag'][proj]['graft_angle_deg']['mean']:.4f}"
        else:
            gn = gd = gn_ld = gd_ld = "nan"
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
            gn_ld,
            gd_ld,
        ]
        dat_lines.append(" ".join(row))
    (HERE / "figs" / "mechanism_bars.dat").write_text("\n".join(dat_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="Recompute raw diagnostics from PVC checkpoints on CPU.")
    parser.add_argument("--from-scratch-dir", type=Path, help="Build mechanism_summary.json from raw JSONs in DIR.")
    args = parser.parse_args()

    summary_path = HERE / "results" / "mechanism_summary.json"
    default_scratch = Path("/root/.gemini/antigravity-cli/brain/f197fc22-55b7-4ffd-9eba-272d4e72d387/scratch")

    if args.recompute:
        import subprocess
        import sys
        script_mot = default_scratch / "compute_motivation_null_all.py"
        if script_mot.exists():
            subprocess.run([sys.executable, str(script_mot)], check=True)
        args.from_scratch_dir = default_scratch

    if args.from_scratch_dir is not None:
        mot_raw = json.loads((args.from_scratch_dir / "motivation_null_all_diagnostics.json").read_text())
        moment_raw = json.loads((args.from_scratch_dir / "moment_diagnostics.json").read_text())
        step_raw = json.loads((args.from_scratch_dir / "step_size_diagnostics.json").read_text())
        summary = build_summary_from_raw(mot_raw, moment_raw, step_raw)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    else:
        summary = json.loads(summary_path.read_text())

    write_outputs(summary)

    print("=== MOTIVATING 6+1 TABLE (wave1/null_all, step 2500 32-layer medians) ===")
    print(f"{'moment':12s} {'d':>5s} {'erank%':>7s} {'CV_dg':>7s} {'rho_rms':>8s} "
          f"{'DiagE%':>7s} {'DgAni%':>7s} {'R2_L%':>7s} {'Drft16':>7s}")
    m25 = summary["motivation_null_all"]["step_002500"]
    mdr = summary["motivation_null_all"]["grassmann_drift_250steps"]["step_002250_to_step_002500"]
    for k in MOTIVATION_ORDER:
        s = m25[k]
        g = mdr[k]
        print(
            f"{k:12s} {s['width']:5d} {s['erank_pct']['median']:7.1f} "
            f"{s['diag_cv']['median']:7.2f} {s['offdiag_corr_rms']['median']:8.4f} "
            f"{s['diag_energy_pct']['median']:7.1f} {s['diag_aniso_pct']['median']:7.1f} "
            f"{s['exp_var_L_pct']['median']:7.1f} {g['drift_top16']['mean']:7.3f}"
        )
    mfb = summary["motivation_null_all"]["fallback_totals_8rank"]["step_005120"]
    mder = summary["motivation_null_all"]["derived"]
    print(
        f"\n8-rank fallbacks (0..5120): input={mfb['input_fallbacks_8rank']} vs output={mfb['output_fallbacks_8rank']} "
        f"({mfb['raw_out_over_in_fb_ratio']:.1f}x raw, {mfb['per_stat_out_over_in_fb_ratio']:.1f}x/stat); "
        f"post-250: {mder['post250_input_fallbacks_8rank']} vs {mder['post250_output_fallbacks_8rank']} "
        f"({mder['post250_raw_out_over_in_ratio']:.0f}x raw)"
    )


if __name__ == "__main__":
    main()
