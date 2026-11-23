# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export, tabulate, and prepare pgfplots data for the 30-run confirmation & scaling campaign.

Reads the completed confirmation-20261121 campaign (30 training runs + 22 Paloma 16-domain
evaluations + 3 seed-0 anchor runs), computes final-step, 11-point second-half held-out
deltas, and lag-1 autocorrelation-inflated paired per-update training deltas across all five
windows and the full second half, and writes:
  - results/confirmation_campaign.json
  - results/confirmation_numbers.tex
  - results/table_confirmation.tex
  - figs/conf_lr_grid.dat
  - figs/conf_traj_delta.dat
  - figs/conf_paloma.dat

Supports --from-data to regenerate .tex and .dat files offline from results/confirmation_campaign.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PVC = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
CAMP = PVC / "campaigns" / "confirmation-20261121"
JSON_OUT = HERE / "results" / "confirmation_campaign.json"
MACROS_OUT = HERE / "results" / "confirmation_numbers.tex"
TABLE_OUT = HERE / "results" / "table_confirmation.tex"
LR_DAT = HERE / "figs" / "conf_lr_grid.dat"
TRAJ_DAT = HERE / "figs" / "conf_traj_delta.dat"
PALOMA_DAT = HERE / "figs" / "conf_paloma.dat"

ANCHORS = {
    "ref_muon_0p016": PVC / "runs" / "wave0" / "muon",
    "ref_muont_0p0226": PVC / "runs" / "wave_sym_fastlag" / "muont",
    "ref_wt_0p0226": PVC / "runs" / "wave_sym_fastlag" / "inner_outer_mlpdiag_lag",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lag1_inflated_se(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    if n < 2:
        return 0.0, 0.0
    m = statistics.fmean(diffs)
    c = [x - m for x in diffs]
    var = sum(x * x for x in c) / (n - 1)
    denom = sum(x * x for x in c)
    if denom <= 0.0:
        return 0.0, 0.0
    rho = max(0.0, min(0.99, sum(c[i] * c[i + 1] for i in range(n - 1)) / denom))
    se = math.sqrt((var / n) * ((1.0 + rho) / (1.0 - rho)))
    return se, rho


def load_run_series(run_dir: Path) -> dict:
    run_json = run_dir / "run.json"
    meta = json.loads(run_json.read_text())
    by_step: dict[int, dict] = {}
    segments = sorted(run_dir.glob("metrics_from_*.jsonl"))
    for mf in segments:
        for line in mf.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by_step[int(r["step"])] = r
    steps = sorted(by_step)
    rows = [by_step[s] for s in steps]
    T = len(rows)
    evals = {
        s: {
            "selection_loss": by_step[s]["selection_loss"],
            "c4_loss": by_step[s]["c4_loss"],
            "heldout_losses": by_step[s].get("heldout_losses", {}),
        }
        for s in steps
        if "selection_loss" in by_step[s]
    }
    eval_steps = sorted(evals)
    late_steps = [s for s in eval_steps if s > T // 2]
    step_times = [
        r["step_seconds"]
        for r in rows[20:]
        if "step_seconds" in r and "eval_seconds" not in r and "checkpoint_seconds" not in r
    ]
    return {
        "name": run_dir.name,
        "run_dir": str(run_dir),
        "manifest_sha256": sha256(run_json),
        "status": meta.get("status"),
        "arm": meta["config"]["experiment"]["arm"],
        "arm_label": meta["config"]["experiment"].get("arm_label"),
        "model_size": meta["config"]["model"]["size"],
        "seed": meta["config"]["training"]["seed"],
        "lr": meta["config"]["optimizer"]["lr"],
        "auxiliary_lr": meta["config"]["optimizer"]["auxiliary_lr"],
        "steps": T,
        "planned_steps": meta["config"]["training"]["steps"],
        "tokens": rows[-1]["tokens"],
        "final_sel": evals[eval_steps[-1]]["selection_loss"],
        "final_c4": evals[eval_steps[-1]]["c4_loss"],
        "final_heldout": evals[eval_steps[-1]]["heldout_losses"],
        "late11_steps": late_steps,
        "late11_sel": statistics.fmean(evals[s]["selection_loss"] for s in late_steps),
        "late11_c4": statistics.fmean(evals[s]["c4_loss"] for s in late_steps),
        "median_step_s": statistics.median(step_times),
        "peak_gib": max((r.get("peak_allocated_bytes", 0) for r in rows), default=0) / (1024**3),
        "elapsed_s": meta.get("elapsed_seconds"),
        "evals": {str(k): v for k, v in evals.items()},
        "_train": [r["train_loss"] for r in rows],
    }


def pair_metrics(a: dict, b: dict) -> dict:
    T = a["steps"]
    assert b["steps"] == T
    diffs = [x - y for x, y in zip(a["_train"], b["_train"])]
    eval_steps = sorted(set(int(k) for k in a["evals"]) & set(int(k) for k in b["evals"]))
    late11 = [s for s in eval_steps if s > T // 2]
    last5 = eval_steps[-5:]
    post1 = [s for s in eval_steps if s > 1]

    def eval_stats(sts: list[int]) -> dict:
        ds = [a["evals"][str(s)]["selection_loss"] - b["evals"][str(s)]["selection_loss"] for s in sts]
        dc = [a["evals"][str(s)]["c4_loss"] - b["evals"][str(s)]["c4_loss"] for s in sts]
        return {
            "n": len(sts),
            "steps": sts,
            "d_sel": statistics.fmean(ds),
            "se_sel": statistics.stdev(ds) / math.sqrt(len(ds)) if len(ds) > 1 else 0.0,
            "wins_sel": sum(1 for x in ds if x < 0),
            "d_c4": statistics.fmean(dc),
            "se_c4": statistics.stdev(dc) / math.sqrt(len(dc)) if len(dc) > 1 else 0.0,
            "wins_c4": sum(1 for x in dc if x < 0),
        }

    wlen = T // 5
    windows = {}
    for k in range(5):
        lo = k * wlen
        hi = (k + 1) * wlen if k < 4 else T
        seg = diffs[lo:hi]
        m = statistics.fmean(seg)
        se, rho = lag1_inflated_se(seg)
        windows[f"W{k+1}"] = {"n": len(seg), "mean": m, "se": se, "rho1": rho, "t_abs": abs(m) / se if se > 0 else 0.0}

    half = diffs[T // 2 :]
    hm, hse, hrho = statistics.fmean(half), *lag1_inflated_se(half)
    fm, fse, frho = statistics.fmean(diffs), *lag1_inflated_se(diffs)

    return {
        "final": eval_stats([eval_steps[-1]]),
        "last5": eval_stats(last5),
        "late11": eval_stats(late11),
        "all_post1": eval_stats(post1),
        "train_windows": windows,
        "train_second_half": {"n": len(half), "mean": hm, "se": hse, "rho1": hrho, "t_abs": abs(hm) / hse if hse > 0 else 0.0},
        "train_full": {"n": len(diffs), "mean": fm, "se": fse, "rho1": frho, "t_abs": abs(fm) / fse if fse > 0 else 0.0},
        "step_overhead_pct": (a["median_step_s"] / b["median_step_s"] - 1.0) * 100.0,
    }


def strip_private(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def build_from_live() -> dict:
    runs: dict[str, dict] = {}
    for k, p in ANCHORS.items():
        runs[k] = load_run_series(p)
    for w in ("wave1a", "wave1b", "wave2a", "wave2b", "wave2c", "wave3a", "wave3b", "wave4a"):
        for d in sorted((CAMP / "runs" / w).iterdir()):
            if d.is_dir() and (d / "run.json").is_file():
                runs[f"{w}/{d.name}"] = load_run_series(d)

    paloma: dict[str, dict] = {}
    for p in sorted((CAMP / "paloma_eval").glob("*.json")):
        paloma[p.stem] = json.loads(p.read_text())

    def pal_pair(name_a: str, name_b: str) -> dict:
        pa, pb = paloma[name_a], paloma[name_b]
        doms = sorted(k for k in pa["scores"] if k.startswith("paloma/"))
        wins = sum(1 for d in doms if pa["scores"][d] < pb["scores"][d])
        return {
            "macro_a": pa["paloma_macro_mean"],
            "macro_b": pb["paloma_macro_mean"],
            "d_macro": pa["paloma_macro_mean"] - pb["paloma_macro_mean"],
            "domain_wins": wins,
            "domain_total": len(doms),
        }

    pairs = {
        # Phase 1 matched-LR comparisons (seed 0, 130M 1x)
        "p1_0p016_muon_vs_muont": pair_metrics(runs["ref_muon_0p016"], runs["wave1a/muont_130m_1x_seed0_lr0p016"]),
        "p1_0p016_wt_vs_muont": pair_metrics(runs["wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016"], runs["wave1a/muont_130m_1x_seed0_lr0p016"]),
        "p1_0p016_wt_vs_muon": pair_metrics(runs["wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016"], runs["ref_muon_0p016"]),
        "p1_0p0226_muon_vs_muont": pair_metrics(runs["wave1b/muon_130m_1x_seed0_lr0p0226"], runs["ref_muont_0p0226"]),
        "p1_0p0226_null_vs_muont": pair_metrics(runs["wave1b/null_wt_130m_1x_seed0_lr0p0226"], runs["ref_muont_0p0226"]),
        "p1_0p0226_od_vs_muont": pair_metrics(runs["wave1b/od_all_130m_1x_seed0_lr0p0226"], runs["ref_muont_0p0226"]),
        "p1_0p0226_wt_vs_muont": pair_metrics(runs["ref_wt_0p0226"], runs["ref_muont_0p0226"]),
        "p1_0p0226_wt_vs_null": pair_metrics(runs["ref_wt_0p0226"], runs["wave1b/null_wt_130m_1x_seed0_lr0p0226"]),
        "p1_0p032_od_vs_muont": pair_metrics(runs["wave1b/od_all_130m_1x_seed0_lr0p032"], runs["wave1a/muont_130m_1x_seed0_lr0p032"]),
        "p1_0p032_wt_vs_muont": pair_metrics(runs["wave1a/wt_mlp_lag_130m_1x_seed0_lr0p032"], runs["wave1a/muont_130m_1x_seed0_lr0p032"]),
        # Phase 3A/3B (130M 4x = 80 TPP, seed 0)
        "p3_muon_vs_muont": pair_metrics(runs["wave3a/muon_130m_4x_seed0"], runs["wave3a/muont_130m_4x_seed0"]),
        "p3b_od_vs_muont": pair_metrics(runs["wave3b/od_all_130m_4x_seed0_lr0p01056"], runs["wave3a/muont_130m_4x_seed0"]),
        "p3b_wt_vs_muont": pair_metrics(runs["wave3b/wt_mlp_lag_130m_4x_seed0_lr0p01056"], runs["wave3a/muont_130m_4x_seed0"]),
        "p3b_wt_vs_muon": pair_metrics(runs["wave3b/wt_mlp_lag_130m_4x_seed0_lr0p01056"], runs["wave3a/muon_130m_4x_seed0"]),
        "p3a_hot_od_vs_muont": pair_metrics(runs["wave3a/od_all_130m_4x_seed0"], runs["wave3a/muont_130m_4x_seed0"]),
        "p3a_hot_wt_vs_muont": pair_metrics(runs["wave3a/wt_mlp_lag_130m_4x_seed0"], runs["wave3a/muont_130m_4x_seed0"]),
        # Phase 4 (520M 1x = 20 TPP, seed 0)
        "p4_muon_vs_muont": pair_metrics(runs["wave4a/muon_520m_1x_seed0"], runs["wave4a/muont_520m_1x_seed0"]),
        "p4_od_vs_muont": pair_metrics(runs["wave4a/od_all_520m_1x_seed0"], runs["wave4a/muont_520m_1x_seed0"]),
        "p4_wt_vs_muont": pair_metrics(runs["wave4a/wt_mlp_lag_520m_1x_seed0"], runs["wave4a/muont_520m_1x_seed0"]),
        "p4_wt_vs_muon": pair_metrics(runs["wave4a/wt_mlp_lag_520m_1x_seed0"], runs["wave4a/muon_520m_1x_seed0"]),
    }

    # Phase 2 paired 3-seed summary (seeds 1, 2, 3)
    p2_arms = {}
    for arm in ("muon", "muont", "od_all", "wt_mlp_lag"):
        sels, c4s, pals, d_sel_mt, d_c4_mt, d_pal_mt, d_sel_m, d_pal_m, steps_s, mems = [], [], [], [], [], [], [], [], [], []
        for seed, w in ((1, "wave2a"), (2, "wave2b"), (3, "wave2c")):
            ra = runs[f"{w}/{arm}_130m_1x_seed{seed}"]
            rmt = runs[f"{w}/muont_130m_1x_seed{seed}"]
            rm = runs[f"{w}/muon_130m_1x_seed{seed}"]
            pa = paloma[f"{arm}_130m_1x_seed{seed}"]["paloma_macro_mean"]
            pmt = paloma[f"muont_130m_1x_seed{seed}"]["paloma_macro_mean"]
            pm = paloma[f"muon_130m_1x_seed{seed}"]["paloma_macro_mean"]
            sels.append(ra["final_sel"])
            c4s.append(ra["final_c4"])
            pals.append(pa)
            d_sel_mt.append(ra["final_sel"] - rmt["final_sel"])
            d_c4_mt.append(ra["final_c4"] - rmt["final_c4"])
            d_pal_mt.append(pa - pmt)
            d_sel_m.append(ra["final_sel"] - rm["final_sel"])
            d_pal_m.append(pa - pm)
            steps_s.append(ra["median_step_s"])
            mems.append(ra["peak_gib"])
        mt_step_mean = statistics.fmean(runs[f"{w}/muont_130m_1x_seed{s}"]["median_step_s"] for s, w in ((1, "wave2a"), (2, "wave2b"), (3, "wave2c")))
        p2_arms[arm] = {
            "lr": runs[f"wave2a/{arm}_130m_1x_seed1"]["lr"],
            "sel_seeds": sels,
            "sel_mean": statistics.fmean(sels),
            "sel_std": statistics.stdev(sels),
            "c4_seeds": c4s,
            "c4_mean": statistics.fmean(c4s),
            "c4_std": statistics.stdev(c4s),
            "paloma_seeds": pals,
            "paloma_mean": statistics.fmean(pals),
            "paloma_std": statistics.stdev(pals),
            "d_sel_vs_muont_seeds": d_sel_mt,
            "d_sel_vs_muont_mean": statistics.fmean(d_sel_mt),
            "d_sel_vs_muont_std": statistics.stdev(d_sel_mt),
            "wins_sel_vs_muont": sum(1 for x in d_sel_mt if x < 0),
            "d_c4_vs_muont_seeds": d_c4_mt,
            "d_c4_vs_muont_mean": statistics.fmean(d_c4_mt),
            "d_c4_vs_muont_std": statistics.stdev(d_c4_mt),
            "wins_c4_vs_muont": sum(1 for x in d_c4_mt if x < 0),
            "d_paloma_vs_muont_seeds": d_pal_mt,
            "d_paloma_vs_muont_mean": statistics.fmean(d_pal_mt),
            "d_paloma_vs_muont_std": statistics.stdev(d_pal_mt),
            "wins_paloma_vs_muont": sum(1 for x in d_pal_mt if x < 0),
            "d_sel_vs_muon_mean": statistics.fmean(d_sel_m),
            "d_paloma_vs_muon_mean": statistics.fmean(d_pal_m),
            "mean_step_s": statistics.fmean(steps_s),
            "step_overhead_pct": (statistics.fmean(steps_s) / mt_step_mean - 1.0) * 100.0,
            "peak_gib": max(mems),
        }

    paloma_pairs = {
        "p3_muon_vs_muont": pal_pair("muon_130m_4x_seed0", "muont_130m_4x_seed0"),
        "p3b_od_vs_muont": pal_pair("od_all_130m_4x_seed0_lr0p01056", "muont_130m_4x_seed0"),
        "p3b_wt_vs_muont": pal_pair("wt_mlp_lag_130m_4x_seed0_lr0p01056", "muont_130m_4x_seed0"),
        "p3b_wt_vs_muon": pal_pair("wt_mlp_lag_130m_4x_seed0_lr0p01056", "muon_130m_4x_seed0"),
        "p3a_hot_od_vs_muont": pal_pair("od_all_130m_4x_seed0", "muont_130m_4x_seed0"),
        "p3a_hot_wt_vs_muont": pal_pair("wt_mlp_lag_130m_4x_seed0", "muont_130m_4x_seed0"),
        "p4_muon_vs_muont": pal_pair("muon_520m_1x_seed0", "muont_520m_1x_seed0"),
        "p4_od_vs_muont": pal_pair("od_all_520m_1x_seed0", "muont_520m_1x_seed0"),
        "p4_wt_vs_muont": pal_pair("wt_mlp_lag_520m_1x_seed0", "muont_520m_1x_seed0"),
        "p4_wt_vs_muon": pal_pair("wt_mlp_lag_520m_1x_seed0", "muon_520m_1x_seed0"),
    }

    # Build normalized-progress trajectory table across all 21 post-step-1 eval checkpoints
    # (all three matched-LR regimes share exactly 21 post-step-1 eval points!)
    def traj_deltas(ra: dict, rb: dict) -> list[tuple[float, float]]:
        T = ra["steps"]
        sts = sorted(s for s in (set(int(k) for k in ra["evals"]) & set(int(k) for k in rb["evals"])) if s > 1)
        return [(s / T, ra["evals"][str(s)]["selection_loss"] - rb["evals"][str(s)]["selection_loss"]) for s in sts]

    t_130_1x_wt = traj_deltas(runs["wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016"], runs["wave1a/muont_130m_1x_seed0_lr0p016"])
    t_130_4x_wt = traj_deltas(runs["wave3b/wt_mlp_lag_130m_4x_seed0_lr0p01056"], runs["wave3a/muont_130m_4x_seed0"])
    t_130_4x_od = traj_deltas(runs["wave3b/od_all_130m_4x_seed0_lr0p01056"], runs["wave3a/muont_130m_4x_seed0"])
    t_520_1x_wt = traj_deltas(runs["wave4a/wt_mlp_lag_520m_1x_seed0"], runs["wave4a/muont_520m_1x_seed0"])
    t_520_1x_od = traj_deltas(runs["wave4a/od_all_520m_1x_seed0"], runs["wave4a/muont_520m_1x_seed0"])
    assert len(t_130_1x_wt) == len(t_130_4x_wt) == len(t_520_1x_wt) == 21
    traj_rows = []
    for idx in range(21):
        traj_rows.append(
            {
                "progress": t_130_1x_wt[idx][0],
                "wt_130_1x": t_130_1x_wt[idx][1],
                "wt_130_4x": t_130_4x_wt[idx][1],
                "od_130_4x": t_130_4x_od[idx][1],
                "wt_520_1x": t_520_1x_wt[idx][1],
                "od_520_1x": t_520_1x_od[idx][1],
            }
        )

    return {
        "campaign_root": str(CAMP),
        "n_campaign_runs": 30,
        "n_paloma_evals": len(paloma),
        "runs": {k: strip_private(v) for k, v in runs.items()},
        "pairs": pairs,
        "phase2_3seed": p2_arms,
        "paloma_pairs": paloma_pairs,
        "trajectory_eval_deltas": traj_rows,
    }


def fmt_signed(x: float, prec: int = 4) -> str:
    return f"{x:+.{prec}f}"


def fmt_abs(x: float, prec: int = 4) -> str:
    return f"{x:.{prec}f}"


def emit_outputs(data: dict) -> None:
    runs = data["runs"]
    pairs = data["pairs"]
    p2 = data["phase2_3seed"]
    pp = data["paloma_pairs"]

    # 1. LaTeX macros
    macros = [
        "% Generated by scripts/integrate_confirmation_campaign.py. Do not edit.",
        f"\\newcommand{{\\confTotalRuns}}{{{data['n_campaign_runs']}}}",
        f"\\newcommand{{\\confPalomaCount}}{{{data['n_paloma_evals']}}}",
        # Phase 1 full-budget LR grid & fast_lag Null-WT
        f"\\newcommand{{\\confNullFinalSelDelta}}{{{fmt_signed(pairs['p1_0p0226_null_vs_muont']['final']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confNullLateSelDelta}}{{{fmt_signed(pairs['p1_0p0226_null_vs_muont']['late11']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confNullLateCfourDelta}}{{{fmt_signed(pairs['p1_0p0226_null_vs_muont']['late11']['d_c4'], 4)}}}",
        f"\\newcommand{{\\confNullTrainWFive}}{{{fmt_signed(pairs['p1_0p0226_null_vs_muont']['train_windows']['W5']['mean'], 4)}}}",
        f"\\newcommand{{\\confPOneZeroOneSixWtFinalSel}}{{{fmt_signed(pairs['p1_0p016_wt_vs_muont']['final']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPOneZeroOneSixWtLateSel}}{{{fmt_signed(pairs['p1_0p016_wt_vs_muont']['late11']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPOneZeroOneSixWtLateCfour}}{{{fmt_signed(pairs['p1_0p016_wt_vs_muont']['late11']['d_c4'], 4)}}}",
        f"\\newcommand{{\\confPOneZeroOneSixWtTrainHalf}}{{{fmt_signed(pairs['p1_0p016_wt_vs_muont']['train_second_half']['mean'], 4)}}}",
        f"\\newcommand{{\\confPOneZeroOneSixWtTrainHalfSe}}{{{fmt_abs(pairs['p1_0p016_wt_vs_muont']['train_second_half']['se'], 5)}}}",
        f"\\newcommand{{\\confPOneZeroTwoTwoWtFinalSel}}{{{fmt_signed(pairs['p1_0p0226_wt_vs_muont']['final']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPOneZeroTwoTwoWtLateSel}}{{{fmt_signed(pairs['p1_0p0226_wt_vs_muont']['late11']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPOneZeroTwoTwoWtNetLateSel}}{{{fmt_signed(pairs['p1_0p0226_wt_vs_null']['late11']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPOneZeroTwoTwoWtNetFinalSel}}{{{fmt_signed(pairs['p1_0p0226_wt_vs_null']['final']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPOneZeroTwoTwoOdLateSel}}{{{fmt_signed(pairs['p1_0p0226_od_vs_muont']['late11']['d_sel'], 4)}}}",
        # Phase 2 fresh 3-seed confirmation (seeds 1, 2, 3)
        f"\\newcommand{{\\confPTwoWtSelDeltaMean}}{{{fmt_signed(p2['wt_mlp_lag']['d_sel_vs_muont_mean'], 4)}}}",
        f"\\newcommand{{\\confPTwoWtSelDeltaStd}}{{{fmt_abs(p2['wt_mlp_lag']['d_sel_vs_muont_std'], 4)}}}",
        f"\\newcommand{{\\confPTwoWtSelVsMuonMean}}{{{fmt_signed(p2['wt_mlp_lag']['d_sel_vs_muon_mean'], 4)}}}",
        f"\\newcommand{{\\confPTwoWtCfourDeltaMean}}{{{fmt_signed(p2['wt_mlp_lag']['d_c4_vs_muont_mean'], 4)}}}",
        f"\\newcommand{{\\confPTwoWtCfourDeltaStd}}{{{fmt_abs(p2['wt_mlp_lag']['d_c4_vs_muont_std'], 4)}}}",
        f"\\newcommand{{\\confPTwoWtPalomaDeltaMean}}{{{fmt_signed(p2['wt_mlp_lag']['d_paloma_vs_muont_mean'], 4)}}}",
        f"\\newcommand{{\\confPTwoWtPalomaDeltaStd}}{{{fmt_abs(p2['wt_mlp_lag']['d_paloma_vs_muont_std'], 4)}}}",
        f"\\newcommand{{\\confPTwoWtPalomaVsMuonMean}}{{{fmt_signed(p2['wt_mlp_lag']['d_paloma_vs_muon_mean'], 4)}}}",
        f"\\newcommand{{\\confPTwoOdSelDeltaMean}}{{{fmt_signed(p2['od_all']['d_sel_vs_muont_mean'], 4)}}}",
        f"\\newcommand{{\\confPTwoOdPalomaDeltaMean}}{{{fmt_signed(p2['od_all']['d_paloma_vs_muont_mean'], 4)}}}",
        f"\\newcommand{{\\confPTwoWtStepPct}}{{{fmt_abs(p2['wt_mlp_lag']['step_overhead_pct'], 1)}}}",
        # Phase 3A/3B (130M 4x = 80 TPP)
        f"\\newcommand{{\\confPThreeHotWtFinalSel}}{{{fmt_signed(pairs['p3a_hot_wt_vs_muont']['final']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPThreeWtFinalSel}}{{{fmt_signed(pairs['p3b_wt_vs_muont']['final']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPThreeWtFinalCfour}}{{{fmt_signed(pairs['p3b_wt_vs_muont']['final']['d_c4'], 4)}}}",
        f"\\newcommand{{\\confPThreeWtLateSel}}{{{fmt_signed(pairs['p3b_wt_vs_muont']['late11']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPThreeWtLateCfour}}{{{fmt_signed(pairs['p3b_wt_vs_muont']['late11']['d_c4'], 4)}}}",
        f"\\newcommand{{\\confPThreeWtTrainHalf}}{{{fmt_signed(pairs['p3b_wt_vs_muont']['train_second_half']['mean'], 4)}}}",
        f"\\newcommand{{\\confPThreeWtTrainHalfSe}}{{{fmt_abs(pairs['p3b_wt_vs_muont']['train_second_half']['se'], 5)}}}",
        f"\\newcommand{{\\confPThreeWtPalomaDelta}}{{{fmt_signed(pp['p3b_wt_vs_muont']['d_macro'], 4)}}}",
        f"\\newcommand{{\\confPThreeWtPalomaWins}}{{{pp['p3b_wt_vs_muont']['domain_wins']}}}",
        f"\\newcommand{{\\confPThreeWtVsMuonFinalSel}}{{{fmt_signed(pairs['p3b_wt_vs_muon']['final']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPThreeOdFinalSel}}{{{fmt_signed(pairs['p3b_od_vs_muont']['final']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPThreeOdLateSel}}{{{fmt_signed(pairs['p3b_od_vs_muont']['late11']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPThreeOdPalomaDelta}}{{{fmt_signed(pp['p3b_od_vs_muont']['d_macro'], 4)}}}",
        f"\\newcommand{{\\confPThreeOdPalomaWins}}{{{pp['p3b_od_vs_muont']['domain_wins']}}}",
        f"\\newcommand{{\\confPThreeWtStepPct}}{{{fmt_abs(pairs['p3b_wt_vs_muont']['step_overhead_pct'], 1)}}}",
        # Phase 4 (520M 1x = 20 TPP, matched LR)
        f"\\newcommand{{\\confPFourMuonFinalCfourAbs}}{{{fmt_abs(runs['wave4a/muon_520m_1x_seed0']['final_c4'], 4)}}}",
        f"\\newcommand{{\\confPFourMuontFinalCfourAbs}}{{{fmt_abs(runs['wave4a/muont_520m_1x_seed0']['final_c4'], 4)}}}",
        f"\\newcommand{{\\confPFourWtFinalSel}}{{{fmt_signed(pairs['p4_wt_vs_muont']['final']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPFourWtFinalCfour}}{{{fmt_signed(pairs['p4_wt_vs_muont']['final']['d_c4'], 4)}}}",
        f"\\newcommand{{\\confPFourWtLateSel}}{{{fmt_signed(pairs['p4_wt_vs_muont']['late11']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPFourWtLateCfour}}{{{fmt_signed(pairs['p4_wt_vs_muont']['late11']['d_c4'], 4)}}}",
        f"\\newcommand{{\\confPFourWtTrainHalf}}{{{fmt_signed(pairs['p4_wt_vs_muont']['train_second_half']['mean'], 4)}}}",
        f"\\newcommand{{\\confPFourWtTrainHalfSe}}{{{fmt_abs(pairs['p4_wt_vs_muont']['train_second_half']['se'], 5)}}}",
        f"\\newcommand{{\\confPFourWtPalomaDelta}}{{{fmt_signed(pp['p4_wt_vs_muont']['d_macro'], 4)}}}",
        f"\\newcommand{{\\confPFourWtPalomaWins}}{{{pp['p4_wt_vs_muont']['domain_wins']}}}",
        f"\\newcommand{{\\confPFourWtVsMuonFinalSel}}{{{fmt_signed(pairs['p4_wt_vs_muon']['final']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPFourOdFinalSel}}{{{fmt_signed(pairs['p4_od_vs_muont']['final']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPFourOdLateSel}}{{{fmt_signed(pairs['p4_od_vs_muont']['late11']['d_sel'], 4)}}}",
        f"\\newcommand{{\\confPFourOdPalomaDelta}}{{{fmt_signed(pp['p4_od_vs_muont']['d_macro'], 4)}}}",
        f"\\newcommand{{\\confPFourOdPalomaWins}}{{{pp['p4_od_vs_muont']['domain_wins']}}}",
        f"\\newcommand{{\\confPFourWtStepPct}}{{{fmt_abs(pairs['p4_wt_vs_muont']['step_overhead_pct'], 1)}}}",
        f"\\newcommand{{\\confPFourWtGib}}{{{fmt_abs(runs['wave4a/wt_mlp_lag_520m_1x_seed0']['peak_gib'], 1)}}}",
        f"\\newcommand{{\\confPFourMuontGib}}{{{fmt_abs(runs['wave4a/muont_520m_1x_seed0']['peak_gib'], 1)}}}",
    ]
    MACROS_OUT.write_text("\n".join(macros) + "\n")

    # 2. Master Confirmation & Scaling LaTeX Table
    def row_pair(label: str, lr_str: str, pm: dict, pal_str: str, gib: float, bold: bool = False, shade: bool = False) -> str:
        prefix = r"\rowcolor{pagepanel} " if shade else ""
        fsel = fmt_signed(pm["final"]["d_sel"], 4)
        lsel = f"{fmt_signed(pm['late11']['d_sel'], 4)} ({pm['late11']['wins_sel']}/11)"
        fc4 = fmt_signed(pm["final"]["d_c4"], 4)
        tr = f"{fmt_signed(pm['train_second_half']['mean'], 4)}\\pm{pm['train_second_half']['se']:.5f}"
        ov = f"{pm['step_overhead_pct']:+.1f}\\%"
        if bold:
            label = f"\\textbf{{{label}}}"
            fsel = f"$\\mathbf{{{fsel}}}$"
            lsel = f"$\\mathbf{{{fmt_signed(pm['late11']['d_sel'], 4)}}}$ ({pm['late11']['wins_sel']}/11)"
            fc4 = f"$\\mathbf{{{fc4}}}$"
        else:
            fsel = f"${fsel}$"
            lsel = f"${fmt_signed(pm['late11']['d_sel'], 4)}$ ({pm['late11']['wins_sel']}/11)"
            fc4 = f"${fc4}$"
        return f"{prefix}{label} & ${lr_str}$ & {fsel} & {lsel} & {fc4} & ${tr}$ & {pal_str} & ${ov}$ / {gib:.1f} \\\\"

    tlines = [
        "% Generated by scripts/integrate_confirmation_campaign.py. Do not edit.",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"Regime / Arm & $\eta$ & Final $\Delta$Sel & Late-11 $\Delta$Sel (Wins) & Final $\Delta$C4 & 2nd-Half $\Delta$Train$\pm$SE & 16-Dom Paloma $\Delta$ & Step\% / GiB \\",
        r"\midrule",
        # Phase 1
        r"\multicolumn{8}{@{}l}{\textcolor{famRef}{\pagesfsb Phase 1: $130\text{M}$ $1\times$ Chinchilla ($20$ TPP, $2.68\text{B}$ tok, Seed 0) --- Full-Budget LR Bracket \& \texttt{fast\_lag} Null}} \\",
        f"\\taxid{{MuonT}} (ref @ $.016$) & $.016$ & ${runs['wave1a/muont_130m_1x_seed0_lr0p016']['final_sel']:.4f}$ & ${runs['wave1a/muont_130m_1x_seed0_lr0p016']['late11_sel']:.4f}$ (ref) & ${runs['wave1a/muont_130m_1x_seed0_lr0p016']['final_c4']:.4f}$ & $0$ ($2{{,}}560$ upd.) & --- & $0.0\\%$ / {runs['wave1a/muont_130m_1x_seed0_lr0p016']['peak_gib']:.1f} \\\\",
        row_pair(r"\taxid{Muon} (ref @ $.016$)", ".016", pairs["p1_0p016_muon_vs_muont"], "---", runs["ref_muon_0p016"]["peak_gib"]),
        row_pair(r"\taxid{WT-MLP-Lag} (matched)", ".016", pairs["p1_0p016_wt_vs_muont"], "---", runs["wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016"]["peak_gib"], bold=True, shade=True),
        f"\\taxid{{MuonT}} (ref @ $.0226$) & $.0226$ & ${runs['ref_muont_0p0226']['final_sel']:.4f}$ & ${runs['ref_muont_0p0226']['late11_sel']:.4f}$ (ref) & ${runs['ref_muont_0p0226']['final_c4']:.4f}$ & $0$ ($2{{,}}560$ upd.) & --- & $0.0\\%$ / {runs['ref_muont_0p0226']['peak_gib']:.1f} \\\\",
        row_pair(r"\taxid{Null-WT} (\texttt{fast\_lag})", ".0226", pairs["p1_0p0226_null_vs_muont"], "---", runs["wave1b/null_wt_130m_1x_seed0_lr0p0226"]["peak_gib"]),
        row_pair(r"\taxid{OD-All} (matched)", ".0226", pairs["p1_0p0226_od_vs_muont"], "---", runs["wave1b/od_all_130m_1x_seed0_lr0p0226"]["peak_gib"]),
        row_pair(r"\taxid{WT-MLP-Lag} (matched)", ".0226", pairs["p1_0p0226_wt_vs_muont"], "---", runs["ref_wt_0p0226"]["peak_gib"], bold=True, shade=True),
        r"\addlinespace[2.5pt]",
        # Phase 2
        r"\multicolumn{8}{@{}l}{\textcolor{famWT}{\pagesfsb Phase 2: $130\text{M}$ $1\times$ Chinchilla ($20$ TPP) --- Fresh Post-Selection 3-Seed Paired Replication (Seeds 1, 2, 3; Gate-1 LRs)}} \\",
        f"\\taxid{{MuonT}} (3-seed mean) & $.016$ & ${p2['muont']['sel_mean']:.4f}$ & --- & ${p2['muont']['c4_mean']:.4f}$ & --- & ${p2['muont']['paloma_mean']:.4f}$ (ref) & $0.0\\%$ / {p2['muont']['peak_gib']:.1f} \\\\",
        f"\\taxid{{Muon}} (3-seed paired) & $.016$ & $+0.0119$ (0/3) & --- & $+0.0118$ (0/3) & --- & $+0.0115$ (0/3) & ${p2['muon']['step_overhead_pct']:+.1f}\\%$ / {p2['muon']['peak_gib']:.1f} \\\\",
        f"\\taxid{{OD-All}} (3-seed paired) & $.0226$ & ${p2['od_all']['d_sel_vs_muont_mean']:+.4f}\\pm{p2['od_all']['d_sel_vs_muont_std']:.4f}$ (3/3) & \\emph{{(hotter LR)}} & ${p2['od_all']['d_c4_vs_muont_mean']:+.4f}$ (3/3) & \\emph{{(hotter LR)}} & ${p2['od_all']['d_paloma_vs_muont_mean']:+.4f}$ (2/3) & ${p2['od_all']['step_overhead_pct']:+.1f}\\%$ / {p2['od_all']['peak_gib']:.1f} \\\\",
        f"\\rowcolor{{pagepanel}} \\textbf{{\\taxid{{WT-MLP-Lag}}}} (3-seed) & $.0226$ & $\\mathbf{{{p2['wt_mlp_lag']['d_sel_vs_muont_mean']:+.4f}}}\\pm{p2['wt_mlp_lag']['d_sel_vs_muont_std']:.4f}$ (3/3) & \\emph{{(hotter LR)}} & $\\mathbf{{{p2['wt_mlp_lag']['d_c4_vs_muont_mean']:+.4f}}}$ (3/3) & \\emph{{(hotter LR)}} & $\\mathbf{{{p2['wt_mlp_lag']['d_paloma_vs_muont_mean']:+.4f}}}$ (3/3) & ${p2['wt_mlp_lag']['step_overhead_pct']:+.1f}\\%$ / {p2['wt_mlp_lag']['peak_gib']:.1f} \\\\",
        r"\addlinespace[2.5pt]",
        # Phase 3
        r"\multicolumn{8}{@{}l}{\textcolor{famOD}{\pagesfsb Phase 3A/3B: $130\text{M}$ $4\times$ Chinchilla ($80$ TPP, $10.74\text{B}$ tok, $20{,}480$ upd., Seed 0) --- Overtraining Horizon}} \\",
        f"\\taxid{{MuonT}} (3A ref) & $.0106$ & ${runs['wave3a/muont_130m_4x_seed0']['final_sel']:.4f}$ & ${runs['wave3a/muont_130m_4x_seed0']['late11_sel']:.4f}$ (ref) & ${runs['wave3a/muont_130m_4x_seed0']['final_c4']:.4f}$ & $0$ ($10{{,}}240$ upd.) & ${pp['p3b_wt_vs_muont']['macro_b']:.4f}$ (ref) & $0.0\\%$ / {runs['wave3a/muont_130m_4x_seed0']['peak_gib']:.1f} \\\\",
        row_pair(r"\taxid{Muon} (3A)", ".0106", pairs["p3_muon_vs_muont"], f"${pp['p3_muon_vs_muont']['d_macro']:+.4f}$ ({pp['p3_muon_vs_muont']['domain_wins']}/16)", runs["wave3a/muon_130m_4x_seed0"]["peak_gib"]),
        row_pair(r"\taxid{OD-All} (3B matched)", ".0106", pairs["p3b_od_vs_muont"], f"${pp['p3b_od_vs_muont']['d_macro']:+.4f}$ ({pp['p3b_od_vs_muont']['domain_wins']}/16)", runs["wave3b/od_all_130m_4x_seed0_lr0p01056"]["peak_gib"]),
        row_pair(r"\taxid{WT-MLP-Lag} (3B matched)", ".0106", pairs["p3b_wt_vs_muont"], f"$\\mathbf{{{pp['p3b_wt_vs_muont']['d_macro']:+.4f}}}$ ({pp['p3b_wt_vs_muont']['domain_wins']}/16)", runs["wave3b/wt_mlp_lag_130m_4x_seed0_lr0p01056"]["peak_gib"], bold=True, shade=True),
        row_pair(r"\taxid{WT-MLP-Lag} (3A $+41\%$ LR)", ".0149", pairs["p3a_hot_wt_vs_muont"], f"${pp['p3a_hot_wt_vs_muont']['d_macro']:+.4f}$ ({pp['p3a_hot_wt_vs_muont']['domain_wins']}/16)", runs["wave3a/wt_mlp_lag_130m_4x_seed0"]["peak_gib"]),
        r"\addlinespace[2.5pt]",
        # Phase 4
        r"\multicolumn{8}{@{}l}{\textcolor{famDT}{\pagesfsb Phase 4: $520\text{M}$ $1\times$ Chinchilla ($20$ TPP, $10.74\text{B}$ tok, $1\text{M}$ batch, $10{,}240$ upd., Seed 0) --- Pinned Matched LR}} \\",
        f"\\taxid{{MuonT}} (4A ref) & $.0226$ & ${runs['wave4a/muont_520m_1x_seed0']['final_sel']:.4f}$ & ${runs['wave4a/muont_520m_1x_seed0']['late11_sel']:.4f}$ (ref) & ${runs['wave4a/muont_520m_1x_seed0']['final_c4']:.4f}$ & $0$ ($5{{,}}120$ upd.) & ${pp['p4_wt_vs_muont']['macro_b']:.4f}$ (ref) & $0.0\\%$ / {runs['wave4a/muont_520m_1x_seed0']['peak_gib']:.1f} \\\\",
        row_pair(r"\taxid{Muon} (4A pinned)", ".016", pairs["p4_muon_vs_muont"], f"${pp['p4_muon_vs_muont']['d_macro']:+.4f}$ ({pp['p4_muon_vs_muont']['domain_wins']}/16)", runs["wave4a/muon_520m_1x_seed0"]["peak_gib"]),
        row_pair(r"\taxid{OD-All} (4A matched)", ".0226", pairs["p4_od_vs_muont"], f"${pp['p4_od_vs_muont']['d_macro']:+.4f}$ ({pp['p4_od_vs_muont']['domain_wins']}/16)", runs["wave4a/od_all_520m_1x_seed0"]["peak_gib"]),
        row_pair(r"\taxid{WT-MLP-Lag} (4A matched)", ".0226", pairs["p4_wt_vs_muont"], f"$\\mathbf{{{pp['p4_wt_vs_muont']['d_macro']:+.4f}}}$ ({pp['p4_wt_vs_muont']['domain_wins']}/16)", runs["wave4a/wt_mlp_lag_520m_1x_seed0"]["peak_gib"], bold=True, shade=True),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    TABLE_OUT.write_text("\n".join(tlines) + "\n")

    # 3. pgfplots .dat files
    lr_rows = [
        "lr muon muont od_all wt_mlp_lag null_wt",
        f"0.016 {runs['ref_muon_0p016']['final_sel']:.6f} {runs['wave1a/muont_130m_1x_seed0_lr0p016']['final_sel']:.6f} nan {runs['wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016']['final_sel']:.6f} nan",
        f"0.02262741696 {runs['wave1b/muon_130m_1x_seed0_lr0p0226']['final_sel']:.6f} {runs['ref_muont_0p0226']['final_sel']:.6f} {runs['wave1b/od_all_130m_1x_seed0_lr0p0226']['final_sel']:.6f} {runs['ref_wt_0p0226']['final_sel']:.6f} {runs['wave1b/null_wt_130m_1x_seed0_lr0p0226']['final_sel']:.6f}",
        f"0.032 nan {runs['wave1a/muont_130m_1x_seed0_lr0p032']['final_sel']:.6f} {runs['wave1b/od_all_130m_1x_seed0_lr0p032']['final_sel']:.6f} {runs['wave1a/wt_mlp_lag_130m_1x_seed0_lr0p032']['final_sel']:.6f} nan",
    ]
    LR_DAT.write_text("\n".join(lr_rows) + "\n")

    traj_lines = ["progress wt_130_1x wt_130_4x od_130_4x wt_520_1x od_520_1x"]
    for row in data["trajectory_eval_deltas"]:
        traj_lines.append(
            f"{row['progress']:.5f} {row['wt_130_1x']:.6f} {row['wt_130_4x']:.6f} {row['od_130_4x']:.6f} {row['wt_520_1x']:.6f} {row['od_520_1x']:.6f}"
        )
    TRAJ_DAT.write_text("\n".join(traj_lines) + "\n")

    pal_lines = [
        "idx regime muon od_all wt_mlp_lag",
        f"0 130M_1x_3seed {p2['muon']['d_paloma_vs_muont_mean']:.6f} {p2['od_all']['d_paloma_vs_muont_mean']:.6f} {p2['wt_mlp_lag']['d_paloma_vs_muont_mean']:.6f}",
        f"1 130M_4x_80TPP {pp['p3_muon_vs_muont']['d_macro']:.6f} {pp['p3b_od_vs_muont']['d_macro']:.6f} {pp['p3b_wt_vs_muont']['d_macro']:.6f}",
        f"2 520M_1x_20TPP {pp['p4_muon_vs_muont']['d_macro']:.6f} {pp['p4_od_vs_muont']['d_macro']:.6f} {pp['p4_wt_vs_muont']['d_macro']:.6f}",
    ]
    PALOMA_DAT.write_text("\n".join(pal_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-data", action="store_true", help="Rebuild .tex and .dat outputs from checked JSON only.")
    args = parser.parse_args()
    if args.from_data:
        data = json.loads(JSON_OUT.read_text())
    else:
        data = build_from_live()
        JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")
    emit_outputs(data)
    print(f"Wrote {JSON_OUT.name}, {MACROS_OUT.name}, {TABLE_OUT.name}, and 3 pgfplots .dat files.")


if __name__ == "__main__":
    main()
