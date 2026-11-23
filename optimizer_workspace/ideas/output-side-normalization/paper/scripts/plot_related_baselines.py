# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export, tabulate, and plot the paired related-baseline comparison (Mousse & GO-MUON).

Reads the completed 18-trial seed-0 short LR screen and 12-trial seed-1/2 full-budget
comparison campaigns, writes checked full-precision per-update training histories,
provenance and summary JSON, LaTeX macros and comparison table, and renders a
three-panel publication figure highlighting training-loss dynamics and LR censoring.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

HERE = Path(__file__).resolve().parents[1]
PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
SHORT_ROOT = PVC_ROOT / "campaigns/related-baselines-20261121"
FULL_ROOT = PVC_ROOT / "campaigns/related-selected-full-seed12-20261121"

ARMS = (
    "muon_control",
    "inner_outer_mlpdiag",
    "mousse_control",
    "mousse_paper",
    "go_muon_control",
    "go_muon_paper",
)
COMPARE_ARMS = tuple(a for a in ARMS if a != "muon_control")
SEEDS = (1, 2)
SHORT_LRS = (0, 1, 2)

FIELDS = (
    "step",
    "tokens",
    *(f"s{seed}_{arm}" for seed in SEEDS for arm in ARMS),
)

ARM_META = {
    "muon_control": {
        "short_label": "Muon (ctrl)",
        "tex_label": r"\taxid{Muon} {\scriptsize(\texttt{muon\_ctrl})}",
        "geom": r"None ($\mathcal{N}_{10}$)",
        "group": "shared",
        "color": "#24313A",
        "ls": "-",
    },
    "inner_outer_mlpdiag": {
        "short_label": "WT-MLP (ours)",
        "tex_label": r"\textbf{\taxid{WT-MLP}} {\scriptsize(\texttt{mlpdiag})}",
        "geom": r"Act/Cot $(\tfrac{1}{2},\tfrac{1}{8})$",
        "group": "shared",
        "color": "#1B4F9C",
        "ls": "-",
    },
    "mousse_control": {
        "short_label": "Mousse (ctrl)",
        "tex_label": r"\taxid{Mousse-Ctrl}",
        "geom": r"Grad Gram $(\tfrac{1}{8},\tfrac{1}{8})$",
        "group": "control",
        "color": "#0E7470",
        "ls": "-",
    },
    "go_muon_control": {
        "short_label": "GO-MUON (ctrl)",
        "tex_label": r"\taxid{GO-MUON-Ctrl}",
        "geom": r"Act/Cot $(\tfrac{1}{4},\tfrac{1}{4})$",
        "group": "control",
        "color": "#C0392B",
        "ls": "-",
    },
    "mousse_paper": {
        "short_label": "Mousse (paper)",
        "tex_label": r"\taxid{Mousse-Paper}",
        "geom": r"Grad Gram $(\tfrac{1}{8},\tfrac{1}{8})$",
        "group": "paper",
        "color": "#4CA8A1",
        "ls": "--",
    },
    "go_muon_paper": {
        "short_label": "GO-MUON (paper)",
        "tex_label": r"\taxid{GO-MUON-Paper}",
        "geom": r"Act/Cot $(\tfrac{1}{4},\tfrac{1}{4})$",
        "group": "paper",
        "color": "#B97720",
        "ls": "--",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lag1_inflated_se(diffs: list[float]) -> float:
    n = len(diffs)
    if n < 2:
        return 0.0
    mean = statistics.fmean(diffs)
    centered = [x - mean for x in diffs]
    var = sum(x * x for x in centered) / (n - 1)
    denom = sum(x * x for x in centered)
    if denom <= 0:
        return 0.0
    num = sum(centered[i] * centered[i + 1] for i in range(n - 1))
    rho = max(0.0, min(0.99, num / denom))
    return math.sqrt((var / n) * ((1.0 + rho) / (1.0 - rho)))


def load_run(run_dir: Path, expected_arm: str, expected_steps: int) -> tuple[dict, list[dict], dict]:
    manifest_path = run_dir / "run.json"
    metrics_path = run_dir / "metrics_from_000000.jsonl"
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    steps = config["training"]["steps"]
    batch_tokens = config["training"]["batch_tokens"]
    if (
        manifest["status"] != "complete"
        or manifest["completed_steps"] != expected_steps
        or steps != expected_steps
        or manifest.get("execution_check", False)
        or config["experiment"]["arm"] != expected_arm
    ):
        raise ValueError(f"{run_dir}: expected completed {expected_steps}-step run for {expected_arm}")
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if len(rows) != expected_steps:
        raise ValueError(f"{run_dir}: expected {expected_steps} rows, got {len(rows)}")
    for idx, row in enumerate(rows, 1):
        if row["step"] != idx or row["tokens"] != idx * batch_tokens:
            raise ValueError(f"{run_dir}: step/token mismatch at {idx}")
        if not math.isfinite(row["train_loss"]) or row["train_loss"] < 0:
            raise ValueError(f"{run_dir}: invalid train_loss at {idx}")
    source = {
        "run": run_dir.name,
        "arm": expected_arm,
        "seed": config["training"]["seed"],
        "steps": steps,
        "matrix_lr": config["optimizer"]["lr"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256(metrics_path),
        "source_revision": manifest["source_revision"],
        "data_receipt_sha256": manifest["data_receipt_sha256"],
        "gpu": manifest["gpu"],
        "world_size": manifest["world_size"],
    }
    return manifest, rows, source


def eval_summary(rows: list[dict], total_tokens: int) -> dict:
    eval_rows = [r for r in rows if "selection_loss" in r]
    if not eval_rows:
        raise ValueError("No selection_loss evaluations found")
    half = total_tokens / 2.0
    late_rows = [r for r in eval_rows if r["tokens"] > half]
    step251 = next((r for r in eval_rows if r["step"] == 251), None)
    return {
        "n_evals": len(eval_rows),
        "n_late": len(late_rows),
        "final_selection_loss": eval_rows[-1]["selection_loss"],
        "final_c4_loss": eval_rows[-1]["c4_loss"],
        "late_selection_loss": statistics.fmean(r["selection_loss"] for r in late_rows) if late_rows else None,
        "late_c4_loss": statistics.fmean(r["c4_loss"] for r in late_rows) if late_rows else None,
        "step251_selection_loss": step251["selection_loss"] if step251 else None,
    }


def crossover_diagnostics(diffs: list[float]) -> dict:
    """Summarize early lead and crossover step for arm - muon_control."""
    min_val = min(diffs[:250])
    min_step = diffs[:250].index(min_val) + 1
    first_pos_after_50 = None
    sustained_pos = None
    for idx in range(50, len(diffs) - 20):
        if diffs[idx] > 0:
            if first_pos_after_50 is None:
                first_pos_after_50 = idx + 1
            if all(d > 0 for d in diffs[idx : idx + 20]):
                sustained_pos = idx + 1
                break
    return {
        "min_delta_early": min_val,
        "min_delta_step": min_step,
        "delta_step50": diffs[49] if len(diffs) >= 50 else None,
        "delta_step100": diffs[99] if len(diffs) >= 100 else None,
        "delta_step150": diffs[149] if len(diffs) >= 150 else None,
        "delta_step180": diffs[179] if len(diffs) >= 180 else None,
        "delta_step200": diffs[199] if len(diffs) >= 200 else None,
        "first_positive_step_after_50": first_pos_after_50,
        "sustained_positive_step": sustained_pos,
    }


def load_campaigns(short_root: Path, full_root: Path) -> tuple[list[dict], dict]:
    sources = []

    # 1. Short 320-step seed-0 LR screen (18 runs)
    short_runs: dict[tuple[str, int], dict] = {}
    short_histories: dict[tuple[str, int], list[dict]] = {}
    for lr_idx in SHORT_LRS:
        for arm in ARMS:
            run_dir = short_root / "runs" / f"{arm}_0p5m_seed0_lr{lr_idx}"
            manifest, rows, src = load_run(run_dir, arm, expected_steps=320)
            sources.append(src)
            total_tokens = manifest["config"]["training"]["tokens"]
            ev = eval_summary(rows, total_tokens)
            short_runs[(arm, lr_idx)] = {
                "matrix_lr": manifest["config"]["optimizer"]["lr"],
                "final_selection_loss": ev["final_selection_loss"],
                "final_c4_loss": ev["final_c4_loss"],
                "total_seconds": rows[-1].get("end_to_end_seconds", manifest["elapsed_seconds"]),
            }
            short_histories[(arm, lr_idx)] = rows

    short_screen: dict[str, dict] = {}
    for arm in ARMS:
        entries = {}
        for lr_idx in SHORT_LRS:
            rec = dict(short_runs[(arm, lr_idx)])
            base = short_runs[("muon_control", lr_idx)]
            rec["d_sel_vs_muon"] = rec["final_selection_loss"] - base["final_selection_loss"]
            rec["d_c4_vs_muon"] = rec["final_c4_loss"] - base["final_c4_loss"]
            if arm != "muon_control":
                diffs = [
                    a["train_loss"] - b["train_loss"]
                    for a, b in zip(short_histories[(arm, lr_idx)], short_histories[("muon_control", lr_idx)], strict=True)
                ]
                rec["early_diag"] = crossover_diagnostics(diffs)
            entries[f"lr{lr_idx}"] = rec
        best_idx = min(SHORT_LRS, key=lambda idx: entries[f"lr{idx}"]["final_selection_loss"])
        short_screen[arm] = {
            "selected_lr_index": best_idx,
            "selected_matrix_lr": entries[f"lr{best_idx}"]["matrix_lr"],
            "trials": entries,
        }

    # 2. Full 5,120-step seed-1 & seed-2 runs (12 runs)
    full_histories: dict[tuple[int, str], list[dict]] = {}
    full_manifests: dict[tuple[int, str], dict] = {}
    for seed in SEEDS:
        for arm in ARMS:
            run_dir = full_root / "runs" / f"{arm}_0p5m_seed{seed}_lr2"
            manifest, rows, src = load_run(run_dir, arm, expected_steps=5120)
            sources.append(src)
            full_histories[(seed, arm)] = rows
            full_manifests[(seed, arm)] = manifest

    # Verify strict data/protocol consistency across full runs within each seed
    stable_steps = full_manifests[(1, "muon_control")]["config"]["schedule"]["stable_steps"]
    total_tokens = full_manifests[(1, "muon_control")]["config"]["training"]["tokens"]
    batch_tokens = full_manifests[(1, "muon_control")]["config"]["training"]["batch_tokens"]
    selected_lr = full_manifests[(1, "muon_control")]["config"]["optimizer"]["lr"]

    for seed in SEEDS:
        ref = full_manifests[(seed, "muon_control")]
        for arm in ARMS:
            man = full_manifests[(seed, arm)]
            for field in ("data_receipt_sha256", "gpu", "world_size", "source_revision"):
                if man[field] != ref[field]:
                    raise ValueError(f"Seed {seed} {arm} differs in {field}")
            if abs(man["config"]["optimizer"]["lr"] - selected_lr) > 1e-12:
                raise ValueError(f"Seed {seed} {arm} matrix_lr mismatch")

    # Export exact 5,120-update training histories (all 6 arms x 2 seeds)
    exported_rows: list[dict] = []
    for idx in range(5120):
        row = {
            "step": idx + 1,
            "tokens": (idx + 1) * batch_tokens,
        }
        for seed in SEEDS:
            for arm in ARMS:
                row[f"s{seed}_{arm}"] = full_histories[(seed, arm)][idx]["train_loss"]
        exported_rows.append(row)

    # Compute per-arm full summaries & paired deltas vs muon_control
    full_summary: dict[str, dict] = {}
    for arm in ARMS:
        per_seed = {}
        for seed in SEEDS:
            rows = full_histories[(seed, arm)]
            base_rows = full_histories[(seed, "muon_control")]
            man = full_manifests[(seed, arm)]
            ev = eval_summary(rows, total_tokens)
            base_ev = eval_summary(base_rows, total_tokens)
            diffs = [r["train_loss"] - b["train_loss"] for r, b in zip(rows, base_rows, strict=True)]
            windows = [
                statistics.fmean(diffs[start : start + stable_steps])
                for start in range(0, len(diffs), stable_steps)
            ]
            win5_se = lag1_inflated_se(diffs[4 * stable_steps : 5 * stable_steps])
            step_times = [r["step_seconds"] for r in rows]
            peak_gib = max(r["peak_allocated_bytes"] for r in rows) / (1024**3)
            total_s = rows[-1].get("end_to_end_seconds", man["elapsed_seconds"])
            base_total_s = base_rows[-1].get("end_to_end_seconds", full_manifests[(seed, "muon_control")]["elapsed_seconds"])
            per_seed[str(seed)] = {
                "final_selection_loss": ev["final_selection_loss"],
                "late_selection_loss": ev["late_selection_loss"],
                "final_c4_loss": ev["final_c4_loss"],
                "late_c4_loss": ev["late_c4_loss"],
                "d_sel_final": ev["final_selection_loss"] - base_ev["final_selection_loss"],
                "d_sel_late": ev["late_selection_loss"] - base_ev["late_selection_loss"],
                "d_c4_final": ev["final_c4_loss"] - base_ev["final_c4_loss"],
                "d_c4_late": ev["late_c4_loss"] - base_ev["late_c4_loss"],
                "d_sel_step251": (
                    ev["step251_selection_loss"] - base_ev["step251_selection_loss"]
                    if ev["step251_selection_loss"] is not None and base_ev["step251_selection_loss"] is not None
                    else None
                ),
                "d_train_windows": windows,
                "d_train_win5_se": win5_se,
                "early_diag": crossover_diagnostics(diffs) if arm != "muon_control" else None,
                "total_seconds": total_s,
                "total_overhead_pct": 100.0 * (total_s / base_total_s - 1.0),
                "median_step_seconds": statistics.median(step_times),
                "peak_allocated_gib": peak_gib,
            }
        s1, s2 = per_seed["1"], per_seed["2"]
        mean_rec = {
            "final_selection_loss": statistics.fmean((s1["final_selection_loss"], s2["final_selection_loss"])),
            "late_selection_loss": statistics.fmean((s1["late_selection_loss"], s2["late_selection_loss"])),
            "final_c4_loss": statistics.fmean((s1["final_c4_loss"], s2["final_c4_loss"])),
            "late_c4_loss": statistics.fmean((s1["late_c4_loss"], s2["late_c4_loss"])),
            "d_sel_final": statistics.fmean((s1["d_sel_final"], s2["d_sel_final"])),
            "d_sel_late": statistics.fmean((s1["d_sel_late"], s2["d_sel_late"])),
            "d_c4_final": statistics.fmean((s1["d_c4_final"], s2["d_c4_final"])),
            "d_c4_late": statistics.fmean((s1["d_c4_late"], s2["d_c4_late"])),
            "d_sel_step251": (
                statistics.fmean((s1["d_sel_step251"], s2["d_sel_step251"]))
                if s1["d_sel_step251"] is not None and s2["d_sel_step251"] is not None
                else None
            ),
            "d_train_windows": [
                statistics.fmean((s1["d_train_windows"][i], s2["d_train_windows"][i])) for i in range(5)
            ],
            "total_seconds": statistics.fmean((s1["total_seconds"], s2["total_seconds"])),
            "total_overhead_pct": statistics.fmean((s1["total_overhead_pct"], s2["total_overhead_pct"])),
            "median_step_seconds": statistics.fmean((s1["median_step_seconds"], s2["median_step_seconds"])),
            "peak_allocated_gib": max(s1["peak_allocated_gib"], s2["peak_allocated_gib"]),
        }
        full_summary[arm] = {"per_seed": per_seed, "mean": mean_rec}

    # Explicit head-to-head differences of note
    go_paper_minus_ctrl = {
        "s1_final_sel": full_summary["go_muon_paper"]["per_seed"]["1"]["final_selection_loss"]
        - full_summary["go_muon_control"]["per_seed"]["1"]["final_selection_loss"],
        "s2_final_sel": full_summary["go_muon_paper"]["per_seed"]["2"]["final_selection_loss"]
        - full_summary["go_muon_control"]["per_seed"]["2"]["final_selection_loss"],
        "mean_final_sel": full_summary["go_muon_paper"]["mean"]["final_selection_loss"]
        - full_summary["go_muon_control"]["mean"]["final_selection_loss"],
    }

    receipt = {
        "short_campaign": str(short_root),
        "full_campaign": str(full_root),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "stable_steps": stable_steps,
        "full_steps": 5120,
        "short_steps": 320,
        "batch_tokens": batch_tokens,
        "total_tokens": total_tokens,
        "selected_matrix_lr": selected_lr,
        "matrix_decay": "isotropic",
        "weight_decay": 0.1,
        "rows": len(exported_rows),
        "smoothing": "none",
        "downsampling": "none",
        "path_simplification": False,
        "short_screen": short_screen,
        "full_runs": full_summary,
        "go_muon_paper_minus_control": go_paper_minus_ctrl,
        "sources": sources,
    }
    return exported_rows, receipt


def export_data(rows: list[dict], receipt: dict, data_path: Path, receipt_path: Path) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter=" ", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    receipt["data_sha256"] = sha256(data_path)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


def read_export(data_path: Path, receipt_path: Path) -> tuple[list[dict], dict]:
    receipt = json.loads(receipt_path.read_text())
    if sha256(data_path) != receipt["data_sha256"]:
        raise ValueError("Exported related-baselines training history differs from its receipt")
    with data_path.open(newline="") as handle:
        rows = [
            {k: int(v) if k in ("step", "tokens") else float(v) for k, v in row.items()}
            for row in csv.DictReader(handle, delimiter=" ")
        ]
    if len(rows) != receipt["rows"]:
        raise ValueError("Exported related-baselines training history has wrong row count")
    return rows, receipt


def fmt_signed(val: float, decimals: int = 4) -> str:
    if abs(val) < 0.5 * (10 ** (-decimals)):
        return f"{0.0:.{decimals}f}"
    return f"{val:+.{decimals}f}"


def write_macros(receipt: dict, macro_path: Path) -> None:
    full = receipt["full_runs"]
    short = receipt["short_screen"]
    gp_diff = receipt["go_muon_paper_minus_control"]

    def m(name: str, val: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{val}}}"

    gc = full["go_muon_control"]
    gp = full["go_muon_paper"]
    mc = full["mousse_control"]
    mp = full["mousse_paper"]
    wt = full["inner_outer_mlpdiag"]
    mu = full["muon_control"]

    lines = [
        "% Generated by scripts/plot_related_baselines.py. Do not edit.",
        # Full-run final selection deltas vs muon_control
        m("relWtFinalSelDeltaSOne", fmt_signed(wt["per_seed"]["1"]["d_sel_final"], 4)),
        m("relWtFinalSelDeltaSTwo", fmt_signed(wt["per_seed"]["2"]["d_sel_final"], 4)),
        m("relWtFinalSelDeltaMean", fmt_signed(wt["mean"]["d_sel_final"], 4)),
        m("relWtLateSelDeltaMean", fmt_signed(wt["mean"]["d_sel_late"], 4)),
        m("relWtFinalCfourDeltaMean", fmt_signed(wt["mean"]["d_c4_final"], 4)),
        m("relMousseCtrlFinalSelDeltaSOne", fmt_signed(mc["per_seed"]["1"]["d_sel_final"], 4)),
        m("relMousseCtrlFinalSelDeltaSTwo", fmt_signed(mc["per_seed"]["2"]["d_sel_final"], 4)),
        m("relMousseCtrlFinalSelDeltaMean", fmt_signed(mc["mean"]["d_sel_final"], 4)),
        m("relMoussePaperFinalSelDeltaSOne", fmt_signed(mp["per_seed"]["1"]["d_sel_final"], 4)),
        m("relMoussePaperFinalSelDeltaSTwo", fmt_signed(mp["per_seed"]["2"]["d_sel_final"], 4)),
        m("relMoussePaperFinalSelDeltaMean", fmt_signed(mp["mean"]["d_sel_final"], 4)),
        m("relGoCtrlFinalSelDeltaSOne", fmt_signed(gc["per_seed"]["1"]["d_sel_final"], 4)),
        m("relGoCtrlFinalSelDeltaSTwo", fmt_signed(gc["per_seed"]["2"]["d_sel_final"], 4)),
        m("relGoCtrlFinalSelDeltaMean", fmt_signed(gc["mean"]["d_sel_final"], 4)),
        m("relGoCtrlFinalSelDeltaSix", fmt_signed(gc["mean"]["d_sel_final"], 6)),
        m("relGoCtrlLateSelDeltaMean", fmt_signed(gc["mean"]["d_sel_late"], 4)),
        m("relGoCtrlFinalCfourDeltaMean", fmt_signed(gc["mean"]["d_c4_final"], 4)),
        m("relGoPaperFinalSelDeltaSOne", fmt_signed(gp["per_seed"]["1"]["d_sel_final"], 4)),
        m("relGoPaperFinalSelDeltaSTwo", fmt_signed(gp["per_seed"]["2"]["d_sel_final"], 4)),
        m("relGoPaperFinalSelDeltaMean", fmt_signed(gp["mean"]["d_sel_final"], 4)),
        m("relGoPaperFinalSelDeltaSix", fmt_signed(gp["mean"]["d_sel_final"], 6)),
        m("relGoPaperLateSelDeltaMean", fmt_signed(gp["mean"]["d_sel_late"], 4)),
        m("relGoPaperFinalCfourDeltaMean", fmt_signed(gp["mean"]["d_c4_final"], 4)),
        m("relGoPaperMinusCtrlFinalSel", fmt_signed(gp_diff["mean_final_sel"], 4)),
        # Early lead & crossover diagnostics (GO-MUON control & paper)
        m("relGoCtrlEarlyMinSOne", fmt_signed(gc["per_seed"]["1"]["early_diag"]["min_delta_early"], 3)),
        m("relGoCtrlEarlyMinStepSOne", str(gc["per_seed"]["1"]["early_diag"]["min_delta_step"])),
        m("relGoCtrlEarlyMinSTwo", fmt_signed(gc["per_seed"]["2"]["early_diag"]["min_delta_early"], 3)),
        m("relGoCtrlEarlyMinStepSTwo", str(gc["per_seed"]["2"]["early_diag"]["min_delta_step"])),
        m("relGoCtrlStepFiftySOne", fmt_signed(gc["per_seed"]["1"]["early_diag"]["delta_step50"], 3)),
        m("relGoCtrlCrossStepSOne", str(gc["per_seed"]["1"]["early_diag"]["sustained_positive_step"])),
        m("relGoCtrlCrossStepSTwo", str(gc["per_seed"]["2"]["early_diag"]["sustained_positive_step"])),
        m("relGoPaperCrossStepSOne", str(gp["per_seed"]["1"]["early_diag"]["sustained_positive_step"])),
        m("relGoPaperCrossStepSTwo", str(gp["per_seed"]["2"]["early_diag"]["sustained_positive_step"])),
        m("relGoCtrlStepTwoFiveOneSel", fmt_signed(gc["mean"]["d_sel_step251"], 3)),
        # Window 1 & Window 5 paired training deltas (2-seed means)
        m("relGoCtrlWinOneMean", fmt_signed(gc["mean"]["d_train_windows"][0], 4)),
        m("relGoCtrlWinFiveMean", fmt_signed(gc["mean"]["d_train_windows"][4], 4)),
        m("relGoPaperWinOneMean", fmt_signed(gp["mean"]["d_train_windows"][0], 4)),
        m("relGoPaperWinFiveMean", fmt_signed(gp["mean"]["d_train_windows"][4], 4)),
        m("relWtWinOneMean", fmt_signed(wt["mean"]["d_train_windows"][0], 4)),
        m("relWtWinFiveMean", fmt_signed(wt["mean"]["d_train_windows"][4], 4)),
        # Short-run LR sensitivity for GO-MUON
        m("relGoCtrlShortLrZero", fmt_signed(short["go_muon_control"]["trials"]["lr0"]["d_sel_vs_muon"], 4)),
        m("relGoCtrlShortLrOne", fmt_signed(short["go_muon_control"]["trials"]["lr1"]["d_sel_vs_muon"], 4)),
        m("relGoCtrlShortLrTwo", fmt_signed(short["go_muon_control"]["trials"]["lr2"]["d_sel_vs_muon"], 4)),
        m("relGoPaperShortLrZero", fmt_signed(short["go_muon_paper"]["trials"]["lr0"]["d_sel_vs_muon"], 4)),
        m("relGoPaperShortLrOne", fmt_signed(short["go_muon_paper"]["trials"]["lr1"]["d_sel_vs_muon"], 4)),
        m("relGoPaperShortLrTwo", fmt_signed(short["go_muon_paper"]["trials"]["lr2"]["d_sel_vs_muon"], 4)),
        m("relWtShortLrTwo", fmt_signed(short["inner_outer_mlpdiag"]["trials"]["lr2"]["d_sel_vs_muon"], 4)),
        m("relMousseCtrlShortLrTwo", fmt_signed(short["mousse_control"]["trials"]["lr2"]["d_sel_vs_muon"], 4)),
        m("relMoussePaperShortLrTwo", fmt_signed(short["mousse_paper"]["trials"]["lr2"]["d_sel_vs_muon"], 4)),
        # Wall-clock runtime (mean total seconds)
        m("relMuonTotalS", f"{mu['mean']['total_seconds']:,.0f}"),
        m("relWtTotalS", f"{wt['mean']['total_seconds']:,.0f}"),
        m("relMousseCtrlTotalS", f"{mc['mean']['total_seconds']:,.0f}"),
        m("relMoussePaperTotalS", f"{mp['mean']['total_seconds']:,.0f}"),
        m("relGoCtrlTotalS", f"{gc['mean']['total_seconds']:,.0f}"),
        m("relGoPaperTotalS", f"{gp['mean']['total_seconds']:,.0f}"),
        m("relGoCtrlOverheadPct", f"{gc['mean']['total_overhead_pct']:+.1f}"),
        m("relGoPaperOverheadPct", f"{gp['mean']['total_overhead_pct']:+.1f}"),
    ]
    macro_path.write_text("\n".join(lines) + "\n")


def write_table(receipt: dict, table_path: Path) -> None:
    full = receipt["full_runs"]
    short = receipt["short_screen"]

    def row_tex(arm: str, highlight: bool = False) -> str:
        meta = ARM_META[arm]
        st = short[arm]["trials"]
        fr = full[arm]
        s1, s2, mn = fr["per_seed"]["1"], fr["per_seed"]["2"], fr["mean"]
        if arm == "muon_control":
            short_cell = f"${st['lr0']['final_selection_loss']:.3f} / {st['lr1']['final_selection_loss']:.3f} / {st['lr2']['final_selection_loss']:.3f}$"
            s12_sel = f"${s1['final_selection_loss']:.4f} / {s2['final_selection_loss']:.4f}$"
            mn_sel = r"$0.0000$"
            late_sel = r"$0.0000$"
            c4_cell = f"${mn['final_c4_loss']:.4f}$"
            tr_cell = r"$0.000 \to 0.000$"
            time_cell = f"${mn['total_seconds']:,.0f}$ (ref)"
        else:
            short_cell = (
                f"${fmt_signed(st['lr0']['d_sel_vs_muon'], 3)} / "
                f"{fmt_signed(st['lr1']['d_sel_vs_muon'], 3)} / "
                f"{fmt_signed(st['lr2']['d_sel_vs_muon'], 3)}$"
            )
            s12_sel = f"${fmt_signed(s1['d_sel_final'], 4)} / {fmt_signed(s2['d_sel_final'], 4)}$"
            mn_val = fmt_signed(mn["d_sel_final"], 4)
            mn_sel = f"$\\mathbf{{{mn_val}}}$" if highlight else f"${mn_val}$"
            late_sel = f"${fmt_signed(mn['d_sel_late'], 4)}$"
            c4_cell = f"${fmt_signed(mn['d_c4_final'], 4)}$"
            tr_cell = (
                f"${fmt_signed(mn['d_train_windows'][0], 3)} \\to "
                f"{fmt_signed(mn['d_train_windows'][4], 3)}$"
            )
            time_cell = f"${mn['total_seconds']:,.0f}$ (${mn['total_overhead_pct']:+.0f}\\%$)"
        prefix = r"\rowcolor{pagepanel} " if highlight else ""
        return (
            f"{prefix}{meta['tex_label']} & {meta['geom']} & {short_cell} & "
            f"{s12_sel} & {mn_sel} & {late_sel} & {c4_cell} & "
            f"{tr_cell} & {time_cell} & ${mn['peak_allocated_gib']:.1f}$ \\\\"
        )

    lines = [
        "% Generated by scripts/plot_related_baselines.py. Do not edit.",
        r"\begin{tabular}{@{}llcccccccc@{}}",
        r"\toprule",
        r"Arm & Moments (in, out) & Short $\Delta$Sel ($.016/.023/.032$) & S1 / S2 $\Delta$Sel & Mean $\Delta$Sel & Late-11 $\Delta$Sel & Final $\Delta$C4 & $\Delta$Train (W1$\to$W5) & Total s (\%) & GiB \\",
        r"\midrule",
        r"\multicolumn{10}{@{}l}{\textcolor{famRef}{\pagesfsb I.\ Shared Controls (Deduplicated Across Both Panels; Baseline Row Reports Absolute Losses)}} \\",
        row_tex("muon_control"),
        row_tex("inner_outer_mlpdiag", highlight=True),
        r"\addlinespace[2.5pt]",
        r"\multicolumn{10}{@{}l}{\textcolor{famOD}{\pagesfsb II.\ Common-Setting Formula Controls (EMA $\beta=0.938$, Ridge $10^{-3}$, Refresh $4$, 10-Step BF16 NS)}} \\",
        row_tex("mousse_control"),
        row_tex("go_muon_control"),
        r"\addlinespace[2.5pt]",
        r"\multicolumn{10}{@{}l}{\textcolor{famAbl}{\pagesfsb III.\ Paper-Recipe Adapters (Author Numerics/Refresh; Common Model, Data, Schedule, Isotropic $\lambda=0.1$)}} \\",
        row_tex("mousse_paper"),
        row_tex("go_muon_paper"),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    table_path.write_text("\n".join(lines) + "\n")


def plot_figure(rows: list[dict], receipt: dict, output: Path) -> None:
    steps = [r["step"] for r in rows]
    stable = receipt["stable_steps"]
    short = receipt["short_screen"]
    full = receipt["full_runs"]

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "path.simplify": False,
        }
    )

    fig = Figure(figsize=(9.4, 7.1), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
    ax_full = fig.add_subplot(grid[0, :])
    ax_early = fig.add_subplot(grid[1, 0])
    ax_lr = fig.add_subplot(grid[1, 1])

    # Precompute exact per-update paired differences vs muon_control for both seeds
    diffs_s1 = {
        arm: [r[f"s1_{arm}"] - r["s1_muon_control"] for r in rows] for arm in COMPARE_ARMS
    }
    diffs_s2 = {
        arm: [r[f"s2_{arm}"] - r["s2_muon_control"] for r in rows] for arm in COMPARE_ARMS
    }

    draw_order = (
        "mousse_control",
        "mousse_paper",
        "inner_outer_mlpdiag",
        "go_muon_control",
        "go_muon_paper",
    )

    # Panel (a): Full-budget exact per-update paired training loss deltas (updates 1-5,120)
    ax_full.axhspan(-0.145, 0.0, color="#EEF3F8", alpha=0.55, zorder=0)
    ax_full.axhline(0.0, color="#24313A", linewidth=0.85, linestyle="-", zorder=2)
    ax_full.axvline(stable, color="#5F6B73", linewidth=0.85, linestyle="--", zorder=2)

    for arm in draw_order:
        meta = ARM_META[arm]
        lw_s1 = 0.85 if arm in ("inner_outer_mlpdiag", "go_muon_control") else 0.72
        ax_full.plot(
            steps,
            diffs_s2[arm],
            color=meta["color"],
            linewidth=0.55,
            linestyle=meta["ls"],
            alpha=0.42,
            zorder=3,
        )
        ax_full.plot(
            steps,
            diffs_s1[arm],
            color=meta["color"],
            linewidth=lw_s1,
            linestyle=meta["ls"],
            alpha=0.92,
            label=meta["short_label"],
            zorder=4,
        )

    ax_full.set_xlim(1, steps[-1])
    ax_full.set_ylim(-0.140, 0.092)
    ax_full.set_title(
        "(a) Exact full-budget paired training-loss trajectories vs. Muon (all 5,120 updates; Seed 1 dark, Seed 2 light)",
        loc="left",
        weight="bold",
    )
    ax_full.set_xlabel("Completed training updates (524,288 tokens/update; 2.68B tokens total)")
    ax_full.set_ylabel("Paired $\\Delta$ train loss: arm $-$ Muon (nats)")
    ax_full.annotate(
        "Linear LR decay begins (update 1,024)",
        xy=(stable, 0.080),
        xytext=(stable + 130, 0.079),
        color="#52606D",
        fontsize=7.8,
        va="center",
        arrowprops=dict(arrowstyle="-", color="#5F6B73", lw=0.65),
    )
    ax_full.text(
        5030,
        0.062,
        "GO-MUON ctrl & paper\noverlap (+0.033 to +0.046)",
        color="#8F2217",
        fontsize=7.6,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#E0C3BF", alpha=0.92),
    )
    ax_full.text(
        5030,
        -0.024,
        "WT-MLP stays below Muon\n(-0.005 to -0.007 in W5)",
        color="#1B4F9C",
        fontsize=7.6,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#C5D5EA", alpha=0.92),
    )
    ax_full.text(
        1120,
        -0.128,
        "Initial plunge reaches -0.50 (S1) / -0.64 (S2) nats in updates 17–19 (see panel b)",
        color="#52606D",
        fontsize=7.4,
        ha="left",
        va="bottom",
    )

    legend_handles = [
        Line2D([0], [0], color=ARM_META[a]["color"], lw=1.7, ls=ARM_META[a]["ls"], label=ARM_META[a]["short_label"])
        for a in ("inner_outer_mlpdiag", "mousse_control", "mousse_paper", "go_muon_control", "go_muon_paper")
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=5,
        frameon=True,
        facecolor="white",
        edgecolor="#D2D7DB",
        fontsize=8.0,
        handlelength=2.2,
        columnspacing=1.2,
    )

    # Panel (b): Early constant-LR reversal zoom (updates 1-400)
    early_end = 400
    early_steps = steps[:early_end]
    ax_early.axhspan(-0.34, 0.0, color="#EEF3F8", alpha=0.55, zorder=0)
    ax_early.axhline(0.0, color="#24313A", linewidth=0.85, zorder=2)

    for arm in draw_order:
        meta = ARM_META[arm]
        lw_s1 = 1.05 if arm in ("inner_outer_mlpdiag", "go_muon_control") else 0.85
        ax_early.plot(
            early_steps,
            diffs_s2[arm][:early_end],
            color=meta["color"],
            linewidth=0.65,
            linestyle=meta["ls"],
            alpha=0.45,
            zorder=3,
        )
        ax_early.plot(
            early_steps,
            diffs_s1[arm][:early_end],
            color=meta["color"],
            linewidth=lw_s1,
            linestyle=meta["ls"],
            alpha=0.95,
            zorder=4,
        )

    cross_s1 = full["go_muon_control"]["per_seed"]["1"]["early_diag"]["sustained_positive_step"]
    cross_paper_s1 = full["go_muon_paper"]["per_seed"]["1"]["early_diag"]["sustained_positive_step"]
    min_s1 = full["go_muon_control"]["per_seed"]["1"]["early_diag"]["min_delta_early"]
    min_step_s1 = full["go_muon_control"]["per_seed"]["1"]["early_diag"]["min_delta_step"]
    min_s2 = full["go_muon_control"]["per_seed"]["2"]["early_diag"]["min_delta_early"]
    ax_early.axvspan(cross_paper_s1, cross_s1, color="#C0392B", alpha=0.10, zorder=1)
    ax_early.axvline(cross_s1, color="#C0392B", linewidth=0.8, linestyle=":", zorder=2)
    ax_early.annotate(
        f"GO-MUON crosses zero\n(steps {cross_paper_s1}–{cross_s1}, flat LR)",
        xy=(cross_paper_s1, 0.002),
        xytext=(42, 0.056),
        color="#8F2217",
        fontsize=7.6,
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.85, connectionstyle="arc3,rad=-0.12"),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#E0C3BF", alpha=0.94),
    )
    ax_early.text(
        392,
        -0.318,
        f"Early peak lead at step {min_step_s1}:\n{min_s1:+.2f} (S1) / {min_s2:+.2f} (S2) nats",
        color="#52606D",
        fontsize=7.3,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#D2D7DB", alpha=0.92),
    )
    ax_early.set_xlim(1, early_end)
    ax_early.set_ylim(-0.335, 0.125)
    ax_early.set_title("(b) Constant-LR early zoom: updates 1–400", loc="left", weight="bold")
    ax_early.set_xlabel("Completed training updates (flat LR = 0.032)")
    ax_early.set_ylabel("Paired $\\Delta$ train loss: arm $-$ Muon (nats)")

    # Panel (c): 320-step Seed-0 LR sensitivity & upper-boundary censoring
    lrs = [short["muon_control"]["trials"][f"lr{i}"]["matrix_lr"] for i in SHORT_LRS]
    ax_lr.axhspan(-0.16, 0.0, color="#EEF3F8", alpha=0.55, zorder=0)
    ax_lr.axhline(0.0, color="#24313A", linewidth=0.85, zorder=2)
    ax_lr.axvline(lrs[-1], color="#5F6B73", linewidth=0.9, linestyle="--", zorder=2)
    ax_lr.axvspan(lrs[-1], 0.036, color="#F3E5D0", alpha=0.45, zorder=1)

    markers = {
        "inner_outer_mlpdiag": "o",
        "mousse_control": "s",
        "mousse_paper": "D",
        "go_muon_control": "^",
        "go_muon_paper": "v",
    }
    for arm in draw_order:
        meta = ARM_META[arm]
        ys = [short[arm]["trials"][f"lr{i}"]["d_sel_vs_muon"] for i in SHORT_LRS]
        ax_lr.plot(
            lrs,
            ys,
            color=meta["color"],
            linestyle=meta["ls"],
            marker=markers[arm],
            markersize=5.2,
            linewidth=1.4,
            label=meta["short_label"],
            zorder=4,
        )

    ax_lr.annotate(
        "GO-MUON deficit narrows\nas LR increases",
        xy=(lrs[1], short["go_muon_control"]["trials"]["lr1"]["d_sel_vs_muon"]),
        xytext=(0.0238, 0.015),
        color="#8F2217",
        fontsize=7.4,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8, connectionstyle="arc3,rad=0.12"),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#E0C3BF", alpha=0.94),
    )
    ax_lr.text(
        lrs[-1] * 0.985,
        0.130,
        "Upper grid boundary (0.032)\nselected by all 6 arms",
        color="#52606D",
        fontsize=7.3,
        ha="right",
        va="top",
    )
    ax_lr.set_xscale("log", base=2)
    ax_lr.set_xlim(0.0148, 0.0355)
    ax_lr.set_xticks(lrs)
    ax_lr.set_xticklabels(["0.016", "0.0226", "0.032"])
    ax_lr.set_ylim(-0.155, 0.140)
    ax_lr.set_title("(c) 320-step LR screen: boundary censoring (Seed 0)", loc="left", weight="bold")
    ax_lr.set_xlabel("Matrix learning rate $\\eta$ (log$_2$ scale)")
    ax_lr.set_ylabel("Final 320-step $\\Delta$ selection loss vs. Muon (nats)")

    for ax in (ax_full, ax_early, ax_lr):
        ax.grid(color="#D2D7DB", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=8)
    ax_full.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax_early.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-root", type=Path, default=SHORT_ROOT)
    parser.add_argument("--full-root", type=Path, default=FULL_ROOT)
    parser.add_argument("--from-data", action="store_true", help="Rebuild outputs offline from checked export.")
    parser.add_argument("--data", type=Path, default=HERE / "figs/related_baselines_training.dat")
    parser.add_argument("--receipt", type=Path, default=HERE / "results/related_baselines.json")
    parser.add_argument("--macros", type=Path, default=HERE / "results/related_numbers.tex")
    parser.add_argument("--table", type=Path, default=HERE / "results/table_related.tex")
    parser.add_argument("--figure", type=Path, default=HERE / "figs/related_baselines_training.pdf")
    args = parser.parse_args()

    if args.from_data:
        rows, receipt = read_export(args.data, args.receipt)
    else:
        rows, receipt = load_campaigns(args.short_root, args.full_root)
        export_data(rows, receipt, args.data, args.receipt)

    write_macros(receipt, args.macros)
    write_table(receipt, args.table)
    plot_figure(rows, receipt, args.figure)
    print(f"Exported {len(rows)} exact updates x 6 arms x 2 seeds -> {args.figure}")


if __name__ == "__main__":
    main()
