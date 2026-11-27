# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render Hyperball campaign results in the exact column format of Paper v4 Table 4 (tab:main_scaling)."""

from __future__ import annotations

import gzip
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
CAMPAIGN_ROOT = PVC_ROOT / "campaigns" / "muonh-20261127"
LR_BRACKET_ROOT = PVC_ROOT / "campaigns" / "lr-bracket-20261125"
CONFIRM_ROOT = PVC_ROOT / "campaigns" / "confirmation-20261121"
WD_ROOT = PVC_ROOT / "campaigns" / "wd-ablation-20261127"
ARCHIVE_REC = Path("/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization/experiment_records")

RUN_DIRS_SEED0 = {
    ("muon", "0p016"): PVC_ROOT / "runs" / "wave0" / "muon",
    ("muont", "0p016"): CONFIRM_ROOT / "runs" / "wave1a" / "muont_130m_1x_seed0_lr0p016",
    ("od_all", "0p016"): LR_BRACKET_ROOT / "runs" / "od_all_130m_1x_seed0_lr0p016",
    ("wt_mlp_lag", "0p016"): CONFIRM_ROOT / "runs" / "wave1a" / "wt_mlp_lag_130m_1x_seed0_lr0p016",
    ("wt_mlp_lag_wd_iso", "0p016"): WD_ROOT / "runs" / "wt_mlp_lag_wd_iso_130m_1x_seed0_lr0p016",
    ("muonh", "0p0113"): CAMPAIGN_ROOT / "runs" / "muonh_130m_1x_seed0_lr0p0113",
    ("muonh", "0p016"): CAMPAIGN_ROOT / "runs" / "muonh_130m_1x_seed0_lr0p016",
    ("muonth", "0p0113"): CAMPAIGN_ROOT / "runs" / "muonth_130m_1x_seed0_lr0p0113",
    ("muonth", "0p016"): CAMPAIGN_ROOT / "runs" / "muonth_130m_1x_seed0_lr0p016",
    ("wt_mlp_lag_muonh", "0p016"): CAMPAIGN_ROOT / "runs" / "wt_mlp_lag_muonh_130m_1x_seed0_lr0p016",
    ("wt_mlp_lag_muonh", "0p0113"): CAMPAIGN_ROOT / "runs" / "wt_mlp_lag_muonh_130m_1x_seed0_lr0p0113",
}

SEEDS123_DIRS = {
    ("muon", "0p016"): [CONFIRM_ROOT / "runs" / f"wave2{c}" / f"muon_130m_1x_seed{s}" for s, c in ((1, "a"), (2, "b"), (3, "c"))],
    ("muont", "0p016"): [CONFIRM_ROOT / "runs" / f"wave2{c}" / f"muont_130m_1x_seed{s}" for s, c in ((1, "a"), (2, "b"), (3, "c"))],
    ("od_all", "0p016"): [LR_BRACKET_ROOT / "runs" / f"od_all_130m_1x_seed{s}_lr0p016" for s in (1, 2, 3)],
    ("wt_mlp_lag", "0p016"): [LR_BRACKET_ROOT / "runs" / f"wt_mlp_lag_130m_1x_seed{s}_lr0p016" for s in (1, 2, 3)],
    ("wt_mlp_lag_wd_iso", "0p016"): [WD_ROOT / "runs" / f"wt_mlp_lag_wd_iso_130m_1x_seed{s}_lr0p016" for s in (1, 2, 3)],
    ("muonh", "0p0113"): [CAMPAIGN_ROOT / "runs" / f"muonh_130m_1x_seed{s}_lr0p0113" for s in (1, 2, 3)],
    ("muonth", "0p0113"): [CAMPAIGN_ROOT / "runs" / f"muonth_130m_1x_seed{s}_lr0p0113" for s in (1, 2, 3)],
    ("wt_mlp_lag_muonh", "0p016"): [CAMPAIGN_ROOT / "runs" / f"wt_mlp_lag_muonh_130m_1x_seed{s}_lr0p016" for s in (1, 2, 3)],
    ("wt_mlp_lag_muonh", "0p0113"): [CAMPAIGN_ROOT / "runs" / f"wt_mlp_lag_muonh_130m_1x_seed{s}_lr0p0113" for s in (1, 2, 3)],
}

PALOMA_SEEDS123 = {
    ("muon", "0p016"): [ARCHIVE_REC / "paloma_eval" / f"muon_130m_1x_seed{s}.json" for s in (1, 2, 3)],
    ("muont", "0p016"): [ARCHIVE_REC / "paloma_eval" / f"muont_130m_1x_seed{s}.json" for s in (1, 2, 3)],
    ("od_all", "0p016"): [LR_BRACKET_ROOT / "paloma_eval" / f"od_all_130m_1x_seed{s}_lr0p016.json" for s in (1, 2, 3)],
    ("wt_mlp_lag", "0p016"): [LR_BRACKET_ROOT / "paloma_eval" / f"wt_mlp_lag_130m_1x_seed{s}_lr0p016.json" for s in (1, 2, 3)],
    ("wt_mlp_lag_wd_iso", "0p016"): [WD_ROOT / "paloma_eval" / f"wt_mlp_lag_wd_iso_130m_1x_seed{s}_lr0p016.json" for s in (1, 2, 3)],
    ("muonh", "0p0113"): [CAMPAIGN_ROOT / "paloma_eval" / f"muonh_130m_1x_seed{s}_lr0p0113.json" for s in (1, 2, 3)],
    ("muonth", "0p0113"): [CAMPAIGN_ROOT / "paloma_eval" / f"muonth_130m_1x_seed{s}_lr0p0113.json" for s in (1, 2, 3)],
    ("wt_mlp_lag_muonh", "0p016"): [CAMPAIGN_ROOT / "paloma_eval" / f"wt_mlp_lag_muonh_130m_1x_seed{s}_lr0p016.json" for s in (1, 2, 3)],
    ("wt_mlp_lag_muonh", "0p0113"): [CAMPAIGN_ROOT / "paloma_eval" / f"wt_mlp_lag_muonh_130m_1x_seed{s}_lr0p0113.json" for s in (1, 2, 3)],
}


def load_complete_run(run_dir: Path) -> dict | None:
    rj = run_dir / "run.json"
    if not rj.is_file():
        return None
    meta = json.loads(rj.read_text())
    if meta.get("status") != "complete":
        return None
    rows = []
    plain = sorted(run_dir.glob("metrics_from_*.jsonl"))
    if plain:
        for p in plain:
            for ln in p.read_text(errors="ignore").splitlines():
                if ln.strip():
                    rows.append(json.loads(ln))
    else:
        for p in sorted(run_dir.glob("metrics_from_*.jsonl.gz")):
            with gzip.open(p, "rt") as fh:
                for ln in fh:
                    if ln.strip():
                        rows.append(json.loads(ln))
    by_s = {int(r["step"]): r for r in rows}
    if max(by_s, default=0) != 5120:
        return None
    evals = [by_s[s] for s in sorted(by_s) if "selection_loss" in by_s[s]]
    late = [r for r in evals if int(r["step"]) >= 2751]
    return {
        "lr": float(meta["config"]["optimizer"]["lr"]),
        "sel": float(evals[-1]["selection_loss"]),
        "c4": float(evals[-1]["c4_loss"]),
        "late11": statistics.mean(float(r["selection_loss"]) for r in late),
    }


def fmt_pm(vals: list[float]) -> str:
    return f"{statistics.mean(vals):+.2f} ± {statistics.stdev(vals):.2f}"


def paloma_domain_map(pal: dict) -> dict[str, float]:
    if "paloma_domains" in pal:
        return {k.removeprefix("paloma/"): float(v) for k, v in pal["paloma_domains"].items()}
    if "domains" in pal:
        return {k.removeprefix("paloma/"): float(v) for k, v in pal["domains"].items()}
    return {k.removeprefix("paloma/"): float(v) for k, v in pal.get("scores", {}).items() if k.startswith("paloma/")}


def render() -> str:
    s0 = {k: load_complete_run(p) for k, p in RUN_DIRS_SEED0.items()}
    ref0 = s0[("muon", "0p016")]
    assert ref0 is not None

    lines = [
        "# Hyperball Results in Paper v4 Table 4 Format (`130M`, `20 TPP = 2.68B tokens`)",
        "",
        f"Generated: `{datetime.now(UTC).isoformat()}`",
        "",
        "Conventions match Table 4 (`tab:main_scaling`): `Held-out` is final $\\mathcal{L}_{\\mathrm{sel}}$ in **nats**; all $\\Delta$ columns report differences from same-regime `Muon` ($\\eta^* = 0.01600$) in **millinats** (negative favors the row method).",
        "",
        "## 1. Seed 0 Verified Optima (`130M`, `20 TPP`, `2.68B` tokens, Seed 0)",
        "",
        "| Method | LR $\\eta$ | Held-out (nats) | $\\Delta$ Held-out | Late-11 $\\Delta$ | $\\Delta$ C4 | $\\Delta$ Paloma | Wins |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    seed0_rows = [
        ("Muon ($Q=W$)", ("muon", "0p016"), True),
        ("MuonT ($Q=W\\hat R^2$)", ("muont", "0p016"), False),
        ("Muon2-Diag ($Q=W\\hat R^2$)", ("od_all", "0p016"), False),
        ("**Muon2-ExpDiag** ($Q=W\\hat R^2$)", ("wt_mlp_lag", "0p016"), False),
        ("Muon2-ExpDiag ($Q=W$)", ("wt_mlp_lag_wd_iso", "0p016"), False),
        ("MuonH (Hyperball, $\\eta^*$)", ("muonh", "0p0113"), False),
        ("MuonTH (Hyperball, $\\eta^*$)", ("muonth", "0p0113"), False),
        ("**Muon2-ExpDiag + MuonH** ($\\eta^*$)", ("wt_mlp_lag_muonh", "0p016"), False),
        ("Muon2-ExpDiag + MuonH (matched $0.01131$)", ("wt_mlp_lag_muonh", "0p0113"), False),
        ("MuonH (matched $0.01600$)", ("muonh", "0p016"), False),
        ("MuonTH (matched $0.01600$)", ("muonth", "0p016"), False),
    ]
    for label, key, is_base in seed0_rows:
        r = s0.get(key)
        if r is None:
            continue
        if is_base:
            lines.append(f"| {label} | `{r['lr']:.5f}` | `{r['sel']:.4f}` | `+0.00` | `+0.00` | `+0.00` | `+0.00` | — |")
        else:
            dh = 1000.0 * (r["sel"] - ref0["sel"])
            dl = 1000.0 * (r["late11"] - ref0["late11"])
            dc = 1000.0 * (r["c4"] - ref0["c4"])
            lines.append(
                f"| {label} | `{r['lr']:.5f}` | `{r['sel']:.4f}` | `{dh:+.2f}` | `{dl:+.2f}` | `{dc:+.2f}` | *pending Stage 3* | — |"
            )

    lines += [
        "",
        "## 2. Fresh Seeds 1–3 Confirmation (`130M`, `20 TPP`, `2.68B` tokens, paired mean ± SD)",
        "",
        "| Method | LR $\\eta$ | Held-out (nats) | $\\Delta$ Held-out | Late-11 $\\Delta$ | $\\Delta$ C4 | $\\Delta$ Paloma | Wins |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    ref123 = [load_complete_run(d) for d in SEEDS123_DIRS[("muon", "0p016")]]
    ref_pal = [json.loads(p.read_text()) for p in PALOMA_SEEDS123[("muon", "0p016")]]

    s123_rows = [
        ("Muon ($Q=W$)", ("muon", "0p016"), True),
        ("MuonT ($Q=W\\hat R^2$)", ("muont", "0p016"), False),
        ("Muon2-Diag ($Q=W\\hat R^2$)", ("od_all", "0p016"), False),
        ("**Muon2-ExpDiag** ($Q=W\\hat R^2$)", ("wt_mlp_lag", "0p016"), False),
        ("Muon2-ExpDiag ($Q=W$)", ("wt_mlp_lag_wd_iso", "0p016"), False),
        ("MuonH (Hyperball, $\\eta^*$)", ("muonh", "0p0113"), False),
        ("MuonTH (Hyperball, $\\eta^*$)", ("muonth", "0p0113"), False),
        ("**Muon2-ExpDiag + MuonH** ($\\eta^*$)", ("wt_mlp_lag_muonh", "0p016"), False),
        ("Muon2-ExpDiag + MuonH (matched $0.01131$)", ("wt_mlp_lag_muonh", "0p0113"), False),
    ]
    for label, key, is_base in s123_rows:
        runs = [load_complete_run(d) for d in SEEDS123_DIRS[key]]
        if not all(runs):
            done_n = sum(r is not None for r in runs)
            lines.append(f"| {label} | — | *running ({done_n}/3 seeds)* | — | — | — | — | — |")
            continue
        held_mean = statistics.mean(r["sel"] for r in runs)
        lr_val = runs[0]["lr"]
        if is_base:
            lines.append(f"| {label} | `{lr_val:.5f}` | `{held_mean:.4f}` | `+0.00` | `+0.00` | `+0.00` | `+0.00` | — |")
            continue
        dh = [1000.0 * (a["sel"] - b["sel"]) for a, b in zip(runs, ref123)]
        dl = [1000.0 * (a["late11"] - b["late11"]) for a, b in zip(runs, ref123)]
        dc = [1000.0 * (a["c4"] - b["c4"]) for a, b in zip(runs, ref123)]
        pal_paths = PALOMA_SEEDS123[key]
        if all(p.is_file() for p in pal_paths):
            pals = [json.loads(p.read_text()) for p in pal_paths]
            dp = [1000.0 * (a["paloma_macro_mean"] - b["paloma_macro_mean"]) for a, b in zip(pals, ref_pal)]
            p_maps = [paloma_domain_map(p) for p in pals]
            r_maps = [paloma_domain_map(rp) for rp in ref_pal]
            doms = sorted(p_maps[0].keys())
            wins = sum(
                statistics.mean(pm[dom] for pm in p_maps) < statistics.mean(rm[dom] for rm in r_maps)
                for dom in doms
            )
            pal_s = f"`{fmt_pm(dp)}`"
            win_s = f"`{wins}/16`"
        else:
            pal_s = "*pending Stage 3*"
            win_s = "—"
        lines.append(
            f"| {label} | `{lr_val:.5f}` | `{held_mean:.4f}` | `{fmt_pm(dh)}` | `{fmt_pm(dl)}` | `{fmt_pm(dc)}` | {pal_s} | {win_s} |"
        )

    out_md = "\n".join(lines) + "\n"
    (CAMPAIGN_ROOT / "TABLE4_MUONH_RESULTS.md").write_text(out_md)
    (TASK / "results" / "TABLE4_MUONH_RESULTS.md").write_text(out_md)
    return out_md


if __name__ == "__main__":
    print(render())
