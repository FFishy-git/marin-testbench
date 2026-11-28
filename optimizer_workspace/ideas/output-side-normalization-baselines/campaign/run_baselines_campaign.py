# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""4-node (32xA100) campaign orchestrator for 130M 1x Chinchilla Muon-family baselines."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))

from campaign.configure_baselines_campaign import (
    BOUNDARY_LRS,
    LR_BY_TAG,
    PRIMARY_ARMS,
    SECONDARY_ARMS,
    STAGE1_LRS,
    generate_all,
    run_name,
)
from campaign.deploy_baselines_campaign import deploy

PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
CAMPAIGN_ROOT = PVC_ROOT / "campaigns" / "baselines-20261128"
CONFIRM_ROOT = PVC_ROOT / "campaigns" / "confirmation-20261121"
LR_BRACKET_ROOT = PVC_ROOT / "campaigns" / "lr-bracket-20261125"
MUONH_ROOT = PVC_ROOT / "campaigns" / "muonh-20261127"
DATA_DIR = PVC_ROOT / "data"
DATA_RECEIPT = DATA_DIR / "experiment_receipt.json"

NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]
ORCH_GEN = 101
LOG_LOCK = threading.Lock()

REFERENCE_RUNS: dict[str, dict[int, Path]] = {
    "Muon (WD=0.1) @ 0.016": {
        0: PVC_ROOT / "runs" / "wave0" / "muon",
        1: CONFIRM_ROOT / "runs" / "wave2a" / "muon_130m_1x_seed1",
        2: CONFIRM_ROOT / "runs" / "wave2b" / "muon_130m_1x_seed2",
        3: CONFIRM_ROOT / "runs" / "wave2c" / "muon_130m_1x_seed3",
    },
    "MuonT (WD=0.1) @ 0.016": {
        0: CONFIRM_ROOT / "runs" / "wave1a" / "muont_130m_1x_seed0_lr0p016",
        1: CONFIRM_ROOT / "runs" / "wave2a" / "muont_130m_1x_seed1",
        2: CONFIRM_ROOT / "runs" / "wave2b" / "muont_130m_1x_seed2",
        3: CONFIRM_ROOT / "runs" / "wave2c" / "muont_130m_1x_seed3",
    },
    "WT-MLP-Lag (WD=0.1) @ 0.016": {
        0: CONFIRM_ROOT / "runs" / "wave1a" / "wt_mlp_lag_130m_1x_seed0_lr0p016",
        1: LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed1_lr0p016",
        2: LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed2_lr0p016",
        3: LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed3_lr0p016",
    },
    "MuonH (WSD) @ 0.01131": {
        0: MUONH_ROOT / "runs" / "muonh_130m_1x_seed0_lr0p0113",
        1: MUONH_ROOT / "runs" / "muonh_130m_1x_seed1_lr0p0113",
        2: MUONH_ROOT / "runs" / "muonh_130m_1x_seed2_lr0p0113",
        3: MUONH_ROOT / "runs" / "muonh_130m_1x_seed3_lr0p0113",
    },
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def log(msg: str) -> None:
    line = f"[{now()}] [baselines] {msg}"
    with LOG_LOCK:
        print(line, flush=True)
        CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
        with (CAMPAIGN_ROOT / "orchestrator.log").open("a") as fh:
            fh.write(line + "\n")


def read_run_summary(run_dir: Path) -> dict | None:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return None
    meta = json.loads(run_json.read_text())
    if meta.get("status") != "complete":
        return None
    metrics_files = sorted(run_dir.glob("metrics_from_*.jsonl"))
    if not metrics_files:
        return None
    last_eval = None
    step_times = []
    peak_mem = 0
    for mf in metrics_files:
        for line in mf.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "selection_loss" in row:
                last_eval = row
            if row.get("step", 0) > 10 and "step_seconds" in row:
                step_times.append(float(row["step_seconds"]))
            if "peak_allocated_bytes" in row:
                peak_mem = max(peak_mem, int(row["peak_allocated_bytes"]))
    if last_eval is None:
        return None
    return {
        "selection_loss": float(last_eval["selection_loss"]),
        "c4_loss": float(last_eval.get("c4_loss", float("nan"))),
        "median_step_s": statistics.median(step_times) if step_times else float("nan"),
        "elapsed_s": float(meta.get("elapsed_seconds", float("nan"))),
        "peak_gb": peak_mem / (1024**3),
    }


def wait_for_campaign(completion_file: Path, poll_seconds: int = 60) -> None:
    log(f"Waiting for queue barrier: {completion_file}")
    while not completion_file.exists():
        time.sleep(poll_seconds)
    log(f"Queue barrier satisfied: {completion_file}")


def run_job_on_node(node: str, name: str, cfg_path: Path, out_dir: Path) -> tuple[str, int]:
    if read_run_summary(out_dir) is not None:
        log(f"[SKIP] {name} already complete at {out_dir}")
        return name, 0

    alloc_dir = CAMPAIGN_ROOT / "allocations"
    alloc_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    node_tag = node.replace(".", "_")
    state_path = alloc_dir / f"train_{name}_{ts}_{node_tag}.state.json"
    log_path = alloc_dir / f"train_{name}_{ts}_{node_tag}.log"

    cmd = [
        PYTHON,
        str(TASK / "campaign" / "allocation.py"),
        "--state",
        str(state_path),
        "--log",
        str(log_path),
        "--orch-gen",
        str(ORCH_GEN),
        "--",
        PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "experiment.train",
        "--config",
        str(cfg_path),
        "--data",
        str(DATA_DIR),
        "--data-receipt",
        str(DATA_RECEIPT),
        "--out",
        str(out_dir),
    ]
    local_ips = set(subprocess.check_output(["hostname", "-I"], text=True).split())
    log(f"[LAUNCH] {name} on {node}")
    if node in local_ips:
        proc = subprocess.run(cmd, cwd=str(TASK))
    else:
        remote_cmd = f"cd {shlex.quote(str(TASK))} && {shlex.join(cmd)}"
        proc = subprocess.run([*SSH, f"root@{node}", remote_cmd])
    log(f"[DONE] {name} on {node} rc={proc.returncode}")
    return name, proc.returncode


def dispatch_wave(jobs: list[tuple[str, Path, Path]], nodes: tuple[str, ...]) -> list[tuple[str, int]]:
    if not jobs:
        return []
    queue = list(jobs)
    q_lock = threading.Lock()
    results: list[tuple[str, int]] = []

    def worker(node: str) -> None:
        while True:
            with q_lock:
                if not queue:
                    return
                name, cfg_path, out_dir = queue.pop(0)
            res = run_job_on_node(node, name, cfg_path, out_dir)
            with q_lock:
                results.append(res)

    with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        futs = [pool.submit(worker, node) for node in nodes]
        for fut in as_completed(futs):
            fut.result()
    return results


def best_seed0_lr(arm_key: str) -> tuple[str | None, float]:
    best_tag, best_loss = None, float("inf")
    for lr_tag in LR_BY_TAG:
        out_dir = CAMPAIGN_ROOT / "runs" / run_name(arm_key, 0, lr_tag)
        summary = read_run_summary(out_dir)
        if summary and summary["selection_loss"] < best_loss:
            best_loss = summary["selection_loss"]
            best_tag = lr_tag
    return best_tag, best_loss


def write_results_report(arms: tuple[str, ...]) -> Path:
    lines = [
        "# 130M 1x Chinchilla Baseline Comparison Results (`5,120 steps`, `2.68B tokens`)",
        "",
        f"Generated: `{now()}`",
        "",
        "## 1. Seed 0 Full-Budget Learning-Rate Bracket (Held-out Selection Loss)",
        "",
        "| Method | `0.008` | `0.0113` | `0.016` | `0.0226` | `0.032` | Best Seed-0 (LR) | Step (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_key in arms:
        cells = []
        best_tag, best_loss, best_step = None, float("inf"), float("nan")
        for lr_tag in ("0p008", "0p0113", "0p016", "0p0226", "0p032"):
            s = read_run_summary(CAMPAIGN_ROOT / "runs" / run_name(arm_key, 0, lr_tag))
            if s is None:
                cells.append("—")
            else:
                cells.append(f"{s['selection_loss']:.5f}")
                if s["selection_loss"] < best_loss:
                    best_loss, best_tag, best_step = s["selection_loss"], lr_tag, s["median_step_s"]
        best_str = f"**{best_loss:.5f}** (`{best_tag}`)" if best_tag else "—"
        step_str = f"{best_step:.3f}" if best_tag else "—"
        lines.append(f"| `{arm_key}` | {' | '.join(cells)} | {best_str} | {step_str} |")

    lines += [
        "",
        "## 2. Reference Archived Baselines (130M 1x Chinchilla, Same Data & Schedule)",
        "",
        "| Reference Method | Seed 0 Held-out | Seeds 1–3 Mean ± SD | C4 Mean ± SD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, seed_map in REFERENCE_RUNS.items():
        s0 = read_run_summary(seed_map[0])
        s0_str = f"{s0['selection_loss']:.5f}" if s0 else "—"
        s13 = [read_run_summary(seed_map[k]) for k in (1, 2, 3)]
        s13_ok = [x for x in s13 if x is not None]
        if len(s13_ok) == 3:
            m = statistics.mean(x["selection_loss"] for x in s13_ok)
            sd = statistics.stdev(x["selection_loss"] for x in s13_ok)
            cm = statistics.mean(x["c4_loss"] for x in s13_ok)
            csd = statistics.stdev(x["c4_loss"] for x in s13_ok)
            s13_str = f"{m:.5f} ± {sd:.5f}"
            c4_str = f"{cm:.5f} ± {csd:.5f}"
        else:
            s13_str, c4_str = "—", "—"
        lines.append(f"| {label} | {s0_str} | {s13_str} | {c4_str} |")

    out_path = TASK / "results" / "BASELINES_RESULTS.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", choices=("stage1", "stage2", "all", "report"), default="stage1")
    parser.add_argument("--arms", nargs="+", default=list(PRIMARY_ARMS), help="Subset of baseline arms to run.")
    parser.add_argument("--include-secondary", action="store_true", help="Also include muon2_f and muoneq_rc.")
    parser.add_argument("--wait-for", type=Path, help="Optional completion.json path to wait for before starting.")
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    args = parser.parse_args()

    arms = tuple(dict.fromkeys([*args.arms, *(SECONDARY_ARMS if args.include_secondary else ())]))
    index = generate_all()

    if args.wave == "report":
        report = write_results_report(arms)
        print(f"Wrote {report}")
        return

    stage1_jobs = []
    for lr_tag in STAGE1_LRS:
        for arm_key in arms:
            name = run_name(arm_key, 0, lr_tag)
            stage1_jobs.append((name, Path(index[name]["config"]), CAMPAIGN_ROOT / "runs" / name))

    if args.dry_run:
        print(f"Campaign root: {CAMPAIGN_ROOT}")
        print(f"Arms ({len(arms)}): {', '.join(arms)}")
        print(f"Stage 1 jobs ({len(stage1_jobs)}):")
        for name, cfg, out in stage1_jobs:
            print(f"  - {name}  cfg={cfg.name}  out={out}")
        return

    if args.wait_for:
        wait_for_campaign(args.wait_for)

    if not args.skip_deploy:
        deploy()

    if args.wave in ("stage1", "all"):
        log(f"Starting Stage 1 ({len(stage1_jobs)} runs across {len(NODES)} nodes)")
        dispatch_wave(stage1_jobs, NODES)

        # Adaptive boundary guards if any arm hits 0p0113 or 0p0226 edge
        guard_jobs = []
        for arm_key in arms:
            best_tag, _ = best_seed0_lr(arm_key)
            if best_tag in BOUNDARY_LRS:
                g_tag = BOUNDARY_LRS[best_tag]
                name = run_name(arm_key, 0, g_tag)
                guard_jobs.append((name, Path(index[name]["config"]), CAMPAIGN_ROOT / "runs" / name))
        if guard_jobs:
            log(f"Starting Stage 1 boundary guards ({len(guard_jobs)} runs)")
            dispatch_wave(guard_jobs, NODES)

    if args.wave in ("stage2", "all"):
        stage2_jobs = []
        for arm_key in arms:
            best_tag, best_loss = best_seed0_lr(arm_key)
            if best_tag is None:
                log(f"[WARN] No completed Seed-0 run for {arm_key}; skipping Stage 2")
                continue
            log(f"Stage 2 for {arm_key}: using Seed-0 best LR {best_tag} (loss={best_loss:.5f})")
            for seed in (1, 2, 3):
                name = run_name(arm_key, seed, best_tag)
                stage2_jobs.append((name, Path(index[name]["config"]), CAMPAIGN_ROOT / "runs" / name))
        dispatch_wave(stage2_jobs, NODES)

    report = write_results_report(arms)
    log(f"Updated report: {report}")


if __name__ == "__main__":
    main()
