# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dynamic 4-node (32xA100) orchestrator for the 1x3 weight-decay operator ablation campaign."""

from __future__ import annotations

import argparse
import gzip
import json
import math
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

from campaign.configure_wd_campaign import ARM_SPECS, LR_BY_TAG, LR_GRID, generate_all, run_name

PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
CAMPAIGN_ROOT = PVC_ROOT / "campaigns" / "wd-ablation-20261127"
LR_BRACKET_ROOT = PVC_ROOT / "campaigns" / "lr-bracket-20261125"
CONFIRM_ROOT = PVC_ROOT / "campaigns" / "confirmation-20261121"
ARCHIVE_REC = Path("/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization/experiment_records")
DATA_DIR = PVC_ROOT / "data"
DATA_RECEIPT = DATA_DIR / "experiment_receipt.json"
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]

# Archived 130M 1x Chinchilla baselines & W*R^2 reference runs (identical model, data receipt, and base runner)
EXISTING_RUNS: dict[tuple[str, str, int], Path] = {
    # Baselines (seed 0)
    ("muon", "0p016", 0): PVC_ROOT / "runs" / "wave0" / "muon",
    ("muon", "0p0226", 0): CONFIRM_ROOT / "runs" / "wave1b" / "muon_130m_1x_seed0_lr0p0226",
    ("muont", "0p016", 0): CONFIRM_ROOT / "runs" / "wave1a" / "muont_130m_1x_seed0_lr0p016",
    ("muont", "0p0226", 0): PVC_ROOT / "runs" / "wave_sym_fastlag" / "muont",
    ("od_all", "0p016", 0): LR_BRACKET_ROOT / "runs" / "od_all_130m_1x_seed0_lr0p016",
    ("od_all", "0p0226", 0): CONFIRM_ROOT / "runs" / "wave1b" / "od_all_130m_1x_seed0_lr0p0226",
    # WT-MLP-Lag with Q = W R^2 (seed 0)
    ("wt_mlp_lag", "0p016", 0): CONFIRM_ROOT / "runs" / "wave1a" / "wt_mlp_lag_130m_1x_seed0_lr0p016",
    ("wt_mlp_lag", "0p0226", 0): PVC_ROOT / "runs" / "wave_sym_fastlag" / "inner_outer_mlpdiag_lag",
    # Fresh Seeds 1, 2, 3 at shared optimum 0.016
    ("muon", "0p016", 1): CONFIRM_ROOT / "runs" / "wave2a" / "muon_130m_1x_seed1",
    ("muon", "0p016", 2): CONFIRM_ROOT / "runs" / "wave2b" / "muon_130m_1x_seed2",
    ("muon", "0p016", 3): CONFIRM_ROOT / "runs" / "wave2c" / "muon_130m_1x_seed3",
    ("muont", "0p016", 1): CONFIRM_ROOT / "runs" / "wave2a" / "muont_130m_1x_seed1",
    ("muont", "0p016", 2): CONFIRM_ROOT / "runs" / "wave2b" / "muont_130m_1x_seed2",
    ("muont", "0p016", 3): CONFIRM_ROOT / "runs" / "wave2c" / "muont_130m_1x_seed3",
    ("od_all", "0p016", 1): LR_BRACKET_ROOT / "runs" / "od_all_130m_1x_seed1_lr0p016",
    ("od_all", "0p016", 2): LR_BRACKET_ROOT / "runs" / "od_all_130m_1x_seed2_lr0p016",
    ("od_all", "0p016", 3): LR_BRACKET_ROOT / "runs" / "od_all_130m_1x_seed3_lr0p016",
    ("wt_mlp_lag", "0p016", 1): LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed1_lr0p016",
    ("wt_mlp_lag", "0p016", 2): LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed2_lr0p016",
    ("wt_mlp_lag", "0p016", 3): LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed3_lr0p016",
    # Fresh Seeds 1, 2, 3 at 0.02263
    ("od_all", "0p0226", 1): CONFIRM_ROOT / "runs" / "wave2a" / "od_all_130m_1x_seed1",
    ("od_all", "0p0226", 2): CONFIRM_ROOT / "runs" / "wave2b" / "od_all_130m_1x_seed2",
    ("od_all", "0p0226", 3): CONFIRM_ROOT / "runs" / "wave2c" / "od_all_130m_1x_seed3",
    ("wt_mlp_lag", "0p0226", 1): CONFIRM_ROOT / "runs" / "wave2a" / "wt_mlp_lag_130m_1x_seed1",
    ("wt_mlp_lag", "0p0226", 2): CONFIRM_ROOT / "runs" / "wave2b" / "wt_mlp_lag_130m_1x_seed2",
    ("wt_mlp_lag", "0p0226", 3): CONFIRM_ROOT / "runs" / "wave2c" / "wt_mlp_lag_130m_1x_seed3",
}

EXISTING_PALOMA: dict[tuple[str, str, int], Path] = (
    {("muon", "0p016", s): ARCHIVE_REC / "paloma_eval" / f"muon_130m_1x_seed{s}.json" for s in (1, 2, 3)}
    | {("muont", "0p016", s): ARCHIVE_REC / "paloma_eval" / f"muont_130m_1x_seed{s}.json" for s in (1, 2, 3)}
    | {("od_all", "0p016", s): LR_BRACKET_ROOT / "paloma_eval" / f"od_all_130m_1x_seed{s}_lr0p016.json" for s in (0, 1, 2, 3)}
    | {("wt_mlp_lag", "0p016", s): LR_BRACKET_ROOT / "paloma_eval" / f"wt_mlp_lag_130m_1x_seed{s}_lr0p016.json" for s in (1, 2, 3)}
    | {("od_all", "0p0226", s): ARCHIVE_REC / "paloma_eval" / f"od_all_130m_1x_seed{s}.json" for s in (1, 2, 3)}
    | {("wt_mlp_lag", "0p0226", s): ARCHIVE_REC / "paloma_eval" / f"wt_mlp_lag_130m_1x_seed{s}.json" for s in (1, 2, 3)}
)

NEW_ARMS = ("wt_mlp_lag_wd_iso", "wt_mlp_lag_wd_lr2")
REPORT_ARMS = ("muon", "muont", "od_all", "wt_mlp_lag_wd_iso", "wt_mlp_lag", "wt_mlp_lag_wd_lr2")
ARM_LABELS = {
    "muon": "Muon (Q = W, unprecond.)",
    "muont": "MuonT (Q = W R^2, input only)",
    "od_all": "OD-All (Q = W R^2)",
    "wt_mlp_lag_wd_iso": "WT-MLP-Lag [Q = W]",
    "wt_mlp_lag": "WT-MLP-Lag [Q = W R^2] (current)",
    "wt_mlp_lag_wd_lr2": "WT-MLP-Lag [Q = L^2 W R^2]",
}

LOG_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    with LOG_LOCK:
        print(line, flush=True)
        CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
        with (CAMPAIGN_ROOT / "orchestrator.log").open("a") as fh:
            fh.write(line + "\n")


def run_remote(node: str, argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [*SSH, f"root@{node}", shlex.join(argv)]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"[{node}] command failed (rc={proc.returncode}): {' '.join(argv)}\n"
            f"STDOUT:\n{proc.stdout[-2500:]}\nSTDERR:\n{proc.stderr[-2500:]}"
        )
    return proc


def _iter_metric_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    plain = sorted(run_dir.glob("metrics_from_*.jsonl"))
    if plain:
        for mf in plain:
            raw = mf.read_bytes()
            complete = raw[: raw.rfind(b"\n") + 1]
            for line in complete.splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    for mf in sorted(run_dir.glob("metrics_from_*.jsonl.gz")):
        with gzip.open(mf, "rt") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def read_run_summary(run_dir: Path) -> dict | None:
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        return None
    meta = json.loads(run_json.read_text())
    rows = _iter_metric_rows(run_dir)
    if not rows:
        return None
    by_step = {int(r["step"]): r for r in rows}
    planned = int(meta.get("config", {}).get("training", {}).get("steps", 5120))
    if meta.get("status") != "complete" or max(by_step) != planned:
        return None
    ordered = [by_step[s] for s in sorted(by_step)]
    eval_rows = [r for r in ordered if "selection_loss" in r]
    if not eval_rows:
        return None
    final_eval = eval_rows[-1]
    late_evals = [r for r in eval_rows if int(r["step"]) >= 2751]
    step_times = [
        r["step_seconds"]
        for r in ordered[20:]
        if "step_seconds" in r and "eval_seconds" not in r and "checkpoint_seconds" not in r
    ]
    return {
        "name": run_dir.name,
        "status": "complete",
        "arm": meta.get("config", {}).get("experiment", {}).get("arm"),
        "arm_label": meta.get("config", {}).get("experiment", {}).get("arm_label"),
        "decay_mode": meta.get("config", {}).get("experiment", {}).get("decay_mode", "default"),
        "seed": int(meta.get("config", {}).get("training", {}).get("seed", 0)),
        "lr": float(meta.get("config", {}).get("optimizer", {}).get("lr")),
        "steps": max(by_step),
        "planned_steps": planned,
        "selection_loss": float(final_eval["selection_loss"]),
        "c4_loss": float(final_eval["c4_loss"]),
        "late11_sel": statistics.mean(float(r["selection_loss"]) for r in late_evals) if late_evals else None,
        "late11_c4": statistics.mean(float(r["c4_loss"]) for r in late_evals) if late_evals else None,
        "late11_evals": {str(int(r["step"])): float(r["selection_loss"]) for r in late_evals},
        "median_step_seconds": statistics.median(step_times) if step_times else None,
        "peak_allocated_gib": max((r.get("peak_allocated_bytes", 0) for r in ordered), default=0) / (1024**3),
        "elapsed_seconds": meta.get("elapsed_seconds"),
        "run_dir": str(run_dir),
    }


def run_owned_on_node(node: str, tag: str, inner_cmd: list[str]) -> dict:
    alloc_dir = CAMPAIGN_ROOT / "allocations"
    state_path = alloc_dir / f"{tag}_{node.replace('.', '_')}.state.json"
    log_path = alloc_dir / f"{tag}_{node.replace('.', '_')}.log"
    argv = [
        PYTHON,
        str(TASK / "campaign" / "allocation.py"),
        "--state",
        str(state_path),
        "--log",
        str(log_path),
        "--",
        *inner_cmd,
    ]
    log(f"START [{node}] {tag} (log: {log_path})")
    t0 = time.perf_counter()
    run_remote(node, argv, check=True)
    elapsed = time.perf_counter() - t0
    log(f"DONE  [{node}] {tag} in {elapsed / 60.0:.1f} min")
    return {"node": node, "tag": tag, "elapsed_seconds": elapsed, "log": str(log_path)}


def preflight_gates(configs_index: dict[str, dict]) -> None:
    smoke_root = CAMPAIGN_ROOT / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)

    # Gate 1: GPU decay-operator accuracy & default bitwise check
    math_marker = smoke_root / "decay_modes_gpu_verified.json"
    if math_marker.is_file():
        log(f"GPU decay-operator gate already passed ({math_marker}), skipping.")
    else:
        run_owned_on_node(
            NODES[0],
            "verify_decay_modes_gpu",
            [PYTHON, str(TASK / "campaign" / "verify_decay_modes_gpu.py"), "--out", str(math_marker)],
        )
        log(f"GPU decay-operator accuracy & default bitwise gate PASSED ({math_marker})")

    # Gate 2: 8-step bitwise parity check against archived wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016
    parity_dir = smoke_root / "parity_8step"
    marker = parity_dir / "passed.json"
    if marker.is_file():
        log(f"8-step parity gate already passed ({marker}), skipping.")
        return

    entry = configs_index["wt_mlp_lag_130m_1x_seed0_lr0p016"]
    out_dir = parity_dir / entry["name"]
    script_sh = parity_dir / "run_smoke.sh"
    parity_dir.mkdir(parents=True, exist_ok=True)
    script_sh.write_text(
        "set -euo pipefail\n"
        f"rm -rf {shlex.quote(str(out_dir))}\n"
        f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
        f"-m experiment.train --config {shlex.quote(entry['config'])} "
        f"--data {shlex.quote(str(DATA_DIR))} --data-receipt {shlex.quote(str(DATA_RECEIPT))} "
        f"--out {shlex.quote(str(out_dir))} --max-steps 8\n"
    )
    run_owned_on_node(NODES[0], "smoke_parity_8step", ["bash", str(script_sh)])

    new_rows = _iter_metric_rows(out_dir)
    ref_dir = EXISTING_RUNS[("wt_mlp_lag", "0p016", 0)]
    ref_rows = [r for r in _iter_metric_rows(ref_dir) if int(r["step"]) <= 8]
    new_by_step = {int(r["step"]): r for r in new_rows}
    ref_by_step = {int(r["step"]): r for r in ref_rows}
    diffs = {s: abs(float(new_by_step[s]["train_loss"]) - float(ref_by_step[s]["train_loss"])) for s in range(1, 9)}
    max_diff = max(diffs.values())
    sel_diff = abs(float(new_by_step[1]["selection_loss"]) - float(ref_by_step[1]["selection_loss"]))
    log(
        f"8-step parity vs archived wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016: "
        f"max |d train_loss| = {max_diff:.3e}, |d sel_loss_step1| = {sel_diff:.3e}"
    )
    if max_diff > 1e-5 or sel_diff > 1e-5:
        raise RuntimeError(f"Parity check failed: max_diff={max_diff}, sel_diff={sel_diff} ({diffs})")
    marker.write_text(
        json.dumps(
            {
                "time": now(),
                "max_abs_train_loss_diff": max_diff,
                "step1_selection_loss_diff": sel_diff,
                "exact_zero": max_diff == 0.0 and sel_diff == 0.0,
                "diffs": diffs,
            },
            indent=2,
        )
        + "\n"
    )


class CampaignScheduler:
    """Thread-safe scheduler for Stage 1 (seed 0 @ 0.016 & 0.0226) and Stage 2 (seeds 1-3)."""

    def __init__(self, configs_index: dict[str, dict]) -> None:
        self.configs_index = configs_index
        self.lock = threading.Lock()
        self.completed: dict[tuple[str, str, int], dict] = {}
        self.in_flight: dict[tuple[str, str, int], str] = {}
        self.failed: list[tuple[tuple[str, str, int], str]] = []

        for key, path in EXISTING_RUNS.items():
            s = read_run_summary(path)
            if s is None:
                raise RuntimeError(f"Missing expected archived reference run {key} at {path}")
            self.completed[key] = s

        runs_root = CAMPAIGN_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        for arm_key in NEW_ARMS:
            for lr_tag, _ in LR_GRID:
                for seed in (0, 1, 2, 3):
                    key = (arm_key, lr_tag, seed)
                    out_dir = runs_root / run_name(arm_key, seed, lr_tag)
                    s = read_run_summary(out_dir)
                    if s is not None:
                        self.completed[key] = s
                        log(f"Loaded already-completed campaign run {out_dir.name}: sel={s['selection_loss']:.5f}")

    def _desired_jobs(self) -> list[tuple[int, tuple[str, str, int]]]:
        candidates: list[tuple[int, tuple[str, str, int]]] = []

        # Stage 1 (Wave 1 across 4 nodes): Seed 0 at both 0.016 and 0.02263 for both new decay variants
        stage1 = [
            (1, ("wt_mlp_lag_wd_iso", "0p016", 0)),
            (2, ("wt_mlp_lag_wd_lr2", "0p016", 0)),
            (3, ("wt_mlp_lag_wd_iso", "0p0226", 0)),
            (4, ("wt_mlp_lag_wd_lr2", "0p0226", 0)),
        ]
        for prio, key in stage1:
            if key not in self.completed:
                candidates.append((prio, key))

        # Stage 2 (Waves 2-3): Fresh paired seeds 1, 2, 3 at shared optimum 0.0160
        for base_p, arm_key in ((10, "wt_mlp_lag_wd_iso"), (20, "wt_mlp_lag_wd_lr2")):
            for s in (1, 2, 3):
                k = (arm_key, "0p016", s)
                if k not in self.completed:
                    candidates.append((base_p + s, k))

            # Adaptive check: if seed-0 at 0.02263 strictly improves on 0.0160 by > 0.5 mnat, also replicate 0.02263
            k16 = (arm_key, "0p016", 0)
            k22 = (arm_key, "0p0226", 0)
            if k16 in self.completed and k22 in self.completed:
                if self.completed[k22]["selection_loss"] < self.completed[k16]["selection_loss"] - 0.0005:
                    for s in (1, 2, 3):
                        k_hot = (arm_key, "0p0226", s)
                        if k_hot not in self.completed:
                            candidates.append((30 + base_p + s, k_hot))

        return sorted(candidates, key=lambda item: item[0])

    def acquire_next(self, node: str) -> tuple[tuple[str, str, int] | None, bool]:
        with self.lock:
            if self.failed:
                return None, True
            desired = self._desired_jobs()
            if not desired and not self.in_flight:
                return None, True
            for _, key in desired:
                if key not in self.in_flight and key not in self.completed:
                    self.in_flight[key] = node
                    return key, False
            return None, False

    def mark_complete(self, key: tuple[str, str, int], summary: dict) -> None:
        with self.lock:
            self.in_flight.pop(key, None)
            self.completed[key] = summary

    def mark_failed(self, key: tuple[str, str, int], err: str) -> None:
        with self.lock:
            self.in_flight.pop(key, None)
            self.failed.append((key, err))

    def live_snapshot(self) -> dict:
        with self.lock:
            in_flight_copy = dict(self.in_flight)
            completed_keys = sorted(f"{a}@{t}:seed{s}" for (a, t, s) in self.completed if a in NEW_ARMS)
            pending_keys = [f"{a}@{t}:seed{s}" for _, (a, t, s) in self._desired_jobs() if (a, t, s) not in in_flight_copy]
            failed_copy = [(f"{a}@{t}:seed{s}", err) for (a, t, s), err in self.failed]

        active = []
        for (arm_key, lr_tag, seed), node in sorted(in_flight_copy.items()):
            name = run_name(arm_key, seed, lr_tag)
            out_dir = CAMPAIGN_ROOT / "runs" / name
            rows = _iter_metric_rows(out_dir) if out_dir.is_dir() else []
            rec: dict = {"name": name, "node": node, "lr": LR_BY_TAG[lr_tag], "seed": seed}
            if rows:
                last = rows[-1]
                times = [r["step_seconds"] for r in rows[20:] if "eval_seconds" not in r and "checkpoint_seconds" not in r]
                med = statistics.median(times[-100:]) if times else None
                rec.update(
                    step=int(last["step"]),
                    planned_steps=5120,
                    train_loss=float(last["train_loss"]),
                    finite=all(math.isfinite(float(r["train_loss"])) for r in rows),
                    median_step_seconds=med,
                    eta_minutes=round((5120 - int(last["step"])) * med / 60.0, 1) if med else None,
                )
            else:
                rec.update(step=0, planned_steps=5120, status="compiling_or_starting")
            active.append(rec)

        return {
            "observed": now(),
            "orchestrator_pid": os.getpid(),
            "active_runs": active,
            "completed_new_runs": completed_keys,
            "queued_runs": pending_keys,
            "failed_runs": failed_copy,
        }


def execute_training_job(node: str, key: tuple[str, str, int], configs_index: dict[str, dict]) -> dict:
    arm_key, lr_tag, seed = key
    name = run_name(arm_key, seed, lr_tag)
    entry = configs_index[name]
    out_dir = CAMPAIGN_ROOT / "runs" / name
    resume_args: list[str] = []
    latest = out_dir / "latest_checkpoint.json"
    if latest.is_file():
        ckpt = json.loads(latest.read_text())["path"]
        resume_args = ["--resume", ckpt]
    elif out_dir.exists():
        s = read_run_summary(out_dir)
        if s is not None:
            return s
        backup = out_dir.with_name(f"{out_dir.name}.incomplete.{int(time.time())}")
        out_dir.rename(backup)

    cmd = [
        PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "experiment.train",
        "--config",
        entry["config"],
        "--data",
        str(DATA_DIR),
        "--data-receipt",
        str(DATA_RECEIPT),
        "--out",
        str(out_dir),
        *resume_args,
    ]
    tag = f"train_{name}_{int(time.time())}"
    run_owned_on_node(node, tag, cmd)
    summary = read_run_summary(out_dir)
    if summary is None:
        raise RuntimeError(f"Training job {name} on {node} exited 0 without complete metrics!")
    log(
        f"SUMMARY {name:42s} sel={summary['selection_loss']:.5f}  c4={summary['c4_loss']:.5f}  "
        f"late11_sel={summary['late11_sel']:.5f}  step_s={summary['median_step_seconds']:.3f}"
    )
    return summary


def run_paloma_stage() -> dict[str, dict]:
    paloma_dir = CAMPAIGN_ROOT / "paloma_eval"
    paloma_dir.mkdir(parents=True, exist_ok=True)
    runs_root = CAMPAIGN_ROOT / "runs"
    todo: list[tuple[Path, Path]] = []

    # All newly completed runs in this campaign
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if read_run_summary(run_dir) is None:
            continue
        if not (run_dir / "latest_checkpoint.json").is_file():
            continue
        out_json = paloma_dir / f"{run_dir.name}.json"
        if not out_json.is_file():
            todo.append((run_dir, out_json))

    # Also evaluate Paloma on archived seed-0 references if checkpoints exist and Paloma JSON is missing
    extra_refs = [
        ("wt_mlp_lag_130m_1x_seed0_lr0p016", EXISTING_RUNS[("wt_mlp_lag", "0p016", 0)]),
        ("wt_mlp_lag_130m_1x_seed0_lr0p0226", EXISTING_RUNS[("wt_mlp_lag", "0p0226", 0)]),
        ("muont_130m_1x_seed0_lr0p016", EXISTING_RUNS[("muont", "0p016", 0)]),
        ("muont_130m_1x_seed0_lr0p0226", EXISTING_RUNS[("muont", "0p0226", 0)]),
    ]
    for alias, rdir in extra_refs:
        if (rdir / "latest_checkpoint.json").is_file():
            out_json = paloma_dir / f"{alias}.json"
            if not out_json.is_file():
                todo.append((rdir, out_json))

    if todo:
        log(f"Evaluating 16-domain Paloma on {len(todo)} checkpoints across {min(len(NODES), len(todo))} nodes...")
        per_node: dict[str, list[tuple[Path, Path]]] = {n: [] for n in NODES}
        for idx, item in enumerate(todo):
            per_node[NODES[idx % len(NODES)]].append(item)

        def eval_node_batch(node: str, batch: list[tuple[Path, Path]]) -> None:
            lines = ["set -euo pipefail"]
            for rdir, ojson in batch:
                lines.append(
                    f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
                    f"{shlex.quote(str(TASK / 'campaign' / 'eval_paloma.py'))} "
                    f"--run-dir {shlex.quote(str(rdir))} "
                    f"--data {shlex.quote(str(DATA_DIR))} "
                    f"--out {shlex.quote(str(ojson))}"
                )
            batch_sh = paloma_dir / f"run_paloma_{node.replace('.', '_')}.sh"
            batch_sh.write_text("\n".join(lines) + "\n")
            run_owned_on_node(node, f"paloma_{int(time.time())}", ["bash", str(batch_sh)])

        with ThreadPoolExecutor(max_workers=len(NODES)) as pool:
            futs = [pool.submit(eval_node_batch, n, b) for n, b in per_node.items() if b]
            for fut in as_completed(futs):
                fut.result()

    results: dict[str, dict] = {}
    for p in sorted(paloma_dir.glob("*.json")):
        results[p.stem] = json.loads(p.read_text())
    return results


def write_final_reports(sched: CampaignScheduler, paloma_new: dict[str, dict]) -> None:
    def get_paloma(arm_key: str, lr_tag: str, seed: int) -> dict | None:
        if (arm_key, lr_tag, seed) in EXISTING_PALOMA and EXISTING_PALOMA[(arm_key, lr_tag, seed)].is_file():
            return json.loads(EXISTING_PALOMA[(arm_key, lr_tag, seed)].read_text())
        name = run_name(arm_key, seed, lr_tag) if arm_key in ARM_SPECS else f"{arm_key}_130m_1x_seed{seed}_lr{lr_tag}"
        return paloma_new.get(name)

    seed0_table: dict[str, dict[str, dict]] = {a: {} for a in REPORT_ARMS}
    for arm_key in REPORT_ARMS:
        for t in ("0p016", "0p0226"):
            if (arm_key, t, 0) in sched.completed:
                rec = dict(sched.completed[(arm_key, t, 0)])
                pal = get_paloma(arm_key, t, 0)
                rec["paloma_macro_mean"] = pal["paloma_macro_mean"] if pal else None
                seed0_table[arm_key][t] = rec

    seeds123: dict[str, dict] = {}
    for arm_key in REPORT_ARMS:
        for t in ("0p016", "0p0226"):
            if all((arm_key, t, s) in sched.completed for s in (1, 2, 3)):
                s_runs = [sched.completed[(arm_key, t, s)] for s in (1, 2, 3)]
                pals = [get_paloma(arm_key, t, s) for s in (1, 2, 3)]
                entry: dict = {
                    "arm_key": arm_key,
                    "label": ARM_LABELS[arm_key],
                    "lr_tag": t,
                    "lr": LR_BY_TAG[t],
                    "sel_seeds": [r["selection_loss"] for r in s_runs],
                    "sel_mean": statistics.mean(r["selection_loss"] for r in s_runs),
                    "sel_std": statistics.stdev(r["selection_loss"] for r in s_runs),
                    "c4_seeds": [r["c4_loss"] for r in s_runs],
                    "c4_mean": statistics.mean(r["c4_loss"] for r in s_runs),
                    "c4_std": statistics.stdev(r["c4_loss"] for r in s_runs),
                    "late11_sel_seeds": [r["late11_sel"] for r in s_runs],
                    "late11_sel_mean": statistics.mean(r["late11_sel"] for r in s_runs),
                    "paloma_seeds": [p["paloma_macro_mean"] for p in pals] if all(pals) else None,
                    "paloma_mean": statistics.mean(p["paloma_macro_mean"] for p in pals) if all(pals) else None,
                    "paloma_std": statistics.stdev(p["paloma_macro_mean"] for p in pals) if all(pals) else None,
                }
                # Paired differences in millinats vs MuonT@0p016 and WT-MLP-Lag(W R^2)@same_lr
                if all(("muont", "0p016", s) in sched.completed for s in (1, 2, 3)):
                    mt_runs = [sched.completed[("muont", "0p016", s)] for s in (1, 2, 3)]
                    d_mt = [1000.0 * (a["selection_loss"] - b["selection_loss"]) for a, b in zip(s_runs, mt_runs, strict=True)]
                    entry["paired_delta_vs_muont_0p016_mnat"] = {
                        "seeds": d_mt,
                        "mean": statistics.mean(d_mt),
                        "std": statistics.stdev(d_mt),
                    }
                if all(("wt_mlp_lag", t, s) in sched.completed for s in (1, 2, 3)):
                    wt_runs = [sched.completed[("wt_mlp_lag", t, s)] for s in (1, 2, 3)]
                    d_wt = [1000.0 * (a["selection_loss"] - b["selection_loss"]) for a, b in zip(s_runs, wt_runs, strict=True)]
                    entry["paired_delta_vs_wt_wr2_mnat"] = {
                        "seeds": d_wt,
                        "mean": statistics.mean(d_wt),
                        "std": statistics.stdev(d_wt),
                    }
                seeds123[f"{arm_key}@{t}"] = entry

    payload = {
        "generated_at": now(),
        "seed0_two_lr_table": seed0_table,
        "seeds123_summaries": seeds123,
    }
    res_dir = TASK / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(res_dir / "wd_ablation_summary.json", payload)
    atomic_json(CAMPAIGN_ROOT / "wd_ablation_summary.json", payload)

    md = [
        "# Weight-Decay Operator Ablation Results (`130M`, `20 TPP = 2.68B tokens`, `5,120 steps`)",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## 1. Seed 0 Two-LR Comparison (`η = 0.01600` vs `η = 0.02263`)",
        "",
        "| Method | `η = 0.01600` Held-out (C4 / Late-11) | `η = 0.02263` Held-out (C4 / Late-11) | Δ vs `W R^2` @ `0.016` (mnat) |",
        "| --- | ---: | ---: | ---: |",
    ]
    ref_s0 = seed0_table.get("wt_mlp_lag", {}).get("0p016", {}).get("selection_loss")
    for arm_key in REPORT_ARMS:
        cells = []
        for t in ("0p016", "0p0226"):
            if t in seed0_table[arm_key]:
                s = seed0_table[arm_key][t]
                cells.append(f"{s['selection_loss']:.5f} ({s['c4_loss']:.5f} / {s['late11_sel']:.5f})")
            else:
                cells.append("*(pending)*")
        if "0p016" in seed0_table[arm_key] and ref_s0 is not None:
            delta_str = f"{1000.0 * (seed0_table[arm_key]['0p016']['selection_loss'] - ref_s0):+.2f}"
        else:
            delta_str = "—"
        md.append(f"| {ARM_LABELS[arm_key]} | {cells[0]} | {cells[1]} | {delta_str} |")

    md += [
        "",
        "## 2. Fresh Paired `Seeds 1, 2, 3` Confirmation (`130M`, `2.68B` tokens)",
        "",
        "| Method @ LR | Final Held-out (Mean ± SD) | Δ vs `MuonT` (mnat) | Δ vs `W R^2` (mnat) | Late-11 Mean | Final C4 (Mean ± SD) | Paloma 16-Dom (Mean ± SD) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t in ("0p016", "0p0226"):
        for arm_key in REPORT_ARMS:
            key = f"{arm_key}@{t}"
            if key not in seeds123:
                continue
            v = seeds123[key]
            d_mt = v.get("paired_delta_vs_muont_0p016_mnat")
            d_wt = v.get("paired_delta_vs_wt_wr2_mnat")
            d_mt_str = f"{d_mt['mean']:+.2f} ± {d_mt['std']:.2f}" if d_mt else "—"
            d_wt_str = f"{d_wt['mean']:+.2f} ± {d_wt['std']:.2f}" if d_wt else "—"
            pal_str = f"{v['paloma_mean']:.5f} ± {v['paloma_std']:.5f}" if v["paloma_mean"] is not None else "—"
            md.append(
                f"| {v['label']} @ `{v['lr']:.5f}` | {v['sel_mean']:.5f} ± {v['sel_std']:.5f} | "
                f"{d_mt_str} | {d_wt_str} | {v['late11_sel_mean']:.5f} | "
                f"{v['c4_mean']:.5f} ± {v['c4_std']:.5f} | {pal_str} |"
            )

    md_text = "\n".join(md) + "\n"
    (res_dir / "WD_ABLATION_RESULTS.md").write_text(md_text)
    (CAMPAIGN_ROOT / "WD_ABLATION_RESULTS.md").write_text(md_text)
    log(f"Wrote consolidated results to {res_dir / 'WD_ABLATION_RESULTS.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print initial job queue without launching GPU jobs.")
    args = parser.parse_args()

    configs_index = generate_all()
    sched = CampaignScheduler(configs_index)
    if args.dry_run:
        print("Initial desired jobs:", json.dumps([(p, k) for p, k in sched._desired_jobs()], indent=2))
        return

    log("=== Deploying wd-ablation-20261127 workspace to all 4 A100 nodes ===")
    subprocess.run([PYTHON, str(TASK / "campaign" / "deploy_wd_campaign.py")], check=True)

    log("=== Running GPU decay-operator & 8-step bitwise parity preflight gates ===")
    preflight_gates(configs_index)

    log("=== Starting 4-node (32xA100) dynamic worker pool ===")
    report_lock = threading.Lock()
    stop_monitor = threading.Event()

    def monitor_loop() -> None:
        while not stop_monitor.is_set():
            try:
                snap = sched.live_snapshot()
                atomic_json(CAMPAIGN_ROOT / "progress.json", snap)
                atomic_json(TASK / "results" / "progress.json", snap)
            except Exception as exc:
                log(f"Monitor snapshot warning: {exc!r}")
            stop_monitor.wait(60.0)

    mon_thread = threading.Thread(target=monitor_loop, daemon=True)
    mon_thread.start()

    def node_worker(node: str) -> None:
        while True:
            key, all_done = sched.acquire_next(node)
            if all_done:
                return
            if key is None:
                time.sleep(15.0)
                continue
            try:
                summary = execute_training_job(node, key, configs_index)
                sched.mark_complete(key, summary)
                with report_lock:
                    write_final_reports(sched, {})
            except BaseException as exc:
                log(f"ERROR on {node} for {key}: {exc!r}")
                sched.mark_failed(key, repr(exc))
                raise

    try:
        with ThreadPoolExecutor(max_workers=len(NODES)) as pool:
            futs = {pool.submit(node_worker, n): n for n in NODES}
            for fut in as_completed(futs):
                fut.result()
    finally:
        stop_monitor.set()
        mon_thread.join(timeout=5.0)
        snap = sched.live_snapshot()
        atomic_json(CAMPAIGN_ROOT / "progress.json", snap)
        atomic_json(TASK / "results" / "progress.json", snap)

    if sched.failed:
        raise RuntimeError(f"Campaign encountered failures: {sched.failed}")

    log("=== All Stage 1 & Stage 2 training jobs COMPLETE; starting Stage 3 16-domain Paloma evaluation ===")
    paloma_new = run_paloma_stage()
    write_final_reports(sched, paloma_new)
    atomic_json(
        CAMPAIGN_ROOT / "completion.json",
        {"time": now(), "status": "complete", "summary": str(CAMPAIGN_ROOT / "wd_ablation_summary.json")},
    )
    log("=== wd-ablation-20261127 CAMPAIGN ALL COMPLETE ===")


if __name__ == "__main__":
    main()
