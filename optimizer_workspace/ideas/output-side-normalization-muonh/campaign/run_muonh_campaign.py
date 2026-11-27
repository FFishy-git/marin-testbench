# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Queue-aware dynamic 4-node (32xA100) orchestrator for MuonH, MuonTH, and WT-MLP-Lag + MuonH (Toner-matched & Marin-exact)."""

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

from campaign.configure_muonh_campaign import (
    ARM_SPECS,
    LR_BY_TAG,
    LR_GRID,
    MARIN_STAGE1_LRS,
    generate_all,
    run_name,
)
from experiment.records import compare_continuation

PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
CAMPAIGN_ROOT = PVC_ROOT / "campaigns" / "muonh-20261127"
LR_BRACKET_ROOT = PVC_ROOT / "campaigns" / "lr-bracket-20261125"
CONFIRM_ROOT = PVC_ROOT / "campaigns" / "confirmation-20261121"
ARCHIVE_REC = Path("/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization/experiment_records")
DATA_DIR = PVC_ROOT / "data"
DATA_RECEIPT = DATA_DIR / "experiment_receipt.json"
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]
ORCH_GEN = 3
WSD_STAGE1_LRS = ("0p0113", "0p016", "0p0226", "0p032")

EXISTING_RUNS: dict[tuple[str, str, int], Path] = {
    ("muon", "0p0113", 0): LR_BRACKET_ROOT / "runs" / "muon_130m_1x_seed0_lr0p0113",
    ("muon", "0p016", 0): PVC_ROOT / "runs" / "wave0" / "muon",
    ("muon", "0p0226", 0): CONFIRM_ROOT / "runs" / "wave1b" / "muon_130m_1x_seed0_lr0p0226",
    ("muon", "0p032", 0): LR_BRACKET_ROOT / "runs" / "muon_130m_1x_seed0_lr0p032",
    ("muont", "0p0113", 0): LR_BRACKET_ROOT / "runs" / "muont_130m_1x_seed0_lr0p0113",
    ("muont", "0p016", 0): CONFIRM_ROOT / "runs" / "wave1a" / "muont_130m_1x_seed0_lr0p016",
    ("muont", "0p0226", 0): PVC_ROOT / "runs" / "wave_sym_fastlag" / "muont",
    ("muont", "0p032", 0): LR_BRACKET_ROOT / "runs" / "muont_130m_1x_seed0_lr0p032",
    ("wt_mlp_lag", "0p0113", 0): LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed0_lr0p0113",
    ("wt_mlp_lag", "0p016", 0): CONFIRM_ROOT / "runs" / "wave1a" / "wt_mlp_lag_130m_1x_seed0_lr0p016",
    ("wt_mlp_lag", "0p0226", 0): PVC_ROOT / "runs" / "wave_sym_fastlag" / "inner_outer_mlpdiag_lag",
    ("wt_mlp_lag", "0p032", 0): LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed0_lr0p032",
    # Fresh Seeds 1, 2, 3 at 0.016
    ("muon", "0p016", 1): CONFIRM_ROOT / "runs" / "wave2a" / "muon_130m_1x_seed1",
    ("muon", "0p016", 2): CONFIRM_ROOT / "runs" / "wave2b" / "muon_130m_1x_seed2",
    ("muon", "0p016", 3): CONFIRM_ROOT / "runs" / "wave2c" / "muon_130m_1x_seed3",
    ("muont", "0p016", 1): CONFIRM_ROOT / "runs" / "wave2a" / "muont_130m_1x_seed1",
    ("muont", "0p016", 2): CONFIRM_ROOT / "runs" / "wave2b" / "muont_130m_1x_seed2",
    ("muont", "0p016", 3): CONFIRM_ROOT / "runs" / "wave2c" / "muont_130m_1x_seed3",
    ("wt_mlp_lag", "0p016", 1): LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed1_lr0p016",
    ("wt_mlp_lag", "0p016", 2): LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed2_lr0p016",
    ("wt_mlp_lag", "0p016", 3): LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed3_lr0p016",
}

EXISTING_PALOMA: dict[tuple[str, str, int], Path] = (
    {("muon", "0p016", s): ARCHIVE_REC / "paloma_eval" / f"muon_130m_1x_seed{s}.json" for s in (1, 2, 3)}
    | {("muont", "0p016", s): ARCHIVE_REC / "paloma_eval" / f"muont_130m_1x_seed{s}.json" for s in (1, 2, 3)}
    | {("wt_mlp_lag", "0p016", s): LR_BRACKET_ROOT / "paloma_eval" / f"wt_mlp_lag_130m_1x_seed{s}_lr0p016.json" for s in (1, 2, 3)}
)

PRIMARY_ARMS = ("muonh", "muonth", "wt_mlp_lag_muonh")
ADAMH_ARMS = ("muonh_adamh", "muonth_adamh", "wt_mlp_lag_muonh_adamh")
MARIN_ARMS = ("muonh_marin", "muonth_marin", "wt_mlp_lag_muonh_marin")
NEW_ARMS = (*PRIMARY_ARMS, *ADAMH_ARMS, *MARIN_ARMS)
REPORT_ARMS = ("muon", "muont", "wt_mlp_lag", *PRIMARY_ARMS, *ADAMH_ARMS, *MARIN_ARMS)
ARM_LABELS = {
    "muon": "Muon (WD=0.1, Q=W)",
    "muont": "MuonT (WD=0.1, Q=W R^2)",
    "wt_mlp_lag": "WT-MLP-Lag (WD=0.1, Q=W R^2)",
    "muonh": "MuonH (WSD, head=AdamW)",
    "muonth": "MuonTH [MuonT + MuonH] (WSD, head=AdamW)",
    "wt_mlp_lag_muonh": "WT-MLP-Lag + MuonH (WSD, head=AdamW)",
    "muonh_adamh": "MuonH + AdamH (WSD, full Hyperball)",
    "muonth_adamh": "MuonTH + AdamH (WSD, full Hyperball)",
    "wt_mlp_lag_muonh_adamh": "WT-MLP-Lag + MuonH + AdamH (WSD, full Hyperball)",
    "muonh_marin": "MuonH (Marin exact: 1pct-noclip + AdamH + quintic5)",
    "muonth_marin": "MuonTH (Marin exact: 1pct-noclip + AdamH + quintic5)",
    "wt_mlp_lag_muonh_marin": "WT-MLP-Lag + MuonH (Marin exact: 1pct-noclip + AdamH + quintic5)",
}

LOG_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def log(msg: str) -> None:
    line = f"[{now()}] [gen{ORCH_GEN}] {msg}"
    with LOG_LOCK:
        print(line, flush=True)
        CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
        with (CAMPAIGN_ROOT / "orchestrator.log").open("a") as fh:
            fh.write(line + "\n")


def pid_alive(pid: int | None) -> bool:
    return pid is not None and pid > 0 and Path(f"/proc/{pid}").exists()


def run_remote(node: str, argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [*SSH, f"root@{node}", shlex.join(argv)]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"[{node}] command failed (rc={proc.returncode}): {' '.join(argv)}\n"
            f"STDOUT:\n{proc.stdout[-2500:]}\nSTDERR:\n{proc.stderr[-2500:]}"
        )
    return proc


def node_idle_of_training(node: str) -> tuple[bool, str]:
    """Check whether a single node has zero active training / torchrun processes."""
    snippet = (
        "import pathlib\n"
        "busy = []\n"
        "for p in pathlib.Path('/proc').iterdir():\n"
        "    if not p.name.isdigit(): continue\n"
        "    try:\n"
        "        cmd = ' '.join(x.decode() for x in (p/'cmdline').read_bytes().split(b'\\0') if x)\n"
        "    except Exception:\n"
        "        continue\n"
        "    if ('experiment.train' in cmd or 'torch.distributed.run' in cmd) and 'allocation.py' not in cmd and 'run_wd_campaign.py' not in cmd and 'run_muonh_campaign.py' not in cmd:\n"
        "        busy.append(f'{p.name}:{cmd[:80]}')\n"
        "print('; '.join(busy))\n"
    )
    proc = run_remote(node, [PYTHON, "-c", snippet], check=False)
    out = proc.stdout.strip()
    if proc.returncode != 0 or out:
        return False, out or f"ssh_rc={proc.returncode}"
    return True, ""


def cluster_idle_of_training() -> tuple[bool, dict[str, str]]:
    details: dict[str, str] = {}
    for node in NODES:
        idle, desc = node_idle_of_training(node)
        if not idle:
            details[node] = desc
    return (len(details) == 0), details


def scan_running_allocations(name_to_key: dict[str, tuple[str, str, int]]) -> dict[tuple[str, str, int], str]:
    """Inspect allocations/*.state.json to identify any active v1 or v2 training jobs by (arm_key, lr_tag, seed) -> node."""
    alloc_dir = CAMPAIGN_ROOT / "allocations"
    active: dict[tuple[str, str, int], str] = {}
    if not alloc_dir.is_dir():
        return active
    for st_file in sorted(alloc_dir.glob("train_*.state.json")):
        try:
            st = json.loads(st_file.read_text())
        except Exception:
            continue
        if st.get("status") != "running":
            continue
        cmd = st.get("command") or []
        cfg_stem = None
        for idx, tok in enumerate(cmd):
            if tok == "--config" and idx + 1 < len(cmd):
                cfg_stem = Path(cmd[idx + 1]).stem
                break
        if cfg_stem and cfg_stem in name_to_key:
            node = next((n for n in NODES if st_file.name.endswith(f"_{n.replace('.', '_')}.state.json")), "unknown")
            active[name_to_key[cfg_stem]] = node
    return active


def wait_for_node_drain(node: str) -> None:
    """Wait until any pre-existing v1 job on `node` completes cleanly and `node` settles."""
    slug = node.replace(".", "_")
    alloc_dir = CAMPAIGN_ROOT / "allocations"
    logged_wait = False
    while True:
        running_states = []
        if alloc_dir.is_dir():
            for sf in alloc_dir.glob(f"train_*_{slug}.state.json"):
                try:
                    st = json.loads(sf.read_text())
                    if st.get("status") == "running":
                        running_states.append(sf.name)
                except Exception:
                    pass
        idle, busy_desc = node_idle_of_training(node)
        if idle and not running_states:
            if logged_wait:
                log(f"Node {node} finished draining v1 job; settling 8s before gen2 takeover...")
                time.sleep(8.0)
                idle2, _ = node_idle_of_training(node)
                if not idle2:
                    continue
            return
        if not logged_wait:
            log(f"Node {node} currently finishing v1 job ({running_states or busy_desc}); waiting non-intrusively...")
            logged_wait = True
        time.sleep(15.0)


def _iter_metric_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    plain = sorted(run_dir.glob("metrics_from_*.jsonl"))
    if plain:
        for mf in plain:
            raw = mf.read_bytes().replace(b"\x00", b"")
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
    try:
        meta = json.loads(run_json.read_text())
    except Exception:
        return None
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
        "head_mode": meta.get("config", {}).get("experiment", {}).get("head_mode", "adamw"),
        "recipe": meta.get("config", {}).get("experiment", {}).get("recipe", "toner_wsd"),
        "seed": int(meta.get("config", {}).get("training", {}).get("seed", 0)),
        "lr": float(meta.get("config", {}).get("optimizer", {}).get("lr")),
        "steps": max(by_step),
        "planned_steps": planned,
        "selection_loss": float(final_eval["selection_loss"]),
        "c4_loss": float(final_eval["c4_loss"]),
        "late11_sel": statistics.mean(float(r["selection_loss"]) for r in late_evals) if late_evals else None,
        "late11_c4": statistics.mean(float(r["c4_loss"]) for r in late_evals) if late_evals else None,
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
        "--orch-gen",
        str(ORCH_GEN),
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

    math_marker = smoke_root / "hyperball_gpu_verified.json"
    if math_marker.is_file():
        log(f"GPU Hyperball math gate already passed ({math_marker}), skipping.")
    else:
        run_owned_on_node(
            NODES[0],
            "verify_hyperball_gpu",
            [PYTHON, str(TASK / "campaign" / "verify_hyperball_gpu.py"), "--out", str(math_marker)],
        )
        log(f"GPU Hyperball accuracy & 50-step Frobenius conservation gate PASSED ({math_marker})")

    parity_dir = smoke_root / "parity_8step"
    marker = parity_dir / "passed.json"
    if marker.is_file():
        log(f"8-step parity gate already passed ({marker}), skipping.")
    else:
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
        if max_diff > 1e-5 or sel_diff > 1e-5:
            raise RuntimeError(f"Parity check failed: max_diff={max_diff}, sel_diff={sel_diff} ({diffs})")
        atomic_json(marker, {"time": now(), "max_abs_train_loss_diff": max_diff, "step1_selection_loss_diff": sel_diff})

    hb_dir = smoke_root / "hyperball_resume_8step"
    hb_marker = hb_dir / "passed.json"
    if hb_marker.is_file():
        log(f"8-step Hyperball + AdamH resume gate already passed ({hb_marker}), skipping.")
        return

    hb_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = json.loads(Path(configs_index["wt_mlp_lag_muonh_marin_130m_1x_seed0_lr0p016"]["config"]).read_text())
    base_cfg["experiment"]["checkpoint_every"] = 4
    smoke_cfg_path = hb_dir / "smoke_cfg.json"
    smoke_cfg_path.write_text(json.dumps(base_cfg, indent=2) + "\n")
    unint_out = hb_dir / "uninterrupted_8"
    res_out = hb_dir / "resumed_4_plus_4"
    hb_sh = hb_dir / "run_hb_smoke.sh"
    hb_sh.write_text(
        "set -euo pipefail\n"
        f"rm -rf {shlex.quote(str(unint_out))} {shlex.quote(str(res_out))}\n"
        f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
        f"-m experiment.train --config {shlex.quote(str(smoke_cfg_path))} "
        f"--data {shlex.quote(str(DATA_DIR))} --data-receipt {shlex.quote(str(DATA_RECEIPT))} "
        f"--out {shlex.quote(str(unint_out))} --max-steps 8\n"
        f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
        f"-m experiment.train --config {shlex.quote(str(smoke_cfg_path))} "
        f"--data {shlex.quote(str(DATA_DIR))} --data-receipt {shlex.quote(str(DATA_RECEIPT))} "
        f"--out {shlex.quote(str(res_out))} --max-steps 4\n"
        f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
        f"-m experiment.train --config {shlex.quote(str(smoke_cfg_path))} "
        f"--data {shlex.quote(str(DATA_DIR))} --data-receipt {shlex.quote(str(DATA_RECEIPT))} "
        f"--out {shlex.quote(str(res_out))} --resume {shlex.quote(str(res_out / 'checkpoints' / 'step_000004'))} --max-steps 8\n"
    )
    run_owned_on_node(NODES[0], "smoke_hyperball_resume_8step", ["bash", str(hb_sh)])
    comp = compare_continuation(
        unint_out / "checkpoints" / "step_000008",
        res_out / "checkpoints" / "step_000008",
        world=8,
    )
    atomic_json(hb_marker, {"time": now(), "continuation": comp})


class CampaignScheduler:
    """Thread-safe scheduler finishing Toner-matched 0.016 triad + full Marin-exact MuonH bracket & multi-seed."""

    def __init__(self, configs_index: dict[str, dict]) -> None:
        self.configs_index = configs_index
        self.name_to_key = {
            name: (info["arm_key"], info["lr_tag"], int(info["seed"])) for name, info in configs_index.items()
        }
        self.lock = threading.Lock()
        self.completed: dict[tuple[str, str, int], dict] = {}
        self.in_flight: dict[tuple[str, str, int], str] = {}
        self.failed: list[tuple[tuple[str, str, int], str]] = []

        for key, path in EXISTING_RUNS.items():
            if path.exists():
                s = read_run_summary(path)
                if s is not None:
                    self.completed[key] = s

        self.refresh_from_disk()

    def refresh_from_disk(self) -> int:
        runs_root = CAMPAIGN_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        newly_loaded = 0
        with self.lock:
            for arm_key in NEW_ARMS:
                for lr_tag, _ in LR_GRID:
                    for seed in (0, 1, 2, 3):
                        key = (arm_key, lr_tag, seed)
                        if key in self.completed:
                            continue
                        out_dir = runs_root / run_name(arm_key, seed, lr_tag)
                        s = read_run_summary(out_dir)
                        if s is not None:
                            self.completed[key] = s
                            newly_loaded += 1
                            log(f"Loaded completed campaign run {out_dir.name}: sel={s['selection_loss']:.5f}")
        return newly_loaded

    def _wsd_stage1_optimum_unlocked(self, arm_key: str) -> str | None:
        if not all((arm_key, t, 0) in self.completed for t in WSD_STAGE1_LRS):
            return None
        core_best = min(WSD_STAGE1_LRS, key=lambda t: self.completed[(arm_key, t, 0)]["selection_loss"])
        if core_best == "0p0113" and (arm_key, "0p008", 0) not in self.completed:
            return None
        if core_best == "0p032" and (arm_key, "0p0453", 0) not in self.completed:
            return None
        tested = [t for t, _ in LR_GRID if (arm_key, t, 0) in self.completed]
        return min(tested, key=lambda t: self.completed[(arm_key, t, 0)]["selection_loss"])

    def _marin_stage1_optimum_unlocked(self, arm_key: str) -> str | None:
        if not all((arm_key, t, 0) in self.completed for t in MARIN_STAGE1_LRS):
            return None
        core_best = min(MARIN_STAGE1_LRS, key=lambda t: self.completed[(arm_key, t, 0)]["selection_loss"])
        if core_best == "0p0113" and (arm_key, "0p008", 0) not in self.completed:
            return None
        if core_best == "0p0226" and (arm_key, "0p032", 0) not in self.completed:
            return None
        tested = [t for t, _ in LR_GRID if (arm_key, t, 0) in self.completed]
        return min(tested, key=lambda t: self.completed[(arm_key, t, 0)]["selection_loss"])

    def _desired_jobs_unlocked(self) -> list[tuple[int, tuple[str, str, int]]]:
        candidates: list[tuple[int, tuple[str, str, int]]] = []

        # Option A Stage 1a: Complete the Toner-matched WSD -> 0.0 Seed-0 bracket (0.01131 missing cells)
        prio = 1
        for lr_tag in WSD_STAGE1_LRS:  # ("0p0113", "0p016", "0p0226", "0p032")
            for arm_key in PRIMARY_ARMS:
                k = (arm_key, lr_tag, 0)
                if k not in self.completed:
                    candidates.append((prio, k))
                prio += 1

        # Proactive lower-boundary guard for MuonH on the 4th node during the 3-node 0.01131 wave
        if ("muonh", "0p008", 0) not in self.completed:
            candidates.append((15, ("muonh", "0p008", 0)))

        # Option A Stage 1b: Adaptive Boundary Guards for WSD Hyperball arms
        for idx, arm_key in enumerate(PRIMARY_ARMS):
            if all((arm_key, t, 0) in self.completed for t in WSD_STAGE1_LRS):
                core_best = min(WSD_STAGE1_LRS, key=lambda t: self.completed[(arm_key, t, 0)]["selection_loss"])
                if core_best == "0p0113" and (arm_key, "0p008", 0) not in self.completed:
                    candidates.append((20 + idx, (arm_key, "0p008", 0)))
                elif core_best == "0p032" and (arm_key, "0p0453", 0) not in self.completed:
                    candidates.append((23 + idx, (arm_key, "0p0453", 0)))

        # Option A Stage 2: Fresh paired Seeds 1, 2, 3 at each WSD Hyperball arm's verified interior optimum
        for idx, arm_key in enumerate(PRIMARY_ARMS):
            opt_t = self._wsd_stage1_optimum_unlocked(arm_key)
            if opt_t is not None:
                for s in (1, 2, 3):
                    k = (arm_key, opt_t, s)
                    if k not in self.completed:
                        candidates.append((30 + idx * 5 + s, k))

        # Matched-LR multi-seed safeguard if WT-MLP-Lag+MuonH (WSD) and MuonTH (WSD) pick different Seed-0 LRs
        opt_mt = self._wsd_stage1_optimum_unlocked("muonth")
        opt_wt = self._wsd_stage1_optimum_unlocked("wt_mlp_lag_muonh")
        if opt_mt is not None and opt_wt is not None and opt_wt != opt_mt:
            for s in (1, 2, 3):
                k = ("wt_mlp_lag_muonh", opt_mt, s)
                if k not in self.completed:
                    candidates.append((50 + s, k))

        return sorted(candidates, key=lambda item: item[0])

    def acquire_next(self, node: str) -> tuple[tuple[str, str, int] | None, bool]:
        self.refresh_from_disk()
        external_running = scan_running_allocations(self.name_to_key)
        with self.lock:
            if self.failed:
                return None, True
            v1_only = {k: n for k, n in external_running.items() if k not in self.in_flight}
            desired = self._desired_jobs_unlocked()
            if not desired and not self.in_flight and not v1_only:
                return None, True
            for _, key in desired:
                if key not in self.in_flight and key not in v1_only and key not in self.completed:
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

    def live_snapshot(self, phase: str = "running", v1_pid: int | None = None) -> dict:
        self.refresh_from_disk()
        external_running = scan_running_allocations(self.name_to_key)
        with self.lock:
            in_flight_copy = dict(self.in_flight)
            v1_only = {k: n for k, n in external_running.items() if k not in in_flight_copy}
            all_active_map = {**v1_only, **in_flight_copy}
            completed_keys = sorted(f"{a}@{t}:seed{s}" for (a, t, s) in self.completed if a in NEW_ARMS)
            pending_keys = [
                f"{a}@{t}:seed{s}"
                for _, (a, t, s) in self._desired_jobs_unlocked()
                if (a, t, s) not in all_active_map and (a, t, s) not in self.completed
            ]
            failed_copy = [(f"{a}@{t}:seed{s}", err) for (a, t, s), err in self.failed]
            wsd_optima = {a: self._wsd_stage1_optimum_unlocked(a) for a in PRIMARY_ARMS}
            marin_optima = {a: self._marin_stage1_optimum_unlocked(a) for a in MARIN_ARMS}

        active = []
        for (arm_key, lr_tag, seed), node in sorted(all_active_map.items()):
            name = run_name(arm_key, seed, lr_tag)
            out_dir = CAMPAIGN_ROOT / "runs" / name
            rows = _iter_metric_rows(out_dir) if out_dir.is_dir() else []
            gen_label = "prior_gen_draining" if (arm_key, lr_tag, seed) in v1_only else f"gen{ORCH_GEN}"
            rec: dict = {"name": name, "node": node, "owner": gen_label, "lr": LR_BY_TAG[lr_tag], "seed": seed}
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
            "phase": phase,
            "orchestrator_gen": ORCH_GEN,
            "orchestrator_pid": os.getpid(),
            "draining_prior_pid": v1_pid,
            "draining_prior_alive": pid_alive(v1_pid),
            "wsd_stage1_optima": wsd_optima,
            "marin_stage1_optima": marin_optima,
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
        f"SUMMARY {name:48s} sel={summary['selection_loss']:.5f}  c4={summary['c4_loss']:.5f}  "
        f"late11_sel={summary['late11_sel']:.5f}  step_s={summary['median_step_seconds']:.3f}"
    )
    return summary


def run_paloma_stage() -> dict[str, dict]:
    paloma_dir = CAMPAIGN_ROOT / "paloma_eval"
    paloma_dir.mkdir(parents=True, exist_ok=True)
    runs_root = CAMPAIGN_ROOT / "runs"
    todo: list[tuple[Path, Path]] = []

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if read_run_summary(run_dir) is None:
            continue
        if not (run_dir / "latest_checkpoint.json").is_file():
            continue
        out_json = paloma_dir / f"{run_dir.name}.json"
        if not out_json.is_file():
            todo.append((run_dir, out_json))

    if todo:
        log(f"Evaluating 16-domain Paloma + Frobenius drift on {len(todo)} runs across {min(len(NODES), len(todo))} nodes...")
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
    sched.refresh_from_disk()

    def get_paloma(arm_key: str, lr_tag: str, seed: int) -> dict | None:
        if (arm_key, lr_tag, seed) in EXISTING_PALOMA and EXISTING_PALOMA[(arm_key, lr_tag, seed)].is_file():
            return json.loads(EXISTING_PALOMA[(arm_key, lr_tag, seed)].read_text())
        name = run_name(arm_key, seed, lr_tag) if arm_key in ARM_SPECS else f"{arm_key}_130m_1x_seed{seed}_lr{lr_tag}"
        return paloma_new.get(name)

    with sched.lock:
        completed_copy = dict(sched.completed)
        opt_muonh = sched._wsd_stage1_optimum_unlocked("muonh")
        opt_muonth = sched._wsd_stage1_optimum_unlocked("muonth")
        wsd_optima = {a: sched._wsd_stage1_optimum_unlocked(a) for a in PRIMARY_ARMS}
        marin_optima = {a: sched._marin_stage1_optimum_unlocked(a) for a in MARIN_ARMS}

    lr_tags_all = [t for t, _ in LR_GRID if any((a, t, 0) in completed_copy for a in REPORT_ARMS) or t in ("0p0113", "0p016", "0p0226", "0p032")]
    seed0_table: dict[str, dict[str, dict]] = {a: {} for a in REPORT_ARMS}
    for arm_key in REPORT_ARMS:
        for t, _ in LR_GRID:
            if (arm_key, t, 0) in completed_copy:
                rec = dict(completed_copy[(arm_key, t, 0)])
                pal = get_paloma(arm_key, t, 0)
                rec["paloma_macro_mean"] = pal["paloma_macro_mean"] if pal else None
                rec["norm_conservation"] = pal.get("norm_conservation") if pal else None
                seed0_table[arm_key][t] = rec

    seeds123: dict[str, dict] = {}
    for arm_key in REPORT_ARMS:
        for t, _ in LR_GRID:
            if all((arm_key, t, s) in completed_copy for s in (1, 2, 3)):
                s_runs = [completed_copy[(arm_key, t, s)] for s in (1, 2, 3)]
                pals = [get_paloma(arm_key, t, s) for s in (1, 2, 3)]
                max_norm_drift = (
                    max(p.get("norm_conservation", {}).get("max_block_rel_norm_drift", 0.0) for p in pals)
                    if all(pals)
                    else None
                )
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
                    "late11_sel_mean": statistics.mean(r["late11_sel"] for r in s_runs),
                    "paloma_mean": statistics.mean(p["paloma_macro_mean"] for p in pals) if all(pals) else None,
                    "paloma_std": statistics.stdev(p["paloma_macro_mean"] for p in pals) if all(pals) else None,
                    "max_block_rel_norm_drift": max_norm_drift,
                }
                if all(("muont", "0p016", s) in completed_copy for s in (1, 2, 3)):
                    mt_runs = [completed_copy[("muont", "0p016", s)] for s in (1, 2, 3)]
                    d_mt = [1000.0 * (a["selection_loss"] - b["selection_loss"]) for a, b in zip(s_runs, mt_runs, strict=True)]
                    entry["paired_delta_vs_muont_0p016_mnat"] = {"mean": statistics.mean(d_mt), "std": statistics.stdev(d_mt), "seeds": d_mt}
                if opt_muonh and all(("muonh", opt_muonh, s) in completed_copy for s in (1, 2, 3)):
                    mh_runs = [completed_copy[("muonh", opt_muonh, s)] for s in (1, 2, 3)]
                    d_mh = [1000.0 * (a["selection_loss"] - b["selection_loss"]) for a, b in zip(s_runs, mh_runs, strict=True)]
                    entry["paired_delta_vs_muonh_opt_mnat"] = {"mean": statistics.mean(d_mh), "std": statistics.stdev(d_mh), "seeds": d_mh}
                if opt_muonth and all(("muonth", opt_muonth, s) in completed_copy for s in (1, 2, 3)):
                    mth_runs = [completed_copy[("muonth", opt_muonth, s)] for s in (1, 2, 3)]
                    d_mth = [1000.0 * (a["selection_loss"] - b["selection_loss"]) for a, b in zip(s_runs, mth_runs, strict=True)]
                    entry["paired_delta_vs_muonth_opt_mnat"] = {"mean": statistics.mean(d_mth), "std": statistics.stdev(d_mth), "seeds": d_mth}
                seeds123[f"{arm_key}@{t}"] = entry

    payload = {
        "generated_at": now(),
        "wsd_stage1_optima": wsd_optima,
        "marin_stage1_optima": marin_optima,
        "seed0_lr_bracket": seed0_table,
        "seeds123_summaries": seeds123,
    }
    res_dir = TASK / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(res_dir / "muonh_summary.json", payload)
    atomic_json(CAMPAIGN_ROOT / "muonh_summary.json", payload)

    header_cols = " | ".join(f"`{LR_BY_TAG[t]:.5f}`" for t in lr_tags_all)
    align_cols = " | ".join("---:" for _ in lr_tags_all)
    md = [
        "# Hyperball (`MuonH`, `MuonTH`, `WT-MLP-Lag + MuonH`) Results (`130M`, `20 TPP = 2.68B tokens`, `5,120 steps`)",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## 1. Seed 0 Full-Budget Learning-Rate Bracket (Held-out Loss)",
        "",
        f"| Method | {header_cols} | Best Seed-0 (LR) |",
        f"| --- | {align_cols} | ---: |",
    ]
    for arm_key in REPORT_ARMS:
        if not seed0_table[arm_key]:
            continue
        cells = []
        best_pair: tuple[float, str] | None = None
        for t in lr_tags_all:
            if t in seed0_table[arm_key]:
                val = seed0_table[arm_key][t]["selection_loss"]
                cells.append(f"{val:.5f}")
                if best_pair is None or val < best_pair[0]:
                    best_pair = (val, t)
            else:
                cells.append("—")
        best_str = f"**{best_pair[0]:.5f}** (`{LR_BY_TAG[best_pair[1]]:.5f}`)" if best_pair else "—"
        md.append(f"| {ARM_LABELS[arm_key]} | " + " | ".join(cells) + f" | {best_str} |")

    md += [
        "",
        "## 2. Fresh Paired `Seeds 1, 2, 3` Confirmation (`130M`, `2.68B` tokens)",
        "",
        "| Method @ LR | Held-out (Mean ± SD) | Δ vs `MuonH*` (mnat) | Δ vs `MuonTH*` (mnat) | Δ vs `MuonT` (mnat) | Late-11 | C4 (Mean ± SD) | Paloma 16-Dom | Max `\\|W\\|_F` Drift |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t, _ in LR_GRID:
        for arm_key in REPORT_ARMS:
            key = f"{arm_key}@{t}"
            if key not in seeds123:
                continue
            v = seeds123[key]
            d_mh = v.get("paired_delta_vs_muonh_opt_mnat")
            d_mth = v.get("paired_delta_vs_muonth_opt_mnat")
            d_mt = v.get("paired_delta_vs_muont_0p016_mnat")
            mh_s = f"{d_mh['mean']:+.2f} ± {d_mh['std']:.2f}" if d_mh else "—"
            mth_s = f"{d_mth['mean']:+.2f} ± {d_mth['std']:.2f}" if d_mth else "—"
            mt_s = f"{d_mt['mean']:+.2f} ± {d_mt['std']:.2f}" if d_mt else "—"
            pal_s = f"{v['paloma_mean']:.5f} ± {v['paloma_std']:.5f}" if v["paloma_mean"] is not None else "—"
            drift_s = f"`{v['max_block_rel_norm_drift']:.2e}`" if v["max_block_rel_norm_drift"] is not None else "—"
            md.append(
                f"| {v['label']} @ `{v['lr']:.5f}` | {v['sel_mean']:.5f} ± {v['sel_std']:.5f} | "
                f"{mh_s} | {mth_s} | {mt_s} | {v['late11_sel_mean']:.5f} | "
                f"{v['c4_mean']:.5f} ± {v['c4_std']:.5f} | {pal_s} | {drift_s} |"
            )

    md_text = "\n".join(md) + "\n"
    atomic_text(res_dir / "MUONH_RESULTS.md", md_text)
    atomic_text(CAMPAIGN_ROOT / "MUONH_RESULTS.md", md_text)
    log(f"Wrote consolidated results to {res_dir / 'MUONH_RESULTS.md'}")


def ensure_marin_exact_smoke_gate(node: str, configs_index: dict[str, dict], gate_lock: threading.Lock) -> None:
    """Run an 8-step (4+4 resume) GPU smoke gate for Marin-exact `wt_mlp_lag_muonh_marin` on the first freed node."""
    hb_dir = CAMPAIGN_ROOT / "smoke" / "marin_exact_resume_8step"
    hb_marker = hb_dir / "passed.json"
    if hb_marker.is_file():
        return
    with gate_lock:
        if hb_marker.is_file():
            return
        hb_dir.mkdir(parents=True, exist_ok=True)
        base_cfg = json.loads(Path(configs_index["wt_mlp_lag_muonh_marin_130m_1x_seed0_lr0p016"]["config"]).read_text())
        base_cfg["experiment"]["checkpoint_every"] = 4
        smoke_cfg_path = hb_dir / "smoke_cfg.json"
        smoke_cfg_path.write_text(json.dumps(base_cfg, indent=2) + "\n")
        unint_out = hb_dir / "uninterrupted_8"
        res_out = hb_dir / "resumed_4_plus_4"
        hb_sh = hb_dir / "run_marin_hb_smoke.sh"
        hb_sh.write_text(
            "set -euo pipefail\n"
            f"rm -rf {shlex.quote(str(unint_out))} {shlex.quote(str(res_out))}\n"
            f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
            f"-m experiment.train --config {shlex.quote(str(smoke_cfg_path))} "
            f"--data {shlex.quote(str(DATA_DIR))} --data-receipt {shlex.quote(str(DATA_RECEIPT))} "
            f"--out {shlex.quote(str(unint_out))} --max-steps 8\n"
            f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
            f"-m experiment.train --config {shlex.quote(str(smoke_cfg_path))} "
            f"--data {shlex.quote(str(DATA_DIR))} --data-receipt {shlex.quote(str(DATA_RECEIPT))} "
            f"--out {shlex.quote(str(res_out))} --max-steps 4\n"
            f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
            f"-m experiment.train --config {shlex.quote(str(smoke_cfg_path))} "
            f"--data {shlex.quote(str(DATA_DIR))} --data-receipt {shlex.quote(str(DATA_RECEIPT))} "
            f"--out {shlex.quote(str(res_out))} --resume {shlex.quote(str(res_out / 'checkpoints' / 'step_000004'))} --max-steps 8\n"
        )
        run_owned_on_node(node, "smoke_marin_exact_resume_8step", ["bash", str(hb_sh)])
        comp = compare_continuation(
            unint_out / "checkpoints" / "step_000008",
            res_out / "checkpoints" / "step_000008",
            world=8,
        )
        log(f"8-step Marin-Exact (1pct-noclip + AdamH + quintic5) continuation gate PASSED on {node}: {comp}")
        atomic_json(hb_marker, {"time": now(), "node": node, "continuation": comp})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print initial job queue without launching GPU jobs.")
    parser.add_argument("--drain-v1-pid", type=int, default=None, help="Existing gen1 orchestrator PID draining active runs.")
    args = parser.parse_args()

    configs_index = generate_all()
    sched = CampaignScheduler(configs_index)
    if args.dry_run:
        snap = sched.live_snapshot("dry_run", v1_pid=args.drain_v1_pid)
        print(json.dumps(snap, indent=2))
        return

    if args.drain_v1_pid is not None:
        drain_json = CAMPAIGN_ROOT / "drain_orch.json"
        atomic_json(
            drain_json,
            {"time": now(), "min_orch_gen": ORCH_GEN, "draining_prior_pid": args.drain_v1_pid, "gen_pid": os.getpid()},
        )
        log(f"Activated {drain_json} (min_orch_gen={ORCH_GEN}) to cleanly drain prior orchestrator pid={args.drain_v1_pid}.")

    log("=== Deploying updated Option-A MuonH workspace to all 4 A100 nodes ===")
    subprocess.run([PYTHON, str(TASK / "campaign" / "deploy_muonh_campaign.py")], check=True)

    log("=== Checking preflight gates ===")
    preflight_gates(configs_index)

    log(f"=== Starting 4-node (32xA100) gen{ORCH_GEN} dynamic worker pool (Option A WSD -> 0.0) ===")
    report_lock = threading.Lock()
    marin_gate_lock = threading.Lock()
    stop_monitor = threading.Event()

    def write_snapshots(phase: str) -> None:
        snap = sched.live_snapshot(phase, v1_pid=args.drain_v1_pid)
        atomic_json(CAMPAIGN_ROOT / "progress_v3.json", snap)
        atomic_json(TASK / "results" / "progress_v3.json", snap)
        atomic_json(CAMPAIGN_ROOT / "progress_v2.json", snap)
        atomic_json(TASK / "results" / "progress_v2.json", snap)
        atomic_json(CAMPAIGN_ROOT / "progress.json", snap)
        atomic_json(TASK / "results" / "progress.json", snap)

    write_snapshots("running")

    def monitor_loop() -> None:
        while not stop_monitor.is_set():
            try:
                write_snapshots("running")
            except Exception as exc:
                log(f"Monitor snapshot warning: {exc!r}")
            stop_monitor.wait(30.0)

    mon_thread = threading.Thread(target=monitor_loop, daemon=True)
    mon_thread.start()

    def node_worker(node: str) -> None:
        wait_for_node_drain(node)
        with report_lock:
            write_final_reports(sched, {})
        write_snapshots("running")
        while True:
            key, all_done = sched.acquire_next(node)
            if all_done:
                return
            if key is None:
                time.sleep(15.0)
                continue
            write_snapshots("running")
            try:
                if key[0] in MARIN_ARMS:
                    ensure_marin_exact_smoke_gate(node, configs_index, marin_gate_lock)
                summary = execute_training_job(node, key, configs_index)
                sched.mark_complete(key, summary)
                with report_lock:
                    write_final_reports(sched, {})
                write_snapshots("running")
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
        write_snapshots("stage3_or_finished")

    if sched.failed:
        raise RuntimeError(f"Campaign encountered failures: {sched.failed}")

    for node in NODES:
        wait_for_node_drain(node)

    log("=== All training jobs COMPLETE; starting Stage 3 16-domain Paloma evaluation ===")
    paloma_new = run_paloma_stage()
    write_final_reports(sched, paloma_new)
    atomic_json(
        CAMPAIGN_ROOT / "completion.json",
        {"time": now(), "status": "complete", "orch_gen": ORCH_GEN, "summary": str(CAMPAIGN_ROOT / "muonh_summary.json")},
    )
    log("=== muonh-20261127 CAMPAIGN (GEN2 MARIN-EXACT + TONER-MATCHED) ALL COMPLETE ===")


if __name__ == "__main__":
    main()
