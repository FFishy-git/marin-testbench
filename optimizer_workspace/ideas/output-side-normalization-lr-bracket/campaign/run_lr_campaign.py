# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dynamic 4-node orchestrator for the full-budget LR-bracket & matched-optimum campaign."""

from __future__ import annotations

import argparse
import gzip
import json
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

from campaign.configure_lr_campaign import ARM_MAP, LR_BY_TAG, LR_GRID, generate_all, run_name

PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
CAMPAIGN_ROOT = PVC_ROOT / "campaigns" / "lr-bracket-20261125"
ARCHIVE_REC = Path("/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization/experiment_records")
DATA_DIR = PVC_ROOT / "data"
DATA_RECEIPT = DATA_DIR / "experiment_receipt.json"
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
LOGGING_PYTHON = PVC_ROOT / ".venv-logging" / "bin" / "python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]

# Existing verified 130M 1x Chinchilla runs (identical model, data receipt, and runner SHA256)
EXISTING_RUNS: dict[tuple[str, str, int], Path] = {
    ("muon", "0p016", 0): ARCHIVE_REC / "runs" / "wave0" / "muon",
    ("muon", "0p0226", 0): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave1b" / "muon_130m_1x_seed0_lr0p0226",
    ("muont", "0p016", 0): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave1a" / "muont_130m_1x_seed0_lr0p016",
    ("muont", "0p0226", 0): ARCHIVE_REC / "runs" / "wave_sym_fastlag" / "muont",
    ("muont", "0p032", 0): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave1a" / "muont_130m_1x_seed0_lr0p032",
    ("od_all", "0p0226", 0): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave1b" / "od_all_130m_1x_seed0_lr0p0226",
    ("od_all", "0p032", 0): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave1b" / "od_all_130m_1x_seed0_lr0p032",
    ("wt_mlp_lag", "0p016", 0): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave1a" / "wt_mlp_lag_130m_1x_seed0_lr0p016",
    ("wt_mlp_lag", "0p0226", 0): ARCHIVE_REC / "runs" / "wave_sym_fastlag" / "inner_outer_mlpdiag_lag",
    ("wt_mlp_lag", "0p032", 0): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave1a" / "wt_mlp_lag_130m_1x_seed0_lr0p032",
    # Seeds 1, 2, 3 from confirmation-20261121
    ("muon", "0p016", 1): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave2a" / "muon_130m_1x_seed1",
    ("muon", "0p016", 2): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave2b" / "muon_130m_1x_seed2",
    ("muon", "0p016", 3): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave2c" / "muon_130m_1x_seed3",
    ("muont", "0p016", 1): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave2a" / "muont_130m_1x_seed1",
    ("muont", "0p016", 2): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave2b" / "muont_130m_1x_seed2",
    ("muont", "0p016", 3): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave2c" / "muont_130m_1x_seed3",
    ("od_all", "0p0226", 1): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave2a" / "od_all_130m_1x_seed1",
    ("od_all", "0p0226", 2): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave2b" / "od_all_130m_1x_seed2",
    ("od_all", "0p0226", 3): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave2c" / "od_all_130m_1x_seed3",
    ("wt_mlp_lag", "0p0226", 1): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave2a" / "wt_mlp_lag_130m_1x_seed1",
    ("wt_mlp_lag", "0p0226", 2): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave2b" / "wt_mlp_lag_130m_1x_seed2",
    ("wt_mlp_lag", "0p0226", 3): ARCHIVE_REC / "campaigns" / "confirmation-20261121" / "runs" / "wave2c" / "wt_mlp_lag_130m_1x_seed3",
}

EXISTING_PALOMA: dict[tuple[str, str, int], Path] = {
    ("muon", "0p016", s): ARCHIVE_REC / "paloma_eval" / f"muon_130m_1x_seed{s}.json" for s in (1, 2, 3)
} | {
    ("muont", "0p016", s): ARCHIVE_REC / "paloma_eval" / f"muont_130m_1x_seed{s}.json" for s in (1, 2, 3)
} | {
    ("od_all", "0p0226", s): ARCHIVE_REC / "paloma_eval" / f"od_all_130m_1x_seed{s}.json" for s in (1, 2, 3)
} | {
    ("wt_mlp_lag", "0p0226", s): ARCHIVE_REC / "paloma_eval" / f"wt_mlp_lag_130m_1x_seed{s}.json" for s in (1, 2, 3)
}

LOG_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(UTC).isoformat()


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
            for line in mf.read_text().splitlines():
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


def smoke_parity_check(configs_index: dict[str, dict]) -> None:
    smoke_root = CAMPAIGN_ROOT / "smoke" / "parity_8step"
    marker = smoke_root / "passed.json"
    if marker.is_file():
        log(f"Smoke parity check already passed ({marker}), skipping.")
        return

    entry = configs_index["wt_mlp_lag_130m_1x_seed0_lr0p016"]
    out_dir = smoke_root / entry["name"]
    script_sh = smoke_root / "run_smoke.sh"
    smoke_root.mkdir(parents=True, exist_ok=True)
    script_sh.write_text(
        "set -euo pipefail\n"
        f"rm -rf {shlex.quote(str(out_dir))}\n"
        f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
        f"-m experiment.train --config {shlex.quote(entry['config'])} "
        f"--data {shlex.quote(str(DATA_DIR))} --data-receipt {shlex.quote(str(DATA_RECEIPT))} "
        f"--out {shlex.quote(str(out_dir))} --max-steps 8\n"
    )
    run_owned_on_node(NODES[0], "smoke_parity_8step", ["bash", str(script_sh)])

    new_rows = {int(r["step"]): float(r["train_loss"]) for r in _iter_metric_rows(out_dir)}
    ref_dir = EXISTING_RUNS[("wt_mlp_lag", "0p016", 0)]
    ref_rows = {int(r["step"]): float(r["train_loss"]) for r in _iter_metric_rows(ref_dir) if int(r["step"]) <= 8}
    diffs = {s: abs(new_rows[s] - ref_rows[s]) for s in range(1, 9)}
    max_diff = max(diffs.values())
    log(f"8-step parity vs archived wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016: max |d train_loss| = {max_diff:.3e}")
    if max_diff > 1e-5:
        raise RuntimeError(f"Parity check failed: max_diff={max_diff} ({diffs})")
    marker.write_text(json.dumps({"time": now(), "max_abs_train_loss_diff": max_diff, "diffs": diffs}, indent=2) + "\n")


class CampaignScheduler:
    """Thread-safe state machine that brackets seed-0 LRs and schedules missing seeds 1-3 runs."""

    def __init__(self, configs_index: dict[str, dict]) -> None:
        self.configs_index = configs_index
        self.lock = threading.Lock()
        self.completed: dict[tuple[str, str, int], dict] = {}
        self.in_flight: set[tuple[str, str, int]] = set()
        self.failed: list[tuple[tuple[str, str, int], str]] = []

        # Load all archived existing runs and any already-completed runs in CAMPAIGN_ROOT/runs
        for key, path in EXISTING_RUNS.items():
            s = read_run_summary(path)
            if s is None:
                raise RuntimeError(f"Missing expected existing reference run {key} at {path}")
            self.completed[key] = s

        runs_root = CAMPAIGN_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        for arm_key in ARM_MAP:
            for lr_tag, _ in LR_GRID:
                for seed in (0, 1, 2, 3):
                    key = (arm_key, lr_tag, seed)
                    if key in self.completed:
                        continue
                    out_dir = runs_root / run_name(arm_key, seed, lr_tag)
                    s = read_run_summary(out_dir)
                    if s is not None:
                        self.completed[key] = s
                        log(f"Loaded already-completed campaign run {out_dir.name}: sel={s['selection_loss']:.5f}")

    def _seed0_bracket_status(self, arm_key: str) -> tuple[bool, str | None, list[tuple[int, tuple[str, str, int]]]]:
        """Return (is_bracketed, best_lr_tag, needed_seed0_jobs_with_priority)."""
        needed: list[tuple[int, tuple[str, str, int]]] = []
        # Core 4-point grid {0p0113, 0p016, 0p0226, 0p032}
        core_tags = ("0p0113", "0p016", "0p0226", "0p032")
        # Wave-1 ordering priority so .9/.80/.144/.139 start the 4 diagnostic runs immediately
        base_prio = {
            ("muon", "0p0113"): 1,
            ("muont", "0p0113"): 2,
            ("od_all", "0p016"): 3,
            ("wt_mlp_lag", "0p0113"): 4,
            ("od_all", "0p0113"): 5,
            ("muon", "0p032"): 6,
        }
        missing_core = False
        for t in core_tags:
            k = (arm_key, t, 0)
            if k not in self.completed:
                missing_core = True
                needed.append((base_prio.get((arm_key, t), 10), k))
        if missing_core:
            return False, None, needed

        # All 4 core tags are complete; check where the minimum sits along LR_GRID
        tag_order = [t for t, _ in LR_GRID]  # ["0p0057", "0p008", "0p0113", "0p016", "0p0226", "0p032"]
        avail = [(idx, t, self.completed[(arm_key, t, 0)]["selection_loss"]) for idx, t in enumerate(tag_order) if (arm_key, t, 0) in self.completed]
        best_idx, best_tag, _ = min(avail, key=lambda item: item[2])

        # If best is at 0p0113 and 0p008 is not yet run, extend left to 0p008
        if best_idx == 2 and (arm_key, "0p008", 0) not in self.completed:
            needed.append((7, (arm_key, "0p008", 0)))
            return False, None, needed
        # If best is at 0p008 and 0p0057 is not yet run, extend left to 0p0057
        if best_idx == 1 and (arm_key, "0p0057", 0) not in self.completed:
            needed.append((8, (arm_key, "0p0057", 0)))
            return False, None, needed

        return True, best_tag, []

    def _desired_jobs(self) -> list[tuple[int, tuple[str, str, int]]]:
        candidates: list[tuple[int, tuple[str, str, int]]] = []
        bracketed: dict[str, str] = {}
        for arm_key in ("muon", "muont", "wt_mlp_lag", "od_all"):
            ok, best_tag, req = self._seed0_bracket_status(arm_key)
            candidates.extend(req)
            if ok and best_tag is not None:
                bracketed[arm_key] = best_tag

        # Stage 2: once an arm is bracketed on seed 0, ensure seeds 1, 2, 3 exist at its true optimum
        arm_seed_prio = {"wt_mlp_lag": 20, "od_all": 30, "muont": 40, "muon": 50}
        for arm_key, opt_tag in bracketed.items():
            for s in (1, 2, 3):
                k = (arm_key, opt_tag, s)
                if k not in self.completed:
                    candidates.append((arm_seed_prio[arm_key] + s, k))

        # Also ensure strictly matched-LR seeds 1-3 against MuonT's true optimum for WT-MLP-Lag and OD-All
        if "muont" in bracketed:
            mt_opt = bracketed["muont"]
            for extra_prio, arm_key in ((60, "wt_mlp_lag"), (70, "od_all")):
                if arm_key in bracketed and bracketed[arm_key] != mt_opt:
                    for s in (1, 2, 3):
                        k = (arm_key, mt_opt, s)
                        if k not in self.completed:
                            candidates.append((extra_prio + s, k))

        # Deduplicate keeping lowest priority number
        best_prio: dict[tuple[str, str, int], int] = {}
        for p, k in candidates:
            if k not in best_prio or p < best_prio[k]:
                best_prio[k] = p
        return sorted((p, k) for k, p in best_prio.items())

    def acquire_next(self) -> tuple[tuple[str, str, int] | None, bool]:
        """Return (job_key_or_None, all_done). If job_key is None and not all_done, worker should wait."""
        with self.lock:
            if self.failed:
                return None, True
            desired = self._desired_jobs()
            if not desired and not self.in_flight:
                return None, True
            for _, key in desired:
                if key not in self.in_flight and key not in self.completed:
                    self.in_flight.add(key)
                    return key, False
            return None, False

    def mark_complete(self, key: tuple[str, str, int], summary: dict) -> None:
        with self.lock:
            self.in_flight.discard(key)
            self.completed[key] = summary

    def mark_failed(self, key: tuple[str, str, int], err: str) -> None:
        with self.lock:
            self.in_flight.discard(key)
            self.failed.append((key, err))


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
        f"SUMMARY {name:36s} sel={summary['selection_loss']:.5f}  c4={summary['c4_loss']:.5f}  "
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
        if out_json.is_file():
            continue
        todo.append((run_dir, out_json))

    if todo:
        log(f"Evaluating 16-domain Paloma on {len(todo)} newly completed checkpoints across {min(len(NODES), len(todo))} nodes...")
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
    tag_order = [t for t, _ in LR_GRID]
    seed0_grid: dict[str, dict[str, dict]] = {arm: {} for arm in ARM_MAP}
    opt_tags: dict[str, str] = {}
    for arm_key in ("muon", "muont", "od_all", "wt_mlp_lag"):
        for t in tag_order:
            k = (arm_key, t, 0)
            if k in sched.completed:
                seed0_grid[arm_key][t] = sched.completed[k]
        best_t = min(seed0_grid[arm_key], key=lambda t: seed0_grid[arm_key][t]["selection_loss"])
        opt_tags[arm_key] = best_t

    def get_paloma(arm_key: str, lr_tag: str, seed: int) -> dict | None:
        if (arm_key, lr_tag, seed) in EXISTING_PALOMA:
            return json.loads(EXISTING_PALOMA[(arm_key, lr_tag, seed)].read_text())
        name = run_name(arm_key, seed, lr_tag)
        return paloma_new.get(name)

    seeds123: dict[str, dict] = {}
    for arm_key in ("muon", "muont", "od_all", "wt_mlp_lag"):
        for t in tag_order:
            if all((arm_key, t, s) in sched.completed for s in (1, 2, 3)):
                s_runs = [sched.completed[(arm_key, t, s)] for s in (1, 2, 3)]
                pals = [get_paloma(arm_key, t, s) for s in (1, 2, 3)]
                seeds123[f"{arm_key}@{t}"] = {
                    "arm_key": arm_key,
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

    payload = {
        "generated_at": now(),
        "optimal_seed0_lr_tags": opt_tags,
        "optimal_seed0_lrs": {k: LR_BY_TAG[t] for k, t in opt_tags.items()},
        "seed0_grid": seed0_grid,
        "seeds123_summaries": seeds123,
    }
    res_dir = TASK / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "lr_bracket_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (CAMPAIGN_ROOT / "lr_bracket_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    md = [
        "# Full-Budget LR-Bracket & Matched-Optimum Campaign Results",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## 1. Seed 0 Full-Budget (`5,120` steps = `2.68B` tokens) LR Sweep",
        "",
        "| Arm | " + " | ".join(f"`{t}` (`{LR_BY_TAG[t]:.5f}`)" for t in tag_order if any(t in seed0_grid[a] for a in ARM_MAP)) + " | True `η*` |",
        "| --- | " + " | ".join("---:" for t in tag_order if any(t in seed0_grid[a] for a in ARM_MAP)) + " | ---: |",
    ]
    used_tags = [t for t in tag_order if any(t in seed0_grid[a] for a in ARM_MAP)]
    for arm_key in ("muon", "muont", "od_all", "wt_mlp_lag"):
        cells = []
        for t in used_tags:
            if t in seed0_grid[arm_key]:
                s = seed0_grid[arm_key][t]
                val = f"{s['selection_loss']:.5f} (c4 {s['c4_loss']:.5f})"
                if t == opt_tags[arm_key]:
                    val = f"**{val}**"
                cells.append(val)
            else:
                cells.append("—")
        md.append(f"| `{arm_key}` | " + " | ".join(cells) + f" | `{LR_BY_TAG[opt_tags[arm_key]]:.5f}` |")

    md += [
        "",
        "## 2. Fresh Paired `seeds 1, 2, 3` (`130M 1×`, `2.68B` tokens) Summaries",
        "",
        "| Arm @ LR | Final `sel` (Mean ± SD) | Late-11 `sel` Mean | Final `c4` (Mean ± SD) | Paloma 16-Dom (Mean ± SD) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for k, v in sorted(seeds123.items()):
        pal_str = f"{v['paloma_mean']:.5f} ± {v['paloma_std']:.5f}" if v["paloma_mean"] is not None else "—"
        md.append(
            f"| `{k}` (`{v['lr']:.5f}`) | {v['sel_mean']:.5f} ± {v['sel_std']:.5f} | "
            f"{v['late11_sel_mean']:.5f} | {v['c4_mean']:.5f} ± {v['c4_std']:.5f} | {pal_str} |"
        )
    md_text = "\n".join(md) + "\n"
    (res_dir / "LR_BRACKET_RESULTS.md").write_text(md_text)
    (CAMPAIGN_ROOT / "LR_BRACKET_RESULTS.md").write_text(md_text)
    log(f"Wrote consolidated results to {res_dir / 'LR_BRACKET_RESULTS.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print initial job queue without launching GPU jobs.")
    args = parser.parse_args()

    configs_index = generate_all()
    sched = CampaignScheduler(configs_index)
    if args.dry_run:
        print("Initial desired jobs:", json.dumps([(p, k) for p, k in sched._desired_jobs()], indent=2))
        return

    log("=== Deploying lr-bracket-20261125 workspace to all 4 A100 nodes ===")
    subprocess.run([PYTHON, str(TASK / "campaign" / "deploy_lr_campaign.py")], check=True)

    log("=== Running 8-step bitwise parity smoke gate ===")
    smoke_parity_check(configs_index)

    log("=== Starting 4-node dynamic worker pool ===")

    report_lock = threading.Lock()

    def node_worker(node: str) -> None:
        while True:
            key, all_done = sched.acquire_next()
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
                    if LOGGING_PYTHON.is_file():
                        subprocess.run(
                            [str(LOGGING_PYTHON), "-u", str(TASK / "campaign" / "swanlab_upload_lr_campaign.py")],
                            check=False,
                        )
            except BaseException as exc:
                log(f"ERROR on {node} for {key}: {exc!r}")
                sched.mark_failed(key, repr(exc))
                raise

    with ThreadPoolExecutor(max_workers=len(NODES)) as pool:
        futs = {pool.submit(node_worker, n): n for n in NODES}
        for fut in as_completed(futs):
            fut.result()

    if sched.failed:
        raise RuntimeError(f"Campaign encountered failures: {sched.failed}")

    log("=== All Stage 1 & Stage 2 training jobs COMPLETE; starting Stage 3 16-domain Paloma evaluation ===")
    paloma_new = run_paloma_stage()
    write_final_reports(sched, paloma_new)

    if LOGGING_PYTHON.is_file():
        log("=== Uploading completed runs to SwanLab ===")
        subprocess.run([str(LOGGING_PYTHON), "-u", str(TASK / "campaign" / "swanlab_upload_lr_campaign.py")], check=True)
    log("=== lr-bracket-20261125 CAMPAIGN ALL COMPLETE ===")


if __name__ == "__main__":
    main()
