# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""4-node (32xA100) campaign runner for Muon2-ExpDiag efficiency & loss-vs-time Pareto research.

All control-aligned (`*_ctrl_*`) runs use `compile_ns=False` and identical `torch.compile` settings
as Muon, MuonH, MuonT, and Muon2-ExpDiag (paper_v4).
"""

from __future__ import annotations

import argparse
import gzip
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

from campaign.configure_eff_campaign import LR_BY_TAG, build_eff_130m_config
from campaign.deploy_eff_campaign import deploy

PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
CAMPAIGN_ROOT = PVC_ROOT / "campaigns" / "efficiency-20261128"
MUONH_ROOT = PVC_ROOT / "campaigns" / "muonh-20261127"
LR_BRACKET_ROOT = PVC_ROOT / "campaigns" / "lr-bracket-20261125"
CONFIRM_ROOT = PVC_ROOT / "campaigns" / "confirmation-20261121"
DATA_DIR = PVC_ROOT / "data"
DATA_RECEIPT = DATA_DIR / "experiment_receipt.json"
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]
ORCH_GEN = 99
LOG_LOCK = threading.Lock()

BASELINE_RUNS: dict[str, Path] = {
    "muon_seed0": PVC_ROOT / "runs" / "wave0" / "muon",
    "muon_seed1": CONFIRM_ROOT / "runs" / "wave2a" / "muon_130m_1x_seed1",
    "muon_seed2": CONFIRM_ROOT / "runs" / "wave2b" / "muon_130m_1x_seed2",
    "muon_seed3": CONFIRM_ROOT / "runs" / "wave2c" / "muon_130m_1x_seed3",
    "muonh_seed0": MUONH_ROOT / "runs" / "muonh_130m_1x_seed0_lr0p0113",
    "muonh_seed1": MUONH_ROOT / "runs" / "muonh_130m_1x_seed1_lr0p0113",
    "muonh_seed2": MUONH_ROOT / "runs" / "muonh_130m_1x_seed2_lr0p0113",
    "muonh_seed3": MUONH_ROOT / "runs" / "muonh_130m_1x_seed3_lr0p0113",
    "muont_seed0": CONFIRM_ROOT / "runs" / "wave1a" / "muont_130m_1x_seed0_lr0p016",
    "muont_seed1": CONFIRM_ROOT / "runs" / "wave2a" / "muont_130m_1x_seed1",
    "muont_seed2": CONFIRM_ROOT / "runs" / "wave2b" / "muont_130m_1x_seed2",
    "muont_seed3": CONFIRM_ROOT / "runs" / "wave2c" / "muont_130m_1x_seed3",
    "muonth_seed0": MUONH_ROOT / "runs" / "muonth_130m_1x_seed0_lr0p0113",
    "muonth_seed1": MUONH_ROOT / "runs" / "muonth_130m_1x_seed1_lr0p0113",
    "muonth_seed2": MUONH_ROOT / "runs" / "muonth_130m_1x_seed2_lr0p0113",
    "muonth_seed3": MUONH_ROOT / "runs" / "muonth_130m_1x_seed3_lr0p0113",
    "expdiag_orig_seed0": CONFIRM_ROOT / "runs" / "wave1a" / "wt_mlp_lag_130m_1x_seed0_lr0p016",
    "expdiag_orig_seed1": LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed1_lr0p016",
    "expdiag_orig_seed2": LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed2_lr0p016",
    "expdiag_orig_seed3": LR_BRACKET_ROOT / "runs" / "wt_mlp_lag_130m_1x_seed3_lr0p016",
    "expdiagh_orig_seed0": MUONH_ROOT / "runs" / "wt_mlp_lag_muonh_130m_1x_seed0_lr0p0113",
    "expdiagh_orig_seed1": MUONH_ROOT / "runs" / "wt_mlp_lag_muonh_130m_1x_seed1_lr0p0113",
    "expdiagh_orig_seed2": MUONH_ROOT / "runs" / "wt_mlp_lag_muonh_130m_1x_seed2_lr0p0113",
    "expdiagh_orig_seed3": MUONH_ROOT / "runs" / "wt_mlp_lag_muonh_130m_1x_seed3_lr0p0113",
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def log(msg: str) -> None:
    line = f"[{now()}] [eff] {msg}"
    with LOG_LOCK:
        print(line, flush=True)
        CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
        with (CAMPAIGN_ROOT / "orchestrator.log").open("a") as fh:
            fh.write(line + "\n")


def lazy_preset(every: int = 8, decay: float = 0.75, backend: str = "async_host_both") -> dict:
    return {
        "enabled": True,
        "input_every": every,
        "output_every": every,
        "warmup_until": 16,
        "warmup_every": 2,
        "capture_microbatches": 1,
        "input_ema_decay": decay,
        "output_ema_decay": decay,
        "root_backend": backend,
        "lag_input": backend == "async_host_both",
    }


def build_catalog() -> dict[str, dict]:
    l8 = lazy_preset(every=8, decay=0.75, backend="async_host_both")
    l12 = lazy_preset(every=12, decay=0.68, backend="async_host_both")
    l16 = lazy_preset(every=16, decay=0.60, backend="async_host_both")
    specs: dict[str, dict] = {
        # --- Strictly Control-Aligned Seed-0 Runs (compile_ns=False, identical compile settings to Muon/MuonH/MuonT) ---
        "expdiag_lazy8_ctrl_s0_lr0p016": build_eff_130m_config(
            label="Muon2-ExpDiag-Lazy8 (control-aligned, compile_ns=False)",
            arm="inner_outer_mlpdiag_lag",
            seed=0,
            lr=LR_BY_TAG["0p016"],
            lazy=l8,
            compile_ns=False,
        ),
        "headdiag_qk_lazy12_ctrl_s0_lr0p016": build_eff_130m_config(
            label="Muon2-HeadDiag-Lazy12 (control-aligned, compile_ns=False)",
            arm="headdiag_qk_mlpdiag_lag",
            seed=0,
            lr=LR_BY_TAG["0p016"],
            lazy=l12,
            compile_ns=False,
        ),
        "headdiag_qk_lazy8_ctrl_s0_lr0p016": build_eff_130m_config(
            label="Muon2-HeadDiag-Lazy8 (control-aligned, compile_ns=False)",
            arm="headdiag_qk_mlpdiag_lag",
            seed=0,
            lr=LR_BY_TAG["0p016"],
            lazy=l8,
            compile_ns=False,
        ),
        "headdiag_qk_lazy16_ctrl_s0_lr0p016": build_eff_130m_config(
            label="Muon2-HeadDiag-Lazy16 (control-aligned, compile_ns=False)",
            arm="headdiag_qk_mlpdiag_lag",
            seed=0,
            lr=LR_BY_TAG["0p016"],
            lazy=l16,
            compile_ns=False,
        ),
        "expdiag_metrichb_wd0p2_lazy12_ctrl_s0_lr0p0113": build_eff_130m_config(
            label="Muon2-ExpDiag-MetricHB-WD0.2-Lazy12 (control-aligned, compile_ns=False)",
            arm="inner_outer_mlpdiag_lag",
            seed=0,
            lr=LR_BY_TAG["0p0113"],
            decay_mode="metric_hyperball",
            weight_decay=0.2,
            lazy=l12,
            compile_ns=False,
        ),
        "headdiag_qk_metrichb_wd0p15_lazy12_ctrl_s0_lr0p0135": build_eff_130m_config(
            label="Muon2-HeadDiag-MetricHB-WD0.15-Lazy12 (lr=0.01345, control-aligned, compile_ns=False)",
            arm="headdiag_qk_mlpdiag_lag",
            seed=0,
            lr=LR_BY_TAG["0p0135"],
            decay_mode="metric_hyperball",
            weight_decay=0.15,
            lazy=l12,
            compile_ns=False,
        ),
        "headdiag_qk_metrichb_wd0p2_lazy12_ctrl_s0_lr0p0113": build_eff_130m_config(
            label="Muon2-HeadDiag-MetricHB-WD0.2-Lazy12 (lr=0.01131, control-aligned, compile_ns=False)",
            arm="headdiag_qk_mlpdiag_lag",
            seed=0,
            lr=LR_BY_TAG["0p0113"],
            decay_mode="metric_hyperball",
            weight_decay=0.2,
            lazy=l12,
            compile_ns=False,
        ),
        "headdiag_qk_metrichb_wd0p15_lazy12_iso5030_ctrl_s0_lr0p0135": build_eff_130m_config(
            label="Muon2-HeadDiag-MetricHB-Lazy12 (Iso-Time 5030 steps, control-aligned, compile_ns=False)",
            arm="headdiag_qk_mlpdiag_lag",
            seed=0,
            lr=LR_BY_TAG["0p0135"],
            steps=5030,
            decay_mode="metric_hyperball",
            weight_decay=0.15,
            lazy=l12,
            compile_ns=False,
        ),
    }
    # Fresh multi-seed (Seeds 1, 2, 3) variants (all strictly control-aligned: compile_ns=False)
    for s in (1, 2, 3):
        specs[f"expdiag_lazy8_ctrl_s{s}_lr0p016"] = build_eff_130m_config(
            label="Muon2-ExpDiag-Lazy8 (control-aligned)",
            arm="inner_outer_mlpdiag_lag",
            seed=s,
            lr=LR_BY_TAG["0p016"],
            lazy=l8,
            compile_ns=False,
        )
        specs[f"headdiag_qk_lazy12_ctrl_s{s}_lr0p016"] = build_eff_130m_config(
            label="Muon2-HeadDiag-Lazy12 (control-aligned)",
            arm="headdiag_qk_mlpdiag_lag",
            seed=s,
            lr=LR_BY_TAG["0p016"],
            lazy=l12,
            compile_ns=False,
        )
        specs[f"headdiag_qk_lazy8_ctrl_s{s}_lr0p016"] = build_eff_130m_config(
            label="Muon2-HeadDiag-Lazy8 (control-aligned)",
            arm="headdiag_qk_mlpdiag_lag",
            seed=s,
            lr=LR_BY_TAG["0p016"],
            lazy=l8,
            compile_ns=False,
        )
        specs[f"expdiag_metrichb_wd0p2_lazy12_ctrl_s{s}_lr0p0113"] = build_eff_130m_config(
            label="Muon2-ExpDiag-MetricHB-WD0.2-Lazy12 (control-aligned)",
            arm="inner_outer_mlpdiag_lag",
            seed=s,
            lr=LR_BY_TAG["0p0113"],
            decay_mode="metric_hyperball",
            weight_decay=0.2,
            lazy=l12,
            compile_ns=False,
        )
        specs[f"headdiag_qk_metrichb_wd0p15_lazy12_ctrl_s{s}_lr0p0135"] = build_eff_130m_config(
            label="Muon2-HeadDiag-MetricHB-WD0.15-Lazy12 (control-aligned)",
            arm="headdiag_qk_mlpdiag_lag",
            seed=s,
            lr=LR_BY_TAG["0p0135"],
            decay_mode="metric_hyperball",
            weight_decay=0.15,
            lazy=l12,
            compile_ns=False,
        )
        specs[f"headdiag_qk_metrichb_wd0p2_lazy12_ctrl_s{s}_lr0p0113"] = build_eff_130m_config(
            label="Muon2-HeadDiag-MetricHB-WD0.2-Lazy12 (control-aligned)",
            arm="headdiag_qk_mlpdiag_lag",
            seed=s,
            lr=LR_BY_TAG["0p0113"],
            decay_mode="metric_hyperball",
            weight_decay=0.2,
            lazy=l12,
            compile_ns=False,
        )
    return specs


WAVES: dict[str, list[str]] = {
    "wave2_ctrl": [
        "expdiag_lazy8_ctrl_s0_lr0p016",
        "headdiag_qk_lazy12_ctrl_s0_lr0p016",
        "expdiag_metrichb_wd0p2_lazy12_ctrl_s0_lr0p0113",
        "headdiag_qk_metrichb_wd0p15_lazy12_ctrl_s0_lr0p0135",
    ],
    "wave3_confirm": [
        "headdiag_qk_metrichb_wd0p15_lazy12_ctrl_s1_lr0p0135",
        "headdiag_qk_metrichb_wd0p15_lazy12_ctrl_s2_lr0p0135",
        "headdiag_qk_metrichb_wd0p15_lazy12_ctrl_s3_lr0p0135",
        "headdiag_qk_metrichb_wd0p15_lazy12_iso5030_ctrl_s0_lr0p0135",
    ],
}


def iter_metric_rows(run_dir: Path) -> list[dict]:
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


def pure_step_seconds(row: dict) -> float:
    return max(
        0.0,
        float(row.get("step_seconds", 0.0))
        - float(row.get("eval_seconds", 0.0))
        - float(row.get("checkpoint_seconds", 0.0)),
    )


def read_run_summary(run_dir: Path) -> dict | None:
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        return None
    try:
        meta = json.loads(run_json.read_text())
    except Exception:
        return None
    rows = iter_metric_rows(run_dir)
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
    pure_all = [pure_step_seconds(r) for r in ordered]
    steady_pure = [pure_step_seconds(r) for r in ordered[20:] if "eval_seconds" not in r and "checkpoint_seconds" not in r]
    cfg_opt = meta.get("config", {}).get("optimizer", {})
    return {
        "name": run_dir.name,
        "status": "complete",
        "arm": meta.get("config", {}).get("experiment", {}).get("arm"),
        "arm_label": meta.get("config", {}).get("experiment", {}).get("arm_label"),
        "decay_mode": meta.get("config", {}).get("experiment", {}).get("decay_mode", "default"),
        "compile_ns": bool(cfg_opt.get("compile_ns", False)),
        "seed": int(meta.get("config", {}).get("training", {}).get("seed", 0)),
        "lr": float(cfg_opt.get("lr")),
        "steps": max(by_step),
        "planned_steps": planned,
        "selection_loss": float(final_eval["selection_loss"]),
        "c4_loss": float(final_eval["c4_loss"]),
        "late11_sel": statistics.mean(float(r["selection_loss"]) for r in late_evals) if late_evals else None,
        "late11_c4": statistics.mean(float(r["c4_loss"]) for r in late_evals) if late_evals else None,
        "pure_train_minutes": sum(pure_all) / 60.0,
        "steady_median_step_s": statistics.median(steady_pure) if steady_pure else None,
        "steady_mean_step_s": statistics.mean(steady_pure) if steady_pure else None,
        "peak_allocated_gib": max((r.get("peak_allocated_bytes", 0) for r in ordered), default=0) / (1024**3),
        "elapsed_seconds": meta.get("elapsed_seconds"),
        "run_dir": str(run_dir),
    }


def run_owned_on_node(node: str, tag: str, inner_cmd: list[str]) -> dict:
    alloc_dir = CAMPAIGN_ROOT / "allocations"
    alloc_dir.mkdir(parents=True, exist_ok=True)
    slug = node.replace(".", "_")
    state_path = alloc_dir / f"{tag}_{slug}.state.json"
    log_path = alloc_dir / f"{tag}_{slug}.log"
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
    local_ips = set(subprocess.check_output(["hostname", "-I"], text=True).split())
    if node in local_ips:
        proc = subprocess.run(argv, cwd=TASK, text=True, capture_output=True)
    else:
        proc = subprocess.run(
            [*SSH, f"root@{node}", f"cd {shlex.quote(str(TASK))} && {shlex.join(argv)}"],
            text=True,
            capture_output=True,
        )
    if proc.returncode != 0:
        tail = log_path.read_text()[-3000:] if log_path.exists() else ""
        raise RuntimeError(
            f"[{node}] {tag} failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout[-1500:]}\n"
            f"STDERR:\n{proc.stderr[-1500:]}\nLOG TAIL:\n{tail}"
        )
    elapsed = time.perf_counter() - t0
    log(f"DONE  [{node}] {tag} in {elapsed / 60.0:.2f} min")
    return {"node": node, "tag": tag, "elapsed_seconds": elapsed}


def run_job_on_node(node: str, job_name: str, cfg_path: Path, with_paloma: bool = False) -> dict:
    runs_root = CAMPAIGN_ROOT / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    out_dir = runs_root / job_name
    existing = read_run_summary(out_dir)
    if existing is None:
        resume_args: list[str] = []
        latest = out_dir / "latest_checkpoint.json"
        if latest.is_file():
            ckpt = json.loads(latest.read_text())["path"]
            resume_args = ["--resume", ckpt]
        elif out_dir.exists():
            backup = out_dir.with_name(f"{out_dir.name}.incomplete.{int(time.time())}")
            out_dir.rename(backup)
        train_cmd = [
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
            *resume_args,
        ]
        run_owned_on_node(node, f"train_{job_name}_{int(time.time())}", train_cmd)
        existing = read_run_summary(out_dir)
        if existing is None:
            raise RuntimeError(f"Job {job_name} exited 0 without complete summary!")
    else:
        log(f"SKIP  [{node}] {job_name} already complete (sel={existing['selection_loss']:.5f}, pure={existing['pure_train_minutes']:.2f}m)")

    if with_paloma:
        pal_dir = CAMPAIGN_ROOT / "paloma_eval"
        pal_dir.mkdir(parents=True, exist_ok=True)
        pal_out = pal_dir / f"{job_name}.json"
        if not pal_out.is_file():
            pal_cmd = [
                PYTHON,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                str(TASK / "campaign" / "eval_paloma.py"),
                "--run-dir",
                str(out_dir),
                "--data",
                str(DATA_DIR),
                "--out",
                str(pal_out),
            ]
            run_owned_on_node(node, f"paloma_{job_name}_{int(time.time())}", pal_cmd)
        existing["paloma_macro_mean"] = json.loads(pal_out.read_text())["paloma_macro_mean"]

    log(
        f"RESULT {job_name:48s} | compile_ns={existing['compile_ns']} | sel={existing['selection_loss']:.5f} "
        f"| c4={existing['c4_loss']:.5f} | late11={existing['late11_sel']:.5f} | pure={existing['pure_train_minutes']:.2f}m "
        f"| med={existing['steady_median_step_s']:.4f}s | mean={existing['steady_mean_step_s']:.4f}s"
    )
    return existing


def write_summary() -> dict:
    baselines = {}
    for b_key, b_path in BASELINE_RUNS.items():
        if b_path.exists():
            s = read_run_summary(b_path)
            if s is not None:
                baselines[b_key] = s
    campaign_runs = {}
    runs_root = CAMPAIGN_ROOT / "runs"
    pal_dir = CAMPAIGN_ROOT / "paloma_eval"
    if runs_root.is_dir():
        for rdir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            s = read_run_summary(rdir)
            if s is not None:
                pfile = pal_dir / f"{rdir.name}.json"
                if pfile.is_file():
                    s["paloma_macro_mean"] = json.loads(pfile.read_text())["paloma_macro_mean"]
                campaign_runs[rdir.name] = s
    payload = {"updated_at": now(), "baselines": baselines, "campaign_runs": campaign_runs}
    atomic_json(CAMPAIGN_ROOT / "eff_summary.json", payload)
    atomic_json(TASK / "results" / "eff_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", choices=sorted(WAVES.keys()), help="Named wave of jobs to execute.")
    parser.add_argument("--jobs", help="Comma-separated job names from catalog (or custom JSON stems in configs/campaign/).")
    parser.add_argument("--with-paloma", action="store_true", help="Run 16-domain Paloma evaluation immediately on completion.")
    parser.add_argument("--summary-only", action="store_true", help="Refresh eff_summary.json and print table.")
    args = parser.parse_args()

    cfg_dir = TASK / "configs" / "campaign"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog()
    for name, cfg in catalog.items():
        (cfg_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")

    if args.summary_only:
        payload = write_summary()
        print(json.dumps(payload, indent=2))
        return

    job_names: list[str] = []
    if args.wave:
        job_names.extend(WAVES[args.wave])
    if args.jobs:
        job_names.extend([x.strip() for x in args.jobs.split(",") if x.strip()])
    if not job_names:
        raise ValueError("Specify --wave, --jobs, or --summary-only")

    for jn in job_names:
        if not (cfg_dir / f"{jn}.json").is_file():
            raise FileNotFoundError(f"Unknown job {jn}; config not found at {cfg_dir / f'{jn}.json'}")

    deploy()

    queue_lock = threading.Lock()
    pending = list(job_names)
    results: dict[str, dict] = {}
    errors: list[tuple[str, str]] = []

    def node_worker(node: str) -> None:
        while True:
            with queue_lock:
                if errors or not pending:
                    return
                jn = pending.pop(0)
            try:
                res = run_job_on_node(node, jn, cfg_dir / f"{jn}.json", with_paloma=args.with_paloma)
                with queue_lock:
                    results[jn] = res
                write_summary()
            except Exception as exc:
                log(f"ERROR [{node}] {jn}: {exc}")
                with queue_lock:
                    errors.append((jn, repr(exc)))
                return

    with ThreadPoolExecutor(max_workers=min(len(NODES), len(job_names))) as pool:
        futs = [pool.submit(node_worker, node) for node in NODES[: min(len(NODES), len(job_names))]]
        for fut in as_completed(futs):
            fut.result()

    write_summary()
    if errors:
        raise RuntimeError(f"Campaign had {len(errors)} failed job(s): {errors}")
    log(f"Finished {len(results)} job(s) cleanly. Summary written to {TASK / 'results' / 'eff_summary.json'}.")


if __name__ == "__main__":
    main()
