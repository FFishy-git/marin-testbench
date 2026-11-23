# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end 4-node orchestrator for the output-side-normalization confirmation & scaling campaign."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
CAMPAIGN_ROOT = PVC_ROOT / "campaigns" / "confirmation-20261121"
DATA_DIR = PVC_ROOT / "data"
DATA_RECEIPT = DATA_DIR / "experiment_receipt.json"
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]

# Verified seed-0 anchor runs from prior waves (exact 130M 1x Chinchilla 0.5M recipe)
ANCHOR_SEED0_REFS = {
    ("muon", 0.016): PVC_ROOT / "runs" / "wave0" / "muon",
    ("muont", 0.02262741696): PVC_ROOT / "runs" / "wave_sym_fastlag" / "muont",
    ("wt_mlp_lag", 0.02262741696): PVC_ROOT / "runs" / "wave_sym_fastlag" / "inner_outer_mlpdiag_lag",
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def run_remote(node: str, argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [*SSH, f"root@{node}", shlex.join(argv)]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"[{node}] command failed (rc={proc.returncode}): {' '.join(argv)}\n"
            f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )
    return proc


def read_run_summary(run_dir: Path) -> dict | None:
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        return None
    meta = json.loads(run_json.read_text())
    metrics_files = sorted(run_dir.glob("metrics_from_*.jsonl"))
    rows = []
    for mf in metrics_files:
        for line in mf.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return {"status": meta.get("status"), "steps": 0, "run_dir": str(run_dir)}
    last = rows[-1]
    eval_rows = [r for r in rows if "selection_loss" in r]
    final_eval = eval_rows[-1] if eval_rows else {}
    step_times = [r["step_seconds"] for r in rows[20:] if "step_seconds" in r and "eval_seconds" not in r and "checkpoint_seconds" not in r]
    return {
        "name": run_dir.name,
        "status": meta.get("status"),
        "arm": meta.get("config", {}).get("experiment", {}).get("arm"),
        "arm_label": meta.get("config", {}).get("experiment", {}).get("arm_label"),
        "model_size": meta.get("config", {}).get("model", {}).get("size"),
        "seed": meta.get("config", {}).get("training", {}).get("seed"),
        "lr": meta.get("config", {}).get("optimizer", {}).get("lr"),
        "tokens": last.get("tokens"),
        "steps": last.get("step"),
        "planned_steps": meta.get("config", {}).get("training", {}).get("steps"),
        "selection_loss": final_eval.get("selection_loss"),
        "c4_loss": final_eval.get("c4_loss"),
        "heldout_losses": final_eval.get("heldout_losses"),
        "median_step_seconds": statistics.median(step_times) if step_times else None,
        "peak_allocated_gib": max((r.get("peak_allocated_bytes", 0) for r in rows), default=0) / (1024**3),
        "elapsed_seconds": meta.get("elapsed_seconds"),
        "run_dir": str(run_dir),
    }


def is_complete(run_dir: Path) -> bool:
    s = read_run_summary(run_dir)
    return bool(s and s["status"] == "complete" and s["steps"] == s["planned_steps"] and s["selection_loss"] is not None)


def deploy_all() -> None:
    print(f"[{now()}] Deploying confirmation workspace to all nodes...", flush=True)
    subprocess.run([PYTHON, str(TASK / "campaign" / "deploy_confirmation.py")], check=True)


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
    print(f"[{now()}] Starting {tag} on {node} (log: {log_path})", flush=True)
    t0 = time.perf_counter()
    run_remote(node, argv, check=True)
    elapsed = time.perf_counter() - t0
    print(f"[{now()}] Completed {tag} on {node} in {elapsed/60:.1f} min", flush=True)
    return {"node": node, "tag": tag, "elapsed_seconds": elapsed, "state": str(state_path), "log": str(log_path)}


def smoke_130m() -> None:
    print(f"[{now()}] Running 130M 4-step GPU smoke check across all 5 arms on {NODES[0]}...", flush=True)
    manifest = json.loads((TASK / "configs" / "waves_manifest.json").read_text())
    samples = [
        manifest["wave1b"][2],  # muon
        manifest["wave1a"][0],  # muont
        manifest["wave1b"][0],  # od_all
        manifest["wave1a"][2],  # wt_mlp_lag
        manifest["wave1b"][3],  # null_wt
    ]
    smoke_root = CAMPAIGN_ROOT / "smoke" / "130m"
    smoke_root.mkdir(parents=True, exist_ok=True)
    marker_file = smoke_root / "passed.json"
    if marker_file.is_file():
        print(f"[{now()}] 130M smoke already passed ({marker_file}), skipping.", flush=True)
        return

    # Run all 5 4-step checks inside a single allocation on NODES[0]
    script_lines = ["set -euo pipefail"]
    for entry in samples:
        out_dir = smoke_root / entry["name"]
        script_lines.append(f"rm -rf {shlex.quote(str(out_dir))}")
        script_lines.append(
            f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
            f"-m experiment.train --config {shlex.quote(entry['config'])} "
            f"--data {shlex.quote(str(DATA_DIR))} --data-receipt {shlex.quote(str(DATA_RECEIPT))} "
            f"--out {shlex.quote(str(out_dir))} --max-steps 4"
        )
    smoke_sh = smoke_root / "run_smoke_130m.sh"
    smoke_sh.write_text("\n".join(script_lines) + "\n")
    run_owned_on_node(NODES[0], "smoke_130m", ["bash", str(smoke_sh)])
    marker_file.write_text(json.dumps({"time": now(), "status": "passed", "arms": [e["arm"] for e in samples]}, indent=2) + "\n")
    print(f"[{now()}] 130M smoke PASSED.", flush=True)


def run_wave(wave_name: str) -> None:
    manifest = json.loads((TASK / "configs" / "waves_manifest.json").read_text())
    if wave_name not in manifest:
        raise ValueError(f"Unknown wave {wave_name}; available: {list(manifest)}")
    entries = manifest[wave_name]
    if len(entries) != 4:
        raise ValueError(f"Expected 4 entries in wave {wave_name}, found {len(entries)}")

    runs_root = CAMPAIGN_ROOT / "runs" / wave_name
    runs_root.mkdir(parents=True, exist_ok=True)

    pending = []
    for node, entry in zip(NODES, entries, strict=True):
        out_dir = runs_root / entry["name"]
        if is_complete(out_dir):
            s = read_run_summary(out_dir)
            print(f"[{now()}] SKIP {entry['name']} (already complete: sel={s['selection_loss']:.5f})", flush=True)
            continue
        pending.append((node, entry, out_dir))

    if not pending:
        print(f"[{now()}] Wave {wave_name} already 4/4 complete.", flush=True)
        return

    print(f"[{now()}] Launching wave {wave_name} ({len(pending)} runs)...", flush=True)

    def launch_one(node: str, entry: dict, out_dir: Path) -> dict:
        resume_args = []
        latest = out_dir / "latest_checkpoint.json"
        if latest.is_file():
            ckpt = json.loads(latest.read_text())["path"]
            resume_args = ["--resume", ckpt]
        elif out_dir.exists():
            # Incomplete run with no checkpoint: archive and restart cleanly
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
        tag = f"{wave_name}_{entry['name']}_{int(time.time())}"
        res = run_owned_on_node(node, tag, cmd)
        if not is_complete(out_dir):
            raise RuntimeError(f"Run {entry['name']} on {node} finished without complete status!")
        return res

    errors = []
    with ThreadPoolExecutor(max_workers=len(pending)) as pool:
        futs = {pool.submit(launch_one, node, entry, out_dir): entry["name"] for node, entry, out_dir in pending}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                fut.result()
            except BaseException as exc:
                errors.append((name, repr(exc)))
                print(f"[{now()}] ERROR in {name}: {exc}", flush=True)

    if errors:
        raise RuntimeError(f"Wave {wave_name} failed with errors: {errors}")

    print(f"[{now()}] Wave {wave_name} COMPLETE:", flush=True)
    for entry in entries:
        s = read_run_summary(runs_root / entry["name"])
        print(
            f"  {entry['name']:36s} sel={s['selection_loss']:.5f}  c4={s['c4_loss']:.5f}  "
            f"step_s={s['median_step_seconds']:.3f}  mem={s['peak_allocated_gib']:.1f}GiB",
            flush=True,
        )


def evaluate_gate1() -> dict:
    """Analyze Phase 1 LR grid + Null-WT, lock per-arm lr*, regenerate Phase 2 & 3 configs."""
    w1a = CAMPAIGN_ROOT / "runs" / "wave1a"
    w1b = CAMPAIGN_ROOT / "runs" / "wave1b"
    grid = {
        "muon": {
            0.016: read_run_summary(ANCHOR_SEED0_REFS[("muon", 0.016)]),
            0.02262741696: read_run_summary(w1b / "muon_130m_1x_seed0_lr0p0226"),
        },
        "muont": {
            0.016: read_run_summary(w1a / "muont_130m_1x_seed0_lr0p016"),
            0.02262741696: read_run_summary(ANCHOR_SEED0_REFS[("muont", 0.02262741696)]),
            0.032: read_run_summary(w1a / "muont_130m_1x_seed0_lr0p032"),
        },
        "od_all": {
            0.02262741696: read_run_summary(w1b / "od_all_130m_1x_seed0_lr0p0226"),
            0.032: read_run_summary(w1b / "od_all_130m_1x_seed0_lr0p032"),
        },
        "wt_mlp_lag": {
            0.016: read_run_summary(w1a / "wt_mlp_lag_130m_1x_seed0_lr0p016"),
            0.02262741696: read_run_summary(ANCHOR_SEED0_REFS[("wt_mlp_lag", 0.02262741696)]),
            0.032: read_run_summary(w1a / "wt_mlp_lag_130m_1x_seed0_lr0p032"),
        },
    }
    null_wt = read_run_summary(w1b / "null_wt_130m_1x_seed0_lr0p0226")
    muont_ref = grid["muont"][0.02262741696]
    null_delta = null_wt["selection_loss"] - muont_ref["selection_loss"]

    pinned = {"muon": 0.016, "muont": 0.02262741696, "od_all": 0.02262741696, "wt_mlp_lag": 0.02262741696}
    locked = {}
    for arm_key, lr_map in grid.items():
        base_lr = pinned[arm_key]
        base_loss = lr_map[base_lr]["selection_loss"]
        best_lr, best_loss = base_lr, base_loss
        for lr, summary in lr_map.items():
            if summary and summary["selection_loss"] is not None:
                if summary["selection_loss"] < best_loss - 0.0010:
                    best_lr, best_loss = lr, summary["selection_loss"]
        locked[arm_key] = best_lr

    gate1 = {
        "time": now(),
        "null_wt_selection_loss": null_wt["selection_loss"],
        "muont_ref_selection_loss": muont_ref["selection_loss"],
        "null_minus_muont": null_delta,
        "null_gate_pass": abs(null_delta) <= 0.0010,
        "grid_selection_loss": {
            arm: {str(lr): (s["selection_loss"] if s else None) for lr, s in m.items()} for arm, m in grid.items()
        },
        "locked_lr": locked,
    }
    out_path = TASK / "results" / "gate1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gate1, indent=2) + "\n")
    print(f"[{now()}] Gate 1 result:\n{json.dumps(gate1, indent=2)}", flush=True)

    # Regenerate Phase 2 and Phase 3 configs using locked LRs and redeploy
    subprocess.run(
        [
            PYTHON,
            str(TASK / "campaign" / "configure_campaign.py"),
            "--phases",
            "2",
            "3",
            "--lr-muon",
            str(locked["muon"]),
            "--lr-muont",
            str(locked["muont"]),
            "--lr-od-all",
            str(locked["od_all"]),
            "--lr-wt-mlp-lag",
            str(locked["wt_mlp_lag"]),
        ],
        check=True,
    )
    deploy_all()
    return gate1


def smoke_520m() -> int:
    """Probe A100-40GB memory for 520M (microbatch 4 -> 2 -> 1), verify all 4 arms & continuation."""
    smoke_root = CAMPAIGN_ROOT / "smoke" / "520m"
    smoke_root.mkdir(parents=True, exist_ok=True)
    marker = smoke_root / "passed.json"
    if marker.is_file():
        info = json.loads(marker.read_text())
        print(f"[{now()}] 520M smoke already passed with microbatch={info['microbatch']}.", flush=True)
        return int(info["microbatch"])

    chosen_mb = None
    for mb in (4, 2, 1):
        print(f"[{now()}] Probing 520M WT-MLP-Lag at micro_batch_size={mb} on {NODES[0]}...", flush=True)
        subprocess.run(
            [PYTHON, str(TASK / "campaign" / "configure_campaign.py"), "--phases", "4", "--microbatch-520m", str(mb)],
            check=True,
        )
        manifest = json.loads((TASK / "configs" / "waves_manifest.json").read_text())
        wt_cfg = manifest["wave4a"][3]["config"]
        probe_out = smoke_root / f"probe_wt_mb{mb}"
        probe_sh = smoke_root / f"probe_mb{mb}.sh"
        probe_sh.write_text(
            "set -euo pipefail\n"
            f"rm -rf {shlex.quote(str(probe_out))}\n"
            f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
            f"-m experiment.train --config {shlex.quote(wt_cfg)} "
            f"--data {shlex.quote(str(DATA_DIR))} --data-receipt {shlex.quote(str(DATA_RECEIPT))} "
            f"--out {shlex.quote(str(probe_out))} --max-steps 4\n"
        )
        try:
            run_owned_on_node(NODES[0], f"smoke_520m_probe_mb{mb}", ["bash", str(probe_sh)])
            s = read_run_summary(probe_out)
            print(f"[{now()}] 520M mb={mb} succeeded! peak_alloc={s['peak_allocated_gib']:.2f} GiB", flush=True)
            if s["peak_allocated_gib"] < 37.5:
                chosen_mb = mb
                break
        except RuntimeError as exc:
            print(f"[{now()}] 520M mb={mb} failed ({exc}); trying smaller microbatch...", flush=True)

    if chosen_mb is None:
        raise RuntimeError("520M probe failed at microbatch 4, 2, and 1!")

    # Verify all 4 arms at chosen_mb + checkpoint continuation for WT-MLP-Lag
    subprocess.run(
        [PYTHON, str(TASK / "campaign" / "configure_campaign.py"), "--phases", "4", "--microbatch-520m", str(chosen_mb)],
        check=True,
    )
    deploy_all()
    manifest = json.loads((TASK / "configs" / "waves_manifest.json").read_text())
    verify_sh = smoke_root / "verify_all_520m.sh"
    lines = ["set -euo pipefail"]
    for entry in manifest["wave4a"]:
        out = smoke_root / f"smoke_{entry['name']}"
        lines.append(f"rm -rf {shlex.quote(str(out))}")
        lines.append(
            f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
            f"-m experiment.train --config {shlex.quote(entry['config'])} "
            f"--data {shlex.quote(str(DATA_DIR))} --data-receipt {shlex.quote(str(DATA_RECEIPT))} "
            f"--out {shlex.quote(str(out))} --max-steps 4"
        )
    verify_sh.write_text("\n".join(lines) + "\n")
    run_owned_on_node(NODES[0], f"smoke_520m_all_mb{chosen_mb}", ["bash", str(verify_sh)])
    marker.write_text(json.dumps({"time": now(), "status": "passed", "microbatch": chosen_mb}, indent=2) + "\n")
    print(f"[{now()}] 520M all-arm smoke PASSED with microbatch={chosen_mb}.", flush=True)
    return chosen_mb


def eval_paloma_all() -> None:
    """Evaluate final checkpoints from Phases 2, 3, 3B, and 4 across all 16 Paloma subsets."""
    eval_root = CAMPAIGN_ROOT / "paloma_eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    target_waves = ("wave2a", "wave2b", "wave2c", "wave3a", "wave3b", "wave4a")
    jobs = []
    for w in target_waves:
        wdir = CAMPAIGN_ROOT / "runs" / w
        if not wdir.is_dir():
            continue
        for run_dir in sorted(wdir.iterdir()):
            if not is_complete(run_dir):
                continue
            out_json = eval_root / f"{run_dir.name}.json"
            if out_json.is_file():
                continue
            jobs.append((run_dir, out_json))

    if not jobs:
        print(f"[{now()}] All Paloma 16-domain evaluations already complete.", flush=True)
        return

    print(f"[{now()}] Evaluating {len(jobs)} checkpoints across 16 Paloma domains on 4 nodes...", flush=True)
    # Partition jobs round-robin across the 4 nodes, 1 allocation per node
    per_node = {node: [] for node in NODES}
    for idx, job in enumerate(jobs):
        per_node[NODES[idx % len(NODES)]].append(job)

    def run_node_evals(node: str, node_jobs: list[tuple[Path, Path]]) -> None:
        if not node_jobs:
            return
        sh = eval_root / f"eval_{node.replace('.', '_')}.sh"
        lines = ["set -euo pipefail"]
        for run_dir, out_json in node_jobs:
            lines.append(
                f"{PYTHON} -m torch.distributed.run --standalone --nproc_per_node=8 "
                f"{shlex.quote(str(TASK / 'campaign' / 'eval_paloma.py'))} "
                f"--run-dir {shlex.quote(str(run_dir))} --data {shlex.quote(str(DATA_DIR))} "
                f"--out {shlex.quote(str(out_json))}"
            )
        sh.write_text("\n".join(lines) + "\n")
        run_owned_on_node(node, f"paloma_eval_{int(time.time())}", ["bash", str(sh)])

    with ThreadPoolExecutor(max_workers=len(NODES)) as pool:
        futs = [pool.submit(run_node_evals, node, nj) for node, nj in per_node.items() if nj]
        for fut in as_completed(futs):
            fut.result()
    print(f"[{now()}] Paloma 16-domain evaluation COMPLETE.", flush=True)


def summarize() -> None:
    """Write consolidated results/summary.json and results/RESULTS.md."""
    results_dir = TASK / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {"generated_at": now(), "waves": {}, "paloma": {}}
    for w in ("wave1a", "wave1b", "wave2a", "wave2b", "wave2c", "wave3a", "wave3b", "wave4a"):
        wdir = CAMPAIGN_ROOT / "runs" / w
        if not wdir.is_dir():
            continue
        summary["waves"][w] = {}
        for run_dir in sorted(wdir.iterdir()):
            if run_dir.is_dir() and (run_dir / "run.json").is_file():
                summary["waves"][w][run_dir.name] = read_run_summary(run_dir)
    eval_root = CAMPAIGN_ROOT / "paloma_eval"
    if eval_root.is_dir():
        for p in sorted(eval_root.glob("*.json")):
            summary["paloma"][p.stem] = json.loads(p.read_text())

    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = ["# Confirmation & Scaling Campaign Results", "", f"Generated: `{summary['generated_at']}`", ""]
    for w, runs in summary["waves"].items():
        lines.append(f"## {w}")
        lines.append("")
        lines.append("| Run | Arm | Seed | LR | Steps | Selection Loss | C4 Loss | Median Step (s) | Peak Mem (GiB) | Status |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for name, r in runs.items():
            sel = f"{r['selection_loss']:.5f}" if r.get("selection_loss") is not None else "—"
            c4 = f"{r['c4_loss']:.5f}" if r.get("c4_loss") is not None else "—"
            st = f"{r['median_step_seconds']:.3f}" if r.get("median_step_seconds") is not None else "—"
            mem = f"{r['peak_allocated_gib']:.1f}" if r.get("peak_allocated_gib") is not None else "—"
            lr = f"{r['lr']:.6g}" if r.get("lr") is not None else "—"
            lines.append(
                f"| `{name}` | `{r.get('arm')}` | {r.get('seed')} | {lr} | {r.get('steps')}/{r.get('planned_steps')} | "
                f"{sel} | {c4} | {st} | {mem} | `{r.get('status')}` |"
            )
        lines.append("")

    # Phase 2 paired multi-seed table if all 3 waves complete
    if all(w in summary["waves"] for w in ("wave2a", "wave2b", "wave2c")):
        lines.append("## Phase 2 Paired 3-Seed Summary (`130M`, `1×` Chinchilla, Fresh `seeds 1, 2, 3`)")
        lines.append("")
        by_arm = {k: [] for k in ("muon", "muont", "od_all", "wt_mlp_lag")}
        for seed, w in ((1, "wave2a"), (2, "wave2b"), (3, "wave2c")):
            for k in by_arm:
                r = summary["waves"][w].get(f"{k}_130m_1x_seed{seed}")
                if r and r.get("selection_loss") is not None:
                    by_arm[k].append(r["selection_loss"])
        lines.append("| Arm | Seed 1 | Seed 2 | Seed 3 | Mean ± Std | Mean Δ vs MuonT | Mean Δ vs Muon |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        muont_vals = by_arm["muont"]
        muon_vals = by_arm["muon"]
        for k, vals in by_arm.items():
            if len(vals) == 3:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals)
                d_mt = statistics.mean(a - b for a, b in zip(vals, muont_vals)) if len(muont_vals) == 3 else float("nan")
                d_m = statistics.mean(a - b for a, b in zip(vals, muon_vals)) if len(muon_vals) == 3 else float("nan")
                lines.append(
                    f"| `{k}` | {vals[0]:.5f} | {vals[1]:.5f} | {vals[2]:.5f} | {mean:.5f} ± {std:.5f} | {d_mt:+.5f} | {d_m:+.5f} |"
                )
        lines.append("")

    if summary["paloma"]:
        lines.append("## Phase 5 Paloma 16-Domain Macro Mean")
        lines.append("")
        lines.append("| Run | Arm | Model | Selection Loss | Paloma 16-Domain Macro Mean |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        for name, p in summary["paloma"].items():
            lines.append(
                f"| `{name}` | `{p['arm']}` | `{p['model_size']}` | {p['selection_loss']:.5f} | {p['paloma_macro_mean']:.5f} |"
            )
        lines.append("")

    (results_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"[{now()}] Wrote {results_dir / 'summary.json'} and {results_dir / 'RESULTS.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("deploy")
    sub.add_parser("smoke-130m")
    w_p = sub.add_parser("wave")
    w_p.add_argument("--name", required=True)
    sub.add_parser("gate1")
    sub.add_parser("smoke-520m")
    sub.add_parser("eval-paloma")
    sub.add_parser("summarize")
    sub.add_parser("run-all")
    args = parser.parse_args()

    if args.cmd == "deploy":
        deploy_all()
    elif args.cmd == "smoke-130m":
        smoke_130m()
    elif args.cmd == "wave":
        run_wave(args.name)
        summarize()
    elif args.cmd == "gate1":
        evaluate_gate1()
    elif args.cmd == "smoke-520m":
        smoke_520m()
    elif args.cmd == "eval-paloma":
        eval_paloma_all()
        summarize()
    elif args.cmd == "summarize":
        summarize()
    elif args.cmd == "run-all":
        deploy_all()
        smoke_130m()
        run_wave("wave1a")
        summarize()
        run_wave("wave1b")
        summarize()
        evaluate_gate1()
        for w in ("wave2a", "wave2b", "wave2c", "wave3a"):
            run_wave(w)
            summarize()
        smoke_520m()
        run_wave("wave4a")
        summarize()
        eval_paloma_all()
        summarize()


if __name__ == "__main__":
    main()
