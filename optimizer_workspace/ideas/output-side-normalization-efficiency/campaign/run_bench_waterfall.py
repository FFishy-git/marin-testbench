# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run 8-GPU numerical/continuation gate + 8-arm steady-state step-time waterfall across all 4 A100 nodes."""

from __future__ import annotations

import json
import shlex
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))

from campaign.configure_eff_campaign import build_eff_130m_config
from campaign.deploy_eff_campaign import deploy
from experiment.records import compare_continuation

PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
CAMPAIGN_ROOT = PVC_ROOT / "campaigns" / "efficiency-20261128"
DATA_DIR = PVC_ROOT / "data"
DATA_RECEIPT = DATA_DIR / "experiment_receipt.json"
INIT_CKPT = PVC_ROOT / "runs" / "wave0" / "muon" / "checkpoints" / "step_002500"
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]


def now() -> str:
    return datetime.now(UTC).isoformat()


def run_owned_on_node(node: str, tag: str, train_args: list[str]) -> None:
    alloc_dir = CAMPAIGN_ROOT / "allocations"
    alloc_dir.mkdir(parents=True, exist_ok=True)
    slug = node.replace(".", "_")
    state_path = alloc_dir / f"{tag}_{slug}.state.json"
    log_path = alloc_dir / f"{tag}_{slug}.log"
    remote_cmd = [
        PYTHON,
        str(TASK / "campaign" / "allocation.py"),
        "--orch-gen",
        "99",
        "--state",
        str(state_path),
        "--log",
        str(log_path),
        "--",
        PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "experiment.train",
        *train_args,
    ]
    local_ips = set(subprocess.check_output(["hostname", "-I"], text=True).split())
    if node in local_ips:
        proc = subprocess.run(remote_cmd, cwd=TASK, text=True, capture_output=True)
    else:
        proc = subprocess.run(
            [*SSH, f"root@{node}", f"cd {shlex.quote(str(TASK))} && {shlex.join(remote_cmd)}"],
            text=True,
            capture_output=True,
        )
    if proc.returncode != 0:
        tail = log_path.read_text()[-3000:] if log_path.exists() else ""
        raise RuntimeError(
            f"[{node}] {tag} failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout[-1500:]}\n"
            f"STDERR:\n{proc.stderr[-1500:]}\nLOG TAIL:\n{tail}"
        )


def verify_8gpu_gate(cfg_dir: Path) -> dict:
    """Verify (1) 8-step bitwise parity with archived wt_mlp_lag wave1a and (2) 4+4 resume equivalence on lazy HeadDiag."""
    gate_root = CAMPAIGN_ROOT / "gates"
    # 1. Legacy parity config
    legacy_cfg = build_eff_130m_config(label="gate_legacy", arm="inner_outer_mlpdiag_lag", seed=0, lr=0.016)
    legacy_cfg_path = cfg_dir / "gate_legacy.json"
    legacy_cfg_path.write_text(json.dumps(legacy_cfg, indent=2) + "\n")

    # 2. Lazy HeadDiag 4+4 continuation config (checkpoint_every=4 so step 4 saves)
    lazy_cfg = build_eff_130m_config(
        label="gate_lazy_headdiag",
        arm="headdiag_qk_mlpdiag_lag",
        seed=0,
        lr=0.016,
        lazy={
            "enabled": True,
            "input_every": 3,
            "output_every": 3,
            "warmup_until": 2,
            "warmup_every": 1,
            "capture_microbatches": 1,
            "input_ema_decay": 0.75,
            "output_ema_decay": 0.75,
            "exact_gpu_roots": True,
        },
    )
    lazy_cfg["experiment"]["checkpoint_every"] = 4
    lazy_cfg_path = cfg_dir / "gate_lazy_headdiag.json"
    lazy_cfg_path.write_text(json.dumps(lazy_cfg, indent=2) + "\n")

    deploy()

    out_legacy = gate_root / "legacy_8step"
    out_lazy_full = gate_root / "lazy_uninterrupted_8step"
    out_lazy_res = gate_root / "lazy_resumed_4plus4"

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_leg = pool.submit(
            run_owned_on_node,
            NODES[0],
            "gate_legacy",
            [
                "--config",
                str(legacy_cfg_path),
                "--data",
                str(DATA_DIR),
                "--data-receipt",
                str(DATA_RECEIPT),
                "--out",
                str(out_legacy),
                "--max-steps",
                "8",
            ],
        )
        f_full = pool.submit(
            run_owned_on_node,
            NODES[1],
            "gate_lazy_full",
            [
                "--config",
                str(lazy_cfg_path),
                "--data",
                str(DATA_DIR),
                "--data-receipt",
                str(DATA_RECEIPT),
                "--out",
                str(out_lazy_full),
                "--max-steps",
                "8",
            ],
        )
        f_part1 = pool.submit(
            run_owned_on_node,
            NODES[2],
            "gate_lazy_part1",
            [
                "--config",
                str(lazy_cfg_path),
                "--data",
                str(DATA_DIR),
                "--data-receipt",
                str(DATA_RECEIPT),
                "--out",
                str(out_lazy_res),
                "--max-steps",
                "4",
            ],
        )
        for fut in (f_leg, f_full, f_part1):
            fut.result()

    # Resume 4 -> 8 on NODES[2]
    run_owned_on_node(
        NODES[2],
        "gate_lazy_part2",
        [
            "--config",
            str(lazy_cfg_path),
            "--data",
            str(DATA_DIR),
            "--data-receipt",
            str(DATA_RECEIPT),
            "--out",
            str(out_lazy_res),
            "--resume",
            str(out_lazy_res / "checkpoints" / "step_000004"),
            "--max-steps",
            "8",
        ],
    )

    # Check 8-step loss parity against archived confirmation wave1a
    arch_metrics = PVC_ROOT / "campaigns/confirmation-20261121/runs/wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016/metrics_from_000000.jsonl"
    arch_rows = [json.loads(x) for x in arch_metrics.read_text().splitlines()[:8]]
    new_rows = [json.loads(x) for x in (out_legacy / "metrics_from_000000.jsonl").read_text().splitlines()[:8]]
    max_loss_diff = max(abs(a["train_loss"] - b["train_loss"]) for a, b in zip(arch_rows, new_rows, strict=True))
    if max_loss_diff > 1e-6:
        raise RuntimeError(f"Legacy 8-step parity drift too large: {max_loss_diff}")

    cont = compare_continuation(
        out_lazy_full / "checkpoints" / "step_000008",
        out_lazy_res / "checkpoints" / "step_000008",
        world=8,
    )
    receipt = {"time": now(), "legacy_8step_max_loss_diff": max_loss_diff, "lazy_headdiag_4plus4_continuation": cont}
    (CAMPAIGN_ROOT / "gate_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("GATE PASSED:", json.dumps(receipt, indent=2), flush=True)
    return receipt


def build_bench_specs() -> list[tuple[str, dict]]:
    lazy8 = {
        "enabled": True,
        "input_every": 8,
        "output_every": 8,
        "warmup_until": 16,
        "warmup_every": 2,
        "capture_microbatches": 1,
        "input_ema_decay": 0.75,
        "output_ema_decay": 0.75,
        "exact_gpu_roots": True,
    }
    lazy16 = {**lazy8, "input_every": 16, "output_every": 16, "input_ema_decay": 0.60, "output_ema_decay": 0.60}
    fast8 = {
        "enabled": True,
        "input_every": 8,
        "output_every": 8,
        "warmup_until": 16,
        "warmup_every": 2,
        "capture_microbatches": 1,
        "input_ema_decay": 0.75,
        "output_ema_decay": 0.75,
        "root_backend": "async_host",
        "input_inline_steps": 1,
        "input_refresh_every": 2,
    }
    fullasync8 = {
        "enabled": True,
        "input_every": 8,
        "output_every": 8,
        "warmup_until": 16,
        "warmup_every": 2,
        "capture_microbatches": 1,
        "input_ema_decay": 0.75,
        "output_ema_decay": 0.75,
        "root_backend": "async_host_both",
        "lag_input": True,
    }
    fullasync16 = {
        **fullasync8,
        "input_every": 16,
        "output_every": 16,
        "input_ema_decay": 0.60,
        "output_ema_decay": 0.60,
    }
    return [
        ("bench_muonh", build_eff_130m_config(label="bench_muonh", arm="none_all", lr=0.01131370849898476, decay_mode="hyperball")),
        ("bench_expdiag_orig", build_eff_130m_config(label="bench_expdiag_orig", arm="inner_outer_mlpdiag_lag", lr=0.016)),
        ("bench_expdiag_lazy8", build_eff_130m_config(label="bench_expdiag_lazy8", arm="inner_outer_mlpdiag_lag", lr=0.016, lazy=lazy8)),
        ("bench_expdiag_lazy16", build_eff_130m_config(label="bench_expdiag_lazy16", arm="inner_outer_mlpdiag_lag", lr=0.016, lazy=lazy16)),
        ("bench_headdiag_qk_lazy8", build_eff_130m_config(label="bench_headdiag_qk_lazy8", arm="headdiag_qk_mlpdiag_lag", lr=0.016, lazy=lazy8)),
        ("bench_headdiag_qkv_lazy8", build_eff_130m_config(label="bench_headdiag_qkv_lazy8", arm="headdiag_qkv_mlpdiag_lag", lr=0.016, lazy=lazy8)),
        ("bench_resdense_lazy8", build_eff_130m_config(label="bench_resdense_lazy8", arm="resdense_alldiag_lag", lr=0.016, lazy=lazy8)),
        ("bench_headdiag_qk_lazy8_cns", build_eff_130m_config(label="bench_headdiag_qk_lazy8_cns", arm="headdiag_qk_mlpdiag_lag", lr=0.016, lazy=lazy8, compile_ns=True)),
        # Wave 3: Zero-cuSOLVER async_host on outputs + FastTonerCore on inputs
        ("bench_expdiag_fast8", build_eff_130m_config(label="bench_expdiag_fast8", arm="inner_outer_mlpdiag_lag", lr=0.016, lazy=fast8)),
        ("bench_headdiag_qk_fast8", build_eff_130m_config(label="bench_headdiag_qk_fast8", arm="headdiag_qk_mlpdiag_lag", lr=0.016, lazy=fast8)),
        ("bench_headdiag_qk_fast8_cns", build_eff_130m_config(label="bench_headdiag_qk_fast8_cns", arm="headdiag_qk_mlpdiag_lag", lr=0.016, lazy=fast8, compile_ns=True)),
        ("bench_headdiag_qkv_fast8_cns", build_eff_130m_config(label="bench_headdiag_qkv_fast8_cns", arm="headdiag_qkv_mlpdiag_lag", lr=0.016, lazy=fast8, compile_ns=True)),
        # Wave 4: Full async_host_both + lag_input (0 GPU cuSOLVER on BOTH sides)
        ("bench_expdiag_fullasync8_cns", build_eff_130m_config(label="bench_expdiag_fullasync8_cns", arm="inner_outer_mlpdiag_lag", lr=0.016, lazy=fullasync8, compile_ns=True)),
        ("bench_headdiag_qk_fullasync8_cns", build_eff_130m_config(label="bench_headdiag_qk_fullasync8_cns", arm="headdiag_qk_mlpdiag_lag", lr=0.016, lazy=fullasync8, compile_ns=True)),
        ("bench_headdiag_qkv_fullasync8_cns", build_eff_130m_config(label="bench_headdiag_qkv_fullasync8_cns", arm="headdiag_qkv_mlpdiag_lag", lr=0.016, lazy=fullasync8, compile_ns=True)),
        ("bench_headdiag_qk_fullasync16_cns", build_eff_130m_config(label="bench_headdiag_qk_fullasync16_cns", arm="headdiag_qk_mlpdiag_lag", lr=0.016, lazy=fullasync16, compile_ns=True)),
    ]


def main() -> None:
    cfg_dir = TASK / "configs" / "bench"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)

    specs = build_bench_specs()
    for name, cfg in specs:
        (cfg_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")

    if not (CAMPAIGN_ROOT / "gate_receipt.json").exists():
        verify_8gpu_gate(cfg_dir)
    else:
        deploy()

    bench_root = CAMPAIGN_ROOT / "bench_waterfall"
    for wave_idx in range(0, len(specs), 4):
        chunk = specs[wave_idx : wave_idx + 4]
        print(f"[{now()}] Launching bench wave {wave_idx // 4 + 1}: {[n for n, _ in chunk]}", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {}
            for (name, _), node in zip(chunk, NODES, strict=False):
                out_dir = bench_root / name
                if (out_dir / "run.json").exists() and json.loads((out_dir / "run.json").read_text()).get("status") == "execution_check_complete":
                    continue
                futs[
                    pool.submit(
                        run_owned_on_node,
                        node,
                        name,
                        [
                            "--config",
                            str(cfg_dir / f"{name}.json"),
                            "--data",
                            str(DATA_DIR),
                            "--data-receipt",
                            str(DATA_RECEIPT),
                            "--init-model-from",
                            str(INIT_CKPT),
                            "--bench-midtrain",
                            "--max-steps",
                            "96",
                            "--out",
                            str(out_dir),
                        ],
                    )
                ] = name
            for fut in as_completed(futs):
                fut.result()
                print(f"[{now()}] Finished {futs[fut]}", flush=True)

    # Summarize steps 33..96 (64 steps = exact multiple of K=8 and K=16)
    summary = []
    base_med = base_mean = None
    for name, _ in specs:
        mpath = bench_root / name / "metrics_from_000000.jsonl"
        rows = [json.loads(x) for x in mpath.read_text().splitlines() if x.strip()]
        win = [r["step_seconds"] for r in rows if 33 <= r["step"] <= 96]
        med = statistics.median(win)
        mn = statistics.mean(win)
        peak_gib = max(r["peak_allocated_bytes"] for r in rows) / (1024**3)
        if base_med is None:
            base_med, base_mean = med, mn
        summary.append(
            {
                "name": name,
                "median_s": med,
                "median_delta_pct": (med / base_med - 1.0) * 100.0,
                "mean_s": mn,
                "mean_delta_pct": (mn / base_mean - 1.0) * 100.0,
                "est_5120_min": mn * 5120.0 / 60.0,
                "peak_gib": peak_gib,
            }
        )
    (CAMPAIGN_ROOT / "bench_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (TASK / "results").mkdir(parents=True, exist_ok=True)
    (TASK / "results" / "bench_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== 8xA100 STEADY-STATE WATERFALL (steps 33..96, 64 steps) ===")
    for row in summary:
        print(
            f"  {row['name']:28s} | med={row['median_s']:.4f}s ({row['median_delta_pct']:+5.2f}%) "
            f"| mean={row['mean_s']:.4f}s ({row['mean_delta_pct']:+5.2f}%) "
            f"| est_5120={row['est_5120_min']:6.2f}m | peak={row['peak_gib']:.2f} GiB"
        )


if __name__ == "__main__":
    main()
