# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run end-to-end CPU optimizer/checkpoint test, deploy to all 4 nodes, and launch queued MuonH orchestrator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
PVC_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
WD_CAMPAIGN = PVC_ROOT / "campaigns" / "wd-ablation-20261127"
MUONH_CAMPAIGN = PVC_ROOT / "campaigns" / "muonh-20261127"
DATA_RECEIPT = PVC_ROOT / "data" / "experiment_receipt.json"
PYTHON = "/opt/conda/envs/torch-base/bin/python"
WD_PID = 3851349

E2E_CPU_SNIPPET = """
import copy, json, tempfile
from pathlib import Path
import torch
from experiment.capture import build_model
from experiment.optimizer import AdamH, build
from experiment.records import checkpoint_identity, compare_continuation, restore_checkpoint, save_checkpoint
from reproduce.marin.model import initialize_model

task = Path.cwd()
receipt = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/data/experiment_receipt.json")

for arm_name in ("muonh", "muonh_adamh"):
    cfg = copy.deepcopy(json.loads((task / "configs" / "seed0" / f"{arm_name}_130m_1x_seed0_lr0p032.json").read_text()))
    cfg["model"]["layers"] = 2
    cfg["model"]["hidden"] = 64
    cfg["model"]["mlp"] = 256
    cfg["model"]["heads"] = 4
    cfg["model"]["vocab_size"] = 512
    cfg["training"]["world_size"] = 1
    cfg["optimizer"]["ns_mode"] = "f32"

    def make_env():
        torch.manual_seed(0)
        m = build_model(cfg, positions=None, row_capacity=0, rows_per_microbatch=0, device="cpu")
        initialize_model(m, 0, cfg["model"]["embedding_init_scale"])
        opts, fast, clip = build(
            m, cfg, None, positions=None, rows_per_microbatch=0, capacity=0, rows_total=1, world=1, rank=0, device="cpu"
        )
        return m, opts, fast

    m1, opts1, fast1 = make_env()
    init_norms = {n: p.detach().norm().item() for n, p in m1.named_parameters() if p.ndim == 2}
    ident = checkpoint_identity(cfg, receipt)
    gen = torch.Generator().manual_seed(123)
    grad_steps = [
        {n: torch.randn(p.shape, generator=gen) * 0.01 for n, p in m1.named_parameters()}
        for _ in range(4)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        u_out = tmp_path / "unint"
        r_out = tmp_path / "res"

        # Uninterrupted 4 steps
        for step_idx in range(4):
            for opt in opts1:
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * (1.0 - 0.1 * step_idx)
            for n, p in m1.named_parameters():
                p.grad = grad_steps[step_idx][n].clone()
            for opt in opts1:
                opt.step()
        save_checkpoint(u_out, m1, opts1, fast1, None, step=4, config=cfg, rank=0, world=1, identity=ident)

        # Check Frobenius sphere conservation across every block matrix (and head when adamh)
        for n, p in m1.named_parameters():
            if n.startswith("body.blocks.") and p.ndim == 2:
                rel = abs(p.detach().norm().item() - init_norms[n]) / init_norms[n]
                assert rel < 1e-5, f"{arm_name} {n}: drift={rel}"
            if arm_name == "muonh_adamh" and n == "head.weight":
                rel = abs(p.detach().norm().item() - init_norms[n]) / init_norms[n]
                assert rel < 1e-5, f"{arm_name} head.weight: drift={rel}"

        # 2 steps + save + restore + 2 steps
        m2, opts2, fast2 = make_env()
        for step_idx in range(2):
            for opt in opts2:
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * (1.0 - 0.1 * step_idx)
            for n, p in m2.named_parameters():
                p.grad = grad_steps[step_idx][n].clone()
            for opt in opts2:
                opt.step()
        save_checkpoint(r_out, m2, opts2, fast2, None, step=2, config=cfg, rank=0, world=1, identity=ident)

        m3, opts3, fast3 = make_env()
        st = restore_checkpoint(r_out / "checkpoints" / "step_000002", m3, opts3, fast3, None, config=cfg, rank=0, world=1, identity=ident)
        assert st == 2
        for step_idx in range(2, 4):
            for opt in opts3:
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * (1.0 - 0.1 * step_idx)
            for n, p in m3.named_parameters():
                p.grad = grad_steps[step_idx][n].clone()
            for opt in opts3:
                opt.step()
        save_checkpoint(r_out, m3, opts3, fast3, None, step=4, config=cfg, rank=0, world=1, identity=ident)

        comp = compare_continuation(u_out / "checkpoints" / "step_000004", r_out / "checkpoints" / "step_000004", world=1)
        assert comp["max_absolute_error"] == 0.0, f"{arm_name} continuation mismatch: {comp}"

print("End-to-end CPU model + BlockToner(hyperball) + AdamH + checkpoint save/restore PASSED (0.0 max error)!")
"""


def main() -> None:
    print("1. Running end-to-end CPU model + optimizer + checkpoint roundtrip check...", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    toner_root = TASK / "artifacts" / "external-sources" / "Toner-dab7f3777c75e740536df822f50ffa1d929a340c"
    env["PYTHONPATH"] = f"{TASK}:{toner_root / 'src'}:{toner_root}"
    subprocess.run([PYTHON, "-c", E2E_CPU_SNIPPET], cwd=TASK, env=env, check=True)

    print("2. Deploying output-side-normalization-muonh workspace to all 4 A100 nodes...", flush=True)
    subprocess.run([PYTHON, str(TASK / "campaign" / "deploy_muonh_campaign.py")], cwd=TASK, check=True)

    MUONH_CAMPAIGN.mkdir(parents=True, exist_ok=True)
    pid_file = MUONH_CAMPAIGN / "orchestrator.pid"
    if pid_file.is_file():
        existing_pid = int(pid_file.read_text().strip())
        if Path(f"/proc/{existing_pid}").exists():
            print(f"MuonH orchestrator already running (PID {existing_pid}); skipping duplicate launch.")
            return

    print(f"3. Launching detached muonh-20261127 orchestrator queued behind {WD_CAMPAIGN} (PID {WD_PID})...", flush=True)
    stdout_log = MUONH_CAMPAIGN / "launcher_stdout.log"
    with stdout_log.open("a") as stream:
        proc = subprocess.Popen(
            [
                PYTHON,
                str(TASK / "campaign" / "run_muonh_campaign.py"),
                "--wait-for-campaign",
                str(WD_CAMPAIGN),
                "--wait-for-pid",
                str(WD_PID),
            ],
            cwd=TASK,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(f"{proc.pid}\n")
    print(f"Launched detached MuonH orchestrator PID {proc.pid}", flush=True)

    prog_path = MUONH_CAMPAIGN / "progress.json"
    for _ in range(20):
        if prog_path.is_file():
            break
        time.sleep(0.5)

    assert Path(f"/proc/{proc.pid}").exists(), f"MuonH orchestrator PID {proc.pid} exited unexpectedly!"
    assert Path(f"/proc/{WD_PID}").exists(), f"Upstream wd-ablation PID {WD_PID} not found!"
    print("4. Live queue status from muonh-20261127/progress.json:")
    print(prog_path.read_text())


if __name__ == "__main__":
    main()
