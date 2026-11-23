# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Wave 3B: matched-LR (0.0105561) 4x Chinchilla follow-up for OD-All and WT-MLP-Lag."""

from __future__ import annotations

import json
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))

from campaign.run_campaign import (
    CAMPAIGN_ROOT,
    DATA_DIR,
    DATA_RECEIPT,
    PYTHON,
    deploy_all,
    eval_paloma_all,
    is_complete,
    now,
    read_run_summary,
    run_owned_on_node,
    summarize,
)

# Match wave3a hardware assignment (.144 for OD-All, .139 for WT-MLP-Lag); leave .9 and .80 idle with keepers intact.
WAVE3B_SPECS = (
    (
        "28.216.17.144",
        "od_all_130m_4x_seed0_lr0p01056",
        TASK / "configs" / "phase3_matched_lr" / "od_all_130m_4x_seed0_lr0p01056.json",
    ),
    (
        "28.216.17.139",
        "wt_mlp_lag_130m_4x_seed0_lr0p01056",
        TASK / "configs" / "phase3_matched_lr" / "wt_mlp_lag_130m_4x_seed0_lr0p01056.json",
    ),
)


def main() -> None:
    log_file = CAMPAIGN_ROOT / "orchestrator.log"

    def log(msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with log_file.open("a") as fh:
            fh.write(line + "\n")

    log("=== Starting Wave 3B: matched-LR (0.0105561) 4x Chinchilla follow-up ===")
    deploy_all()

    runs_root = CAMPAIGN_ROOT / "runs" / "wave3b"
    runs_root.mkdir(parents=True, exist_ok=True)

    pending = []
    for node, name, cfg in WAVE3B_SPECS:
        out_dir = runs_root / name
        if is_complete(out_dir):
            s = read_run_summary(out_dir)
            log(f"SKIP {name} (already complete: sel={s['selection_loss']:.5f})")
            continue
        pending.append((node, name, cfg, out_dir))

    def launch_one(node: str, name: str, cfg: Path, out_dir: Path) -> dict:
        resume_args = []
        latest = out_dir / "latest_checkpoint.json"
        if latest.is_file():
            ckpt = json.loads(latest.read_text())["path"]
            resume_args = ["--resume", ckpt]
        elif out_dir.exists():
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
            str(cfg),
            "--data",
            str(DATA_DIR),
            "--data-receipt",
            str(DATA_RECEIPT),
            "--out",
            str(out_dir),
            *resume_args,
        ]
        tag = f"wave3b_{name}_{int(time.time())}"
        res = run_owned_on_node(node, tag, cmd)
        if not is_complete(out_dir):
            raise RuntimeError(f"Run {name} on {node} finished without complete status!")
        return res

    if pending:
        log(f"Launching wave3b ({len(pending)} runs in parallel on .144 and .139)...")
        errors = []
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            futs = {pool.submit(launch_one, n, name, cfg, out): name for n, name, cfg, out in pending}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    fut.result()
                except BaseException as exc:
                    errors.append((name, repr(exc)))
                    log(f"ERROR in {name}: {exc}")
        if errors:
            raise RuntimeError(f"Wave wave3b failed: {errors}")

    log("Wave wave3b COMPLETE:")
    for _, name, _ in WAVE3B_SPECS:
        s = read_run_summary(runs_root / name)
        log(
            f"  {name:38s} sel={s['selection_loss']:.5f}  c4={s['c4_loss']:.5f}  "
            f"step_s={s['median_step_seconds']:.3f}  mem={s['peak_allocated_gib']:.1f}GiB"
        )

    eval_paloma_all()
    summarize()
    log("Uploading newly completed waves (wave4a, wave3b) to SwanLab project output-side-normalization-confirmation...")
    logging_python = "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/.venv-logging/bin/python"
    import subprocess
    subprocess.run([logging_python, "-u", str(TASK / "campaign" / "swanlab_upload_confirmation.py")], check=True)
    log("=== Wave 3B + Paloma evaluation + summary + SwanLab upload COMPLETE ===")


if __name__ == "__main__":
    main()
