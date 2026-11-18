# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record executed checks without claiming pending accelerator gates passed."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from reproduce.marin.model import Model

from .bootstrap import verify_snapshot
from .data_access import require_receipt
from .launch import BLOCK_CHECKS, ENVIRONMENT, OUTPUT_ROOT, WAVE, config_path, run_name
from .records import atomic_json, compare_continuation, sha256, utc_now
from .reference import TASK, UPSTREAM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-receipt", type=Path)
    parser.add_argument("--work", type=Path, default=OUTPUT_ROOT / "validation")
    args = parser.parse_args()
    verify_snapshot()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-c", "/dev/null", "experiment/tests", "-q"],
        cwd=TASK,
        text=True,
        capture_output=True,
    )
    configuration = json.loads((UPSTREAM / "reproduce/marin/configs/130m/muont_0p5m.json").read_text())
    with torch.device("meta"):
        model = Model(configuration["model"])
    count = sum(parameter.numel() for parameter in model.parameters())
    receipt = {
        "time": utc_now(),
        "environment": ENVIRONMENT,
        "source_sha256": {
            str(path.relative_to(TASK)): sha256(path) for path in sorted((TASK / "experiment").rglob("*.py"))
        },
        "checks": {
            "cpu_numerics": result.returncode == 0,
            "cpu_distributed_statistics": result.returncode == 0,
            "model_parameter_count": count == 265585152,
            "gpu_collector": False,
            "distributed_statistics": False,
            "fa2": False,
            "compiled_model": False,
            "real_data_four_arm_smoke": False,
            "checkpoint_continuation": False,
            "gpu_down_updates": False,
            "cached_root_continuation": False,
            "microbatch_partition": False,
            **dict.fromkeys(BLOCK_CHECKS, False),
        },
        "evidence": {
            "cpu_test_command": result.args,
            "cpu_test_exit_code": result.returncode,
            "cpu_test_stdout": result.stdout,
            "cpu_test_stderr": result.stderr,
            "model_parameter_count": count,
        },
        "limitation": "CPU checks do not validate A100 kernels, NCCL, full-model compilation or real-data training.",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.receipt, receipt)
    if result.returncode:
        raise SystemExit(result.stdout + result.stderr)
    if not args.gpu:
        print(json.dumps(receipt["checks"], indent=2))
        return
    if args.data is None or args.data_receipt is None:
        parser.error("--gpu requires exact --data and --data-receipt")
    require_receipt(args.data, args.data_receipt)
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if occupied:
        raise RuntimeError("GPU validation refused while existing CUDA workloads occupy this node")
    args.work.mkdir(parents=True, exist_ok=False)
    receipt["data_receipt_sha256"] = sha256(args.data_receipt)
    receipt["config_sha256"] = {
        run_name(arm, seed, batch): sha256(config_path(arm, seed, batch)) for arm, seed, batch in WAVE
    }
    receipt["evidence"]["gpu_commands"] = []

    def execute(command, log):
        with log.open("x") as stream:
            status = subprocess.run(
                command, cwd=TASK, stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, **ENVIRONMENT}
            ).returncode
        record = {"command": command, "log": str(log), "exit_code": status, "log_sha256": sha256(log)}
        receipt["evidence"]["gpu_commands"].append(record)
        atomic_json(args.receipt, receipt)
        if status:
            raise RuntimeError(f"Validation failed: inspect {log}")

    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    gpu_receipt = args.work / "gpu_checks.json"
    execute(
        [*launcher, "--nproc_per_node=2", "-m", "experiment.gpu_checks", "--receipt", str(gpu_receipt)],
        args.work / "gpu_checks.log",
    )
    checked = json.loads(gpu_receipt.read_text())
    for name in (
        "gpu_collector",
        "distributed_statistics",
        "fa2",
        "gpu_down_updates",
        "cached_root_continuation",
        "microbatch_partition",
        *BLOCK_CHECKS,
    ):
        receipt["checks"][name] = checked["checks"][name]
    receipt["evidence"]["gpu_receipt"] = str(gpu_receipt)
    smoke_records = {}
    for arm, seed, batch in WAVE:
        config = config_path(arm, seed, batch)
        name = run_name(arm, seed, batch)
        full = args.work / f"{name}_continuous"
        interrupted = args.work / f"{name}_resumed"
        common = [
            *launcher,
            "--nproc_per_node=8",
            "-m",
            "experiment.train",
            "--config",
            str(config),
            "--data",
            str(args.data),
            "--data-receipt",
            str(args.data_receipt),
        ]
        execute([*common, "--out", str(full), "--max-steps", "3"], args.work / f"{name}_continuous.log")
        execute(
            [*common, "--out", str(interrupted), "--max-steps", "2"],
            args.work / f"{name}_interrupted.log",
        )
        execute(
            [
                *common,
                "--out",
                str(interrupted),
                "--max-steps",
                "3",
                "--resume",
                str(interrupted / "checkpoints/step_000002"),
            ],
            args.work / f"{name}_resumed.log",
        )
        comparison = compare_continuation(full / "checkpoints/step_000003", interrupted / "checkpoints/step_000003", 8)
        status = json.loads((full / "run.json").read_text())
        if status["status"] != "execution_check_complete" or status["completed_steps"] != 3:
            raise ValueError(f"Full-model smoke did not complete for {name}")
        smoke_records[name] = {"out": str(full), "continuation": comparison}
    receipt["evidence"]["real_data_smokes"] = smoke_records
    receipt["checks"].update(compiled_model=True, real_data_four_arm_smoke=True, checkpoint_continuation=True)
    receipt["time"] = utc_now()
    receipt["limitation"] = "Execution gates passed; short smoke losses are not optimizer quality measurements."
    receipt["evidence"]["hostname"] = os.uname().nodename
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt["checks"], indent=2))


if __name__ == "__main__":
    main()
