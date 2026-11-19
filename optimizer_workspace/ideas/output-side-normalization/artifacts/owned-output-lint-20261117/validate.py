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
from .execution_owner import require_gpu_owner
from .launch import BLOCK_CHECKS, ENVIRONMENT, OUTPUT_ROOT, WAVE, config_path, run_name
from .records import atomic_json, compare_continuation, sha256, utc_now
from .reference import TASK, UPSTREAM
from .upstream_validation import ARMS, SYNTHETIC_GATES, require_upstream_capture_history, validation_commands


def require_cached_root_checkpoint(directory, world):
    """Inspect each rank's saved numerical state and graph-use evidence before resume."""
    groups = 0
    for rank in range(world):
        state = torch.load(directory / f"rank_{rank}.pt", map_location="cpu", weights_only=False)["fast"]
        engines = (state["input"], state["output"]) if "input" in state else (state,)
        for engine in engines:
            if engine is None:
                continue
            if engine["refresh_install_age"] != 1:
                raise ValueError(f"Rank {rank} checkpoint has a different refresh-installation policy")
            if engine["steps"] < 22:
                raise ValueError(f"Rank {rank} checkpoint precedes the cached-root transition")
            for saved in engine["groups"].values():
                if saved["C"].shape[0] == 0:
                    continue
                if not saved["cached_root_graph_active"] or saved["S2"] is None:
                    raise ValueError(f"Rank {rank} did not exercise cached roots before its checkpoint")
                groups += 1
    if groups == 0:
        raise ValueError("No owned cached-root groups were exercised")
    return {"checkpoint": str(directory), "ranks": world, "owned_graph_groups": groups}


def require_fixed_age_history(directory):
    """Check the actual reference and block installation boundaries on the logged rank."""
    rows = [
        json.loads(line)
        for path in sorted(directory.glob("metrics_from_*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    if [row["step"] for row in rows] != list(range(1, 25)):
        raise ValueError("Fixed-age scheduling evidence requires all 24 consecutive updates")
    installations = 0
    for row in rows:
        for record in row["input_refresh_installations"]:
            if (
                record["installation_step"] != row["step"] - 1
                or record["installation_step"] - record["source_step"] != 1
            ):
                raise ValueError("Actual host-refresh installation did not use fixed update age 1")
            installations += 1
    if not installations:
        raise ValueError("Actual full-model history did not exercise host-refresh installation")
    return {
        "rank": 0,
        "updates": 24,
        "installation_age": 1,
        "observed_installations": installations,
    }


def require_outer_capture_history(directory):
    """Require exact bounded history coverage and both actual-capture gates."""
    rows = [
        json.loads(line)
        for path in sorted(directory.glob("metrics_from_*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    if [row["step"] for row in rows] != list(range(1, 25)):
        raise ValueError("Outer capture parity requires all 24 consecutive updates")
    for row in rows:
        if (
            row.get("diagonal_capture_parity_passed") is not True
            or row.get("outer_diagonal_update_parity_passed") is not True
        ):
            raise ValueError("Actual outer-only update parity is missing")
    return {"updates": 24, "captured_rows_and_outer_updates": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-receipt", type=Path)
    parser.add_argument("--work", type=Path, default=OUTPUT_ROOT / "validation")
    args = parser.parse_args()
    owner = require_gpu_owner() if args.gpu else None
    verify_snapshot()
    data_receipt = args.data_receipt or OUTPUT_ROOT / "data/experiment_receipt.json"
    if not data_receipt.is_file():
        raise FileNotFoundError("Initial candidate receipt requires the prepared-data identity")
    config_hashes = {run_name(arm, seed, batch): sha256(config_path(arm, seed, batch)) for arm, seed, batch in WAVE}
    if WAVE != tuple((arm, 0, "0p5m") for arm in ARMS):
        raise ValueError("Upstream validation requires the exact four full-budget configurations")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-p",
            "no:cacheprovider",
            "experiment/tests",
            "-q",
            "-m",
            "not slow and not integration and not data_integration and not cluster "
            "and not requires_cluster and not docker and not manual",
        ],
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
        "owner": owner,
        "environment": ENVIRONMENT,
        "candidate_path": str(TASK.resolve()),
        "source_sha256": {
            str(path.relative_to(TASK)): sha256(path) for path in sorted((TASK / "experiment").rglob("*.py"))
        },
        "cluster_sha256": sha256(TASK / "experiment/cluster.json"),
        "upstream_manifest_sha256": sha256(TASK / "experiment/upstream_manifest.json"),
        "config_sha256": config_hashes,
        "config_paths": {run_name(arm, seed, batch): str(config_path(arm, seed, batch)) for arm, seed, batch in WAVE},
        "data_receipt": str(data_receipt.resolve()),
        "data_receipt_sha256": sha256(data_receipt),
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
            "real_data_cached_root_continuation": False,
            "real_data_fused_diagonal_capture": False,
            "real_data_outer_diagonal_update_parity": False,
            "real_data_upstream_product_parity": False,
            **dict.fromkeys(BLOCK_CHECKS, False),
        },
        "evidence": {
            "refresh_install_age": 1,
            "cpu_test_command": result.args,
            "cpu_test_exit_code": result.returncode,
            "cpu_test_stdout": result.stdout,
            "cpu_test_stderr": result.stderr,
            "model_parameter_count": count,
            "validation_command": [
                sys.executable,
                "-m",
                "experiment.validate",
                *sys.argv[1:],
            ],
            "pending_gpu_cli": {
                "synthetic": [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    "-m",
                    "experiment.gpu_checks",
                    "--receipt",
                    str(args.work / "gpu_checks.json"),
                ],
                "real_data": {
                    "module": "experiment.train",
                    "ranks": 8,
                    "configuration_names": list(config_hashes),
                    "continuous_max_steps": 24,
                    "interrupted_max_steps": 22,
                    "resumed_max_steps": 24,
                    "execution_boundary": 22,
                    "fused_flags": [
                        "--verify-diagonal-capture",
                        "--verify-outer-diagonal-updates",
                    ],
                    "full_reference_lr_schedule": True,
                },
            },
            "outer_gate_scope": {
                "synthetic_check": "outer_diagonal_pipeline",
                "real_data_check": "real_data_outer_diagonal_update_parity",
                "arm": "outer_leftdiag_fused",
                "ns_source_and_output": "exact",
                "momentum": "exact",
                "update_tolerance": {"rtol": 0.003, "atol": 5e-5},
                "statistic_and_root_tolerance": {"rtol": 1e-5, "atol": 1e-6},
                "continuation_tolerance": {"rtol": 1e-5, "atol": 1e-6},
                "captured_updates_per_trajectory": 24,
                "comparison": "Torch outer factors within the fused trajectory, not independent training trajectories",
                "resume_shadow_initialization": "Restored production moments; independent subsequent increments",
                "original_inner_fused_test": "Preserved but not executed or approved by outer-only gates",
                "timing": "Execution-only verifier overhead is not throughput evidence",
            },
            "upstream_gate_scope": {
                "arms": list(ARMS[2:]),
                "exact": ["momentum", "ns_source", "ns_output", "null_delta"],
                "output_equation_tolerance": {"rtol": 1e-5, "atol": 1e-6},
                "changing_scalar_gate": "gpu_upstream_scalar_step",
                "coverage": "Every owned shape, every update, both actual trajectories",
            },
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
                command,
                cwd=TASK,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, **ENVIRONMENT},
            ).returncode
        record = {
            "command": command,
            "log": str(log),
            "exit_code": status,
            "log_sha256": sha256(log),
        }
        receipt["evidence"]["gpu_commands"].append(record)
        atomic_json(args.receipt, receipt)
        if status:
            raise RuntimeError(f"Validation failed: inspect {log}")

    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    gpu_receipt = args.work / "gpu_checks.json"
    execute(
        [
            *launcher,
            "--nproc_per_node=2",
            "-m",
            "experiment.gpu_checks",
            "--receipt",
            str(gpu_receipt),
        ],
        args.work / "gpu_checks.log",
    )
    checked = json.loads(gpu_receipt.read_text())
    if checked["world"] != 2 or checked["source_sha256"] != receipt["source_sha256"]:
        raise ValueError("Synthetic GPU source or rank coverage differs")
    for name in SYNTHETIC_GATES:
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
            "--execution-boundary",
            "22",
        ]
        if arm.endswith("_fused"):
            common.append("--verify-diagonal-capture")
        if arm.endswith("_fused"):
            common.append("--verify-outer-diagonal-updates")
        execute(
            [*common, "--out", str(full), "--max-steps", "24"],
            args.work / f"{name}_continuous.log",
        )
        execute(
            [*common, "--out", str(interrupted), "--max-steps", "22"],
            args.work / f"{name}_interrupted.log",
        )
        cached = require_cached_root_checkpoint(interrupted / "checkpoints/step_000022", 8)
        boundary_comparison = compare_continuation(
            full / "checkpoints/step_000022", interrupted / "checkpoints/step_000022", 8
        )
        execute(
            [
                *common,
                "--out",
                str(interrupted),
                "--max-steps",
                "24",
                "--resume",
                str(interrupted / "checkpoints/step_000022"),
            ],
            args.work / f"{name}_resumed.log",
        )
        comparison = compare_continuation(full / "checkpoints/step_000024", interrupted / "checkpoints/step_000024", 8)
        scheduling = {
            "continuous": require_fixed_age_history(full),
            "resumed": require_fixed_age_history(interrupted),
        }
        status = json.loads((full / "run.json").read_text())
        if status["status"] != "execution_check_complete" or status["completed_steps"] != 24:
            raise ValueError(f"Full-model smoke did not complete for {name}")
        if arm.endswith("_fused"):
            for directory in (full, interrupted):
                rows = [
                    json.loads(line)
                    for path in directory.glob("metrics_from_*.jsonl")
                    for line in path.read_text().splitlines()
                ]
                if len(rows) != 24 or not all(row["diagonal_capture_parity_passed"] for row in rows):
                    raise ValueError(f"Full-model captured diagonal parity was not established for {name}")
                if arm == "outer_leftdiag_fused":
                    require_outer_capture_history(directory)
                else:
                    require_upstream_capture_history(directory, arm)
        smoke_records[name] = {
            "out": str(full),
            "continuation": comparison,
            "boundary_continuation": boundary_comparison,
            "cached_root_checkpoint": cached,
            "fixed_age_history": scheduling,
        }
    if [item["command"] for item in receipt["evidence"]["gpu_commands"]] != validation_commands(
        sys.executable, TASK, OUTPUT_ROOT, args.work
    ):
        raise ValueError("Executed validation commands differ from the canonical plan")
    receipt["evidence"]["real_data_smokes"] = smoke_records
    receipt["checks"].update(
        compiled_model=True,
        real_data_four_arm_smoke=True,
        checkpoint_continuation=True,
        real_data_cached_root_continuation=True,
        real_data_fused_diagonal_capture=True,
        real_data_outer_diagonal_update_parity=True,
        real_data_upstream_product_parity=True,
    )
    receipt["time"] = utc_now()
    receipt["limitation"] = "Execution gates passed; short smoke losses are not optimizer quality measurements."
    receipt["evidence"]["hostname"] = os.uname().nodename
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt["checks"], indent=2))


if __name__ == "__main__":
    main()
