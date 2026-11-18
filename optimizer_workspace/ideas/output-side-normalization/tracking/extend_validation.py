# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate revised wave entries while retaining verified, unchanged launch gates."""

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from experiment.bootstrap import verify_snapshot
from experiment.data_access import require_receipt
from experiment.launch import ENVIRONMENT, WAVE, config_path, run_name
from experiment.records import atomic_json, compare_continuation, sha256, utc_now
from experiment.reference import TASK


def extend(
    original: Path, receipt_path: Path, work: Path, data: Path, data_receipt: Path,
    configurations: dict[str, Path],
) -> None:
    verify_snapshot()
    require_receipt(data, data_receipt)
    prior = json.loads(original.read_text())
    if not all(prior["checks"].values()):
        raise ValueError("Original validation is incomplete")
    expected = {str(path.relative_to(TASK)): sha256(path) for path in sorted((TASK / "experiment").rglob("*.py"))}
    if prior["source_sha256"] != expected or prior["environment"] != ENVIRONMENT:
        raise ValueError("Cannot reuse validation for different source or environment")
    if prior["data_receipt_sha256"] != sha256(data_receipt):
        raise ValueError("Cannot reuse validation for different prepared data")
    for command in prior["evidence"]["gpu_commands"]:
        if command["exit_code"] != 0 or sha256(Path(command["log"])) != command["log_sha256"]:
            raise ValueError("Original command evidence failed or changed")
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if occupied:
        raise RuntimeError("Validation extension refused while CUDA workloads occupy this node")
    work.mkdir(parents=True, exist_ok=False)
    result = copy.deepcopy(prior)
    result["reused_receipt"] = {"path": str(original), "sha256": sha256(original)}
    result["checks"].update(compiled_model=False, real_data_four_arm_smoke=False, checkpoint_continuation=False)
    result["config_sha256"] = {name: sha256(config) for name, config in configurations.items()}
    smokes = {}
    for name, config in configurations.items():
        if prior["config_sha256"].get(name) == result["config_sha256"][name]:
            smokes[name] = prior["evidence"]["real_data_smokes"][name]
            continue
        full, interrupted = work / f"{name}_continuous", work / f"{name}_resumed"
        command = [
            sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
            "-m", "experiment.train", "--config", str(config),
            "--data", str(data), "--data-receipt", str(data_receipt),
        ]
        commands = (
            ([*command, "--out", str(full), "--max-steps", "3"], work / f"{name}_continuous.log"),
            ([*command, "--out", str(interrupted), "--max-steps", "2"], work / f"{name}_interrupted.log"),
            (
                [
                    *command, "--out", str(interrupted), "--max-steps", "3",
                    "--resume", str(interrupted / "checkpoints/step_000002"),
                ],
                work / f"{name}_resumed.log",
            ),
        )
        for execution, log in commands:
            with log.open("x") as stream:
                status = subprocess.run(
                    execution, cwd=TASK, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, **ENVIRONMENT},
                ).returncode
            result["evidence"]["gpu_commands"].append({
                "command": execution, "log": str(log), "exit_code": status, "log_sha256": sha256(log),
            })
            atomic_json(receipt_path, result)
            if status:
                raise RuntimeError(f"Validation failed: {log}")
        continuation = compare_continuation(
            full / "checkpoints/step_000003", interrupted / "checkpoints/step_000003", 8
        )
        manifest = json.loads((full / "run.json").read_text())
        if manifest["status"] != "execution_check_complete" or manifest["completed_steps"] != 3:
            raise ValueError(f"{name}: execution check did not complete")
        smokes[name] = {"out": str(full), "continuation": continuation}
    result["evidence"]["real_data_smokes"] = smokes
    result["checks"].update(compiled_model=True, real_data_four_arm_smoke=True, checkpoint_continuation=True)
    result["time"] = utc_now()
    atomic_json(receipt_path, result)
    print(json.dumps({"receipt": str(receipt_path), "checks": result["checks"], "smokes": smokes}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Validate only this exact configuration for a single follow-up")
    args = parser.parse_args()
    configurations = {run_name(arm, seed, batch): config_path(arm, seed, batch) for arm, seed, batch in WAVE}
    if args.config is not None:
        config = json.loads(args.config.read_text())
        training = config["training"]
        batches = {524288: "0p5m", 1048576: "1m", 2097152: "2m", 4194304: "4m"}
        arm, seed, batch = config["experiment"]["arm"], training["seed"], batches[training["batch_tokens"]]
        if args.config.resolve() != config_path(arm, seed, batch).resolve():
            raise ValueError("Single configuration must be the canonical arm/batch/seed file")
        configurations = {run_name(arm, seed, batch): args.config}
    extend(args.original, args.receipt, args.work, args.data, args.data_receipt, configurations)


if __name__ == "__main__":
    main()
