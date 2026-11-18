# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute and reconstruct twelve bounded continuous/resumed profiler validation commands."""

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from launch_wave5 import read, sha256
from selected_lr_stage_profile_contract import ENVIRONMENT, MODES, PREPARATION, locations
from selected_lr_stage_profile_validation import plan, seed_resume_prefix, trial_evidence
from watch_wave5 import write

PASS_STATUS = "passed_selected_lr_stage_profile_continuation_and_trace_gates"


def execution_owner(candidate: Path):
    sys.path.insert(0, str(candidate))
    module = importlib.import_module("experiment.selected_lr_stage_profile_owner")
    if Path(module.__file__).resolve() != candidate / "experiment/selected_lr_stage_profile_owner.py":
        raise RuntimeError("Stage-profile validator imported a different candidate owner")
    return module


def reconstruct(root: Path, candidate: Path, receipt_path: Path) -> dict:
    actual = read(receipt_path)
    prepared = read(candidate / PREPARATION)
    expected = plan(candidate, root, locations(root)[3])
    if (
        actual["status"] != PASS_STATUS
        or "error" in actual
        or any(
            actual[key] != value for key, value in expected.items() if key not in ("status", "trials", "gpu_commands")
        )
        or len(actual["trials"]) != len(expected["trials"])
        or len(actual["gpu_commands"]) != len(expected["trials"]) * len(MODES)
    ):
        raise ValueError("Stage-profile GPU receipt is incomplete or has a different bounded plan")
    for item, measured in zip(expected["trials"], actual["trials"], strict=True):
        if measured != item | {"evidence": trial_evidence(item, candidate, prepared)}:
            raise ValueError("Stage-profile original trajectory or rank-trace reconstruction differs")
    executions = iter(actual["gpu_commands"])
    for item in expected["trials"]:
        for mode, command in zip(MODES, item["commands"], strict=True):
            executed = next(executions)
            log = locations(root)[3] / f"{item['arm']}_{mode}.log"
            if (
                executed["arm"] != item["arm"]
                or executed["mode"] != mode
                or executed["command"] != command
                or executed["log"] != str(log)
                or executed["log_sha256"] != sha256(log)
                or type(executed["exit_code"]) is not int
                or executed["exit_code"] != 0
            ):
                raise ValueError("Stage-profile original command, exit or log identity differs")
    return actual


def run(root: Path, candidate: Path, work: Path, receipt_path: Path) -> dict:
    expected = plan(candidate, root, work)
    if receipt_path != locations(root)[4] or work.exists() or receipt_path.exists():
        raise FileExistsError("Stage-profile validation requires unused canonical execution evidence")
    owner = execution_owner(candidate)
    expected["owner"] = owner.require_gpu_owner(root)
    if any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()):
        raise RuntimeError("Stage-profile validator CUDA environment differs")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Stage-profile validator refused occupied GPUs")
    prepared = read(candidate / PREPARATION)
    work.mkdir(parents=True)
    expected["status"] = "validating"
    write(receipt_path, expected)
    try:
        for item in expected["trials"]:
            for mode, command in zip(MODES, item["commands"], strict=True):
                if mode != "continuous":
                    seed_resume_prefix(
                        Path(item["outputs"]["continuous"]),
                        Path(item["outputs"][mode]),
                        read(Path(item["config"])),
                        candidate,
                        prepared,
                    )
                state_path = locations(root)[2]
                state = read(state_path)
                state["active_command"] = command
                write(state_path, state)
                log = work / f"{item['arm']}_{mode}.log"
                with log.open("x") as stream:
                    try:
                        code = subprocess.run(
                            command,
                            cwd=candidate,
                            env=os.environ | ENVIRONMENT,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            timeout=2400,
                        ).returncode
                    except subprocess.TimeoutExpired:
                        code = 124
                expected["gpu_commands"].append(
                    {
                        "arm": item["arm"],
                        "mode": mode,
                        "command": command,
                        "exit_code": code,
                        "log": str(log),
                        "log_sha256": sha256(log),
                    }
                )
                write(receipt_path, expected)
                if code:
                    raise RuntimeError(f"Stage-profile command failed; preserve {log}")
            item["evidence"] = trial_evidence(item, candidate, prepared)
            write(receipt_path, expected)
        expected["status"] = PASS_STATUS
        write(receipt_path, expected)
        return reconstruct(root, candidate, receipt_path)
    except BaseException as error:
        expected.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        write(receipt_path, expected)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--reconstruct", action="store_true")
    args = parser.parse_args()
    result = (
        reconstruct(args.root.resolve(), args.candidate.resolve(), args.receipt.resolve())
        if args.reconstruct
        else run(args.root.resolve(), args.candidate.resolve(), args.work.resolve(), args.receipt.resolve())
    )
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
