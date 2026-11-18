# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute and independently reconstruct four actual-config 24-versus-22+resume gates."""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from analyze_tuning import executed_sources, normalized_sources
from launch_wave5 import read, sha256
from selected_lr_validation import ARMS, BOUNDARY, ENDPOINT, ENVIRONMENT, PREPARATION, WORLD, locations, plan
from validate_fixed_age_tuning import executed_trajectory
from watch_wave5 import write


def numerical_modules(candidate: Path) -> tuple:
    sys.path.insert(0, str(candidate))
    names = ("experiment.validate", "experiment.records", "experiment.selected_lr_owner")
    modules = tuple(importlib.import_module(name) for name in names)
    for module in modules:
        if not Path(module.__file__).resolve().is_relative_to(candidate.resolve()):
            raise ValueError("Selected-LR numerical validation imported a different candidate")
    return modules


def checkpoint_evidence(directory: Path, config: dict, candidate: Path, frozen: dict, step: int) -> dict:
    """Check rank-local metadata and identities independently of pairwise numerical agreement."""
    checkpoint = directory / f"checkpoints/step_{step:06d}"
    receipt = read(checkpoint / "complete.json")
    expected_sources = executed_sources(candidate, frozen)
    hashes, sizes = {}, []
    if type(receipt["step"]) is not int or receipt["step"] != step or receipt["world"] != WORLD:
        raise ValueError("Selected-LR checkpoint completion metadata differs")
    for rank in range(WORLD):
        path = checkpoint / f"rank_{rank}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        identity = state["identity"]
        if (
            state["rank"] != rank
            or state["world"] != WORLD
            or state["step"] != step
            or state["config"] != config
            or state["next_global_sequence"] != step * config["training"]["global_batch_sequences"]
            or identity["config_sha256"] != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            or identity["data_receipt_sha256"] != frozen["data_receipt_sha256"]
            or normalized_sources(candidate, identity["source_sha256"]) != expected_sources
            or (rank == 0 and not state.get("model"))
        ):
            raise ValueError("Selected-LR checkpoint rank, config, source or data position differs")
        hashes[path.name] = sha256(path)
        sizes.append(path.stat().st_size)
    return {"receipt_sha256": sha256(checkpoint / "complete.json"), "rank_bytes": sizes, "rank_sha256": hashes}


def trial_evidence(item: dict, candidate: Path, frozen: dict) -> dict:
    """Recompute numerical checks from original checkpoints and histories at unchanged tolerances."""
    validation, records, _ = numerical_modules(candidate)
    config = read(Path(item["config"]))
    continuous, resumed = Path(item["continuous"]), Path(item["resumed"])
    result = dict(item)
    result["cached_root_checkpoint"] = validation.require_cached_root_checkpoint(
        resumed / f"checkpoints/step_{BOUNDARY:06d}", WORLD
    )
    result["boundary_continuation"] = records.compare_continuation(
        continuous / f"checkpoints/step_{BOUNDARY:06d}", resumed / f"checkpoints/step_{BOUNDARY:06d}", WORLD
    )
    result["post_cached_root_continuation"] = records.compare_continuation(
        continuous / f"checkpoints/step_{ENDPOINT:06d}", resumed / f"checkpoints/step_{ENDPOINT:06d}", WORLD
    )
    result["fixed_age_history"] = {
        name: validation.require_fixed_age_history(Path(item[name])) for name in ("continuous", "resumed")
    }
    result["trajectory_identity"] = {
        name: executed_trajectory(Path(item[name]), config, candidate, frozen) for name in ("continuous", "resumed")
    }
    result["checkpoint_identity"] = {
        name: {
            str(step): checkpoint_evidence(Path(item[name]), config, candidate, frozen, step)
            for step in (BOUNDARY, ENDPOINT)
        }
        for name in ("continuous", "resumed")
    }
    if item["family"] == "outer_leftdiag_fused":
        result["capture_history"] = {
            name: validation.require_outer_capture_history(Path(item[name])) for name in ("continuous", "resumed")
        }
    result["checks"] = dict.fromkeys(item["checks"], True)
    return result


def reconstruct(root: Path, candidate: Path, receipt_path: Path) -> dict:
    """Accept only a complete exact command set and independently reproduced all-rank evidence."""
    frozen = read(candidate / PREPARATION)
    expected = plan(candidate, root, locations(root)[3])
    actual = read(receipt_path)
    if (
        actual["status"] != "passed_actual_full_budget_config_gates"
        or any(
            actual[key] != value for key, value in expected.items() if key not in ("status", "trials", "gpu_commands")
        )
        or len(actual["trials"]) != len(ARMS)
        or len(actual["gpu_commands"]) != 3 * len(ARMS)
        or "error" in actual
    ):
        raise ValueError("Selected-LR actual-config GPU receipt is incomplete or differs")
    for item, executed in zip(expected["trials"], actual["trials"], strict=True):
        if executed != trial_evidence(item, candidate, frozen):
            raise ValueError("Selected-LR checkpoint/history evidence differs from independent reconstruction")
    commands = [
        (item["name"], index, command) for item in expected["trials"] for index, command in enumerate(item["commands"])
    ]
    for execution, (arm, index, command) in zip(actual["gpu_commands"], commands, strict=True):
        log = locations(root)[3] / f"{arm}_{('continuous', 'interrupted', 'resumed')[index]}.log"
        if (
            execution["trial"] != arm
            or execution["command"] != command
            or type(execution["exit_code"]) is not int
            or execution["exit_code"] != 0
            or execution["log"] != str(log)
            or execution["log_sha256"] != sha256(log)
        ):
            raise ValueError("Selected-LR actual-config command or original log differs")
    return actual


def run(root: Path, candidate: Path, work: Path, receipt_path: Path) -> dict:
    result = plan(candidate, root, work)
    if receipt_path.resolve() != locations(root.resolve())[4]:
        raise ValueError("Selected-LR GPU receipt requires its canonical path")
    if work.exists() or receipt_path.exists():
        raise FileExistsError("Selected-LR GPU evidence already exists; preserve it without retry")
    validation, _, owner = numerical_modules(candidate)
    result["owner"] = owner.require_gpu_owner()
    validation.require_receipt(root / "data", root / "data/experiment_receipt.json")
    if any(os.environ.get(name) != value for name, value in ENVIRONMENT.items()):
        raise RuntimeError("Selected-LR GPU validation environment differs")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Selected-LR validation refused occupied GPUs")
    frozen = read(candidate / PREPARATION)
    work.mkdir(parents=True, exist_ok=False)
    result["status"] = "validating"
    write(receipt_path, result)
    try:
        for item in result["trials"]:
            for index, command in enumerate(item["commands"]):
                log = work / f"{item['name']}_{('continuous', 'interrupted', 'resumed')[index]}.log"
                with log.open("x") as stream:
                    try:
                        code = subprocess.run(
                            command,
                            cwd=candidate,
                            env=os.environ | ENVIRONMENT,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            timeout=900,
                        ).returncode
                    except subprocess.TimeoutExpired:
                        code = 124
                result["gpu_commands"].append(
                    {
                        "trial": item["name"],
                        "command": command,
                        "exit_code": code,
                        "log": str(log),
                        "log_sha256": sha256(log),
                    }
                )
                write(receipt_path, result)
                if code:
                    raise RuntimeError(f"Selected-LR actual-config command failed; preserve {log}")
            evidence = trial_evidence(item, candidate, frozen)
            item.update(evidence)
            write(receipt_path, result)
        result["status"] = "passed_actual_full_budget_config_gates"
        write(receipt_path, result)
        return result
    except BaseException as error:
        result.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        write(receipt_path, result)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--reconstruct", action="store_true")
    args = parser.parse_args()
    if args.reconstruct and args.receipt is None:
        parser.error("Reconstruction requires --receipt")
    result = (
        reconstruct(args.root, args.candidate, args.receipt)
        if args.reconstruct
        else (
            run(args.root, args.candidate, args.work, args.receipt)
            if args.receipt
            else plan(args.candidate, args.root, args.work)
        )
    )
    print(json.dumps({"status": result["status"], "trials": len(result["trials"])}))


if __name__ == "__main__":
    main()
