# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain exact per-LR gates under a future distinct owner; never use a lower-LR release."""

import argparse
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import ns_lr_full_model_contract as contract
from fused_attribution_validation import numerical_modules, save_interrupted_manifest
from fused_attribution_validation_controller import acknowledged_process, finish_marked_child
from fused_continuation_validation import checkpoint_pair, loss_pair
from ns_full_model_validation import capture, require_entry, trajectory
from selected_lr_release_owner import candidate_files, read, sha256
from watch_wave5 import write

STATUS = "ns_actual_lr_full_model_gates_reconstructed_not_production"
COMMAND_TIMEOUT = 900


def plan(root: Path, candidate: Path) -> dict:
    work = contract.locations(root)[3]
    commands = contract.commands(candidate, root, work)
    return {
        "root": str(root),
        "candidate": str(candidate),
        "world": contract.WORLD,
        "environment": contract.ENVIRONMENT,
        "trials": [
            spec
            | {
                "config_sha256": sha256(candidate / spec["config_name"]),
                "continuous": str(work / f"{spec['name']}_continuous"),
                "resumed": str(work / f"{spec['name']}_resumed"),
                "commands": commands[index * 3 : (index + 1) * 3],
            }
            for index, spec in enumerate(contract.specs())
        ],
        "commands": commands,
        "checkpoint_steps": list(contract.CHECKPOINTS),
        "tolerance": contract.TOLERANCE,
        "planned_tokens": len(contract.specs()) * 2 * contract.ENDPOINT * contract.BATCH_TOKENS,
        "scope": contract.SCOPE,
    }


def source(root: Path, candidate: Path) -> dict:
    path = contract.locations(root)[0]
    raw, frozen = path.read_bytes(), read(path)
    expected = plan(root, candidate)
    if (
        frozen["status"] != contract.FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != str(candidate)
        or frozen["files_sha256"] != candidate_files(candidate)
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["plan"] != expected
        or frozen["commands"] != expected["commands"]
        or path.read_bytes() != raw
    ):
        raise ValueError("NS actual-LR frozen source/configuration/data/protocol differs")
    return frozen


def trial(item: dict, candidate: Path, frozen: dict) -> dict:
    """Apply accepted readers unchanged to the exact actual LR and retain every rejection."""
    spec = {key: item[key] for key in contract.specs()[0]}
    config = contract.require_config(candidate, spec)
    continuous, resumed = Path(item["continuous"]), Path(item["resumed"])
    trajectories = {
        mode: trajectory(path, config, candidate, frozen, mode)
        for mode, path in (("continuous", continuous), ("interrupted", resumed), ("resumed", resumed))
    }
    comparisons = {str(step): checkpoint_pair(resumed, continuous, step) for step in contract.CHECKPOINTS}
    loss = loss_pair(resumed, continuous)
    captures = (
        {mode: capture(path, spec["family"]) for mode, path in (("continuous", continuous), ("resumed", resumed))}
        if spec["family"] != "muont"
        else {}
    )
    gates = {
        "actual_lr_complete_trajectories": True,
        "boundary_all_rank_continuation": comparisons["22"]["within_tolerance"],
        "post_cached_root_all_rank_continuation": comparisons["24"]["within_tolerance"],
        "scalar_loss_continuation": loss["within_tolerance"],
        "captured_row_update_work": True,
    }
    return item | {
        "trajectories": trajectories,
        "continuation": comparisons,
        "loss": loss,
        "capture": captures,
        "first_resumed_diagnostic_only": True,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def run_command(root: Path, candidate: Path, command: list[str], log: Path) -> dict:
    state_path = contract.locations(root)[2]
    state, marker, generations = read(state_path), uuid4().hex, {}
    state["active_command"] = command
    write(state_path, state)
    entry, child = {"command": command, "marker": marker, "spawn_status": "starting"}, None
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(
                command,
                cwd=candidate,
                env=os.environ | contract.ENVIRONMENT | {contract.COMMAND_MARKER: marker},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            generation = acknowledged_process(
                child, command, candidate, contract.COMMAND_MARKER, marker, contract.ENVIRONMENT
            )
            entry.update(spawn_status="acknowledged", generation=generation)
            entry["exit_code"] = child.wait(timeout=COMMAND_TIMEOUT)
    except BaseException as failure:
        entry["error"] = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        try:
            if child is not None:
                finish_marked_child(child, marker, generations, contract.COMMAND_MARKER)
                entry.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
        except BaseException as failure:
            entry["cleanup_error"] = f"{type(failure).__name__}: {failure}"
            raise
        finally:
            entry.update(log=str(log), log_sha256=sha256(log) if log.is_file() else None, generations=generations)
            write(log.with_suffix(".execution.json"), entry)
    return entry


def execute(root: Path, candidate: Path) -> dict:
    # The old lower-LR adapter cannot supply this distinct owner boundary.
    _, _, owner = numerical_modules(candidate)
    owned = owner.require_ns_actual_lr_gpu_owner(root)
    frozen = source(root, candidate)
    binding, _, _, work, receipt = contract.locations(root)
    if work.exists() or receipt.exists():
        raise FileExistsError("NS actual-LR execution identity is one-shot")
    work.mkdir(parents=True)
    result = plan(root, candidate) | {
        "status": "validating",
        "owner": owned,
        "binding_sha256": sha256(binding),
        "gpu_commands": [],
        "results": [],
    }
    write(receipt, result)
    try:
        for item in result["trials"]:
            config = contract.require_config(candidate, {key: item[key] for key in contract.specs()[0]})
            for mode, command in zip(contract.MODES, item["commands"], strict=True):
                if source(root, candidate) != frozen or owner.require_ns_actual_lr_gpu_owner(root) != owned:
                    raise RuntimeError("NS actual-LR source or original distinct owner changed")
                log = work / f"{item['name']}_{mode}.log"
                entry = run_command(root, candidate, command, log)
                require_entry(entry, command, log)
                directory = Path(item["continuous"] if mode == "continuous" else item["resumed"])
                if mode == "interrupted":
                    save_interrupted_manifest(directory)
                entry.update(
                    trial=item["name"], mode=mode, trajectory=trajectory(directory, config, candidate, frozen, mode)
                )
                result["gpu_commands"].append(entry)
                write(receipt, result)
            result["results"].append(trial(item, candidate, frozen))
            write(receipt, result)
        result.update(
            status=STATUS,
            accepted_cases=[item["name"] for item in result["results"] if item["accepted"]],
            rejected_cases=[item["name"] for item in result["results"] if not item["accepted"]],
            all_cases_accepted=all(item["accepted"] for item in result["results"]),
        )
    except BaseException as failure:
        result.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        raise
    finally:
        write(receipt, result)
    return result


def reconstruct(root: Path, candidate: Path) -> dict:
    """Rederive all per-LR results and ancestry without using lower-LR decisions."""
    frozen, expected = source(root, candidate), plan(root, candidate)
    binding, _, state_path, work, receipt = contract.locations(root)
    raw, result, state = receipt.read_bytes(), read(receipt), read(state_path)
    expected_owner = {
        "controller_pid": state["controller_pid"],
        "controller_start_ticks": state["controller_start_ticks"],
        "validator_pid": state["child_pid"],
        "validator_start_ticks": state["child_start_ticks"],
    }
    if (
        result["status"] != STATUS
        or "error" in result
        or result["binding_sha256"] != sha256(binding)
        or state["binding_sha256"] != sha256(binding)
        or result["owner"] != expected_owner
        or any(result[key] != value for key, value in expected.items())
        or len(result["results"]) != len(expected["trials"])
        or len(result["gpu_commands"]) != len(expected["commands"])
        or result["accepted_cases"] != [item["name"] for item in result["results"] if item["accepted"]]
        or result["rejected_cases"] != [item["name"] for item in result["results"] if not item["accepted"]]
        or type(result["all_cases_accepted"]) is not bool
        or result["all_cases_accepted"] != all(item["accepted"] for item in result["results"])
    ):
        raise ValueError("NS actual-LR retained protocol is incomplete or source/owner/decisions differ")
    rebuilt = []
    for index, item in enumerate(expected["trials"]):
        for offset, (mode, command) in enumerate(zip(contract.MODES, item["commands"], strict=True)):
            entry = result["gpu_commands"][index * 3 + offset]
            require_entry(entry, command, work / f"{item['name']}_{mode}.log")
            if (
                entry["trial"] != item["name"]
                or entry["mode"] != mode
                or entry["generation"]["parent"] != result["owner"]["validator_pid"]
            ):
                raise ValueError("NS actual-LR retained command ancestry differs")
        measured = trial(item, candidate, frozen)
        if any(
            result["gpu_commands"][index * 3 + offset]["trajectory"] != measured["trajectories"][mode]
            for offset, mode in enumerate(contract.MODES)
        ):
            raise ValueError("NS actual-LR retained command trajectory differs")
        rebuilt.append(measured)
    if (
        rebuilt != result["results"]
        or result["accepted_cases"] != [item["name"] for item in rebuilt if item["accepted"]]
        or result["rejected_cases"] != [item["name"] for item in rebuilt if not item["accepted"]]
        or result["all_cases_accepted"] != all(item["accepted"] for item in rebuilt)
        or receipt.read_bytes() != raw
        or source(root, candidate) != frozen
    ):
        raise ValueError("NS actual-LR retained numerical decisions changed during reconstruction")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("plan", "execute", "reconstruct"))
    args = parser.parse_args()
    root, candidate = args.root.resolve(strict=True), args.candidate.resolve(strict=True)
    if args.phase == "plan":
        print(plan(root, candidate)["planned_tokens"], flush=True)
    elif args.phase == "execute":
        print(execute(root, candidate)["status"], flush=True)
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("NS actual-LR retained reconstruction requires hidden CUDA")
        print(reconstruct(root, candidate)["status"], flush=True)


if __name__ == "__main__":
    main()
