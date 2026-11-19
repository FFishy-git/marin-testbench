# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate one frozen full-model validator and its exact eight-rank commands."""

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import ns_full_model_contract as contract
import ns_full_model_execution_source as source
import ns_full_model_predecessor as predecessor
import ns_full_model_validation as validation
import owned_output_handoff as handoff
from fused_attribution_validation_controller import (
    process_context,
    require_acknowledged_controller,
    require_helper_bytes,
)
from prepare_selected_lr_validation import require_archive
from saved_ns_gpu_owner import require_held_lock
from selected_lr_failure_handoff import MANAGER_LOCK, released_keeper
from selected_lr_release_owner import candidate_files, read, sha256
from selected_lr_release_owner import observed as process_details

NODE = "28.216.17.9"
TIMEOUT = 54 * 900 + 14400


def controller_command(root: Path) -> list[str]:
    return [
        contract.PYTHON,
        "-P",
        str(contract.locations(root)[1] / "ns_full_model_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "execute",
    ]


def validator_command(root: Path, candidate: Path) -> list[str]:
    return [
        contract.PYTHON,
        "-P",
        str(contract.locations(root)[1] / "ns_full_model_validation.py"),
        "--root",
        str(root),
        "--candidate",
        str(candidate),
        "--phase",
        "execute",
    ]


def require_binding(root: Path) -> dict:
    path, helpers, _, _, _ = contract.locations(root)
    raw, frozen = path.read_bytes(), read(path)
    cpu = source.require_cpu(root, Path(frozen["cpu_receipt"]))
    candidate = Path(frozen["candidate"])
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    if (
        root != root.resolve(strict=True)
        or frozen["status"] != contract.FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != cpu["candidate"]
        or frozen["files_sha256"] != candidate_files(candidate)
        or frozen["preparation_sha256"] != sha256(candidate / source.PREPARATION)
        or frozen["helpers"] != str(helpers)
        or frozen["helpers_sha256"] != cpu["helpers_sha256"]
        or frozen["cpu_receipt_sha256"] != sha256(Path(frozen["cpu_receipt"]))
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["node"] != NODE
        or frozen["environment"] != contract.ENVIRONMENT
        or frozen["allocation_predecessor"] != predecessor.evidence(root)
        or frozen["devices"] != frozen["allocation_predecessor"]["restored_keeper"]["devices"]
        or frozen["commands"] != contract.commands(candidate, root, contract.locations(root)[3])
        or frozen["plan"] != validation.plan(root, candidate)
        or frozen["controller_command"] != controller_command(root)
        or frozen["validator_command"] != validator_command(root, candidate)
        or frozen["validation_timeout"] != TIMEOUT
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
    ):
        raise ValueError("NS full-model frozen owner/source/protocol/predecessor identity differs")
    require_archive(Path(frozen["archive"]), cpu["inventory_sha256"], alias=True)
    for name in frozen["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"NS full-model imported mutable execution helper: {name}")
    if Path(__file__).resolve() != helpers / Path(__file__).name or path.read_bytes() != raw:
        raise ValueError("NS full-model issued owner entrypoint or binding changed")
    return frozen


def process(pid: int, command: list[str], cwd: Path, marker_name: str, marker: str) -> dict:
    return process_context(pid, command, cwd, marker_name, marker, contract.ENVIRONMENT)


def require_controller(root: Path, candidate: Path) -> tuple[dict, dict]:
    frozen = require_binding(root)
    binding, helpers, state_path, _, _ = contract.locations(root)
    intent, deadline = read(helpers.parent / "controller.json"), time.monotonic() + 30
    while True:
        state = read(state_path)
        if state.get("spawn_status") == "acknowledged" and intent["status"] == "acknowledged":
            break
        if state.get("status") not in ("preflight", "validating") or time.monotonic() >= deadline:
            raise RuntimeError("NS full-model lacks original controller/validator acknowledgment")
        time.sleep(0.01)
        intent = read(helpers.parent / "controller.json")
    if (
        frozen["candidate"] != str(candidate)
        or state["status"] != "validating"
        or state["binding_sha256"] != sha256(binding)
        or intent["binding_sha256"] != sha256(binding)
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or state["allocation_predecessor"] != frozen["allocation_predecessor"]
        or state["keeper_release_complete"] is not True
        or not released_keeper(state["keeper"], frozen["allocation_predecessor"]["restored_keeper"])
        or intent["command"] != controller_command(root)
        or state["command"] != validator_command(root, candidate)
        or state["child_generation"]["pid"] != state["child_pid"]
        or state["child_generation"]["start_ticks"] != state["child_start_ticks"]
        or state["child_generation"]["parent"] != state["controller_pid"]
        or state["child_generation"]["command"] != state["command"]
        or Path.cwd().resolve() != candidate
        or any(os.environ.get(name) != value for name, value in contract.ENVIRONMENT.items())
        or NODE not in subprocess.check_output(["hostname", "-I"], text=True, timeout=20).split()
    ):
        raise RuntimeError("NS full-model source, original owner, keeper transfer or environment differs")
    controller = process(
        state["controller_pid"], intent["command"], helpers, contract.CONTROLLER_MARKER, intent["marker"]
    )
    require_acknowledged_controller(intent, controller)
    validator = process(state["child_pid"], state["command"], candidate, contract.CHILD_MARKER, state["child_marker"])
    if (
        validator["parent"] != controller["pid"]
        or validator["start_ticks"] != state["child_start_ticks"]
        or validator["group"] != validator["pid"]
        or validator["session"] != validator["pid"]
        or not handoff.same_generation(validator, state["child_generation"])
    ):
        raise RuntimeError("NS full-model validator original generation or direct-child ancestry differs")
    for name, path in (
        ("node", root / "run-preparation" / f".wave7-node-{NODE}.lock"),
        ("manager", MANAGER_LOCK),
    ):
        require_held_lock(controller["pid"], state["lock_descriptors"][name], path)
    devices, occupied = (
        list(csv.reader(subprocess.check_output(argv, text=True, timeout=20).splitlines(), skipinitialspace=True))
        for argv in (
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        )
    )
    if devices != frozen["devices"] or handoff.KEEPER_PID.exists():
        raise RuntimeError("NS full-model physical allocation or keeper release differs")
    marked = handoff.marker_processes(contract.CHILD_MARKER, state["child_marker"])
    launchers = {
        entry["pid"]: entry
        for entry in marked.values()
        if entry["command"] == state.get("active_command") and entry["parent"] == validator["pid"]
    }
    ranks = {
        entry["pid"]: entry
        for entry in marked.values()
        if entry["parent"] in launchers
        and entry["command"][:5] == [contract.PYTHON, "-u", "-m", "experiment.train", "--config"]
    }
    if len(launchers) > 1:
        raise RuntimeError("NS full-model physical occupancy has competing active launchers")
    for pid, uuid in occupied:
        rank = ranks.get(int(pid))
        if rank is None:
            raise RuntimeError("NS full-model CUDA occupancy includes an unowned process")
        environment = process_details(int(pid))["environment"]
        local = environment.get("LOCAL_RANK")
        if local not in tuple(str(index) for index in range(contract.WORLD)) or uuid != devices[int(local)][1]:
            raise RuntimeError("NS full-model rank occupies a different physical GPU")
    if not handoff.same_generation(
        controller,
        process(controller["pid"], intent["command"], helpers, contract.CONTROLLER_MARKER, intent["marker"]),
    ) or not handoff.same_generation(
        validator,
        process(validator["pid"], state["command"], candidate, contract.CHILD_MARKER, state["child_marker"]),
    ):
        raise RuntimeError("NS full-model owner changed during physical allocation inspection")
    return state, frozen


def require_gpu_owner(root: Path, candidate: Path) -> dict:
    state, _ = require_controller(root, candidate)
    if os.getpid() != state["child_pid"] or os.getppid() != state["controller_pid"]:
        raise RuntimeError("NS full-model validator is not the original direct child")
    return {
        "controller_pid": state["controller_pid"],
        "controller_start_ticks": state["controller_start_ticks"],
        "validator_pid": state["child_pid"],
        "validator_start_ticks": state["child_start_ticks"],
    }


def require_rank(root: Path, candidate: Path, args: object) -> None:
    state, frozen = require_controller(root, candidate)
    invocation = [contract.PYTHON, "-m", "experiment.train", *sys.argv[1:]]
    matches = [argv for argv in frozen["commands"] if invocation == [argv[0], *argv[5:]]]
    if len(matches) != 1:
        raise RuntimeError("NS full-model rank is outside its exact bounded command protocol")
    command = matches[0]
    if (
        state.get("active_command") != command
        or args.launch_release != contract.locations(root)[0]
        or str(args.config) != command[command.index("--config") + 1]
        or os.environ.get("WORLD_SIZE") != str(contract.WORLD)
        or os.environ.get("LOCAL_WORLD_SIZE") != str(contract.WORLD)
        or os.environ.get("RANK") not in tuple(str(index) for index in range(contract.WORLD))
        or os.environ.get("LOCAL_RANK") != os.environ.get("RANK")
    ):
        raise RuntimeError("NS full-model active command or rank topology differs")
    marker = os.environ.get(contract.COMMAND_MARKER, "")
    if not marker:
        raise RuntimeError("NS full-model rank has no per-command ownership marker")
    rank = process(os.getpid(), [command[0], "-u", *command[5:]], candidate, contract.COMMAND_MARKER, marker)
    launcher = process(rank["parent"], command, candidate, contract.COMMAND_MARKER, marker)
    if (
        rank["group"] != rank["pid"]
        or rank["session"] != rank["pid"]
        or launcher["group"] != launcher["pid"]
        or launcher["session"] != launcher["pid"]
        or launcher["parent"] != state["child_pid"]
        or process_details(rank["pid"])["environment"].get(contract.CHILD_MARKER) != state["child_marker"]
        or process_details(launcher["pid"])["environment"].get(contract.CHILD_MARKER) != state["child_marker"]
    ):
        raise RuntimeError("NS full-model rank lacks exact independently sessioned launcher ancestry")
