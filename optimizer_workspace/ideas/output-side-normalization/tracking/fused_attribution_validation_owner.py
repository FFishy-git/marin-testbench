# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail closed before CUDA unless the distinct validator owns this exact bounded rank."""

import os
import subprocess
import sys
import time
from pathlib import Path

from .fused_attribution_validation_contract import (
    CHILD_MARKER,
    CONTROLLER_MARKER,
    ENVIRONMENT,
    FROZEN_STATUS,
    NODE,
    PREPARATION,
    PREPARED_STATUS,
    PYTHON,
    WORLD,
    controller_command,
    locations,
    validation_commands,
    validator_command,
)
from .selected_lr_release_owner import candidate_files, observed, read, sha256

TASK = Path(__file__).resolve().parents[1]


def require_process(pid: int, marker_name: str, marker: str, cwd: Path) -> dict:
    process = observed(pid)
    if (
        not marker
        or process["cwd"] != str(cwd)
        or process["environment"].get(marker_name) != marker
        or any(process["environment"].get(name) != value for name, value in ENVIRONMENT.items())
    ):
        raise RuntimeError("Attribution process cwd, marker or explicit eight-GPU environment differs")
    return process


def require_controller(release_path: Path) -> tuple[dict, dict]:
    prepared_path = TASK / PREPARATION
    if not prepared_path.is_file():
        raise RuntimeError("Attribution validation has no separate prepared execution owner")
    prepared = read(prepared_path)
    root = Path(prepared["root"])
    binding_path, helpers, state_path, work, _ = locations(root)
    if root != root.resolve() or release_path != binding_path or not binding_path.is_file():
        raise RuntimeError("Attribution validation has no canonical frozen execution binding")
    frozen = read(binding_path)
    intent_path = helpers.parent / "controller.json"
    deadline = time.monotonic() + 30
    while True:
        state, intent = read(state_path), read(intent_path)
        if state.get("spawn_status") == "acknowledged" and intent.get("status") == "acknowledged":
            break
        if state.get("status") not in ("preflight", "validating") or time.monotonic() >= deadline:
            raise RuntimeError("Attribution validation has no acknowledged controller/validator generations")
        time.sleep(0.05)
    files = candidate_files(TASK)
    predecessor = prepared["allocation_predecessor"]
    keeper = predecessor["restored_keeper"]
    generations = {int(pid): process for pid, process in keeper["generation"].items()}
    compact = state["keeper"]
    if (
        prepared["status"] != PREPARED_STATUS
        or prepared["candidate"] != str(TASK)
        or prepared["files_sha256"] != files
        or frozen["status"] != FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != str(TASK)
        or frozen["files_sha256"] != files
        or frozen["preparation_sha256"] != sha256(prepared_path)
        or files["experiment/fused_attribution_owner.py"] != sha256(Path(__file__))
        or frozen["helpers"] != str(helpers)
        or helpers.is_symlink()
        or frozen["node"] != NODE
        or frozen["environment"] != ENVIRONMENT
        or frozen["commands"] != validation_commands(TASK, root, work)
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or any(
            Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != digest
            for name, digest in frozen["helpers_sha256"].items()
        )
        or frozen["data_receipt_sha256"] != prepared["data_receipt_sha256"]
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["allocation_predecessor"] != predecessor
        or state["allocation_predecessor"] != predecessor
        or any(sha256(root / name) != digest for name, digest in predecessor["evidence_sha256"].items())
        or compact["pid"] != keeper["pid"]
        or compact["start_ticks"] != generations[keeper["pid"]]["start_ticks"]
        or sorted(compact["generation"])
        != sorted([pid, process["start_ticks"], process["group"]] for pid, process in generations.items())
        or state["status"] != "validating"
        or state["binding_sha256"] != sha256(binding_path)
        or intent["binding_sha256"] != sha256(binding_path)
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or intent["command"] != controller_command(root)
        or state["command"] != validator_command(root, TASK)
        or state.get("keeper_release_complete") is not True
        or os.environ.get(CHILD_MARKER) != state["child_marker"]
        or Path.cwd().resolve() != TASK
        or any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())
        or NODE not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("Attribution validation source, command, binding or predecessor allocation differs")
    controller = require_process(state["controller_pid"], CONTROLLER_MARKER, intent["marker"], helpers)
    validator = require_process(state["child_pid"], CHILD_MARKER, state["child_marker"], TASK)
    if (
        controller["start_ticks"] != state["controller_start_ticks"]
        or controller["command"] != controller_command(root)
        or controller["group"] != controller["pid"]
        or controller["session"] != controller["pid"]
        or validator["start_ticks"] != state["child_start_ticks"]
        or validator["command"] != validator_command(root, TASK)
        or validator["parent"] != controller["pid"]
        or validator["group"] != validator["pid"]
        or validator["session"] != validator["pid"]
    ):
        raise RuntimeError("Attribution acknowledged controller/validator generation differs")
    current, seen = os.getpid(), set()
    while current != validator["pid"]:
        if current <= 1 or current in seen:
            raise RuntimeError("Attribution rank escaped its acknowledged validator ancestry")
        seen.add(current)
        current = require_process(current, CHILD_MARKER, state["child_marker"], TASK)["parent"]
    if observed(controller["pid"]) != controller or observed(validator["pid"]) != validator:
        raise RuntimeError("Attribution ownership changed during ancestry inspection")
    if sha256(binding_path) != state["binding_sha256"] or candidate_files(TASK) != files:
        raise RuntimeError("Attribution binding or candidate changed during ownership inspection")
    return state, frozen


def require_gpu_owner(root: Path) -> dict:
    state, _ = require_controller(locations(root)[0])
    if os.getpid() != state["child_pid"] or os.getppid() != state["controller_pid"]:
        raise RuntimeError("Attribution validator is not the original directly owned child")
    return {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}


def require_fused_attribution_owner(args: object) -> None:
    state, frozen = require_controller(args.launch_release)
    root = Path(frozen["root"])
    invocation = [PYTHON, "-m", "experiment.train", *sys.argv[1:]]
    matches = [command for command in frozen["commands"] if invocation == [command[0], *command[5:]]]
    if len(matches) != 1:
        raise RuntimeError("Attribution rank command is not one exact bounded protocol command")
    command = matches[0]
    if (
        state.get("active_command") != command
        or str(args.config) != command[command.index("--config") + 1]
        or args.launch_release != locations(root)[0]
        or os.environ.get("WORLD_SIZE") != str(WORLD)
        or os.environ.get("LOCAL_WORLD_SIZE") != str(WORLD)
        or os.environ.get("RANK") not in tuple(str(rank) for rank in range(WORLD))
        or os.environ.get("LOCAL_RANK") != os.environ.get("RANK")
    ):
        raise RuntimeError("Attribution active command, actual configuration or eight-rank identity differs")
    rank = require_process(os.getpid(), CHILD_MARKER, state["child_marker"], TASK)
    if (
        rank["command"] not in ([command[0], *command[5:]], [command[0], "-u", *command[5:]])
        or rank["group"] != rank["pid"]
        or rank["session"] != rank["pid"]
    ):
        raise RuntimeError("Attribution rank command or independent Torch session differs")
    launcher = require_process(rank["parent"], CHILD_MARKER, state["child_marker"], TASK)
    if (
        launcher["command"] != command
        or launcher["parent"] != state["child_pid"]
        or launcher["group"] != launcher["pid"]
        or launcher["session"] != launcher["pid"]
    ):
        raise RuntimeError("Attribution rank has no exact independently sessioned parent launcher")
