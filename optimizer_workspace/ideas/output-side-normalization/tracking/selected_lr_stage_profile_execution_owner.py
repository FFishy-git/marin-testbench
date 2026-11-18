# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorize only acknowledged bounded profiler ranks before creating any CUDA context."""

import os
import subprocess
import sys
import time
from pathlib import Path

from .selected_lr_release_owner import candidate_files, observed, read, sha256
from .selected_lr_stage_profile_contract import (
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

TASK = Path(__file__).resolve().parents[1]


def require_process(pid: int, marker_name: str, marker: str, cwd: Path) -> dict:
    process = observed(pid)
    if (
        process["cwd"] != str(cwd)
        or process["environment"].get(marker_name) != marker
        or any(process["environment"].get(name) != value for name, value in ENVIRONMENT.items())
    ):
        raise RuntimeError("Stage-profile process cwd, marker or CUDA environment differs")
    return process


def require_controller(release_path: Path) -> tuple[dict, dict]:
    prepared_path = TASK / PREPARATION
    prepared = read(prepared_path)
    root = Path(prepared["root"])
    binding_path, helpers, state_path, work, _ = locations(root)
    if root != root.resolve() or release_path != binding_path:
        raise RuntimeError("Stage-profile execution requires its canonical distinct release")
    frozen = read(binding_path)
    intent_path = helpers.parent / "controller.json"
    deadline = time.monotonic() + 30
    while True:
        state, intent = read(state_path), read(intent_path)
        if state.get("spawn_status") == "acknowledged" and intent.get("status") == "acknowledged":
            break
        if state.get("status") not in ("preflight", "validating") or time.monotonic() >= deadline:
            raise RuntimeError("Stage-profile execution has no acknowledged controller/validator generations")
        time.sleep(0.05)
    files = candidate_files(TASK)
    commands = validation_commands(TASK, root, work)
    if (
        prepared["status"] != PREPARED_STATUS
        or prepared["candidate"] != str(TASK)
        or prepared["files_sha256"] != files
        or frozen["status"] != FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != str(TASK)
        or frozen["preparation_sha256"] != sha256(prepared_path)
        or frozen["files_sha256"] != files
        or files["experiment/selected_lr_stage_profile_owner.py"] != sha256(Path(__file__))
        or frozen["helpers"] != str(helpers)
        or frozen["node"] != NODE
        or frozen["environment"] != ENVIRONMENT
        or frozen["commands"] != commands
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or any(
            Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != digest
            for name, digest in frozen["helpers_sha256"].items()
        )
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["predecessor_release_sha256"] != sha256(root / "run-preparation/wave12-launch-binding.json")
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
        raise RuntimeError("Stage-profile binding, source, data or allocation differs")
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
        raise RuntimeError("Stage-profile acknowledged controller or validator generation differs")
    current, seen = os.getpid(), set()
    while current != validator["pid"]:
        if current <= 1 or current in seen:
            raise RuntimeError("Stage-profile rank is outside its acknowledged validator ancestry")
        seen.add(current)
        current = require_process(current, CHILD_MARKER, state["child_marker"], TASK)["parent"]
    if observed(controller["pid"]) != controller or observed(validator["pid"]) != validator:
        raise RuntimeError("Stage-profile ownership changed during ancestry inspection")
    return state, frozen


def require_gpu_owner(root: Path) -> dict:
    state, _ = require_controller(locations(root)[0])
    if os.getpid() != state["child_pid"] or os.getppid() != state["controller_pid"]:
        raise RuntimeError("Stage-profile validator is not the directly owned marked child")
    return {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}


def require_selected_lr_stage_profile_owner(args: object) -> None:
    """Require the exact bounded launcher and each rank's independent Torch process session."""
    state, frozen = require_controller(args.launch_release)
    root = Path(frozen["root"])
    invocation = [PYTHON, "-m", "experiment.train", *sys.argv[1:]]
    matches = [command for command in frozen["commands"] if invocation == [command[0], *command[5:]]]
    if len(matches) != 1:
        raise RuntimeError("Stage-profile rank command differs from the exact bounded protocol")
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
        raise RuntimeError("Stage-profile rank configuration or world ownership differs")
    rank = require_process(os.getpid(), CHILD_MARKER, state["child_marker"], TASK)
    if (
        rank["command"] not in ([command[0], *command[5:]], [command[0], "-u", *command[5:]])
        or rank["group"] != rank["pid"]
        or rank["session"] != rank["pid"]
    ):
        raise RuntimeError("Stage-profile training rank command or independent session differs")
    launcher = require_process(rank["parent"], CHILD_MARKER, state["child_marker"], TASK)
    if (
        launcher["command"] != command
        or launcher["parent"] != state["child_pid"]
        or launcher["group"] != launcher["pid"]
        or launcher["session"] != launcher["pid"]
    ):
        raise RuntimeError("Stage-profile training rank has no exact bounded parent launcher")
