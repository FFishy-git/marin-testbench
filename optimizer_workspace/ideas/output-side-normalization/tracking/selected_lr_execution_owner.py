# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorize only exact bounded descendants of the separate Wave-11 validator."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .reference import TASK
from .selected_lr_validation import (
    CHILD_MARKER,
    CONTROLLER_MARKER,
    ENVIRONMENT,
    MASK,
    NODE,
    PREPARATION,
    STATUS,
    controller_command,
    locations,
    validation_commands,
    validator_command,
)

PROC_ROOT = Path("/proc")


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def observed(pid):
    directory = PROC_ROOT / str(pid)
    before = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    argv = [item.decode() for item in (directory / "cmdline").read_bytes().split(b"\0") if item]
    environment = dict(item.decode().split("=", 1) for item in (directory / "environ").read_bytes().split(b"\0") if item)
    cwd = (directory / "cwd").resolve()
    after = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    if before[19] != after[19] or before[1:4] != after[1:4] or after[0] in ("Z", "X", "x"):
        raise RuntimeError("Selected-LR process exited or changed generation during observation")
    return {
        "pid": pid,
        "parent": int(after[1]),
        "group": int(after[2]),
        "session": int(after[3]),
        "start_ticks": after[19],
        "command": argv,
        "environment": environment,
        "cwd": str(cwd),
    }


def candidate_files():
    alias = TASK / "artifacts/external-sources"
    pinned = TASK / "artifacts/pinned-sources-materialized"
    if not alias.is_symlink() or alias.readlink() != Path("pinned-sources-materialized") or pinned.is_symlink():
        raise RuntimeError("Selected-LR pinned source alias differs")
    files = {}
    for directory in (TASK / "experiment", pinned):
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError("Selected-LR source directory is missing or aliased")
        for path in directory.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise RuntimeError("Selected-LR source inventory contains an alias")
            if path.is_file() and (directory == pinned or path.suffix in (".py", ".json")):
                files[str(path.relative_to(TASK))] = sha256(path)
    return files


def require_process(pid, marker_name, marker, cwd):
    process = observed(pid)
    if (
        process["cwd"] != str(cwd.resolve())
        or process["environment"].get(marker_name) != marker
        or any(process["environment"].get(name) != value for name, value in ENVIRONMENT.items())
    ):
        raise RuntimeError("Selected-LR process cwd, marker or GPU environment differs")
    return process


def require_controller():
    """Require an immutable binding and acknowledged live controller/validator generations."""
    prepared_path = TASK / PREPARATION
    prepared = read(prepared_path)
    root = Path(prepared["root"])
    binding_path, helpers, state_path, work, _ = locations(root)
    frozen = read(binding_path)
    intent_path = helpers.parent / "controller.json"
    deadline = time.monotonic() + 30
    while True:
        state, intent = read(state_path), read(intent_path)
        if state.get("spawn_status") == "acknowledged" and intent.get("status") == "acknowledged":
            break
        if state.get("status") not in ("preflight", "validating") or time.monotonic() >= deadline:
            raise RuntimeError("Selected-LR validator has no acknowledged process generation")
        time.sleep(0.05)
    files = candidate_files()
    if (
        frozen["status"] != STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != str(TASK.resolve())
        or prepared["candidate"] != str(TASK.resolve())
        or frozen["preparation_sha256"] != sha256(prepared_path)
        or frozen["files_sha256"] != files
        or files != prepared["files_sha256"]
        or files.get("experiment/selected_lr_owner.py") != sha256(Path(__file__))
        or frozen["node"] != NODE
        or frozen["helpers"] != str(helpers)
        or frozen["environment"] != ENVIRONMENT
        or frozen["commands"] != validation_commands(frozen["python"], TASK, root, work)
        or frozen["python"] != controller_command(root)[0]
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or not {"run_selected_lr_validation.py", "validate_selected_lr_full_budget.py"}
        <= frozen["helpers_sha256"].keys()
        or any(
            not (helpers / name).resolve().is_relative_to(helpers.resolve()) or sha256(helpers / name) != digest
            for name, digest in frozen["helpers_sha256"].items()
        )
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or state["status"] != "validating"
        or state["binding_sha256"] != sha256(binding_path)
        or intent["binding_sha256"] != sha256(binding_path)
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or intent["command"] != controller_command(root)
        or state["command"] != validator_command(root, TASK)
        or os.environ.get(CHILD_MARKER) != state["child_marker"]
        or os.environ.get("CUDA_VISIBLE_DEVICES") != MASK
        or any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())
        or Path.cwd().resolve() != TASK.resolve()
        or NODE not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("Selected-LR validation source, data, binding or ownership differs")
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
        raise RuntimeError("Selected-LR acknowledged controller or validator generation differs")
    current, seen = os.getpid(), set()
    while current != validator["pid"]:
        if current <= 1 or current in seen:
            raise RuntimeError("Selected-LR CUDA rank is outside its validator ancestry")
        seen.add(current)
        process = require_process(current, CHILD_MARKER, state["child_marker"], TASK)
        current = process["parent"]
    if observed(controller["pid"]) != controller or observed(validator["pid"]) != validator:
        raise RuntimeError("Selected-LR ownership changed while inspecting ancestry")
    return state, frozen


def require_gpu_owner():
    state, _ = require_controller()
    if os.getpid() != state["child_pid"] or os.getppid() != state["controller_pid"]:
        raise RuntimeError("Selected-LR validator must be the directly owned marked child")
    return {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}


def require_selected_lr_owner():
    raise RuntimeError("Selected-LR validation permits only its actual-config validator and bounded training ranks")


def require_selected_lr_training(args):
    """Reject unbounded production commands before any CUDA context is created."""
    state, frozen = require_controller()
    invocation = [frozen["python"], "-m", "experiment.train", *sys.argv[1:]]
    matched = [command for command in frozen["commands"] if invocation == [command[0], *command[5:]]]
    if len(matched) != 1:
        raise RuntimeError("Selected-LR training differs from its exact bounded validation command")
    command = matched[0]
    if str(args.config) != command[command.index("--config") + 1]:
        raise RuntimeError("Selected-LR training configuration differs")
    rank = observed(os.getpid())
    if rank["command"] not in ([command[0], *command[5:]], [command[0], "-u", *command[5:]]):
        raise RuntimeError("Selected-LR CUDA rank command differs")
    current = rank
    while current["pid"] != state["child_pid"]:
        if current["parent"] == state["child_pid"]:
            if current["command"] != command:
                raise RuntimeError("Selected-LR bounded launcher command differs")
            return
        current = observed(current["parent"])
    raise RuntimeError("Selected-LR CUDA rank has no exact bounded launcher")
