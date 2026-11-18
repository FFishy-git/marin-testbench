# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorize only frozen, marked owned-output validation descendants before CUDA."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .launch import ENVIRONMENT, OUTPUT_ROOT, PYTHON
from .owned_output_validation import VALIDATION_ENVIRONMENT, validation_commands
from .records import sha256
from .reference import TASK

MARKER = "OUTPUT_MARIN_DIAGNOSTIC_ID"
NODE = "28.216.17.139"
MASK = "0,1,2,3,4,5,6,7"
PROC_ROOT = Path("/proc")


def observed(pid):
    directory = PROC_ROOT / str(pid)
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    argv = [arg.decode() for arg in (directory / "cmdline").read_bytes().split(b"\0") if arg]
    return fields, argv


def require_inherited(pid, marker):
    directory = PROC_ROOT / str(pid)
    environment = dict(
        entry.decode().split("=", 1) for entry in (directory / "environ").read_bytes().split(b"\0") if entry
    )
    if (
        (directory / "cwd").resolve() != TASK.resolve()
        or environment.get(MARKER) != marker
        or environment.get("CUDA_VISIBLE_DEVICES") != MASK
        or any(environment.get(name) != value for name, value in VALIDATION_ENVIRONMENT.items())
    ):
        raise RuntimeError("Owned-output descendant cwd, mask, environment or marker changed")


def candidate_files():
    materialized = TASK / "artifacts/pinned-sources-materialized"
    alias = TASK / "artifacts/external-sources"
    if (
        not alias.is_symlink()
        or alias.readlink() != Path("pinned-sources-materialized")
        or materialized.is_symlink()
        or not materialized.is_dir()
        or alias.resolve() != materialized.resolve()
    ):
        raise RuntimeError("Owned-output pinned sources are not candidate-local")
    names = set()
    for directory in (TASK / "experiment", materialized):
        if directory.is_symlink():
            raise RuntimeError("Owned-output candidate source directory is aliased")
        for path in directory.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise RuntimeError("Owned-output candidate source inventory contains an alias")
            if path.is_file() and (directory == materialized or path.suffix in (".py", ".json")):
                names.add(str(path.relative_to(TASK)))
    return names


def require_controller():
    """Require frozen source and live acknowledged validator ancestry."""
    preparation = OUTPUT_ROOT / "run-preparation"
    binding_path = preparation / "owned-output-full-model-binding.json"
    state_path = preparation / "owned-output-full-model-state.json"
    helpers = OUTPUT_ROOT / "controllers/owned-output-full-model-validation-20261117/tracking"
    work = OUTPUT_ROOT / "validation/owned_output_full_model"
    expected = [
        PYTHON,
        str(helpers / "run_owned_output_validation.py"),
        "--root",
        str(OUTPUT_ROOT),
        "--phase",
        "validate",
    ]
    deadline = time.monotonic() + 30
    generation = None
    while True:
        state = json.loads(state_path.read_text())
        identity = (state["controller_pid"], state["controller_start_ticks"])
        if generation is None:
            generation = identity
        fields, argv = observed(state["controller_pid"])
        if fields[0] == "Z" or fields[19] != identity[1] or identity != generation or argv != expected:
            raise RuntimeError("Owned-output validation controller generation or command changed")
        if (
            state["status"] == "validating"
            and "child_start_ticks" in state
            and state.get("spawn_status") == "acknowledged"
        ):
            break
        if state["status"] not in ("preflight", "validating") or time.monotonic() >= deadline:
            raise RuntimeError("Owned-output validation has no matching marked-child acknowledgment")
        time.sleep(0.05)
    frozen = json.loads(binding_path.read_text())
    commands = validation_commands(PYTHON, TASK, OUTPUT_ROOT, work)
    files = frozen["files_sha256"]
    if (
        frozen["status"] != "frozen_pending_completed_ceiling_and_owned_gpu_validation"
        or frozen["root"] != str(OUTPUT_ROOT)
        or frozen["candidate"] != str(TASK.resolve())
        or frozen["node"] != NODE
        or frozen["helpers"] != str(helpers)
        or state["node"] != NODE
        or state["candidate"] != frozen["candidate"]
        or state["binding_sha256"] != sha256(binding_path)
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or not files
        or set(files) != candidate_files()
        or any(
            not (TASK / name).resolve().is_relative_to(TASK.resolve()) or sha256(TASK / name) != digest
            for name, digest in files.items()
        )
        or files.get("experiment/execution_owner.py") != sha256(Path(__file__))
        or frozen["source_sha256"]
        != {name: digest for name, digest in files.items() if name.startswith("experiment/") and name.endswith(".py")}
        or not frozen["helpers_sha256"]
        or "run_owned_output_validation.py" not in frozen["helpers_sha256"]
        or any(
            not (helpers / name).resolve().is_relative_to(helpers.resolve()) or sha256(helpers / name) != digest
            for name, digest in frozen["helpers_sha256"].items()
        )
        or frozen["data_receipt_sha256"] != sha256(OUTPUT_ROOT / "data/experiment_receipt.json")
        or frozen["commands"] != commands
        or any(
            str(Path(command[command.index("--config") + 1]).relative_to(TASK)) not in files for command in commands[2:]
        )
        or frozen["environment"] != VALIDATION_ENVIRONMENT
        or ENVIRONMENT != VALIDATION_ENVIRONMENT
        or os.environ.get("CUDA_VISIBLE_DEVICES") != MASK
        or any(os.environ.get(name) != value for name, value in VALIDATION_ENVIRONMENT.items())
        or not state["child_marker"]
        or os.environ.get(MARKER) != state["child_marker"]
        or Path.cwd().resolve() != TASK.resolve()
        or NODE not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("Owned-output validation source, plan, data, environment or allocation changed")
    fields, argv = observed(state["child_pid"])
    expected_child = [
        PYTHON,
        "-m",
        "experiment.validate",
        "--gpu",
        "--data",
        str(OUTPUT_ROOT / "data"),
        "--data-receipt",
        str(OUTPUT_ROOT / "data/experiment_receipt.json"),
        "--work",
        str(work),
        "--receipt",
        str(preparation / "validation-owned-output-full-model-gpu.json"),
    ]
    if (
        fields[0] == "Z"
        or fields[19] != state["child_start_ticks"]
        or int(fields[1]) != state["controller_pid"]
        or argv != expected_child
        or state["command"] != expected_child
    ):
        raise RuntimeError("Owned-output validator generation, command or parent changed")
    require_inherited(state["child_pid"], state["child_marker"])
    current, seen = os.getpid(), set()
    while current != state["child_pid"]:
        if current <= 1 or current in seen:
            raise RuntimeError("CUDA process is not descended from the bound owned-output validator")
        seen.add(current)
        require_inherited(current, state["child_marker"])
        fields, _ = observed(current)
        if fields[0] == "Z":
            raise RuntimeError("Owned-output validation descendant exited")
        current = int(fields[1])
    return state


def require_gpu_owner():
    state = require_controller()
    if state["child_pid"] != os.getpid() or os.getppid() != state["controller_pid"]:
        raise RuntimeError("Owned-output validator must be the directly owned marked child")
    return {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}


def require_launcher(state, command):
    _, argv = observed(os.getpid())
    rank = [command[0], *command[5:]]
    if argv not in (rank, [command[0], "-u", *command[5:]]):
        raise RuntimeError("Owned-output CUDA rank process differs from its exact bounded command")
    current = os.getpid()
    while current != state["child_pid"]:
        fields, argv = observed(current)
        parent = int(fields[1])
        if parent == state["child_pid"]:
            if argv != command:
                raise RuntimeError("Owned-output CUDA launcher differs from its exact bounded command")
            return
        if parent <= 1:
            break
        current = parent
    raise RuntimeError("Owned-output CUDA rank has no exact bounded launcher")


def require_synthetic_owner():
    """Restrict synthetic CUDA ranks to command zero's two-rank launch."""
    state = require_controller()
    command = validation_commands(PYTHON, TASK, OUTPUT_ROOT, OUTPUT_ROOT / "validation/owned_output_full_model")[0]
    if [PYTHON, "-m", "experiment.gpu_checks", *sys.argv[1:]] != [command[0], *command[5:]]:
        raise RuntimeError("Synthetic CUDA rank differs from its exact two-rank command")
    require_launcher(state, command)
    return state


def require_owned_synthetic_owner():
    """Restrict owned-factor CUDA ranks to command one's eight-rank launch."""
    state = require_controller()
    command = validation_commands(PYTHON, TASK, OUTPUT_ROOT, OUTPUT_ROOT / "validation/owned_output_full_model")[1]
    if [PYTHON, "-m", "experiment.owned_output_gpu_checks", *sys.argv[1:]] != [command[0], *command[5:]]:
        raise RuntimeError("Owned-factor CUDA rank differs from its exact eight-rank command")
    require_launcher(state, command)
    return state


def require_training_owner(args, config):
    """Reject every production release; authorize only exact bounded training ranks."""
    if args.launch_release is not None:
        raise RuntimeError("owned output full-budget production release has not been issued")
    state = require_controller()
    commands = validation_commands(PYTHON, TASK, OUTPUT_ROOT, OUTPUT_ROOT / "validation/owned_output_full_model")[2:]
    invocation = [PYTHON, "-m", "experiment.train", *sys.argv[1:]]
    for command in commands:
        if invocation == [command[0], *command[5:]]:
            if str(args.config) != command[command.index("--config") + 1] or config != json.loads(
                args.config.read_text()
            ):
                raise RuntimeError("Owned-output CUDA training configuration changed")
            require_launcher(state, command)
            return
    raise RuntimeError("Owned-output CUDA training differs from the exact bounded validation command")
