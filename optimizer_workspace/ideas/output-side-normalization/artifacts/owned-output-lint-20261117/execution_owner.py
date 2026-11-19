# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorize only exact bounded upstream-product validation descendants before CUDA."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .launch import ENVIRONMENT, OUTPUT_ROOT, PYTHON
from .records import sha256
from .reference import TASK
from .production_owner import require_production_owner
from .upstream_validation import validation_commands

MARKER = "OUTPUT_MARIN_DIAGNOSTIC_ID"
NODE = "28.216.17.139"


def observed(pid):
    directory = Path("/proc") / str(pid)
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    argv = [arg.decode() for arg in (directory / "cmdline").read_bytes().split(b"\0") if arg]
    return fields, argv


def require_controller():
    """Require immutable source and live marked-validator ancestry, including spawn acknowledgment."""
    preparation = OUTPUT_ROOT / "run-preparation"
    binding_path = preparation / "upstream-full-model-corrected-binding.json"
    state_path = preparation / "upstream-full-model-corrected-state.json"
    helpers = OUTPUT_ROOT / "controllers/upstream-full-model-corrected-20261117/tracking"
    expected = [PYTHON, str(helpers / "run_upstream_full_model.py"), "--root", str(OUTPUT_ROOT), "--phase", "validate"]
    deadline = time.monotonic() + 30
    while True:
        state = json.loads(state_path.read_text())
        fields, argv = observed(state["controller_pid"])
        if fields[0] == "Z" or fields[19] != state["controller_start_ticks"] or argv != expected:
            raise RuntimeError("Upstream validation controller generation or command changed")
        if state["status"] == "validating" and "child_start_ticks" in state:
            break
        if state["status"] not in ("preflight", "validating") or time.monotonic() >= deadline:
            raise RuntimeError("Upstream validation has no matching marked-child acknowledgment")
        time.sleep(0.05)
    frozen = json.loads(binding_path.read_text())
    if (
        frozen["status"] != "frozen_pending_upper_lr_and_synthetic_probe"
        or frozen["root"] != str(OUTPUT_ROOT)
        or frozen["candidate"] != str(TASK.resolve())
        or frozen["node"] != NODE
        or frozen["helpers"] != str(helpers)
        or state["node"] != NODE
        or state["candidate"] != frozen["candidate"]
        or state["binding_sha256"] != sha256(binding_path)
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or not frozen["files_sha256"]
        or any(sha256(TASK / name) != digest for name, digest in frozen["files_sha256"].items())
        or not frozen["helpers_sha256"]
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or frozen["data_receipt_sha256"] != sha256(OUTPUT_ROOT / "data/experiment_receipt.json")
        or frozen["environment"] != ENVIRONMENT
        or any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())
        or os.environ.get(MARKER) != state["child_marker"]
        or NODE not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("Upstream validation source, data, environment or allocation ownership changed")
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
        str(OUTPUT_ROOT / "validation/upstream_full_model_corrected"),
        "--receipt",
        str(preparation / "validation-upstream-full-model-corrected-gpu.json"),
    ]
    if (
        fields[0] == "Z"
        or fields[19] != state["child_start_ticks"]
        or int(fields[1]) != state["controller_pid"]
        or argv != expected_child
        or state["command"] != expected_child
    ):
        raise RuntimeError("Upstream validator generation, command or parent changed")
    current = os.getpid()
    while current != state["child_pid"]:
        fields, _ = observed(current)
        current = int(fields[1])
        if current <= 1:
            raise RuntimeError("CUDA process is not descended from the bound upstream validator")
    return state


def require_gpu_owner():
    state = require_controller()
    if state["child_pid"] != os.getpid() or os.getppid() != state["controller_pid"]:
        raise RuntimeError("Upstream validator must be the directly owned marked child")
    return {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}


def require_synthetic_owner():
    """Restrict synthetic CUDA ranks to the exact two-rank launcher."""
    state = require_controller()
    current = os.getpid()
    while current != state["child_pid"]:
        fields, argv = observed(current)
        parent = int(fields[1])
        if parent == state["child_pid"]:
            expected = validation_commands(
                PYTHON, TASK, OUTPUT_ROOT, OUTPUT_ROOT / "validation/upstream_full_model_corrected"
            )[0]
            if argv != expected:
                raise RuntimeError("Synthetic CUDA launcher differs from its exact two-rank command")
            return state
        current = parent
    raise RuntimeError("Synthetic CUDA checks must descend from their exact two-rank launcher")


def require_training_owner(args, config):
    """Deny production until separately released; authorize only canonical smoke commands."""
    if args.launch_release is not None:
        require_production_owner(args, config)
        return
    state = require_controller()
    commands = validation_commands(PYTHON, TASK, OUTPUT_ROOT, OUTPUT_ROOT / "validation/upstream_full_model_corrected")[
        1:
    ]
    invocation = [PYTHON, "-m", "experiment.train", *sys.argv[1:]]
    allowed = [[command[0], *command[5:]] for command in commands]
    if invocation not in allowed or config != json.loads(args.config.read_text()):
        raise RuntimeError("Upstream CUDA training differs from the exact bounded validation command")
    current = os.getpid()
    while current != state["child_pid"]:
        fields, argv = observed(current)
        parent = int(fields[1])
        if parent == state["child_pid"]:
            if argv not in commands or [argv[0], *argv[5:]] != invocation:
                raise RuntimeError("Upstream rank is not owned by its exact eight-rank launcher")
            return
        current = parent
    raise RuntimeError("Upstream CUDA rank has no exact eight-rank launcher")
