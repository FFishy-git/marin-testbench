# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorize only the separately released Wave-9 full-budget worker descendants."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .launch import ENVIRONMENT, NODES, OUTPUT_ROOT, PYTHON
from .records import sha256
from .reference import TASK
from .upstream_validation import ARMS

WORKER_MARKER = "OUTPUT_MARIN_WAVE9_WORKER_ID"
MARKER = "OUTPUT_MARIN_DIAGNOSTIC_ID"
CHANGES = {
    "experiment/execution_owner.py",
    "experiment/production_owner.py",
    "experiment/cluster.json",
    "experiment/tests/test_upstream_release.py",
}


def observed(pid):
    directory = Path("/proc") / str(pid)
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    argv = [arg.decode() for arg in (directory / "cmdline").read_bytes().split(b"\0") if arg]
    return fields, argv


def require_release(release_path):
    """Check original numerical evidence and the explicitly bounded ownership-only migration."""
    preparation = OUTPUT_ROOT / "run-preparation"
    if release_path != preparation / "wave9-launch-binding.json":
        raise RuntimeError("Upstream full-budget production release has not been issued for this noncanonical path")
    release = json.loads(release_path.read_text())
    original_path = preparation / "upstream-full-model-corrected-binding.json"
    original = json.loads(original_path.read_text())
    state_path = preparation / "upstream-full-model-corrected-state.json"
    validation_path = preparation / "validation-upstream-full-model-corrected-gpu.json"
    state = json.loads(state_path.read_text())
    validation = json.loads(validation_path.read_text())
    helpers = OUTPUT_ROOT / "controllers/wave9-launch-20261117/tracking"
    previous = Path(original["candidate"])
    changed = {
        name
        for name in release["files_sha256"].keys() | original["files_sha256"].keys()
        if release["files_sha256"].get(name) != original["files_sha256"].get(name)
    }
    commands = validation["evidence"].get("gpu_commands", [])
    if (
        release["status"] != "frozen_validated_launch_release"
        or release["root"] != str(OUTPUT_ROOT)
        or release["candidate"] != str(TASK.resolve())
        or release["helpers"] != str(helpers)
        or changed != CHANGES
        or release["validation_binding_sha256"] != sha256(original_path)
        or release["validation_sha256"] != sha256(validation_path)
        or release["state_sha256"] != sha256(state_path)
        or release["data_receipt_sha256"] != original["data_receipt_sha256"]
        or release["data_receipt_sha256"] != sha256(OUTPUT_ROOT / "data/experiment_receipt.json")
        or tuple(release["arms"]) != ARMS
        or tuple(release["nodes"]) != NODES
        or release["environment"] != ENVIRONMENT
        or original["environment"] != ENVIRONMENT
        or validation["environment"] != ENVIRONMENT
        or any(os.environ.get(key) != value for key, value in ENVIRONMENT.items())
        or original["archive_sha256"] != sha256(Path(original["archive"]))
        or any(sha256(previous / name) != digest for name, digest in original["files_sha256"].items())
        or any(sha256(TASK / name) != digest for name, digest in release["files_sha256"].items())
        or not release["helpers_sha256"]
        or "upstream_production.py" not in release["helpers_sha256"]
        or any(sha256(helpers / name) != digest for name, digest in release["helpers_sha256"].items())
        or release["archive_sha256"] != sha256(Path(release["archive"]))
        or release["archive"] != str(preparation / "source-wave9-launch-20261117.tar")
        or validation["source_sha256"] != original["source_sha256"]
        or validation["data_receipt_sha256"] != original["data_receipt_sha256"]
        or validation["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or not all(validation["checks"].get(gate) is True for gate in original["required_gates"])
        or [entry["command"] for entry in commands] != original["commands"]
        or any(
            type(entry["exit_code"]) is not int
            or entry["exit_code"] != 0
            or sha256(Path(entry["log"])) != entry["log_sha256"]
            for entry in commands
        )
        or state["status"] != "passed"
        or state["binding_sha256"] != sha256(original_path)
        or state["validation_sha256"] != sha256(validation_path)
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or "cleanup_error" in state
        or release["preparation_sha256"] != sha256(Path(release["preparation"]))
        or release["cpu_receipt_sha256"] != sha256(Path(release["cpu_receipt"]))
    ):
        raise RuntimeError("Wave-9 production source migration, original gates or release identity changed")
    return release


def require_production_owner(args, config):
    """Require the exact full-budget command, live worker generation and marked eight-rank launcher."""
    release = require_release(args.launch_release)
    arm = config["experiment"]["arm"]
    if arm not in ARMS:
        raise RuntimeError("Wave-9 production arm is not released")
    node = NODES[ARMS.index(arm)]
    release_path = OUTPUT_ROOT / "run-preparation/wave9-launch-binding.json"
    command = [
        PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "experiment.train",
        "--config",
        str(TASK / f"experiment/configs/{arm}_0p5m_seed0.json"),
        "--data",
        str(OUTPUT_ROOT / "data"),
        "--data-receipt",
        str(OUTPUT_ROOT / "data/experiment_receipt.json"),
        "--out",
        str(OUTPUT_ROOT / "runs/wave9" / arm),
        "--launch-release",
        str(release_path),
    ]
    if (
        release["commands"][node] != command
        or [PYTHON, "-m", "experiment.train", *sys.argv[1:]] != [command[0], *command[5:]]
        or args.config != TASK / f"experiment/configs/{arm}_0p5m_seed0.json"
        or config != json.loads(args.config.read_text())
        or args.max_steps is not None
        or args.resume is not None
        or args.execution_boundary is not None
        or args.verify_diagonal_capture
        or args.verify_outer_diagonal_updates
        or node not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("Wave-9 production differs from its exact full-budget allocation or command")
    state_path = OUTPUT_ROOT / "run-preparation" / f"wave9-worker-{node}.json"
    deadline = time.monotonic() + 30
    while True:
        state = json.loads(state_path.read_text())
        fields, argv = observed(state["worker_pid"])
        expected = [
            PYTHON,
            str(Path(release["helpers"]) / "upstream_production.py"),
            "--root",
            str(OUTPUT_ROOT),
            "--phase",
            "worker",
            "--arm",
            arm,
        ]
        if (
            fields[0] == "Z"
            or fields[19] != state["worker_start_ticks"]
            or argv != expected
            or state["arm"] != arm
            or state["node"] != node
            or state["release_binding_sha256"] != sha256(release_path)
            or os.environ.get(WORKER_MARKER) != state["worker_marker"]
        ):
            raise RuntimeError("Wave-9 production worker generation, command or release changed")
        if state["status"] == "running" and "child_start_ticks" in state:
            break
        if state["status"] not in ("starting", "running") or time.monotonic() >= deadline:
            raise RuntimeError("Wave-9 production has no matching launcher acknowledgment")
        time.sleep(0.05)
    fields, argv = observed(state["child_pid"])
    if (
        fields[0] == "Z"
        or fields[19] != state["child_start_ticks"]
        or int(fields[1]) != state["worker_pid"]
        or argv != command
        or state["command"] != command
        or os.environ.get(MARKER) != state["child_marker"]
    ):
        raise RuntimeError("Wave-9 production launcher generation, command or parent changed")
    current = os.getpid()
    while current != state["child_pid"]:
        fields, _ = observed(current)
        current = int(fields[1])
        if current <= 1:
            raise RuntimeError("Wave-9 rank is not descended from its marked eight-rank launcher")
