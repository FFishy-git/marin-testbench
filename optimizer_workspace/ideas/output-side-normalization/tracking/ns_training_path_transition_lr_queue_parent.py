# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observe the transition terminal archive without merging its execution helpers."""

import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import ns_training_path_followup as followup
import ns_training_path_transition_lr_parent as bridge
from fused_attribution_validation_controller import acknowledged_process, finish_marked_child

SEAL = bridge.SEAL
OBSERVER_MARKER = "OUTPUT_MARIN_NS_TRANSITION_LR_QUEUE_OBSERVER"
OBSERVER_TIMEOUT = 900
PROGRAM = """import json,sys
if sys.stdin.read(1) != "1":
    raise RuntimeError("Transition-LR observer lost its caller acknowledgement")
from pathlib import Path
import prepare_ns_training_path_transition_lr_predecessor as preparation
import ns_training_path_transition_lr_predecessor as predecessor
root,path,operation=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
preparation.accepted(root,path)
value=(predecessor.readiness if operation=='readiness' else predecessor.evidence)(root)
print(json.dumps(value,allow_nan=False))
"""


def inspect(root: Path, operation: str) -> dict:
    """Run the pinned external archive's read-only observer with its own import origins."""
    if operation not in ("readiness", "evidence"):
        raise ValueError("Transition-LR queue parent operation is unknown")
    before = bridge.cpu_sources(root)
    helpers = Path(before["helpers"])
    argv = [bridge.PYTHON, "-B", "-P", "-c", PROGRAM, str(root), str(root / bridge.CPU), operation]
    marker = uuid4().hex
    child = subprocess.Popen(
        argv,
        cwd=helpers,
        env=os.environ | bridge.ENVIRONMENT | {"PYTHONPATH": str(helpers), "UV_OFFLINE": "1", OBSERVER_MARKER: marker},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        acknowledged_process(child, argv, helpers, OBSERVER_MARKER, marker, bridge.ENVIRONMENT)
        stdout, stderr = child.communicate(input="1", timeout=OBSERVER_TIMEOUT)
    finally:
        finish_marked_child(child, marker, {}, OBSERVER_MARKER)
    if bridge.cpu_sources(root) != before:
        raise ValueError("Transition-LR queue parent changed during observation")
    if child.returncode != 0:
        raise RuntimeError(
            f"Transition-LR external {operation} reader refused evidence " f"(exit {child.returncode}): {stderr[-2000:]}"
        )
    return json.loads(stdout)


def readiness(root: Path) -> dict:
    return inspect(root, "readiness")


def evidence(root: Path) -> dict:
    return inspect(root, "evidence")


def require_seal(root: Path, digest: str) -> dict:
    return bridge.require_seal(root, digest)


def handle(expected: dict, cwd: Path, marker_name: str, marker: str, environment: dict) -> dict:
    return followup.handle(expected, cwd, marker_name, marker, environment)
