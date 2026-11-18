# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Temporary persisted fixtures for attribution protocol/preparer tests."""

import json

from fused_attribution_validation_contract import NODE, predecessor_locations
from launch_wave5 import sha256


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def keeper():
    generation = {
        str(pid): {
            "pid": pid,
            "status": "R",
            "parent": 1 if pid == 100 else 100,
            "group": pid,
            "session": pid,
            "start_ticks": str(pid + 900),
            "command": ["keeper", str(pid)],
        }
        for pid in range(100, 109)
    }
    return {
        "pid": 100,
        "generation": generation,
        "devices": [[str(i), f"GPU-{i}", "NVIDIA A100-SXM4-40GB"] for i in range(8)],
        "occupied": [[str(101 + i), f"GPU-{i}"] for i in range(8)],
    }


def write_failed_profiler(root):
    binding_path, helpers, state_path, receipt_path = predecessor_locations(root)
    helpers.mkdir(parents=True)
    helper = helpers / "run_selected_lr_stage_profile.py"
    helper.write_text("# archived profiler controller\n")
    archive = root / "source-profile.tar"
    archive.write_bytes(b"fixture immutable profile archive")
    binding = {
        "root": str(root),
        "helpers": str(helpers),
        "node": NODE,
        "candidate": str(root / "profile"),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "helpers_sha256": {helper.name: sha256(helper)},
    }
    record(binding_path, binding)
    log = helpers.parent / "validator.log"
    log.write_text("preserved trace coverage failure\n")
    owner = {"controller_pid": 201, "controller_start_ticks": "301"}
    state = owner | {
        "status": "needs_attention",
        "node": NODE,
        "candidate": binding["candidate"],
        "binding_sha256": sha256(binding_path),
        "child_pid": 202,
        "child_start_ticks": "302",
        "child_marker": "old-child",
        "exit_code": 1,
        "cleanup_exit_code": 1,
        "keeper_restore": 0,
        "marked_descendants_absent": True,
        "keeper_release_complete": True,
        "restored_keeper": keeper(),
        "log": str(log),
        "log_sha256": sha256(log),
        "error": "ValueError: original profiler trace failed",
    }
    record(state_path, state)
    record(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "pid": 201,
            "start_ticks": "301",
            "marker": "old-controller",
            "binding_sha256": sha256(binding_path),
        },
    )
    record(receipt_path, {"status": "needs_attention", "owner": owner, "error": state["error"]})
