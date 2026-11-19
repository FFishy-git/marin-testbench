# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate the completed accuracy trial as the allocation predecessor."""

import os
from pathlib import Path

import owned_output_handoff as handoff
import root_start_gpu_owner as previous
from fused_attribution_validation import require_keeper_shape
from selected_lr_failure_handoff import same_keeper
from selected_lr_release_owner import read, sha256

TERMINAL = "run-preparation/root-start-accuracy-terminal-independent-20261118T2001.json"
TERMINAL_SHA256 = "6d32ada12cd2f199920b391b61179c6bc03d0390db87029707b47b3b2c4750da"
BINDING_SHA256 = "d8c64fc07c110d997bda28a734ac25cb0b44d2ed452f1dd7f98fdb3148cab9d8"
STATE_SHA256 = "cfb3e7518d937637719fd5db56125073929f30010bee90e2290b5b307f03b147"
COMPARISON_SHA256 = "a9f0a2da341ed9e7314e5cbd6bd3012be901da0bb7760cfcb818113d1dd393fe"


def evidence(root: Path) -> dict:
    binding_path, helpers, state_path, work = previous.locations(root)
    paths = {
        TERMINAL: TERMINAL_SHA256,
        str(binding_path.relative_to(root)): BINDING_SHA256,
        str(state_path.relative_to(root)): STATE_SHA256,
        str((work / "comparison.json").relative_to(root)): COMPARISON_SHA256,
    }
    if any(sha256(root / name) != value for name, value in paths.items()):
        raise ValueError("Rational continuation start-accuracy predecessor changed")
    terminal, state, frozen, intent = (
        read(path)
        for path in (
            root / TERMINAL,
            state_path,
            binding_path,
            helpers.parent / "controller.json",
        )
    )
    result = terminal["comparison"]
    if (
        terminal["status"] != "start_accuracy_terminal_independently_reconstructed_exact_restoration_verified"
        or terminal["binding_sha256"] != BINDING_SHA256
        or terminal["terminal_time"] != state["time"]
        or state["status"] != "diagnostic_complete"
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
        or state["controller_pid"] != terminal["controller_pid_absent"]
        or terminal["worker_pids_absent"] != [entry["pid"] for entry in state["executions"]]
        or terminal["verifier_sha256"] != frozen["helpers_sha256"]["root_start_gpu_controller.py"]
        or intent["status"] != "acknowledged"
        or intent["pid"] != state["controller_pid"]
        or intent["start_ticks"] != state["controller_start_ticks"]
        or intent["binding_sha256"] != BINDING_SHA256
        or not same_keeper(terminal["keeper"], state["restored_keeper"])
        or any(
            result[name] is not True
            for name in (
                "all_cross_process_exact_finite",
                "all_repeat_outputs_exact_finite",
                "all_unaffected_outputs_exact_finite",
                "all_rational_O_within_fp64_tolerance",
            )
        )
        or any(
            sha256(root / name) != value
            for name, value in terminal["evidence_sha256"].items()
            if not name.endswith(".pt")
        )
    ):
        raise RuntimeError("Rational continuation predecessor lacks accepted accuracy and restoration")
    require_keeper_shape(terminal["keeper"])
    paths[str((helpers.parent / "controller.json").relative_to(root))] = sha256(helpers.parent / "controller.json")
    return {
        "status": "completed_start_accuracy_original_owner_absent_pending_exact_live_handoff",
        "evidence_sha256": paths,
        "restored_keeper": terminal["keeper"],
        "controller_pid": state["controller_pid"],
        "controller_start_ticks": state["controller_start_ticks"],
        "controller_marker": intent["marker"],
        "workers": state["executions"],
    }


def require_absent(predecessor: dict) -> None:
    if handoff.observed(predecessor["controller_pid"]) is not None or any(
        handoff.observed(entry["pid"]) is not None or handoff.marker_processes(previous.CHILD_MARKER, entry["marker"])
        for entry in predecessor["workers"]
    ):
        raise RuntimeError("Rational continuation predecessor still has original live descendants")
    marked = handoff.marker_processes(previous.CONTROLLER_MARKER, predecessor["controller_marker"])
    if not set(marked).issubset({int(pid) for pid in predecessor["restored_keeper"]["generation"]}):
        raise RuntimeError("Rational continuation predecessor has nonkeeper controller descendants")


def live(root: Path, expected: dict) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "0,1,2,3,4,5,6,7") or evidence(root) != expected:
        raise RuntimeError("Rational continuation predecessor source or explicit environment differs")
    require_absent(expected)
    actual = handoff.keeper_observation()
    require_keeper_shape(actual)
    if not same_keeper(actual, expected["restored_keeper"]):
        raise RuntimeError("Rational continuation exact restored predecessor keeper changed")
    return actual
