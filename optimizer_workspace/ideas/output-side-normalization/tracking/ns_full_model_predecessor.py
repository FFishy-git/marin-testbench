# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require original NS diagnostic reconstruction, queue exit and exact restoration."""

from datetime import datetime
from pathlib import Path

import ns_budget_gpu_owner as previous
import owned_output_handoff as handoff
from fused_attribution_validation import require_keeper_shape
from fused_attribution_validation_controller import require_helper_bytes
from prepare_selected_lr_validation import require_archive
from root_start_gpu_owner import require_cpu_archive
from selected_lr_failure_handoff import same_keeper
from selected_lr_release_owner import read, sha256

TERMINAL = "run-preparation/ns-budget-terminal-independent-20261119.json"
QUEUE = "run-preparation/ns-budget-queue-20261119"
QUEUE_CPU = "run-preparation/ns-budget-queue-cpu/20261119T0024/receipt.json"
QUEUE_CPU_SHA256 = "b268b833c4d79187984dffa29686e8533a72bfeac74162c87d9571eb6a441796"
QUEUE_INTENT_SHA256 = "1a0660e164fc069300c90e8b058138ea828b3b9f7377a0da4117278248240e05"
QUEUE_MARKER = "OUTPUT_MARIN_NS_BUDGET_QUEUE"
OWNER_CPU = "run-preparation/ns-budget-gpu-owner-cpu/20261119T0003/receipt.json"
OWNER_CPU_SHA256 = "18e0b21734ebdebca8934319753fc6dbed37e5e565c5e28df5d0d1f453f81c69"


def absent(intent: dict, state: dict) -> None:
    if handoff.observed(intent["pid"]) is not None or handoff.marker_processes(QUEUE_MARKER, intent["marker"]):
        raise RuntimeError("NS full-model predecessor original queue remains present")
    for entry in state.get("phases", []):
        if handoff.observed(entry["pid"]) is not None or handoff.marker_processes(
            previous.CONTROLLER_MARKER, entry["marker"]
        ):
            raise RuntimeError("NS full-model predecessor original queue phase remains present")


def evidence(root: Path) -> dict:
    queue_path = root / QUEUE
    intent, queue_state = read(queue_path / "intent.json"), read(queue_path / "state.json")
    absent(intent, queue_state)
    if (
        sha256(queue_path / "intent.json") != QUEUE_INTENT_SHA256
        or sha256(root / QUEUE_CPU) != QUEUE_CPU_SHA256
        or sha256(root / OWNER_CPU) != OWNER_CPU_SHA256
    ):
        raise ValueError("NS full-model predecessor accepted queue/owner identity changed")
    queue_cpu = require_cpu_archive(root, root / QUEUE_CPU, "ns_budget_queue_cpu_archive_verified_gpu_unissued", 21)
    owner_cpu = require_cpu_archive(root, root / OWNER_CPU, previous.CPU_STATUS, 75)
    binding, helpers, state_path, work = previous.locations(root)
    frozen, state, owner_intent, terminal, comparison = (
        read(path)
        for path in (binding, state_path, helpers.parent / "controller.json", root / TERMINAL, work / "comparison.json")
    )
    if handoff.observed(state["controller_pid"]) is not None or any(
        handoff.observed(entry["pid"]) is not None or handoff.marker_processes(previous.CHILD_MARKER, entry["marker"])
        for entry in state["executions"]
    ):
        raise RuntimeError("NS full-model predecessor original diagnostic owner remains present")
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    require_archive(
        Path(frozen["archive"]),
        {f"tracking/{name}": digest for name, digest in frozen["helpers_sha256"].items()},
        alias=False,
    )
    phases = queue_state["phases"]
    expected_queue_command = [
        previous.PYTHON,
        "-P",
        str(Path(queue_cpu["helpers"]) / "ns_budget_queue.py"),
        "--root",
        str(root),
        "--cpu-receipt",
        str(root / QUEUE_CPU),
        "--phase",
        "execute",
    ]
    expected_commands = {
        "freeze": [
            previous.PYTHON,
            "-P",
            str(Path(owner_cpu["helpers"]) / "ns_budget_gpu_controller.py"),
            "--root",
            str(root),
            "--phase",
            "freeze",
            "--cpu-receipt",
            str(root / OWNER_CPU),
        ],
        "check": [
            previous.PYTHON,
            "-P",
            str(helpers / "ns_budget_gpu_controller.py"),
            "--root",
            str(root),
            "--phase",
            "check",
        ],
        "start": [
            previous.PYTHON,
            "-P",
            str(helpers / "ns_budget_gpu_controller.py"),
            "--root",
            str(root),
            "--phase",
            "start",
        ],
        "terminal": [
            previous.PYTHON,
            "-P",
            str(helpers / "ns_budget_gpu_controller.py"),
            "--root",
            str(root),
            "--phase",
            "terminal",
            "--out",
            str(root / TERMINAL),
        ],
    }
    if (
        queue_state["status"] != "ns_diagnostic_terminal_reconstructed"
        or intent["status"] != "acknowledged"
        or intent["pid"] != queue_state["pid"]
        or intent["start_ticks"] != queue_state["start_ticks"]
        or intent["generation"]["pid"] != intent["pid"]
        or intent["generation"]["start_ticks"] != intent["start_ticks"]
        or intent["generation"]["command"] != intent["command"]
        or intent["generation"]["group"] != intent["pid"]
        or intent["generation"]["session"] != intent["pid"]
        or intent["command"] != expected_queue_command
        or intent["parent_cpu_receipt_sha256"] != OWNER_CPU_SHA256
        or queue_state["parent_cpu_receipt_sha256"] != OWNER_CPU_SHA256
        or intent["runner_sha256"] != queue_cpu["helpers_sha256"]["ns_budget_queue.py"]
        or queue_state["runner_sha256"] != intent["runner_sha256"]
        or intent["cpu_receipt_sha256"] != QUEUE_CPU_SHA256
        or queue_state["cpu_receipt_sha256"] != QUEUE_CPU_SHA256
        or intent["interval"] != 900
        or queue_state["interval"] != 900
        or [entry["phase"] for entry in phases] != list(expected_commands)
        or any(key in queue_state for key in ("error", "cleanup_error"))
        or frozen["status"] != previous.STATUS
        or frozen["root"] != str(root)
        or frozen["node"] != previous.NODE
        or frozen["cpu_receipt"] != str(root / OWNER_CPU)
        or frozen["cpu_receipt_sha256"] != OWNER_CPU_SHA256
        or frozen["helpers_sha256"] != owner_cpu["helpers_sha256"]
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or state["status"] != "diagnostic_complete"
        or state["binding_sha256"] != sha256(binding)
        or state["predecessor"] != frozen["predecessor"]
        or state["controller_pid"] != owner_intent["pid"]
        or state["controller_start_ticks"] != owner_intent["start_ticks"]
        or owner_intent["status"] != "acknowledged"
        or owner_intent["command"] != previous.controller_command(root)
        or owner_intent["binding_sha256"] != sha256(binding)
        or owner_intent["generation"]["pid"] != owner_intent["pid"]
        or owner_intent["generation"]["start_ticks"] != owner_intent["start_ticks"]
        or owner_intent["generation"]["command"] != owner_intent["command"]
        or owner_intent["generation"]["group"] != owner_intent["pid"]
        or owner_intent["generation"]["session"] != owner_intent["pid"]
        or state["keeper_release_attempted"] is not True
        or state["keeper_release_complete"] is not True
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
        or len(state["executions"]) != 2
        or state["active_worker"] != state["executions"][-1]
        or terminal["status"] != "ns_budget_terminal_independently_reconstructed_exact_restoration_verified"
        or terminal["binding_sha256"] != sha256(binding)
        or terminal["terminal_time"] != state["time"]
        or terminal["controller_pid_absent"] != state["controller_pid"]
        or terminal["worker_pids_absent"] != [entry["pid"] for entry in state["executions"]]
        or terminal["verifier_sha256"] != frozen["helpers_sha256"]["ns_budget_gpu_controller.py"]
        or terminal["comparison"] != {key: value for key, value in comparison.items() if key != "time"}
        or state["comparison_sha256"] != sha256(work / "comparison.json")
        or queue_state["terminal_sha256"] != sha256(root / TERMINAL)
        or queue_state["issued_binding_sha256"] != sha256(binding)
        or queue_state["historical_controls_interpretable"] is not True
        or not same_keeper(terminal["keeper"], state["restored_keeper"])
        or terminal["keeper"]["devices"] != frozen["devices"]
        or terminal["comparison"]["historical_controls_interpretable"] is not True
    ):
        raise RuntimeError("NS full-model predecessor terminal reconstruction/control/restoration is incomplete")
    for entry in phases:
        generation, phase = entry["generation"], entry["phase"]
        log = queue_path / f"{phase}.log"
        if (
            entry["command"] != expected_commands[phase]
            or entry["spawn_status"] != "acknowledged"
            or generation["command"] != entry["command"]
            or generation["pid"] != entry["pid"]
            or generation["start_ticks"] != entry["start_ticks"]
            or generation["parent"] != intent["pid"]
            or generation["group"] != entry["pid"]
            or generation["session"] != entry["pid"]
            or any(type(entry[key]) is not int or entry[key] != 0 for key in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
            or entry["log"] != str(log)
            or entry["log_sha256"] != sha256(log)
        ):
            raise ValueError("NS full-model original queue command/ancestry/cleanup differs")
    for index, entry in enumerate(state["executions"]):
        generation = entry["generation"]
        if (
            entry["index"] != index
            or entry["command"] != previous.worker_command(root, sha256(binding), index)
            or generation["command"] != entry["command"]
            or generation["pid"] != entry["pid"]
            or generation["start_ticks"] != entry["start_ticks"]
            or generation["parent"] != state["controller_pid"]
            or entry["controller_pid"] != state["controller_pid"]
            or entry["controller_start_ticks"] != state["controller_start_ticks"]
            or generation["group"] != entry["pid"]
            or generation["session"] != entry["pid"]
            or any(type(entry[key]) is not int or entry[key] != 0 for key in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
            or entry["log_sha256"] != sha256(Path(entry["log"]))
            or entry["report_sha256"] != sha256(Path(entry["report"]))
        ):
            raise ValueError("NS full-model original diagnostic execution/cleanup differs")
    require_keeper_shape(terminal["keeper"])
    marked = handoff.marker_processes(previous.CONTROLLER_MARKER, owner_intent["marker"])
    if not set(marked).issubset({int(pid) for pid in terminal["keeper"]["generation"]}):
        raise RuntimeError("NS full-model predecessor has nonkeeper controller descendants")
    times = [datetime.fromisoformat(value["time"]) for value in (comparison, state, terminal, queue_state)]
    if any(value.tzinfo is None for value in times) or times != sorted(times):
        raise ValueError("NS full-model predecessor completion chronology differs")
    paths = [
        binding,
        state_path,
        helpers.parent / "controller.json",
        helpers.parent / "controller.log",
        work / "comparison.json",
        root / TERMINAL,
        queue_path / "intent.json",
        queue_path / "state.json",
        queue_path / "queue.log",
        *(queue_path / f"{phase}.log" for phase in expected_commands),
    ]
    hashes = terminal["evidence_sha256"] | {str(path.relative_to(root)): sha256(path) for path in paths}
    if any(sha256(root / name) != digest for name, digest in hashes.items()):
        raise ValueError("NS full-model predecessor canonical evidence changed")
    absent(intent, queue_state)
    return {
        "status": "original_ns_terminal_queue_absent_pending_exact_full_model_handoff",
        "evidence_sha256": hashes,
        "restored_keeper": terminal["keeper"],
        "controller_marker": owner_intent["marker"],
        "historical_controls_interpretable": True,
    }


def live(root: Path, expected: dict) -> dict:
    if evidence(root) != expected:
        raise RuntimeError("NS full-model predecessor evidence changed before transfer")
    keeper = handoff.keeper_observation()
    require_keeper_shape(keeper)
    if not same_keeper(keeper, expected["restored_keeper"]):
        raise RuntimeError("NS full-model exact restored predecessor keeper generation differs")
    return keeper
