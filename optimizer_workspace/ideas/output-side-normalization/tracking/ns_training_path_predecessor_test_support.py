# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist an original diagnostic terminal chain for CPU evidence-path tests."""

import copy
import shutil
import tarfile
from pathlib import Path

import ns_training_path_predecessor as predecessor
from fused_attribution_validation_test_support import keeper, record
from selected_lr_release_owner import read, sha256


def generation(pid: int, parent: int, command: list[str]) -> dict:
    return {
        "pid": pid,
        "parent": parent,
        "command": command,
        "start_ticks": str(pid + 100),
        "group": pid,
        "session": pid,
        "status": "S",
    }


def completed(root: Path, receipts: list[dict], comparison: dict) -> dict:
    """Write source, queue, worker and independent terminal fixtures without numerical execution."""
    binding, helpers, state_path, work = predecessor.diagnostic.locations(root)
    helpers.mkdir(parents=True)
    work.mkdir(parents=True)
    owner_receipt = copy.deepcopy(receipts[0])
    original = Path(owner_receipt["helpers"]).parent
    owner_directory = root / "cpu-owner"
    shutil.copytree(original, owner_directory)
    original_helpers = owner_directory / "tracking"
    owner_receipt["helpers"] = str(original_helpers)
    for name in ("archive", "tests", "log"):
        owner_receipt[name] = str(owner_directory / Path(owner_receipt[name]).name)
    controller = original_helpers / "ns_budget_gpu_controller.py"
    controller.write_text("# original accepted diagnostic controller\n")
    owner_receipt["helpers_sha256"][controller.name] = sha256(controller)
    archive = Path(owner_receipt["archive"])
    archive.unlink()
    with tarfile.open(archive, "x") as stream:
        for name in owner_receipt["helpers_sha256"]:
            stream.add(original_helpers / name, arcname=f"tracking/{name}", recursive=False)
    owner_receipt["archive_sha256"] = sha256(archive)
    owner_path = original_helpers.parent / "receipt.json"
    record(owner_path, owner_receipt)
    for name in owner_receipt["helpers_sha256"]:
        (helpers / name).write_bytes((original_helpers / name).read_bytes())
    frozen_archive = root / "run-preparation/original-ns-diagnostic.tar"
    with tarfile.open(frozen_archive, "x") as stream:
        for name in owner_receipt["helpers_sha256"]:
            stream.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    restored = keeper()
    frozen = {
        "status": predecessor.diagnostic.STATUS,
        "root": str(root),
        "helpers": str(helpers),
        "work": str(work),
        "node": predecessor.diagnostic.NODE,
        "environment": predecessor.diagnostic.ENVIRONMENT,
        "cpu_receipt": str(owner_path),
        "cpu_receipt_sha256": sha256(owner_path),
        "helpers_sha256": owner_receipt["helpers_sha256"],
        "archive": str(frozen_archive),
        "archive_sha256": sha256(frozen_archive),
        "devices": restored["devices"],
        "predecessor": {"status": "retained original completed parent"},
    }
    record(binding, frozen)
    intent = {
        "status": "acknowledged",
        "pid": 820001,
        "start_ticks": str(820101),
        "command": predecessor.diagnostic.controller_command(root),
        "marker": "original-diagnostic-controller",
        "binding_sha256": sha256(binding),
    }
    intent["generation"] = generation(intent["pid"], 1, intent["command"])
    record(helpers.parent / "controller.json", intent)
    (helpers.parent / "controller.log").write_text("Original completed controller\n")
    entries = []
    for index in range(2):
        log, report, output = (work / f"process-{index}{suffix}" for suffix in (".log", ".json", ".pt"))
        log.write_text("Original completed diagnostic worker\n")
        record(report, {"worker_index": index, "status": "retained original output"})
        output.write_bytes(f"original retained payload {index}".encode())
        command = predecessor.diagnostic.worker_command(root, sha256(binding), index)
        pid = 820010 + index
        own = generation(pid, intent["pid"], command)
        entries.append(
            {
                "index": index,
                "command": command,
                "marker": f"original-worker-{index}",
                "spawn_status": "acknowledged",
                "controller_pid": intent["pid"],
                "controller_start_ticks": intent["start_ticks"],
                "pid": pid,
                "start_ticks": own["start_ticks"],
                "generation": own,
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "marked_descendants_absent": True,
                "log": str(log),
                "log_sha256": sha256(log),
                "report": str(report),
                "report_sha256": sha256(report),
            }
        )
    comparison = copy.deepcopy(comparison)
    comparison["time"] = "2026-11-19T00:00:00+00:00"
    record(work / "comparison.json", comparison)
    state = {
        "status": "diagnostic_complete",
        "controller_pid": intent["pid"],
        "controller_start_ticks": intent["start_ticks"],
        "binding_sha256": sha256(binding),
        "comparison_sha256": sha256(work / "comparison.json"),
        "keeper_release_attempted": True,
        "keeper_release_complete": True,
        "keeper_restore": 0,
        "restored_keeper": restored,
        "predecessor": frozen["predecessor"],
        "executions": entries,
        "time": "2026-11-19T00:00:01+00:00",
    }
    record(state_path, state)
    terminal = {
        "status": "ns_budget_terminal_independently_reconstructed_exact_restoration_verified",
        "binding_sha256": sha256(binding),
        "terminal_time": state["time"],
        "controller_pid_absent": intent["pid"],
        "worker_pids_absent": [entry["pid"] for entry in entries],
        "comparison": {key: value for key, value in comparison.items() if key != "time"},
        "verifier_sha256": frozen["helpers_sha256"]["ns_budget_gpu_controller.py"],
        "keeper": restored,
        "evidence_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in (
                binding,
                state_path,
                helpers.parent / "controller.json",
                helpers.parent / "controller.log",
                work / "comparison.json",
                *(work / f"process-{index}{suffix}" for index in range(2) for suffix in (".log", ".json", ".pt")),
            )
        },
        "time": "2026-11-19T00:00:02+00:00",
    }
    record(root / predecessor.TERMINAL, terminal)
    queue_path = root / predecessor.NS_QUEUE
    queue_intent, queue_state = read(queue_path / "intent.json"), read(queue_path / "state.json")
    phases = []
    for index, phase in enumerate(("freeze", "check", "start", "terminal")):
        command = [
            predecessor.diagnostic.PYTHON,
            "-P",
            str((original_helpers if phase == "freeze" else helpers) / "ns_budget_gpu_controller.py"),
            "--root",
            str(root),
            "--phase",
            phase,
        ]
        if phase == "freeze":
            command += ["--cpu-receipt", str(owner_path)]
        elif phase == "terminal":
            command += ["--out", str(root / predecessor.TERMINAL)]
        log = queue_path / f"{phase}.log"
        log.write_text(f"Original completed {phase}\n")
        pid = 820020 + index
        own = generation(pid, queue_intent["pid"], command)
        phases.append(
            {
                "phase": phase,
                "command": command,
                "spawn_status": "acknowledged",
                "pid": pid,
                "start_ticks": own["start_ticks"],
                "generation": own,
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "marked_descendants_absent": True,
                "log": str(log),
                "log_sha256": sha256(log),
            }
        )
    queue_state.update(
        phases=phases,
        terminal_sha256=sha256(root / predecessor.TERMINAL),
        issued_binding_sha256=sha256(binding),
        time="2026-11-19T00:00:03+00:00",
    )
    record(queue_path / "state.json", queue_state)
    for relative in (predecessor.NS_QUEUE, predecessor.LOWER_QUEUE, predecessor.ACTUAL_QUEUE):
        (root / relative / "queue.log").write_text("Original completed queue log\n")
    return {
        "owner_spec": (
            str(owner_path.relative_to(root)),
            sha256(owner_path),
            owner_receipt["status"],
            predecessor.ARCHIVES[0][3],
        ),
        "terminal": terminal,
        "state": state,
        "restored_keeper": restored,
    }
