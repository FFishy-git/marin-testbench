# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist compact original-terminal evidence at the external GPU/CPU-receipt boundary."""

import tarfile
from pathlib import Path

import ns_lr_full_model_predecessor as predecessor
import ns_lr_full_model_test_support as numerical
from fused_attribution_validation_test_support import keeper
from selected_lr_release_owner import candidate_files, read, sha256


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


def completed(tmp_path: Path, monkeypatch) -> tuple[Path, dict, dict, dict, dict, dict]:
    root, candidate, _ = numerical.protocol_candidate(tmp_path)
    binding, helpers, state_path, work, result_path = predecessor.contract.locations(root)
    helpers.mkdir(parents=True)
    work.mkdir(parents=True)
    record = numerical.record
    prepared_path = candidate / predecessor.owner.source.PREPARATION
    record(prepared_path, {"status": "CPU external receipt fixture"})
    helper = helpers / "ns_full_model_gpu_controller.py"
    helper.write_text("# Original archived CPU fixture controller\n")
    helper_hashes = {helper.name: sha256(helper)}
    files = candidate_files(candidate)
    inventory = files | {f"tracking/{helper.name}": sha256(helper)}
    archive = root / "run-preparation/original-full-model-source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in inventory:
            path = helper if name.startswith("tracking/") else candidate / name
            bundle.add(path, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    owner_cpu = {
        "candidate": str(candidate),
        "helpers_sha256": helper_hashes,
        "files_sha256": files,
        "preparation_sha256": sha256(prepared_path),
        "inventory_sha256": inventory,
    }
    record(root / predecessor.queue.PARENT, owner_cpu)
    monkeypatch.setattr(predecessor.queue, "PARENT_SHA256", sha256(root / predecessor.queue.PARENT))
    cpu = {
        "candidate": str(candidate),
        "helpers_sha256": {predecessor.queue.NEW_HELPERS[0]: "original-accepted-queue-fixture"},
        "time": "2026-11-19T00:00:00+00:00",
    }
    record(root / predecessor.reporting.CPU, cpu)
    monkeypatch.setattr(predecessor.reporting, "CPU_SHA256", sha256(root / predecessor.reporting.CPU))
    restored = keeper()
    previous = {"status": "original NS terminal fixture", "restored_keeper": restored}
    planned = predecessor.validation.plan(root, candidate)
    frozen = {
        "status": predecessor.contract.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "helpers_sha256": helper_hashes,
        "files_sha256": files,
        "cpu_receipt": str(root / predecessor.queue.PARENT),
        "cpu_receipt_sha256": predecessor.queue.PARENT_SHA256,
        "preparation_sha256": sha256(prepared_path),
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "environment": predecessor.contract.ENVIRONMENT,
        "node": predecessor.owner.NODE,
        "allocation_predecessor": previous,
        "devices": restored["devices"],
        "commands": planned["commands"],
        "plan": planned,
        "controller_command": predecessor.owner.controller_command(root),
        "validator_command": predecessor.owner.validator_command(root, candidate),
        "validation_timeout": predecessor.owner.TIMEOUT,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
    }
    record(binding, frozen)
    queue_work = root / predecessor.queue.WORK
    queue_work.mkdir(parents=True)
    queue_command = predecessor.queue.command(root, root / predecessor.reporting.CPU)
    intent = {
        "status": "acknowledged",
        "pid": 700001,
        "start_ticks": "700101",
        "command": queue_command,
        "generation": generation(700001, 1, queue_command),
        "marker": "original-cpu-queue-fixture",
        "cpu_receipt": str(root / predecessor.reporting.CPU),
        "cpu_receipt_sha256": predecessor.reporting.CPU_SHA256,
        "parent_cpu_receipt_sha256": predecessor.queue.PARENT_SHA256,
        "runner_sha256": cpu["helpers_sha256"][predecessor.queue.NEW_HELPERS[0]],
        "interval": predecessor.queue.INTERVAL,
    }
    record(queue_work / "intent.json", intent)
    monkeypatch.setattr(predecessor.reporting, "INTENT_SHA256", sha256(queue_work / "intent.json"))
    owner_intent = {
        "status": "acknowledged",
        "pid": 700002,
        "start_ticks": "700102",
        "command": frozen["controller_command"],
        "generation": generation(700002, 1, frozen["controller_command"]),
        "marker": "original-cpu-controller-fixture",
        "binding_sha256": sha256(binding),
        "time": "2026-11-19T00:00:01+00:00",
    }
    record(helpers.parent / "controller.json", owner_intent)
    for name in ("controller.log", "validator.log"):
        (helpers.parent / name).write_text("Original completed external CPU fixture\n")
    entries, decisions = [], []
    for index, item in enumerate(planned["trials"]):
        measured = item | {
            "accepted": index != 0,
            "gates": {
                "actual_config_complete_trajectories": True,
                "boundary_all_rank_continuation": True,
                "post_cached_root_all_rank_continuation": index != 0,
                "scalar_loss_continuation": True,
                "captured_row_update_work": True,
            },
            "trajectories": {mode: {"fixture": mode, "trial": item["name"]} for mode in predecessor.contract.MODES},
        }
        decisions.append(measured)
        for offset, (mode, command) in enumerate(zip(predecessor.contract.MODES, item["commands"], strict=True)):
            log = work / f"{item['name']}_{mode}.log"
            log.write_text("Original retained external command fixture\n")
            pid = 710000 + index * 3 + offset
            entry = {
                "command": command,
                "generation": generation(pid, 700003, command),
                "marker": f"original-command-{pid}",
                "spawn_status": "acknowledged",
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "marked_descendants_absent": True,
                "log": str(log),
                "log_sha256": sha256(log),
                "generations": {},
            }
            record(log.with_suffix(".execution.json"), entry)
            entries.append(entry | {"trial": item["name"], "mode": mode, "trajectory": measured["trajectories"][mode]})
    summary = {
        "accepted_cases": [item["name"] for item in decisions if item["accepted"]],
        "rejected_cases": [item["name"] for item in decisions if not item["accepted"]],
        "all_cases_accepted": False,
    }
    result = planned | {
        "status": predecessor.validation.STATUS,
        "binding_sha256": sha256(binding),
        "owner": {
            "controller_pid": 700002,
            "controller_start_ticks": "700102",
            "validator_pid": 700003,
            "validator_start_ticks": "700103",
        },
        "gpu_commands": entries,
        "results": decisions,
        **summary,
        "time": "2026-11-19T00:00:02+00:00",
    }
    record(result_path, result)
    state = {
        "status": "validation_complete",
        "binding_sha256": sha256(binding),
        "controller_pid": 700002,
        "controller_start_ticks": "700102",
        "child_pid": 700003,
        "child_start_ticks": "700103",
        "child_generation": generation(700003, 700002, frozen["validator_command"]),
        "child_marker": "original-cpu-validator-fixture",
        "command": frozen["validator_command"],
        "allocation_predecessor": previous,
        "spawn_status": "acknowledged",
        "keeper_release_attempted": True,
        "keeper_release_complete": True,
        "marked_descendants_absent": True,
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "restored_keeper": restored,
        "log": str(helpers.parent / "validator.log"),
        "log_sha256": sha256(helpers.parent / "validator.log"),
        "result_sha256": sha256(result_path),
        **summary,
        "time": "2026-11-19T00:00:03+00:00",
    }
    record(state_path, state)
    terminal_paths = (
        binding,
        state_path,
        helpers.parent / "controller.json",
        helpers.parent / "controller.log",
        helpers.parent / "validator.log",
        result_path,
    )
    terminal = {
        "status": "ns_full_model_terminal_independently_reconstructed_exact_restoration",
        "binding_sha256": sha256(binding),
        "result_sha256": sha256(result_path),
        "terminal_time": state["time"],
        "controller_pid_absent": 700002,
        "validator_pid_absent": 700003,
        "verifier_sha256": sha256(helper),
        "keeper": restored,
        "evidence_sha256": {str(path.relative_to(root)): sha256(path) for path in terminal_paths},
        **summary,
        "time": "2026-11-19T00:00:04+00:00",
    }
    record(root / predecessor.queue.TERMINAL, terminal)
    phases = []
    commands = predecessor.queue.phases(root)
    commands.append(
        (
            "terminal",
            [
                predecessor.contract.PYTHON,
                "-P",
                str(helpers / "ns_full_model_gpu_controller.py"),
                "--root",
                str(root),
                "--phase",
                "terminal",
                "--out",
                str(root / predecessor.queue.TERMINAL),
            ],
            helpers,
        )
    )
    for index, (phase, command, cwd) in enumerate(commands):
        log = queue_work / f"{phase}.log"
        log.write_text("Original completed queue phase fixture\n")
        pid = 720000 + index
        phases.append(
            {
                "phase": phase,
                "command": command,
                "cwd": str(cwd),
                "marker": f"original-queue-phase-{pid}",
                "spawn_status": "acknowledged",
                "pid": pid,
                "start_ticks": str(pid + 100),
                "generation": generation(pid, intent["pid"], command),
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "marked_descendants_absent": True,
                "log": str(log),
                "log_sha256": sha256(log),
            }
        )
    (queue_work / "queue.log").write_text("Original completed queue fixture\n")
    queue_state = {
        "status": "full_model_terminal_reconstructed",
        "pid": intent["pid"],
        "start_ticks": intent["start_ticks"],
        **{
            name: intent[name]
            for name in ("cpu_receipt_sha256", "parent_cpu_receipt_sha256", "runner_sha256", "interval")
        },
        "phases": phases,
        "issued_controller": owner_intent,
        "issued_binding_sha256": sha256(binding),
        "terminal": str(root / predecessor.queue.TERMINAL),
        "terminal_sha256": sha256(root / predecessor.queue.TERMINAL),
        **summary,
        "time": "2026-11-19T00:00:05+00:00",
    }
    record(queue_work / "state.json", queue_state)
    return root, cpu, intent, queue_state, terminal, previous


def rebind_terminal(inputs: tuple) -> None:
    root, _, _, queue_state, terminal, _ = inputs
    binding, helpers, state_path, _, result_path = predecessor.contract.locations(root)
    state = read(state_path)
    state["result_sha256"] = sha256(result_path)
    numerical.record(state_path, state)
    terminal["result_sha256"] = sha256(result_path)
    terminal["evidence_sha256"] = {
        str(path.relative_to(root)): sha256(path)
        for path in (
            binding,
            state_path,
            helpers.parent / "controller.json",
            helpers.parent / "controller.log",
            helpers.parent / "validator.log",
            result_path,
        )
    }
    numerical.record(root / predecessor.queue.TERMINAL, terminal)
    queue_state["terminal_sha256"] = sha256(root / predecessor.queue.TERMINAL)
    numerical.record(root / predecessor.queue.WORK / "state.json", queue_state)
