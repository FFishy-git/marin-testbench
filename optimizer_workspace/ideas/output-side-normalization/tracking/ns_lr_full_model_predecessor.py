# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate the original lower-LR terminal handoff without tensor replay or launch."""

from datetime import datetime
from pathlib import Path

import ns_full_model_contract as contract
import ns_full_model_gpu_owner as owner
import ns_full_model_predecessor as previous
import ns_full_model_queue as queue
import ns_full_model_queue_reporting as reporting
import ns_full_model_validation as validation
import owned_output_handoff as handoff
from fused_attribution_validation import require_keeper_shape
from fused_attribution_validation_controller import require_helper_bytes
from ns_budget_queue import handle
from prepare_selected_lr_validation import require_archive
from selected_lr_failure_handoff import same_keeper
from selected_lr_release_owner import candidate_files, read, sha256

STATUS = "original_lower_lr_full_model_terminal_queue_absent_actual_lr_unissued"
SCOPE = (
    "Read-only original terminal/source/command/cleanup/restoration handoff. Trust the exact "
    "original independently reconstructed terminal receipt, not a new tensor reconstruction. "
    "Keep every lower-LR rejection. Lower-LR acceptance grants no actual-LR, changed-owner, "
    "production, full-budget quality, elapsed-efficiency or private SwanLab authority."
)


def readiness(root: Path) -> dict:
    """Observe the exact original queue generation before expensive archive authentication."""
    work = root / queue.WORK
    paths = (work / "intent.json", work / "state.json")
    hashes = {path: sha256(path) for path in paths}
    if hashes[paths[0]] != reporting.INTENT_SHA256:
        raise ValueError("NS actual-LR predecessor original full-model queue acknowledgment changed")
    intent, state = (read(path) for path in paths)
    owners = {"original_queue": handle(intent["generation"], work, queue.MARKER, intent["marker"], queue.ENVIRONMENT)}
    marked = handoff.marker_processes(queue.MARKER, intent["marker"])
    if marked:
        owners["queue_marked_descendants"] = {"status": "marked_descendants_present", "pids": sorted(marked)}
    phases = list(state.get("phases", []))
    active = state.get("active_phase", {})
    if active.get("spawn_status") == "acknowledged" and active not in phases:
        phases.append(active)
    for entry in phases:
        owners[f"phase_{entry['phase']}"] = handle(
            entry["generation"], Path(entry["cwd"]), queue.PHASE_MARKER, entry["marker"], queue.ENVIRONMENT
        )
        marked = handoff.marker_processes(queue.PHASE_MARKER, entry["marker"])
        if marked:
            owners[f"phase_{entry['phase']}_descendants"] = {
                "status": "marked_descendants_present",
                "pids": sorted(marked),
            }
    if hashes != {path: sha256(path) for path in paths}:
        raise RuntimeError("NS actual-LR predecessor original queue changed during observation")
    return {
        "status": (
            "original_full_model_queue_absent"
            if all(item["status"] == "absent" for item in owners.values())
            else "waiting_for_original_full_model_queue"
        ),
        "ready": all(item["status"] == "absent" for item in owners.values()),
        "owners": owners,
    }


def retained(root: Path, cpu: dict, intent: dict, queue_state: dict, terminal: dict, predecessor: dict) -> dict:
    """Bind the original terminal to its exact source, command records and lower-LR decisions."""
    binding, helpers, state_path, work, result_path = contract.locations(root)
    paths = [
        binding,
        state_path,
        helpers.parent / "controller.json",
        helpers.parent / "controller.log",
        helpers.parent / "validator.log",
        result_path,
        root / reporting.CPU,
        root / queue.PARENT,
        root / queue.WORK / "intent.json",
        root / queue.WORK / "state.json",
        root / queue.WORK / "queue.log",
        root / queue.TERMINAL,
        *(root / queue.WORK / f"{phase}.log" for phase in ("freeze", "check", "start", "terminal")),
    ]
    if root != root.resolve() or any(
        path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 for path in paths
    ):
        raise ValueError("NS actual-LR predecessor requires private canonical original evidence")
    hashes = {str(path.relative_to(root)): sha256(path) for path in paths}
    frozen, state, owner_intent, result, owner_cpu = (
        read(path)
        for path in (binding, state_path, helpers.parent / "controller.json", result_path, root / queue.PARENT)
    )
    candidate = Path(frozen["candidate"])
    expected = validation.plan(root, candidate)
    child, controller = state["child_generation"], owner_intent["generation"]
    if (
        hashes[reporting.CPU] != reporting.CPU_SHA256
        or hashes[queue.PARENT] != queue.PARENT_SHA256
        or hashes[f"{queue.WORK}/intent.json"] != reporting.INTENT_SHA256
        or queue_state["status"] != "full_model_terminal_reconstructed"
        or len(queue_state["phases"]) != 4
        or queue_state["issued_controller"] != owner_intent
        or queue_state["issued_binding_sha256"] != sha256(binding)
        or queue_state["terminal"] != str(root / queue.TERMINAL)
        or queue_state["terminal_sha256"] != sha256(root / queue.TERMINAL)
        or queue_state["pid"] != intent["pid"]
        or queue_state["start_ticks"] != intent["start_ticks"]
        or frozen["status"] != contract.FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != cpu["candidate"]
        or frozen["candidate"] != owner_cpu["candidate"]
        or frozen["helpers"] != str(helpers)
        or frozen["helpers_sha256"] != owner_cpu["helpers_sha256"]
        or frozen["files_sha256"] != candidate_files(candidate)
        or frozen["files_sha256"] != owner_cpu["files_sha256"]
        or frozen["cpu_receipt"] != str(root / queue.PARENT)
        or frozen["cpu_receipt_sha256"] != queue.PARENT_SHA256
        or frozen["preparation_sha256"] != owner_cpu["preparation_sha256"]
        or frozen["preparation_sha256"] != sha256(candidate / owner.source.PREPARATION)
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["environment"] != contract.ENVIRONMENT
        or frozen["node"] != owner.NODE
        or frozen["allocation_predecessor"] != predecessor
        or frozen["devices"] != predecessor["restored_keeper"]["devices"]
        or frozen["commands"] != expected["commands"]
        or frozen["plan"] != expected
        or frozen["controller_command"] != owner.controller_command(root)
        or frozen["validator_command"] != owner.validator_command(root, candidate)
        or frozen["validation_timeout"] != owner.TIMEOUT
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or owner_intent["status"] != "acknowledged"
        or owner_intent["binding_sha256"] != sha256(binding)
        or owner_intent["command"] != owner.controller_command(root)
        or controller["pid"] != owner_intent["pid"]
        or controller["start_ticks"] != owner_intent["start_ticks"]
        or controller["command"] != owner_intent["command"]
        or controller["group"] != controller["pid"]
        or controller["session"] != controller["pid"]
        or state["status"] != "validation_complete"
        or state["binding_sha256"] != sha256(binding)
        or state["controller_pid"] != owner_intent["pid"]
        or state["controller_start_ticks"] != owner_intent["start_ticks"]
        or state["allocation_predecessor"] != predecessor
        or state["spawn_status"] != "acknowledged"
        or child["pid"] != state["child_pid"]
        or child["start_ticks"] != state["child_start_ticks"]
        or child["parent"] != state["controller_pid"]
        or child["command"] != owner.validator_command(root, candidate)
        or child["group"] != child["pid"]
        or child["session"] != child["pid"]
        or state["command"] != child["command"]
        or state["keeper_release_attempted"] is not True
        or state["keeper_release_complete"] is not True
        or state["marked_descendants_absent"] is not True
        or any(
            type(state[key]) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or state["log"] != str(helpers.parent / "validator.log")
        or state["log_sha256"] != sha256(Path(state["log"]))
        or state["result_sha256"] != sha256(result_path)
        or terminal["status"] != "ns_full_model_terminal_independently_reconstructed_exact_restoration"
        or terminal["binding_sha256"] != sha256(binding)
        or terminal["result_sha256"] != sha256(result_path)
        or terminal["terminal_time"] != state["time"]
        or terminal["controller_pid_absent"] != state["controller_pid"]
        or terminal["validator_pid_absent"] != state["child_pid"]
        or terminal["verifier_sha256"] != frozen["helpers_sha256"]["ns_full_model_gpu_controller.py"]
        or terminal["keeper"]["devices"] != frozen["devices"]
        or not same_keeper(terminal["keeper"], state["restored_keeper"])
        or result["status"] != validation.STATUS
        or result["binding_sha256"] != sha256(binding)
        or result["owner"]
        != {
            "controller_pid": state["controller_pid"],
            "controller_start_ticks": state["controller_start_ticks"],
            "validator_pid": state["child_pid"],
            "validator_start_ticks": state["child_start_ticks"],
        }
        or any(result[key] != value for key, value in expected.items())
        or len(result["results"]) != len(expected["trials"])
        or len(result["gpu_commands"]) != len(expected["commands"])
        or any(
            key in value
            for value in (queue_state, state, result)
            for key in ("error", "cleanup_error", "deferred_signals")
        )
    ):
        raise ValueError("NS actual-LR predecessor original source/owner/terminal/restoration differs")
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    require_archive(Path(frozen["archive"]), owner_cpu["inventory_sha256"], alias=True)
    require_keeper_shape(terminal["keeper"])
    measured = result["results"]
    for index, item in enumerate(expected["trials"]):
        saved = measured[index]
        if (
            any(saved[key] != value for key, value in item.items())
            or type(saved["accepted"]) is not bool
            or set(saved["gates"])
            != {
                "actual_config_complete_trajectories",
                "boundary_all_rank_continuation",
                "post_cached_root_all_rank_continuation",
                "scalar_loss_continuation",
                "captured_row_update_work",
            }
            or any(type(value) is not bool for value in saved["gates"].values())
            or saved["accepted"] != all(saved["gates"].values())
        ):
            raise ValueError("NS actual-LR predecessor lower-LR case decisions differ")
        for offset, (mode, command) in enumerate(zip(contract.MODES, item["commands"], strict=True)):
            entry = result["gpu_commands"][index * 3 + offset]
            log = work / f"{item['name']}_{mode}.log"
            validation.require_entry(entry, command, log)
            if (
                entry["trial"] != item["name"]
                or entry["mode"] != mode
                or entry["generation"]["parent"] != state["child_pid"]
                or entry["trajectory"] != saved["trajectories"][mode]
            ):
                raise ValueError("NS actual-LR predecessor retained command ancestry or trajectory differs")
            if handoff.observed(entry["generation"]["pid"]) is not None or handoff.marker_processes(
                contract.COMMAND_MARKER, entry["marker"]
            ):
                raise RuntimeError("NS actual-LR predecessor original training command remains present")
            hashes.update(
                {str(path.relative_to(root)): sha256(path) for path in (log, log.with_suffix(".execution.json"))}
            )
    decisions = {
        "accepted_cases": [item["name"] for item in measured if item["accepted"]],
        "rejected_cases": [item["name"] for item in measured if not item["accepted"]],
        "all_cases_accepted": all(item["accepted"] for item in measured),
    }
    if any(
        value[key] != decision or type(value[key]) is not type(decision)
        for value in (result, state, terminal, queue_state)
        for key, decision in decisions.items()
    ):
        raise ValueError("NS actual-LR predecessor accepted/rejected summaries differ")
    required_terminal_paths = {
        str(path.relative_to(root))
        for path in (
            binding,
            state_path,
            helpers.parent / "controller.json",
            helpers.parent / "controller.log",
            helpers.parent / "validator.log",
            result_path,
        )
    }
    if set(terminal["evidence_sha256"]) != required_terminal_paths or any(
        hashes[name] != digest for name, digest in terminal["evidence_sha256"].items()
    ):
        raise ValueError("NS actual-LR predecessor original terminal evidence coverage differs")
    times = [datetime.fromisoformat(value["time"]) for value in (owner_intent, result, state, terminal, queue_state)]
    if any(value.tzinfo is None for value in times) or times != sorted(times):
        raise ValueError("NS actual-LR predecessor original completion chronology differs")
    if any(
        handoff.observed(state[key]) is not None for key in ("controller_pid", "child_pid")
    ) or handoff.marker_processes(contract.CHILD_MARKER, state["child_marker"]):
        raise RuntimeError("NS actual-LR predecessor original full-model owner remains present")
    if not set(handoff.marker_processes(contract.CONTROLLER_MARKER, owner_intent["marker"])).issubset(
        {int(pid) for pid in terminal["keeper"]["generation"]}
    ):
        raise RuntimeError("NS actual-LR predecessor has nonkeeper controller descendants")
    if any(sha256(root / name) != digest for name, digest in hashes.items()):
        raise RuntimeError("NS actual-LR predecessor canonical evidence changed during checking")
    return {
        "status": STATUS,
        "evidence_sha256": hashes,
        "restored_keeper": terminal["keeper"],
        "controller_marker": owner_intent["marker"],
        "lower_lr_decisions": decisions,
        "actual_lr_execution_authorized": False,
        "production_authorized": False,
        "scope": SCOPE,
    }


def evidence(root: Path) -> dict:
    if not readiness(root)["ready"]:
        raise RuntimeError("NS actual-LR predecessor original full-model queue remains present")
    cpu, intent, state, terminal = reporting.evidence(root)
    if not terminal:
        raise RuntimeError("NS actual-LR predecessor original full-model terminal is incomplete")
    result = retained(root, cpu, intent, state, terminal, previous.evidence(root))
    if not readiness(root)["ready"]:
        raise RuntimeError("NS actual-LR predecessor original queue changed during authentication")
    return result
