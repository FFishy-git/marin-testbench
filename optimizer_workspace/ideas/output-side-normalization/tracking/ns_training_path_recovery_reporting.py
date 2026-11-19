# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report the separately accepted recovery attempt without issuing execution authority."""

import hashlib
import json
from pathlib import Path

import ns_training_path_recovery_predecessor as predecessor
import ns_training_path_recovery_source as source
import ns_training_path_reporting as original_report
import prepare_ns_training_path_recovery as preparation

TITLE = "NS Marker-Exit Recovery Diagnostic"
CPU_SHA256 = "6c5b8af36ba473ea042fbb061153a4e94a819960a51255651198b41c9c345638"
SEAL_SHA256 = "19f4bdeb9b3b520413fdeb542bb33d22588afe3483835603fd298179eb4cc3bb"
BINDING_SHA256: str | None = "8c7afc88680630425b0662ce19a063e0d433413ea215360e9b77c2535f479332"
INTENT_SHA256: str | None = "85013b6805e40957bfe98e610002cbbf233c1cc705ddafc3011e21a881f742d9"
RESULT_STATUS = source.LITERALS["ns_training_path_actual_config_gates_reconstructed_not_production"]
TERMINAL_STATUS = source.LITERALS[
    "ns_training_path_terminal_independently_reconstructed_exact_restoration_not_production"
]


def terminal_evidence(root: Path, frozen: dict, intent: dict, state: dict, progress: dict, records: dict) -> dict:
    """Check canonical independent-terminal receipts without reconstructing tensor payloads."""
    generation = {
        "binding_sha256": intent["binding_sha256"],
        "controller_pid": intent["pid"],
        "controller_start_ticks": intent["start_ticks"],
        "controller_marker": intent["marker"],
    }
    token = hashlib.sha256(json.dumps(generation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = root / "run-preparation" / f"{source.NAMESPACE}-terminal-{token}.json"
    directory = Path(frozen["helpers"]).parent
    attempt_path, outcome_path = directory / "terminal-intent.json", directory / "terminal-outcome.json"
    if not any(item.exists() for item in (path, attempt_path, outcome_path)):
        return {}
    attempt = original_report._saved(attempt_path, records)
    if (
        attempt["status"] != "reconstruction_claimed"
        or attempt["output"] != str(path)
        or attempt["binding_sha256"] != BINDING_SHA256
        or attempt["controller_pid"] != intent["pid"]
        or attempt["controller_start_ticks"] != intent["start_ticks"]
        or attempt["controller_marker"] != intent["marker"]
        or attempt["verifier_sha256"] != frozen["helpers_sha256"]["ns_training_path_recovery_gpu_controller.py"]
    ):
        raise ValueError("Recovery reporting canonical terminal attempt differs")
    if not outcome_path.exists():
        return {"status": "reconstruction_claimed", "time": attempt["time"]}
    outcome = original_report._saved(outcome_path, records)
    if (
        outcome["output"] != str(path)
        or outcome["reconstruction_intent_sha256"] != records[attempt_path][0]
        or outcome["status"] not in ("complete", "needs_attention")
    ):
        raise ValueError("Recovery reporting terminal outcome differs")
    if outcome["status"] == "needs_attention":
        if path.exists():
            raise ValueError("Recovery reporting failed terminal also has a canonical receipt")
        return outcome
    terminal = original_report._saved(path, records)
    if (
        records[path][0] != outcome["terminal_sha256"]
        or terminal["status"] != TERMINAL_STATUS
        or terminal["binding_sha256"] != BINDING_SHA256
        or terminal["result_sha256"] != state["result_sha256"]
        or terminal["reconstruction_intent_sha256"] != records[attempt_path][0]
        or terminal["controller_pid_absent"] != intent["pid"]
        or terminal["controller_start_ticks"] != intent["start_ticks"]
        or terminal["controller_marker"] != intent["marker"]
        or terminal["validator_pid_absent"] != state["child_pid"]
        or terminal["verifier_sha256"] != attempt["verifier_sha256"]
        or terminal["predecessor_acceptance"] != frozen["predecessor_acceptance"]
        or terminal["production_authorized"] is not False
        or terminal["scope"] != source.SCOPE
        or state["status"] != "validation_complete"
        or any(
            type(state[name]) is not int or state[name] != 0
            for name in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or any(name in state for name in ("error", "cleanup_error", "deferred_signals"))
        or progress["completed_commands"] != 54
        or len(progress["decisions"]) != 18
        or terminal["accepted_cases"] != progress["accepted_cases"]
        or terminal["rejected_cases"] != progress["rejected_cases"]
        or terminal["all_cases_accepted"] != (not progress["rejected_cases"])
        or not original_report._same_keeper(terminal["keeper"], state["restored_keeper"])
    ):
        raise ValueError("Recovery reporting complete terminal coverage, decisions or restoration differs")
    for name, digest in terminal["evidence_sha256"].items():
        original_report._checked(root / name, digest, records)
    return terminal


def evidence(root: Path) -> dict:
    """Authenticate saved CPU acceptance, failed predecessor and distinct progress receipts."""
    records = {}
    cpu_path = root / preparation.CPU_PATH
    original_report._checked(cpu_path, CPU_SHA256, records)
    cpu = preparation.require_cpu(root, cpu_path)
    seal_path = root / predecessor.SEAL
    sealed = original_report._saved(seal_path, records)
    if (
        records[seal_path][0] != SEAL_SHA256
        or sealed["status"] != predecessor.STATUS
        or sealed["root"] != str(root)
        or sealed["scope"] != predecessor.SCOPE
        or sealed["gpu_execution_authorized"] is not False
        or sealed["production_authorized"] is not False
        or sealed["retained_summary"]["completed_commands"] != 44
        or sealed["retained_summary"]["case_decisions"] != 14
        or sealed["retained_summary"]["coverage_complete"] is not False
        or sealed["retained_summary"]["numerically_accepted"] is not False
    ):
        raise ValueError("Recovery reporting failed predecessor scope or incomplete evidence differs")
    binding = root / source.LITERALS["run-preparation/ns-training-path-binding-20261119.json"]
    progress = {"status": "gpu_unissued", "completed_commands": 0, "decisions": []}
    if binding.exists():
        if BINDING_SHA256 is None:
            raise ValueError("Recovery reporting has no pinned issued binding")
        frozen = original_report._saved(binding, records)
        if (
            records[binding][0] != BINDING_SHA256
            or frozen["status"] != source.LITERALS["frozen_ns_training_path_diagnostic_not_production"]
            or frozen["root"] != str(root)
            or frozen["cpu_receipt_sha256"] != CPU_SHA256
            or frozen["files_sha256"] != cpu["files_sha256"]
            or frozen["helpers_sha256"] != cpu["helpers_sha256"]
            or frozen["allocation_predecessor"] != sealed
            or frozen["predecessor_acceptance"]["seal_sha256"] != SEAL_SHA256
            or frozen["scope"] != source.SCOPE
            or frozen["commands"] != frozen["plan"]["commands"]
            or len(frozen["commands"]) != 54
            or len(frozen["plan"]["trials"]) != 18
        ):
            raise ValueError("Recovery reporting issued source or numerical protocol differs")
        preparation_path = Path(cpu["candidate"]) / source.PREPARATION
        original_report._checked(preparation_path, frozen["preparation_sha256"], records)
        helpers = Path(frozen["helpers"])
        original_report._checked(Path(frozen["archive"]), frozen["archive_sha256"], records)
        for name, digest in frozen["helpers_sha256"].items():
            original_report._checked(helpers / name, digest, records)
        intent_path = helpers.parent / "controller.json"
        if intent_path.exists():
            intent = original_report._saved(intent_path, records)
            if INTENT_SHA256 is None or records[intent_path][0] != INTENT_SHA256:
                raise ValueError("Recovery reporting original acknowledgment is not pinned")
            original_report._generation(
                intent["generation"], frozen["controller_command"], intent["pid"], intent["start_ticks"]
            )
            if intent["status"] != "acknowledged" or intent["binding_sha256"] != BINDING_SHA256:
                raise ValueError("Recovery reporting original owner acknowledgment differs")
            state_path = root / source.LITERALS["run-preparation/ns-training-path-state-20261119.json"]
            state = original_report._saved(state_path, records) if state_path.exists() else {}
            if state and (
                state["controller_pid"] != intent["pid"]
                or state["controller_start_ticks"] != intent["start_ticks"]
                or state["binding_sha256"] != BINDING_SHA256
                or state["status"] not in ("preflight", "validating", "validation_complete", "needs_attention")
            ):
                raise ValueError("Recovery reporting saved controller state identity differs")
            progress |= {"status": state.get("status", "acknowledged"), "intent": intent, "state": state}
            result_path = root / source.LITERALS["run-preparation/validation-ns-training-path-gpu-20261119.json"]
            if result_path.exists():
                result = original_report._saved(result_path, records)
                if (
                    result["binding_sha256"] != BINDING_SHA256
                    or result["trials"] != frozen["plan"]["trials"]
                    or result["status"] not in ("validating", "needs_attention", RESULT_STATUS)
                    or len(result["gpu_commands"]) > 54
                    or result["owner"]["controller_pid"] != intent["pid"]
                    or result["owner"]["validator_pid"] != state["child_pid"]
                    or result["owner"]["validator_start_ticks"] != state["child_start_ticks"]
                ):
                    raise ValueError("Recovery reporting retained execution owner or command coverage differs")
                work = root / source.LITERALS["validation/ns_training_path_20261119"]
                for index, entry in enumerate(result["gpu_commands"]):
                    item, mode = frozen["plan"]["trials"][index // 3], predecessor.original.MODES[index % 3]
                    if entry["trial"] != item["name"] or entry["mode"] != mode:
                        raise ValueError("Recovery reporting retained command order differs")
                    original_report._command(
                        entry,
                        frozen["commands"][index],
                        work / f"{item['name']}_{mode}.log",
                        state["child_pid"],
                        records,
                    )
                projected = result | {
                    "status": original_report.RESULT_STATUS if result["status"] == RESULT_STATUS else result["status"]
                }
                decisions = original_report._decisions(projected, frozen["plan"], len(result["gpu_commands"]))
                progress |= {
                    "completed_commands": len(result["gpu_commands"]),
                    "decisions": decisions["cases"],
                    "accepted_cases": decisions["accepted_cases"],
                    "rejected_cases": decisions["rejected_cases"],
                    "result": result,
                }
                if result["status"] == "needs_attention" and len(result["gpu_commands"]) < 54:
                    index = len(result["gpu_commands"])
                    item, mode = frozen["plan"]["trials"][index // 3], predecessor.original.MODES[index % 3]
                    log = work / f"{item['name']}_{mode}.log"
                    entry = original_report._saved(log.with_suffix(".execution.json"), records)
                    original_report._generation(
                        entry["generation"],
                        frozen["commands"][index],
                        entry["generation"]["pid"],
                        entry["generation"]["start_ticks"],
                        state["child_pid"],
                    )
                    if (
                        entry["command"] != frozen["commands"][index]
                        or entry["spawn_status"] != "acknowledged"
                        or entry["log"] != str(log)
                        or any(
                            type(entry[name]) is not int or entry[name] != 1
                            for name in ("exit_code", "cleanup_exit_code")
                        )
                        or entry["marked_descendants_absent"] is not True
                        or "cleanup_error" in entry
                    ):
                        raise ValueError("Recovery reporting failed command identity or cleanup differs")
                    original_report._checked(log, entry["log_sha256"], records)
                    progress["failed_command"] = entry | {"trial": item["name"], "mode": mode}
            progress["terminal"] = terminal_evidence(root, frozen, intent, state, progress, records)
    if any(original_report._sha256(name) != entry[0] for name, entry in records.items()):
        raise ValueError("Recovery reporting evidence changed during inspection")
    return {"cpu": cpu, "sealed": sealed, "progress": progress}


def lines(root: Path, local_time) -> list[str]:
    if not (root / preparation.CPU_PATH).is_file():
        return []
    title = f"### {TITLE}"
    try:
        saved = evidence(root)
    except original_report.EVIDENCE_ERRORS as failure:
        return [title, "", f"Recovery evidence withheld: `{type(failure).__name__}: {failure}`.", ""]
    cpu, sealed, progress = saved["cpu"], saved["sealed"], saved["progress"]
    result = [
        title,
        "",
        f"CPU/source acceptance at {local_time(cpu['time'])}: 103 passing cases, "
        f"{len(cpu['helpers_sha256'])} helpers and {len(cpu['inventory_sha256'])} inventory entries.",
        "The failed original remains 44/54 commands, 14/18 decisions, six accepted/eight rejected; "
        f"its separate preservation seal is dated {local_time(sealed['time'])}.",
        f"Recovery saved status: `{progress['status']}`; {progress['completed_commands']}/54 completed commands "
        f"and {len(progress['decisions'])}/18 case decisions.",
        "Source/configuration changes are restricted to the owner adapter and the measured handoff helper. "
        "Original numerical readers, tolerances, compilation and CUDA graphs are retained.",
    ]
    state = progress.get("state", {})
    if state:
        result.append(
            f"Controller receipt at {local_time(state['time'])}: "
            f"`{state['controller_pid']}/{state['controller_start_ticks']}`. "
            "This is a saved observation and does not establish current liveness or GPU occupancy."
        )
    if "error" in state:
        result.append(f"Retained execution error: `{state['error']}`.")
    failed_command = progress.get("failed_command")
    if failed_command:
        result.append(
            f"Failed command: `{failed_command['trial']}/{failed_command['mode']}` at "
            f"{local_time(failed_command['time'])}; execution/cleanup exits1/1, "
            "marked descendants absent. It is excluded from completed trajectory counts."
        )
    terminal = progress.get("terminal", {})
    if terminal:
        result.append(f"Independent terminal saved status: `{terminal['status']}` at {local_time(terminal['time'])}.")
    for case in progress["decisions"]:
        failed = [name for name, passed in case["gates"].items() if not passed]
        result.append(
            f"- `{case['name']}`: {'accepted' if case['accepted'] else 'rejected'}; "
            f"failed gates: {', '.join(failed) if failed else 'none'}."
        )
    result.extend(
        [
            "Independent complete terminal and successor validation remain separately gated. "
            "This diagnostic supplies no full-budget optimizer-quality, whole-training efficiency "
            "or SwanLab import result.",
            "",
        ]
    )
    return result
