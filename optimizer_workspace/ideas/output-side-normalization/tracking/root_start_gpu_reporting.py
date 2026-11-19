# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report source-bound saved start trials without executing CUDA or rederiving numerical results."""

from datetime import datetime
from pathlib import Path

import root_start_accuracy as accuracy
import root_start_gpu_owner as owner
from prepare_selected_lr_validation import require_archive
from root_stage_precision_reporting import cpu_evidence
from selected_lr_failure_handoff import same_keeper
from selected_lr_release_owner import read, sha256

CPU = "run-preparation/root-start-gpu-owner-cpu/20261118T1952/receipt.json"
CPU_SHA256 = "5c5172704bc7f1d3c4e8e2e97c4276013f63bef824238f853243c79e4de985b3"
BINDING_SHA256 = "d8c64fc07c110d997bda28a734ac25cb0b44d2ed452f1dd7f98fdb3148cab9d8"
TERMINAL = "run-preparation/root-start-accuracy-terminal-independent-20261118T2001.json"
TERMINAL_SHA256 = "6d32ada12cd2f199920b391b61179c6bc03d0390db87029707b47b3b2c4750da"
TERMINAL_STATUS = "start_accuracy_terminal_independently_reconstructed_exact_restoration_verified"


def evidence(root: Path) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Authenticate saved receipt/source/owner ancestry, not current processes or retained arithmetic."""
    cpu = cpu_evidence(root, root / CPU, CPU_SHA256, owner.CPU_STATUS, 121)
    binding, helpers, state_path, work = owner.locations(root)
    frozen = read(binding)
    if (
        sha256(binding) != BINDING_SHA256
        or frozen["status"] != owner.STATUS
        or frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or frozen["work"] != str(work)
        or frozen["cpu_receipt"] != str(root / CPU)
        or frozen["cpu_receipt_sha256"] != CPU_SHA256
        or frozen["helpers_sha256"] != cpu["helpers_sha256"]
        or frozen["controller_command"] != owner.controller_command(root)
        or frozen["environment"] != owner.ENVIRONMENT
        or frozen["worker_environment"] != owner.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": frozen["devices"][0][1]}
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or any(sha256(helpers / name) != value for name, value in frozen["helpers_sha256"].items())
    ):
        raise ValueError("Saved start-accuracy binding or accepted source differs")
    require_archive(
        Path(frozen["archive"]),
        {f"tracking/{name}": value for name, value in frozen["helpers_sha256"].items()},
        alias=False,
    )
    paths = [
        path for path in (helpers.parent / "controller.json", state_path, work / "comparison.json") if path.is_file()
    ]
    hashes = {path: sha256(path) for path in paths}
    intent, state, comparison = (
        read(path) if path.is_file() else {}
        for path in (
            helpers.parent / "controller.json",
            state_path,
            work / "comparison.json",
        )
    )
    if intent and (
        intent["binding_sha256"] != BINDING_SHA256
        or intent["command"] != owner.controller_command(root)
        or intent["status"] not in ("starting", "acknowledged", "needs_attention")
        or (
            intent["status"] == "acknowledged"
            and (
                intent["pid"] != intent["generation"]["pid"]
                or intent["start_ticks"] != intent["generation"]["start_ticks"]
                or intent["generation"]["command"] != intent["command"]
            )
        )
    ):
        raise ValueError("Saved start-accuracy acknowledgment differs")
    if state and (
        intent.get("status") != "acknowledged"
        or state["binding_sha256"] != BINDING_SHA256
        or state["predecessor"] != frozen["predecessor"]
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or state["status"] not in ("preflight", "replaying", "diagnostic_complete", "needs_attention")
        or len(state["executions"]) > 2
    ):
        raise ValueError("Saved start-accuracy controller state or ancestry differs")
    for index, entry in enumerate(state.get("executions", [])):
        generation = entry["generation"]
        if (
            entry["index"] != index
            or entry["command"] != owner.worker_command(root, BINDING_SHA256, index)
            or generation["command"] != entry["command"]
            or generation["pid"] != entry["pid"]
            or generation["start_ticks"] != entry["start_ticks"]
            or generation["parent"] != state["controller_pid"]
            or entry["controller_pid"] != state["controller_pid"]
            or entry["controller_start_ticks"] != state["controller_start_ticks"]
            or any(type(entry[name]) is not int or entry[name] != 0 for name in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
            or entry["report"] != str(work / f"process-{index}.json")
            or entry["report_sha256"] != sha256(Path(entry["report"]))
            or entry["log"] != str(work / f"process-{index}.log")
            or entry["log_sha256"] != sha256(Path(entry["log"]))
        ):
            raise ValueError("Saved start-accuracy completed worker or retained report differs")
    terminal_path = root / TERMINAL
    terminal = read(terminal_path) if terminal_path.is_file() else {}
    if state.get("status") == "diagnostic_complete":
        if (
            len(state["executions"]) != 2
            or not comparison
            or state["comparison_sha256"] != sha256(work / "comparison.json")
            or state["keeper_release_complete"] is not True
            or type(state["keeper_restore"]) is not int
            or state["keeper_restore"] != 0
            or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
            or comparison["executions"] != state["executions"]
            or comparison["status"] != "saved_start_accuracy_independently_reconstructed_not_continuation_or_production"
            or len(comparison["processes"]) != 2
            or [row["coordinate"] for row in comparison["cross_process"]]
            != sorted([[row[key] for key in ("variant", "group", "mode", "phase")] for row in accuracy.coordinates(0)])
            or datetime.fromisoformat(comparison["time"]) > datetime.fromisoformat(state["time"])
        ):
            raise ValueError("Saved start-accuracy completion lacks bound comparison or restoration")
        for index, process in enumerate(comparison["processes"]):
            report = read(work / f"process-{index}.json")
            if (
                process["worker_index"] != index
                or process["report_sha256"] != state["executions"][index]["report_sha256"]
                or process["analysis"] != report["analysis"]
                or process["analysis"]["coordinates"] != accuracy.coordinates(index)
                or len(process["analysis"]["O_fp64_accuracy"]) != 256
                or process["analysis"]["tolerance"] != accuracy.TOLERANCE
            ):
                raise ValueError("Saved start-accuracy comparison differs from completed process reports")
    if terminal:
        if (
            state.get("status") != "diagnostic_complete"
            or sha256(terminal_path) != TERMINAL_SHA256
            or terminal["status"] != TERMINAL_STATUS
            or terminal["binding_sha256"] != BINDING_SHA256
            or terminal["terminal_time"] != state["time"]
            or terminal["controller_pid_absent"] != state["controller_pid"]
            or terminal["worker_pids_absent"] != [entry["pid"] for entry in state["executions"]]
            or terminal["verifier_sha256"] != frozen["helpers_sha256"]["root_start_gpu_controller.py"]
            or terminal["comparison"] != {key: value for key, value in comparison.items() if key != "time"}
            or not same_keeper(terminal["keeper"], state["restored_keeper"])
            or any(
                sha256(root / name) != value
                for name, value in terminal["evidence_sha256"].items()
                if not name.endswith(".pt")
            )
            or datetime.fromisoformat(terminal["time"]) < datetime.fromisoformat(state["time"])
        ):
            raise ValueError("Saved start-accuracy terminal source, result or restoration differs")
    if hashes != {path: sha256(path) for path in paths}:
        raise ValueError("Start-accuracy saved state changed during reporting")
    return cpu, frozen, intent, state, comparison, terminal


def lines(root: Path, local_time) -> list[str]:
    if not owner.locations(root)[0].is_file():
        return []
    cpu, frozen, intent, state, comparison, terminal = evidence(root)
    current = [
        "### Separately Owned CUDA Start-Accuracy Trial",
        "",
        f"Saved CPU acceptance: `PVC/{CPU}` at {local_time(cpu['time'])}; 121 passing archived tests.",
        f"Saved freeze at {local_time(frozen['time'])}; two full-batch opposite-order processes.",
    ]
    if intent:
        current.append(f"Saved controller intent `{intent['status']}` at {local_time(intent['time'])}.")
    if state:
        current.append(
            f"Saved controller state `{state['status']}` at {local_time(state['time'])}; "
            f"{len(state['executions'])}/2 workers with integer-zero execution and cleanup exits."
        )
        if "keeper_restore" in state:
            current.append(f"Recorded keeper restoration: `{state['keeper_restore']}`.")
        if "error" in state:
            current.append(f"Preserved controller failure: `{state['error']}`.")
    if state.get("status") == "diagnostic_complete":
        for name in (
            "all_cross_process_exact_finite",
            "all_repeat_outputs_exact_finite",
            "all_unaffected_outputs_exact_finite",
            "all_rational_O_within_fp64_tolerance",
        ):
            current.append(f"Saved predicate `{name}`: `{comparison[name]}`.")
        for process in comparison["processes"]:
            rows = process["analysis"]["O_fp64_accuracy"]
            for group in accuracy.GROUPS:
                selected = [row for row in rows if row["group"] == group]
                counts = {
                    variant: sum(not row["within_tolerance"] for row in selected if row["variant"] == variant)
                    for variant in accuracy.VARIANTS
                }
                current.append(
                    f"Saved process {process['worker_index']} `{group}` O/FP64 failures across both modes/four phases: "
                    f"pinned {counts['pinned']}/{len(selected) // 2}, rational "
                    f"{counts['rational']}/{len(selected) // 2}."
                )
    if terminal:
        current.append(f"Saved independent retained-output/restoration reconstruction: `PVC/{TERMINAL}`.")
    current.append(
        "Saved state is not current liveness. The renderer authenticates small receipt/source bytes, "
        "not retained tensor arithmetic or current CUDA occupancy. Full-root accuracy, original capture-context "
        "reproduction, full-model continuation, runtime, quality and production gates remain required. "
        "No production wave, efficiency improvement or SwanLab import is inferred."
    )
    return [*current, ""]
