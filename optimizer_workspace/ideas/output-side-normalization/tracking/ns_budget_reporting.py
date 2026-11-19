# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report source-bound NS queue and diagnostic receipts without executing tensors."""

import math
from datetime import datetime
from pathlib import Path

import ns_budget_execution as execution
import ns_budget_gpu_owner as owner
import ns_budget_queue as queue
from prepare_selected_lr_validation import require_archive
from root_stage_precision_reporting import cpu_evidence
from selected_lr_failure_handoff import same_keeper
from selected_lr_release_owner import read, sha256

QUEUE_CPU = "run-preparation/ns-budget-queue-cpu/20261119T0024/receipt.json"
QUEUE_CPU_SHA256 = "b268b833c4d79187984dffa29686e8533a72bfeac74162c87d9571eb6a441796"
QUEUE_INTENT_SHA256 = "1a0660e164fc069300c90e8b058138ea828b3b9f7377a0da4117278248240e05"
QUEUE_STATUSES = (
    "waiting_for_original_rational_terminal",
    "original_terminal_ready",
    "issuing_ns_diagnostic",
    "ns_diagnostic_controller_acknowledged",
    "waiting_for_issued_ns_owner",
    "issued_ns_terminal_ready",
    "ns_diagnostic_terminal_reconstructed",
    "needs_attention",
)


def queue_evidence(root: Path) -> tuple[dict, dict, dict, dict]:
    """Authenticate saved source and phase origin; do not infer current liveness."""
    path, work = root / QUEUE_CPU, root / queue.WORK
    cpu = cpu_evidence(root, path, QUEUE_CPU_SHA256, queue.CPU_STATUS, 21)
    inherited = cpu_evidence(root, root / queue.PARENT, queue.PARENT_SHA256, owner.CPU_STATUS, 75)
    sources = cpu["helpers_sha256"]
    if (
        cpu["parent_cpu_receipt_sha256"] != queue.PARENT_SHA256
        or cpu["fixtures_sha256"] != inherited["fixtures_sha256"]
        or set(sources) != set(inherited["helpers_sha256"]) | set(queue.NEW_HELPERS)
        or any(sources[name] != digest for name, digest in inherited["helpers_sha256"].items())
        or cpu["command"] != queue.cpu_command(path.parent)
        or cpu["cwd"] != str(path.parent)
        or cpu["environment"] != owner.ENVIRONMENT
        or sha256(Path(queue.__file__)) != sources["ns_budget_queue.py"]
    ):
        raise ValueError("Saved NS queue accepted protocol or dependency differs")
    intent_path, state_path = work / "intent.json", work / "state.json"
    paths = [item for item in (intent_path, state_path) if item.is_file()]
    hashes = {item: sha256(item) for item in paths}
    intent, state = (read(item) if item.is_file() else {} for item in (intent_path, state_path))
    if intent and (
        sha256(intent_path) != QUEUE_INTENT_SHA256
        or intent["status"] != "acknowledged"
        or intent["command"] != queue.command(root, path)
        or intent["cpu_receipt"] != str(path)
        or intent["cpu_receipt_sha256"] != QUEUE_CPU_SHA256
        or intent["runner_sha256"] != sources["ns_budget_queue.py"]
        or intent["parent_cpu_receipt_sha256"] != queue.PARENT_SHA256
        or intent["interval"] != 900
        or intent["pid"] != intent["generation"]["pid"]
        or intent["start_ticks"] != intent["generation"]["start_ticks"]
        or intent["generation"]["command"] != intent["command"]
        or intent["generation"]["group"] != intent["pid"]
        or intent["generation"]["session"] != intent["pid"]
        or not intent["marker"]
    ):
        raise ValueError("Saved NS queue original acknowledgment changed")
    if state and (
        not intent
        or state["status"] not in QUEUE_STATUSES
        or state["pid"] != intent["pid"]
        or state["start_ticks"] != intent["start_ticks"]
        or any(
            state[name] != intent[name]
            for name in ("cpu_receipt_sha256", "parent_cpu_receipt_sha256", "runner_sha256", "interval")
        )
        or len(state["phases"]) > 4
        or [entry["phase"] for entry in state["phases"]]
        != ["freeze", "check", "start", "terminal"][: len(state["phases"])]
        or ("error" in state and state["status"] != "needs_attention")
    ):
        raise ValueError("Saved NS queue state or phase order changed")
    commands = {name: (argv, cwd) for name, argv, cwd in queue.phases(root, inherited)}
    helpers = owner.locations(root)[1]
    commands["terminal"] = (
        [
            owner.PYTHON,
            "-P",
            str(helpers / "ns_budget_gpu_controller.py"),
            "--root",
            str(root),
            "--phase",
            "terminal",
            "--out",
            str(root / queue.TERMINAL),
        ],
        helpers,
    )
    for entry in state.get("phases", []):
        argv, cwd = commands[entry["phase"]]
        log = work / f"{entry['phase']}.log"
        if (
            entry["command"] != argv
            or entry["cwd"] != str(cwd)
            or entry["generation"]["command"] != argv
            or entry["generation"]["pid"] != entry["pid"]
            or entry["generation"]["start_ticks"] != entry["start_ticks"]
            or entry["generation"]["parent"] != state["pid"]
            or entry["generation"]["group"] != entry["pid"]
            or entry["generation"]["session"] != entry["pid"]
            or entry["spawn_status"] != "acknowledged"
            or not entry["marker"]
            or any(type(entry[name]) is not int or entry[name] != 0 for name in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
            or entry["log"] != str(log)
            or entry["log_sha256"] != sha256(log)
        ):
            raise ValueError("Saved NS queue completed phase or log origin differs")
    if hashes != {item: sha256(item) for item in paths}:
        raise ValueError("NS queue saved state changed during reporting")
    return cpu, inherited, intent, state


def diagnostic_evidence(root: Path, cpu: dict, queue_state: dict) -> tuple[dict, dict, dict, dict, dict]:
    binding, helpers, state_path, work = owner.locations(root)
    if not binding.is_file():
        return {}, {}, {}, {}, {}
    frozen = read(binding)
    if (
        frozen["status"] != owner.STATUS
        or frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or frozen["work"] != str(work)
        or frozen["cpu_receipt"] != str(root / queue.PARENT)
        or frozen["cpu_receipt_sha256"] != queue.PARENT_SHA256
        or frozen["helpers_sha256"] != cpu["helpers_sha256"]
        or frozen["controller_command"] != owner.controller_command(root)
        or frozen["environment"] != owner.ENVIRONMENT
        or frozen["worker_environment"] != owner.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": frozen["devices"][0][1]}
        or frozen["workers"] != 2
        or frozen["worker_timeout"] != owner.WORKER_TIMEOUT
        or frozen["coordinates"] != [execution.coordinates(index) for index in (0, 1)]
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or (queue_state.get("issued_binding_sha256") and queue_state["issued_binding_sha256"] != sha256(binding))
    ):
        raise ValueError("Saved NS diagnostic binding or accepted source changed")
    require_archive(
        Path(frozen["archive"]),
        {f"tracking/{name}": digest for name, digest in frozen["helpers_sha256"].items()},
        alias=False,
    )
    paths = [
        item
        for item in (helpers.parent / "controller.json", state_path, work / "comparison.json", root / queue.TERMINAL)
        if item.is_file()
    ]
    hashes = {item: sha256(item) for item in paths}
    intent, state, comparison, terminal = (
        read(item) if item.is_file() else {}
        for item in (helpers.parent / "controller.json", state_path, work / "comparison.json", root / queue.TERMINAL)
    )
    if intent and (
        intent["command"] != frozen["controller_command"]
        or intent["binding_sha256"] != sha256(binding)
        or intent["status"] not in ("starting", "acknowledged", "needs_attention")
        or (
            intent["status"] == "acknowledged"
            and (
                intent["generation"]["pid"] != intent["pid"]
                or intent["generation"]["start_ticks"] != intent["start_ticks"]
                or intent["generation"]["command"] != intent["command"]
            )
        )
    ):
        raise ValueError("Saved NS diagnostic controller acknowledgment changed")
    if state and (
        intent.get("status") != "acknowledged"
        or state["status"] not in ("preflight", "replaying", "diagnostic_complete", "needs_attention")
        or state["binding_sha256"] != sha256(binding)
        or state["predecessor"] != frozen["predecessor"]
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or len(state["executions"]) > 2
    ):
        raise ValueError("Saved NS diagnostic state or ancestry changed")
    for index, entry in enumerate(state.get("executions", [])):
        generation = entry["generation"]
        if (
            entry["index"] != index
            or entry["command"] != owner.worker_command(root, sha256(binding), index)
            or entry["controller_pid"] != state["controller_pid"]
            or entry["controller_start_ticks"] != state["controller_start_ticks"]
            or generation["command"] != entry["command"]
            or generation["pid"] != entry["pid"]
            or generation["start_ticks"] != entry["start_ticks"]
            or generation["parent"] != state["controller_pid"]
            or any(type(entry[name]) is not int or entry[name] != 0 for name in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
            or entry["report"] != str(work / f"process-{index}.json")
            or entry["report_sha256"] != sha256(Path(entry["report"]))
            or entry["log"] != str(work / f"process-{index}.log")
            or entry["log_sha256"] != sha256(Path(entry["log"]))
        ):
            raise ValueError("Saved NS completed worker report or log origin changed")
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
            or comparison["status"] != "ns_budget_diagnostic_independently_reconstructed_not_continuation_or_production"
            or len(comparison["processes"]) != 2
            or datetime.fromisoformat(comparison["time"]) > datetime.fromisoformat(state["time"])
        ):
            raise ValueError("Saved NS completion lacks retained comparison or restoration")
        for index, process in enumerate(comparison["processes"]):
            report = read(work / f"process-{index}.json")
            if (
                process["worker_index"] != index
                or process["report_sha256"] != state["executions"][index]["report_sha256"]
                or process["analysis"] != report["analysis"]
                or process["timings"] != report["timings"]
                or report["coordinates"] != execution.coordinates(index)
                or process["outputs_sha256"] != report["outputs_sha256"]
            ):
                raise ValueError("Saved NS comparison differs from completed process reports")
        predicates = (
            "all_ten_step_tpu_phases_reproduce_saved_outputs",
            "all_compiled_phases_match_pinned_eager",
            "all_graph_phases_match_compiled_call",
            "all_same_input_repeats_exact_finite",
        )
        if any(
            comparison[name] is not all(process["analysis"][name] for process in comparison["processes"])
            for name in predicates
        ) or comparison["historical_controls_interpretable"] is not (
            comparison["cross_process"]["all_cross_process_exact_finite"]
            and all(comparison[name] for name in predicates)
        ):
            raise ValueError("Saved NS aggregate predicates differ from completed process comparisons")
    if terminal and (
        state.get("status") != "diagnostic_complete"
        or terminal["status"] != "ns_budget_terminal_independently_reconstructed_exact_restoration_verified"
        or terminal["binding_sha256"] != sha256(binding)
        or terminal["terminal_time"] != state["time"]
        or terminal["controller_pid_absent"] != state["controller_pid"]
        or terminal["worker_pids_absent"] != [entry["pid"] for entry in state["executions"]]
        or terminal["verifier_sha256"] != frozen["helpers_sha256"]["ns_budget_gpu_controller.py"]
        or terminal["comparison"] != {key: value for key, value in comparison.items() if key != "time"}
        or not same_keeper(terminal["keeper"], state["restored_keeper"])
        or any(
            sha256(root / name) != digest
            for name, digest in terminal["evidence_sha256"].items()
            if not name.endswith(".pt")
        )
        or (queue_state.get("terminal_sha256") and queue_state["terminal_sha256"] != sha256(root / queue.TERMINAL))
    ):
        raise ValueError("Saved NS independent terminal evidence changed")
    if hashes != {item: sha256(item) for item in paths}:
        raise ValueError("NS diagnostic saved evidence changed during reporting")
    return frozen, intent, state, comparison, terminal


def numeric_range(values: list[float]) -> str:
    if not values or any(value is None or not math.isfinite(value) for value in values):
        return "nonfinite/unavailable"
    return f"{min(values):.6g} to {max(values):.6g}"


def completed_lines(comparison: dict) -> list[str]:
    """Render saved per-arm error and matched replay timings without training-efficiency claims."""
    current = [
        "",
        "| Process | Operand Mode | Steps | Pair Gate Failures | Pair Relative Error | "
        "Replay Median ms | Matched q10/Arm Time Ratio |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for process in comparison["processes"]:
        timings = process["timings"]
        baseline = {
            (item["mode"], item["source"], item["call_index"]): item["median_milliseconds"]
            for item in timings
            if item["steps"] == 10
        }
        for arm in process["analysis"]["arms"]:
            pairs = arm["source_pairs"]
            selected = [item for item in timings if (item["mode"], item["steps"]) == (arm["mode"], arm["steps"])]
            ratios = [
                baseline[(item["mode"], item["source"], item["call_index"])] / item["median_milliseconds"]
                for item in selected
            ]
            current.append(
                f"| {process['worker_index']} | `{arm['mode']}` | {arm['steps']} | "
                f"{sum(pair['within_tolerance'] is not True for pair in pairs)}/{len(pairs)} | "
                f"{numeric_range([pair['relative_frobenius_error'] for pair in pairs])} | "
                f"{numeric_range([item['median_milliseconds'] for item in selected])} | "
                f"{numeric_range(ratios)} |"
            )
    return [*current, "", "Ranges span saved phases/repeats or matched calls, not confidence intervals.", ""]


def lines(root: Path, local_time) -> list[str]:
    if not (root / QUEUE_CPU).is_file():
        return []
    try:
        cpu, inherited, intent, state = queue_evidence(root)
        frozen, controller, diagnostic, comparison, terminal = diagnostic_evidence(root, inherited, state)
    except (OSError, ValueError, RuntimeError, KeyError) as failure:
        return [
            "### NS Budget Diagnostic And Queue",
            "",
            f"Saved NS report refused: `{type(failure).__name__}: {failure}`.",
            "",
        ]
    current = [
        "### NS Budget Diagnostic And Queue",
        "",
        f"Saved queue CPU acceptance at {local_time(cpu['time'])}; 21 passing tests, 124 helpers and two fixtures.",
        f"Saved owner CPU acceptance at {local_time(inherited['time'])}; 75 passing tests.",
        "Six pinned BF16-operand/FP32 arms at five/seven/ten steps; two full-batch opposite-order processes.",
    ]
    if intent:
        current.append(f"Saved queue acknowledgment at {local_time(intent['time'])}; 900-second original-owner checks.")
    if state:
        current.append(
            f"Saved queue `{state['status']}` at {local_time(state['time'])}; {len(state['phases'])}/4 phases."
        )
        if state.get("original_observation", {}).get("owners"):
            current.append(
                "Saved original-owner observation: "
                + ", ".join(
                    f"{name} `{entry['status']}`" for name, entry in state["original_observation"]["owners"].items()
                )
                + "."
            )
        if "error" in state:
            current.append(f"Preserved queue failure: `{state['error']}`.")
    if frozen:
        current.append(f"Saved diagnostic freeze at {local_time(frozen['time'])}; separate execution binding.")
    if controller:
        current.append(
            f"Saved diagnostic controller intent `{controller['status']}` at {local_time(controller['time'])}."
        )
    if diagnostic:
        current.append(
            f"Saved diagnostic `{diagnostic['status']}` at {local_time(diagnostic['time'])}; "
            f"{len(diagnostic['executions'])}/2 completed workers."
        )
        if "error" in diagnostic:
            current.append(f"Preserved diagnostic failure: `{diagnostic['error']}`.")
        if "keeper_restore" in diagnostic:
            current.append(f"Recorded keeper restoration: `{diagnostic['keeper_restore']}`.")
    if diagnostic.get("status") == "diagnostic_complete":
        for name in (
            "all_ten_step_tpu_phases_reproduce_saved_outputs",
            "all_compiled_phases_match_pinned_eager",
            "all_graph_phases_match_compiled_call",
            "all_same_input_repeats_exact_finite",
            "historical_controls_interpretable",
        ):
            current.append(f"Saved diagnostic predicate `{name}`: `{comparison[name]}`.")
        current.append(
            "Saved cross-process exact finite predicate: "
            f"`{comparison['cross_process']['all_cross_process_exact_finite']}`."
        )
        current.extend(completed_lines(comparison))
    if terminal:
        current.append(f"Saved independent retained-output/restoration reconstruction: `PVC/{queue.TERMINAL}`.")
    current.append(
        "Saved receipts do not prove current liveness or rederive retained tensors. CUDA-event timing excludes "
        "compilation, capture, transfers and training. Changed budgets are separate numerical arms; "
        "full-model gates, fair LR tuning, large shared-token windows and verified private histories remain required. "
        "No optimizer quality, elapsed-efficiency, continuation, production or SwanLab acceptance follows."
    )
    return [*current, ""]
