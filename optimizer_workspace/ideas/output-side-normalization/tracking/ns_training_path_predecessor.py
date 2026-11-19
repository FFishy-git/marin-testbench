# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate completed NS evidence for a distinct training-path diagnostic."""

import argparse
import ast
import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_budget_execution as execution
import ns_budget_gpu_owner as diagnostic
import owned_output_handoff as handoff
from fused_attribution_validation import require_keeper_shape
from fused_attribution_validation_controller import require_helper_bytes
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from selected_lr_failure_handoff import keeper_guard, same_keeper
from selected_lr_release_owner import read, sha256

STATUS = "original_ns_and_failed_queues_sealed_for_distinct_training_path_diagnostic"
SEAL = "run-preparation/ns-training-path-predecessor-seal-20261119.json"
TERMINAL = "run-preparation/ns-budget-terminal-independent-20261119.json"
NS_QUEUE = "run-preparation/ns-budget-queue-20261119"
LOWER_QUEUE = "run-preparation/ns-full-model-queue-20261119"
ACTUAL_QUEUE = "run-preparation/ns-actual-lr-queue-20261119"
TRAINING_CPU = "run-preparation/ns-full-model-owner-cpu/single-pass-20261119T0300/receipt.json"
RECORDS = {
    TERMINAL: "fdff8e32afad96e6664c98abf949374cb43c5291a7d8a5cf00d0e5248e9894ec",
    "run-preparation/ns-budget-binding-20261118.json": (
        "825f9c07698e6134ff92109adab680f031bf35fa16f42c3f1cfa4a22d2f9fcab"
    ),
    "run-preparation/ns-budget-state-20261118.json": "6c551a6b37a701e217bad8cc1d200793675151b27bc7d18bb88680a578444c69",
    "diagnostics/ns-budget-20261118/comparison.json": "05a1f82b4ff17316ec20ac2bd1b87ea7121bbc6abc9957039034a882f146b6fe",
    f"{NS_QUEUE}/intent.json": "1a0660e164fc069300c90e8b058138ea828b3b9f7377a0da4117278248240e05",
    f"{NS_QUEUE}/state.json": "979ab6b4c947a939eb194d92e6073f1ae03635be138f8d7d1e66e4fb1373605c",
    f"{LOWER_QUEUE}/intent.json": "d4679f0fdfe04a939a7d95f1ff2c4e57cdff282275d7660564ec50b95cd197d9",
    f"{LOWER_QUEUE}/state.json": "7772d2d812e2f6599fc36c7bb8763737f0265a317c208cc3957038c5e13597d5",
    f"{LOWER_QUEUE}/queue.log": "8b043ab665987f67ca92e4866200d5dfed2bb1c8ddb61db7d5dd416c66086436",
    f"{ACTUAL_QUEUE}/intent.json": "eb656f9a934583d5ca6a8fbbb974a89b04a2f7096b5d206b0caaadd779ed20b1",
    f"{ACTUAL_QUEUE}/state.json": "12bd3c4d81b9cfb7ffa8f3821bb98f8e9a2346277e87aea41d96714d660ff81b",
    f"{ACTUAL_QUEUE}/queue.log": "2149836a683b183ab7e6e655104df0f5641c616e2b5bb6931619d69df272c8f6",
    TRAINING_CPU: "cbbd6514d9c6431bfd616ed73f52de13a5eb72011b96a24986e64e9b8e06eaff",
}
ARCHIVES = (
    (
        "run-preparation/ns-budget-queue-cpu/20261119T0024/receipt.json",
        "b268b833c4d79187984dffa29686e8533a72bfeac74162c87d9571eb6a441796",
        "ns_budget_queue_cpu_archive_verified_gpu_unissued",
        21,
    ),
    (
        "run-preparation/ns-full-model-queue-cpu/20261119T0318/receipt.json",
        "5c3722a9d740154d4b34ad9bc5aa7b40eda276354af4d0b9b9a1ced24f72123c",
        "ns_full_model_queue_cpu_archive_verified_gpu_unissued",
        25,
    ),
    (
        "run-preparation/ns-actual-lr-queue-cpu/20261119T0534/receipt.json",
        "3b42f4992490e39e191f60ea94a1bc5b9131bc87683c5c1bb6cf68713ff06335",
        "ns_actual_lr_queue_cpu_archive_verified_gpu_unissued",
        27,
    ),
)
OWNER_ARCHIVE = (
    "run-preparation/ns-budget-gpu-owner-cpu/20261119T0003/receipt.json",
    "18e0b21734ebdebca8934319753fc6dbed37e5e565c5e28df5d0d1f453f81c69",
    diagnostic.CPU_STATUS,
    75,
)
QUEUES = (
    (NS_QUEUE, "OUTPUT_MARIN_NS_BUDGET_QUEUE", "ns_budget_queue.py"),
    (LOWER_QUEUE, "OUTPUT_MARIN_NS_FULL_MODEL_QUEUE", "ns_full_model_queue.py"),
    (ACTUAL_QUEUE, "OUTPUT_MARIN_NS_ACTUAL_LR_QUEUE", "ns_lr_full_model_queue.py"),
)
ERRORS = {
    LOWER_QUEUE: "RuntimeError: NS full-model predecessor terminal reconstruction/control/restoration is incomplete",
    ACTUAL_QUEUE: "ValueError: NS full-model queue source is absent or aliased",
}
MUON_ROOT = "artifacts/pinned-sources-materialized/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/src/toner/fast"
TRAINING_FILES = {
    "experiment/train.py": "7df444033a0630a22ab99708cc1971ee3feb770c8c1154b96794f92cc1576980",
    "experiment/optimizer.py": "a5ec7893960b08e729da6f9fea4825d4af625685c786ee9ad1d4c6729647a635",
    "experiment/block_toner.py": "9ff7a2b2690736ec94d8f647a735476ad90fb56b7d29296143621a3dd252142c",
    "experiment/refresh_toner.py": "3d3ef4f01f206ef29fbb3a40847764711d271e197e525edec76ef17eb563acf5",
    f"{MUON_ROOT}/muon.py": "b03f10c80ce46c61d4db788dda481bc8407273e809b252bd99132c49d665bb42",
    f"{MUON_ROOT}/engine.py": "1887e09312bb7276a2e3057cf9a41c644c353aa1b1c53dbaa1359c32f43892cb",
}
SCOPE = (
    "Read-only original independently reconstructed NS terminal, rejected controls, failed "
    "queue archives/ancestry/absence and exact keeper evidence. Eager reproduction identifies "
    "the training source's NS boundary; it does not correct source-pair sensitivity or full-model "
    "continuation. No tensor reconstruction, GPU release, launch, production, optimizer quality, "
    "elapsed-efficiency or SwanLab authority. Existing archives and failed predicates stay unchanged."
)


def require_hidden_cuda() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Training-path predecessor inspection requires hidden CUDA")


def require_retained_identities(root: Path, saved: dict) -> None:
    if set(saved["file_identity"]) != set(saved["evidence_sha256"]) or any(
        file_identity(root / name) != identity for name, identity in saved["file_identity"].items()
    ):
        raise ValueError("Training-path predecessor retained evidence changed during checking")


def dependencies(root: Path, archives: list[dict], frozen: dict, training_cpu: dict) -> dict:
    """Retain every authenticated archive dependency under its canonical original path."""
    result = {}

    def retain(path: Path, digest: str) -> None:
        canonical = path.resolve(strict=True)
        name = str(canonical.relative_to(root))
        if name in result and result[name] != digest:
            raise ValueError("Training-path predecessor canonical dependency digests conflict")
        result[name] = digest

    entries = [(root / spec[0], spec[1], cpu) for spec, cpu in zip(ARCHIVES, archives, strict=True)]
    entries.append((Path(frozen["cpu_receipt"]), frozen["cpu_receipt_sha256"], read(Path(frozen["cpu_receipt"]))))
    for path, receipt_digest, cpu in entries:
        retain(path, receipt_digest)
        for name in ("archive", "tests", "log"):
            retain(Path(cpu[name]), cpu[f"{name}_sha256"])
        inventory = cpu.get("inventory_sha256")
        if inventory is None:
            inventory = {f"tracking/{name}": digest for name, digest in cpu["helpers_sha256"].items()}
            inventory |= cpu["fixtures_sha256"]
        for name, digest in inventory.items():
            retain(path.parent / name, digest)
    retain(Path(frozen["archive"]), frozen["archive_sha256"])
    retain(Path(frozen["cpu_receipt"]), frozen["cpu_receipt_sha256"])
    for name, digest in frozen["helpers_sha256"].items():
        retain(Path(frozen["helpers"]) / name, digest)
    retain(root / TRAINING_CPU, RECORDS[TRAINING_CPU])
    for name, digest in TRAINING_FILES.items():
        retain(Path(training_cpu["candidate"]) / name, digest)
    return result


def accepted_archive(root: Path, spec: tuple) -> dict:
    """Authenticate an original archive without executing its nested mutable-path verifier."""
    relative, digest, status, count = spec
    path, cpu = root / relative, read(root / relative)
    helpers = path.parent / "tracking"
    require_helper_bytes(helpers, cpu["helpers_sha256"])
    inventory = cpu.get("inventory_sha256")
    if inventory is None:
        inventory = {f"tracking/{name}": value for name, value in cpu["helpers_sha256"].items()}
        inventory |= cpu["fixtures_sha256"]
    suites = list(ET.parse(cpu["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        sha256(path) != digest
        or cpu["status"] != status
        or cpu["root"] != str(root)
        or cpu["helpers"] != str(helpers)
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["cuda_visible_devices"] != ""
        or "error" in cpu
        or len(cases) != count
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != count
        or sum(int(suite.attrib["tests"]) for suite in suites) != count
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
        or any(
            cpu[name] != str(path.parent / filename) or cpu[f"{name}_sha256"] != sha256(path.parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or any(sha256(path.parent / name) != value for name, value in inventory.items())
    ):
        raise ValueError("Training-path predecessor original CPU/source/archive evidence differs")
    require_archive(Path(cpu["archive"]), inventory, alias="inventory_sha256" in cpu)
    return cpu


def exact_finite(comparisons: list[dict]) -> bool:
    return all(item["exact"] is True and item["finite"] is True for item in comparisons)


def retained_numerics(comparison: dict) -> dict:
    """Project full saved comparison coverage while retaining rejected numerical decisions."""
    if comparison["status"] != "ns_budget_diagnostic_independently_reconstructed_not_continuation_or_production":
        raise ValueError("Training-path predecessor lacks the original reconstructed diagnostic")
    processes = comparison["processes"]
    if len(processes) != 2 or [item["worker_index"] for item in processes] != [0, 1]:
        raise ValueError("Training-path predecessor needs both original opposite-order processes")
    eager, compiled, pairs, repeats = [], [], [], []
    for process in processes:
        arms = process["analysis"]["arms"]
        if [(arm["mode"], arm["steps"]) for arm in arms] != list(execution.ARMS):
            raise ValueError("Training-path predecessor numerical arm coverage differs")
        for arm in arms:
            calls = arm["calls"]
            if len(calls) != 4 or [call["call_index"] for call in calls] != list(range(4)):
                raise ValueError("Training-path predecessor repeated-input call coverage differs")
            for call in calls:
                phases = call["phases"]
                if [phase["phase"] for phase in phases] != list(execution.PHASES):
                    raise ValueError("Training-path predecessor numerical phase coverage differs")
                if arm["mode"] == "tpu" and arm["steps"] == 10:
                    eager.append(phases[0]["vs_saved_gpu_output"])
                    compiled.extend(phase["vs_saved_gpu_output"] for phase in phases[1:])
            expected_pairs = [(index, phase) for index in range(2) for phase in execution.PHASES]
            if [(pair["pair_index"], pair["phase"]) for pair in arm["source_pairs"]] != expected_pairs:
                raise ValueError("Training-path predecessor source-pair coverage differs")
            pairs.append(
                {
                    "process": process["worker_index"],
                    "mode": arm["mode"],
                    "steps": arm["steps"],
                    "comparisons": arm["source_pairs"],
                }
            )
            repeats.extend(arm["same_input_repeats"])
    cross = comparison["cross_process"]["comparisons"]
    expected_cross = {
        (mode, steps, source, occurrence, phase)
        for mode, steps in execution.ARMS
        for source in ("continuous", "resumed")
        for occurrence in range(2)
        for phase in execution.PHASES
    }
    if (
        len(cross) != len(expected_cross)
        or {(item["mode"], item["steps"], item["source"], item["occurrence"], item["phase"]) for item in cross}
        != expected_cross
    ):
        raise ValueError("Training-path predecessor cross-process coverage differs")
    if (
        not exact_finite(eager)
        or not exact_finite(cross)
        or not exact_finite(repeats)
        or comparison["cross_process"]["all_cross_process_exact_finite"] is not True
        or comparison["all_same_input_repeats_exact_finite"] is not True
        or comparison["all_graph_phases_match_compiled_call"] is not True
        or comparison["all_compiled_phases_match_pinned_eager"] is not False
        or comparison["all_ten_step_tpu_phases_reproduce_saved_outputs"] is not False
        or comparison["historical_controls_interpretable"] is not False
        or len(eager) != 8
        or len(compiled) != 40
        or any(item["within_tolerance"] is not False or item["finite"] is not True for item in compiled)
        or any(
            item["finite"] is not True or not math.isfinite(item["relative_frobenius_error"])
            for record in pairs
            for item in record["comparisons"]
        )
    ):
        raise ValueError("Training-path predecessor original reproduction or rejected control changed")
    return {
        "historical_eager_exact_finite": True,
        "historical_eager_comparisons": len(eager),
        "historical_compiled_within_tolerance": False,
        "historical_compiled_failed_comparisons": len(compiled),
        "historical_controls_interpretable": False,
        "source_pairs": pairs,
        "all_source_pairs_within_tolerance": all(
            item["within_tolerance"] for record in pairs for item in record["comparisons"]
        ),
        "original_comparison": comparison,
    }


def queues(root: Path, archives: list[dict]) -> list[dict]:
    result = []
    for (relative, marker_name, filename), cpu in zip(QUEUES, archives, strict=True):
        work = root / relative
        intent, state = read(work / "intent.json"), read(work / "state.json")
        command = [
            diagnostic.PYTHON,
            "-P",
            str(Path(cpu["helpers"]) / filename),
            "--root",
            str(root),
            "--cpu-receipt",
            intent["cpu_receipt"],
            "--phase",
            "execute",
        ]
        generation = intent["generation"]
        if (
            intent["status"] != "acknowledged"
            or intent["command"] != command
            or generation["command"] != command
            or generation["pid"] != intent["pid"]
            or generation["start_ticks"] != intent["start_ticks"]
            or generation["group"] != intent["pid"]
            or generation["session"] != intent["pid"]
            or state["pid"] != intent["pid"]
            or state["start_ticks"] != intent["start_ticks"]
            or intent["runner_sha256"] != cpu["helpers_sha256"][filename]
            or intent["cpu_receipt"] != str(root / ARCHIVES[len(result)][0])
            or intent["cpu_receipt_sha256"] != ARCHIVES[len(result)][1]
            or intent["parent_cpu_receipt_sha256"] != cpu["parent_cpu_receipt_sha256"]
            or intent["interval"] != 900
            or any(
                state[name] != intent[name]
                for name in ("cpu_receipt_sha256", "parent_cpu_receipt_sha256", "runner_sha256", "interval")
            )
        ):
            raise ValueError("Training-path predecessor original queue ancestry or accepted source differs")
        if handoff.observed(intent["pid"]) is not None or handoff.marker_processes(marker_name, intent["marker"]):
            raise RuntimeError("Training-path predecessor original queue or descendants remain present")
        if relative in ERRORS:
            if state["status"] != "needs_attention" or state["error"] != ERRORS[relative] or state["phases"]:
                raise ValueError("Training-path predecessor failed queue decision or unissued phases changed")
        elif (
            state["status"] != "ns_diagnostic_terminal_reconstructed"
            or state["historical_controls_interpretable"] is not False
            or len(state["phases"]) != 4
        ):
            raise ValueError("Training-path predecessor original NS queue terminal differs")
        result.append({"queue": relative, "marker_name": marker_name, "intent": intent, "state": state})
    return result


def training_boundary(candidate: Path, cpu: dict) -> dict:
    """Bind the audited NS boundary to exact accepted trainer and numerical source."""
    if cpu["candidate"] != str(candidate) or any(
        cpu["files_sha256"].get(name) != digest or sha256(candidate / name) != digest
        for name, digest in TRAINING_FILES.items()
    ):
        raise ValueError("Training-path predecessor accepted compiler-boundary source differs")
    tree = ast.parse((candidate / f"{MUON_ROOT}/muon.py").read_bytes())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "newton_schulz_batched"
    )
    if function.decorator_list:
        raise ValueError("Training-path predecessor NS routine acquired a decorator")
    return {
        "candidate": str(candidate),
        "files_sha256": TRAINING_FILES,
        "model_body_and_loss_compiled": True,
        "pre_post_ns_helpers_compiled": True,
        "cached_root_graphs_preserved": True,
        "ns_called_outside_compiled_model_and_product_helpers": True,
        "scope": "Exact accepted source-defined boundary; no running-training or causal claim.",
    }


def absent_owners(root: Path, saved: dict) -> None:
    for record in saved["queues"]:
        intent = record["intent"]
        if handoff.observed(intent["pid"]) is not None or handoff.marker_processes(
            record["marker_name"], intent["marker"]
        ):
            raise RuntimeError("Training-path predecessor queue returned during checking")
    state, intent = saved["diagnostic_state"], saved["diagnostic_intent"]
    if handoff.observed(state["controller_pid"]) is not None:
        raise RuntimeError("Training-path predecessor original diagnostic controller remains present")
    for entry in state["executions"]:
        if handoff.observed(entry["pid"]) is not None or handoff.marker_processes(
            diagnostic.CHILD_MARKER, entry["marker"]
        ):
            raise RuntimeError("Training-path predecessor original diagnostic worker remains present")
    allowed = {int(pid) for pid in saved["restored_keeper"]["generation"]}
    if not set(handoff.marker_processes(diagnostic.CONTROLLER_MARKER, intent["marker"])).issubset(allowed):
        raise RuntimeError("Training-path predecessor has nonkeeper controller descendants")
    for entry in saved["queues"][0]["state"]["phases"]:
        if handoff.observed(entry["pid"]) is not None:
            raise RuntimeError("Training-path predecessor original NS queue phase remains present")


def evidence(root: Path) -> dict:
    """Hash complete original files once and preserve their independent terminal decisions."""
    require_hidden_cuda()
    if root != root.resolve(strict=True):
        raise ValueError("Training-path predecessor needs a canonical original root")
    hashes = {name: sha256(root / name) for name in RECORDS}
    if hashes != RECORDS:
        raise ValueError("Training-path predecessor pinned original records changed")
    archives = [accepted_archive(root, spec) for spec in ARCHIVES]
    owner_cpu = accepted_archive(root, OWNER_ARCHIVE)
    recorded_queues = queues(root, archives)
    binding, helpers, state_path, work = diagnostic.locations(root)
    frozen, state, intent, terminal, comparison = (
        read(path)
        for path in (binding, state_path, helpers.parent / "controller.json", root / TERMINAL, work / "comparison.json")
    )
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    require_archive(
        Path(frozen["archive"]),
        {f"tracking/{name}": digest for name, digest in frozen["helpers_sha256"].items()},
        alias=False,
    )
    ns_state = recorded_queues[0]["state"]
    if (
        frozen["root"] != str(root)
        or frozen["status"] != diagnostic.STATUS
        or frozen["helpers"] != str(helpers)
        or frozen["work"] != str(work)
        or frozen["node"] != diagnostic.NODE
        or frozen["environment"] != diagnostic.ENVIRONMENT
        or frozen["cpu_receipt"] != str(root / OWNER_ARCHIVE[0])
        or frozen["cpu_receipt_sha256"] != OWNER_ARCHIVE[1]
        or frozen["helpers_sha256"] != owner_cpu["helpers_sha256"]
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or state["status"] != "diagnostic_complete"
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or state["binding_sha256"] != sha256(binding)
        or state["comparison_sha256"] != sha256(work / "comparison.json")
        or state["keeper_release_complete"] is not True
        or state["keeper_release_attempted"] is not True
        or state["predecessor"] != frozen["predecessor"]
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or len(state["executions"]) != 2
        or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
        or terminal["status"] != "ns_budget_terminal_independently_reconstructed_exact_restoration_verified"
        or terminal["binding_sha256"] != sha256(binding)
        or terminal["terminal_time"] != state["time"]
        or terminal["controller_pid_absent"] != state["controller_pid"]
        or terminal["worker_pids_absent"] != [entry["pid"] for entry in state["executions"]]
        or terminal["comparison"] != {key: value for key, value in comparison.items() if key != "time"}
        or terminal["verifier_sha256"] != frozen["helpers_sha256"]["ns_budget_gpu_controller.py"]
        or ns_state["terminal_sha256"] != hashes[TERMINAL]
        or ns_state["issued_binding_sha256"] != sha256(binding)
        or not same_keeper(terminal["keeper"], state["restored_keeper"])
        or terminal["keeper"]["devices"] != frozen["devices"]
    ):
        raise ValueError("Training-path predecessor independently reconstructed terminal or restoration differs")
    for entry in state["executions"]:
        if (
            entry["command"] != diagnostic.worker_command(root, sha256(binding), entry["index"])
            or entry["generation"]["command"] != entry["command"]
            or entry["generation"]["pid"] != entry["pid"]
            or entry["generation"]["start_ticks"] != entry["start_ticks"]
            or entry["generation"]["group"] != entry["pid"]
            or entry["generation"]["session"] != entry["pid"]
            or entry["log_sha256"] != sha256(Path(entry["log"]))
            or entry["report_sha256"] != sha256(Path(entry["report"]))
            or entry["controller_pid"] != state["controller_pid"]
            or entry["controller_start_ticks"] != state["controller_start_ticks"]
            or any(type(entry[name]) is not int or entry[name] != 0 for name in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
            or entry["generation"]["parent"] != state["controller_pid"]
        ):
            raise ValueError("Training-path predecessor original diagnostic cleanup or ancestry differs")
    if (
        intent["status"] != "acknowledged"
        or intent["command"] != diagnostic.controller_command(root)
        or intent["binding_sha256"] != sha256(binding)
        or intent["generation"]["pid"] != intent["pid"]
        or intent["generation"]["start_ticks"] != intent["start_ticks"]
        or intent["generation"]["command"] != intent["command"]
        or intent["generation"]["group"] != intent["pid"]
        or intent["generation"]["session"] != intent["pid"]
    ):
        raise ValueError("Training-path predecessor original controller acknowledgment differs")
    for entry in ns_state["phases"]:
        phase = entry["phase"]
        command = [
            diagnostic.PYTHON,
            "-P",
            (
                str(Path(owner_cpu["helpers"]) / "ns_budget_gpu_controller.py")
                if phase == "freeze"
                else str(helpers / "ns_budget_gpu_controller.py")
            ),
            "--root",
            str(root),
            "--phase",
            phase,
        ]
        if phase == "freeze":
            command += ["--cpu-receipt", str(root / OWNER_ARCHIVE[0])]
        elif phase == "terminal":
            command += ["--out", str(root / TERMINAL)]
        if (
            [item["phase"] for item in ns_state["phases"]] != ["freeze", "check", "start", "terminal"]
            or entry["command"] != command
            or entry["log"] != str(root / NS_QUEUE / f"{phase}.log")
            or entry["spawn_status"] != "acknowledged"
            or entry["generation"]["command"] != entry["command"]
            or entry["generation"]["pid"] != entry["pid"]
            or entry["generation"]["start_ticks"] != entry["start_ticks"]
            or entry["generation"]["parent"] != recorded_queues[0]["intent"]["pid"]
            or entry["generation"]["group"] != entry["pid"]
            or entry["generation"]["session"] != entry["pid"]
            or any(type(entry[name]) is not int or entry[name] != 0 for name in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
            or entry["log_sha256"] != sha256(Path(entry["log"]))
        ):
            raise ValueError("Training-path predecessor original NS queue phase/cleanup differs")
    times = [
        datetime.fromisoformat(read(root / name)["time"])
        for name in ("diagnostics/ns-budget-20261118/comparison.json", TERMINAL, f"{NS_QUEUE}/state.json")
    ]
    if any(value.tzinfo is None for value in times) or times != sorted(times):
        raise ValueError("Training-path predecessor original completion chronology differs")
    numerical = retained_numerics(comparison)
    training_cpu = read(root / TRAINING_CPU)
    boundary = training_boundary(Path(training_cpu["candidate"]), training_cpu)
    required_terminal_paths = {
        str(binding.relative_to(root)),
        str(state_path.relative_to(root)),
        str((helpers.parent / "controller.json").relative_to(root)),
        str((helpers.parent / "controller.log").relative_to(root)),
        str((work / "comparison.json").relative_to(root)),
        *(
            str((work / f"process-{index}{suffix}").relative_to(root))
            for index in range(2)
            for suffix in (".log", ".json", ".pt")
        ),
    }
    if set(terminal["evidence_sha256"]) != required_terminal_paths:
        raise ValueError("Training-path predecessor terminal lacks complete retained output identities")
    hashes |= terminal["evidence_sha256"]
    hashes |= {
        f"{NS_QUEUE}/{name}": sha256(root / NS_QUEUE / name)
        for name in ("queue.log", "freeze.log", "check.log", "start.log", "terminal.log")
    }
    identities = {name: file_identity(root / name) for name in hashes}
    dependency_hashes = dependencies(root, archives, frozen, training_cpu)
    for name, digest in dependency_hashes.items():
        if name in hashes and hashes[name] != digest:
            raise ValueError("Training-path predecessor dependency disagrees with original evidence")
        hashes[name] = digest
        identities[name] = file_identity(root / name)
    if any(sha256(root / name) != digest for name, digest in hashes.items()):
        raise ValueError("Training-path predecessor original retained file identity changed")
    if identities != {name: file_identity(root / name) for name in hashes}:
        raise RuntimeError("Training-path predecessor original evidence changed during hashing")
    result = {
        "status": STATUS,
        "root": str(root),
        "evidence_sha256": hashes,
        "file_identity": identities,
        "dependency_sha256": dependency_hashes,
        "queues": recorded_queues,
        "diagnostic_state": state,
        "diagnostic_intent": intent,
        "restored_keeper": terminal["keeper"],
        "numerical_evidence": numerical,
        "training_boundary": boundary,
        "gpu_execution_authorized": False,
        "production_authorized": False,
        "scope": SCOPE,
    }
    absent_owners(root, result)
    require_retained_identities(root, result)
    return result


def seal(root: Path) -> dict:
    require_hidden_cuda()
    out = root / SEAL
    if out.exists():
        raise FileExistsError("Training-path predecessor seal is one-shot")
    saved = evidence(root)
    with keeper_guard():
        keeper = handoff.keeper_observation()
        require_keeper_shape(keeper)
        if not same_keeper(keeper, saved["restored_keeper"]):
            raise RuntimeError("Training-path predecessor exact original keeper is not restored")
        absent_owners(root, saved)
        require_retained_identities(root, saved)
        saved["time"] = datetime.now(UTC).isoformat()
        with tempfile.TemporaryDirectory(prefix=".ns-training-path-seal-", dir=out.parent) as directory:
            staged = Path(directory) / "seal.json"
            with staged.open("x") as stream:
                stream.write(json.dumps(saved, indent=2, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(staged, out)
            staged.unlink()
    return read(out)


def require_seal(root: Path, digest: str) -> dict:
    """Require the caller-bound seal and unchanged files without rereading tensor payloads."""
    require_hidden_cuda()
    path = root / SEAL
    identity = file_identity(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest or file_identity(path) != identity:
        raise ValueError("Training-path predecessor caller-bound evidence seal changed")
    saved = json.loads(raw)
    if (
        saved["status"] != STATUS
        or saved["root"] != str(root)
        or saved["scope"] != SCOPE
        or {name: saved["evidence_sha256"].get(name) for name in RECORDS} != RECORDS
        or saved["numerical_evidence"] != retained_numerics(saved["numerical_evidence"]["original_comparison"])
        or saved["gpu_execution_authorized"] is not False
        or saved["production_authorized"] is not False
        or set(saved["file_identity"]) != set(saved["evidence_sha256"])
        or any(saved["evidence_sha256"].get(name) != value for name, value in saved["dependency_sha256"].items())
        or any(file_identity(root / name) != identity for name, identity in saved["file_identity"].items())
        or any(sha256(root / name) != value for name, value in RECORDS.items())
        or path.is_symlink()
    ):
        raise ValueError("Training-path predecessor sealed original evidence or decisions changed")
    absent_owners(root, saved)
    require_retained_identities(root, saved)
    if file_identity(path) != identity:
        raise ValueError("Training-path predecessor seal changed during checking")
    return saved


def live(root: Path, digest: str) -> dict:
    saved = require_seal(root, digest)
    keeper = handoff.keeper_observation()
    require_keeper_shape(keeper)
    if not same_keeper(keeper, saved["restored_keeper"]):
        raise RuntimeError("Training-path predecessor exact restored keeper generation differs")
    require_retained_identities(root, saved)
    return keeper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("seal", "check"), required=True)
    parser.add_argument("--seal-sha256")
    args = parser.parse_args()
    require_hidden_cuda()
    root = args.root.resolve(strict=True)
    if args.phase == "seal":
        result = seal(root)
    else:
        if args.seal_sha256 is None:
            parser.error("Checking requires the caller-bound --seal-sha256")
        result = require_seal(root, args.seal_sha256)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
