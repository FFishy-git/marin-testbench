# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate a separate NS budget diagnostic after original rational-owner restoration."""

import ast
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_budget_execution as execution
import owned_output_handoff as handoff
import prepare_ns_budget_execution as preparation
import rational_continuation_contract as rational
from fused_attribution_validation import require_keeper_shape
from fused_attribution_validation_controller import (
    process_context,
    require_acknowledged_controller,
    require_helper_bytes,
)
from prepare_selected_lr_validation import require_archive
from root_start_gpu_owner import require_cpu_archive
from saved_ns_gpu_owner import require_held_lock
from selected_lr_failure_handoff import MANAGER_LOCK, same_keeper
from selected_lr_release_owner import read, sha256

NODE = "28.216.17.9"
PYTHON = "/opt/conda/envs/torch-base/bin/python"
CONTROLLER_MARKER = "OUTPUT_MARIN_NS_BUDGET_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_NS_BUDGET_CHILD"
STATUS = "frozen_ns_budget_diagnostic_not_continuation_or_production"
CPU_STATUS = "ns_budget_gpu_owner_cpu_archive_verified_gpu_unissued"
WORKER_TIMEOUT = 3600
PARENT = "run-preparation/ns-budget-execution-cpu/20261118T2340/receipt.json"
PARENT_SHA256 = "4ad93501569a4839f57a8c4f53532fe8139d344611c69a64f91dc7e08cdbe510"
TERMINAL = "run-preparation/rational-continuation-terminal-independent-20261118.json"
FOLLOWUP = "run-preparation/rational-continuation-terminal-followup-20261118"
FOLLOWUP_CPU = "run-preparation/rational-terminal-followup-cpu/20261118T2222/receipt.json"
FOLLOWUP_CPU_SHA256 = "ad5197a53a05b225c2ebc26193aa4008bbea55b5f4145fb288173f28a3c9830a"
FOLLOWUP_INTENT_SHA256 = "6d79465415fe352bd559d498cc1340e01c99503a66bb494c54d2d4d92f2c06d7"
FOLLOWUP_MARKER = "OUTPUT_MARIN_RATIONAL_CONTINUATION_FOLLOWUP"
FOLLOWUP_CHILD_MARKER = "OUTPUT_MARIN_RATIONAL_CONTINUATION_READONLY"
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONFAULTHANDLER": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "NCCL_INSPECTOR_ENABLE": "0",
}
TESTS = ("test_replay_saved_ns.py", "test_ns_budget_gpu.py")
NEW_HELPERS = ("ns_budget_gpu_owner.py", "ns_budget_gpu_controller.py", "ns_budget_gpu_worker.py", TESTS[-1])


def locations(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / "run-preparation/ns-budget-binding-20261118.json",
        root / "controllers/ns-budget-20261118/tracking",
        root / "run-preparation/ns-budget-state-20261118.json",
        root / "diagnostics/ns-budget-20261118",
    )


def controller_command(root: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "ns_budget_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "execute",
    ]


def worker_command(root: Path, binding_hash: str, index: int) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "ns_budget_gpu_worker.py"),
        "--root",
        str(root),
        "--binding-sha256",
        binding_hash,
        "--state",
        str(locations(root)[2]),
        "--worker-index",
        str(index),
    ]


def helper_sources(root: Path) -> dict[str, Path]:
    parent_path = root / PARENT
    if sha256(parent_path) != PARENT_SHA256:
        raise ValueError("NS owner execution-component parent receipt changed")
    parent = preparation.require_cpu(root, parent_path)
    if parent["status"] != preparation.STATUS:
        raise ValueError("NS owner lacks accepted numerical execution/reconstruction code")
    sources = {name: Path(parent["helpers"]) / name for name in parent["helpers_sha256"]}
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"NS owner new helper is absent or aliased: {name}")
        for node in ast.walk(ast.parse(path.read_bytes(), filename=name)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError(f"NS owner helper imports outside its accepted closure: {name}")
    return sources


def cpu_command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        PYTHON,
        "--with",
        "pytest-timeout",
        "python",
        "-P",
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "--rootdir",
        str(out),
        "-p",
        "no:cacheprovider",
        "-p",
        "root_diagnostic_pytest_guard",
        "--timeout=60",
        "--session-timeout=600",
        "--import-mode=prepend",
        "-m",
        preparation.SAFE_MARKERS,
        "-q",
        *(str(out / "tracking" / name) for name in TESTS),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    accepted = require_cpu_archive(root, path, CPU_STATUS, None)
    sources = helper_sources(root)
    cases = list(ET.parse(accepted["tests"]).getroot().iter("testcase"))
    if (
        accepted["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or accepted["command"] != cpu_command(path.parent)
        or accepted["cwd"] != str(path.parent)
        or accepted["environment"] != ENVIRONMENT
        or accepted["fixtures_sha256"] != read(root / PARENT)["fixtures_sha256"]
        or accepted["helpers_sha256"] != {name: sha256(source) for name, source in sources.items()}
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(name).stem for name in TESTS}
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != len(cases)
    ):
        raise ValueError("NS owner CPU source closure or complete tested-module coverage differs")
    for name, source in sources.items():
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and sha256(Path(module.__file__)) != sha256(source):
            raise RuntimeError(f"NS owner imported different accepted helper bytes: {name}")
    return accepted


def require_original_absence(state: dict) -> None:
    """Reject actual original owner handles before any terminal evidence can authorize a successor."""
    for name in ("controller_pid", "child_pid"):
        if handoff.observed(state[name]) is not None:
            raise RuntimeError("NS budget predecessor original rational owner handle remains present")
    if handoff.marker_processes(rational.CHILD_MARKER, state["child_marker"]):
        raise RuntimeError("NS budget predecessor original rational training descendants remain")


def predecessor(root: Path) -> dict:
    binding_path, helpers, state_path, _, result_path = rational.locations(root)
    state, intent = read(state_path), read(helpers.parent / "controller.json")
    require_original_absence(state)
    if sha256(binding_path) != preparation.BINDING_SHA256:
        raise ValueError("NS budget predecessor original rational binding changed")
    frozen, result, terminal = (read(path) for path in (binding_path, result_path, root / TERMINAL))
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    follow_path = root / FOLLOWUP
    if (
        sha256(root / FOLLOWUP_CPU) != FOLLOWUP_CPU_SHA256
        or sha256(follow_path / "intent.json") != FOLLOWUP_INTENT_SHA256
    ):
        raise ValueError("NS budget predecessor terminal waiter accepted identity changed")
    follow, follow_intent, cpu = (
        read(path) for path in (follow_path / "state.json", follow_path / "intent.json", root / FOLLOWUP_CPU)
    )
    runner = (root / FOLLOWUP_CPU).parent / "rational_continuation_followup.py"
    terminal_command = [
        PYTHON,
        "-P",
        str(helpers / "rational_continuation_controller.py"),
        "--root",
        str(root),
        "--phase",
        "terminal",
        "--out",
        str(root / TERMINAL),
    ]
    digest, result_digest = sha256(binding_path), sha256(result_path)
    if (
        state["status"] != "diagnostic_complete"
        or state["binding_sha256"] != digest
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or intent["status"] != "acknowledged"
        or intent["binding_sha256"] != digest
        or intent["command"] != frozen["controller_command"]
        or intent["generation"]["pid"] != state["controller_pid"]
        or intent["generation"]["start_ticks"] != state["controller_start_ticks"]
        or intent["generation"]["command"] != intent["command"]
        or intent["generation"]["group"] != state["controller_pid"]
        or intent["generation"]["session"] != state["controller_pid"]
        or state["command"] != frozen["validator_command"]
        or state["child_generation"]["pid"] != state["child_pid"]
        or state["child_generation"]["start_ticks"] != state["child_start_ticks"]
        or state["child_generation"]["parent"] != state["controller_pid"]
        or state["child_generation"]["command"] != state["command"]
        or state["child_generation"]["group"] != state["child_pid"]
        or state["child_generation"]["session"] != state["child_pid"]
        or state["spawn_status"] != "acknowledged"
        or state["marked_descendants_absent"] is not True
        or state["keeper_release_attempted"] is not True
        or state["keeper_release_complete"] is not True
        or any(
            type(state[key]) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
        or state["result_sha256"] != result_digest
        or state["log_sha256"] != sha256(Path(state["log"]))
        or result["status"] != "completed_rational_start_full_model_diagnostic_not_production"
        or "error" in result
        or result["binding_sha256"] != digest
        or result["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or result["commands"] != frozen["commands"]
        or len(result["gpu_commands"]) != 12
        or any(
            entry["command"] != command
            or type(entry["exit_code"]) is not int
            or entry["exit_code"] != 0
            or entry["log_sha256"] != sha256(Path(entry["log"]))
            for entry, command in zip(result["gpu_commands"], frozen["commands"], strict=True)
        )
        or terminal["status"] != "rational_continuation_terminal_independently_reconstructed_exact_restoration"
        or terminal["binding_sha256"] != digest
        or terminal["result_sha256"] != result_digest
        or terminal["terminal_time"] != state["time"]
        or terminal["controller_pid_absent"] != state["controller_pid"]
        or terminal["validator_pid_absent"] != state["child_pid"]
        or terminal["verifier_sha256"] != frozen["helpers_sha256"]["rational_continuation_controller.py"]
        or terminal["analysis"] != result["analysis"]
        or state["intervention_accepted"] is not result["analysis"]["diagnostic_intervention_accepted"]
        or not same_keeper(terminal["keeper"], state["restored_keeper"])
        or follow["status"] != "terminal_reconstructed"
        or follow["pid"] != follow_intent["pid"]
        or follow["start_ticks"] != follow_intent["start_ticks"]
        or follow["runner_sha256"] != sha256(runner)
        or cpu["files_sha256"]["rational_continuation_followup.py"] != sha256(runner)
        or follow["cpu_receipt_sha256"] != FOLLOWUP_CPU_SHA256
        or follow["binding_sha256"] != digest
        or follow_intent["status"] != "acknowledged"
        or follow_intent["binding_sha256"] != digest
        or follow_intent["runner_sha256"] != sha256(runner)
        or follow_intent["cpu_receipt_sha256"] != FOLLOWUP_CPU_SHA256
        or follow_intent["generation"]["pid"] != follow["pid"]
        or follow_intent["generation"]["start_ticks"] != follow["start_ticks"]
        or follow_intent["generation"]["command"] != follow_intent["command"]
        or follow_intent["generation"]["group"] != follow["pid"]
        or follow_intent["generation"]["session"] != follow["pid"]
        or follow["terminal_command"] != terminal_command
        or follow["terminal_generation"]["command"] != terminal_command
        or follow["terminal_generation"]["pid"] != follow["terminal_pid"]
        or follow["terminal_generation"]["parent"] != follow["pid"]
        or follow["terminal_generation"]["group"] != follow["terminal_pid"]
        or follow["terminal_generation"]["session"] != follow["terminal_pid"]
        or any(
            type(follow[key]) is not int or follow[key] != 0
            for key in ("terminal_exit_code", "terminal_cleanup_exit_code")
        )
        or follow["terminal_descendants_absent"] is not True
        or follow["terminal_sha256"] != sha256(root / TERMINAL)
        or follow["terminal_status"] != terminal["status"]
        or follow["terminal_log"] != str(follow_path / "terminal.log")
        or follow["terminal_log_sha256"] != sha256(follow_path / "terminal.log")
        or any(key in follow for key in ("error", "cleanup_error"))
        or any(handoff.observed(follow[key]) is not None for key in ("pid", "terminal_pid"))
        or handoff.marker_processes(FOLLOWUP_MARKER, follow_intent["marker"])
        or handoff.marker_processes(FOLLOWUP_CHILD_MARKER, follow["terminal_marker"])
    ):
        raise RuntimeError("NS budget predecessor original reconstruction, cleanup or restoration is incomplete")
    completed, restored, verified = (datetime.fromisoformat(value["time"]) for value in (result, state, terminal))
    if any(value.tzinfo is None for value in (completed, restored, verified)) or not completed <= restored <= verified:
        raise ValueError("NS budget predecessor terminal chronology differs")
    require_keeper_shape(terminal["keeper"])
    if any(sha256(root / name) != value for name, value in terminal["evidence_sha256"].items()):
        raise ValueError("NS budget predecessor terminal canonical evidence changed")
    marked = handoff.marker_processes(rational.CONTROLLER_MARKER, intent["marker"])
    if not set(marked).issubset({int(pid) for pid in terminal["keeper"]["generation"]}):
        raise RuntimeError("NS budget predecessor original rational controller has nonkeeper descendants")
    paths = [
        binding_path,
        state_path,
        result_path,
        helpers.parent / "controller.json",
        helpers.parent / "controller.log",
        helpers.parent / "validator.log",
        root / TERMINAL,
        follow_path / "intent.json",
        follow_path / "state.json",
        follow_path / "terminal.log",
    ]
    paths.extend(Path(entry["log"]) for entry in result["gpu_commands"])
    evidence = {str(path.relative_to(root)): sha256(path) for path in paths}
    observed = {
        "status": "original_rational_terminal_reconstructed_pending_exact_live_ns_handoff",
        "evidence_sha256": evidence,
        "restored_keeper": terminal["keeper"],
        "controller_marker": intent["marker"],
        "intervention_accepted": result["analysis"]["diagnostic_intervention_accepted"],
    }
    if evidence != {str(path.relative_to(root)): sha256(path) for path in paths}:
        raise ValueError("NS budget predecessor evidence changed during handoff authentication")
    return observed


def require_live_predecessor(root: Path, expected: dict) -> dict:
    if predecessor(root) != expected:
        raise RuntimeError("NS budget original predecessor evidence changed")
    keeper = handoff.keeper_observation()
    require_keeper_shape(keeper)
    if not same_keeper(keeper, expected["restored_keeper"]):
        raise RuntimeError("NS budget original restored keeper generation changed")
    return keeper


def require_frozen(root: Path, expected_hash: str) -> dict:
    path, helpers, _, work = locations(root)
    if root != root.resolve(strict=True) or sha256(path) != expected_hash:
        raise ValueError("NS budget frozen binding root or digest differs")
    frozen = read(path)
    cpu_path = Path(frozen["cpu_receipt"])
    if sha256(cpu_path) != frozen["cpu_receipt_sha256"]:
        raise ValueError("NS budget frozen CPU receipt identity changed")
    accepted = require_cpu(root, cpu_path)
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    if (
        frozen["status"] != STATUS
        or frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or frozen["work"] != str(work)
        or frozen["node"] != NODE
        or frozen["helpers_sha256"] != accepted["helpers_sha256"]
        or frozen["environment"] != ENVIRONMENT
        or frozen["worker_environment"] != ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": frozen["devices"][0][1]}
        or frozen["devices"] != frozen["predecessor"]["restored_keeper"]["devices"]
        or frozen["predecessor"] != predecessor(root)
        or frozen["controller_command"] != controller_command(root)
        or frozen["workers"] != 2
        or frozen["worker_timeout"] != WORKER_TIMEOUT
        or frozen["coordinates"] != [execution.coordinates(index) for index in (0, 1)]
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
    ):
        raise ValueError("NS budget frozen source, protocol or predecessor differs")
    require_archive(
        Path(frozen["archive"]),
        {f"tracking/{name}": value for name, value in frozen["helpers_sha256"].items()},
        alias=False,
    )
    for name in frozen["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"NS budget imported a different helper directory: {name}")
    if sha256(path) != expected_hash:
        raise ValueError("NS budget binding changed during authentication")
    return frozen


def require_gpu_owner(root: Path, binding_hash: str, state_path: Path, index: int) -> dict:
    _, helpers, canonical_state, _ = locations(root)
    if state_path != canonical_state or index not in (0, 1):
        raise RuntimeError("NS budget worker lacks canonical execution identity")
    frozen = require_frozen(root, binding_hash)
    intent, deadline = read(helpers.parent / "controller.json"), time.monotonic() + 30
    while True:
        state = read(state_path)
        if state.get("active_worker", {}).get("spawn_status") == "acknowledged":
            break
        if state.get("status") != "replaying" or time.monotonic() >= deadline:
            raise RuntimeError("NS budget worker lacks executable acknowledgment")
        time.sleep(0.01)
    worker = state["active_worker"]
    if (
        state["status"] != "replaying"
        or state["binding_sha256"] != binding_hash
        or state["keeper_release_complete"] is not True
        or state["predecessor"] != frozen["predecessor"]
        or worker["index"] != index
        or worker["command"] != worker_command(root, binding_hash, index)
        or worker["pid"] != os.getpid()
        or worker["controller_pid"] != os.getppid()
        or worker["controller_pid"] != state["controller_pid"]
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or intent["binding_sha256"] != binding_hash
        or intent["command"] != controller_command(root)
        or Path.cwd().resolve() != helpers
        or any(os.environ.get(name) != value for name, value in frozen["worker_environment"].items())
        or NODE not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("NS budget binding, keeper transfer or direct-child identity differs")
    controller = process_context(
        state["controller_pid"], controller_command(root), helpers, CONTROLLER_MARKER, intent["marker"], ENVIRONMENT
    )
    require_acknowledged_controller(intent, controller)
    own = process_context(
        os.getpid(), worker["command"], helpers, CHILD_MARKER, worker["marker"], frozen["worker_environment"]
    )
    if own["parent"] != controller["pid"] or own["start_ticks"] != worker["start_ticks"]:
        raise RuntimeError("NS budget worker generation differs")
    for name, path in (("node", root / "run-preparation" / f".wave7-node-{NODE}.lock"), ("manager", MANAGER_LOCK)):
        require_held_lock(controller["pid"], state["lock_descriptors"][name], path)
    devices, occupied = (
        list(csv.reader(subprocess.check_output(command, text=True).splitlines(), skipinitialspace=True))
        for command in (
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        )
    )
    if (
        devices != frozen["devices"]
        or handoff.KEEPER_PID.exists()
        or any(row != [str(own["pid"]), devices[0][1]] for row in occupied)
        or not handoff.same_generation(controller, handoff.observed(controller["pid"]))
    ):
        raise RuntimeError("NS budget physical CUDA ownership differs")
    return {
        "binding": frozen,
        "device": "cuda:0",
        "physical_uuid": devices[0][1],
        "controller_pid": controller["pid"],
        "controller_start_ticks": controller["start_ticks"],
        "worker_pid": own["pid"],
        "worker_start_ticks": own["start_ticks"],
    }
