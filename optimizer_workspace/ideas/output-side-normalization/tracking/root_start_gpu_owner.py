# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate a separate saved-start accuracy trial and its completed root-stage predecessor."""

import ast
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import owned_output_handoff as handoff
from fused_attribution_validation import require_keeper_shape
from fused_attribution_validation_controller import (
    process_context,
    require_acknowledged_controller,
    require_helper_bytes,
)
from prepare_selected_lr_validation import require_archive
from root_stage_failed_predecessor import file_identity
from saved_ns_gpu_owner import require_held_lock
from selected_lr_failure_handoff import MANAGER_LOCK, keeper_guard, same_keeper
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

NODE = "28.216.17.9"
PYTHON = "/opt/conda/envs/torch-base/bin/python"
CONTROLLER_MARKER = "OUTPUT_MARIN_ROOT_START_ACCURACY_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_ROOT_START_ACCURACY_CHILD"
STATUS = "frozen_saved_start_accuracy_trial_not_continuation_or_production"
CPU_STATUS = "root_start_gpu_owner_cpu_archive_verified_gpu_unissued"
WORKER_TIMEOUT = 3600
PARENT = "run-preparation/root-start-execution-cpu/20261118T1922/receipt.json"
PARENT_SHA256 = "e7b824e87631c417843cb1ecb0815b52e55d1e33a7083dea8f1c1e86a4a01904"
PARENT_STATUS = "root_start_execution_reconstruction_cpu_archive_verified_cuda_owner_unissued"
TERMINAL = "run-preparation/root-stage-terminal-independently-reconstructed-20261118T1718.json"
TERMINAL_SHA256 = "7eaec796d8a2cb2261afddd607c697e4ab32309f5303b01c828e3b94afbfc4ed"
FOLLOW_WORK = "run-preparation/root-stage-terminal-followup-20261118T1718"
FOLLOW_STATE_SHA256 = "fad83110413def89e8eed68221ed4834536cefe4a5f6c93534bdb41ced0ce8db"
PRECISION = "run-preparation/root-stage-precision-independent-20261118T1718.json"
PRECISION_SHA256 = "85ed5d3e978bb870c9341da5278d90eb12ab8bdd39a1c5b69db1183839b2ba58"
PREVIOUS_BINDING = "run-preparation/root-stage-replay-binding-20261118.json"
PREVIOUS_BINDING_SHA256 = "00964875514bcfc73bb11ab51847692ae538dbd39b31740cf73b139f71f6cc93"
PREVIOUS_CONTROLLER_MARKER = "OUTPUT_MARIN_ROOT_STAGE_REPLAY_CONTROLLER"
PREVIOUS_CHILD_MARKER = "OUTPUT_MARIN_ROOT_STAGE_REPLAY_CHILD"
SEAL = "run-preparation/root-start-predecessor-seal-20261118.json"
SEAL_STATUS = "root_stage_terminal_and_precision_sealed_for_separate_start_accuracy_trial"
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONFAULTHANDLER": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "NCCL_INSPECTOR_ENABLE": "0",
    "TONER_SYM": "1",
}
TESTS = (
    "test_root_stage_trace.py",
    "test_root_stage_gpu.py",
    "test_root_preparation_reconstruction.py",
    "test_analyze_root_stage_precision.py",
    "test_root_start_rational.py",
    "test_root_start_accuracy.py",
    "test_root_start_execution.py",
    "test_root_start_gpu.py",
)
NEW_HELPERS = ("root_start_gpu_owner.py", "root_start_gpu_worker.py", "root_start_gpu_controller.py", TESTS[-1])
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)


def locations(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / "run-preparation/root-start-accuracy-binding-20261118.json",
        root / "controllers/root-start-accuracy-20261118/tracking",
        root / "run-preparation/root-start-accuracy-state-20261118.json",
        root / "diagnostics/root-start-accuracy-20261118",
    )


def controller_command(root: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "root_start_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "execute",
    ]


def worker_command(root: Path, binding_hash: str, index: int) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "root_start_gpu_worker.py"),
        "--root",
        str(root),
        "--binding-sha256",
        binding_hash,
        "--state",
        str(locations(root)[2]),
        "--worker-index",
        str(index),
    ]


def require_cpu_archive(root: Path, path: Path, expected_status: str, count: int | None) -> dict:
    cpu = read(path)
    helpers = path.parent / "tracking"
    require_helper_bytes(helpers, cpu["helpers_sha256"])
    suites = list(ET.parse(cpu["tests"]).getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        cpu["status"] != expected_status
        or cpu["root"] != str(root)
        or cpu["helpers"] != str(helpers)
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["cuda_visible_devices"] != ""
        or "error" in cpu
        or not cases
        or (count is not None and len(cases) != count)
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or any(int(suite.attrib[key]) for suite in suites for key in ("errors", "failures", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("error", "failure", "skipped"))
        or any(
            cpu[name] != str(path.parent / filename) or cpu[f"{name}_sha256"] != sha256(Path(cpu[name]))
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or any(sha256(path.parent / name) != value for name, value in cpu["fixtures_sha256"].items())
    ):
        raise ValueError("Start owner CPU execution, source or complete test evidence differs")
    require_archive(
        Path(cpu["archive"]),
        {f"tracking/{name}": value for name, value in cpu["helpers_sha256"].items()} | cpu["fixtures_sha256"],
        alias=False,
    )
    return cpu


def helper_sources(root: Path) -> dict[str, Path]:
    path = root / PARENT
    if sha256(path) != PARENT_SHA256:
        raise ValueError("Start owner parent CPU receipt changed")
    parent = require_cpu_archive(root, path, PARENT_STATUS, 109)
    sources = {name: Path(parent["helpers"]) / name for name in parent["helpers_sha256"]}
    sources.update({name: Path(__file__).parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError("Start owner new helper is absent or aliased")
        for node in ast.walk(ast.parse(path.read_bytes(), filename=name)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("Start owner import escapes its accepted closure")
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
        "--import-mode=importlib",
        "-m",
        SAFE_MARKERS,
        "-q",
        *(str(out / "tracking" / name) for name in TESTS),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def require_cpu(root: Path, path: Path) -> dict:
    cpu = require_cpu_archive(root, path, CPU_STATUS, None)
    sources = helper_sources(root)
    cases = list(ET.parse(cpu["tests"]).getroot().iter("testcase"))
    if (
        cpu["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or cpu["command"] != cpu_command(path.parent)
        or cpu["helpers_sha256"] != {name: sha256(source) for name, source in sources.items()}
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(name).stem for name in TESTS}
    ):
        raise ValueError("Start owner CPU source closure or tested module coverage differs")
    for name, source in sources.items():
        module = sys.modules.get(Path(name).stem)
        if module is not None and sha256(Path(module.__file__)) != sha256(source):
            raise RuntimeError(f"Start owner loaded different accepted bytes: {name}")
    return cpu


def predecessor_records(root: Path) -> tuple[dict, dict, dict, dict]:
    pinned = {
        TERMINAL: TERMINAL_SHA256,
        PRECISION: PRECISION_SHA256,
        f"{FOLLOW_WORK}/state.json": FOLLOW_STATE_SHA256,
        PREVIOUS_BINDING: PREVIOUS_BINDING_SHA256,
    }
    if any(sha256(root / name) != value for name, value in pinned.items()):
        raise ValueError("Start trial original terminal or precision evidence changed")
    terminal, follow, binding = (read(root / name) for name in (TERMINAL, f"{FOLLOW_WORK}/state.json", PREVIOUS_BINDING))
    state = read(root / "run-preparation/root-stage-replay-state-20261118.json")
    intent = read(Path(binding["helpers"]).parent / "controller.json")
    if (
        terminal["status"] != "root_stage_replay_terminal_independently_reconstructed_exact_restoration_verified"
        or terminal["binding_sha256"] != PREVIOUS_BINDING_SHA256
        or terminal["intermediates_interpretable"] is not True
        or terminal["terminal_time"] != state["time"]
        or state["status"] != "diagnostic_complete"
        or state["binding_sha256"] != PREVIOUS_BINDING_SHA256
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or intent["status"] != "acknowledged"
        or intent["binding_sha256"] != PREVIOUS_BINDING_SHA256
        or intent["command"] != binding["controller_command"]
        or terminal["controller_pid_absent"] != state["controller_pid"]
        or terminal["verifier_sha256"] != binding["helpers_sha256"]["root_stage_gpu_controller.py"]
        or terminal["comparison_time"] != read(root / "diagnostics/root-stage-replay-20261118/comparison.json")["time"]
        or terminal["worker_pids_absent"] != [entry["pid"] for entry in state["executions"]]
        or len(state["executions"]) != 2
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or state["keeper_release_complete"] is not True
        or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
        or not same_keeper(terminal["keeper"], state["restored_keeper"])
        or follow["status"] != "terminal_and_precision_reconstructed"
        or follow["terminal_sha256"] != TERMINAL_SHA256
        or follow["precision_sha256"] != PRECISION_SHA256
        or [entry["phase"] for entry in follow["commands"]] != ["terminal", "precision"]
        or any(
            entry["status"] != "complete"
            or any(type(entry[key]) is not int or entry[key] != 0 for key in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
            or entry["log_sha256"] != sha256(Path(entry["log"]))
            for entry in follow["commands"]
        )
        or any(
            entry["spawn_status"] != "acknowledged"
            or any(type(entry[key]) is not int or entry[key] != 0 for key in ("exit_code", "cleanup_exit_code"))
            or entry["marked_descendants_absent"] is not True
            for entry in state["executions"]
        )
    ):
        raise RuntimeError("Start trial predecessor reconstruction or owner cleanup is incomplete")
    require_keeper_shape(terminal["keeper"])
    return terminal, follow, binding, intent


def require_predecessor_absent(root: Path, terminal: dict, follow: dict, intent: dict) -> None:
    pids = [terminal["controller_pid_absent"], *terminal["worker_pids_absent"], follow["pid"]]
    pids.extend(entry["pid"] for entry in follow["commands"])
    if any(handoff.observed(pid) is not None for pid in pids):
        raise RuntimeError("Start trial original predecessor handle remains present")
    if any(
        handoff.marker_processes(PREVIOUS_CHILD_MARKER, entry["marker"])
        for entry in read(root / "run-preparation/root-stage-replay-state-20261118.json")["executions"]
    ):
        raise RuntimeError("Start trial original GPU worker descendants remain")
    runner = read(root / FOLLOW_WORK / "intent.json")
    if handoff.marker_processes("OUTPUT_MARIN_ROOT_STAGE_TERMINAL_FOLLOWUP", runner["marker"]) or any(
        handoff.marker_processes("OUTPUT_MARIN_ROOT_STAGE_READONLY_CHILD", entry["marker"])
        for entry in follow["commands"]
    ):
        raise RuntimeError("Start trial original read-only follow-up descendants remain")
    marked = handoff.marker_processes(PREVIOUS_CONTROLLER_MARKER, intent["marker"])
    if not set(marked).issubset({int(pid) for pid in terminal["keeper"]["generation"]}):
        raise RuntimeError("Start trial original controller has nonkeeper descendants")


def seal(root: Path) -> dict:
    path = root / SEAL
    if path.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise FileExistsError("Start predecessor sealing requires hidden CUDA and unused evidence")
    terminal, follow, binding, intent = predecessor_records(root)
    require_predecessor_absent(root, terminal, follow, intent)
    runner = read(root / FOLLOW_WORK / "intent.json")
    if sha256(Path(runner["command"][2])) != runner["runner_sha256"]:
        raise ValueError("Start predecessor original follow-up source changed")
    require_helper_bytes(Path(binding["helpers"]), binding["helpers_sha256"])
    if sha256(Path(binding["archive"])) != binding["archive_sha256"]:
        raise ValueError("Original root-stage issued archive changed")
    require_archive(
        Path(binding["archive"]),
        {f"tracking/{name}": value for name, value in binding["helpers_sha256"].items()},
        alias=False,
    )
    evidence = dict(terminal["evidence_sha256"])
    evidence.update(
        {
            name: sha256(root / name)
            for name in (
                TERMINAL,
                PRECISION,
                f"{FOLLOW_WORK}/state.json",
                f"{FOLLOW_WORK}/intent.json",
                f"{FOLLOW_WORK}/terminal.log",
                f"{FOLLOW_WORK}/precision.log",
            )
        }
    )
    identities = {name: file_identity(root / name) for name in evidence}
    if any(sha256(root / name) != value for name, value in evidence.items()) or identities != {
        name: file_identity(root / name) for name in evidence
    }:
        raise ValueError("Start predecessor evidence changed while sealing")
    with keeper_guard():
        keeper = handoff.keeper_observation()
        require_keeper_shape(keeper)
        if not same_keeper(keeper, terminal["keeper"]):
            raise RuntimeError("Start predecessor exact restored keeper changed")
    result = {
        "status": SEAL_STATUS,
        "evidence_sha256": evidence,
        "file_identities": identities,
        "restored_keeper": terminal["keeper"],
        "terminal_sha256": TERMINAL_SHA256,
        "precision_sha256": PRECISION_SHA256,
        "sealer_sha256": sha256(Path(__file__)),
        "scope": (
            "Sealed completed diagnostic evidence only; historical attribution and production gates remain unchanged."
        ),
    }
    write(path, result)
    return result


def predecessor(root: Path) -> dict:
    terminal, follow, _, intent = predecessor_records(root)
    require_predecessor_absent(root, terminal, follow, intent)
    sealed = read(root / SEAL)
    if (
        sealed["status"] != SEAL_STATUS
        or sealed["terminal_sha256"] != TERMINAL_SHA256
        or sealed["precision_sha256"] != PRECISION_SHA256
        or sealed["sealer_sha256"] != sha256(Path(__file__))
        or not same_keeper(sealed["restored_keeper"], terminal["keeper"])
        or sealed["file_identities"] != {name: file_identity(root / name) for name in sealed["evidence_sha256"]}
        or any(sealed["evidence_sha256"].get(name) != value for name, value in terminal["evidence_sha256"].items())
        or any(
            sha256(root / name) != value for name, value in sealed["evidence_sha256"].items() if not name.endswith(".pt")
        )
    ):
        raise ValueError("Start predecessor seal, canonical evidence identity or receipt bytes changed")
    return {
        "status": SEAL_STATUS,
        "seal_sha256": sha256(root / SEAL),
        "terminal_sha256": TERMINAL_SHA256,
        "precision_sha256": PRECISION_SHA256,
        "restored_keeper": terminal["keeper"],
        "controller_marker": intent["marker"],
    }


def require_live_predecessor(root: Path, expected: dict) -> dict:
    if predecessor(root) != expected:
        raise RuntimeError("Start accuracy predecessor changed")
    keeper = handoff.keeper_observation()
    require_keeper_shape(keeper)
    if not same_keeper(keeper, expected["restored_keeper"]):
        raise RuntimeError("Start accuracy exact predecessor keeper changed")
    return keeper


def require_frozen(root: Path, expected_hash: str) -> dict:
    path, helpers, _, work = locations(root)
    if root != root.resolve(strict=True) or sha256(path) != expected_hash:
        raise ValueError("Start accuracy frozen binding identity differs")
    frozen = read(path)
    cpu_path = Path(frozen["cpu_receipt"])
    if sha256(cpu_path) != frozen["cpu_receipt_sha256"]:
        raise ValueError("Start accuracy CPU receipt identity changed")
    cpu = require_cpu(root, cpu_path)
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    if (
        frozen["status"] != STATUS
        or frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or frozen["work"] != str(work)
        or frozen["node"] != NODE
        or frozen["helpers_sha256"] != cpu["helpers_sha256"]
        or frozen["environment"] != ENVIRONMENT
        or frozen["worker_environment"] != ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": frozen["devices"][0][1]}
        or frozen["devices"] != frozen["predecessor"]["restored_keeper"]["devices"]
        or frozen["predecessor"] != predecessor(root)
        or frozen["controller_command"] != controller_command(root)
        or frozen["workers"] != 2
        or frozen["worker_timeout"] != WORKER_TIMEOUT
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
    ):
        raise ValueError("Start accuracy frozen source, protocol or predecessor differs")
    require_archive(
        Path(frozen["archive"]),
        {f"tracking/{name}": value for name, value in frozen["helpers_sha256"].items()},
        alias=False,
    )
    for name in frozen["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Start accuracy imported a different helper directory: {name}")
    if sha256(path) != expected_hash:
        raise ValueError("Start accuracy binding changed during authentication")
    return frozen


def require_gpu_owner(root: Path, binding_hash: str, state_path: Path, index: int) -> dict:
    _, helpers, canonical_state, _ = locations(root)
    if state_path != canonical_state or index not in (0, 1):
        raise RuntimeError("Start accuracy worker lacks canonical execution identity")
    frozen = require_frozen(root, binding_hash)
    intent = read(helpers.parent / "controller.json")
    deadline = time.monotonic() + 30
    while True:
        state = read(state_path)
        if state.get("active_worker", {}).get("spawn_status") == "acknowledged":
            break
        if state.get("status") != "replaying" or time.monotonic() >= deadline:
            raise RuntimeError("Start accuracy worker lacks executable acknowledgment")
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
        raise RuntimeError("Start accuracy binding, keeper transfer or direct-child identity differs")
    controller = process_context(
        state["controller_pid"], controller_command(root), helpers, CONTROLLER_MARKER, intent["marker"], ENVIRONMENT
    )
    require_acknowledged_controller(intent, controller)
    own = process_context(
        os.getpid(), worker["command"], helpers, CHILD_MARKER, worker["marker"], frozen["worker_environment"]
    )
    if own["parent"] != controller["pid"] or own["start_ticks"] != worker["start_ticks"]:
        raise RuntimeError("Start accuracy worker generation differs")
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
        raise RuntimeError("Start accuracy physical CUDA ownership differs")
    return {
        "binding": frozen,
        "device": "cuda:0",
        "physical_uuid": devices[0][1],
        "controller_pid": controller["pid"],
        "controller_start_ticks": controller["start_ticks"],
        "worker_pid": own["pid"],
        "worker_start_ticks": own["start_ticks"],
    }
