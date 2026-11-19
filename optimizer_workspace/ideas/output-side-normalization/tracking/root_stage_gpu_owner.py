# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own a distinct two-process root-stage diagnostic after exact predecessor restoration."""

import ast
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import owned_output_handoff as handoff
import root_preparation_contract as previous
import root_stage_failed_predecessor as failed_predecessor
from fused_attribution_validation import require_keeper_shape
from fused_attribution_validation_controller import (
    process_context,
    require_acknowledged_controller,
    require_helper_bytes,
)
from prepare_selected_lr_validation import require_archive
from root_preparation_controller import inventory as previous_inventory
from saved_ns_gpu_owner import require_held_lock
from selected_lr_failure_handoff import MANAGER_LOCK, same_keeper
from selected_lr_release_owner import candidate_files, read, sha256

NODE = previous.NODE
PYTHON = previous.PYTHON
CONTROLLER_MARKER = "OUTPUT_MARIN_ROOT_STAGE_REPLAY_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_ROOT_STAGE_REPLAY_CHILD"
STATUS = "frozen_root_stage_replay_diagnostic_only"
CPU_STATUS = "root_stage_replay_cpu_archive_verified_gpu_unissued"
PREVIOUS_BINDING_SHA256 = "a98ae6fc80841f9fe0c6c9bbf5e4cad01857c6fef690bf5db31ff31cfa03ecc5"
TERMINAL_PROBE_SHA256 = "195cafd5c965cf1b601d1a02b3c93a884e91c80c6825de77c8d4e14cab85d323"
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONFAULTHANDLER": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "NCCL_INSPECTOR_ENABLE": "0",
    "TONER_SYM": "1",
}
TESTS = ("test_root_stage_trace.py", "test_root_stage_gpu.py", "test_root_preparation_reconstruction.py")
NEW_HELPERS = (
    "root_stage_gpu_owner.py",
    "root_stage_gpu_worker.py",
    "root_stage_gpu_controller.py",
    "root_stage_trace.py",
    "replay_saved_roots.py",
    "root_preparation_reconstruction.py",
    "root_stage_failed_predecessor.py",
    *TESTS,
)
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
WORKER_TIMEOUT = 3600


def locations(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / "run-preparation/root-stage-replay-binding-20261118.json",
        root / "controllers/root-stage-replay-20261118/tracking",
        root / "run-preparation/root-stage-replay-state-20261118.json",
        root / "diagnostics/root-stage-replay-20261118",
    )


def terminal_observation(root: Path) -> Path:
    return root / "run-preparation/root-terminal-observation-20261118.json"


def controller_command(root: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "root_stage_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "execute",
    ]


def worker_command(root: Path, binding_hash: str, index: int) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "root_stage_gpu_worker.py"),
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
    """Extend the original frozen closure without replacing issued bytes."""
    frozen = read(previous.locations(root)[0])
    inherited = Path(frozen["helpers"])
    require_helper_bytes(inherited, frozen["helpers_sha256"])
    sources = {name: inherited / name for name in frozen["helpers_sha256"]}
    local = Path(__file__).parent
    pending, seen = list(NEW_HELPERS), set()
    external = sys.stdlib_module_names | {
        "torch",
        "triton",
        "numpy",
        "pytest",
        "pytest_timeout",
        "swanlab",
        "dotenv",
        "experiment",
    }
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = local / name
        if name in sources:
            if path.is_file() and sha256(path) != sha256(sources[name]):
                raise ValueError(f"Root replay attempted to replace an issued helper: {name}")
            path = sources[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Root replay helper is absent or aliased: {name}")
        sources[name] = path
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            for module in modules:
                head = module.split(".")[0]
                dependency = f"{head}.py"
                if dependency in sources or (local / dependency).is_file():
                    pending.append(dependency)
                elif head not in external:
                    raise ValueError(f"Unbound root replay import: {name}: {module}")
    return sources


def require_loaded_sources(sources: dict[str, Path]) -> None:
    for name, path in sources.items():
        loaded = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if loaded is not None and sha256(Path(loaded.__file__)) != sha256(path):
            raise RuntimeError(f"Root replay loaded different helper bytes: {name}")


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


def require_cpu(root: Path, receipt_path: Path) -> dict:
    raw = receipt_path.read_bytes()
    cpu = read(receipt_path)
    out = receipt_path.parent
    sources = helper_sources(root)
    require_loaded_sources(sources)
    require_helper_bytes(out / "tracking", cpu["helpers_sha256"])
    suites = list(ET.parse(out / "tests.xml").getroot().iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        cpu["status"] != CPU_STATUS
        or cpu["root"] != str(root)
        or cpu["helpers"] != str(out / "tracking")
        or cpu["command"] != cpu_command(out)
        or cpu["helpers_sha256"] != {name: sha256(path) for name, path in sources.items()}
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["cuda_visible_devices"] != ""
        or "error" in cpu
        or not cases
        or sum(int(suite.attrib["tests"]) for suite in suites) != len(cases)
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(name).stem for name in TESTS}
        or any(case.find(field) is not None for case in cases for field in ("failure", "error", "skipped"))
        or any(int(suite.attrib[field]) for suite in suites for field in ("failures", "errors", "skipped"))
        or any(
            cpu[f"{name}_sha256"] != sha256(out / filename) or cpu[name] != str(out / filename)
            for name, filename in (("log", "cpu.log"), ("archive", "source.tar"), ("tests", "tests.xml"))
        )
        or any(sha256(out / name) != expected for name, expected in cpu["fixtures_sha256"].items())
    ):
        raise ValueError("Root replay CPU execution/source/archive evidence differs")
    require_archive(
        out / "source.tar",
        {f"tracking/{name}": value for name, value in cpu["helpers_sha256"].items()} | cpu["fixtures_sha256"],
        alias=False,
    )
    if receipt_path.read_bytes() != raw:
        raise ValueError("Root replay CPU receipt changed during authentication")
    return cpu


def require_terminal(state: dict, intent: dict, diagnostic_hash: str) -> None:
    if (
        state.get("status") != "diagnostic_complete"
        or state.get("binding_sha256") != PREVIOUS_BINDING_SHA256
        or state.get("validation_sha256") != diagnostic_hash
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or state.get("keeper_release_complete") is not True
        or state.get("marked_descendants_absent") is not True
        or "error" in state
        or "cleanup_error" in state
        or intent.get("status") != "acknowledged"
        or intent.get("binding_sha256") != PREVIOUS_BINDING_SHA256
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
    ):
        raise RuntimeError("Original root diagnostic reconstruction, cleanup or restoration is incomplete")


def predecessor(root: Path) -> dict:
    """Require independently reconstructed terminal evidence before archive or allocation work."""
    binding_path, helpers, state_path, _, gpu_path = previous.locations(root)
    state = read(state_path)
    if state.get("status") == "needs_attention":
        return failed_predecessor.predecessor(root)
    intent_path = helpers.parent / "controller.json"
    intent = read(intent_path)
    # A live state must fail before any future terminal receipt or GPU inventory is read.
    require_terminal(state, intent, state.get("validation_sha256", ""))
    if any(handoff.observed(pid) is not None for pid in (state["controller_pid"], state["child_pid"])):
        raise RuntimeError("Original root controller or validator handle remains present")
    observation_path = terminal_observation(root)
    observation = read(observation_path)
    paths = (
        binding_path,
        state_path,
        gpu_path,
        intent_path,
        Path(state["log"]),
        observation_path,
        helpers.parent / "controller.log",
    )
    hashes = {str(path.relative_to(root)): sha256(path) for path in paths}
    frozen, gpu = read(binding_path), read(gpu_path)
    require_terminal(state, intent, sha256(gpu_path))
    if (
        sha256(binding_path) != PREVIOUS_BINDING_SHA256
        or observation["status"] != "root_diagnostic_terminal_independently_reconstructed_exact_restoration_verified"
        or observation["probe_sha256"] != TERMINAL_PROBE_SHA256
        or observation["controller_pid_absent"] != state["controller_pid"]
        or observation["validator_pid_absent"] != state["child_pid"]
        or observation["marked_descendants_absent"] is not True
        or observation["commands"] != 12
        or observation["terminal_time"] != state["time"]
        or not same_keeper(observation["keeper"], state["restored_keeper"])
        or observation["gates"] != gpu["analysis"]["gates"]
        or observation["diagnostic_intervention_accepted"] != gpu["analysis"]["diagnostic_intervention_accepted"]
        or any(hashes.get(name) != expected for name, expected in observation["evidence_sha256"].items())
        or state["log_sha256"] != sha256(Path(state["log"]))
        or gpu["status"] != "completed_root_preparation_diagnostic_not_production_gate"
        or gpu["execution_binding_sha256"] != PREVIOUS_BINDING_SHA256
        or gpu["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or gpu["tolerance"] != {"rtol": 1e-5, "atol": 1e-6}
        or len(gpu["gpu_commands"]) != 12
        or any(
            entry["command"] != command or type(entry["exit_code"]) is not int or entry["exit_code"] != 0
            for entry, command in zip(gpu["gpu_commands"], frozen["commands"], strict=True)
        )
        or frozen["files_sha256"] != candidate_files(Path(frozen["candidate"]))
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
    ):
        raise ValueError("Original root terminal reconstruction or source identity differs")
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    require_archive(Path(frozen["archive"]), previous_inventory(frozen), alias=True)
    require_keeper_shape(state["restored_keeper"])
    if hashes != {str(path.relative_to(root)): sha256(path) for path in paths}:
        raise ValueError("Original root terminal evidence changed during authentication")
    return {
        "status": "independent_root_terminal_exact_restoration_pending_live_replay_handoff",
        "evidence_sha256": hashes,
        "controller_pid": state["controller_pid"],
        "child_pid": state["child_pid"],
        "controller_marker": intent["marker"],
        "child_marker": state["child_marker"],
        "restored_keeper": state["restored_keeper"],
    }


def require_live_predecessor(root: Path, expected: dict) -> dict:
    if predecessor(root) != expected:
        raise RuntimeError("Original root terminal predecessor changed")
    keeper = handoff.keeper_observation()
    require_keeper_shape(keeper)
    if not same_keeper(keeper, expected["restored_keeper"]):
        raise RuntimeError("Original root exact restored keeper generation changed")
    if not set(handoff.marker_processes(previous.CONTROLLER_MARKER, expected["controller_marker"])).issubset(
        {int(pid) for pid in keeper["generation"]}
    ) or handoff.marker_processes(previous.CHILD_MARKER, expected["child_marker"]):
        raise RuntimeError("Original root owner markers have nonkeeper descendants")
    return keeper


def require_frozen(root: Path, expected_hash: str) -> dict:
    binding_path, helpers, _, work = locations(root)
    if root != root.resolve(strict=True) or sha256(binding_path) != expected_hash:
        raise ValueError("Root replay binding identity differs")
    frozen = read(binding_path)
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    cpu_path = Path(frozen["cpu_receipt"])
    if sha256(cpu_path) != frozen["cpu_receipt_sha256"]:
        raise ValueError("Root replay CPU receipt bytes differ")
    cpu = require_cpu(root, cpu_path)
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
        raise ValueError("Root replay frozen protocol/source/predecessor differs")
    require_archive(
        Path(frozen["archive"]),
        {f"tracking/{name}": value for name, value in frozen["helpers_sha256"].items()},
        alias=False,
    )
    for name in frozen["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Root replay imported a mutable helper: {name}")
    if sha256(binding_path) != expected_hash:
        raise ValueError("Root replay binding changed during authentication")
    return frozen


def require_gpu_owner(root: Path, binding_hash: str, state_path: Path, index: int) -> dict:
    _, helpers, canonical_state, _ = locations(root)
    if state_path != canonical_state or index not in (0, 1):
        raise RuntimeError("Root replay has no canonical worker identity")
    frozen = require_frozen(root, binding_hash)
    intent = read(helpers.parent / "controller.json")
    deadline = time.monotonic() + 30
    while True:
        state = read(state_path)
        if state.get("active_worker", {}).get("spawn_status") == "acknowledged":
            break
        if state.get("status") != "replaying" or time.monotonic() >= deadline:
            raise RuntimeError("Root replay worker lacks executable acknowledgment")
        time.sleep(0.01)
    worker = state["active_worker"]
    environment = frozen["worker_environment"]
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
        or Path.cwd().resolve() != helpers
        or any(os.environ.get(name) != expected for name, expected in environment.items())
        or NODE not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("Root replay binding, transfer or direct-child identity differs")
    controller = process_context(
        state["controller_pid"], controller_command(root), helpers, CONTROLLER_MARKER, intent["marker"], ENVIRONMENT
    )
    require_acknowledged_controller(intent, controller)
    process = process_context(os.getpid(), worker["command"], helpers, CHILD_MARKER, worker["marker"], environment)
    if process["parent"] != controller["pid"] or process["start_ticks"] != worker["start_ticks"]:
        raise RuntimeError("Root replay worker generation or ancestry differs")
    for name, path in (("node", root / "run-preparation" / f".wave7-node-{NODE}.lock"), ("manager", MANAGER_LOCK)):
        require_held_lock(controller["pid"], state["lock_descriptors"][name], path)
    devices = list(
        csv.reader(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True
            ).splitlines(),
            skipinitialspace=True,
        )
    )
    occupied = list(
        csv.reader(
            subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True
            ).splitlines(),
            skipinitialspace=True,
        )
    )
    if (
        devices != frozen["devices"]
        or environment["CUDA_VISIBLE_DEVICES"] != devices[0][1]
        or handoff.KEEPER_PID.exists()
        or any(row != [str(os.getpid()), devices[0][1]] for row in occupied)
        or not handoff.same_generation(controller, handoff.observed(controller["pid"]))
    ):
        raise RuntimeError("Root replay physical mapping, compute ownership or parent changed")
    return {
        "binding": frozen,
        "device": "cuda:0",
        "physical_uuid": devices[0][1],
        "controller_pid": controller["pid"],
        "controller_start_ticks": controller["start_ticks"],
        "worker_pid": process["pid"],
        "worker_start_ticks": process["start_ticks"],
    }
