# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own only two source-bound saved-input CUDA replay processes, never training."""

import ast
import csv
import fcntl
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import owned_output_handoff as handoff
from diagnose_fused_diagonal import MOMENT_TOLERANCE
from fused_attribution_validation_controller import process_context, require_acknowledged_controller
from fused_continuation_contract import CHILD_MARKER as PREVIOUS_CHILD_MARKER
from fused_continuation_contract import CONTROLLER_MARKER as PREVIOUS_CONTROLLER_MARKER
from fused_continuation_contract import locations as previous_locations
from prepare_selected_lr_validation import require_archive
from selected_lr_failure_handoff import MANAGER_LOCK, same_keeper
from selected_lr_release_owner import candidate_files, read, sha256

NODE = "28.216.17.9"
PYTHON = "/opt/conda/envs/torch-base/bin/python"
CONTROLLER_MARKER = "OUTPUT_MARIN_SAVED_NS_REPLAY_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_SAVED_NS_REPLAY_CHILD"
STATUS = "frozen_saved_ns_replay_diagnostic_only"
RECEIPT_SHA256 = "b09b6276162b6811aef15ab0b729361c45a0ed1202c3c30fc133894f13837a82"
PREVIOUS_BINDING_SHA256 = "3eb36cfe0b1f2387bd030a3d0b419f36c9abaf6b87bd5767ff96b0aa9360e277"
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONFAULTHANDLER": "1",
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}
EXITED = ("Z", "X", "x")
CPU_STATUS = "saved_ns_gpu_adapter_cpu_archive_verified_gpu_unissued"
TESTS = (
    "test_saved_ns_gpu_owner.py",
    "test_saved_ns_gpu_adapter.py",
    "test_saved_ns_gpu_worker.py",
    "test_replay_saved_ns.py",
)
NEW_HELPERS = (
    "saved_ns_gpu_adapter.py",
    "saved_ns_gpu_owner.py",
    "saved_ns_gpu_worker.py",
    "replay_saved_ns.py",
    "analyze_fused_continuation_stages.py",
    "benchmark_diagonal.py",
    "diagnose_continuation.py",
    "diagnose_fused_diagonal.py",
    *TESTS,
)
WORKER_SOURCES = (
    "saved_ns_gpu_worker.py",
    "saved_ns_gpu_owner.py",
    "replay_saved_ns.py",
    "analyze_fused_continuation_stages.py",
    "diagnose_continuation.py",
    "diagnose_fused_diagonal.py",
    "benchmark_diagonal.py",
)
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)


def helper_sources(root: Path) -> dict[str, Path]:
    """Bind the complete new closure without replacing inherited owner implementations."""
    previous = read(previous_locations(root)[0])
    source = Path(previous["helpers"])
    sources = {name: source / name for name in previous["helpers_sha256"]}
    if any(sha256(path) != previous["helpers_sha256"][name] for name, path in sources.items()):
        raise ValueError("Original continuation helper bytes changed")
    local = Path(__file__).parent
    guard = local / "replay_pytest_guard.py"
    sources["replay_pytest_guard.py"] = (
        guard if guard.is_file() else Path(__file__).resolve().parents[4] / "infra/ci/pytest_timeout_guard.py"
    )
    for name in NEW_HELPERS:
        path = local / name
        if name in sources and sha256(path) != sha256(sources[name]):
            raise ValueError("New replay helper would replace an original ownership helper")
        sources[name] = path
    pending, checked = list(NEW_HELPERS), set()
    while pending:
        name = pending.pop()
        if name in checked or not name.endswith(".py"):
            continue
        checked.add(name)
        path = sources[name]
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module] if node.module else []
            else:
                continue
            for module in modules:
                head = module.split(".")[0]
                dependency = f"{head}.py"
                if dependency in sources:
                    pending.append(dependency)
                elif (local / dependency).is_file():
                    sources[dependency] = local / dependency
                    pending.append(dependency)
                elif head not in sys.stdlib_module_names | {
                    "torch",
                    "triton",
                    "pytest",
                    "pytest_timeout",
                    "swanlab",
                    "dotenv",
                    "experiment",
                }:
                    raise ValueError(f"Unbound replay helper import: {name}: {module}")
    return sources


def require_loaded_sources(sources: dict[str, Path]) -> None:
    """Authenticate loaded helper bytes before CPU acceptance or allocation preparation."""
    for name, expected in sources.items():
        module = sys.modules.get(Path(name).stem)
        if module is not None and (
            Path(module.__file__).is_symlink() or sha256(Path(module.__file__)) != sha256(expected)
        ):
            raise RuntimeError("Saved NS replay preparation loaded different helper bytes")


def cpu_command(helpers: Path) -> list[str]:
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
        "-p",
        "no:cacheprovider",
        "-p",
        "replay_pytest_guard",
        "--timeout=60",
        "--session-timeout=600",
        "--import-mode=importlib",
        "-m",
        SAFE_MARKERS,
        "-q",
        *(str(helpers / name) for name in TESTS),
    ]


def require_cpu_contract(root: Path, cpu: dict, frozen: dict) -> None:
    """Reject substituted test commands or omitted owner dependencies independently."""
    expected = {name: sha256(path) for name, path in helper_sources(root).items()}
    if (
        cpu["command"] != cpu_command(Path(cpu["helpers"]))
        or frozen["cpu_command"] != cpu_command(Path(cpu["helpers"]))
        or cpu["helpers_sha256"] != expected
        or frozen["helpers_sha256"] != expected
    ):
        raise ValueError("Saved NS replay lacks the prescribed CPU command or complete source closure")


def worker_sources(frozen: dict) -> dict[str, str]:
    """Identify every helper held open across the numerical worker's execution."""
    helpers = Path(frozen["helpers"])
    return {str(helpers / name): frozen["helpers_sha256"][name] for name in WORKER_SOURCES}


def locations(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / "run-preparation/saved-ns-replay-binding-20261118.json",
        root / "controllers/saved-ns-replay-20261118/tracking",
        root / "run-preparation/saved-ns-replay-state-20261118.json",
        root / "diagnostics/saved-ns-replay-20261118",
    )


def controller_command(root: Path) -> list[str]:
    return [PYTHON, "-P", str(locations(root)[1] / "saved_ns_gpu_adapter.py"), "--root", str(root), "--phase", "execute"]


def worker_command(root: Path, binding_sha256: str, index: int) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "saved_ns_gpu_worker.py"),
        "--root",
        str(root),
        "--binding-sha256",
        binding_sha256,
        "--state",
        str(locations(root)[2]),
        "--worker-index",
        str(index),
    ]


def require_terminal(state: dict, intent: dict, diagnostic_sha256: str) -> None:
    """Require the original owner's reconstruction and cleanup, not just six commands."""
    if (
        state.get("status") != "diagnostic_complete"
        or state.get("binding_sha256") != PREVIOUS_BINDING_SHA256
        or state.get("validation_sha256") != diagnostic_sha256
        or type(state.get("exit_code")) is not int
        or state["exit_code"] != 0
        or type(state.get("cleanup_exit_code")) is not int
        or state["cleanup_exit_code"] != 0
        or state.get("marked_descendants_absent") is not True
        or type(state.get("keeper_restore")) is not int
        or state["keeper_restore"] != 0
        or state.get("keeper_release_complete") is not True
        or "error" in state
        or "cleanup_error" in state
        or intent.get("status") != "acknowledged"
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or intent["binding_sha256"] != PREVIOUS_BINDING_SHA256
        or not state.get("restored_keeper")
    ):
        raise RuntimeError("Original continuation reconstruction, cleanup or restoration is incomplete")


def predecessor(root: Path) -> dict:
    """Authenticate terminal source-bound evidence without rerunning checkpoint reconstruction."""
    binding_path, helpers, state_path, _, receipt_path = previous_locations(root)
    intent_path = helpers.parent / "controller.json"
    state, intent = read(state_path), read(intent_path)
    # Reject the live owner before source/archive work or any allocation action.
    require_terminal(state, intent, RECEIPT_SHA256)
    for pid in (state["controller_pid"], state["child_pid"]):
        process = handoff.observed(pid)
        if process is not None and process["status"] not in EXITED:
            raise RuntimeError("Original continuation controller/validator handle is still live")
    paths = (binding_path, state_path, receipt_path, intent_path, Path(state["log"]), helpers.parent / "controller.log")
    hashes = {str(path.relative_to(root)): sha256(path) for path in paths}
    if sha256(binding_path) != PREVIOUS_BINDING_SHA256 or sha256(receipt_path) != RECEIPT_SHA256:
        raise ValueError("Original continuation binding or diagnostic bytes differ")
    frozen, receipt = read(binding_path), read(receipt_path)
    candidate = Path(frozen["candidate"])
    prepared = read(candidate / "fused-continuation-preparation.json")
    if (
        frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or candidate_files(candidate) != frozen["files_sha256"]
        or frozen["preparation_sha256"] != sha256(candidate / "fused-continuation-preparation.json")
        or state["candidate"] != str(candidate)
        or state["log_sha256"] != sha256(Path(state["log"]))
        or receipt["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or receipt["status"] != "completed_fused_continuation_diagnostic_not_production_gate"
        or receipt["execution_binding_sha256"] != PREVIOUS_BINDING_SHA256
        or receipt["tolerance"] != MOMENT_TOLERANCE
        or any(receipt["analysis"][gate] is not True for gate in ("trace_transparent", "extra_checkpoint_transparent"))
        or receipt["commands"] != frozen["commands"]
        or len(receipt["gpu_commands"]) != 6
        or any(
            execution["command"] != command or type(execution["exit_code"]) is not int or execution["exit_code"] != 0
            for execution, command in zip(receipt["gpu_commands"], frozen["commands"], strict=True)
        )
        or any(
            (helpers / name).is_symlink() or sha256(helpers / name) != value
            for name, value in frozen["helpers_sha256"].items()
        )
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
    ):
        raise ValueError("Terminal continuation source, log or archive identity differs")
    inventory = (
        frozen["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {
            "fused-continuation-preparation.json": frozen["preparation_sha256"],
            "candidate-cpu-receipt.json": frozen["cpu_receipt_sha256"],
            "controller-cpu-receipt.json": frozen["controller_cpu_receipt_sha256"],
        }
        | {f"tracking/{name}": value for name, value in frozen["helpers_sha256"].items()}
    )
    require_archive(Path(frozen["archive"]), inventory, alias=True)
    if hashes != {str(path.relative_to(root)): sha256(path) for path in paths}:
        raise ValueError("Original terminal evidence changed during authentication")
    return {
        "status": "original_continuation_reconstructed_restored_pending_live_handoff",
        "evidence_sha256": hashes,
        "controller_pid": state["controller_pid"],
        "controller_start_ticks": state["controller_start_ticks"],
        "child_pid": state["child_pid"],
        "child_start_ticks": state["child_start_ticks"],
        "controller_marker": intent["marker"],
        "child_marker": state["child_marker"],
        "restored_keeper": state["restored_keeper"],
    }


def require_live_predecessor(root: Path, expected: dict) -> dict:
    """Require the same restored keeper, permitting only its inherited controller marker."""
    if predecessor(root) != expected:
        raise RuntimeError("Original terminal predecessor evidence changed")
    keeper = handoff.keeper_observation()
    if not same_keeper(keeper, expected["restored_keeper"]):
        raise RuntimeError("Original diagnostic restored keeper generation changed")
    marked = handoff.marker_processes(PREVIOUS_CONTROLLER_MARKER, expected["controller_marker"])
    if not set(marked).issubset({int(pid) for pid in keeper["generation"]}):
        raise RuntimeError("Original continuation controller marker has nonkeeper descendants")
    if handoff.marker_processes(PREVIOUS_CHILD_MARKER, expected["child_marker"]):
        raise RuntimeError("Original continuation child marker is still live")
    return keeper


def require_frozen(root: Path, expected_sha256: str) -> dict:
    binding_path, helpers, _, work = locations(root)
    if root != root.resolve(strict=True) or sha256(binding_path) != expected_sha256:
        raise ValueError("Saved NS replay binding root or digest differs")
    frozen = read(binding_path)
    cpu_path = Path(frozen["cpu_receipt"])
    if sha256(cpu_path) != frozen["cpu_receipt_sha256"]:
        raise ValueError("Saved NS replay CPU receipt bytes differ")
    cpu = read(cpu_path)
    require_cpu_contract(root, cpu, frozen)
    if (
        cpu["status"] != CPU_STATUS
        or cpu["root"] != str(root)
        or cpu["helpers_sha256"] != frozen["helpers_sha256"]
        or cpu["command"] != cpu_command(Path(cpu["helpers"]))
        or frozen["cpu_command"] != cpu_command(Path(cpu["helpers"]))
        or frozen["helpers_sha256"] != {name: sha256(path) for name, path in helper_sources(root).items()}
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["cuda_visible_devices"] != ""
        or cpu["archive_sha256"] != sha256(Path(cpu["archive"]))
        or cpu["log_sha256"] != sha256(Path(cpu["log"]))
        or any(sha256(Path(cpu["helpers"]) / name) != value for name, value in cpu["helpers_sha256"].items())
    ):
        raise ValueError("Saved NS replay frozen closure is not the CPU-tested archive")
    require_archive(
        Path(cpu["archive"]),
        {f"tracking/{name}": value for name, value in cpu["helpers_sha256"].items()} | cpu["fixtures_sha256"],
        alias=False,
    )
    if (
        frozen["status"] != STATUS
        or frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or frozen["work"] != str(work)
        or frozen["controller_command"] != controller_command(root)
        or frozen["environment"] != ENVIRONMENT
        or frozen["node"] != NODE
        or frozen["worker_environment"] != ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": frozen["devices"][0][1]}
        or frozen["devices"] != frozen["predecessor"]["restored_keeper"]["devices"]
        or frozen["workers"] != 2
        or frozen["worker_timeout"] != 600
        or helpers.is_symlink()
        or any(
            Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != value
            for name, value in frozen["helpers_sha256"].items()
        )
        or sha256(Path(frozen["archive"])) != frozen["archive_sha256"]
        or frozen["predecessor"] != predecessor(root)
    ):
        raise ValueError("Saved NS replay frozen sources, protocol or predecessor differs")
    require_archive(
        Path(frozen["archive"]),
        {f"tracking/{name}": value for name, value in frozen["helpers_sha256"].items()},
        alias=False,
    )
    for name in frozen["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem)
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError("Saved NS replay imported a mutable helper")
    if sha256(binding_path) != expected_sha256:
        raise ValueError("Saved NS replay binding changed during authentication")
    return frozen


def require_held_lock(pid: int, descriptor: int, path: Path) -> None:
    """Verify parent FD ownership via fdinfo and independent nonblocking contention."""
    if path.is_symlink() or path != path.resolve(strict=True):
        raise RuntimeError("Replay lock path is not a canonical regular file")
    expected = path.stat()
    target = handoff.PROC_ROOT / str(pid) / "fd" / str(descriptor)
    actual = target.stat()
    if (
        not stat.S_ISREG(expected.st_mode)
        or target.readlink() != path
        or (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise RuntimeError("Replay parent lock descriptor identifies a different inode")
    rows = (handoff.PROC_ROOT / str(pid) / "fdinfo" / str(descriptor)).read_text().splitlines()
    locks = [row.split()[2:] for row in rows if row.startswith("lock:")]
    inode = f"{os.major(expected.st_dev):02x}:{os.minor(expected.st_dev):02x}:{expected.st_ino}"
    if not any(row == ["FLOCK", "ADVISORY", "WRITE", str(pid), inode, "0", "EOF"] for row in locks):
        raise RuntimeError("Replay parent does not hold the exact exclusive flock")
    with path.open("r+") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("Replay lock path changed before independent contention probe")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            final = target.stat()
            current = path.stat()
            if (
                target.readlink() != path
                or (final.st_dev, final.st_ino) != (expected.st_dev, expected.st_ino)
                or (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino)
            ):
                raise RuntimeError("Replay lock identity changed during contention probe") from None
            return
        fcntl.flock(stream, fcntl.LOCK_UN)
    raise RuntimeError("Replay parent node/manager lock is no longer held")


def require_gpu_owner(root: Path, binding_sha256: str, state_path: Path, worker_index: int) -> dict:
    """Reject before CUDA unless the exact acknowledged direct child owns this replay."""
    _, helpers, canonical_state, _ = locations(root)
    if state_path != canonical_state or worker_index not in (0, 1):
        raise RuntimeError("Saved NS replay has no canonical worker identity")
    frozen = require_frozen(root, binding_sha256)
    intent = read(helpers.parent / "controller.json")
    deadline = time.monotonic() + 30
    while True:
        state = read(state_path)
        if state.get("active_worker", {}).get("spawn_status") == "acknowledged":
            break
        if state.get("status") != "replaying" or time.monotonic() >= deadline:
            raise RuntimeError("Saved NS replay worker lacks executable acknowledgment")
        time.sleep(0.01)
    worker = state["active_worker"]
    environment = frozen["worker_environment"]
    own_command = worker_command(root, binding_sha256, worker_index)
    if (
        state["status"] != "replaying"
        or state["binding_sha256"] != binding_sha256
        or state["keeper_release_complete"] is not True
        or state["predecessor"] != frozen["predecessor"]
        or worker["index"] != worker_index
        or worker["command"] != own_command
        or worker["pid"] != os.getpid()
        or worker["controller_pid"] != os.getppid()
        or worker["controller_pid"] != state["controller_pid"]
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or intent["binding_sha256"] != binding_sha256
        or Path.cwd().resolve() != helpers
        or any(os.environ.get(name) != value for name, value in environment.items())
        or NODE not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("Saved NS replay binding, transfer or direct child differs")
    controller = process_context(
        state["controller_pid"], controller_command(root), helpers, CONTROLLER_MARKER, intent["marker"], ENVIRONMENT
    )
    require_acknowledged_controller(intent, controller)
    process = process_context(os.getpid(), own_command, helpers, CHILD_MARKER, worker["marker"], environment)
    if process["parent"] != controller["pid"] or process["start_ticks"] != worker["start_ticks"]:
        raise RuntimeError("Saved NS replay worker ancestry or generation differs")
    for name, path in (
        ("node", root / "run-preparation" / f".wave7-node-{NODE}.lock"),
        ("manager", MANAGER_LOCK),
    ):
        require_held_lock(controller["pid"], state["lock_descriptors"][name], path)
    devices = list(
        csv.reader(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True
            ).splitlines(),
            skipinitialspace=True,
        )
    )
    if devices != frozen["devices"] or environment["CUDA_VISIBLE_DEVICES"] != devices[0][1]:
        raise RuntimeError("Saved NS replay physical device mapping differs")
    if handoff.KEEPER_PID.exists():
        raise RuntimeError("Keeper pointer remains during saved NS replay")
    occupied = list(
        csv.reader(
            subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True
            ).splitlines(),
            skipinitialspace=True,
        )
    )
    if any(row != [str(os.getpid()), devices[0][1]] for row in occupied):
        raise RuntimeError("Unexpected compute owner exists during saved NS replay")
    if not handoff.same_generation(controller, handoff.observed(controller["pid"])):
        raise RuntimeError("Saved NS replay parent changed during owner checks")
    return {
        "device": "cuda:0",
        "physical_uuid": devices[0][1],
        "devices": devices,
        "binding": frozen,
        "controller_pid": controller["pid"],
        "controller_start_ticks": controller["start_ticks"],
        "worker_pid": process["pid"],
        "worker_start_ticks": process["start_ticks"],
    }
