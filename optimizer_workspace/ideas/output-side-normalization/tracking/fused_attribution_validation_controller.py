# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive and own the separate nine-config attribution validation, never a production run."""

import argparse
import fcntl
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import fused_attribution_validation as validation
import owned_output_handoff as handoff
import prepare_fused_attribution_validation as preparation
from fused_attribution_validation_contract import (
    CHILD_MARKER,
    CONTROLLER_MARKER,
    ENVIRONMENT,
    FROZEN_STATUS,
    NODE,
    PREPARATION,
    PYTHON,
    controller_command,
    locations,
    predecessor_locations,
    trial_specs,
    validator_command,
)
from launch_wave5 import read, sha256
from prepare_selected_lr_stage_profile_validation import helper_names as dependency_names
from prepare_selected_lr_validation import require_archive
from selected_lr_failure_handoff import (
    keeper_guard,
    pidfd_open,
    release_accepted_keeper,
    restore_accepted_keeper,
)
from watch_wave5 import write
from wave7_release import candidate_inventory, idle_training, local_node, node_guard

TESTS = (
    "test_fused_attribution_validation_controller.py",
    "test_fused_attribution_validation_owner.py",
    "test_selected_lr_failure_handoff.py",
)
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
CONTROLLER_CPU_STATUS = "fused_attribution_validation_controller_cpu_verified_no_gpu_binding"
MANAGER = "recovery_keeper_manager.sh"
CPU_TIMEOUT = 900
VALIDATION_TIMEOUT = 27 * 900 + 8100
ACK_TIMEOUT = 30


def helper_names(directory: Path) -> tuple[str, ...]:
    # The owner tests load a synthetic package whose relative imports are outside flat AST discovery.
    return dependency_names(
        directory,
        (Path(__file__).name, "selected_lr_release_owner.py", *preparation.COPIES, *preparation.TESTS, *TESTS),
    )


def helper_sources(root: Path, directory: Path) -> dict[str, Path]:
    """Use the failed profiler's archived manager, not the historical Wave12 keeper identity."""
    sources = {name: directory / name for name in helper_names(directory)}
    predecessor = read(predecessor_locations(root)[0])
    manager = directory / MANAGER
    if not manager.exists():
        manager = Path(predecessor["helpers"]) / MANAGER
    if manager.is_symlink() or sha256(manager) != predecessor["helpers_sha256"][MANAGER]:
        raise ValueError("Attribution serialized manager differs from the failed profiler archive")
    sources[MANAGER] = manager
    return sources


def require_loaded_helpers(helpers: Path, hashes: dict) -> None:
    """Reject a frozen controller importing mutable modules or candidate-shadowed tracking helpers."""
    for name in hashes:
        if name.endswith(".py"):
            module = sys.modules.get(Path(name).stem)
            if module is not None and Path(module.__file__).resolve() != helpers / name:
                raise RuntimeError(f"Attribution loaded helper origin differs: {name}")
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Attribution controller must execute its frozen helper copy")


def require_helper_bytes(helpers: Path, hashes: dict) -> None:
    if helpers != helpers.resolve() or helpers.is_symlink() or not hashes:
        raise ValueError("Attribution helper directory is not canonical")
    if any(
        Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != digest
        for name, digest in hashes.items()
    ):
        raise ValueError("Attribution archived helper bytes differ")


def controller_cpu_command() -> list[str]:
    return [
        PYTHON,
        "-P",
        "-B",
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "-p",
        "no:cacheprovider",
        "-q",
        "-m",
        SAFE_MARKERS,
        *(f"tracking/{name}" for name in TESTS),
    ]


def fixture_names() -> tuple[str, ...]:
    return tuple(spec["config_name"] for spec in trial_specs())


def controller_cpu(root: Path, candidate_receipt: Path, out: Path) -> dict:
    """Run the exact controller tests on CPU and retain both success and failure archives."""
    root, candidate_receipt, out = root.resolve(), candidate_receipt.resolve(), out.resolve()
    cpu = preparation.require_cpu(root, candidate_receipt)
    candidate = Path(cpu["candidate"])
    if out.exists() or out.is_relative_to(candidate) or candidate.is_relative_to(out):
        raise FileExistsError("Attribution controller CPU evidence needs an unused external directory")
    sources = helper_sources(root, Path(__file__).parent)
    hashes = {name: sha256(path) for name, path in sources.items()}
    if any(hashes.get(name) != digest for name, digest in cpu["helpers_sha256"].items()):
        raise ValueError("Attribution controller differs from candidate CPU helper bytes")
    candidate_receipt_digest = sha256(candidate_receipt)
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    for name, source in sources.items():
        shutil.copy2(source, helpers / name)
    fixtures = {}
    for name in fixture_names():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate / name, target)
        fixtures[name] = sha256(target)
    log, archive = out / "controller.log", out / "controller-source.tar"
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                controller_cpu_command(),
                cwd=out,
                env=os.environ
                | {
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": os.pathsep.join((str(candidate), str(helpers))),
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=CPU_TIMEOUT,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, f"Attribution controller CPU timeout at {CPU_TIMEOUT} seconds"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    expected = {f"tracking/{name}": digest for name, digest in hashes.items()} | fixtures
    try:
        with tarfile.open(archive, "x") as bundle:
            for name in expected:
                bundle.add(out / name, arcname=name, recursive=False)
        require_archive(archive, expected, alias=False)
        if (
            hashes != {name: sha256(helpers / name) for name in hashes}
            or hashes != {name: sha256(path) for name, path in sources.items()}
            or fixtures != {name: sha256(out / name) for name in fixtures}
            or any(cpu["files_sha256"][name] != digest for name, digest in fixtures.items())
            or preparation.require_cpu(root, candidate_receipt) != cpu
            or sha256(candidate_receipt) != candidate_receipt_digest
        ):
            raise ValueError("Attribution candidate or controller changed during CPU verification")
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as failure:
        detail = f"{type(failure).__name__}: {failure}"
        error = detail if error is None else f"{error}; {detail}"
    result = {
        "status": CONTROLLER_CPU_STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate_cpu_receipt": str(candidate_receipt),
        "candidate_cpu_receipt_sha256": candidate_receipt_digest,
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "fixture_files_sha256": fixtures,
        "command": controller_cpu_command(),
        "cwd": str(out),
        "exit_code": code,
        "cuda_visible_devices": "",
        "log": str(log),
        "log_sha256": sha256(log) if log.is_file() else None,
        "archive": str(archive),
        "archive_sha256": sha256(archive) if archive.is_file() else None,
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != CONTROLLER_CPU_STATUS:
        raise RuntimeError(f"Attribution controller CPU failure preserved at {out}")
    try:
        return require_controller_cpu(root, out / "receipt.json", candidate_receipt)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as failure:
        result.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(out / "receipt.json", result)
        raise RuntimeError(f"Attribution controller final CPU/archive verification failed at {out}") from failure


def require_controller_cpu(root: Path, receipt_path: Path, candidate_receipt: Path) -> dict:
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw)
    cpu = preparation.require_cpu(root, candidate_receipt)
    helpers = Path(receipt["helpers"])
    require_helper_bytes(helpers, receipt["helpers_sha256"])
    sources = helper_sources(root, helpers)
    hashes = {name: sha256(path) for name, path in sources.items()}
    fixtures = {name: cpu["files_sha256"][name] for name in fixture_names()}
    if (
        receipt["status"] != CONTROLLER_CPU_STATUS
        or receipt["root"] != str(root)
        or receipt["candidate_cpu_receipt"] != str(candidate_receipt.resolve())
        or receipt["candidate_cpu_receipt_sha256"] != sha256(candidate_receipt)
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["cuda_visible_devices"] != ""
        or receipt["command"] != controller_cpu_command()
        or receipt["cwd"] != str(helpers.parent)
        or receipt["helpers_sha256"] != hashes
        or any(hashes.get(name) != digest for name, digest in cpu["helpers_sha256"].items())
        or receipt["fixture_files_sha256"] != fixtures
        or fixtures != {name: sha256(helpers.parent / name) for name in fixtures}
        or Path(receipt["log"]) != helpers.parent / "controller.log"
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or Path(receipt["archive"]) != helpers.parent / "controller-source.tar"
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
        or "error" in receipt
    ):
        raise ValueError("Attribution controller CPU command, outcome or source differs")
    require_archive(
        Path(receipt["archive"]),
        {f"tracking/{name}": digest for name, digest in hashes.items()} | fixtures,
        alias=False,
    )
    if receipt_path.read_bytes() != raw:
        raise ValueError("Attribution controller CPU receipt changed during verification")
    return receipt


def archive_inventory(frozen: dict) -> dict:
    prepared = read(Path(frozen["candidate"]) / PREPARATION)
    return (
        frozen["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {
            PREPARATION: frozen["preparation_sha256"],
            "candidate-cpu-receipt.json": frozen["cpu_receipt_sha256"],
            "controller-cpu-receipt.json": frozen["controller_cpu_receipt_sha256"],
        }
        | {f"tracking/{name}": digest for name, digest in frozen["helpers_sha256"].items()}
    )


def freeze(root: Path, candidate_receipt: Path, controller_receipt: Path) -> dict:
    """Publish one immutable validation binding without releasing or probing the live allocation."""
    root, candidate_receipt, controller_receipt = (
        root.resolve(),
        candidate_receipt.resolve(),
        controller_receipt.resolve(),
    )
    cpu = preparation.require_cpu(root, candidate_receipt)
    controller = require_controller_cpu(root, controller_receipt, candidate_receipt)
    candidate = Path(cpu["candidate"])
    prepared = read(candidate / PREPARATION)
    predecessor = validation.failed_profiler_evidence(root)
    if prepared["allocation_predecessor"] != predecessor:
        raise ValueError("Attribution preparation does not identify the exact failed-profiler restoration")
    binding_path, helpers, state_path, work, receipt_path = locations(root)
    archive = root / "run-preparation" / f"source-{helpers.parent.name}.tar"
    sources = helper_sources(root, Path(__file__).parent)
    hashes = {name: sha256(path) for name, path in sources.items()}
    if hashes != controller["helpers_sha256"] or any(
        hashes.get(name) != digest for name, digest in cpu["helpers_sha256"].items()
    ):
        raise ValueError("Attribution freeze requires the exact CPU-tested helper closure")
    with (root / "run-preparation/.fused-attribution-validation-freeze.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.exists() for path in (binding_path, helpers.parent, archive, state_path, work, receipt_path)):
            raise FileExistsError("Attribution validation identity already has evidence; preserve it")
        helpers.mkdir(parents=True)
        for name, source in sources.items():
            shutil.copy2(source, helpers / name)
        planned = validation.plan(candidate, root, work)
        frozen = {
            "status": FROZEN_STATUS,
            "root": str(root),
            "candidate": str(candidate),
            "node": NODE,
            "python": PYTHON,
            "environment": ENVIRONMENT,
            "files_sha256": cpu["files_sha256"],
            "preparation_sha256": sha256(candidate / PREPARATION),
            "cpu_receipt": str(candidate_receipt),
            "cpu_receipt_sha256": sha256(candidate_receipt),
            "controller_cpu_receipt": str(controller_receipt),
            "controller_cpu_receipt_sha256": sha256(controller_receipt),
            "helpers": str(helpers),
            "helpers_sha256": hashes,
            "archive": str(archive),
            "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
            "allocation_predecessor": predecessor,
            "plan": planned,
            "commands": [command for trial in planned["trials"] for command in trial["commands"]],
            "work": str(work),
            "receipt": str(receipt_path),
            "validation_timeout": VALIDATION_TIMEOUT,
            "policy": "Validation only; exact failed-profiler keeper transfer under its manager lock. No production.",
        }
        expected = archive_inventory(frozen)
        with tarfile.open(archive, "x") as bundle:
            for name in expected:
                if name.startswith("tracking/"):
                    source = helpers / Path(name).name
                elif name == "candidate-cpu-receipt.json":
                    source = candidate_receipt
                elif name == "controller-cpu-receipt.json":
                    source = controller_receipt
                else:
                    source = candidate / name
                bundle.add(source, arcname=name, recursive=False)
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
        frozen["archive_sha256"] = sha256(archive)
        require_archive(archive, expected, alias=True)
        if (
            hashes != {name: sha256(helpers / name) for name in hashes}
            or preparation.require_cpu(root, candidate_receipt) != cpu
            or require_controller_cpu(root, controller_receipt, candidate_receipt) != controller
            or validation.failed_profiler_evidence(root) != predecessor
        ):
            raise ValueError("Attribution acceptance or predecessor changed during freeze")
        write(binding_path, frozen)
    return frozen


def binding(root: Path) -> dict:
    binding_path, helpers, _, work, receipt = locations(root)
    raw = binding_path.read_bytes()
    frozen = json.loads(raw)
    if frozen["helpers"] != str(helpers):
        raise ValueError("Attribution binding has a noncanonical helper directory")
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    require_loaded_helpers(helpers, frozen["helpers_sha256"])
    cpu = preparation.require_cpu(root, Path(frozen["cpu_receipt"]))
    controller = require_controller_cpu(root, Path(frozen["controller_cpu_receipt"]), Path(frozen["cpu_receipt"]))
    candidate = Path(cpu["candidate"])
    prepared = read(candidate / PREPARATION)
    planned = validation.plan(candidate, root, work)
    predecessor = validation.failed_profiler_evidence(root)
    if (
        root != root.resolve()
        or candidate != candidate.resolve()
        or frozen["status"] != FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != str(candidate)
        or frozen["node"] != NODE
        or frozen["python"] != PYTHON
        or frozen["environment"] != ENVIRONMENT
        or frozen["files_sha256"] != cpu["files_sha256"]
        or frozen["files_sha256"] != candidate_inventory(candidate)
        or frozen["preparation_sha256"] != sha256(candidate / PREPARATION)
        or frozen["cpu_receipt_sha256"] != sha256(Path(frozen["cpu_receipt"]))
        or frozen["controller_cpu_receipt_sha256"] != sha256(Path(frozen["controller_cpu_receipt"]))
        or frozen["helpers_sha256"] != controller["helpers_sha256"]
        or any(frozen["helpers_sha256"].get(name) != digest for name, digest in cpu["helpers_sha256"].items())
        or frozen["helpers_sha256"] != {name: sha256(path) for name, path in helper_sources(root, helpers).items()}
        or frozen["archive"] != str(root / "run-preparation" / f"source-{helpers.parent.name}.tar")
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["data_receipt_sha256"] != prepared["data_receipt_sha256"]
        or frozen["allocation_predecessor"] != predecessor
        or prepared["allocation_predecessor"] != predecessor
        or frozen["plan"] != planned
        or frozen["commands"] != [command for trial in planned["trials"] for command in trial["commands"]]
        or frozen["work"] != str(work)
        or frozen["receipt"] != str(receipt)
        or frozen["validation_timeout"] != VALIDATION_TIMEOUT
    ):
        raise ValueError("Attribution frozen source, command plan, acceptance or predecessor differs")
    require_archive(Path(frozen["archive"]), archive_inventory(frozen), alias=True)
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    if binding_path.read_bytes() != raw:
        raise ValueError("Attribution binding changed during verification")
    return frozen


def process_context(pid: int, command: list[str], cwd: Path, marker_name: str, marker: str, environment: dict) -> dict:
    process = handoff.observed(pid)
    directory = handoff.PROC_ROOT / str(pid)
    if process is None or process["status"] in ("Z", "X", "x"):
        raise RuntimeError("Attribution process is absent or exited")
    actual_environment = dict(
        token.decode().split("=", 1) for token in (directory / "environ").read_bytes().split(b"\0") if token
    )
    if (
        not marker
        or process["command"] != command
        or process["group"] != pid
        or process["session"] != pid
        or (directory / "cwd").resolve() != cwd
        or actual_environment.get(marker_name) != marker
        or any(actual_environment.get(key) != value for key, value in environment.items())
        or not handoff.same_generation(process, handoff.observed(pid))
    ):
        raise RuntimeError("Attribution executable, session, cwd, marker or environment differs")
    return process


def acknowledged_process(
    child: subprocess.Popen, command: list[str], cwd: Path, marker_name: str, marker: str, environment: dict
) -> dict:
    """Wait for exec, never acknowledge the transient starter command or an already exited child."""
    deadline = time.monotonic() + ACK_TIMEOUT
    while True:
        process = handoff.observed(child.pid)
        if process is not None and process["command"] == command:
            return process_context(child.pid, command, cwd, marker_name, marker, environment)
        if child.poll() is not None or time.monotonic() >= deadline:
            raise RuntimeError("Attribution child exited or never reached its exact executable acknowledgment")
        time.sleep(0.01)


def require_acknowledged_controller(intent: dict, own: dict) -> None:
    expected = intent["generation"]
    if (
        intent["status"] != "acknowledged"
        or intent["pid"] != own["pid"]
        or intent["start_ticks"] != own["start_ticks"]
        or own["parent"] not in (expected["parent"], 1)
        or any(own[key] != expected[key] for key in ("pid", "start_ticks", "group", "session", "command"))
        or own["status"] in ("Z", "X", "x")
    ):
        raise RuntimeError("Attribution detached acknowledgment or controller generation differs")


def probe_failed_profiler(root: Path, expected: dict) -> dict:
    """Read a fresh keeper/marker/original-handle verdict while the caller owns the manager lock."""
    if validation.failed_profiler_evidence(root) != expected:
        raise RuntimeError("Attribution failed-profiler evidence changed before live handoff")
    current = handoff.keeper_observation()
    marked = handoff.marker_processes(expected["controller_marker_name"], expected["controller_marker"])
    children = handoff.marker_processes(expected["child_marker_name"], expected["child_marker"])
    handles = {pid: handoff.observed(pid) for pid in (expected["controller_pid"], expected["child_pid"])}
    validation.require_failed_profiler_handoff(expected, current, marked, children, handles)
    if validation.failed_profiler_evidence(root) != expected:
        raise RuntimeError("Attribution failed-profiler evidence changed during live handoff")
    return {
        "status": "failed_profiler_exact_keeper_live_handoff_verified",
        "keeper": current,
        "inherited_controller_marker_pids": sorted(marked),
        "original_handles_absent": True,
        "child_markers_absent": True,
    }


def start(root: Path) -> dict:
    frozen = binding(root)
    if any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()) or local_node() != NODE:
        raise RuntimeError("Attribution start requires its bound node and explicit eight-GPU environment")
    binding_path, helpers, state_path, work, receipt = locations(root)
    intent_path, log = helpers.parent / "controller.json", helpers.parent / "controller.log"
    with (helpers.parent / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.exists() for path in (intent_path, log, state_path, work, receipt)):
            raise FileExistsError("Attribution execution already has intent/evidence; never retry an ambiguous start")
        with keeper_guard():
            idle_training()
            probe_failed_profiler(root, frozen["allocation_predecessor"])
            if binding(root) != frozen:
                raise RuntimeError("Attribution binding changed before detached start")
        intent = {
            "status": "starting",
            "binding_sha256": sha256(binding_path),
            "command": controller_command(root),
            "marker": uuid4().hex,
            "allocation_predecessor": frozen["allocation_predecessor"],
            "log": str(log),
        }
        write(intent_path, intent)
        with log.open("x") as stream:
            child = subprocess.Popen(
                intent["command"],
                cwd=helpers,
                env=os.environ
                | ENVIRONMENT
                | {
                    CONTROLLER_MARKER: intent["marker"],
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(helpers),
                },
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        process = acknowledged_process(
            child, intent["command"], helpers, CONTROLLER_MARKER, intent["marker"], ENVIRONMENT
        )
        intent.update(status="acknowledged", pid=child.pid, start_ticks=process["start_ticks"], generation=process)
        write(intent_path, intent)
    return intent


def finish_marked_child(child: subprocess.Popen, marker: str, generation: dict, marker_name: str) -> None:
    """Clean independent or reparented ranks with pidfds, leaving unrelated processes untouched."""
    with ExitStack() as stack:
        handles, poll = {}, select.poll()
        for signum, timeout in ((signal.SIGTERM, 60), (signal.SIGKILL, 15)):
            deadline, signaled = time.monotonic() + timeout, set()
            while True:
                current = handoff.marker_processes(marker_name, marker)
                generation.update({str(pid): process["start_ticks"] for pid, process in current.items()})
                for pid, expected in current.items():
                    identity = pid, expected["start_ticks"]
                    if identity not in handles:
                        try:
                            handle = pidfd_open(pid)
                        except ProcessLookupError:
                            continue
                        stack.callback(os.close, handle)
                        if not handoff.same_generation(expected, handoff.observed(pid)):
                            raise RuntimeError("Attribution marked process changed while opening cleanup pidfd")
                        handles[identity] = handle
                        poll.register(handle, select.POLLIN)
                    if identity not in signaled:
                        try:
                            signal.pidfd_send_signal(handles[identity], signum)
                        except ProcessLookupError:
                            pass
                        signaled.add(identity)
                if child.poll() is not None and not handoff.marker_processes(marker_name, marker):
                    return
                if time.monotonic() >= deadline:
                    break
                poll.poll(min(100, max(1, int((deadline - time.monotonic()) * 1000))))
        raise TimeoutError("Attribution validator or marked ranks remain after bounded cleanup")


def bounded_validator(root: Path, frozen: dict, state: dict) -> int:
    _, helpers, state_path, _, _ = locations(root)
    candidate = Path(frozen["candidate"])
    command, marker = validator_command(root, candidate), uuid4().hex
    generation, log = {}, helpers.parent / "validator.log"
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=candidate,
            env=os.environ
            | ENVIRONMENT
            | {
                CHILD_MARKER: marker,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join((str(candidate), str(helpers))),
            },
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process = acknowledged_process(child, command, candidate, CHILD_MARKER, marker, ENVIRONMENT)
            state.update(
                status="validating",
                spawn_status="acknowledged",
                command=command,
                child_pid=child.pid,
                child_start_ticks=process["start_ticks"],
                child_marker=marker,
                log=str(log),
            )
            write(state_path, state)
            deadline = time.monotonic() + VALIDATION_TIMEOUT
            while True:
                generation.update(
                    {
                        str(pid): process["start_ticks"]
                        for pid, process in handoff.marker_processes(CHILD_MARKER, marker).items()
                    }
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, VALIDATION_TIMEOUT)
                try:
                    return child.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            try:
                finish_marked_child(child, marker, generation, CHILD_MARKER)
                state["cleanup_exit_code"] = child.returncode
                state["marked_descendants_absent"] = True
            except BaseException as error:
                state["cleanup_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                state.update(child_generation=generation, log=str(log), log_sha256=sha256(log))
                write(state_path, state)


def validate(root: Path) -> dict:
    frozen = binding(root)
    binding_path, helpers, state_path, _, receipt_path = locations(root)
    intent_path = helpers.parent / "controller.json"
    deadline, intent = time.monotonic() + ACK_TIMEOUT, read(intent_path)
    while intent["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.05)
        intent = read(intent_path)
    own = process_context(
        os.getpid(),
        controller_command(root),
        helpers,
        CONTROLLER_MARKER,
        os.environ.get(CONTROLLER_MARKER, ""),
        ENVIRONMENT,
    )
    if (
        intent["binding_sha256"] != sha256(binding_path)
        or intent["command"] != controller_command(root)
        or intent["marker"] != os.environ.get(CONTROLLER_MARKER)
        or local_node() != NODE
        or state_path.exists()
        or intent["allocation_predecessor"] != frozen["allocation_predecessor"]
    ):
        raise RuntimeError("Attribution validation has no original acknowledged controller ownership")
    require_acknowledged_controller(intent, own)
    state = {
        "status": "preflight",
        "binding_sha256": sha256(binding_path),
        "controller_pid": own["pid"],
        "controller_start_ticks": own["start_ticks"],
        "node": NODE,
        "candidate": frozen["candidate"],
        "allocation_predecessor": frozen["allocation_predecessor"],
    }
    write(state_path, state)
    signals = []
    try:
        with node_guard(root, NODE) as signals, keeper_guard() as manager_lock:
            idle_training()
            live = probe_failed_profiler(root, frozen["allocation_predecessor"])
            accepted = live["keeper"]
            subprocess.run(
                [PYTHON, "-m", "experiment.bootstrap", "--check-only"],
                cwd=Path(frozen["candidate"]),
                env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                check=True,
                timeout=240,
            )
            if binding(root) != frozen:
                raise RuntimeError("Attribution frozen identity changed before keeper transfer")
            state["handoff"] = probe_failed_profiler(root, frozen["allocation_predecessor"])
            try:
                release_accepted_keeper(accepted, state, state_path, manager_lock)
                code = bounded_validator(root, frozen, state)
                state["exit_code"] = code
                if type(code) is not int or code != 0:
                    raise RuntimeError("Attribution bounded GPU validator failed")
                actual = validation.reconstruct(root, Path(frozen["candidate"]), receipt_path)
                if actual["status"] != validation.PASS_STATUS or actual["owner"] != {
                    "controller_pid": own["pid"],
                    "controller_start_ticks": own["start_ticks"],
                }:
                    raise ValueError("Attribution validation gates or controller generation differ")
                state.update(status="passed", validation_sha256=sha256(receipt_path))
            finally:
                if state.get("keeper_release_attempted"):
                    try:
                        if (
                            state.get("keeper_release_complete") is not True
                            or "cleanup_error" in state
                            or (
                                state.get("child_marker")
                                and handoff.marker_processes(CHILD_MARKER, state["child_marker"])
                            )
                        ):
                            raise RuntimeError("Attribution incomplete transfer or live descendants prevent restoration")
                        restore_accepted_keeper(manager_lock)
                        restored = handoff.keeper_observation()
                        validation.require_keeper_shape(restored)
                        if restored["devices"] != accepted["devices"]:
                            raise RuntimeError("Attribution restored keeper has different physical GPUs")
                        state.update(restored_keeper=restored, keeper_restore=0)
                    except BaseException as error:
                        state.update(
                            status="needs_attention", keeper_restore=f"refused: {type(error).__name__}: {error}"
                        )
                        raise
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if signals:
            state["deferred_signals"] = signals
        write(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("cpu", "freeze", "check", "start", "validate"), required=True)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--controller-cpu-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase in ("cpu", "freeze") and args.cpu_receipt is None:
        parser.error("cpu/freeze require --cpu-receipt")
    if args.phase == "cpu" and args.out is None:
        parser.error("cpu requires --out")
    if args.phase == "freeze" and args.controller_cpu_receipt is None:
        parser.error("freeze requires --controller-cpu-receipt")
    root = args.root.resolve()
    if args.phase == "cpu":
        result = controller_cpu(root, args.cpu_receipt.resolve(), args.out.resolve())
    elif args.phase == "freeze":
        result = freeze(root, args.cpu_receipt.resolve(), args.controller_cpu_receipt.resolve())
    elif args.phase == "check":
        result = binding(root)
    elif args.phase == "start":
        result = start(root)
    else:
        result = validate(root)
    print(json.dumps({"status": result["status"]}, allow_nan=False))


if __name__ == "__main__":
    main()
