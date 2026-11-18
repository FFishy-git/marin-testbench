# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze and own one validation-only selected-LR profiler after completed Wave 12."""

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

from launch_wave5 import read, sha256
from owned_output_handoff import keeper_observation, marker_processes, observed, same_generation
from prepare_selected_lr_stage_profile_validation import COPIES, require_cpu
from prepare_selected_lr_stage_profile_validation import helper_names as dependency_names
from prepare_selected_lr_validation import require_archive
from selected_lr_failure_handoff import (
    keeper_guard,
    pidfd_open,
    release_accepted_keeper,
    restore_accepted_keeper,
    same_keeper,
)
from selected_lr_stage_profile_contract import (
    CHILD_MARKER,
    CONTROLLER_MARKER,
    ENVIRONMENT,
    FROZEN_STATUS,
    NODE,
    PREPARATION,
    PYTHON,
    controller_command,
    locations,
    validator_command,
)
from selected_lr_stage_profile_handoff import require_completed_wave12
from selected_lr_stage_profile_validation import plan
from validate_selected_lr_stage_profile import reconstruct
from watch_wave5 import write
from wave7_release import candidate_inventory, idle_training, local_node, node_guard

TESTS = (
    "test_selected_lr_stage_profile_controller.py",
    "test_selected_lr_stage_profile_execution_owner.py",
    "test_selected_lr_stage_profile_validation.py",
)
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
CONTROLLER_CPU_STATUS = "selected_lr_stage_profile_controller_cpu_verified_no_gpu_binding"
MANAGER = "recovery_keeper_manager.sh"
VALIDATION_TIMEOUT = 28800
CPU_FIXTURE_CONFIG = "experiment/configs/muont_0p5m_seed0.json"


def helper_names(directory: Path) -> tuple[str, ...]:
    return dependency_names(
        directory,
        (
            Path(__file__).name,
            "validate_selected_lr_stage_profile.py",
            "selected_lr_release_owner.py",
            *COPIES,
            *TESTS,
        ),
    )


def helper_sources(root: Path, directory: Path) -> dict[str, Path]:
    """Keep the original issued manager bytes without editing the candidate's helper closure."""
    sources = {name: directory / name for name in helper_names(directory)}
    release = read(root / "run-preparation/wave12-launch-binding.json")
    manager = directory / MANAGER
    if not manager.exists():
        manager = Path(release["helpers"]) / MANAGER
    if manager.is_symlink() or sha256(manager) != release["helpers_sha256"][MANAGER]:
        raise ValueError("Stage-profile serialized manager differs from the issued Wave12 manager")
    sources[MANAGER] = manager
    return sources


def require_loaded_helpers(helpers: Path, hashes: dict) -> None:
    """Reject preloaded local modules from outside the frozen helper directory."""
    for name in hashes:
        if not name.endswith(".py"):
            continue
        module = sys.modules.get(Path(name).stem)
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Stage-profile loaded helper origin differs: {name}")
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Stage-profile controller must run from its frozen helper directory")


def controller_cpu_command() -> list[str]:
    return [
        PYTHON,
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


def controller_cpu(root: Path, candidate_receipt: Path, out: Path) -> dict:
    """Exercise the archived controller independently of the issued candidate CPU suite."""
    cpu = require_cpu(root, candidate_receipt)
    candidate = Path(cpu["candidate"])
    if out.exists() or out.resolve().is_relative_to(candidate):
        raise FileExistsError("Stage-profile controller CPU evidence requires an unused external directory")
    sources = helper_sources(root, Path(__file__).parent)
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    hashes = {name: sha256(path) for name, path in sources.items()}
    if any(hashes.get(name) != digest for name, digest in cpu["helpers_sha256"].items()):
        raise ValueError("Stage-profile controller changed the issued candidate helper identities")
    for name, source in sources.items():
        shutil.copy2(source, helpers / name)
    fixture = out / CPU_FIXTURE_CONFIG
    fixture.parent.mkdir(parents=True)
    shutil.copy2(candidate / CPU_FIXTURE_CONFIG, fixture)
    fixture_hashes = {CPU_FIXTURE_CONFIG: sha256(fixture)}
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
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Stage-profile controller CPU timeout at 600 seconds"
    with tarfile.open(archive, "x") as bundle:
        for name in hashes:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        bundle.add(fixture, arcname=CPU_FIXTURE_CONFIG, recursive=False)
    require_archive(
        archive,
        {f"tracking/{name}": digest for name, digest in hashes.items()} | fixture_hashes,
        alias=False,
    )
    if (
        hashes != {name: sha256(helpers / name) for name in hashes}
        or hashes != {name: sha256(path) for name, path in sources.items()}
        or fixture_hashes != {CPU_FIXTURE_CONFIG: sha256(fixture)}
        or fixture_hashes != {CPU_FIXTURE_CONFIG: cpu["files_sha256"][CPU_FIXTURE_CONFIG]}
        or require_cpu(root, candidate_receipt) != cpu
    ):
        raise ValueError("Stage-profile candidate or controller changed during CPU acceptance")
    result = {
        "status": CONTROLLER_CPU_STATUS if code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate_cpu_receipt": str(candidate_receipt.resolve()),
        "candidate_cpu_receipt_sha256": sha256(candidate_receipt),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "fixture_files_sha256": fixture_hashes,
        "command": controller_cpu_command(),
        "cwd": str(out),
        "exit_code": code,
        "cuda_visible_devices": "",
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != CONTROLLER_CPU_STATUS:
        raise RuntimeError(f"Stage-profile controller CPU failure preserved at {out}")
    return result


def require_controller_cpu(root: Path, receipt_path: Path, candidate_receipt: Path) -> dict:
    receipt = read(receipt_path)
    helpers = Path(receipt["helpers"])
    sources = helper_sources(root, helpers)
    hashes = {name: sha256(path) for name, path in sources.items()}
    cpu = read(candidate_receipt)
    fixture_hashes = {CPU_FIXTURE_CONFIG: cpu["files_sha256"][CPU_FIXTURE_CONFIG]}
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
        or receipt["fixture_files_sha256"] != fixture_hashes
        or fixture_hashes != {name: sha256(helpers.parent / name) for name in fixture_hashes}
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
        or "error" in receipt
    ):
        raise ValueError("Stage-profile controller CPU command, outcome or archived source differs")
    require_archive(
        Path(receipt["archive"]),
        {f"tracking/{name}": digest for name, digest in hashes.items()} | fixture_hashes,
        alias=False,
    )
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
    """Publish an immutable validation identity without transferring any GPU allocation."""
    cpu = require_cpu(root, candidate_receipt)
    controller = require_controller_cpu(root, controller_receipt, candidate_receipt)
    candidate = Path(cpu["candidate"])
    binding_path, helpers, state_path, work, receipt_path = locations(root)
    archive = root / "run-preparation" / f"source-{helpers.parent.name}.tar"
    sources = helper_sources(root, Path(__file__).parent)
    hashes = {name: sha256(path) for name, path in sources.items()}
    if hashes != controller["helpers_sha256"]:
        raise ValueError("Stage-profile current controller differs from CPU-tested archived bytes")
    if any(hashes.get(name) != digest for name, digest in cpu["helpers_sha256"].items()):
        raise ValueError("Stage-profile frozen controller differs from candidate CPU helper identities")
    with (root / "run-preparation/.selected-lr-stage-profile-freeze.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.exists() for path in (binding_path, helpers, archive, state_path, work, receipt_path)):
            raise FileExistsError("Stage-profile validation identity already exists; preserve it")
        helpers.mkdir(parents=True)
        for name, source in sources.items():
            shutil.copy2(source, helpers / name)
        frozen = {
            "status": FROZEN_STATUS,
            "root": str(root),
            "candidate": str(candidate),
            "node": NODE,
            "python": PYTHON,
            "environment": ENVIRONMENT,
            "files_sha256": cpu["files_sha256"],
            "preparation_sha256": sha256(candidate / PREPARATION),
            "cpu_receipt": str(candidate_receipt.resolve()),
            "cpu_receipt_sha256": sha256(candidate_receipt),
            "controller_cpu_receipt": str(controller_receipt.resolve()),
            "controller_cpu_receipt_sha256": sha256(controller_receipt),
            "helpers": str(helpers),
            "helpers_sha256": hashes,
            "archive": str(archive),
            "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
            "predecessor_release_sha256": sha256(root / "run-preparation/wave12-launch-binding.json"),
            "plan": plan(candidate, root, work),
            "commands": [command for trial in plan(candidate, root, work)["trials"] for command in trial["commands"]],
            "work": str(work),
            "receipt": str(receipt_path),
            "policy": (
                "Validation only; completed original Wave12 workers, five-window analysis and verified cloud "
                "histories precede one-shot execution. Exact keeper transfer is serialized under its manager lock."
            ),
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
            or require_cpu(root, candidate_receipt) != cpu
            or require_controller_cpu(root, controller_receipt, candidate_receipt) != controller
        ):
            raise ValueError("Stage-profile source or acceptance changed during validation freeze")
        write(binding_path, frozen)
    return frozen


def binding(root: Path) -> dict:
    binding_path, helpers, _, work, receipt = locations(root)
    frozen = read(binding_path)
    require_loaded_helpers(helpers, frozen["helpers_sha256"])
    cpu = require_cpu(root, Path(frozen["cpu_receipt"]))
    controller = require_controller_cpu(root, Path(frozen["controller_cpu_receipt"]), Path(frozen["cpu_receipt"]))
    candidate = Path(cpu["candidate"])
    expected_plan = plan(candidate, root, work)
    if (
        frozen["status"] != FROZEN_STATUS
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
        or frozen["helpers"] != str(helpers)
        or frozen["helpers_sha256"] != controller["helpers_sha256"]
        or any(frozen["helpers_sha256"].get(name) != digest for name, digest in cpu["helpers_sha256"].items())
        or frozen["helpers_sha256"] != {name: sha256(path) for name, path in helper_sources(root, helpers).items()}
        or frozen["archive"] != str(root / "run-preparation" / f"source-{helpers.parent.name}.tar")
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["predecessor_release_sha256"] != sha256(root / "run-preparation/wave12-launch-binding.json")
        or frozen["plan"] != expected_plan
        or frozen["commands"] != [command for trial in expected_plan["trials"] for command in trial["commands"]]
        or frozen["work"] != str(work)
        or frozen["receipt"] != str(receipt)
    ):
        raise ValueError("Stage-profile frozen source, command plan or acceptance differs")
    require_archive(Path(frozen["archive"]), archive_inventory(frozen), alias=True)
    return frozen


def process_context(pid: int, command: list[str], cwd: Path, marker_name: str, marker: str) -> dict:
    """Observe the real executable after exec, including cwd and inherited execution environment."""
    process = observed(pid)
    directory = Path("/proc") / str(pid)
    if process is None or process["status"] in ("Z", "X", "x"):
        raise RuntimeError("Stage-profile process is absent or exited")
    environment = dict(
        token.decode().split("=", 1) for token in (directory / "environ").read_bytes().split(b"\0") if token
    )
    if (
        process["command"] != command
        or process["group"] != pid
        or process["session"] != pid
        or (directory / "cwd").resolve() != cwd
        or environment.get(marker_name) != marker
        or any(environment.get(key) != value for key, value in ENVIRONMENT.items())
        or not same_generation(process, observed(pid))
    ):
        raise RuntimeError("Stage-profile executable, session, cwd, marker or environment differs")
    return process


def require_acknowledged_controller(intent: dict, own: dict) -> None:
    """Permit detached-parent exit, but keep PID/start/argv/group/session identity exact."""
    expected = intent["generation"]
    if (
        intent["status"] != "acknowledged"
        or intent["pid"] != own["pid"]
        or intent["start_ticks"] != own["start_ticks"]
        or own["parent"] not in (expected["parent"], 1)
        or any(own[key] != expected[key] for key in ("pid", "start_ticks", "group", "session", "command"))
        or own["status"] in ("Z", "X", "x")
    ):
        raise RuntimeError("Stage-profile detached controller acknowledgment or generation differs")


def start(root: Path) -> dict:
    binding(root)
    if any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()) or local_node() != NODE:
        raise RuntimeError("Stage-profile start requires its original node and explicit eight-GPU environment")
    predecessor = require_completed_wave12(root)
    binding_path, helpers, state_path, work, receipt = locations(root)
    intent_path, log = helpers.parent / "controller.json", helpers.parent / "controller.log"
    with (helpers.parent / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.exists() for path in (intent_path, log, state_path, work, receipt)):
            raise FileExistsError("Stage-profile execution has intent or evidence; ambiguous starts are never retried")
        intent = {
            "status": "starting",
            "binding_sha256": sha256(binding_path),
            "command": controller_command(root),
            "marker": uuid4().hex,
            "predecessor": predecessor,
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
        process = process_context(child.pid, intent["command"], helpers, CONTROLLER_MARKER, intent["marker"])
        intent.update(status="acknowledged", pid=child.pid, start_ticks=process["start_ticks"], generation=process)
        write(intent_path, intent)
    return intent


def finish_marked_child(child: subprocess.Popen, marker: str, generation: dict) -> None:
    """Reap independently sessioned/reparented descendants through generation-stable pidfds."""
    with ExitStack() as stack:
        handles, poll = {}, select.poll()
        for signum, timeout in ((signal.SIGTERM, 60), (signal.SIGKILL, 15)):
            deadline, signaled = time.monotonic() + timeout, set()
            while True:
                current = marker_processes(CHILD_MARKER, marker)
                generation.update({str(pid): process["start_ticks"] for pid, process in current.items()})
                for pid, expected in current.items():
                    identity = pid, expected["start_ticks"]
                    if identity not in handles:
                        try:
                            handle = pidfd_open(pid)
                        except ProcessLookupError:
                            continue
                        stack.callback(os.close, handle)
                        if not same_generation(expected, observed(pid)):
                            raise RuntimeError("Stage-profile marked process changed while binding cleanup handle")
                        handles[identity] = handle
                        poll.register(handle, select.POLLIN)
                    if identity not in signaled:
                        try:
                            signal.pidfd_send_signal(handles[identity], signum)
                        except ProcessLookupError:
                            pass
                        signaled.add(identity)
                if child.poll() is not None and not marker_processes(CHILD_MARKER, marker):
                    return
                if time.monotonic() >= deadline:
                    break
                poll.poll(min(100, max(1, int((deadline - time.monotonic()) * 1000))))
        raise TimeoutError("Stage-profile validator or marked ranks remain after independent cleanup")


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
            process = process_context(child.pid, command, candidate, CHILD_MARKER, marker)
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
                    {str(pid): process["start_ticks"] for pid, process in marker_processes(CHILD_MARKER, marker).items()}
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
                finish_marked_child(child, marker, generation)
                state["cleanup_exit_code"] = child.returncode
                state["marked_descendants_absent"] = True
            except BaseException as error:
                state["cleanup_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                state.update(child_generation=generation, log_sha256=sha256(log))
                write(state_path, state)


def validate(root: Path) -> dict:
    frozen = binding(root)
    binding_path, helpers, state_path, _, receipt_path = locations(root)
    intent_path = helpers.parent / "controller.json"
    deadline = time.monotonic() + 30
    intent = read(intent_path)
    while intent["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.05)
        intent = read(intent_path)
    own = process_context(
        os.getpid(), controller_command(root), helpers, CONTROLLER_MARKER, os.environ.get(CONTROLLER_MARKER, "")
    )
    if (
        intent["status"] != "acknowledged"
        or intent["pid"] != own["pid"]
        or intent["start_ticks"] != own["start_ticks"]
        or intent["binding_sha256"] != sha256(binding_path)
        or intent["command"] != controller_command(root)
        or intent["marker"] != os.environ.get(CONTROLLER_MARKER)
        or local_node() != NODE
        or state_path.exists()
    ):
        raise RuntimeError("Stage-profile validation has no original acknowledged controller ownership")
    require_acknowledged_controller(intent, own)
    state = {
        "status": "preflight",
        "binding_sha256": sha256(binding_path),
        "controller_pid": own["pid"],
        "controller_start_ticks": own["start_ticks"],
        "node": NODE,
        "candidate": frozen["candidate"],
    }
    write(state_path, state)
    signals = []
    try:
        # Frozen production probes take the node lock themselves; reconstruct before acquiring it locally.
        state["predecessor"] = require_completed_wave12(root)
        with node_guard(root, NODE) as signals, keeper_guard() as manager_lock:
            idle_training()
            accepted = state["predecessor"]["workers"]["muont"]["restored_keeper"]
            if not same_keeper(keeper_observation(), accepted):
                raise RuntimeError("Stage-profile allocation differs from the exact completed Wave12 keeper")
            subprocess.run(
                [PYTHON, "-m", "experiment.bootstrap", "--check-only"],
                cwd=Path(frozen["candidate"]),
                env=os.environ | ENVIRONMENT,
                capture_output=True,
                text=True,
                check=True,
                timeout=240,
            )
            if binding(root) != frozen or not same_keeper(keeper_observation(), accepted):
                raise RuntimeError("Stage-profile frozen identity or keeper changed before transfer")
            try:
                release_accepted_keeper(accepted, state, state_path, manager_lock)
                code = bounded_validator(root, frozen, state)
                state["exit_code"] = code
                if type(code) is not int or code != 0:
                    raise RuntimeError("Stage-profile bounded GPU validator failed")
                actual = reconstruct(root, Path(frozen["candidate"]), receipt_path)
                if actual["owner"] != {
                    "controller_pid": own["pid"],
                    "controller_start_ticks": own["start_ticks"],
                }:
                    raise ValueError("Stage-profile GPU evidence belongs to a different controller generation")
                state.update(status="passed", validation_sha256=sha256(receipt_path))
            finally:
                if state.get("keeper_release_attempted"):
                    try:
                        if (
                            state.get("keeper_release_complete") is not True
                            or "cleanup_error" in state
                            or (state.get("child_marker") and marker_processes(CHILD_MARKER, state["child_marker"]))
                        ):
                            raise RuntimeError(
                                "Stage-profile incomplete transfer or marked descendants prevent restoration"
                            )
                        restore_accepted_keeper(manager_lock)
                        restored = keeper_observation()
                        if restored["devices"] != accepted["devices"]:
                            raise RuntimeError("Stage-profile restored keeper has a different physical allocation")
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
