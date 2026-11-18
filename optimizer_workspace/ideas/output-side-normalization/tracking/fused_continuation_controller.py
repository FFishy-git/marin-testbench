# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze and own the separate six-command continuation diagnostic on the exact restored allocation."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from uuid import uuid4

import fused_continuation_validation as validation
import owned_output_handoff as handoff
import prepare_fused_attribution_validation as attribution_preparation
from fused_attribution_validation import require_failed_profiler_handoff, require_keeper_shape
from fused_attribution_validation_contract import locations as parent_locations
from fused_attribution_validation_contract import trial_specs
from fused_attribution_validation_controller import (
    acknowledged_process,
    finish_marked_child,
    process_context,
    require_acknowledged_controller,
    require_helper_bytes,
)
from fused_continuation_acceptance import require_cpu
from fused_continuation_contract import (
    CHILD_MARKER,
    CONTROLLER_MARKER,
    ENVIRONMENT,
    FROZEN_STATUS,
    NODE,
    PREPARATION,
    PYTHON,
    controller_command,
    locations,
    plan,
    validator_command,
)
from prepare_fused_continuation import failed_attribution
from prepare_selected_lr_stage_profile_validation import helper_names as dependencies
from prepare_selected_lr_validation import require_archive
from selected_lr_failure_handoff import keeper_guard, release_accepted_keeper, restore_accepted_keeper
from selected_lr_release_owner import candidate_files, read, sha256
from watch_wave5 import write
from wave7_release import idle_training, local_node, node_guard

TESTS = (
    "test_fused_continuation_validation.py",
    "test_fused_continuation_controller.py",
    "test_fused_attribution_validation_controller.py",
    "test_selected_lr_failure_handoff.py",
)
CONTROLLER_CPU_STATUS = "fused_continuation_controller_cpu_archive_verified_gpu_unissued"
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
MANAGER = "recovery_keeper_manager.sh"
TIMEOUT = 6 * validation.COMMAND_TIMEOUT + 8100


def predecessor(root: Path) -> dict:
    return failed_attribution(root, read(parent_locations(root)[0]))


def helper_sources(root: Path, directory: Path) -> dict[str, Path]:
    names = dependencies(
        directory,
        (
            Path(__file__).name,
            "selected_lr_release_owner.py",
            *TESTS,
            "test_fused_attribution_validation_owner.py",
            *attribution_preparation.COPIES,
            *attribution_preparation.TESTS,
            "fused_continuation_trace.py",
            "test_fused_continuation_trace.py",
            "test_prepare_fused_continuation.py",
        ),
    )
    result = {name: directory / name for name in names}
    parent = read(parent_locations(root)[0])
    manager = directory / MANAGER
    if not manager.exists():
        manager = Path(parent["helpers"]) / MANAGER
    if manager.is_symlink() or sha256(manager) != parent["helpers_sha256"][MANAGER]:
        raise ValueError("Continuation serialized manager differs from original attribution archive")
    result[MANAGER] = manager
    return result


def require_loaded(helpers: Path, hashes: dict) -> None:
    for name in hashes:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Continuation imported mutable helper: {name}")
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Continuation controller must execute its own frozen source")


def cpu_command() -> list[str]:
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
    cpu = require_cpu(root, candidate_receipt)
    receipt_digest = sha256(candidate_receipt)
    candidate = Path(cpu["candidate"])
    if out.exists() or out.is_relative_to(candidate) or candidate.is_relative_to(out):
        raise FileExistsError("Continuation controller CPU evidence needs an unused external directory")
    sources = helper_sources(root, Path(__file__).parent)
    hashes = {name: sha256(path) for name, path in sources.items()}
    if any(hashes.get(name) != value for name, value in cpu["helpers_sha256"].items()):
        raise ValueError("Continuation controller helper closure differs from accepted candidate helpers")
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    for name, path in sources.items():
        shutil.copy2(path, helpers / name)
    fixtures = {}
    for name in fixture_names():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate / name, path)
        fixtures[name] = sha256(path)
    log, archive = out / "controller.log", out / "controller-source.tar"
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(),
                cwd=out,
                env=os.environ
                | {
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": os.pathsep.join((str(candidate), str(helpers))),
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except (OSError, subprocess.TimeoutExpired) as failure:
            code, error = 124, f"{type(failure).__name__}: {failure}"
    expected = {f"tracking/{name}": value for name, value in hashes.items()} | fixtures
    try:
        with tarfile.open(archive, "x") as bundle:
            for name in expected:
                bundle.add(out / name, arcname=name, recursive=False)
        require_archive(archive, expected, alias=False)
        if (
            hashes != {name: sha256(path) for name, path in sources.items()}
            or hashes != {name: sha256(helpers / name) for name in hashes}
            or fixtures != {name: sha256(out / name) for name in fixtures}
            or require_cpu(root, candidate_receipt) != cpu
            or sha256(candidate_receipt) != receipt_digest
        ):
            raise ValueError("Continuation controller or candidate changed during CPU verification")
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as failure:
        error = f"{type(failure).__name__}: {failure}"
    result = {
        "status": CONTROLLER_CPU_STATUS if code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate_cpu_receipt": str(candidate_receipt),
        "candidate_cpu_receipt_sha256": receipt_digest,
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "fixture_files_sha256": fixtures,
        "command": cpu_command(),
        "cwd": str(out),
        "exit_code": code,
        "cuda_visible_devices": "",
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive) if archive.is_file() else None,
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != CONTROLLER_CPU_STATUS:
        raise RuntimeError(f"Continuation controller CPU failure preserved at {out}")
    try:
        return require_controller_cpu(root, out / "receipt.json", candidate_receipt)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as failure:
        result.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(out / "receipt.json", result)
        raise RuntimeError(f"Continuation controller final CPU/archive verification failed at {out}") from failure


def require_controller_cpu(root: Path, receipt_path: Path, candidate_receipt: Path) -> dict:
    raw = receipt_path.read_bytes()
    receipt, cpu = read(receipt_path), require_cpu(root, candidate_receipt)
    helpers = Path(receipt["helpers"])
    require_helper_bytes(helpers, receipt["helpers_sha256"])
    if (
        receipt["status"] != CONTROLLER_CPU_STATUS
        or receipt["root"] != str(root)
        or receipt["candidate_cpu_receipt"] != str(candidate_receipt)
        or receipt["candidate_cpu_receipt_sha256"] != sha256(candidate_receipt)
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["cuda_visible_devices"] != ""
        or "error" in receipt
        or receipt["command"] != cpu_command()
        or receipt["cwd"] != str(helpers.parent)
        or receipt["helpers_sha256"] != {name: sha256(path) for name, path in helper_sources(root, helpers).items()}
        or any(receipt["helpers_sha256"].get(name) != value for name, value in cpu["helpers_sha256"].items())
        or receipt["fixture_files_sha256"] != {name: cpu["files_sha256"][name] for name in fixture_names()}
        or any(sha256(helpers.parent / name) != cpu["files_sha256"][name] for name in fixture_names())
        or receipt["log"] != str(helpers.parent / "controller.log")
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive"] != str(helpers.parent / "controller-source.tar")
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
    ):
        raise ValueError("Continuation controller CPU acceptance differs")
    require_archive(
        Path(receipt["archive"]),
        {f"tracking/{name}": value for name, value in receipt["helpers_sha256"].items()}
        | receipt["fixture_files_sha256"],
        alias=False,
    )
    if receipt_path.read_bytes() != raw:
        raise ValueError("Continuation controller CPU receipt changed during authentication")
    return receipt


def inventory(frozen: dict) -> dict:
    prepared = read(Path(frozen["candidate"]) / PREPARATION)
    return (
        frozen["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {
            PREPARATION: frozen["preparation_sha256"],
            "candidate-cpu-receipt.json": frozen["cpu_receipt_sha256"],
            "controller-cpu-receipt.json": frozen["controller_cpu_receipt_sha256"],
        }
        | {f"tracking/{name}": value for name, value in frozen["helpers_sha256"].items()}
    )


def freeze(root: Path, candidate_receipt: Path, controller_receipt: Path) -> dict:
    cpu = require_cpu(root, candidate_receipt)
    controller = require_controller_cpu(root, controller_receipt, candidate_receipt)
    candidate = Path(cpu["candidate"])
    prepared = read(candidate / PREPARATION)
    previous = predecessor(root)
    binding_path, helpers, state_path, work, receipt_path = locations(root)
    archive = root / "run-preparation/source-fused-continuation-20261118.tar"
    sources = helper_sources(root, Path(__file__).parent)
    hashes = {name: sha256(path) for name, path in sources.items()}
    if previous != prepared["allocation_predecessor"] or hashes != controller["helpers_sha256"]:
        raise ValueError("Continuation freeze requires exact CPU-tested closure and failed predecessor")
    with (root / "run-preparation/.fused-continuation-freeze.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.exists() for path in (binding_path, helpers.parent, archive, state_path, work, receipt_path)):
            raise FileExistsError("Continuation diagnostic identity already has evidence; preserve it")
        helpers.mkdir(parents=True)
        for name, path in sources.items():
            shutil.copy2(path, helpers / name)
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
            "data_receipt_sha256": prepared["data_receipt_sha256"],
            "allocation_predecessor": previous,
            "plan": plan(candidate, root),
            "commands": plan(candidate, root)["commands"],
            "validation_timeout": TIMEOUT,
            "policy": "Diagnostic only; no production authorization.",
        }
        expected = inventory(frozen)
        with tarfile.open(archive, "x") as bundle:
            for name in expected:
                if name.startswith("tracking/"):
                    path = helpers / Path(name).name
                elif name == "candidate-cpu-receipt.json":
                    path = candidate_receipt
                elif name == "controller-cpu-receipt.json":
                    path = controller_receipt
                else:
                    path = candidate / name
                bundle.add(path, arcname=name, recursive=False)
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
        frozen["archive_sha256"] = sha256(archive)
        require_archive(archive, expected, alias=True)
        if (
            require_cpu(root, candidate_receipt) != cpu
            or require_controller_cpu(root, controller_receipt, candidate_receipt) != controller
            or hashes != {name: sha256(path) for name, path in sources.items()}
            or hashes != {name: sha256(helpers / name) for name in hashes}
            or predecessor(root) != previous
        ):
            raise ValueError("Continuation issued source or predecessor changed during freeze")
        write(binding_path, frozen)
    return frozen


def binding(root: Path) -> dict:
    path, helpers, _, _, _ = locations(root)
    raw = path.read_bytes()
    frozen = read(path)
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    require_loaded(helpers, frozen["helpers_sha256"])
    cpu = require_cpu(root, Path(frozen["cpu_receipt"]))
    controller = require_controller_cpu(root, Path(frozen["controller_cpu_receipt"]), Path(frozen["cpu_receipt"]))
    candidate = Path(cpu["candidate"])
    if (
        root != root.resolve()
        or frozen["root"] != str(root)
        or frozen["candidate"] != str(candidate)
        or frozen["status"] != FROZEN_STATUS
        or frozen["node"] != NODE
        or frozen["python"] != PYTHON
        or frozen["environment"] != ENVIRONMENT
        or frozen["files_sha256"] != candidate_files(candidate)
        or frozen["preparation_sha256"] != sha256(candidate / PREPARATION)
        or frozen["cpu_receipt_sha256"] != sha256(Path(frozen["cpu_receipt"]))
        or frozen["controller_cpu_receipt_sha256"] != sha256(Path(frozen["controller_cpu_receipt"]))
        or frozen["helpers"] != str(helpers)
        or frozen["helpers_sha256"] != controller["helpers_sha256"]
        or frozen["helpers_sha256"] != {name: sha256(path) for name, path in helper_sources(root, helpers).items()}
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["allocation_predecessor"] != predecessor(root)
        or frozen["allocation_predecessor"] != read(candidate / PREPARATION)["allocation_predecessor"]
        or frozen["plan"] != plan(candidate, root)
        or frozen["commands"] != plan(candidate, root)["commands"]
        or frozen["validation_timeout"] != TIMEOUT
        or frozen["archive"] != str(root / "run-preparation/source-fused-continuation-20261118.tar")
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
    ):
        raise ValueError("Continuation frozen binding, source, commands or exact predecessor differs")
    require_archive(Path(frozen["archive"]), inventory(frozen), alias=True)
    if path.read_bytes() != raw:
        raise ValueError("Continuation binding changed during authentication")
    return frozen


def probe(root: Path, expected: dict) -> dict:
    if predecessor(root) != expected:
        raise RuntimeError("Continuation original failure evidence changed")
    current = handoff.keeper_observation()
    marked = handoff.marker_processes(expected["controller_marker_name"], expected["controller_marker"])
    children = handoff.marker_processes(expected["child_marker_name"], expected["child_marker"])
    handles = {pid: handoff.observed(pid) for pid in (expected["controller_pid"], expected["child_pid"])}
    previous = expected | {"status": "failed_profiler_exact_restored_keeper_pending_live_handoff"}
    require_failed_profiler_handoff(previous, current, marked, children, handles)
    if predecessor(root) != expected:
        raise RuntimeError("Continuation original failure evidence changed during live handoff")
    return {
        "status": "failed_attribution_exact_keeper_live_handoff_verified",
        "keeper": current,
        "original_handles_absent": True,
        "child_markers_absent": True,
        "inherited_controller_marker_pids": sorted(marked),
    }


def start(root: Path) -> dict:
    frozen = binding(root)
    path, helpers, state, work, receipt = locations(root)
    intent_path, log = helpers.parent / "controller.json", helpers.parent / "controller.log"
    if local_node() != NODE or any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()):
        raise RuntimeError("Continuation start requires its bound node and explicit eight-GPU environment")
    with (helpers.parent / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(target.exists() for target in (intent_path, log, state, work, receipt)):
            raise FileExistsError("Continuation execution already has intent or evidence; never retry")
        with keeper_guard():
            idle_training()
            probe(root, frozen["allocation_predecessor"])
        intent = {
            "status": "starting",
            "binding_sha256": sha256(path),
            "command": controller_command(root),
            "marker": uuid4().hex,
            "allocation_predecessor": frozen["allocation_predecessor"],
        }
        write(intent_path, intent)
        with log.open("x") as stream:
            child = subprocess.Popen(
                intent["command"],
                cwd=helpers,
                env=os.environ
                | ENVIRONMENT
                | {CONTROLLER_MARKER: intent["marker"], "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(helpers)},
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


def bounded_validator(root: Path, frozen: dict, state: dict) -> int:
    _, helpers, state_path, _, _ = locations(root)
    candidate = Path(frozen["candidate"])
    command, marker = validator_command(root, candidate), uuid4().hex
    log, generations = helpers.parent / "validator.log", {}
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
            return child.wait(timeout=TIMEOUT)
        finally:
            try:
                finish_marked_child(child, marker, generations, CHILD_MARKER)
                state.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
            except BaseException as error:
                state["cleanup_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                state.update(child_generation=generations, log_sha256=sha256(log))
                write(state_path, state)


def validate(root: Path) -> dict:
    frozen = binding(root)
    path, helpers, state_path, _, receipt = locations(root)
    intent_path = helpers.parent / "controller.json"
    deadline, intent = time.monotonic() + 30, read(intent_path)
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
    require_acknowledged_controller(intent, own)
    if (
        intent["binding_sha256"] != sha256(path)
        or intent["command"] != controller_command(root)
        or intent["marker"] != os.environ.get(CONTROLLER_MARKER)
        or local_node() != NODE
        or state_path.exists()
        or intent["allocation_predecessor"] != frozen["allocation_predecessor"]
    ):
        raise RuntimeError("Continuation has no exact acknowledged controller ownership")
    state = {
        "status": "preflight",
        "binding_sha256": sha256(path),
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
            accepted = probe(root, frozen["allocation_predecessor"])["keeper"]
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
                raise RuntimeError("Continuation binding changed before exact keeper transfer")
            state["handoff"] = probe(root, frozen["allocation_predecessor"])
            try:
                release_accepted_keeper(accepted, state, state_path, manager_lock)
                code = bounded_validator(root, frozen, state)
                state["exit_code"] = code
                if type(code) is not int or code != 0:
                    raise RuntimeError("Continuation bounded diagnostic validator failed")
                actual = validation.reconstruct(root, Path(frozen["candidate"]), receipt)
                if actual["status"] != validation.COMPLETE_STATUS:
                    raise ValueError("Continuation diagnostic reconstruction did not complete")
                state.update(status="diagnostic_complete", validation_sha256=sha256(receipt))
            finally:
                if state.get("keeper_release_attempted"):
                    if (
                        state.get("keeper_release_complete") is not True
                        or "cleanup_error" in state
                        or (state.get("child_marker") and handoff.marker_processes(CHILD_MARKER, state["child_marker"]))
                    ):
                        state["keeper_restore"] = "refused: incomplete transfer or live descendants"
                        raise RuntimeError("Continuation incomplete transfer prevents keeper restoration")
                    try:
                        restore_accepted_keeper(manager_lock)
                        restored = handoff.keeper_observation()
                        require_keeper_shape(restored)
                        if restored["devices"] != accepted["devices"]:
                            raise RuntimeError("Continuation restored keeper has different physical GPUs")
                        state.update(restored_keeper=restored, keeper_restore=0)
                    except BaseException as failure:
                        state["keeper_restore"] = f"{type(failure).__name__}: {failure}"
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
    root = args.root.resolve()
    if args.phase in ("cpu", "freeze") and args.cpu_receipt is None:
        parser.error("CPU/freeze require --cpu-receipt")
    if args.phase == "cpu" and args.out is None:
        parser.error("CPU requires --out")
    if args.phase == "freeze" and args.controller_cpu_receipt is None:
        parser.error("Freeze requires --controller-cpu-receipt")
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
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
