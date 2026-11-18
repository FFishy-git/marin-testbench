# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own full-budget attribution waves after original numerical acceptance and exact handoff."""

import argparse
import fcntl
import json
import os
import select
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import fused_attribution_run_history as run_history
import prepare_fused_attribution_production as preparation
from fused_attribution_production_contract import (
    CHILD_MARKER,
    CONTROLLER_CPU_STATUS,
    ENVIRONMENT,
    FAMILIES,
    FROZEN_STATUS,
    MONITOR_INTERVAL,
    PREPARATION,
    PYTHON,
    TRAINING_TIMEOUT,
    WAVES,
    WORKER_MARKER,
    commands,
    config_name,
    node_for,
    paths,
    wave_spec,
    worker_command,
    worker_path,
)
from fused_attribution_production_handoff import allocation_observation, require_acceptance, validation_acceptance
from launch_wave5 import read, sha256
from owned_output_handoff import keeper_observation, marker_processes, observed, require_absent, same_generation
from prepare_selected_lr_stage_profile_validation import helper_names
from prepare_selected_lr_validation import require_archive
from selected_lr_failure_handoff import keeper_guard, pidfd_open, release_accepted_keeper, restore_accepted_keeper
from watch_wave5 import write
from wave7_release import idle_training, local_node, node_guard

TESTS = (
    "test_fused_attribution_production.py",
    "test_fused_attribution_production_handoff.py",
    "test_prepare_fused_attribution_production.py",
    "test_fused_attribution_production_owner.py",
    "test_fused_attribution_run_history.py",
)
EXTRA_HELPERS = (
    "fused_attribution_production.py",
    "fused_attribution_production_contract.py",
    "fused_attribution_production_handoff.py",
    "fused_attribution_run_history.py",
    "prepare_fused_attribution_production.py",
    "fused_attribution_production_owner.py",
    "analyze_fused_attribution_run.py",
    "swanlab_fused_attribution_run.py",
)
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
ACK_TIMEOUT = 30


def helper_sources(root: Path, directory: Path) -> dict[str, Path]:
    original = preparation.original(root)
    names = helper_names(directory, (*EXTRA_HELPERS, *TESTS))
    sources = {name: directory / name for name in names}
    parent = Path(original["helpers"])
    for name, digest in original["helpers_sha256"].items():
        source = parent / name
        if source.is_symlink() or sha256(source) != digest:
            raise ValueError("Original attribution validation helper changed")
        if name in sources and sha256(sources[name]) != digest:
            raise ValueError("Production helper conflicts with original attribution validation bytes")
        sources[name] = source
    sources["monitor_fused_attribution_run.sh"] = directory / "monitor_fused_attribution_run.sh"
    return sources


def fixture_names() -> tuple[str, ...]:
    return tuple(config_name(family, wave_spec(wave)["lr_label"]) for wave in WAVES for family in FAMILIES)


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


def controller_cpu(root: Path, cpu_receipt: Path, out: Path) -> dict:
    root, cpu_receipt, out = root.resolve(), cpu_receipt.resolve(), out.resolve()
    cpu = preparation.require_cpu(root, cpu_receipt)
    candidate = Path(cpu["candidate"])
    if out.exists() or out.is_relative_to(candidate) or candidate.is_relative_to(out):
        raise FileExistsError("Production controller evidence requires an unused external directory")
    sources = helper_sources(root, Path(__file__).parent)
    hashes = {name: sha256(source) for name, source in sources.items()}
    if any(hashes.get(name) != digest for name, digest in cpu["helpers_sha256"].items()):
        raise ValueError("Production controller changed tested candidate helpers")
    cpu_receipt_digest = sha256(cpu_receipt)
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    for name, source in sources.items():
        shutil.copy2(source, helpers / name)
    fixtures = {}
    for name in fixture_names():
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate / name, destination)
        fixtures[name] = cpu["files_sha256"][name]
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
                    "PYTHONPATH": str(helpers),
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Production controller CPU suite timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    expected = {f"tracking/{name}": digest for name, digest in hashes.items()} | fixtures
    try:
        with tarfile.open(archive, "x") as bundle:
            for name in expected:
                bundle.add(out / name, arcname=name, recursive=False)
        require_archive(archive, expected, alias=False)
        if (
            any(sha256(helpers / name) != digest or sha256(sources[name]) != digest for name, digest in hashes.items())
            or any(sha256(out / name) != digest for name, digest in fixtures.items())
            or preparation.require_cpu(root, cpu_receipt) != cpu
            or sha256(cpu_receipt) != cpu_receipt_digest
        ):
            raise ValueError("Production candidate or controller changed during CPU suite")
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as failure:
        detail = f"{type(failure).__name__}: {failure}"
        error = detail if error is None else f"{error}; {detail}"
    receipt = {
        "status": CONTROLLER_CPU_STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate": str(candidate),
        "candidate_cpu_receipt": str(cpu_receipt),
        "candidate_cpu_receipt_sha256": cpu_receipt_digest,
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "fixture_files_sha256": fixtures,
        "command": cpu_command(),
        "cwd": str(out),
        "cuda_visible_devices": "",
        "exit_code": code,
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive) if archive.is_file() else None,
    }
    if error is not None:
        receipt["error"] = error
    write(out / "receipt.json", receipt)
    if receipt["status"] != CONTROLLER_CPU_STATUS:
        raise RuntimeError("Production controller CPU suite failed; original receipt and log preserved")
    try:
        return require_controller_cpu(root, out / "receipt.json", cpu_receipt)
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as failure:
        receipt.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(out / "receipt.json", receipt)
        raise RuntimeError("Production controller final CPU/archive failure preserved") from failure


def require_controller_cpu(root: Path, receipt_path: Path, cpu_path: Path) -> dict:
    root, receipt_path, cpu_path = root.resolve(), receipt_path.resolve(), cpu_path.resolve()
    raw = receipt_path.read_bytes()
    result = json.loads(raw)
    cpu = preparation.require_cpu(root, cpu_path)
    helpers = Path(result["helpers"])
    if (
        helpers != receipt_path.parent / "tracking"
        or helpers.is_symlink()
        or {path.name for path in helpers.iterdir() if path.name != "__pycache__"} != set(result["helpers_sha256"])
    ):
        raise ValueError("Production controller CPU helper origin or inventory differs")
    sources = helper_sources(root, helpers)
    hashes = {name: sha256(source) for name, source in sources.items()}
    fixtures = {name: cpu["files_sha256"][name] for name in fixture_names()}
    if (
        result["status"] != CONTROLLER_CPU_STATUS
        or result["root"] != str(root)
        or result["candidate"] != cpu["candidate"]
        or result["candidate_cpu_receipt"] != str(cpu_path)
        or result["candidate_cpu_receipt_sha256"] != sha256(cpu_path)
        or type(result["exit_code"]) is not int
        or result["exit_code"] != 0
        or result["cuda_visible_devices"] != ""
        or result["command"] != cpu_command()
        or result["cwd"] != str(helpers.parent)
        or any(key in result for key in ("error", "errors"))
        or result["helpers_sha256"] != hashes
        or any(hashes.get(name) != digest for name, digest in cpu["helpers_sha256"].items())
        or result["fixture_files_sha256"] != fixtures
        or any(sha256(helpers.parent / name) != digest for name, digest in fixtures.items())
        or result["log"] != str(helpers.parent / "controller.log")
        or result["archive"] != str(helpers.parent / "controller-source.tar")
        or sha256(Path(result["log"])) != result["log_sha256"]
        or sha256(Path(result["archive"])) != result["archive_sha256"]
        or any(
            Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != digest
            for name, digest in result["helpers_sha256"].items()
        )
    ):
        raise ValueError("Production controller CPU acceptance differs")
    expected = {f"tracking/{name}": digest for name, digest in hashes.items()} | fixtures
    require_archive(Path(result["archive"]), expected, alias=False)
    if receipt_path.read_bytes() != raw:
        raise ValueError("Production controller CPU receipt changed during verification")
    return result


def remote_command(root: Path, wave: str, family: str, helpers: Path, phase: str) -> list[str]:
    command = [
        "env",
        *(f"{key}={value}" for key, value in ENVIRONMENT.items()),
        f"PYTHONPATH={helpers}",
        "PYTHONDONTWRITEBYTECODE=1",
        PYTHON,
        "-P",
        str(helpers / Path(__file__).name),
        "--root",
        str(root),
        "--wave",
        wave,
        "--phase",
        phase,
        "--family",
        family,
    ]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        f"root@{node_for(wave, family)}",
        shlex.join(command),
    ]


def remote(root: Path, wave: str, family: str, helpers: Path, phase: str) -> dict:
    result = subprocess.run(
        remote_command(root, wave, family, helpers, phase),
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )
    return json.loads(result.stdout.splitlines()[-1])


def freeze(root: Path, wave: str, cpu_path: Path, controller_path: Path) -> dict:
    binding_path, helpers, launch_path, workers = paths(root, wave)
    with (root / f"run-preparation/.{wave}-freeze.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.exists() for path in (binding_path, helpers.parent, launch_path, workers, root / f"runs/{wave}")):
            raise FileExistsError("Attribution wave already has source or execution evidence")
        cpu = preparation.require_cpu(root, cpu_path)
        controller = require_controller_cpu(root, controller_path, cpu_path)
        candidate = Path(cpu["candidate"])
        acceptance_path = root / "run-preparation/fused-attribution-production-validation-acceptance-20261118.json"
        if not acceptance_path.exists():
            validation_acceptance(root, acceptance_path)
        acceptance = require_acceptance(root, acceptance_path)
        predecessor = wave_spec(wave)["predecessor_wave"]
        preceding = {}
        if predecessor is not None:
            previous_helpers = paths(root, predecessor)[1]
            checked = subprocess.run(
                [
                    PYTHON,
                    "-P",
                    str(previous_helpers / Path(__file__).name),
                    "--root",
                    str(root),
                    "--wave",
                    predecessor,
                    "--phase",
                    "completion",
                ],
                env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(previous_helpers)},
                capture_output=True,
                text=True,
                check=True,
                timeout=1800,
            )
            preceding = json.loads(checked.stdout.splitlines()[-1])
        original_helpers = Path(controller["helpers"])
        allocations = {
            node_for(wave, family): remote(root, wave, family, original_helpers, "allocation") for family in FAMILIES
        }
        helpers.mkdir(parents=True)
        for name, digest in controller["helpers_sha256"].items():
            source = original_helpers / name
            if sha256(source) != digest:
                raise ValueError("Archived production controller changed before freeze")
            shutil.copy2(source, helpers / name)
        archive = root / f"run-preparation/source-{wave}-fused-attribution-20261118.tar"
        prepared = read(candidate / PREPARATION)
        expected = (
            preparation.archive_inventory(prepared, cpu["preparation_sha256"])
            | {
                "candidate-cpu-receipt.json": sha256(cpu_path),
                "controller-cpu-receipt.json": sha256(controller_path),
                "validation-acceptance.json": sha256(acceptance_path),
            }
            | {f"tracking/{name}": digest for name, digest in controller["helpers_sha256"].items()}
        )
        candidate_inventory = preparation.archive_inventory(prepared, cpu["preparation_sha256"])
        if any(
            candidate_inventory.get(f"tracking/{name}", digest) != digest
            for name, digest in controller["helpers_sha256"].items()
        ):
            raise ValueError("Production archive would overwrite candidate-bound helper bytes")
        with tarfile.open(archive, "x") as bundle:
            for name in expected:
                if name.startswith("tracking/"):
                    source = helpers / Path(name).name
                elif name == "candidate-cpu-receipt.json":
                    source = cpu_path
                elif name == "controller-cpu-receipt.json":
                    source = controller_path
                elif name == "validation-acceptance.json":
                    source = acceptance_path
                else:
                    source = candidate / name
                bundle.add(source, arcname=name, recursive=False)
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
        require_archive(archive, expected, alias=True)
        spec = wave_spec(wave)
        release = {
            "status": FROZEN_STATUS,
            "root": str(root),
            "candidate": str(candidate),
            "wave": wave,
            "families": list(FAMILIES),
            "arms": list(FAMILIES),
            "nodes": spec["nodes"],
            "matrix_lr": spec["matrix_lr"],
            "lr_label": spec["lr_label"],
            "environment": ENVIRONMENT,
            "files_sha256": cpu["files_sha256"],
            "preparation_sha256": cpu["preparation_sha256"],
            "cpu_receipt": str(cpu_path),
            "cpu_receipt_sha256": sha256(cpu_path),
            "controller_cpu_receipt": str(controller_path),
            "controller_cpu_receipt_sha256": sha256(controller_path),
            "helpers": str(helpers),
            "helpers_sha256": controller["helpers_sha256"],
            "archive": str(archive),
            "archive_sha256": sha256(archive),
            "validation_binding": str(root / "run-preparation/fused-attribution-validation-binding-20261118.json"),
            "validation_binding_sha256": acceptance["binding_sha256"],
            "validation_acceptance": str(acceptance_path),
            "validation_acceptance_sha256": sha256(acceptance_path),
            "allocation_by_node": allocations,
            "preceding_wave": preceding,
            "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
            "commands": commands(root, candidate, wave),
            "monitor_interval_seconds": MONITOR_INTERVAL,
            "training_timeout_seconds": TRAINING_TIMEOUT,
        }
        if (
            preparation.require_cpu(root, cpu_path) != cpu
            or require_controller_cpu(root, controller_path, cpu_path) != controller
            or require_acceptance(root, acceptance_path) != acceptance
        ):
            raise ValueError("Production source or original numerical acceptance changed during freeze")
        for family in FAMILIES:
            current = remote(root, wave, family, helpers, "allocation")
            if not same_keeper(current["keeper"], allocations[node_for(wave, family)]["keeper"]):
                raise ValueError("Accepted keeper generation changed before production publication")
        if require_controller_cpu(root, controller_path, cpu_path) != controller or any(
            sha256(helpers / name) != digest for name, digest in controller["helpers_sha256"].items()
        ):
            raise ValueError("Production controller changed before publication")
        write(binding_path, release)
    return release


def same_keeper(actual: dict, expected: dict) -> bool:
    before = {int(pid): process for pid, process in expected["generation"].items()}
    after = {int(pid): process for pid, process in actual["generation"].items()}
    return (
        actual["pid"] == expected["pid"]
        and actual["devices"] == expected["devices"]
        and sorted(actual["occupied"]) == sorted(expected["occupied"])
        and set(before) == set(after)
        and all(same_generation(process, after[pid]) for pid, process in before.items())
    )


def require_process(pid: int, command: list[str], cwd: Path, markers: dict[str, str]) -> dict:
    process = observed(pid)
    directory = Path("/proc") / str(pid)
    if process is None or process["status"] in ("Z", "X", "x"):
        raise RuntimeError("Attribution process is absent or exited")
    environment = dict(
        token.decode().split("=", 1) for token in (directory / "environ").read_bytes().split(b"\0") if token
    )
    if (
        process["command"] != command
        or process["group"] != pid
        or process["session"] != pid
        or (directory / "cwd").resolve() != cwd
        or any(not value or environment.get(name) != value for name, value in markers.items())
        or any(environment.get(name) != value for name, value in ENVIRONMENT.items())
        or not same_generation(process, observed(pid))
    ):
        raise RuntimeError("Attribution command, session, cwd, markers or environment differs")
    return process


def acknowledge(child: subprocess.Popen, command: list[str], cwd: Path, markers: dict[str, str]) -> dict:
    deadline = time.monotonic() + ACK_TIMEOUT
    while True:
        process = observed(child.pid)
        if process is not None and process["command"] == command:
            return require_process(child.pid, command, cwd, markers)
        if child.poll() is not None or time.monotonic() >= deadline:
            raise RuntimeError("Attribution child exited or did not reach exact executable acknowledgment")
        time.sleep(0.01)


def finish_child(child: subprocess.Popen, marker: str, generation: dict) -> None:
    """Bind pidfds before signaling independently sessioned or reparented training ranks."""
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
                            raise RuntimeError("Marked attribution process changed during cleanup binding")
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
        raise TimeoutError("Attribution training or marked ranks remain after bounded cleanup")


def run_child(root: Path, wave: str, family: str, release: dict, state: dict, path: Path) -> int:
    command = release["commands"][node_for(wave, family)]
    marker, generation = uuid4().hex, {}
    log = root / f"runs/{wave}/{family}.torchrun.log"
    state.update(command=command, child_marker=marker, log=str(log))
    write(path, state)
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=release["candidate"],
            env=os.environ | ENVIRONMENT | {CHILD_MARKER: marker},
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process = acknowledge(
                child,
                command,
                Path(release["candidate"]),
                {WORKER_MARKER: state["worker_marker"], CHILD_MARKER: marker},
            )
            state.update(status="running", child_pid=child.pid, child_start_ticks=process["start_ticks"])
            write(path, state)
            deadline = time.monotonic() + TRAINING_TIMEOUT
            while True:
                generation.update(
                    {str(pid): item["start_ticks"] for pid, item in marker_processes(CHILD_MARKER, marker).items()}
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, TRAINING_TIMEOUT)
                try:
                    return child.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            try:
                finish_child(child, marker, generation)
                state["cleanup_exit_code"] = child.returncode
                state["marked_descendants_absent"] = True
            except BaseException as error:
                state["cleanup_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                state.update(child_generation=generation, log_sha256=sha256(log))
                write(path, state)


def run_worker(root: Path, wave: str, family: str) -> dict:
    release = run_history.binding(root, wave)
    path = worker_path(root, wave, family)
    intent_path = path.with_suffix(".start.json")
    deadline, intent = time.monotonic() + ACK_TIMEOUT, read(intent_path)
    while intent["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.05)
        intent = read(intent_path)
    own = require_process(
        os.getpid(),
        worker_command(root, wave, family),
        Path(release["candidate"]),
        {WORKER_MARKER: intent["worker_marker"]},
    )
    expected = intent["generation"]
    launch = read(paths(root, wave)[2])
    if (
        intent["status"] != "acknowledged"
        or intent["pid"] != own["pid"]
        or intent["start_ticks"] != own["start_ticks"]
        or own["parent"] not in (expected["parent"], 1)
        or any(own[key] != expected[key] for key in ("pid", "start_ticks", "group", "session", "command"))
        or intent["release_binding_sha256"] != sha256(paths(root, wave)[0])
        or intent["worker_marker"] != launch["worker_markers"][family]
        or launch["binding_sha256"] != intent["release_binding_sha256"]
        or launch["status"] not in ("launching", "launched")
        or path.exists()
    ):
        raise RuntimeError("Attribution worker lacks its original one-shot acknowledgment")
    state = {
        "status": "preflight",
        "family": family,
        "arm": family,
        "wave": wave,
        "node": node_for(wave, family),
        "worker_pid": own["pid"],
        "worker_start_ticks": own["start_ticks"],
        "worker_command": own["command"],
        "worker_marker": intent["worker_marker"],
        "release_binding_sha256": sha256(paths(root, wave)[0]),
        "out": str(root / f"runs/{wave}/{family}"),
    }
    write(path, state)
    try:
        with node_guard(root, node_for(wave, family)) as signals, keeper_guard() as manager_lock:
            if local_node() != node_for(wave, family):
                raise RuntimeError("Attribution worker is on a different node")
            idle_training()
            observed_allocation = allocation_observation(
                root,
                wave,
                family,
                expected=release["allocation_by_node"][state["node"]],
                manager_lock=manager_lock,
            )
            accepted = observed_allocation["keeper"]
            subprocess.run(
                [PYTHON, "-m", "experiment.bootstrap", "--check-only"],
                cwd=release["candidate"],
                env=os.environ | ENVIRONMENT,
                capture_output=True,
                text=True,
                check=True,
                timeout=240,
            )
            if Path(state["out"]).exists():
                raise FileExistsError("Attribution output already exists; preserve without retry")
            if not same_keeper(keeper_observation(), accepted):
                raise RuntimeError("Attribution keeper changed during preflight")
            try:
                release_accepted_keeper(accepted, state, path, manager_lock)
                code = run_child(root, wave, family, release, state, path)
                state["exit_code"] = code
                if type(code) is not int or code != 0:
                    raise RuntimeError("Attribution full-budget training failed")
                state["completed_run"] = run_history.completed_run(root, wave, family, release)
                state["status"] = "complete"
            finally:
                if state.get("keeper_release_attempted"):
                    if (
                        state.get("keeper_release_complete") is not True
                        or "cleanup_error" in state
                        or marker_processes(CHILD_MARKER, state.get("child_marker", ""))
                    ):
                        state["keeper_restore"] = "refused: incomplete release or unverified descendants"
                        raise RuntimeError("Attribution cleanup does not permit keeper restoration")
                    restore_accepted_keeper(manager_lock)
                    keeper = keeper_observation()
                    if keeper["devices"] != accepted["devices"]:
                        raise RuntimeError("Attribution restored keeper occupies different physical GPUs")
                    state.update(restored_keeper=keeper, keeper_restore=0)
                if signals:
                    state["deferred_signals"] = signals
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(path, state)
    return state


def start_worker(root: Path, wave: str, family: str) -> dict:
    release = run_history.binding(root, wave)
    if local_node() != node_for(wave, family):
        raise RuntimeError("Attribution worker starter is on a different node")
    path = worker_path(root, wave, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    intent_path = path.with_suffix(".start.json")
    with path.with_suffix(".start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() or intent_path.exists():
            raise FileExistsError("Attribution worker already has evidence; no retry")
        launch = read(paths(root, wave)[2])
        if launch["status"] != "launching" or launch["binding_sha256"] != sha256(paths(root, wave)[0]):
            raise RuntimeError("Attribution starter has no original launch generation")
        command = worker_command(root, wave, family)
        intent = {
            "status": "starting",
            "family": family,
            "arm": family,
            "node": node_for(wave, family),
            "command": command,
            "release_binding_sha256": sha256(paths(root, wave)[0]),
            "worker_marker": launch["worker_markers"][family],
            "starter": observed(os.getpid()),
        }
        write(intent_path, intent)
        try:
            with path.with_suffix(".log").open("x") as stream:
                child = subprocess.Popen(
                    command,
                    cwd=release["candidate"],
                    env=os.environ | ENVIRONMENT | {WORKER_MARKER: intent["worker_marker"]},
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            process = acknowledge(child, command, Path(release["candidate"]), {WORKER_MARKER: intent["worker_marker"]})
            intent.update(
                status="acknowledged",
                pid=child.pid,
                start_ticks=process["start_ticks"],
                generation=process,
            )
            write(intent_path, intent)
        except BaseException as error:
            intent.update(status="unacknowledged", error=f"{type(error).__name__}: {error}")
            write(intent_path, intent)
            raise
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if path.exists():
                state = read(path)
                if state["worker_pid"] != child.pid or state["worker_start_ticks"] != process["start_ticks"]:
                    raise RuntimeError("Attribution worker changed before state acknowledgment")
                return state
            if child.poll() is not None:
                raise RuntimeError("Attribution worker exited before state acknowledgment")
            time.sleep(0.1)
        raise TimeoutError("Attribution worker state acknowledgment is unknown")


def probe(root: Path, wave: str, family: str) -> dict:
    release = run_history.binding(root, wave)
    path = worker_path(root, wave, family)
    if not path.exists():
        return {"status": "preflight", "family": family, "arm": family, "node": node_for(wave, family)}
    state = read(path)
    result = {"status": state["status"], "family": family, "arm": family, "node": state["node"], "receipt": state}
    if state["release_binding_sha256"] != sha256(paths(root, wave)[0]):
        raise ValueError("Attribution worker state belongs to a different release")
    if state["status"] in ("complete", "needs_attention"):
        require_absent(state["worker_pid"])
        if "child_pid" in state:
            require_absent(state["child_pid"])
        if state.get("child_marker") and marker_processes(CHILD_MARKER, state["child_marker"]):
            raise RuntimeError("Attribution terminal worker retains marked training descendants")
        result.update(original_worker_absent=True, original_child_absent=True)
        keeper = keeper_observation()
        if state.get("keeper_release_attempted"):
            if state.get("keeper_restore") != 0 or not same_keeper(keeper, state["restored_keeper"]):
                raise RuntimeError("Attribution restored keeper differs from terminal evidence")
        else:
            accepted = release["allocation_by_node"][state["node"]]["keeper"]
            if not same_keeper(keeper, accepted):
                raise RuntimeError("Attribution untouched keeper changed after preflight failure")
        marked = marker_processes(WORKER_MARKER, state["worker_marker"])
        generations = {int(pid): item for pid, item in keeper["generation"].items()}
        if any(pid not in generations or not same_generation(item, generations[pid]) for pid, item in marked.items()):
            raise RuntimeError("Attribution predecessor markers exist outside exact restored keeper")
        result["keeper"] = keeper
    else:
        worker = require_process(
            state["worker_pid"],
            worker_command(root, wave, family),
            Path(release["candidate"]),
            {WORKER_MARKER: state["worker_marker"]},
        )
        if worker["start_ticks"] != state["worker_start_ticks"]:
            raise RuntimeError("Attribution running worker changed generation")
        if state["status"] == "running":
            child = require_process(
                state["child_pid"],
                state["command"],
                Path(release["candidate"]),
                {WORKER_MARKER: state["worker_marker"], CHILD_MARKER: state["child_marker"]},
            )
            if child["start_ticks"] != state["child_start_ticks"] or child["parent"] != worker["pid"]:
                raise RuntimeError("Attribution training launcher changed generation")
    if Path(state["out"]).exists():
        result["history"] = run_history.measured_run(root, wave, family, release)
    return result


def snapshot(root: Path, wave: str) -> dict:
    release = run_history.binding(root, wave)
    workers = [remote(root, wave, family, Path(release["helpers"]), "probe") for family in FAMILIES]
    terminal = all(item["status"] in ("complete", "needs_attention") for item in workers)
    status = "complete" if all(item["status"] == "complete" for item in workers) else "running"
    if any(item["status"] == "needs_attention" for item in workers):
        status = "needs_attention"
    return {
        "status": status,
        "wave": wave,
        "all_workers_terminal": terminal,
        "binding_sha256": sha256(paths(root, wave)[0]),
        "workers": workers,
    }


def launch(root: Path, wave: str) -> dict:
    release = run_history.binding(root, wave)
    path = paths(root, wave)[2]
    directory = root / f"run-preparation/monitor-{wave}"
    with (root / f"run-preparation/.{wave}-launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(item.exists() for item in (path, directory, root / f"runs/{wave}")):
            raise FileExistsError("Attribution launch already has intent or evidence")
        for family in FAMILIES:
            current = remote(root, wave, family, Path(release["helpers"]), "allocation")
            if not same_keeper(current["keeper"], release["allocation_by_node"][node_for(wave, family)]["keeper"]):
                raise ValueError("Attribution accepted initial keeper changed before launch")
        (root / f"runs/{wave}").mkdir(parents=True)
        directory.mkdir()
        state = {
            "status": "launching",
            "wave": wave,
            "binding_sha256": sha256(paths(root, wave)[0]),
            "workers": {},
            "worker_markers": {family: uuid4().hex for family in FAMILIES},
        }
        write(path, state)
        try:
            command = ["bash", str(Path(release["helpers"]) / "monitor_fused_attribution_run.sh"), str(root), wave]
            with (directory / "monitor.log").open("x") as log:
                monitor = subprocess.Popen(
                    command,
                    cwd=Path(release["helpers"]),
                    env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": release["helpers"]},
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            deadline = time.monotonic() + ACK_TIMEOUT
            while time.monotonic() < deadline:
                own = observed(monitor.pid)
                if (directory / "pid").exists():
                    if own is None or own["command"] != command or int((directory / "pid").read_text()) != monitor.pid:
                        raise RuntimeError("Attribution monitor acknowledgment differs")
                    state.update(monitor_pid=monitor.pid, monitor_start_ticks=own["start_ticks"])
                    write(path, state)
                    break
                if monitor.poll() is not None:
                    raise RuntimeError("Attribution monitor exited before acknowledgment")
                time.sleep(0.1)
            else:
                raise TimeoutError("Attribution monitor startup remains unknown")
            for family in FAMILIES:
                acknowledgment = remote(root, wave, family, Path(release["helpers"]), "start")
                state["workers"][family] = acknowledgment
                write(path, state)
                if (
                    acknowledgment["family"] != family
                    or acknowledgment["node"] != node_for(wave, family)
                    or acknowledgment["worker_command"] != worker_command(root, wave, family)
                    or acknowledgment["worker_marker"] != state["worker_markers"][family]
                    or acknowledgment["release_binding_sha256"] != state["binding_sha256"]
                    or acknowledgment["status"] not in ("preflight", "starting", "running", "complete")
                ):
                    raise RuntimeError("Attribution worker acknowledgment differs")
            state["status"] = "launched"
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write(path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wave", choices=WAVES, default="wave13")
    parser.add_argument(
        "--phase",
        choices=(
            "cpu",
            "freeze",
            "check",
            "allocation",
            "launch",
            "start",
            "worker",
            "probe",
            "snapshot",
            "completion",
        ),
        required=True,
    )
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--controller-cpu-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--family", choices=FAMILIES)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "cpu":
        if args.cpu_receipt is None or args.out is None:
            parser.error("cpu requires --cpu-receipt and --out")
        result = controller_cpu(root, args.cpu_receipt.resolve(), args.out.resolve())
    elif args.phase == "freeze":
        if args.cpu_receipt is None or args.controller_cpu_receipt is None:
            parser.error("freeze requires candidate and controller CPU receipts")
        result = freeze(root, args.wave, args.cpu_receipt.resolve(), args.controller_cpu_receipt.resolve())
    elif args.phase == "check":
        result = run_history.binding(root, args.wave)
    elif args.phase in ("allocation", "start", "worker", "probe"):
        if args.family is None:
            parser.error("worker phases require --family")
        functions = {
            "allocation": allocation_observation,
            "start": start_worker,
            "worker": run_worker,
            "probe": probe,
        }
        result = functions[args.phase](root, args.wave, args.family)
    elif args.phase == "launch":
        result = launch(root, args.wave)
    elif args.phase == "completion":
        result = run_history.require_completed_wave(root, args.wave)
    else:
        result = snapshot(root, args.wave)
        if args.out is not None:
            write(args.out, result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
