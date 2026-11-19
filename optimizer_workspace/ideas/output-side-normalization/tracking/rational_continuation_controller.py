# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serialize the full-model rational-start diagnostic and exact restoration."""

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import owned_output_handoff as handoff
import rational_continuation_contract as contract
import rational_continuation_predecessor as predecessor
import rational_continuation_validation as validation
import torch
from fused_attribution_validation import require_keeper_shape
from fused_attribution_validation_controller import (
    acknowledged_process,
    finish_marked_child,
    process_context,
    require_acknowledged_controller,
    require_helper_bytes,
)
from prepare_rational_continuation import inventory, require_cpu, require_preparation
from prepare_selected_lr_validation import require_archive
from selected_lr_failure_handoff import keeper_guard, release_accepted_keeper, restore_accepted_keeper, same_keeper
from selected_lr_release_owner import candidate_files, read, sha256
from watch_wave5 import write
from wave7_release import idle_training, local_node, node_guard

TIMEOUT = 12 * validation.COMMAND_TIMEOUT + 14400


def freeze(root: Path, cpu_receipt: Path) -> dict:
    cpu = require_cpu(root, cpu_receipt)
    candidate = Path(cpu["candidate"])
    prepared = require_preparation(root, candidate)
    binding, helpers, state, work, result = contract.locations(root)
    archive = root / "run-preparation/source-rational-continuation-20261118.tar"
    if any(path.exists() for path in (binding, helpers.parent, state, work, result, archive)):
        raise FileExistsError("Rational continuation preparation already has one-shot evidence")
    with keeper_guard():
        predecessor.live(root, prepared["allocation_predecessor"])
    helpers.mkdir(parents=True)
    for name in cpu["helpers_sha256"]:
        shutil.copy2(Path(cpu["helpers"]) / name, helpers / name)
    with tarfile.open(archive, "x") as bundle:
        for name in inventory(prepared):
            source = helpers / Path(name).name if name.startswith("tracking/") else candidate / name
            bundle.add(source, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    require_archive(archive, inventory(prepared), alias=True)
    frozen = {
        "status": contract.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "node": contract.NODE,
        "environment": contract.ENVIRONMENT,
        "helpers": str(helpers),
        "helpers_sha256": cpu["helpers_sha256"],
        "files_sha256": cpu["files_sha256"],
        "preparation_sha256": cpu["preparation_sha256"],
        "cpu_receipt": str(cpu_receipt),
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "data_receipt_sha256": prepared["data_receipt_sha256"],
        "allocation_predecessor": prepared["allocation_predecessor"],
        "plan": contract.plan(candidate, root),
        "commands": contract.plan(candidate, root)["commands"],
        "controller_command": contract.controller_command(root),
        "validator_command": contract.validator_command(root, candidate),
        "validation_timeout": TIMEOUT,
        "scope": "Diagnostic-only original-context start intervention; no production release.",
    }
    if require_preparation(root, candidate) != prepared:
        raise ValueError("Rational continuation source changed during freeze")
    write(binding, frozen)
    return frozen


def binding(root: Path) -> dict:
    path, helpers, _, _, _ = contract.locations(root)
    raw, frozen = path.read_bytes(), read(path)
    cpu = require_cpu(root, Path(frozen["cpu_receipt"]))
    candidate = Path(frozen["candidate"])
    prepared = require_preparation(root, candidate)
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    if (
        frozen["status"] != contract.FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["node"] != contract.NODE
        or frozen["candidate"] != cpu["candidate"]
        or frozen["helpers"] != str(helpers)
        or frozen["environment"] != contract.ENVIRONMENT
        or frozen["helpers_sha256"] != cpu["helpers_sha256"]
        or frozen["files_sha256"] != candidate_files(candidate)
        or frozen["preparation_sha256"] != sha256(candidate / contract.PREPARATION)
        or frozen["cpu_receipt_sha256"] != sha256(Path(frozen["cpu_receipt"]))
        or frozen["allocation_predecessor"] != predecessor.evidence(root)
        or frozen["plan"] != contract.plan(candidate, root)
        or frozen["commands"] != frozen["plan"]["commands"]
        or frozen["controller_command"] != contract.controller_command(root)
        or frozen["validator_command"] != contract.validator_command(root, candidate)
        or frozen["validation_timeout"] != TIMEOUT
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
    ):
        raise ValueError("Rational continuation frozen source/command/predecessor differs")
    require_archive(Path(frozen["archive"]), inventory(prepared), alias=True)
    for name in frozen["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Rational continuation imported mutable helpers: {name}")
    if path.read_bytes() != raw or Path(__file__).resolve() != helpers / Path(__file__).name:
        raise ValueError("Rational continuation binding or issued entrypoint differs")
    return frozen


def start(root: Path) -> dict:
    frozen = binding(root)
    path, helpers, state, work, result = contract.locations(root)
    intent_path, log = helpers.parent / "controller.json", helpers.parent / "controller.log"
    with (helpers.parent / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if local_node() != contract.NODE or any(target.exists() for target in (intent_path, log, state, work, result)):
            raise RuntimeError("Rational continuation allocation differs or execution already exists")
        with keeper_guard():
            idle_training()
            predecessor.live(root, frozen["allocation_predecessor"])
        intent = {
            "status": "starting",
            "binding_sha256": sha256(path),
            "command": contract.controller_command(root),
            "marker": uuid4().hex,
        }
        write(intent_path, intent)
        child = None
        try:
            with log.open("x") as stream:
                child = subprocess.Popen(
                    intent["command"],
                    cwd=helpers,
                    env=os.environ
                    | contract.ENVIRONMENT
                    | {contract.CONTROLLER_MARKER: intent["marker"], "PYTHONPATH": str(helpers)},
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            own = acknowledged_process(
                child, intent["command"], helpers, contract.CONTROLLER_MARKER, intent["marker"], contract.ENVIRONMENT
            )
        except BaseException as failure:
            intent.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
            try:
                if child is not None:
                    finish_marked_child(child, intent["marker"], {}, contract.CONTROLLER_MARKER)
                    intent.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
            except BaseException as cleanup_failure:
                intent["cleanup_error"] = f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                raise
            finally:
                intent["log_sha256"] = sha256(log) if log.exists() else None
                write(intent_path, intent)
            raise
        intent.update(status="acknowledged", pid=own["pid"], start_ticks=own["start_ticks"], generation=own)
        write(intent_path, intent)
    return intent


def bounded_validator(root: Path, frozen: dict, state: dict) -> int:
    _, helpers, state_path, _, _ = contract.locations(root)
    candidate = Path(frozen["candidate"])
    command, marker, generations = contract.validator_command(root, candidate), uuid4().hex, {}
    log = helpers.parent / "validator.log"
    state.update(status="validating", spawn_status="starting", command=command, child_marker=marker, log=str(log))
    write(state_path, state)
    child = None
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(
                command,
                cwd=candidate,
                env=os.environ
                | contract.ENVIRONMENT
                | {contract.CHILD_MARKER: marker, "PYTHONPATH": f"{candidate}:{helpers}"},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            own = acknowledged_process(child, command, candidate, contract.CHILD_MARKER, marker, contract.ENVIRONMENT)
            if own["parent"] != state["controller_pid"]:
                raise RuntimeError("Rational continuation validator is not the direct original child")
            state.update(
                spawn_status="acknowledged",
                child_pid=own["pid"],
                child_start_ticks=own["start_ticks"],
                child_generation=own,
            )
            write(state_path, state)
            return child.wait(timeout=TIMEOUT)
    finally:
        try:
            if child is not None:
                finish_marked_child(child, marker, generations, contract.CHILD_MARKER)
                state.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
        except BaseException as failure:
            state["cleanup_error"] = f"{type(failure).__name__}: {failure}"
            raise
        finally:
            state.update(cleaned_generations=generations, log_sha256=sha256(log) if log.exists() else None)
            write(state_path, state)


def execute(root: Path) -> dict:
    frozen = binding(root)
    path, helpers, state_path, _, result_path = contract.locations(root)
    intent_path = helpers.parent / "controller.json"
    intent, deadline = read(intent_path), time.monotonic() + 30
    while intent["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.01)
        intent = read(intent_path)
    own = process_context(
        os.getpid(),
        contract.controller_command(root),
        helpers,
        contract.CONTROLLER_MARKER,
        os.environ.get(contract.CONTROLLER_MARKER, ""),
        contract.ENVIRONMENT,
    )
    require_acknowledged_controller(intent, own)
    if state_path.exists() or local_node() != contract.NODE or intent["binding_sha256"] != sha256(path):
        raise RuntimeError("Rational continuation controller lacks exact original acknowledgment")
    state = {
        "status": "preflight",
        "binding_sha256": sha256(path),
        "controller_pid": own["pid"],
        "controller_start_ticks": own["start_ticks"],
        "allocation_predecessor": frozen["allocation_predecessor"],
    }
    write(state_path, state)
    signals = []
    try:
        with node_guard(root, contract.NODE) as signals, keeper_guard() as manager:
            idle_training()
            accepted = predecessor.live(root, frozen["allocation_predecessor"])
            subprocess.run(
                [contract.PYTHON, "-m", "experiment.bootstrap", "--check-only"],
                cwd=Path(frozen["candidate"]),
                env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                check=True,
                timeout=240,
            )
            try:
                release_accepted_keeper(accepted, state, state_path, manager)
                code = bounded_validator(root, frozen, state)
                state["exit_code"] = code
                if type(code) is not int or code != 0:
                    raise RuntimeError("Rational continuation validator failed")
                result = validation.reconstruct(root, Path(frozen["candidate"]))
                state.update(
                    status="diagnostic_complete",
                    result_sha256=sha256(result_path),
                    intervention_accepted=result["analysis"]["diagnostic_intervention_accepted"],
                )
            finally:
                if state.get("keeper_release_attempted"):
                    if (
                        state.get("keeper_release_complete") is not True
                        or "cleanup_error" in state
                        or (
                            state.get("child_marker")
                            and handoff.marker_processes(contract.CHILD_MARKER, state["child_marker"])
                        )
                    ):
                        state["keeper_restore"] = "refused: incomplete transfer or live descendants"
                        raise RuntimeError("Rational continuation live descendants prevent restoration")
                    restore_accepted_keeper(manager)
                    restored = handoff.keeper_observation()
                    require_keeper_shape(restored)
                    if restored["devices"] != accepted["devices"]:
                        raise RuntimeError("Rational continuation restored physical devices differ")
                    state.update(keeper_restore=0, restored_keeper=restored)
    except BaseException as failure:
        state.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        raise
    finally:
        if signals:
            state["deferred_signals"] = signals
        write(state_path, state)
    return state


def terminal(root: Path, out: Path) -> dict:
    frozen = binding(root)
    path, helpers, state_path, _, result_path = contract.locations(root)
    state, intent = read(state_path), read(helpers.parent / "controller.json")
    controller, child = intent["generation"], state["child_generation"]
    digest = sha256(path)
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or out.exists()
        or out != out.resolve()
        or not out.is_relative_to(root / "run-preparation")
        or state["status"] != "diagnostic_complete"
        or any(
            type(state[key]) is not int or state[key] != 0
            for key in ("keeper_restore", "exit_code", "cleanup_exit_code")
        )
        or state["keeper_release_attempted"] is not True
        or state["keeper_release_complete"] is not True
        or state["marked_descendants_absent"] is not True
        or state["spawn_status"] != "acknowledged"
        or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
        or intent["status"] != "acknowledged"
        or intent["binding_sha256"] != digest
        or intent["command"] != contract.controller_command(root)
        or intent["pid"] != state["controller_pid"]
        or intent["start_ticks"] != state["controller_start_ticks"]
        or controller["pid"] != intent["pid"]
        or controller["start_ticks"] != intent["start_ticks"]
        or controller["command"] != intent["command"]
        or controller["group"] != controller["pid"]
        or controller["session"] != controller["pid"]
        or child["pid"] != state["child_pid"]
        or child["start_ticks"] != state["child_start_ticks"]
        or child["parent"] != state["controller_pid"]
        or child["command"] != contract.validator_command(root, Path(frozen["candidate"]))
        or child["group"] != child["pid"]
        or child["session"] != child["pid"]
        or state["command"] != child["command"]
        or state["allocation_predecessor"] != frozen["allocation_predecessor"]
        or state["log"] != str(helpers.parent / "validator.log")
        or state["log_sha256"] != sha256(Path(state["log"]))
        or local_node() != contract.NODE
        or any(handoff.observed(state[key]) is not None for key in ("controller_pid", "child_pid"))
        or handoff.marker_processes(contract.CHILD_MARKER, state["child_marker"])
        or state["binding_sha256"] != sha256(path)
        or state["result_sha256"] != sha256(result_path)
    ):
        raise RuntimeError("Rational continuation terminal owner/cleanup/restoration is incomplete")
    paths = (
        path,
        state_path,
        helpers.parent / "controller.json",
        helpers.parent / "controller.log",
        helpers.parent / "validator.log",
        result_path,
    )
    evidence = {str(item.relative_to(root)): sha256(item) for item in paths}
    result = validation.reconstruct(root, Path(frozen["candidate"]))
    initialized, completed, restored = (datetime.fromisoformat(item["time"]) for item in (intent, result, state))
    if (
        any(value.tzinfo is None for value in (initialized, completed, restored))
        or not initialized <= completed <= restored
        or state["intervention_accepted"] is not result["analysis"]["diagnostic_intervention_accepted"]
    ):
        raise ValueError("Rational continuation terminal chronology or intervention result differs")
    with keeper_guard():
        keeper = handoff.keeper_observation()
        require_keeper_shape(keeper)
        if not same_keeper(keeper, state["restored_keeper"]) or not set(
            handoff.marker_processes(contract.CONTROLLER_MARKER, intent["marker"])
        ).issubset({int(pid) for pid in keeper["generation"]}):
            raise RuntimeError("Rational continuation terminal exact keeper differs")
    if any(sha256(root / name) != value for name, value in evidence.items()):
        raise ValueError("Rational continuation terminal evidence changed during reconstruction")
    receipt = {
        "status": "rational_continuation_terminal_independently_reconstructed_exact_restoration",
        "binding_sha256": sha256(path),
        "result_sha256": sha256(result_path),
        "terminal_time": state["time"],
        "controller_pid_absent": state["controller_pid"],
        "validator_pid_absent": state["child_pid"],
        "keeper": keeper,
        "analysis": result["analysis"],
        "evidence_sha256": evidence,
        "verifier_sha256": sha256(Path(__file__)),
        "scope": frozen["scope"],
    }
    write(out, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("freeze", "check", "start", "execute", "terminal"))
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    torch.set_num_threads(8)
    if args.phase == "freeze":
        if args.cpu_receipt is None:
            parser.error("Freeze requires --cpu-receipt")
        result = freeze(root, args.cpu_receipt.resolve(strict=True))
    elif args.phase == "check":
        result = binding(root)
    elif args.phase == "start":
        result = start(root)
    elif args.phase == "terminal":
        if args.out is None:
            parser.error("Terminal requires --out")
        result = terminal(root, args.out.resolve())
    else:
        result = execute(root)
    print(result["status"])


if __name__ == "__main__":
    main()
