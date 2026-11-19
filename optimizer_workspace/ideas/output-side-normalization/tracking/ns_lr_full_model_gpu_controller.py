# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue distinct actual-LR diagnostics with original terminal gates and exact restoration."""

import argparse
import fcntl
import os
import shutil
import subprocess
import tarfile
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import ns_lr_full_model_contract as contract
import ns_lr_full_model_execution_source as source
import ns_lr_full_model_gpu_owner as owner
import ns_lr_full_model_predecessor as predecessor
import ns_lr_full_model_validation as validation
import owned_output_handoff as handoff
import torch
from fused_attribution_validation import require_keeper_shape
from fused_attribution_validation_controller import (
    acknowledged_process,
    finish_marked_child,
    process_context,
    require_acknowledged_controller,
)
from prepare_selected_lr_validation import require_archive
from saved_ns_gpu_adapter import lock_descriptor
from selected_lr_failure_handoff import keeper_guard, release_accepted_keeper, restore_accepted_keeper, same_keeper
from selected_lr_release_owner import read, sha256
from watch_wave5 import write
from wave7_release import idle_training, local_node, node_guard

ARCHIVE = "run-preparation/source-ns-actual-lr-20261119.tar"
TERMINAL_STATUS = "ns_actual_lr_terminal_independently_reconstructed_exact_restoration"


def freeze(root: Path, cpu_receipt: Path) -> dict:
    """Freeze accepted source only after the original lower-LR queue terminal handoff."""
    cpu = source.require_cpu(root, cpu_receipt)
    candidate = Path(cpu["candidate"])
    prepared = read(candidate / source.PREPARATION)
    binding, helpers, state, work, result = contract.locations(root)
    archive = root / ARCHIVE
    if any(path.exists() for path in (binding, helpers.parent, state, work, result, archive)):
        raise FileExistsError("NS actual-LR freeze has existing one-shot evidence")
    previous = predecessor.evidence(root)
    with keeper_guard():
        owner.require_predecessor_keeper(root, previous)
    helpers.mkdir(parents=True)
    for name in cpu["helpers_sha256"]:
        shutil.copy2(Path(cpu["helpers"]) / name, helpers / name)
    expected = cpu["inventory_sha256"]
    with tarfile.open(archive, "x") as bundle:
        for name in expected:
            if name.startswith("tracking/"):
                path = helpers / Path(name).name
            elif (candidate / name).is_file():
                path = candidate / name
            else:
                path = cpu_receipt.parent / name
            bundle.add(path, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    require_archive(archive, expected, alias=True)
    frozen = {
        "status": contract.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "node": owner.NODE,
        "environment": contract.ENVIRONMENT,
        "helpers": str(helpers),
        "helpers_sha256": cpu["helpers_sha256"],
        "files_sha256": prepared["files_sha256"],
        "preparation_sha256": sha256(candidate / source.PREPARATION),
        "cpu_receipt": str(cpu_receipt),
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "data_receipt_sha256": prepared["data_receipt_sha256"],
        "allocation_predecessor": previous,
        "devices": previous["restored_keeper"]["devices"],
        "plan": validation.plan(root, candidate),
        "commands": contract.commands(candidate, root, work),
        "controller_command": owner.controller_command(root),
        "validator_command": owner.validator_command(root, candidate),
        "validation_timeout": owner.TIMEOUT,
        "scope": source.SCOPE,
    }
    if source.require_preparation(root, candidate) != prepared or predecessor.evidence(root) != previous:
        raise RuntimeError("NS actual-LR preparation or predecessor changed during freeze")
    write(binding, frozen)
    return frozen


def start(root: Path) -> dict:
    """Acknowledge exactly one separately sessioned actual-LR controller."""
    frozen = owner.require_binding(root)
    binding, helpers, state, work, result = contract.locations(root)
    intent_path, log = helpers.parent / "controller.json", helpers.parent / "controller.log"
    with (helpers.parent / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if local_node() != owner.NODE or any(path.exists() for path in (intent_path, log, state, work, result)):
            raise RuntimeError("NS actual-LR allocation differs or launch already has evidence")
        with keeper_guard():
            idle_training()
            owner.require_predecessor_keeper(root, frozen["allocation_predecessor"])
        intent = {
            "status": "starting",
            "binding_sha256": sha256(binding),
            "command": owner.controller_command(root),
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
            intent.update(status="acknowledged", pid=own["pid"], start_ticks=own["start_ticks"], generation=own)
        except BaseException as failure:
            intent.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
            if child is not None:
                finish_marked_child(child, intent["marker"], {}, contract.CONTROLLER_MARKER)
                intent.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
            intent["log_sha256"] = sha256(log) if log.is_file() else None
            raise
        finally:
            write(intent_path, intent)
    return intent


def bounded_validator(root: Path, frozen: dict, state: dict) -> int:
    """Retain direct-child acknowledgment and cleanup, including detached descendants."""
    _, helpers, state_path, _, _ = contract.locations(root)
    candidate = Path(frozen["candidate"])
    command, marker, generations = owner.validator_command(root, candidate), uuid4().hex, {}
    log, child = helpers.parent / "validator.log", None
    state.update(status="validating", spawn_status="starting", command=command, child_marker=marker, log=str(log))
    write(state_path, state)
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
                raise RuntimeError("NS actual-LR validator is not the original controller's direct child")
            state.update(
                spawn_status="acknowledged",
                child_pid=own["pid"],
                child_start_ticks=own["start_ticks"],
                child_generation=own,
            )
            write(state_path, state)
            return child.wait(timeout=owner.TIMEOUT)
    finally:
        try:
            if child is not None:
                finish_marked_child(child, marker, generations, contract.CHILD_MARKER)
                state.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
        except BaseException as failure:
            state["cleanup_error"] = f"{type(failure).__name__}: {failure}"
            raise
        finally:
            state.update(cleaned_generations=generations, log_sha256=sha256(log) if log.is_file() else None)
            write(state_path, state)


def execute(root: Path) -> dict:
    """Run bounded diagnostics under both allocation locks and restore on failure."""
    frozen = owner.require_binding(root)
    binding, helpers, state_path, _, result_path = contract.locations(root)
    intent, deadline = read(helpers.parent / "controller.json"), time.monotonic() + 30
    while intent["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.01)
        intent = read(helpers.parent / "controller.json")
    own = process_context(
        os.getpid(),
        owner.controller_command(root),
        helpers,
        contract.CONTROLLER_MARKER,
        os.environ.get(contract.CONTROLLER_MARKER, ""),
        contract.ENVIRONMENT,
    )
    require_acknowledged_controller(intent, own)
    if state_path.exists() or local_node() != owner.NODE or intent["binding_sha256"] != sha256(binding):
        raise RuntimeError("NS actual-LR execution lacks original acknowledgment or allocation")
    state = {
        "status": "preflight",
        "binding_sha256": sha256(binding),
        "controller_pid": own["pid"],
        "controller_start_ticks": own["start_ticks"],
        "allocation_predecessor": frozen["allocation_predecessor"],
    }
    write(state_path, state)
    signals = []
    try:
        with node_guard(root, owner.NODE) as signals, keeper_guard() as manager:
            idle_training()
            accepted = owner.require_predecessor_keeper(root, frozen["allocation_predecessor"])
            subprocess.run(
                [contract.PYTHON, "-m", "experiment.bootstrap", "--check-only"],
                cwd=Path(frozen["candidate"]),
                env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                check=True,
                timeout=240,
            )
            state["lock_descriptors"] = {
                "node": lock_descriptor(root / "run-preparation" / f".wave7-node-{owner.NODE}.lock"),
                "manager": manager,
            }
            try:
                release_accepted_keeper(accepted, state, state_path, manager)
                state["exit_code"] = bounded_validator(root, frozen, state)
                if type(state["exit_code"]) is not int or state["exit_code"] != 0:
                    raise RuntimeError("NS actual-LR bounded validator failed")
                result = validation.reconstruct(root, Path(frozen["candidate"]))
                state.update(
                    status="validation_complete",
                    result_sha256=sha256(result_path),
                    accepted_cases=result["accepted_cases"],
                    rejected_cases=result["rejected_cases"],
                    all_cases_accepted=result["all_cases_accepted"],
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
                        raise RuntimeError("NS actual-LR descendants or incomplete cleanup prevent restoration")
                    restore_accepted_keeper(manager)
                    restored = handoff.keeper_observation()
                    require_keeper_shape(restored)
                    if restored["devices"] != accepted["devices"]:
                        raise RuntimeError("NS actual-LR restored physical devices differ")
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
    """Independently reconstruct retained actual-LR decisions and exact restoration."""
    frozen = owner.require_binding(root)
    binding, helpers, state_path, _, result_path = contract.locations(root)
    state, intent = read(state_path), read(helpers.parent / "controller.json")
    child, controller = state["child_generation"], intent["generation"]
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or out.exists()
        or out != out.resolve()
        or not out.is_relative_to(root / "run-preparation")
        or local_node() != owner.NODE
        or state["status"] != "validation_complete"
        or intent["status"] != "acknowledged"
        or state["binding_sha256"] != sha256(binding)
        or intent["binding_sha256"] != sha256(binding)
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or controller["pid"] != intent["pid"]
        or controller["start_ticks"] != intent["start_ticks"]
        or controller["command"] != owner.controller_command(root)
        or controller["group"] != controller["pid"]
        or controller["session"] != controller["pid"]
        or child["pid"] != state["child_pid"]
        or child["start_ticks"] != state["child_start_ticks"]
        or child["parent"] != state["controller_pid"]
        or child["command"] != owner.validator_command(root, Path(frozen["candidate"]))
        or child["group"] != child["pid"]
        or child["session"] != child["pid"]
        or state["command"] != child["command"]
        or state["spawn_status"] != "acknowledged"
        or state["keeper_release_attempted"] is not True
        or state["keeper_release_complete"] is not True
        or state["marked_descendants_absent"] is not True
        or any(
            type(state[key]) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
        or state["allocation_predecessor"] != frozen["allocation_predecessor"]
        or state["log"] != str(helpers.parent / "validator.log")
        or state["log_sha256"] != sha256(Path(state["log"]))
        or state["result_sha256"] != sha256(result_path)
        or any(handoff.observed(state[key]) is not None for key in ("controller_pid", "child_pid"))
        or handoff.marker_processes(contract.CHILD_MARKER, state["child_marker"])
    ):
        raise RuntimeError("NS actual-LR terminal ownership, cleanup or restoration is incomplete")
    paths = (
        binding,
        state_path,
        helpers.parent / "controller.json",
        helpers.parent / "controller.log",
        helpers.parent / "validator.log",
        result_path,
    )
    evidence = {str(path.relative_to(root)): sha256(path) for path in paths}
    result = validation.reconstruct(root, Path(frozen["candidate"]))
    if any(state[name] != result[name] for name in ("accepted_cases", "rejected_cases", "all_cases_accepted")):
        raise ValueError("NS actual-LR terminal decisions differ from retained reconstruction")
    times = [datetime.fromisoformat(value["time"]) for value in (intent, result, state)]
    if any(value.tzinfo is None for value in times) or times != sorted(times):
        raise ValueError("NS actual-LR terminal chronology differs")
    with keeper_guard():
        keeper = handoff.keeper_observation()
        require_keeper_shape(keeper)
        if (
            not same_keeper(keeper, state["restored_keeper"])
            or keeper["devices"] != frozen["devices"]
            or not set(handoff.marker_processes(contract.CONTROLLER_MARKER, intent["marker"])).issubset(
                {int(pid) for pid in keeper["generation"]}
            )
        ):
            raise RuntimeError("NS actual-LR exact terminal keeper restoration differs")
    if any(sha256(root / name) != digest for name, digest in evidence.items()):
        raise ValueError("NS actual-LR terminal canonical evidence changed")
    receipt = {
        "status": TERMINAL_STATUS,
        "binding_sha256": sha256(binding),
        "result_sha256": sha256(result_path),
        "terminal_time": state["time"],
        "controller_pid_absent": state["controller_pid"],
        "validator_pid_absent": state["child_pid"],
        "keeper": keeper,
        "accepted_cases": result["accepted_cases"],
        "rejected_cases": result["rejected_cases"],
        "all_cases_accepted": result["all_cases_accepted"],
        "evidence_sha256": evidence,
        "verifier_sha256": sha256(Path(__file__)),
        "scope": source.SCOPE,
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
        result = owner.require_binding(root)
    elif args.phase == "start":
        result = start(root)
    elif args.phase == "terminal":
        if args.out is None:
            parser.error("Terminal requires --out")
        result = terminal(root, args.out.resolve())
    else:
        result = execute(root)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
