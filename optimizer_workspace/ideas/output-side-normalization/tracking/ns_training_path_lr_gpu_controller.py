# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue separately accepted actual-LR training-path diagnostics with exact restoration."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import ns_training_path_lr_contract as contract
import ns_training_path_lr_gpu_owner as owner
import ns_training_path_lr_gpu_source as source
import ns_training_path_lr_validation as validation
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

TERMINAL_STATUS = "ns_training_path_actual_lr_terminal_independently_reconstructed_exact_restoration_not_production"


def freeze(root: Path, cpu_receipt: Path, seal_sha256: str) -> dict:
    owner.require_hidden_cuda()
    cpu = source.require_cpu(root, cpu_receipt)
    candidate = Path(cpu["candidate"])
    prepared = owner.require_accepted_adapter(root, candidate)
    binding, helpers, state, work, result = contract.locations(root)
    archive = root / owner.ARCHIVE
    if any(path.exists() or path.is_symlink() for path in (binding, helpers.parent, state, work, result, archive)):
        raise FileExistsError("NS training-path freeze has existing one-shot evidence")
    previous = owner.require_sealed_predecessor(root, seal_sha256)
    accepted_cpu = source.accepted_archive(
        root,
        (
            owner.PREDECESSOR_RECEIPT,
            owner.PREDECESSOR_RECEIPT_SHA256,
            "ns_training_path_actual_lr_predecessor_cpu_archive_verified_owner_unissued",
            12,
        ),
    )
    with keeper_guard():
        owner.require_predecessor_keeper(root, previous, seal_sha256)
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
    paths = {
        str(cpu_receipt): sha256(cpu_receipt),
        str(archive): sha256(archive),
        str(candidate / source.PREPARATION): sha256(candidate / source.PREPARATION),
        str(root / "data/experiment_receipt.json"): prepared["data_receipt_sha256"],
        str(root / owner.PREDECESSOR_RECEIPT): owner.PREDECESSOR_RECEIPT_SHA256,
        str(root / owner.predecessor.SEAL): seal_sha256,
        str(root / source.PARENT): source.PARENT_SHA256,
    }
    paths |= {str(candidate / name): digest for name, digest in prepared["files_sha256"].items()}
    paths |= {str(helpers / name): digest for name, digest in cpu["helpers_sha256"].items()}
    paths |= {str(cpu_receipt.parent / name): digest for name, digest in expected.items()}
    paths |= {
        str(Path(accepted_cpu["helpers"]).parent / name): digest
        for name, digest in accepted_cpu["inventory_sha256"].items()
    }
    for receipt in (cpu, accepted_cpu):
        for name in ("archive", "tests", "log"):
            paths[receipt[name]] = receipt[f"{name}_sha256"]
    adapter_cpu = read(root / source.PARENT)
    for name in ("archive", "tests", "log"):
        paths[adapter_cpu[name]] = adapter_cpu[f"{name}_sha256"]
    paths |= {
        str(Path(adapter_cpu["helpers"]).parent / name): digest
        for name, digest in adapter_cpu["inventory_sha256"].items()
    }
    authenticated = owner.authenticated_inventory(paths)
    # The accepted predecessor seal already authenticates payload hashes; retain its inode
    # evidence without reopening the original tensor payloads.
    for group in ("evidence", "checkpoint_payloads"):
        for name, item in previous[group].items():
            path = str(root / name)
            if path in authenticated and authenticated[path] != item:
                raise ValueError("NS training-path predecessor inventory conflicts with accepted source")
            authenticated[path] = item
    frozen = {
        "status": contract.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "node": owner.NODE,
        "environment": contract.ENVIRONMENT,
        "controller_environment": owner.CONTROLLER_ENVIRONMENT,
        "helpers": str(helpers),
        "helpers_sha256": cpu["helpers_sha256"],
        "files_sha256": prepared["files_sha256"],
        "adapter_cpu_receipt": str(root / source.PARENT),
        "adapter_cpu_receipt_sha256": source.PARENT_SHA256,
        "allowed_transform": [],
        "preparation_sha256": sha256(candidate / source.PREPARATION),
        "cpu_receipt": str(cpu_receipt),
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "data_receipt_sha256": prepared["data_receipt_sha256"],
        "allocation_predecessor": previous,
        "predecessor_acceptance": owner.predecessor_binding(root, seal_sha256),
        "authenticated_files": authenticated,
        "devices": previous["restored_keeper"]["devices"],
        "plan": validation.plan(root, candidate),
        "commands": contract.commands(candidate, root, work),
        "controller_command": owner.controller_command(root),
        "validator_command": owner.validator_command(root, candidate),
        "validation_timeout": owner.TIMEOUT,
        "scope": source.SCOPE,
    }
    if (
        source.require_preparation(root, candidate) != prepared
        or owner.require_sealed_predecessor(root, seal_sha256) != previous
    ):
        raise RuntimeError("NS training-path preparation or sealed predecessor changed during freeze")
    owner.require_operation_inventory(authenticated)
    write(binding, frozen)
    return frozen


def start(root: Path) -> dict:
    owner.require_hidden_cuda()
    frozen = owner.require_binding(root)
    binding, helpers, state, work, result = contract.locations(root)
    intent_path, log = helpers.parent / "controller.json", helpers.parent / "controller.log"
    with (helpers.parent / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if local_node() != owner.NODE or any(path.exists() for path in (intent_path, log, state, work, result)):
            raise RuntimeError("NS training-path allocation differs or launch already has evidence")
        with keeper_guard():
            idle_training()
            owner.require_predecessor_keeper(
                root, frozen["allocation_predecessor"], frozen["predecessor_acceptance"]["seal_sha256"]
            )
        if owner.require_binding(root) != frozen:
            raise RuntimeError("NS training-path independently accepted CPU binding changed before start")
        proof = owner.parent_proof(binding.read_bytes(), frozen)
        intent = {
            "status": "starting",
            "binding_sha256": sha256(binding),
            "command": owner.controller_command(root),
            "marker": uuid4().hex,
            "parent_proof": proof,
        }
        write(intent_path, intent)
        child = None
        try:
            with log.open("x") as stream:
                child = subprocess.Popen(
                    intent["command"],
                    cwd=helpers,
                    env=os.environ
                    | owner.CONTROLLER_ENVIRONMENT
                    | {
                        contract.CONTROLLER_MARKER: intent["marker"],
                        "PYTHONPATH": str(helpers),
                        owner.PARENT_PROOF_MARKER: proof,
                    },
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            own = acknowledged_process(
                child,
                intent["command"],
                helpers,
                contract.CONTROLLER_MARKER,
                intent["marker"],
                owner.CONTROLLER_ENVIRONMENT | {owner.PARENT_PROOF_MARKER: proof},
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
                | {
                    contract.CHILD_MARKER: marker,
                    "PYTHONPATH": f"{candidate}:{helpers}",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            own = acknowledged_process(child, command, candidate, contract.CHILD_MARKER, marker, contract.ENVIRONMENT)
            if own["parent"] != state["controller_pid"]:
                raise RuntimeError("NS training-path validator is not the original controller's direct child")
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
    owner.require_hidden_cuda()
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
        owner.CONTROLLER_ENVIRONMENT
        | {
            owner.PARENT_PROOF_MARKER: owner.parent_proof(binding.read_bytes(), frozen),
        },
    )
    require_acknowledged_controller(intent, own)
    if state_path.exists() or local_node() != owner.NODE or intent["binding_sha256"] != sha256(binding):
        raise RuntimeError("NS training-path execution lacks original acknowledgment or allocation")
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
            accepted = owner.require_predecessor_keeper(
                root, frozen["allocation_predecessor"], frozen["predecessor_acceptance"]["seal_sha256"]
            )
            subprocess.run(
                [contract.PYTHON, "-m", "experiment.bootstrap", "--check-only"],
                cwd=Path(frozen["candidate"]),
                env=os.environ | owner.CONTROLLER_ENVIRONMENT,
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
                    raise RuntimeError("NS training-path bounded validator failed")
                result = validation.reconstruct(root, Path(frozen["candidate"]))
                state.update(
                    status="validation_complete",
                    result_sha256=sha256(result_path),
                    **{key: result[key] for key in ("accepted_cases", "rejected_cases", "all_cases_accepted")},
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
                        raise RuntimeError("NS training-path incomplete cleanup prevents restoration")
                    restore_accepted_keeper(manager)
                    restored = handoff.keeper_observation()
                    require_keeper_shape(restored)
                    if restored["devices"] != accepted["devices"]:
                        raise RuntimeError("NS training-path restored physical devices differ")
                    state.update(keeper_restore=0, restored_keeper=restored)
    except BaseException as failure:
        state.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        raise
    finally:
        if signals:
            state["deferred_signals"] = signals
        write(state_path, state)
    return state


def terminal_paths(root: Path, intent: dict) -> tuple[Path, Path, Path, Path]:
    """Bind the only terminal output to the original acknowledged controller generation."""
    generation = {
        "binding_sha256": intent["binding_sha256"],
        "controller_pid": intent["pid"],
        "controller_start_ticks": intent["start_ticks"],
        "controller_marker": intent["marker"],
    }
    token = hashlib.sha256(json.dumps(generation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    directory = contract.locations(root)[1].parent
    return (
        root / "run-preparation" / f"ns-training-path-actual-lr-terminal-{token}.json",
        directory / "terminal-intent.json",
        directory / "terminal-outcome.json",
        directory / "terminal.lock",
    )


def publish_terminal_record(path: Path, value: dict) -> None:
    """Publish complete JSON exclusively; never replace another terminal or attempt."""
    value = value | {"time": datetime.now(UTC).isoformat()}
    with tempfile.TemporaryDirectory(prefix=".ns-training-path-actual-lr-terminal-", dir=path.parent) as directory:
        staged = Path(directory) / "record.json"
        with staged.open("x") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, path)
        staged.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _terminal_receipt(root: Path, frozen: dict) -> dict:
    binding, helpers, state_path, _, result_path = contract.locations(root)
    state, intent = read(state_path), read(helpers.parent / "controller.json")
    child, controller = state["child_generation"], intent["generation"]
    if (
        local_node() != owner.NODE
        or state["status"] != "validation_complete"
        or intent["status"] != "acknowledged"
        or state["binding_sha256"] != sha256(binding)
        or intent["binding_sha256"] != sha256(binding)
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or controller["pid"] != intent["pid"]
        or controller["start_ticks"] != intent["start_ticks"]
        or intent["command"] != owner.controller_command(root)
        or controller["command"] != intent["command"]
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
        raise RuntimeError("NS training-path terminal ownership, cleanup or restoration is incomplete")
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
        raise ValueError("NS training-path terminal decisions differ from independent reconstruction")
    times = [datetime.fromisoformat(value["time"]) for value in (intent, result, state)]
    if any(value.tzinfo is None for value in times) or times != sorted(times):
        raise ValueError("NS training-path terminal chronology differs")
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
            raise RuntimeError("NS training-path exact terminal keeper restoration differs")
    if any(sha256(root / name) != digest for name, digest in evidence.items()):
        raise ValueError("NS training-path terminal canonical evidence changed")
    receipt = {
        "status": TERMINAL_STATUS,
        "binding_sha256": sha256(binding),
        "result_sha256": sha256(result_path),
        "terminal_time": state["time"],
        "controller_pid_absent": state["controller_pid"],
        "validator_pid_absent": state["child_pid"],
        "keeper": keeper,
        **{name: result[name] for name in ("accepted_cases", "rejected_cases", "all_cases_accepted")},
        "evidence_sha256": evidence,
        "verifier_sha256": sha256(Path(__file__)),
        "predecessor_acceptance": frozen["predecessor_acceptance"],
        "production_authorized": False,
        "scope": source.SCOPE,
    }
    return receipt


def terminal(root: Path, out: Path) -> dict:
    owner.require_hidden_cuda()
    frozen = owner.require_binding(root)
    _, helpers, _, _, _ = contract.locations(root)
    controller_intent = helpers.parent / "controller.json"
    original = controller_intent.read_bytes()
    intent = json.loads(original)
    canonical, attempt_path, outcome_path, lock_path = terminal_paths(root, intent)
    if out != canonical or out != out.resolve() or intent["status"] != "acknowledged":
        raise RuntimeError("NS training-path terminal output is not its original canonical generation")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = os.fstat(lock.fileno())
        if (
            lock_path != lock_path.resolve(strict=True)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != (lock_path.stat().st_dev, lock_path.stat().st_ino)
            or controller_intent.read_bytes() != original
        ):
            raise RuntimeError("NS training-path terminal lock or original generation changed")
        if any(path.exists() or path.is_symlink() for path in (canonical, attempt_path, outcome_path)):
            raise RuntimeError("NS training-path terminal reconstruction already has one-shot evidence")
        attempt = {
            "status": "reconstruction_claimed",
            "binding_sha256": intent["binding_sha256"],
            "controller_pid": intent["pid"],
            "controller_start_ticks": intent["start_ticks"],
            "controller_marker": intent["marker"],
            "output": str(canonical),
            "verifier_sha256": sha256(Path(__file__)),
            "reconstructor_pid": os.getpid(),
        }
        publish_terminal_record(attempt_path, attempt)
        outcome = {
            "status": "needs_attention",
            "output": str(canonical),
            "reconstruction_intent_sha256": sha256(attempt_path),
        }
        try:
            receipt = _terminal_receipt(root, frozen)
            if controller_intent.read_bytes() != original:
                raise RuntimeError("NS training-path terminal controller generation changed during reconstruction")
            receipt.update(
                controller_start_ticks=intent["start_ticks"],
                controller_marker=intent["marker"],
                reconstruction_intent_sha256=outcome["reconstruction_intent_sha256"],
            )
            publish_terminal_record(canonical, receipt)
            outcome.update(status="complete", terminal_sha256=sha256(canonical))
            return read(canonical)
        except BaseException as failure:
            outcome["error"] = f"{type(failure).__name__}: {failure}"
            raise
        finally:
            publish_terminal_record(outcome_path, outcome)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("freeze", "check", "start", "execute", "terminal"))
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--seal-sha256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    owner.require_hidden_cuda()
    root = args.root.resolve(strict=True)
    torch.set_num_threads(8)
    if args.phase == "freeze":
        if args.cpu_receipt is None or args.seal_sha256 is None:
            parser.error("Freeze requires --cpu-receipt and original terminal --seal-sha256")
        result = freeze(root, args.cpu_receipt.resolve(strict=True), args.seal_sha256)
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
