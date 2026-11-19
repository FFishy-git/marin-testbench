# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive, launch and independently reconstruct a separate two-process start-accuracy trial."""

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

import root_stage_trace as tracing
import root_start_accuracy as accuracy
import root_start_execution as execution
import root_start_gpu_owner as owner
import torch
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


def cpu(root: Path, out: Path) -> dict:
    if out.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise FileExistsError("Start owner CPU acceptance requires hidden CUDA and unused external output")
    sources = owner.helper_sources(root)
    hashes = {name: sha256(path) for name, path in sources.items()}
    parent = read(root / owner.PARENT)
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    for name, source in sources.items():
        shutil.copy2(source, helpers / name)
    for name in parent["fixtures_sha256"]:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2((root / owner.PARENT).parent / name, target)
    log, archive = out / "cpu.log", out / "source.tar"
    with log.open("x") as stream:
        code = subprocess.run(
            owner.cpu_command(out),
            cwd=out,
            env=os.environ
            | owner.ENVIRONMENT
            | {"UV_OFFLINE": "1", "PYTHONPATH": f"{helpers}:{Path(torch.__file__).parents[1]}"},
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=900,
        ).returncode
    inventory = {f"tracking/{name}": value for name, value in hashes.items()} | parent["fixtures_sha256"]
    with tarfile.open(archive, "x") as bundle:
        for name in inventory:
            bundle.add(out / name, arcname=name, recursive=False)
    require_archive(archive, inventory, alias=False)
    unchanged = hashes == {name: sha256(path) for name, path in sources.items()}
    result = {
        "status": owner.CPU_STATUS if type(code) is int and code == 0 and unchanged else "needs_attention",
        "root": str(root),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "fixtures_sha256": parent["fixtures_sha256"],
        "parent_cpu_receipt_sha256": owner.PARENT_SHA256,
        "command": owner.cpu_command(out),
        "exit_code": code,
        "cuda_visible_devices": "",
        "log": str(log),
        "log_sha256": sha256(log),
        "tests": str(out / "tests.xml"),
        "tests_sha256": sha256(out / "tests.xml"),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "scope": "CPU/source acceptance only; no keeper transfer, CUDA execution or production authorization.",
    }
    write(out / "receipt.json", result)
    return owner.require_cpu(root, out / "receipt.json")


def freeze(root: Path, cpu_receipt: Path) -> dict:
    accepted = owner.require_cpu(root, cpu_receipt)
    previous = owner.predecessor(root)
    binding, helpers, state, work = owner.locations(root)
    archive = root / "run-preparation/source-root-start-accuracy-20261118.tar"
    if any(path.exists() for path in (binding, helpers.parent, state, work, archive)):
        raise FileExistsError("Start accuracy identity already has one-shot preparation or execution evidence")
    with keeper_guard():
        keeper = owner.require_live_predecessor(root, previous)
    helpers.mkdir(parents=True)
    for name in accepted["helpers_sha256"]:
        shutil.copy2(Path(accepted["helpers"]) / name, helpers / name)
    with tarfile.open(archive, "x") as bundle:
        for name in accepted["helpers_sha256"]:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    frozen = {
        "status": owner.STATUS,
        "root": str(root),
        "helpers": str(helpers),
        "work": str(work),
        "node": owner.NODE,
        "controller_command": owner.controller_command(root),
        "environment": owner.ENVIRONMENT,
        "worker_environment": owner.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": keeper["devices"][0][1]},
        "helpers_sha256": accepted["helpers_sha256"],
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "cpu_receipt": str(cpu_receipt),
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "predecessor": previous,
        "devices": keeper["devices"],
        "workers": 2,
        "worker_timeout": owner.WORKER_TIMEOUT,
        "scope": "Full saved B batches, opposite pinned/rational order, eager/capture/two-repeat start accuracy only.",
    }
    require_archive(
        archive, {f"tracking/{name}": value for name, value in frozen["helpers_sha256"].items()}, alias=False
    )
    if owner.predecessor(root) != previous:
        raise ValueError("Start accuracy predecessor changed during freeze")
    write(binding, frozen)
    return frozen


def start(root: Path) -> dict:
    binding, helpers, state, work = owner.locations(root)
    frozen = owner.require_frozen(root, sha256(binding))
    intent_path, log = helpers.parent / "controller.json", helpers.parent / "controller.log"
    with (helpers.parent / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if local_node() != owner.NODE:
            raise RuntimeError("Start accuracy launch is outside its original allocation")
        if any(path.exists() for path in (intent_path, log, state, work)):
            raise FileExistsError("Start accuracy already has one-shot launch evidence")
        with keeper_guard():
            idle_training()
            owner.require_live_predecessor(root, frozen["predecessor"])
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
                    | owner.ENVIRONMENT
                    | {owner.CONTROLLER_MARKER: intent["marker"], "PYTHONPATH": str(helpers)},
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            own = acknowledged_process(
                child, intent["command"], helpers, owner.CONTROLLER_MARKER, intent["marker"], owner.ENVIRONMENT
            )
        except BaseException as failure:
            intent.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
            try:
                if child is not None:
                    finish_marked_child(child, intent["marker"], {}, owner.CONTROLLER_MARKER)
                    intent.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
            finally:
                intent["log_sha256"] = sha256(log)
                write(intent_path, intent)
            raise
        intent.update(status="acknowledged", pid=own["pid"], start_ticks=own["start_ticks"], generation=own)
        write(intent_path, intent)
    return intent


def run_worker(root: Path, frozen: dict, state: dict, index: int) -> dict:
    binding, helpers, state_path, work = owner.locations(root)
    command, marker, generations = owner.worker_command(root, sha256(binding), index), uuid4().hex, {}
    entry = {
        "index": index,
        "command": command,
        "marker": marker,
        "spawn_status": "starting",
        "controller_pid": state["controller_pid"],
        "controller_start_ticks": state["controller_start_ticks"],
    }
    state["active_worker"] = entry
    write(state_path, state)
    log = work / f"process-{index}.log"
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=helpers,
            env=os.environ | frozen["worker_environment"] | {owner.CHILD_MARKER: marker, "PYTHONPATH": str(helpers)},
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            own = acknowledged_process(child, command, helpers, owner.CHILD_MARKER, marker, frozen["worker_environment"])
            entry.update(pid=own["pid"], start_ticks=own["start_ticks"], generation=own, spawn_status="acknowledged")
            write(state_path, state)
            code = child.wait(timeout=owner.WORKER_TIMEOUT)
            entry["exit_code"] = code
            if type(code) is not int or code != 0:
                raise RuntimeError("Start accuracy one-shot worker failed")
        finally:
            try:
                finish_marked_child(child, marker, generations, owner.CHILD_MARKER)
                entry.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
            except BaseException as failure:
                state["cleanup_error"] = f"{type(failure).__name__}: {failure}"
                raise
            finally:
                entry.update(log=str(log), log_sha256=sha256(log), generations=generations)
                write(state_path, state)
    entry.update(report=str(work / f"process-{index}.json"), report_sha256=sha256(work / f"process-{index}.json"))
    return entry


def expected_owner(root: Path, frozen: dict, entry: dict, index: int) -> dict:
    generation = entry["generation"]
    work = owner.locations(root)[3]
    if (
        entry["index"] != index
        or entry["spawn_status"] != "acknowledged"
        or not entry["marker"]
        or entry["command"] != owner.worker_command(root, sha256(owner.locations(root)[0]), index)
        or any(type(entry[key]) is not int or entry[key] != 0 for key in ("exit_code", "cleanup_exit_code"))
        or entry["marked_descendants_absent"] is not True
        or generation["pid"] != entry["pid"]
        or generation["start_ticks"] != entry["start_ticks"]
        or generation["parent"] != entry["controller_pid"]
        or generation["command"] != entry["command"]
        or generation["group"] != entry["pid"]
        or generation["session"] != entry["pid"]
        or generation["status"] in ("Z", "X", "x")
        or entry["log"] != str(work / f"process-{index}.log")
        or entry["log_sha256"] != sha256(Path(entry["log"]))
        or entry["report"] != str(work / f"process-{index}.json")
    ):
        raise ValueError("Start accuracy report lacks authenticated execution and cleanup origin")
    return {
        "binding": frozen,
        "device": "cuda:0",
        "physical_uuid": frozen["devices"][0][1],
        "controller_pid": entry["controller_pid"],
        "controller_start_ticks": entry["controller_start_ticks"],
        "worker_pid": entry["pid"],
        "worker_start_ticks": entry["start_ticks"],
    }


def compare_results(root: Path, entries: list[dict]) -> dict:
    if (
        len(entries) != 2
        or len({(entry["pid"], entry["start_ticks"]) for entry in entries}) != 2
        or len({entry["marker"] for entry in entries}) != 2
        or len({(entry["controller_pid"], entry["controller_start_ticks"]) for entry in entries}) != 1
    ):
        raise ValueError("Start accuracy requires two distinct opposite-order executions under one owner")
    frozen = owner.require_frozen(root, sha256(owner.locations(root)[0]))
    results, retained = [], []
    for index, entry in enumerate(entries):
        result, output = execution.reconstruct(
            Path(entry["report"]),
            entry["report_sha256"],
            expected_owner(root, frozen, entry, index),
            frozen["helpers_sha256"],
            index,
        )
        results.append(result)
        retained.append(
            {tuple(item[key] for key in ("variant", "group", "mode", "phase")): item["output"] for item in output}
        )
    crossed = [
        {"coordinate": list(key), **tracing.comparison(retained[1][key], retained[0][key], f"cross.{'.'.join(key)}")}
        for key in sorted(retained[0])
    ]
    return {
        "status": "saved_start_accuracy_independently_reconstructed_not_continuation_or_production",
        "executions": entries,
        "processes": results,
        "cross_process": crossed,
        "all_cross_process_exact_finite": all(accuracy.exact_finite(item) for item in crossed),
        "all_repeat_outputs_exact_finite": all(
            result["analysis"]["all_repeat_outputs_exact_finite"] for result in results
        ),
        "all_unaffected_outputs_exact_finite": all(
            result["analysis"]["all_unaffected_outputs_exact_finite"] for result in results
        ),
        "all_rational_O_within_fp64_tolerance": all(
            result["analysis"]["all_rational_O_within_fp64_tolerance"] for result in results
        ),
        "scope": execution.LIMITATION,
    }


def execute(root: Path) -> dict:
    binding, helpers, state_path, work = owner.locations(root)
    digest = sha256(binding)
    frozen = owner.require_frozen(root, digest)
    intent = read(helpers.parent / "controller.json")
    deadline = time.monotonic() + 30
    while intent["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.01)
        intent = read(helpers.parent / "controller.json")
    own = process_context(
        os.getpid(),
        owner.controller_command(root),
        helpers,
        owner.CONTROLLER_MARKER,
        os.environ.get(owner.CONTROLLER_MARKER, ""),
        owner.ENVIRONMENT,
    )
    require_acknowledged_controller(intent, own)
    if intent["binding_sha256"] != digest or local_node() != owner.NODE:
        raise RuntimeError("Start accuracy controller lacks this allocation and binding")
    if state_path.exists() or work.exists():
        raise FileExistsError("Start accuracy already has one-shot execution evidence")
    state = {
        "status": "preflight",
        "binding_sha256": digest,
        "controller_pid": own["pid"],
        "controller_start_ticks": own["start_ticks"],
        "predecessor": frozen["predecessor"],
        "executions": [],
    }
    write(state_path, state)
    signals = []
    try:
        with node_guard(root, owner.NODE) as signals, keeper_guard() as manager:
            idle_training()
            accepted = owner.require_live_predecessor(root, frozen["predecessor"])
            work.mkdir(parents=True)
            state["lock_descriptors"] = {
                "node": lock_descriptor(root / "run-preparation" / f".wave7-node-{owner.NODE}.lock"),
                "manager": manager,
            }
            try:
                release_accepted_keeper(accepted, state, state_path, manager)
                state["status"] = "replaying"
                write(state_path, state)
                for index in range(2):
                    state["executions"].append(run_worker(root, frozen, state, index))
                    write(state_path, state)
                result = compare_results(root, state["executions"])
                write(work / "comparison.json", result)
                state.update(status="diagnostic_complete", comparison_sha256=sha256(work / "comparison.json"))
            finally:
                if state.get("keeper_release_attempted"):
                    markers = [entry["marker"] for entry in state["executions"]]
                    if state.get("active_worker"):
                        markers.append(state["active_worker"]["marker"])
                    if (
                        state.get("keeper_release_complete") is not True
                        or "cleanup_error" in state
                        or any(owner.handoff.marker_processes(owner.CHILD_MARKER, marker) for marker in markers)
                    ):
                        state["keeper_restore"] = "refused: incomplete transfer or live descendants"
                        raise RuntimeError("Start accuracy cleanup prevents keeper restoration")
                    restore_accepted_keeper(manager)
                    restored = owner.handoff.keeper_observation()
                    owner.require_keeper_shape(restored)
                    if restored["devices"] != accepted["devices"]:
                        raise RuntimeError("Start accuracy restored physical devices differ")
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
    binding, helpers, state_path, work = owner.locations(root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or out.exists() or not out.is_relative_to(root / "run-preparation"):
        raise RuntimeError("Start accuracy terminal reconstruction requires hidden CUDA and unused receipt")
    state, intent = read(state_path), read(helpers.parent / "controller.json")
    if (
        state["status"] != "diagnostic_complete"
        or state["binding_sha256"] != sha256(binding)
        or state["keeper_release_complete"] is not True
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
        or intent["status"] != "acknowledged"
        or intent["command"] != owner.controller_command(root)
        or intent["binding_sha256"] != state["binding_sha256"]
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or intent["generation"]["pid"] != intent["pid"]
        or intent["generation"]["start_ticks"] != intent["start_ticks"]
        or intent["generation"]["command"] != intent["command"]
        or intent["generation"]["group"] != intent["pid"]
        or intent["generation"]["session"] != intent["pid"]
        or len(state["executions"]) != 2
        or state["active_worker"] != state["executions"][-1]
        or local_node() != owner.NODE
    ):
        raise RuntimeError("Start accuracy terminal owner or restoration is incomplete")
    if owner.handoff.observed(state["controller_pid"]) is not None or any(
        owner.handoff.observed(entry["pid"]) is not None
        or owner.handoff.marker_processes(owner.CHILD_MARKER, entry["marker"])
        for entry in state["executions"]
    ):
        raise RuntimeError("Start accuracy original owner or worker descendants remain present")
    frozen = owner.require_frozen(root, sha256(binding))
    if state["predecessor"] != frozen["predecessor"] or any(
        entry["controller_pid"] != state["controller_pid"]
        or entry["controller_start_ticks"] != state["controller_start_ticks"]
        for entry in state["executions"]
    ):
        raise ValueError("Start accuracy terminal execution ancestry or allocation predecessor differs")
    paths = (
        binding,
        state_path,
        helpers.parent / "controller.json",
        helpers.parent / "controller.log",
        work / "comparison.json",
        *(work / f"process-{index}{suffix}" for index in range(2) for suffix in (".log", ".json", ".pt")),
    )
    evidence = {str(path.relative_to(root)): sha256(path) for path in paths}
    stored = read(work / "comparison.json")
    comparison_time, terminal_time = (datetime.fromisoformat(value["time"]) for value in (stored, state))
    if comparison_time.tzinfo is None or terminal_time.tzinfo is None or comparison_time > terminal_time:
        raise ValueError("Start accuracy terminal chronology differs")
    comparison = compare_results(root, state["executions"])
    if state["comparison_sha256"] != sha256(work / "comparison.json") or comparison != {
        key: value for key, value in stored.items() if key != "time"
    }:
        raise ValueError("Start accuracy comparison differs from independent retained reconstruction")
    with keeper_guard():
        keeper = owner.handoff.keeper_observation()
        owner.require_keeper_shape(keeper)
        marked = owner.handoff.marker_processes(owner.CONTROLLER_MARKER, intent["marker"])
        if (
            not same_keeper(keeper, state["restored_keeper"])
            or keeper["devices"] != frozen["devices"]
            or not set(marked).issubset({int(pid) for pid in keeper["generation"]})
        ):
            raise RuntimeError("Start accuracy exact restoration or controller marker coverage differs")
    if evidence != {str(path.relative_to(root)): sha256(path) for path in paths}:
        raise ValueError("Start accuracy terminal evidence changed during reconstruction")
    result = {
        "status": "start_accuracy_terminal_independently_reconstructed_exact_restoration_verified",
        "binding_sha256": sha256(binding),
        "terminal_time": state["time"],
        "evidence_sha256": evidence,
        "controller_pid_absent": state["controller_pid"],
        "worker_pids_absent": [entry["pid"] for entry in state["executions"]],
        "keeper": keeper,
        "comparison": comparison,
        "verifier_sha256": sha256(Path(__file__)),
        "scope": execution.LIMITATION,
    }
    write(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--phase", required=True, choices=("cpu", "seal-predecessor", "freeze", "check", "start", "execute", "terminal")
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, args.out.resolve())
    elif args.phase == "seal-predecessor":
        result = owner.seal(root)
    elif args.phase == "freeze":
        if args.cpu_receipt is None:
            parser.error("Freeze requires --cpu-receipt")
        result = freeze(root, args.cpu_receipt.resolve(strict=True))
    elif args.phase == "check":
        result = owner.require_frozen(root, sha256(owner.locations(root)[0]))
    elif args.phase == "start":
        result = start(root)
    elif args.phase == "terminal":
        if args.out is None:
            parser.error("Terminal reconstruction requires --out")
        result = terminal(root, args.out.resolve())
    else:
        result = execute(root)
    print(result["status"])


if __name__ == "__main__":
    main()
