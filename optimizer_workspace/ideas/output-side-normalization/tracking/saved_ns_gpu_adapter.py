# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and execute a bounded two-process NS replay after original-owner restoration."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import tarfile
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import UUID, uuid4

import analyze_fused_continuation_stages as analyzer
import owned_output_handoff as handoff
import replay_saved_ns as replay
import saved_ns_gpu_owner as owner
import torch
from fused_attribution_validation_controller import (
    acknowledged_process,
    finish_marked_child,
    process_context,
    require_acknowledged_controller,
)
from fused_continuation_contract import locations as previous_locations
from prepare_selected_lr_validation import require_archive
from selected_lr_failure_handoff import keeper_guard, release_accepted_keeper, restore_accepted_keeper
from selected_lr_release_owner import read, sha256
from watch_wave5 import write
from wave7_release import idle_training, local_node, node_guard

CPU_STATUS = owner.CPU_STATUS
WORKER_TIMEOUT = 600


def helper_sources(root: Path) -> dict[str, Path]:
    return owner.helper_sources(root)


def cpu_command(helpers: Path) -> list[str]:
    return owner.cpu_command(helpers)


def cpu(root: Path, out: Path) -> dict:
    if out.exists() or out.is_relative_to(Path(__file__).parents[1]):
        raise FileExistsError("Replay CPU verification requires a new external directory")
    sources = helper_sources(root)
    owner.require_loaded_sources(sources)
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    hashes = {name: sha256(path) for name, path in sources.items()}
    for name, path in sources.items():
        shutil.copy2(path, helpers / name)
    candidate = Path(read(previous_locations(root)[0])["candidate"])
    manifest_path = candidate / "experiment/upstream_manifest.json"
    manifest = read(manifest_path)
    relative = f"artifacts/pinned-sources-materialized/Toner-{manifest['revision']}/src/toner/fast/muon.py"
    numerical_path = candidate / relative
    if sha256(numerical_path) != replay.MUON_SHA256:
        raise ValueError("Replay CPU fixture pinned NS bytes differ")
    fixtures = {
        "experiment/upstream_manifest.json": manifest_path,
        relative.replace("pinned-sources-materialized", "external-sources"): numerical_path,
    }
    for name, path in fixtures.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    log, archive = out / "cpu.log", out / "source.tar"
    command = cpu_command(helpers)
    with log.open("x") as stream:
        code = subprocess.run(
            command,
            cwd=out,
            env=os.environ | owner.ENVIRONMENT | {"PYTHONPATH": str(helpers)},
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=900,
        ).returncode
    with tarfile.open(archive, "x") as bundle:
        for name in hashes:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        for name in fixtures:
            bundle.add(out / name, arcname=name, recursive=False)
    fixture_hashes = {name: sha256(path) for name, path in fixtures.items()}
    require_archive(archive, {f"tracking/{name}": value for name, value in hashes.items()} | fixture_hashes, alias=False)
    unchanged = hashes == {name: sha256(path) for name, path in sources.items()}
    result = {
        "status": CPU_STATUS if type(code) is int and code == 0 and unchanged else "needs_attention",
        "root": str(root),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "command": command,
        "cwd": str(out),
        "exit_code": code,
        "cuda_visible_devices": "",
        "fixtures_sha256": fixture_hashes,
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "scope": "CPU/source archive only; no keeper mutation, GPU worker or production authorization.",
    }
    write(out / "receipt.json", result)
    if result["status"] != CPU_STATUS:
        raise RuntimeError(f"Replay CPU failure preserved at {out}")
    return result


def require_cpu(root: Path, receipt_path: Path) -> dict:
    owner.require_loaded_sources(helper_sources(root))
    result = read(receipt_path)
    helpers = Path(result["helpers"])
    if (
        result["status"] != CPU_STATUS
        or result["root"] != str(root)
        or result["command"] != cpu_command(helpers)
        or type(result["exit_code"]) is not int
        or result["exit_code"] != 0
        or result["cuda_visible_devices"] != ""
        or result["helpers_sha256"] != {name: sha256(path) for name, path in helper_sources(root).items()}
        or any(sha256(helpers / name) != value for name, value in result["helpers_sha256"].items())
        or result["log_sha256"] != sha256(Path(result["log"]))
        or result["archive_sha256"] != sha256(Path(result["archive"]))
        or any(sha256(helpers.parent / name) != value for name, value in result["fixtures_sha256"].items())
    ):
        raise ValueError("Replay CPU/source acceptance changed")
    require_archive(
        Path(result["archive"]),
        {f"tracking/{name}": value for name, value in result["helpers_sha256"].items()} | result["fixtures_sha256"],
        alias=False,
    )
    return result


def freeze(root: Path, cpu_receipt: Path) -> dict:
    verified = require_cpu(root, cpu_receipt)
    previous = owner.predecessor(root)
    binding_path, helpers, state, work = owner.locations(root)
    if any(path.exists() for path in (binding_path, helpers.parent, state, work)):
        raise FileExistsError("Replay identity already has preparation or execution evidence")
    with keeper_guard():
        keeper = owner.require_live_predecessor(root, previous)
    helpers.mkdir(parents=True)
    for name in verified["helpers_sha256"]:
        shutil.copy2(Path(verified["helpers"]) / name, helpers / name)
    archive = root / "run-preparation/source-saved-ns-replay-20261118.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in verified["helpers_sha256"]:
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
        "helpers_sha256": verified["helpers_sha256"],
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "cpu_receipt": str(cpu_receipt),
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "predecessor": previous,
        "devices": keeper["devices"],
        "worker_timeout": WORKER_TIMEOUT,
        "workers": 2,
        "scope": "Unchanged pinned NS saved-input CUDA replay only, not a training or continuation gate.",
        "cpu_command": verified["command"],
    }
    if owner.predecessor(root) != previous:
        raise ValueError("Replay predecessor changed during freeze")
    write(binding_path, frozen)
    return frozen


def start(root: Path) -> dict:
    binding_path, helpers, state_path, work = owner.locations(root)
    frozen = owner.require_frozen(root, sha256(binding_path))
    intent_path, log = helpers.parent / "controller.json", helpers.parent / "controller.log"
    if local_node() != owner.NODE:
        raise RuntimeError("Saved NS replay start is not on the original allocation")
    with (helpers.parent / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.exists() for path in (intent_path, log, state_path, work)):
            raise FileExistsError("Replay already has one-shot intent/evidence; preserve it")
        with keeper_guard():
            idle_training()
            owner.require_live_predecessor(root, frozen["predecessor"])
        intent = {
            "status": "starting",
            "binding_sha256": sha256(binding_path),
            "command": owner.controller_command(root),
            "marker": uuid4().hex,
        }
        write(intent_path, intent)
        with log.open("x") as stream:
            child = subprocess.Popen(
                intent["command"],
                cwd=helpers,
                env=os.environ
                | owner.ENVIRONMENT
                | {
                    owner.CONTROLLER_MARKER: intent["marker"],
                    "PYTHONPATH": str(helpers),
                },
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        process = acknowledged_process(
            child, intent["command"], helpers, owner.CONTROLLER_MARKER, intent["marker"], owner.ENVIRONMENT
        )
        intent.update(status="acknowledged", pid=child.pid, start_ticks=process["start_ticks"], generation=process)
        write(intent_path, intent)
    return intent


def lock_descriptor(path: Path) -> int:
    matches = []
    for descriptor in Path("/proc/self/fd").iterdir():
        try:
            if descriptor.resolve() == path:
                matches.append(int(descriptor.name))
        except FileNotFoundError:
            continue
    if len(matches) != 1:
        raise RuntimeError("Replay controller has an ambiguous lock descriptor")
    owner.require_held_lock(os.getpid(), matches[0], path)
    return matches[0]


def run_worker(root: Path, frozen: dict, state: dict, index: int) -> dict:
    binding_path, helpers, state_path, work = owner.locations(root)
    command = owner.worker_command(root, sha256(binding_path), index)
    marker, generations = uuid4().hex, {}
    log = work / f"process-{index}.log"
    worker = {
        "index": index,
        "command": command,
        "marker": marker,
        "controller_pid": state["controller_pid"],
        "controller_start_ticks": state["controller_start_ticks"],
        "spawn_status": "starting",
    }
    state["active_worker"] = worker
    write(state_path, state)
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=helpers,
            env=os.environ
            | frozen["worker_environment"]
            | {
                owner.CHILD_MARKER: marker,
                "PYTHONPATH": str(helpers),
            },
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process = acknowledged_process(
                child, command, helpers, owner.CHILD_MARKER, marker, frozen["worker_environment"]
            )
            worker.update(pid=child.pid, start_ticks=process["start_ticks"], spawn_status="acknowledged")
            write(state_path, state)
            code = child.wait(timeout=WORKER_TIMEOUT)
            worker["exit_code"] = code
            if type(code) is not int or code != 0:
                raise RuntimeError("Saved NS CUDA replay worker failed")
        finally:
            try:
                finish_marked_child(child, marker, generations, owner.CHILD_MARKER)
                worker.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
            except BaseException as failure:
                state["cleanup_error"] = f"{type(failure).__name__}: {failure}"
                raise
            finally:
                worker.update(log=str(log), log_sha256=sha256(log), generations=generations)
                write(state_path, state)
    worker.update(report=str(work / f"process-{index}.json"), report_sha256=sha256(work / f"process-{index}.json"))
    return worker


def compare_results(root: Path, executions: list[dict]) -> dict:
    if len(executions) != 2:
        raise ValueError("Saved NS replay requires two complete process executions")
    binding_hash = sha256(owner.locations(root)[0])
    frozen = owner.require_frozen(root, binding_hash)
    reports, outputs = [], []
    with ExitStack() as stack:
        saved, provenance = replay.load_saved_group(stack)
        config = analyzer._json(
            stack.enter_context(
                analyzer._bound_file(Path(read(root / analyzer.RECEIPT_RELATIVE)["config"]), provenance["config_sha256"])
            )
        )
        for index, execution in enumerate(executions):
            report = analyzer._json(
                stack.enter_context(analyzer._bound_file(Path(execution["report"]), execution["report_sha256"]))
            )
            require_report_origin(report, execution, root, binding_hash, index)
            require_report_context(report, frozen, config["runtime"], provenance)
            stream = stack.enter_context(analyzer._bound_file(Path(report["outputs"]), report["outputs_sha256"]))
            tensors = torch.load(f"/proc/self/fd/{stream.fileno()}", map_location="cpu", weights_only=True, mmap=True)
            if len(tensors) != 4 or any(tuple(value.shape) != replay.SAVED_SHAPE for value in tensors):
                raise ValueError("Saved NS replay output coverage differs")
            rederived = {
                "continuous": replay.compare_outputs(tensors[2], tensors[0]),
                "resumed": replay.compare_outputs(tensors[3], tensors[1]),
            }
            crossed = [replay.compare_outputs(tensors[1], tensors[0]), replay.compare_outputs(tensors[3], tensors[2])]
            saved_comparisons = []
            for call_index, (mode, value) in enumerate(zip(replay.ORDER, tensors, strict=True)):
                other = "resumed" if mode == "continuous" else "continuous"
                saved_comparisons.append(
                    {
                        "call_index": call_index,
                        "source_mode": mode,
                        "own_saved_output": replay.compare_outputs(value, saved[mode]["output"]),
                        "other_saved_output": replay.compare_outputs(value, saved[other]["output"]),
                    }
                )
            source_metrics = replay.compare_outputs(saved["resumed"]["source"], saved["continuous"]["source"])
            if (
                report["repeats"] != rederived
                or report["cross_outputs"] != crossed
                or report["saved_comparisons"] != saved_comparisons
                or report["source_comparison"] != source_metrics
            ):
                raise ValueError("Saved NS worker metrics differ from retained output reconstruction")
            reports.append(report)
            outputs.append(tensors)
        comparisons = [replay.compare_outputs(second, first) for first, second in zip(*outputs, strict=True)]
    return {
        "status": "saved_ns_cuda_replay_compared_not_production_gate",
        "executions": executions,
        "reports": reports,
        "cross_process": comparisons,
        "all_cross_process_exact": all(item["exact"] and item["finite"] for item in comparisons),
    }


def require_report_context(report: dict, frozen: dict, runtime: dict, provenance: dict) -> None:
    """Validate complete execution sources, actual device and raw-layout evidence."""
    physical_uuid = frozen["devices"][0][1]
    if (
        report["owner"]["binding"] != frozen
        or report["environment"] != frozen["worker_environment"]
        or report["runtime"] != runtime
        or report["device"]["physical_uuid"] != physical_uuid
        or UUID(report["device"]["actual_uuid"].removeprefix("GPU-")) != UUID(physical_uuid.removeprefix("GPU-"))
        or report["device"]["logical"] != "cuda:0"
        or report["replay_mode"] != "tpu"
        or report["runtime"]["deterministic"] != report["numerical_settings"]["deterministic_algorithms"]
        or report["numerical_settings"]["tf32_matmul"] is not False
        or report["numerical_settings"]["tf32_cudnn"] is not False
        or report["numerical_settings"]["autocast"] is not False
        or report["numerical_settings"]["cuda_source_uploads"] != 2
        or report["numerical_settings"]["source_values_checked_after_each_call"] is not True
        or report["worker_sources_sha256"] != owner.worker_sources(frozen)
    ):
        raise ValueError("Saved NS report physical environment or executed helper identity differs")
    for mode in replay.analyzer.MODES:
        raw = provenance["raw_layouts"][mode]["source"]
        if any(
            report["source_layouts"][mode][place][key] != raw[key]
            for place in ("cpu", "cuda")
            for key in ("shape", "stride", "storage_offset")
        ):
            raise ValueError("Saved NS report source layout differs from the authenticated raw trace")
    if report["provenance"] != provenance:
        raise ValueError("Saved NS replay reports consume different saved evidence")


def require_report_origin(report: dict, execution: dict, root: Path, binding_hash: str, index: int) -> None:
    """Bind result bytes to the acknowledged worker and canonical diagnostic output paths."""
    work = owner.locations(root)[3]
    process = report["owner"]
    if (
        report["status"] != "saved_ns_cuda_replay_not_production_gate"
        or report["worker_index"] != index
        or report["binding_sha256"] != binding_hash
        or report["order"] != list(replay.ORDER)
        or report["tolerance"] != replay.MOMENT_TOLERANCE
        or report["replay_mode"] != "tpu"
        or report["ns_steps"] != replay.NS_STEPS
        or report["ns_eps"] != replay.NS_EPS
        or process["worker_pid"] != execution["pid"]
        or process["worker_start_ticks"] != execution["start_ticks"]
        or process["controller_pid"] != execution["controller_pid"]
        or process["controller_start_ticks"] != execution["controller_start_ticks"]
        or execution["command"] != owner.worker_command(root, binding_hash, index)
        or type(execution["exit_code"]) is not int
        or execution["exit_code"] != 0
        or type(execution["cleanup_exit_code"]) is not int
        or execution["cleanup_exit_code"] != 0
        or execution["marked_descendants_absent"] is not True
        or Path(report["report_path"]) != work / f"process-{index}.json"
        or Path(execution["report"]) != work / f"process-{index}.json"
        or Path(report["outputs"]) != work / f"process-{index}.pt"
    ):
        raise ValueError("Saved NS report is not from its exact acknowledged replay process")


def execute(root: Path) -> dict:
    binding_path, helpers, state_path, work = owner.locations(root)
    digest = sha256(binding_path)
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
    if (
        intent["binding_sha256"] != digest
        or intent["command"] != owner.controller_command(root)
        or intent["marker"] != os.environ.get(owner.CONTROLLER_MARKER)
        or local_node() != owner.NODE
    ):
        raise RuntimeError("Replay controller acknowledgment does not authorize this binding or node")
    if state_path.exists() or work.exists():
        raise FileExistsError("Replay execution evidence already exists")
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
        with node_guard(root, owner.NODE) as signals, keeper_guard() as manager_lock:
            idle_training()
            keeper = owner.require_live_predecessor(root, frozen["predecessor"])
            owner.require_frozen(root, digest)
            work.mkdir(parents=True)
            state["lock_descriptors"] = {
                "node": lock_descriptor(root / "run-preparation" / f".wave7-node-{owner.NODE}.lock"),
                "manager": manager_lock,
            }
            try:
                release_accepted_keeper(keeper, state, state_path, manager_lock)
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
                    marker = state.get("active_worker", {}).get("marker")
                    if (
                        state.get("keeper_release_complete") is not True
                        or "cleanup_error" in state
                        or (marker and handoff.marker_processes(owner.CHILD_MARKER, marker))
                    ):
                        state["keeper_restore"] = "refused: incomplete transfer or live marked children"
                        raise RuntimeError("Replay cleanup does not permit keeper restoration")
                    restore_accepted_keeper(manager_lock)
                    restored = handoff.keeper_observation()
                    if restored["devices"] != keeper["devices"]:
                        raise RuntimeError("Replay keeper restoration physical mapping differs")
                    state.update(keeper_restore=0, restored_keeper=restored)
    except BaseException as failure:
        state.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        raise
    finally:
        if signals:
            state["deferred_signals"] = signals
        write(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("cpu", "freeze", "check", "start", "execute"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU verification requires --out")
        result = cpu(root, args.out.resolve())
    elif args.phase == "freeze":
        if args.cpu_receipt is None:
            parser.error("Freeze requires --cpu-receipt")
        result = freeze(root, args.cpu_receipt.resolve())
    elif args.phase == "check":
        result = owner.require_frozen(root, sha256(owner.locations(root)[0]))
    elif args.phase == "start":
        result = start(root)
    else:
        result = execute(root)
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
