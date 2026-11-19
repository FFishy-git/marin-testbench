# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive and run the separately owned root-stage replay with exact cleanup/restoration."""

import argparse
import fcntl
import os
import shutil
import subprocess
import tarfile
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import replay_saved_roots as replay
import root_stage_gpu_owner as owner
import root_stage_gpu_worker as worker
import root_stage_trace as tracing
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


def cpu(root: Path, out: Path) -> dict:
    if out.exists() or out.is_relative_to(Path(__file__).resolve().parents[1]):
        raise FileExistsError("Root replay CPU acceptance requires an unused external directory")
    sources = owner.helper_sources(root)
    owner.require_loaded_sources(sources)
    hashes = {name: sha256(path) for name, path in sources.items()}
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    for name, source in sources.items():
        shutil.copy2(source, helpers / name)
    candidate = Path(read(owner.previous.locations(root)[0])["candidate"])
    manifest = read(candidate / "experiment/upstream_manifest.json")
    prefix = f"artifacts/external-sources/Toner-{manifest['revision']}"
    numerical = candidate / prefix
    fixtures = {"experiment/upstream_manifest.json": candidate / "experiment/upstream_manifest.json"}
    fixtures.update(
        {f"{prefix}/{name}": numerical / name for name in ("src/toner/fast/roots.py", "reproduce/marin/optimizer.py")}
    )
    fixtures[f"{prefix}/src/toner/fast/engine.py"] = numerical / "src/toner/fast/engine.py"
    fixtures.update(
        {
            "experiment/records.py": candidate / "experiment/records.py",
            "experiment/configs/inner_outer_leftdiag_0p5m_seed0.json": candidate / owner.previous.CONFIG,
        }
    )
    for name, source in fixtures.items():
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    fixture_hashes = {name: sha256(path) for name, path in fixtures.items()}
    code, error = 124, None
    log = out / "cpu.log"
    try:
        with log.open("x") as stream:
            code = subprocess.run(
                owner.cpu_command(out),
                cwd=out,
                env=os.environ
                | owner.ENVIRONMENT
                | {"PYTHONPATH": f"{helpers}:{Path(torch.__file__).parents[1]}", "UV_OFFLINE": "1"},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
    except (OSError, subprocess.TimeoutExpired) as failure:
        error = f"{type(failure).__name__}: {failure}"
    archive_path = out / "source.tar"
    inventory = {f"tracking/{name}": value for name, value in hashes.items()} | fixture_hashes
    with tarfile.open(archive_path, "x") as archive:
        for name in inventory:
            archive.add(out / name, arcname=name, recursive=False)
    require_archive(archive_path, inventory, alias=False)
    if hashes != {name: sha256(path) for name, path in sources.items()}:
        error = "Root replay checkout source changed during CPU acceptance"
    result = {
        "status": owner.CPU_STATUS if code == 0 and error is None else "needs_attention",
        "root": str(root),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "fixtures_sha256": fixture_hashes,
        "command": owner.cpu_command(out),
        "exit_code": code,
        "cuda_visible_devices": "",
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "tests": str(out / "tests.xml"),
        "tests_sha256": sha256(out / "tests.xml") if (out / "tests.xml").exists() else None,
        "scope": "CPU/source archive only. No predecessor transfer, CUDA allocation or production acceptance.",
    }
    if error:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != owner.CPU_STATUS:
        raise RuntimeError(f"Root replay failed CPU attempt retained at {out}")
    return owner.require_cpu(root, out / "receipt.json")


def freeze(root: Path, cpu_receipt: Path) -> dict:
    accepted = owner.require_cpu(root, cpu_receipt)
    predecessor = owner.predecessor(root)
    binding_path, helpers, state_path, work = owner.locations(root)
    archive_path = root / "run-preparation/source-root-stage-replay-20261118.tar"
    if any(path.exists() for path in (binding_path, helpers.parent, state_path, work, archive_path)):
        raise FileExistsError("Root replay identity already has preparation or execution evidence")
    with keeper_guard():
        keeper = owner.require_live_predecessor(root, predecessor)
    helpers.mkdir(parents=True)
    for name in accepted["helpers_sha256"]:
        shutil.copy2(Path(accepted["helpers"]) / name, helpers / name)
    with tarfile.open(archive_path, "x") as archive:
        for name in accepted["helpers_sha256"]:
            archive.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
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
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "cpu_receipt": str(cpu_receipt),
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "predecessor": predecessor,
        "devices": keeper["devices"],
        "workers": 2,
        "worker_timeout": owner.WORKER_TIMEOUT,
        "scope": (
            "Two fresh processes; opposite trace-off/on order; saved d/f inputs; "
            "unchanged compiled eager/capture/replay."
        ),
    }
    require_archive(
        archive_path, {f"tracking/{name}": value for name, value in frozen["helpers_sha256"].items()}, alias=False
    )
    if owner.predecessor(root) != predecessor or owner.require_cpu(root, cpu_receipt) != accepted:
        raise ValueError("Root replay acceptance changed during freeze")
    write(binding_path, frozen)
    return frozen


def start(root: Path) -> dict:
    binding_path, helpers, state_path, work = owner.locations(root)
    frozen = owner.require_frozen(root, sha256(binding_path))
    intent_path, log = helpers.parent / "controller.json", helpers.parent / "controller.log"
    with (helpers.parent / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if local_node() != owner.NODE:
            raise RuntimeError("Root replay start is outside the original allocation")
        if any(path.exists() for path in (intent_path, log, state_path, work)):
            raise FileExistsError("Root replay already has one-shot intent or evidence")
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
        except OSError as failure:
            intent.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
            if log.exists():
                intent["log_sha256"] = sha256(log)
            write(intent_path, intent)
            raise
        try:
            process = acknowledged_process(
                child, intent["command"], helpers, owner.CONTROLLER_MARKER, intent["marker"], owner.ENVIRONMENT
            )
        except BaseException as failure:
            generations = {}
            intent.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
            try:
                finish_marked_child(child, intent["marker"], generations, owner.CONTROLLER_MARKER)
                intent.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
            except BaseException as cleanup:
                intent["cleanup_error"] = f"{type(cleanup).__name__}: {cleanup}"
                raise
            finally:
                intent.update(generations=generations, log_sha256=sha256(log))
                write(intent_path, intent)
            raise
        intent.update(status="acknowledged", pid=child.pid, start_ticks=process["start_ticks"], generation=process)
        write(intent_path, intent)
    return intent


def run_worker(root: Path, frozen: dict, state: dict, index: int) -> dict:
    binding_path, helpers, state_path, work = owner.locations(root)
    command, marker, generations = owner.worker_command(root, sha256(binding_path), index), uuid4().hex, {}
    execution = {
        "index": index,
        "command": command,
        "marker": marker,
        "spawn_status": "starting",
        "controller_pid": state["controller_pid"],
        "controller_start_ticks": state["controller_start_ticks"],
    }
    state["active_worker"] = execution
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
            process = acknowledged_process(
                child, command, helpers, owner.CHILD_MARKER, marker, frozen["worker_environment"]
            )
            execution.update(
                pid=child.pid,
                start_ticks=process["start_ticks"],
                spawn_status="acknowledged",
                generation=process,
            )
            write(state_path, state)
            code = child.wait(timeout=owner.WORKER_TIMEOUT)
            execution["exit_code"] = code
            if type(code) is not int or code != 0:
                raise RuntimeError("Root replay worker failed; preserve one-shot outputs")
        finally:
            try:
                finish_marked_child(child, marker, generations, owner.CHILD_MARKER)
                execution.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
            except BaseException as failure:
                state["cleanup_error"] = f"{type(failure).__name__}: {failure}"
                raise
            finally:
                execution.update(log=str(log), log_sha256=sha256(log), generations=generations)
                write(state_path, state)
    execution.update(report=str(work / f"process-{index}.json"), report_sha256=sha256(work / f"process-{index}.json"))
    return execution


def require_report(
    report: dict, execution: dict, frozen: dict, root: Path, binding_hash: str, index: int, provenance: dict
) -> None:
    process = report["owner"]
    generation = execution["generation"]
    work = owner.locations(root)[3]
    if (
        report["status"] != worker.STATUS
        or report["worker_index"] != index
        or report["binding_sha256"] != binding_hash
        or process["binding"] != frozen
        or process["controller_pid"] != execution["controller_pid"]
        or process["controller_start_ticks"] != execution["controller_start_ticks"]
        or process["worker_pid"] != execution["pid"]
        or process["worker_start_ticks"] != execution["start_ticks"]
        or process["device"] != "cuda:0"
        or process["physical_uuid"] != frozen["devices"][0][1]
        or execution["command"] != owner.worker_command(root, binding_hash, index)
        or execution["index"] != index
        or execution["spawn_status"] != "acknowledged"
        or not execution["marker"]
        or generation["pid"] != execution["pid"]
        or generation["start_ticks"] != execution["start_ticks"]
        or generation["parent"] != execution["controller_pid"]
        or generation["command"] != execution["command"]
        or generation["group"] != generation["pid"]
        or generation["session"] != generation["pid"]
        or generation["status"] in ("Z", "X", "x")
        or any(type(execution[key]) is not int or execution[key] != 0 for key in ("exit_code", "cleanup_exit_code"))
        or execution["marked_descendants_absent"] is not True
        or execution["log"] != str(work / f"process-{index}.log")
        or execution["log_sha256"] != sha256(Path(execution["log"]))
        or execution["report"] != str(work / f"process-{index}.json")
        or report["outputs"] != str(work / f"process-{index}.pt")
        or report["report_path"] != str(work / f"process-{index}.json")
        or report["environment"] != frozen["worker_environment"]
        or report["provenance"] != provenance
        or report["runtime"] != provenance["d"]["config"]["runtime"]
        or report["tolerance"] != {"rtol": 1e-5, "atol": 1e-6}
        or report["helper_sources_sha256"] != frozen["helpers_sha256"]
        or report["physical_uuid"] != frozen["devices"][0][1]
        or UUID(report["actual_uuid"].removeprefix("GPU-")) != UUID(report["physical_uuid"].removeprefix("GPU-"))
        or report["compilation_preserved"] is not True
        or report["graphs_preserved"] is not True
        or report["lifecycle"]
        != [{**item, "graph_active": item["phase"] != "eager"} for item in worker.coordinates(index)]
    ):
        raise ValueError("Root replay report differs from acknowledged origin, source, GPU or lifecycle")
    for key, layouts in report["source_layouts"].items():
        protocol, group, mode = key.split(":")
        if protocol not in worker.PROTOCOLS:
            raise ValueError("Root replay source-layout protocol differs")
        raw = provenance[group]["trace_pair"][mode]["input_layouts"]
        if any(
            layouts[name][field] != value[field]
            for name, value in raw.items()
            for field in ("shape", "dtype", "stride", "storage_offset")
        ):
            raise ValueError("Root replay uploaded input layout differs from saved evidence")
    if set(report["source_layouts"]) != {
        f"{protocol}:{group}:{mode}"
        for protocol in worker.PROTOCOLS
        for group in worker.GROUPS
        for mode in ("continuous", "resumed")
    }:
        raise ValueError("Root replay source-layout coverage is incomplete")


def compare_results(root: Path, executions: list[dict]) -> dict:
    if (
        len(executions) != 2
        or len({(execution["pid"], execution["start_ticks"]) for execution in executions}) != 2
        or len({execution["marker"] for execution in executions}) != 2
        or len({(execution["controller_pid"], execution["controller_start_ticks"]) for execution in executions}) != 1
    ):
        raise ValueError("Root replay needs two distinct acknowledged worker generations from one controller")
    binding_hash = sha256(owner.locations(root)[0])
    frozen = owner.require_frozen(root, binding_hash)
    retained, reports, saved, provenance = [], [], {}, {}
    with ExitStack() as stack:
        for group in worker.GROUPS:
            saved[group], provenance[group] = replay.load_saved_roots(stack, group)
        for index, execution in enumerate(executions):
            report = replay.analyzer._json(
                stack.enter_context(replay.analyzer._bound_file(Path(execution["report"]), execution["report_sha256"]))
            )
            require_report(report, execution, frozen, root, binding_hash, index, provenance)
            stream = stack.enter_context(replay.analyzer._bound_file(Path(report["outputs"]), report["outputs_sha256"]))
            outputs = torch.load(f"/proc/self/fd/{stream.fileno()}", map_location="cpu", weights_only=True, mmap=True)
            analysis = worker.summarize(outputs, saved, index)
            if report["analysis"] != analysis:
                raise ValueError("Root replay reported metrics differ from independent retained-output reconstruction")
            retained.append(outputs)
            reports.append(report)
        indexed = [
            {tuple(item[key] for key in ("protocol", "group", "mode", "phase")): item for item in output}
            for output in retained
        ]
        cross_process = []
        for key in sorted(indexed[0]):
            first, second = indexed[0][key], indexed[1][key]
            record = {
                "coordinate": list(key),
                "output": tracing.comparison(second["output"], first["output"], f"cross.{'.'.join(key)}"),
                "stages": tracing.compare_root_stages(second["stages"], first["stages"]) if key[0] == "trace" else None,
            }
            cross_process.append(record)
    exact = all(worker._exact_finite(item["output"]) for item in cross_process)
    stages_exact = all(worker.stages_exact(item["stages"]) for item in cross_process if item["stages"] is not None)
    interpretable = (
        exact and stages_exact and all(report["analysis"]["intermediates_interpretable"] for report in reports)
    )
    return {
        "status": "root_stage_replay_reconstructed_not_continuation_or_production_gate",
        "executions": executions,
        "reports": reports,
        "cross_process": cross_process,
        "all_cross_process_outputs_exact": exact,
        "all_cross_process_stages_exact": stages_exact,
        "intermediates_interpretable": interpretable,
        "scope": (
            "Only full source-bound saved-input replay and transparency. Historical-output failures remain explicit; "
            "no model continuation acceptance."
        ),
    }


def execute(root: Path) -> dict:
    binding_path, helpers, state_path, work = owner.locations(root)
    binding_hash = sha256(binding_path)
    frozen = owner.require_frozen(root, binding_hash)
    intent = read(helpers.parent / "controller.json")
    deadline = time.monotonic() + 30
    while intent["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.01)
        intent = read(helpers.parent / "controller.json")
    process = process_context(
        os.getpid(),
        owner.controller_command(root),
        helpers,
        owner.CONTROLLER_MARKER,
        os.environ.get(owner.CONTROLLER_MARKER, ""),
        owner.ENVIRONMENT,
    )
    require_acknowledged_controller(intent, process)
    if intent["binding_sha256"] != binding_hash or local_node() != owner.NODE:
        raise RuntimeError("Root replay controller does not own this node and binding")
    if state_path.exists() or work.exists():
        raise FileExistsError("Root replay already has one-shot execution evidence")
    state = {
        "status": "preflight",
        "binding_sha256": binding_hash,
        "controller_pid": process["pid"],
        "controller_start_ticks": process["start_ticks"],
        "predecessor": frozen["predecessor"],
        "executions": [],
    }
    write(state_path, state)
    signals = []
    try:
        with node_guard(root, owner.NODE) as signals, keeper_guard() as manager_lock:
            idle_training()
            accepted = owner.require_live_predecessor(root, frozen["predecessor"])
            owner.require_frozen(root, binding_hash)
            work.mkdir(parents=True)
            state["lock_descriptors"] = {
                "node": lock_descriptor(root / "run-preparation" / f".wave7-node-{owner.NODE}.lock"),
                "manager": manager_lock,
            }
            try:
                release_accepted_keeper(accepted, state, state_path, manager_lock)
                state["status"] = "replaying"
                write(state_path, state)
                for index in range(2):
                    state["executions"].append(run_worker(root, frozen, state, index))
                    write(state_path, state)
                result = compare_results(root, state["executions"])
                write(work / "comparison.json", result)
                state.update(
                    status="diagnostic_complete",
                    comparison_sha256=sha256(work / "comparison.json"),
                    intermediates_interpretable=result["intermediates_interpretable"],
                )
            finally:
                if state.get("keeper_release_attempted"):
                    markers = [execution["marker"] for execution in state["executions"]]
                    if state.get("active_worker"):
                        markers.append(state["active_worker"]["marker"])
                    if (
                        state.get("keeper_release_complete") is not True
                        or "cleanup_error" in state
                        or any(owner.handoff.marker_processes(owner.CHILD_MARKER, marker) for marker in markers)
                    ):
                        state["keeper_restore"] = "refused: incomplete transfer or live descendants"
                        raise RuntimeError("Root replay cleanup does not permit keeper restoration")
                    restore_accepted_keeper(manager_lock)
                    restored = owner.handoff.keeper_observation()
                    require_keeper_shape(restored)
                    if restored["devices"] != accepted["devices"]:
                        raise RuntimeError("Root replay restored physical devices differ")
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
    """Reconstruct retained outputs and authenticate exact restoration without changing issued evidence."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Root replay terminal reconstruction requires hidden CUDA")
    if out.exists() or not out.parent.is_dir() or not out.is_relative_to(root / "run-preparation"):
        raise FileExistsError("Root replay terminal observation requires a new external receipt path")
    binding_path, helpers, state_path, work = owner.locations(root)
    paths = (
        binding_path,
        state_path,
        helpers.parent / "controller.json",
        helpers.parent / "controller.log",
        work / "comparison.json",
    )
    state, intent = read(state_path), read(helpers.parent / "controller.json")
    if (
        state.get("status") != "diagnostic_complete"
        or state.get("binding_sha256") != sha256(binding_path)
        or state.get("keeper_release_complete") is not True
        or type(state.get("keeper_restore")) is not int
        or state["keeper_restore"] != 0
        or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
        or intent["status"] != "acknowledged"
        or intent["command"] != owner.controller_command(root)
        or intent["binding_sha256"] != state["binding_sha256"]
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or state["controller_pid"] != intent["generation"]["pid"]
        or state["controller_start_ticks"] != intent["generation"]["start_ticks"]
        or intent["generation"]["command"] != intent["command"]
        or intent["generation"]["group"] != state["controller_pid"]
        or intent["generation"]["session"] != state["controller_pid"]
        or len(state["executions"]) != 2
        or state["active_worker"] != state["executions"][-1]
    ):
        raise RuntimeError("Root replay terminal reconstruction, owner or cleanup is incomplete")
    if owner.handoff.observed(state["controller_pid"]) is not None:
        raise RuntimeError("Root replay original controller handle remains present")
    for execution in state["executions"]:
        if (
            execution["controller_pid"] != state["controller_pid"]
            or execution["controller_start_ticks"] != state["controller_start_ticks"]
            or owner.handoff.observed(execution["pid"]) is not None
            or owner.handoff.marker_processes(owner.CHILD_MARKER, execution["marker"])
        ):
            raise RuntimeError("Root replay original worker handle or marked descendants remain present")
    if local_node() != owner.NODE:
        raise RuntimeError("Root replay terminal observation is outside the original allocation")
    frozen = owner.require_frozen(root, state["binding_sha256"])
    if state["predecessor"] != frozen["predecessor"]:
        raise ValueError("Root replay terminal allocation predecessor differs")
    paths += tuple(work / f"process-{index}{suffix}" for index in range(2) for suffix in (".log", ".json", ".pt"))
    evidence = {str(path.relative_to(root)): sha256(path) for path in paths}
    with ExitStack() as stack:
        streams = {
            name: stack.enter_context(replay.analyzer._bound_file(root / name, expected))
            for name, expected in evidence.items()
        }
        comparison = replay.analyzer._json(streams[str((work / "comparison.json").relative_to(root))])
        if (
            replay.analyzer._json(streams[str(state_path.relative_to(root))]) != state
            or replay.analyzer._json(streams[str((helpers.parent / "controller.json").relative_to(root))]) != intent
        ):
            raise ValueError("Root replay terminal state or acknowledgment changed during authentication")
        comparison_time, terminal_time = (datetime.fromisoformat(value["time"]) for value in (comparison, state))
        if (
            state["comparison_sha256"] != evidence[str((work / "comparison.json").relative_to(root))]
            or comparison_time.tzinfo is None
            or terminal_time.tzinfo is None
            or comparison_time > terminal_time
        ):
            raise ValueError("Root replay comparison bytes or terminal chronology differ")
        reconstructed = compare_results(root, state["executions"])
        if (
            set(comparison) != set(reconstructed) | {"time"}
            or reconstructed != {key: value for key, value in comparison.items() if key != "time"}
            or state["intermediates_interpretable"] is not reconstructed["intermediates_interpretable"]
        ):
            raise ValueError("Root replay terminal comparison differs from independent retained-output reconstruction")
        with keeper_guard():
            keeper = owner.handoff.keeper_observation()
            require_keeper_shape(keeper)
            marked = owner.handoff.marker_processes(owner.CONTROLLER_MARKER, intent["marker"])
            if (
                not same_keeper(keeper, state["restored_keeper"])
                or keeper["devices"] != frozen["devices"]
                or not set(marked).issubset({int(pid) for pid in keeper["generation"]})
            ):
                raise RuntimeError("Root replay exact restored keeper or inherited marker coverage differs")
        if evidence != {str(path.relative_to(root)): sha256(path) for path in paths}:
            raise ValueError("Root replay terminal evidence changed during reconstruction")
        result = {
            "status": "root_stage_replay_terminal_independently_reconstructed_exact_restoration_verified",
            "binding_sha256": state["binding_sha256"],
            "terminal_time": state["time"],
            "comparison_time": comparison["time"],
            "controller_pid_absent": state["controller_pid"],
            "worker_pids_absent": [execution["pid"] for execution in state["executions"]],
            "evidence_sha256": evidence,
            "verifier_sha256": sha256(Path(__file__)),
            "keeper": keeper,
            "comparison": reconstructed,
            "intermediates_interpretable": reconstructed["intermediates_interpretable"],
            "scope": (
                "Retained root-stage diagnostic and exact restoration only. Historical-output failures remain "
                "explicit; no continuation, optimizer quality, efficiency or production acceptance."
            ),
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
            parser.error("CPU verification requires --out")
        result = cpu(root, args.out.resolve())
    elif args.phase == "seal-predecessor":
        result = owner.failed_predecessor.seal(root)
    elif args.phase == "freeze":
        if args.cpu_receipt is None:
            parser.error("Freeze requires --cpu-receipt")
        result = freeze(root, args.cpu_receipt.resolve())
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
