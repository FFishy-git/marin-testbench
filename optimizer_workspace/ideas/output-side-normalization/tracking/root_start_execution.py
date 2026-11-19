# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute complete source-bound start trials inside a separately accepted CUDA owner."""

import importlib
import os
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import replay_saved_roots as replay
import root_stage_trace as tracing
import root_start_accuracy as accuracy
import root_start_rational as rational
import root_start_trial as trial
import torch
from benchmark_diagonal import source_definitions
from selected_lr_release_owner import sha256
from watch_wave5 import write

STATUS = "source_bound_start_accuracy_trial_not_continuation_or_production"
LIMITATION = (
    "Numerical worker component only; its separately accepted caller must authenticate an exclusive CUDA owner, "
    "original predecessor restoration and source archive. The caller must execute both opposite-order processes, "
    "independently reconstruct retained evidence and verify cleanup/restoration. This component transfers no GPU, "
    "changes no trainer and establishes no continuation, full-root accuracy, optimizer quality or efficiency."
)


def pinned_arguments(provenance: dict) -> tuple[Callable, float]:
    """Extract the executed raw start and freeze setting from bound upstream files."""
    roots = Path(provenance["pinned_roots"])
    if sha256(roots) != replay.ROOTS_SHA256:
        raise ValueError("Start execution raw pinned source differs")
    raw = source_definitions(
        roots,
        ("_start",),
        {"torch": torch, "EPS": 1e-3, "QUARTER": 0.25, "U32": 2.0**-24, "LOWP": torch.float16},
    )["_start"]
    arguments = source_definitions(
        roots.parents[3] / "reproduce/marin/optimizer.py",
        ("fast_arguments",),
        {"SimpleNamespace": SimpleNamespace},
    )["fast_arguments"](provenance["config"])
    return raw, arguments.toner_freeze


def retain_protocol(sources: dict, index: int, retain: Callable) -> tuple[list[dict], list[dict]]:
    """Retain every variant/group/mode phase while checking original source preservation.

    Args:
        sources: Complete CPU batches keyed by group and continuous/resumed mode.
        index: Fresh-process index selecting the declared opposite variant order.
        retain: Caller-owned function accepting a variant, full source tensor and
            coordinate. It returns all declared phase outputs plus execution evidence.
            CPU tests can supply a CPU function without creating any CUDA owner.

    Returns:
        Complete ordered CPU snapshots and one execution record per source call.
        This function authenticates no file, callable compilation or CUDA owner.
    """
    expected = accuracy.coordinates(index)
    if set(sources) != set(accuracy.GROUPS) or any(set(pair) != set(accuracy.MODES) for pair in sources.values()):
        raise ValueError("Start execution requires every full source group and mode")
    layouts, before = {}, {}
    for group, pair in sources.items():
        for mode, value in pair.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or value.ndim != 3
                or value.shape[-1] != value.shape[-2]
                or not value.numel()
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Start execution requires finite full FP32 CPU square batches")
            layouts[(group, mode)] = (value.shape, value.stride(), value.storage_offset(), value.data_ptr())
            before[(group, mode)] = value.clone(memory_format=torch.preserve_format)
    retained, executions = [], []
    for coordinate in expected[:: len(accuracy.PHASES)]:
        variant, group, mode = (coordinate[name] for name in ("variant", "group", "mode"))
        base = {"variant": variant, "group": group, "mode": mode}
        phases, evidence = retain(variant, sources[group][mode], base)
        if [item["phase"] for item in phases] != list(accuracy.PHASES):
            raise ValueError("Start execution did not retain the complete ordered graph phases")
        for phase in phases:
            accuracy.named_output(phase["output"], sources[group][mode].shape)
            retained.append({**base, "phase": phase["phase"], "output": tracing.cpu_tree(phase["output"])})
        executions.append({**base, "execution": evidence})
        for key, original in before.items():
            value = sources[key[0]][key[1]]
            if (value.shape, value.stride(), value.storage_offset(), value.data_ptr()) != layouts[
                key
            ] or not torch.equal(value, original):
                raise ValueError("Start execution mutated a saved source or its storage layout")
    return retained, executions


def run(
    helpers: Path,
    helper_hashes: dict[str, str],
    index: int,
    output_path: Path,
    report_path: Path,
    require_owner: Callable[[], dict],
) -> dict:
    """Retain a full CUDA start protocol after the caller authenticates exclusive ownership.

    The ownership callback must reject missing source/configuration/allocation
    bindings before any CUDA initialization. It must return the same acknowledged
    worker/controller generations throughout this call.
    """
    owner = require_owner()
    if (
        owner["device"] != "cuda:0"
        or owner["worker_pid"] != os.getpid()
        or not owner["physical_uuid"].startswith("GPU-")
        or output_path.exists()
        or report_path.exists()
        or output_path.parent != report_path.parent
        or not output_path.parent.is_dir()
        or output_path == report_path
    ):
        raise RuntimeError("Start execution requires canonical caller-owned identity and unused output paths")
    expected = accuracy.coordinates(index)
    if helpers != Path(__file__).resolve().parent or helpers != helpers.resolve(strict=True):
        raise ValueError("Start execution component is outside its accepted helper directory")
    with ExitStack() as stack:
        for name, expected_hash in helper_hashes.items():
            if Path(name).name != name:
                raise ValueError("Start execution helper inventory escapes its accepted directory")
            path = helpers / name
            stack.enter_context(replay.analyzer._bound_file(path, expected_hash))
            loaded = sys.modules.get(Path(name).stem)
            if loaded is not None and Path(loaded.__file__).resolve() != path:
                raise RuntimeError(f"Start execution imported a different helper: {name}")
        required = (
            "root_start_execution.py",
            "root_start_trial.py",
            "root_start_accuracy.py",
            "root_start_rational.py",
            "root_stage_trace.py",
            "replay_saved_roots.py",
        )
        if any(name not in helper_hashes for name in required):
            raise ValueError("Start execution lacks its complete numerical helper inventory")
        saved, provenance = {}, {}
        for group in accuracy.GROUPS:
            saved[group], provenance[group] = replay.load_saved_roots(stack, group)
        config = provenance["d"]["config"]
        if config != provenance["f"]["config"]:
            raise ValueError("Start execution source groups have different executed configurations")
        if any(name == "toner" or name.startswith("toner.") for name in sys.modules):
            raise RuntimeError("Start execution process already imported upstream numerical modules")
        pinned_path = Path(provenance["d"]["pinned_roots"])
        upstream = pinned_path.parents[3]
        sys.path.insert(0, str(upstream / "src"))
        roots = importlib.import_module("toner.fast.roots")
        if (
            Path(roots.__file__).resolve() != pinned_path
            or sha256(pinned_path) != replay.ROOTS_SHA256
            or roots.start._torchdynamo_orig_callable is not roots._start
        ):
            raise ValueError("Start execution imported a different pinned compiled start")
        raw_start, freeze = pinned_arguments(provenance["d"])
        if require_owner() != owner:
            raise RuntimeError("Start execution ownership changed before CUDA initialization")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(config["runtime"]["deterministic"])
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Start execution requires one explicitly UUID-bound CUDA device")
        actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if UUID(actual_uuid.removeprefix("GPU-")) != UUID(owner["physical_uuid"].removeprefix("GPU-")):
            raise RuntimeError("Start execution physical CUDA device differs")
        torch.cuda.set_device(0)
        kernels = {
            "pinned": roots.start,
            "rational": torch.compile(rational.rational_start, dynamic=False),
        }
        sources = {
            group: {mode: pair["saved_output"][1] for mode, pair in saved[group].items()} for group in accuracy.GROUPS
        }
        rng = {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(0).clone()}

        def retain(variant, value, coordinate):
            resident = value.to(device="cuda:0", copy=True, memory_format=torch.preserve_format)
            if resident.stride() != value.stride() or resident.storage_offset() != value.storage_offset():
                raise ValueError("Start execution staging changed the saved B layout")
            phases, lifecycle = trial.retain_kernel(resident, freeze, kernels[variant])
            lifecycle["source_shape"] = list(value.shape)
            lifecycle["source_stride"] = list(value.stride())
            lifecycle["source_storage_offset"] = value.storage_offset()
            return phases, lifecycle

        retained, lifecycle = retain_protocol(sources, index, retain)
        torch.cuda.synchronize()
        if not torch.equal(rng["cpu"], torch.get_rng_state()) or not torch.equal(
            rng["cuda"], torch.cuda.get_rng_state(0)
        ):
            raise ValueError("Start execution protocol changed CPU or CUDA RNG")
        analysis = accuracy.summarize(retained, sources, raw_start, freeze, index)
        if require_owner() != owner:
            raise RuntimeError("Start execution ownership changed during numerical work")
        with output_path.open("xb") as stream:
            torch.save(retained, stream)
        report = {
            "status": STATUS,
            "worker_index": index,
            "owner": owner,
            "outputs": str(output_path),
            "outputs_sha256": sha256(output_path),
            "report_path": str(report_path),
            "provenance": provenance,
            "runtime": config["runtime"],
            "physical_uuid": owner["physical_uuid"],
            "actual_uuid": actual_uuid,
            "coordinates": expected,
            "lifecycle": lifecycle,
            "helper_sources_sha256": helper_hashes,
            "freeze": freeze,
            "analysis": analysis,
            "rng_exact": True,
            "compilation_preserved": True,
            "graphs_preserved": True,
            "compile_options": {"dynamic": False},
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "scope": LIMITATION,
        }
        write(report_path, report)
    return report


def reconstruct(
    report_path: Path,
    report_hash: str,
    expected_owner: dict,
    helper_hashes: dict[str, str],
    index: int,
) -> tuple[dict, list[dict]]:
    """Reconstruct authenticated CPU snapshots without claiming owner liveness or cleanup.

    The caller supplies independently authenticated execution generations and
    helper identities. This function compares every saved arithmetic result with
    reconstruction. It does not inspect live processes, retry numerical work or
    accept source archive, CUDA ownership, cleanup or production.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Start reconstruction requires hidden CUDA")
    expected_coordinates = accuracy.coordinates(index)
    with ExitStack() as stack:
        report = replay.analyzer._json(stack.enter_context(replay.analyzer._bound_file(report_path, report_hash)))
        if (
            report["status"] != STATUS
            or report["worker_index"] != index
            or report["owner"] != expected_owner
            or report["report_path"] != str(report_path)
            or report["helper_sources_sha256"] != helper_hashes
            or report["coordinates"] != expected_coordinates
            or report["rng_exact"] is not True
            or report["compilation_preserved"] is not True
            or report["graphs_preserved"] is not True
            or report["compile_options"] != {"dynamic": False}
            or report["physical_uuid"] != expected_owner["physical_uuid"]
            or UUID(report["actual_uuid"].removeprefix("GPU-"))
            != UUID(expected_owner["physical_uuid"].removeprefix("GPU-"))
        ):
            raise ValueError("Start reconstruction report differs from declared execution/source/device")
        output_path = Path(report["outputs"])
        if output_path != report_path.with_suffix(".pt"):
            raise ValueError("Start reconstruction output path differs from its execution report")
        stream = stack.enter_context(replay.analyzer._bound_file(output_path, report["outputs_sha256"]))
        retained = torch.load(f"/proc/self/fd/{stream.fileno()}", map_location="cpu", weights_only=True, mmap=True)
        saved, provenance = {}, {}
        for group in accuracy.GROUPS:
            saved[group], provenance[group] = replay.load_saved_roots(stack, group)
        raw_start, freeze = pinned_arguments(provenance["d"])
        if (
            provenance != report["provenance"]
            or provenance["d"]["config"] != provenance["f"]["config"]
            or report["runtime"] != provenance["d"]["config"]["runtime"]
            or report["freeze"] != freeze
        ):
            raise ValueError("Start reconstruction source/configuration/data provenance differs")
        sources = {
            group: {mode: pair["saved_output"][1] for mode, pair in saved[group].items()} for group in accuracy.GROUPS
        }
        declared_calls = [
            {name: coordinate[name] for name in ("variant", "group", "mode")}
            for coordinate in expected_coordinates[:: len(accuracy.PHASES)]
        ]
        if [
            {name: item[name] for name in ("variant", "group", "mode")} for item in report["lifecycle"]
        ] != declared_calls:
            raise ValueError("Start reconstruction execution lifecycle lacks complete source calls")
        for record in report["lifecycle"]:
            value = sources[record["group"]][record["mode"]]
            lifecycle = record["execution"]
            preserved = lifecycle["input_exact_finite"]
            if (
                lifecycle["shape"] != list(value.shape)
                or lifecycle["stride"] != list(value.stride())
                or lifecycle["storage_offset"] != value.storage_offset()
                or lifecycle["source_shape"] != list(value.shape)
                or lifecycle["source_stride"] != list(value.stride())
                or lifecycle["source_storage_offset"] != value.storage_offset()
                or lifecycle["rng_exact"] is not True
                or lifecycle["compilation_preserved"] is not True
                or lifecycle["graphs_preserved"] is not True
                or lifecycle["captured_graph_replays"] != 3
                or not accuracy.exact_finite(preserved)
            ):
                raise ValueError("Start reconstruction input/layout/RNG/graph lifecycle differs")
        analysis = accuracy.summarize(retained, sources, raw_start, freeze, index)
        if analysis != report["analysis"]:
            raise ValueError("Start reconstruction arithmetic differs from authenticated retained outputs")
        independent = tracing.cpu_tree(retained)
    return {
        "status": "retained_start_outputs_independently_reconstructed_not_owner_or_production_acceptance",
        "worker_index": index,
        "report_sha256": report_hash,
        "outputs_sha256": report["outputs_sha256"],
        "owner": expected_owner,
        "analysis": analysis,
        "scope": LIMITATION,
    }, independent
