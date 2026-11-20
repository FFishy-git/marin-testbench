# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain precision-placement CUDA trials inside a separately authenticated exclusive owner."""

import ast
import json
import os
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from uuid import UUID

import ns_budget_trial as trial
import ns_precision_placement as placement
import ns_precision_placement_maps as maps
import replay_saved_ns as replay
import root_stage_trace as tracing
import torch
from diagnose_fused_diagonal import differences
from ns_budget_execution import bind_helpers, saved_config, source_snapshots
from selected_lr_release_owner import sha256

PHASES = ("pinned_eager", *trial.PHASES)
KEYS = ("schedule", "source", "call_index", "phase")
STATUS = "source_bound_ns_precision_placement_not_continuation_or_production"
RANKING = placement.RANKING
REQUIRED_HELPERS = (
    "ns_precision_placement_execution.py",
    "ns_precision_placement_maps.py",
    "ns_precision_placement.py",
    "ns_compiler_localization.py",
    "ns_budget_execution.py",
    "ns_budget_trial.py",
    "ns_operand_precision.py",
    "ns_step_budget.py",
    "replay_saved_ns.py",
    "root_stage_trace.py",
    "root_preparation_probe.py",
    "analyze_fused_continuation_stages.py",
    "diagnose_continuation.py",
    "benchmark_diagonal.py",
    "diagnose_fused_diagonal.py",
    "selected_lr_release_owner.py",
)
EXTERNAL_MODULES = sys.stdlib_module_names | {"torch", "triton", "numpy"}
SCOPE = (
    "Saved rank0/update23/occurrence0 full-batch CUDA precision-placement diagnostic component. "
    "Five schedules in two fresh opposite-order processes retain eager and compiled controls, "
    "graph phases and raw CUDA-event samples. Higher precision changes finite arithmetic. "
    "Historical eager/compiled and C/R failures stay failed. Independent file reconstruction "
    "does not establish exclusive live ownership, cleanup, restoration, continuation, "
    "optimizer quality, whole-training efficiency, SwanLab or production acceptance. "
    "A separately accepted owner and completed original/actual-LR predecessors are required."
)


def coordinates(index: int) -> list[dict]:
    if index not in (0, 1):
        raise ValueError("Precision placement requires two distinct opposite-order process indices")
    names = list(maps.SCHEDULES)
    order = replay.ORDER
    if index:
        names, order = names[::-1], tuple(reversed(order))
    return [
        {"schedule": name, "source": source, "call_index": call_index, "phase": phase}
        for name in names
        for call_index, source in enumerate(order)
        for phase in PHASES
    ]


def calls(index: int) -> list[dict]:
    return [
        {name: value for name, value in row.items() if name != "phase"} for row in coordinates(index)[:: len(PHASES)]
    ]


def retain_files(groups: dict, index: int, directory: Path, retain: Callable) -> tuple[list, list]:
    """Persist each independent phase before callback buffers can be reused."""
    snapshots, layouts = source_snapshots(groups)
    directory.mkdir()
    manifest, lifecycle = [], []
    for call in calls(index):
        phases, execution = retain(call, groups[call["source"]]["source"])
        if [row["phase"] for row in phases] != list(PHASES):
            raise ValueError("Precision placement eager/compiled/graph phase sequence differs")
        for phase in phases:
            if set(phase) != {"phase", "output"}:
                raise ValueError("Precision placement retained phase fields differ")
            output = phase["output"]
            replay._require_cpu_tensor(output)
            if output.ndim != 3 or output.shape != groups[call["source"]]["source"].shape:
                raise ValueError("Precision placement phase omits part of the saved batch")
            path = directory / f"phase-{len(manifest):03d}.pt"
            row = {**call, **phase}
            with path.open("xb") as stream:
                torch.save(row, stream)
                stream.flush()
                os.fsync(stream.fileno())
            manifest.append({**call, "phase": phase["phase"], "path": str(path), "sha256": sha256(path)})
        lifecycle.append({**call, "execution": tracing.cpu_tree(execution)})
        for key, before in snapshots.items():
            value = groups[key[0]][key[1]]
            if (value.shape, value.stride(), value.storage_offset(), value.data_ptr()) != layouts[
                key
            ] or not torch.equal(value, before):
                raise ValueError("Precision placement changed source/output values or storage")
    return manifest, lifecycle


class PhaseFiles:
    """Authenticate one phase at a time without retaining the complete output inventory."""

    def __init__(self, manifest: list, directory: Path, groups: dict, index: int):
        if (
            directory != directory.resolve(strict=True)
            or any(set(row) != set(KEYS) | {"path", "sha256"} for row in manifest)
            or [{key: row[key] for key in KEYS} for row in manifest] != coordinates(index)
            or any(row["path"] != str(directory / f"phase-{number:03d}.pt") for number, row in enumerate(manifest))
        ):
            raise ValueError("Precision placement phase inventory or canonical paths differ")
        self.rows = {tuple(row[key] for key in KEYS): row for row in manifest}
        self.groups = groups

    def load(self, coordinate: dict) -> torch.Tensor:
        entry = self.rows[tuple(coordinate[key] for key in KEYS)]
        with replay.analyzer._bound_file(Path(entry["path"]), entry["sha256"]) as stream:
            row = torch.load(f"/proc/self/fd/{stream.fileno()}", map_location="cpu", weights_only=True, mmap=True)
            if set(row) != set(KEYS) | {"output"} or any(row[key] != coordinate[key] for key in KEYS):
                raise ValueError("Precision placement retained coordinate differs")
            output = row["output"]
            replay._require_cpu_tensor(output)
            if output.ndim != 3 or output.shape != self.groups[coordinate["source"]]["source"].shape:
                raise ValueError("Precision placement retained output is incomplete")
            output = output.clone(memory_format=torch.preserve_format)
        return output


def summarize(manifest: list, directory: Path, groups: dict, references: dict, index: int) -> dict:
    """Recompute controls, repeated inputs, source pairs and independent finite-map errors."""
    source_snapshots(groups)
    files = PhaseFiles(manifest, directory, groups, index)
    comparisons, pairs, repeats = [], [], []
    for coordinate in coordinates(index):
        actual = files.load(coordinate)
        eager = files.load(coordinate | {"phase": PHASES[0]})
        compiled = files.load(coordinate | {"phase": trial.PHASES[0]})
        source, schedule = coordinate["source"], maps.SCHEDULES[coordinate["schedule"]]
        comparisons.append(
            coordinate
            | {
                "vs_pinned_eager": replay.compare_outputs(actual, eager),
                "vs_compiled_call": replay.compare_outputs(actual, compiled),
                "own_saved_output": replay.compare_outputs(actual, groups[source]["output"]),
                "other_saved_output": replay.compare_outputs(
                    actual, groups["resumed" if source == "continuous" else "continuous"]["output"]
                ),
                "vs_fp64_finite_polynomial": differences(
                    actual, references[source][str(len(schedule))], replay.MOMENT_TOLERANCE
                ),
            }
        )
        if coordinate["call_index"] >= 2:
            previous = files.load(coordinate | {"call_index": coordinate["call_index"] - 2})
            repeats.append(coordinate | replay.compare_outputs(actual, previous))
        if source == "resumed":
            continuous = files.load(
                coordinate
                | {
                    "source": "continuous",
                    "call_index": coordinate["call_index"] + (1 if index else -1),
                }
            )
            pairs.append(coordinate | replay.compare_outputs(actual, continuous))
    historical = [row for row in comparisons if row["schedule"] == "q10_tpu" and row["phase"] == "pinned_eager"]
    compiled_rows = [row for row in comparisons if row["phase"] in trial.PHASES]
    graph_rows = [row for row in comparisons if row["phase"] in trial.PHASES[1:]]
    return {
        "comparisons": comparisons,
        "source_pairs": pairs,
        "same_input_repeats": repeats,
        "source_pair": replay.compare_outputs(groups["resumed"]["source"], groups["continuous"]["source"]),
        "all_ten_step_tpu_eager_reproduce_saved_outputs": all(
            row["own_saved_output"]["exact"] and row["own_saved_output"]["finite"] for row in historical
        ),
        "all_compiled_phases_match_eager_exact_finite": all(
            row["vs_pinned_eager"]["exact"] and row["vs_pinned_eager"]["finite"] for row in compiled_rows
        ),
        "all_graph_phases_match_compiled_exact_finite": all(
            row["vs_compiled_call"]["exact"] and row["vs_compiled_call"]["finite"] for row in graph_rows
        ),
        "all_same_input_repeats_exact_finite": all(row["exact"] and row["finite"] for row in repeats),
        "all_source_pairs_within_tolerance": all(row["within_tolerance"] for row in pairs),
        "scope": SCOPE,
    }


def reconstruct_lifecycle(manifest: list, directory: Path, lifecycle: list, groups: dict, index: int) -> list:
    """Recompute event statistics and reject source/layout/RNG or timed-final drift evidence changes."""
    files = PhaseFiles(manifest, directory, groups, index)
    if [{key: row[key] for key in calls(index)[0]} for row in lifecycle] != calls(index):
        raise ValueError("Precision placement lifecycle call coverage differs")
    timings = []
    for record in lifecycle:
        call = {key: record[key] for key in calls(index)[0]}
        source, observed = groups[call["source"]]["source"], record["execution"]
        unchanged = observed["input_exact_finite"]
        if (
            observed["source_shape"] != list(source.shape)
            or observed["source_stride"] != list(source.stride())
            or observed["source_storage_offset"] != source.storage_offset()
            or observed["rng_exact"] is not True
            or observed["compilation_preserved"] is not True
            or observed["graphs_preserved"] is not True
            or observed["graph_replays"] != 3 + trial.TIMING_REPLAYS
            or unchanged["tensor_comparisons"] != 1
            or unchanged["exact"] is not True
            or unchanged["finite"] is not True
            or unchanged["changed_tensors"]
            or unchanged["scalar_or_structure_differences"]
        ):
            raise ValueError("Precision placement lifecycle source/layout/RNG/graph identity differs")
        retained = [{"phase": phase, "output": files.load(call | {"phase": phase})} for phase in trial.PHASES]
        rebuilt = trial.summarize_kernel(retained, observed["cuda_event_milliseconds"])
        if any(observed[key] != value for key, value in rebuilt.items()):
            raise ValueError("Precision placement retained phase or event statistics differ")
        timings.append(call | rebuilt)
    return timings


def compare_processes(manifests: tuple, directories: tuple, groups: dict) -> dict:
    files = [PhaseFiles(manifests[index], directories[index], groups, index) for index in (0, 1)]
    comparisons = []
    for coordinate in coordinates(0):
        actual = files[1].load(coordinate | {"call_index": coordinate["call_index"] ^ 1})
        expected = files[0].load(coordinate)
        comparisons.append(
            {key: coordinate[key] for key in KEYS if key != "call_index"}
            | {"occurrence": coordinate["call_index"] // 2, **replay.compare_outputs(actual, expected)}
        )
    return {
        "comparisons": comparisons,
        "all_cross_process_exact_finite": all(row["exact"] and row["finite"] for row in comparisons),
        "scope": SCOPE,
    }


def _bind(stack: ExitStack, helpers: Path, hashes: dict) -> None:
    if helpers != Path(__file__).resolve().parent or any(name not in hashes for name in REQUIRED_HELPERS):
        raise ValueError("Precision placement execution lacks its complete accepted numerical closure")
    bind_helpers(stack, helpers, hashes)
    pending, inspected = list(REQUIRED_HELPERS), set()
    while pending:
        name = pending.pop()
        if name in inspected:
            continue
        inspected.add(name)
        if name not in hashes:
            raise ValueError("Precision placement accepted local import closure is incomplete")
        with replay.analyzer._bound_file(helpers / name, hashes[name]) as stream:
            module = ast.parse(stream.read(), filename=str(helpers / name))
        for node in ast.walk(module):
            relative = isinstance(node, ast.ImportFrom) and node.level
            if relative and node.level != 1:
                raise ValueError("Precision placement helper closure cannot import parent-relative sources")
            imports = (
                [entry.name for entry in node.names]
                if isinstance(node, ast.Import)
                else (
                    ([node.module] if node.module else [entry.name for entry in node.names])
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
            )
            for imported in imports:
                dependency = imported.split(".")[0]
                if relative or dependency not in EXTERNAL_MODULES:
                    pending.append(f"{dependency}.py")


def run(helpers: Path, hashes: dict, index: int, report_path: Path, require_owner: Callable) -> dict:
    """Execute only through a caller that reauthenticates exclusive physical-device ownership."""
    owner = require_owner()
    directory = report_path.with_suffix(".phases")
    if (
        owner["device"] != "cuda:0"
        or owner["worker_pid"] != os.getpid()
        or not owner["physical_uuid"].startswith("GPU-")
        or os.environ.get("CUDA_VISIBLE_DEVICES") != owner["physical_uuid"]
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or os.environ.get("NVIDIA_TF32_OVERRIDE") == "1"
        or report_path != report_path.resolve()
        or report_path.exists()
        or directory.exists()
        or not report_path.parent.is_dir()
    ):
        raise RuntimeError("Precision placement requires exclusive UUID-bound CUDA and unused canonical outputs")
    coordinates(index)
    with ExitStack() as stack:
        _bind(stack, helpers, hashes)
        groups, provenance = replay.load_saved_group(stack)
        source_snapshots(groups)
        if any(tuple(pair["source"].shape) != replay.SAVED_SHAPE for pair in groups.values()):
            raise ValueError("Precision placement CUDA needs the full original saved source batches")
        references, reference_provenance = placement.spectral_reference(stack)
        config = saved_config(stack, provenance)
        if require_owner() != owner:
            raise RuntimeError("Precision placement owner changed before CUDA initialization")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(config["runtime"]["deterministic"])
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Precision placement needs one UUID-bound CUDA device")
        actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if UUID(actual_uuid.removeprefix("GPU-")) != UUID(owner["physical_uuid"].removeprefix("GPU-")):
            raise RuntimeError("Precision placement physical CUDA device differs")
        torch.cuda.set_device(0)
        resident = {
            source: pair["source"].to(device="cuda:0", copy=True, memory_format=torch.preserve_format)
            for source, pair in groups.items()
        }
        for source, value in resident.items():
            if (
                value.stride() != groups[source]["source"].stride()
                or value.storage_offset() != groups[source]["source"].storage_offset()
            ):
                raise ValueError("Precision placement upload changed source layout")
        numerical = {
            name: maps.maps(Path(provenance["pinned_muon"]), name, lambda fn: torch.compile(fn, dynamic=False))
            for name in maps.SCHEDULES
        }
        rng = {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(0).clone()}

        def retain(call, source):
            if require_owner() != owner:
                raise RuntimeError("Precision placement owner changed between numerical calls")
            value, arm = resident[call["source"]], numerical[call["schedule"]]
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                eager = arm.eager(value)
            torch.cuda.synchronize()
            phases = [{"phase": PHASES[0], "output": tracing.cpu_tree(eager)}]
            compiled_phases, execution = trial.retain_kernel(value, arm.compiled)
            for name, uploaded in resident.items():
                if not torch.equal(tracing.cpu_tree(uploaded), groups[name]["source"]):
                    raise ValueError("Precision placement changed another resident source")
            execution.update(
                source_shape=list(source.shape),
                source_stride=list(source.stride()),
                source_storage_offset=source.storage_offset(),
            )
            return phases + compiled_phases, execution

        manifest, lifecycle = retain_files(groups, index, directory, retain)
        if not (
            torch.equal(rng["cpu"], torch.get_rng_state()) and torch.equal(rng["cuda"], torch.cuda.get_rng_state(0))
        ):
            raise ValueError("Precision placement complete protocol changed CPU/CUDA RNG")
        analysis = summarize(manifest, directory, groups, references, index)
        timings = reconstruct_lifecycle(manifest, directory, lifecycle, groups, index)
        if require_owner() != owner:
            raise RuntimeError("Precision placement owner changed during retained reconstruction")
        report = {
            "status": STATUS,
            "worker_index": index,
            "owner": owner,
            "report_path": str(report_path),
            "phase_directory": str(directory),
            "phase_files": manifest,
            "coordinates": coordinates(index),
            "schedules": {name: list(schedule) for name, schedule in maps.SCHEDULES.items()},
            "provenance": provenance,
            "reference_provenance": reference_provenance,
            "runtime": config["runtime"],
            "helper_sources_sha256": hashes,
            "physical_uuid": owner["physical_uuid"],
            "actual_uuid": actual_uuid,
            "lifecycle": lifecycle,
            "analysis": analysis,
            "timings": timings,
            "rng_exact": True,
            "compilation_preserved": True,
            "graphs_preserved": True,
            "compile_options": {"dynamic": False},
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "scope": SCOPE,
        }
        with report_path.open("x") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return report


def reconstruct(report_path: Path, report_hash: str, expected_owner: dict, hashes: dict, index: int) -> dict:
    """Independently rederive all reported file-backed comparisons and timing with CUDA hidden."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Precision placement reconstruction requires hidden CUDA")
    with ExitStack() as stack:
        _bind(stack, Path(__file__).resolve().parent, hashes)
        report = replay.analyzer._json(stack.enter_context(replay.analyzer._bound_file(report_path, report_hash)))
        if (
            report["status"] != STATUS
            or report["worker_index"] != index
            or report["owner"] != expected_owner
            or report["report_path"] != str(report_path)
            or report["helper_sources_sha256"] != hashes
            or report["coordinates"] != coordinates(index)
            or report["schedules"] != {name: list(schedule) for name, schedule in maps.SCHEDULES.items()}
            or report["rng_exact"] is not True
            or report["compilation_preserved"] is not True
            or report["graphs_preserved"] is not True
            or report["compile_options"] != {"dynamic": False}
            or report["physical_uuid"] != expected_owner["physical_uuid"]
            or UUID(report["actual_uuid"].removeprefix("GPU-"))
            != UUID(expected_owner["physical_uuid"].removeprefix("GPU-"))
            or report["phase_directory"] != str(report_path.with_suffix(".phases"))
            or report["scope"] != SCOPE
        ):
            raise ValueError("Precision placement retained report source/device/execution identity differs")
        groups, provenance = replay.load_saved_group(stack)
        references, reference_provenance = placement.spectral_reference(stack)
        config = saved_config(stack, provenance)
        if (
            report["provenance"] != provenance
            or report["reference_provenance"] != reference_provenance
            or report["runtime"] != config["runtime"]
        ):
            raise ValueError("Precision placement retained source/configuration/reference provenance differs")
        directory = Path(report["phase_directory"])
        analysis = summarize(report["phase_files"], directory, groups, references, index)
        timings = reconstruct_lifecycle(report["phase_files"], directory, report["lifecycle"], groups, index)
        if report["analysis"] != analysis or report["timings"] != timings:
            raise ValueError("Precision placement retained arithmetic or timing differs from report")
    return {
        "status": "retained_ns_precision_placement_reconstructed_not_owner_or_production_acceptance",
        "worker_index": index,
        "report_sha256": report_hash,
        "owner": expected_owner,
        "phase_files": report["phase_files"],
        "analysis": analysis,
        "timings": timings,
        "scope": SCOPE,
    }
