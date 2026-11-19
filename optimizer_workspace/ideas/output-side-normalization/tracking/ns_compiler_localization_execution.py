# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream source-bound compiler-localization traces inside a separately accepted CUDA owner."""

import json
import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from uuid import UUID

import ns_compiler_localization as localization
import replay_saved_ns as replay
import root_stage_trace as tracing
import torch
from ns_budget_execution import bind_helpers, saved_config, source_snapshots
from selected_lr_release_owner import sha256

STATUS = "source_bound_ns_compiler_localization_not_continuation_or_production"
REQUIRED_HELPERS = (
    "ns_compiler_localization_execution.py",
    "ns_compiler_localization.py",
    "ns_budget_execution.py",
    "ns_budget_trial.py",
    "ns_operand_precision.py",
    "replay_saved_ns.py",
    "root_stage_trace.py",
    "root_preparation_probe.py",
    "analyze_fused_continuation_stages.py",
    "diagnose_continuation.py",
    "benchmark_diagonal.py",
    "diagnose_fused_diagonal.py",
)
KEYS = ("mode", "steps", "boundary", "source", "call_index", "phase")
SCOPE = (
    "Source-bound saved rank0/update23/occurrence0 compiler-boundary diagnostic component. "
    "Each full phase is retained separately and reconstructed with bounded resident traces. "
    "Trace/control exact finite parity is mandatory before stage interpretation. Eager and "
    "full-compiled historical failures remain failures. A separately accepted exclusive owner, "
    "both original/actual-LR predecessor terminals, two fresh opposite-order processes and "
    "independent cleanup/restoration remain required. No standalone CUDA entry, continuation, "
    "causal upstream explanation, timing, production, optimizer superiority or SwanLab authority."
)


def coordinates(index: int) -> list[dict]:
    return [{**call, "phase": phase} for call in localization.call_coordinates(index) for phase in localization.PHASES]


def _key(row: dict) -> tuple:
    return tuple(row[name] for name in KEYS)


def require_phase(row: dict, groups: dict) -> None:
    """Require the complete CPU output and traced source batch before persistence or comparison."""
    if set(row) != set(KEYS) | {"control", "trace"}:
        raise ValueError("NS localization phase fields differ")
    replay._require_cpu_tensor(row["control"])
    localization.require_trace(row["trace"], row["steps"], row["mode"])
    source = groups[row["source"]]["source"]
    if row["control"].shape != source.shape or row["trace"]["output"].shape != source.shape:
        raise ValueError("NS localization phase omits part of the saved source batch")


def retain_files(groups: dict, index: int, directory: Path, retain: Callable) -> list[dict]:
    """Save one completed CPU phase at a time; reject source mutation or incomplete protocols."""
    snapshots, layouts = source_snapshots(groups)
    directory.mkdir()
    manifest = []
    for call in localization.call_coordinates(index):
        observed = []
        for phase in retain(call, groups[call["source"]]["source"]):
            if set(phase) != {"phase", "control", "trace"}:
                raise ValueError("NS localization phase lacks control or traced outputs")
            observed.append(phase["phase"])
            row = {**call, **phase}
            require_phase(row, groups)
            path = directory / f"phase-{len(manifest):03d}.pt"
            with path.open("xb") as stream:
                torch.save(row, stream)
                stream.flush()
                os.fsync(stream.fileno())
            manifest.append({**call, "phase": phase["phase"], "path": str(path), "sha256": sha256(path)})
        if observed != list(localization.PHASES):
            raise ValueError("NS localization phase sequence is missing, duplicated or reordered")
        for key, before in snapshots.items():
            value = groups[key[0]][key[1]]
            layout = (value.shape, value.stride(), value.storage_offset(), value.data_ptr())
            if layout != layouts[key] or not torch.equal(value, before):
                raise ValueError("NS localization changed original source/output values or storage")
    return manifest


class PhaseFiles:
    """Authenticate and load individual phases without caching the complete trace inventory."""

    def __init__(self, manifest: list[dict], directory: Path, groups: dict, index: int):
        expected = coordinates(index)
        if (
            directory != directory.resolve(strict=True)
            or any(set(row) != set(KEYS) | {"path", "sha256"} for row in manifest)
            or [{name: row[name] for name in KEYS} for row in manifest] != expected
            or any(row["path"] != str(directory / f"phase-{position:03d}.pt") for position, row in enumerate(manifest))
        ):
            raise ValueError("NS localization retained inventory or canonical phase paths differ")
        self.rows = {_key(row): row for row in manifest}
        self.groups = groups

    def load(self, coordinate: dict) -> dict:
        record = self.rows[_key(coordinate)]
        with replay.analyzer._bound_file(Path(record["path"]), record["sha256"]) as stream:
            row = torch.load(f"/proc/self/fd/{stream.fileno()}", map_location="cpu", weights_only=True, mmap=True)
            if {name: row[name] for name in KEYS} != {name: record[name] for name in KEYS}:
                raise ValueError("NS localization saved phase differs from its declared coordinate")
            require_phase(row, self.groups)
        return row


def _guard(actual: dict, expected: dict) -> dict:
    return localization.guarded_comparison(
        actual["control"],
        actual["trace"],
        expected["control"],
        expected["trace"],
        steps=actual["steps"],
        mode=actual["mode"],
    )


def summarize(manifest: list[dict], directory: Path, groups: dict, index: int) -> dict:
    """Stream all eager, historical, source-pair, phase and same-input comparisons from files."""
    source_snapshots(groups)
    files = PhaseFiles(manifest, directory, groups, index)
    comparisons, phases, repeats, pairs, historical = [], [], [], [], []
    for coordinate in coordinates(index):
        actual = files.load(coordinate)
        baseline = actual if coordinate["boundary"] == "eager" else files.load(coordinate | {"boundary": "eager"})
        comparisons.append(coordinate | _guard(actual, baseline))
        del baseline
        first = (
            actual
            if coordinate["phase"] == localization.PHASES[0]
            else files.load(coordinate | {"phase": localization.PHASES[0]})
        )
        phases.append(coordinate | {"control": replay.compare_outputs(actual["control"], first["control"])})
        del first
        historical.append(
            coordinate
            | {
                "own_saved_output": replay.compare_outputs(actual["control"], groups[coordinate["source"]]["output"]),
                "other_saved_output": replay.compare_outputs(
                    actual["control"],
                    groups["resumed" if coordinate["source"] == "continuous" else "continuous"]["output"],
                ),
            }
        )
        if coordinate["call_index"] >= 2:
            previous = files.load(coordinate | {"call_index": coordinate["call_index"] - 2})
            repeats.append(
                coordinate
                | {
                    "control": replay.compare_outputs(actual["control"], previous["control"]),
                    "trace": tracing.comparison(actual["trace"], previous["trace"], "ns.same_input"),
                }
            )
            del previous
        if coordinate["source"] == "resumed":
            continuous = files.load(
                coordinate
                | {
                    "source": "continuous",
                    "call_index": coordinate["call_index"] - 1 if index == 0 else coordinate["call_index"] + 1,
                }
            )
            pairs.append(coordinate | {"pair_index": coordinate["call_index"] // 2, **_guard(actual, continuous)})
            del continuous
        del actual
    original = [row for row in historical if row["mode"] == "tpu" and row["boundary"] == "eager"]
    compiled = [row for row in historical if row["mode"] == "tpu" and row["boundary"] == "full_compiled"]
    return {
        "comparisons_to_eager": comparisons,
        "same_input_repeats": repeats,
        "phases_vs_call": phases,
        "source_pairs": pairs,
        "historical_outputs": historical,
        "source_pair": replay.compare_outputs(groups["resumed"]["source"], groups["continuous"]["source"]),
        "all_traces_transparent_exact_finite": all(row["stage_interpretation_permitted"] for row in comparisons),
        "all_control_phases_exact_finite": all(row["control"]["exact"] and row["control"]["finite"] for row in phases),
        "all_same_input_repeats_exact_finite": all(
            row["control"]["exact"] and row["control"]["finite"] and row["trace"]["exact"] and row["trace"]["finite"]
            for row in repeats
        ),
        "all_tpu_eager_phases_reproduce_saved_outputs": all(
            row["own_saved_output"]["exact"] and row["own_saved_output"]["finite"] for row in original
        ),
        "all_tpu_full_compiled_phases_reproduce_saved_outputs": all(
            row["own_saved_output"]["exact"] and row["own_saved_output"]["finite"] for row in compiled
        ),
        "scope": SCOPE,
    }


def compare_processes(first: list[dict], second: list[dict], directories: tuple[Path, Path], groups: dict) -> dict:
    """Pair two opposite-order file inventories by source occurrence and preserve internal drift."""
    files = [PhaseFiles(rows, directories[index], groups, index) for index, rows in enumerate((first, second))]
    comparisons = []
    for coordinate in coordinates(0):
        other = coordinate | {"call_index": coordinate["call_index"] ^ 1}
        actual, expected = files[1].load(other), files[0].load(coordinate)
        comparisons.append(
            {name: coordinate[name] for name in KEYS if name != "call_index"}
            | {
                "occurrence": coordinate["call_index"] // 2,
                **_guard(actual, expected),
                "trace": tracing.comparison(actual["trace"], expected["trace"], "ns.cross_process"),
            }
        )
        del actual, expected
    return {
        "comparisons": comparisons,
        "all_cross_process_exact_finite": all(
            row["controls"]["exact"] and row["controls"]["finite"] and row["trace"]["exact"] and row["trace"]["finite"]
            for row in comparisons
        ),
        "all_cross_process_traces_transparent": all(row["stage_interpretation_permitted"] for row in comparisons),
        "scope": SCOPE,
    }


def retain_cuda_phases(source: torch.Tensor, control: Callable, traced: Callable, evidence: dict) -> Iterator[dict]:
    """Snapshot completed calls and three graph executions while retaining compilation and RNG."""
    if source.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("NS localization requires caller-owned resident CUDA outside capture")
    original = tracing.cpu_tree(source)
    layout = (source.shape, source.stride(), source.storage_offset(), source.data_ptr())
    rng = {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(source.device).clone()}
    current, side = torch.cuda.current_stream(source.device), torch.cuda.Stream(device=source.device)
    side.wait_stream(current)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        with torch.cuda.stream(side):
            called, observed = control(source), traced(source)
        side.synchronize()
        yield {"phase": "call", "control": tracing.cpu_tree(called), "trace": tracing.cpu_tree(observed)}
        del called, observed
        control_graph, trace_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(control_graph, stream=side):
            called = control(source)
        with torch.cuda.graph(trace_graph, stream=side):
            observed = traced(source)
        for phase in localization.PHASES[1:]:
            with torch.cuda.stream(side):
                control_graph.replay()
                trace_graph.replay()
            side.synchronize()
            yield {"phase": phase, "control": tracing.cpu_tree(called), "trace": tracing.cpu_tree(observed)}
    current.wait_stream(side)
    torch.cuda.synchronize(source.device)
    unchanged = tracing.comparison(tracing.cpu_tree(source), original, "ns.source")
    if (
        layout != (source.shape, source.stride(), source.storage_offset(), source.data_ptr())
        or not unchanged["exact"]
        or not unchanged["finite"]
        or not torch.equal(rng["cpu"], torch.get_rng_state())
        or not torch.equal(rng["cuda"], torch.cuda.get_rng_state(source.device))
    ):
        raise ValueError("NS localization changed resident source storage, values or CPU/CUDA RNG")
    evidence.update(
        source_shape=list(source.shape),
        source_stride=list(source.stride()),
        source_storage_offset=source.storage_offset(),
        input_exact_finite=unchanged,
        rng_exact=True,
        graphs_preserved=True,
        graph_replays=3,
    )


def require_lifecycle(lifecycle: list[dict], groups: dict, index: int) -> None:
    """Check every declared call's unchanged full-source layout and graph/RNG evidence."""
    calls = localization.call_coordinates(index)
    if [{name: row[name] for name in calls[0]} for row in lifecycle] != calls:
        raise ValueError("NS localization lifecycle omits or reorders a declared call")
    for row in lifecycle:
        source, observed = groups[row["source"]]["source"], row["execution"]
        preserved = observed["input_exact_finite"]
        if (
            observed["source_shape"] != list(source.shape)
            or observed["source_stride"] != list(source.stride())
            or observed["source_storage_offset"] != source.storage_offset()
            or observed["rng_exact"] is not True
            or observed["graphs_preserved"] is not True
            or observed["graph_replays"] != 3
            or preserved["tensor_comparisons"] != 1
            or preserved["exact"] is not True
            or preserved["finite"] is not True
            or preserved["changed_tensors"]
            or preserved["scalar_or_structure_differences"]
        ):
            raise ValueError("NS localization lifecycle source/layout/RNG/graph evidence differs")


def _bind(stack: ExitStack, helpers: Path, hashes: dict) -> None:
    if any(name not in hashes for name in REQUIRED_HELPERS):
        raise ValueError("NS localization execution lacks its accepted numerical helper inventory")
    if helpers != Path(__file__).resolve().parent:
        raise ValueError("NS localization execution is outside its accepted helper directory")
    bind_helpers(stack, helpers, hashes)


def run(helpers: Path, hashes: dict, index: int, report_path: Path, require_owner: Callable) -> dict:
    """Execute only after a separate source-accepted caller authenticates exclusive CUDA ownership."""
    owner = require_owner()
    directory = report_path.with_suffix(".phases")
    if (
        owner["device"] != "cuda:0"
        or owner["worker_pid"] != os.getpid()
        or not owner["physical_uuid"].startswith("GPU-")
        or report_path.exists()
        or directory.exists()
        or not report_path.parent.is_dir()
        or report_path != report_path.resolve()
        or os.environ.get("CUDA_VISIBLE_DEVICES") != owner["physical_uuid"]
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or os.environ.get("NVIDIA_TF32_OVERRIDE") == "1"
    ):
        raise RuntimeError("NS localization requires caller-owned UUID-bound CUDA and unused canonical paths")
    coordinates(index)
    with ExitStack() as stack:
        _bind(stack, helpers, hashes)
        groups, provenance = replay.load_saved_group(stack)
        source_snapshots(groups)
        if any(tuple(pair["source"].shape) != replay.SAVED_SHAPE for pair in groups.values()):
            raise ValueError("NS localization execution requires the original full saved source batches")
        config = saved_config(stack, provenance)
        if require_owner() != owner:
            raise RuntimeError("NS localization ownership changed before CUDA initialization")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(config["runtime"]["deterministic"])
        if torch.cuda.device_count() != 1:
            raise RuntimeError("NS localization requires one explicitly UUID-bound CUDA device")
        actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if UUID(actual_uuid.removeprefix("GPU-")) != UUID(owner["physical_uuid"].removeprefix("GPU-")):
            raise RuntimeError("NS localization actual physical CUDA device differs")
        torch.cuda.set_device(0)
        resident = {
            source: pair["source"].to(device="cuda:0", copy=True, memory_format=torch.preserve_format)
            for source, pair in groups.items()
        }
        for source, value in resident.items():
            original = groups[source]["source"]
            if value.stride() != original.stride() or value.storage_offset() != original.storage_offset():
                raise ValueError("NS localization upload changed saved source layout")
        rng = {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(0).clone()}
        maps = {
            (mode, steps): localization.compiler_maps(
                Path(provenance["pinned_muon"]),
                lambda fn: torch.compile(fn, dynamic=False),
                mode=mode,
                steps=steps,
                eps=replay.NS_EPS,
            )
            for mode, steps in localization.ARMS
        }
        lifecycle = []

        def retain(call, source):
            if require_owner() != owner:
                raise RuntimeError("NS localization ownership changed between calls")
            numerical, execution = maps[call["mode"], call["steps"]], {}
            yield from retain_cuda_phases(
                resident[call["source"]],
                numerical.controls[call["boundary"]],
                numerical.traces[call["boundary"]],
                execution,
            )
            for name, value in resident.items():
                if not torch.equal(tracing.cpu_tree(value), groups[name]["source"]):
                    raise ValueError("NS localization changed another resident input")
            lifecycle.append({**call, "execution": execution})

        manifest = retain_files(groups, index, directory, retain)
        if not (
            torch.equal(rng["cpu"], torch.get_rng_state()) and torch.equal(rng["cuda"], torch.cuda.get_rng_state(0))
        ):
            raise ValueError("NS localization complete protocol changed CPU or CUDA RNG")
        require_lifecycle(lifecycle, groups, index)
        analysis = summarize(manifest, directory, groups, index)
        if require_owner() != owner:
            raise RuntimeError("NS localization ownership changed during execution")
        report = {
            "status": STATUS,
            "worker_index": index,
            "owner": owner,
            "report_path": str(report_path),
            "phase_directory": str(directory),
            "phase_files": manifest,
            "provenance": provenance,
            "runtime": config["runtime"],
            "helper_sources_sha256": hashes,
            "physical_uuid": owner["physical_uuid"],
            "actual_uuid": actual_uuid,
            "coordinates": coordinates(index),
            "lifecycle": lifecycle,
            "analysis": analysis,
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
    """Independently rederive authenticated phase-file arithmetic with CUDA hidden."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("NS localization reconstruction requires hidden CUDA")
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
            raise ValueError("NS localization retained report execution/source/device identity differs")
        groups, provenance = replay.load_saved_group(stack)
        config = saved_config(stack, provenance)
        if report["provenance"] != provenance or report["runtime"] != config["runtime"]:
            raise ValueError("NS localization retained source/configuration/data provenance differs")
        require_lifecycle(report["lifecycle"], groups, index)
        analysis = summarize(report["phase_files"], Path(report["phase_directory"]), groups, index)
        if report["analysis"] != analysis:
            raise ValueError("NS localization arithmetic differs from independently reconstructed phase files")
        for row in report["phase_files"]:
            with replay.analyzer._bound_file(Path(row["path"]), row["sha256"]):
                pass
    return {
        "status": "retained_ns_compiler_localization_reconstructed_not_owner_or_production_acceptance",
        "worker_index": index,
        "report_sha256": report_hash,
        "owner": expected_owner,
        "phase_files": report["phase_files"],
        "analysis": analysis,
        "scope": SCOPE,
    }
