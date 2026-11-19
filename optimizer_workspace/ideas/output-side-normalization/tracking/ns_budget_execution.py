# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-bound NS budget execution and reconstruction inside a separately accepted CUDA owner."""

import json
import os
import sys
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from uuid import UUID

import ns_budget_trial as trial
import ns_operand_precision as precision
import replay_saved_ns as replay
import root_stage_trace as tracing
import torch
from benchmark_diagonal import source_definitions
from diagnose_fused_diagonal import differences
from selected_lr_release_owner import sha256

ARMS = (("tpu", 10), ("tpu", 7), ("tpu", 5), ("f32", 10), ("f32", 7), ("f32", 5))
PHASES = ("pinned_eager", *trial.PHASES)
STATUS = "source_bound_ns_budget_trial_not_continuation_or_production"
REQUIRED_HELPERS = (
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
LIMITATION = (
    "Saved rank0/update23/occurrence0 NS maps only. The caller must authenticate exclusive CUDA ownership, "
    "source/archive acceptance, original predecessor restoration and both opposite-order fresh processes. "
    "Retained reconstruction does not prove live ownership or cleanup. CUDA-event replay timing excludes "
    "compilation, capture, transfers and training. Changed budgets are distinct numerical arms; no "
    "continuation, optimizer quality, whole-training efficiency, SwanLab or production acceptance."
)


def coordinates(index: int) -> list[dict]:
    """Declare both complete opposite arm/input orders with explicit repeated-call identities."""
    if index not in (0, 1):
        raise ValueError("NS budget protocol requires one of its two fresh-process indices")
    arms = ARMS if index == 0 else ARMS[::-1]
    order = replay.ORDER if index == 0 else tuple(reversed(replay.ORDER))
    return [
        {"mode": mode, "steps": steps, "source": source, "call_index": call_index, "phase": phase}
        for mode, steps in arms
        for call_index, source in enumerate(order)
        for phase in PHASES
    ]


def source_snapshots(groups: dict) -> tuple[dict, dict]:
    """Check complete CPU source/output pairs and retain their values and concrete storage layouts."""
    if set(groups) != set(replay.analyzer.MODES) or any(set(pair) != {"source", "output"} for pair in groups.values()):
        raise ValueError("NS budget execution requires complete continuous/resumed source/output pairs")
    snapshots, layouts = {}, {}
    shape = groups["continuous"]["source"].shape
    for mode, pair in groups.items():
        for field, value in pair.items():
            replay._require_cpu_tensor(value)
            if value.ndim != 3 or value.shape != shape or not torch.isfinite(value).all():
                raise ValueError("NS budget execution requires one complete finite CPU batch shape")
            key = mode, field
            snapshots[key] = value.clone(memory_format=torch.preserve_format)
            layouts[key] = (value.shape, value.stride(), value.storage_offset(), value.data_ptr())
    return snapshots, layouts


def retain_protocol(groups: dict, index: int, retain: Callable) -> tuple[list[dict], list[dict]]:
    """Retain independent CPU phases for every arm and input; the callback owns numerical execution."""
    snapshots, layouts = source_snapshots(groups)
    retained, executions = [], []
    for coordinate in coordinates(index)[:: len(PHASES)]:
        call = {key: coordinate[key] for key in ("mode", "steps", "source", "call_index")}
        source = groups[call["source"]]["source"]
        phases, evidence = retain(call["mode"], call["steps"], source, call)
        if [item["phase"] for item in phases] != list(PHASES):
            raise ValueError("NS budget execution lacks its complete eager/compiled/graph phases")
        for record in phases:
            output = record["output"]
            replay._require_cpu_tensor(output)
            if output.shape != source.shape:
                raise ValueError("NS budget execution retained a partial or differently shaped output")
            retained.append({**call, "phase": record["phase"], "output": tracing.cpu_tree(output)})
        executions.append({**call, "execution": tracing.cpu_tree(evidence)})
        for key, before in snapshots.items():
            value = groups[key[0]][key[1]]
            if (value.shape, value.stride(), value.storage_offset(), value.data_ptr()) != layouts[
                key
            ] or not torch.equal(value, before):
                raise ValueError("NS budget execution changed saved input/output values or storage")
    return retained, executions


def output_inventory(retained: list[dict], groups: dict, index: int) -> dict:
    """Reject missing, duplicated or reordered coordinates before pairing retained outputs."""
    expected = coordinates(index)
    if [{key: item[key] for key in expected[0]} for item in retained] != expected:
        raise ValueError("NS budget retained evidence differs from its complete opposite-order protocol")
    source_snapshots(groups)
    outputs = {}
    for item in retained:
        output = item["output"]
        replay._require_cpu_tensor(output)
        if output.shape != groups[item["source"]]["source"].shape:
            raise ValueError("NS budget retained output is not the complete saved source shape")
        outputs[(item["mode"], item["steps"], item["call_index"], item["phase"])] = output
    return outputs


def summarize(retained: list[dict], groups: dict, coefficients: tuple, index: int) -> dict:
    """Recompute all repeats, historical controls and per-matrix FP64 finite-map comparisons."""
    outputs = output_inventory(retained, groups, index)
    order = replay.ORDER if index == 0 else tuple(reversed(replay.ORDER))
    references = {
        (source, steps): precision.reference_fp64(pair["source"], coefficients, steps=steps, eps=replay.NS_EPS)
        for source, pair in groups.items()
        for steps in (5, 7, 10)
    }
    arms, historical, compiled_parity, graph_repeats, input_repeats = [], [], [], [], []
    for mode, steps in ARMS:
        calls, repeats = [], []
        for call_index, source in enumerate(order):
            eager = outputs[(mode, steps, call_index, PHASES[0])]
            compiled = outputs[(mode, steps, call_index, trial.PHASES[0])]
            phases = []
            for phase in PHASES:
                actual = outputs[(mode, steps, call_index, phase)]
                own = replay.compare_outputs(actual, groups[source]["output"])
                value = {
                    "phase": phase,
                    "vs_pinned_eager": replay.compare_outputs(actual, eager),
                    "vs_compiled_call": replay.compare_outputs(actual, compiled),
                    "vs_saved_gpu_output": own,
                    "vs_same_mode_ten_steps": replay.compare_outputs(actual, outputs[(mode, 10, call_index, phase)]),
                    "vs_fp64_finite_map": differences(actual, references[(source, steps)], replay.MOMENT_TOLERANCE),
                }
                phases.append(value)
                if mode == "tpu" and steps == 10:
                    historical.append(own)
                if phase in trial.PHASES:
                    compiled_parity.append(value["vs_pinned_eager"])
                if phase in trial.PHASES[1:]:
                    graph_repeats.append(value["vs_compiled_call"])
            calls.append({"call_index": call_index, "source": source, "phases": phases})
        for call_index in (2, 3):
            for phase in PHASES:
                metric = replay.compare_outputs(
                    outputs[(mode, steps, call_index, phase)], outputs[(mode, steps, call_index - 2, phase)]
                )
                repeats.append({"source": order[call_index], "phase": phase, **metric})
                input_repeats.append(metric)
        pairs = []
        for beginning in (0, 2):
            continuous = beginning + order[beginning : beginning + 2].index("continuous")
            resumed = beginning + order[beginning : beginning + 2].index("resumed")
            pairs.extend(
                {
                    "pair_index": beginning // 2,
                    "phase": phase,
                    **replay.compare_outputs(
                        outputs[(mode, steps, resumed, phase)], outputs[(mode, steps, continuous, phase)]
                    ),
                }
                for phase in PHASES
            )
        arms.append(
            {
                "mode": mode,
                "steps": steps,
                "nominal_products": 3 * steps,
                "calls": calls,
                "same_input_repeats": repeats,
                "source_pairs": pairs,
            }
        )
    return {
        "arms": arms,
        "source_pair": replay.compare_outputs(groups["resumed"]["source"], groups["continuous"]["source"]),
        "all_ten_step_tpu_phases_reproduce_saved_outputs": all(
            value["exact"] and value["finite"] for value in historical
        ),
        "all_compiled_phases_match_pinned_eager": all(value["exact"] and value["finite"] for value in compiled_parity),
        "all_graph_phases_match_compiled_call": all(value["exact"] and value["finite"] for value in graph_repeats),
        "all_same_input_repeats_exact_finite": all(value["exact"] and value["finite"] for value in input_repeats),
        "fp64_reference": (
            "Per-matrix independent FP64 normalization and finite quintic recurrence; no exact polar target."
        ),
        "scope": LIMITATION,
    }


def reconstruct_lifecycle(retained: list[dict], executions: list[dict], groups: dict, index: int) -> list[dict]:
    """Recompute every phase/timing summary and require the declared full input/RNG lifecycle."""
    outputs = output_inventory(retained, groups, index)
    calls = [
        {key: item[key] for key in ("mode", "steps", "source", "call_index")}
        for item in coordinates(index)[:: len(PHASES)]
    ]
    if [{key: item[key] for key in calls[0]} for item in executions] != calls:
        raise ValueError("NS budget lifecycle omits or reorders a declared full source call")
    summaries = []
    for call, record in zip(calls, executions, strict=True):
        source, execution = groups[call["source"]]["source"], record["execution"]
        preserved = execution["input_exact_finite"]
        if (
            execution["source_shape"] != list(source.shape)
            or execution["source_stride"] != list(source.stride())
            or execution["source_storage_offset"] != source.storage_offset()
            or execution["rng_exact"] is not True
            or execution["compilation_preserved"] is not True
            or execution["graphs_preserved"] is not True
            or execution["graph_replays"] != 3 + trial.TIMING_REPLAYS
            or preserved["tensor_comparisons"] != 1
            or preserved["exact"] is not True
            or preserved["finite"] is not True
            or preserved["changed_tensors"]
            or preserved["scalar_or_structure_differences"]
        ):
            raise ValueError("NS budget lifecycle input/layout/RNG/graph evidence differs")
        phases = [
            {"phase": phase, "output": outputs[(call["mode"], call["steps"], call["call_index"], phase)]}
            for phase in trial.PHASES
        ]
        rebuilt = trial.summarize_kernel(phases, execution["cuda_event_milliseconds"])
        if any(execution[key] != value for key, value in rebuilt.items()):
            raise ValueError("NS budget lifecycle phase or timing summary differs from retained evidence")
        summaries.append({**call, **tracing.cpu_tree(rebuilt)})
    return summaries


def compare_processes(first: list[dict], second: list[dict], groups: dict) -> dict:
    """Pair opposite-order processes by arm/source/occurrence/phase rather than list position."""
    inventories = [output_inventory(values, groups, index) for index, values in enumerate((first, second))]
    pairs = []
    for mode, steps in ARMS:
        for source in replay.analyzer.MODES:
            for occurrence in (0, 1):
                for phase in PHASES:
                    indices = [
                        [
                            item["call_index"]
                            for item in coordinates(index)[:: len(PHASES)]
                            if item["mode"] == mode and item["steps"] == steps and item["source"] == source
                        ][occurrence]
                        for index in (0, 1)
                    ]
                    metric = replay.compare_outputs(
                        inventories[1][(mode, steps, indices[1], phase)],
                        inventories[0][(mode, steps, indices[0], phase)],
                    )
                    pairs.append(
                        {
                            "mode": mode,
                            "steps": steps,
                            "source": source,
                            "occurrence": occurrence,
                            "phase": phase,
                            **metric,
                        }
                    )
    return {
        "comparisons": pairs,
        "all_cross_process_exact_finite": all(value["exact"] and value["finite"] for value in pairs),
        "scope": LIMITATION,
    }


def bind_helpers(stack: ExitStack, helpers: Path, hashes: dict) -> None:
    """Hold complete accepted helper inodes open and reject imports from another source directory."""
    if helpers != Path(__file__).resolve().parent or helpers != helpers.resolve(strict=True):
        raise ValueError("NS budget execution is outside its accepted helper directory")
    if any(name not in hashes for name in REQUIRED_HELPERS):
        raise ValueError("NS budget execution lacks its numerical helper inventory")
    for name, digest in hashes.items():
        if Path(name).name != name:
            raise ValueError("NS budget helper inventory escapes its accepted directory")
        path = helpers / name
        stack.enter_context(replay.analyzer._bound_file(path, digest))
        loaded = sys.modules.get(Path(name).stem)
        if loaded is not None and Path(loaded.__file__).resolve() != path:
            raise RuntimeError(f"NS budget execution imported another helper: {name}")


def pinned_definitions(stack: ExitStack, provenance: dict) -> dict:
    stream = stack.enter_context(
        replay.analyzer._bound_file(Path(provenance["pinned_muon"]), provenance["pinned_muon_sha256"])
    )
    return source_definitions(
        Path(f"/proc/self/fd/{stream.fileno()}"), ("ABC", "_bmm", "newton_schulz_batched"), {"torch": torch}
    )


def saved_config(stack: ExitStack, provenance: dict) -> dict:
    receipt = replay.analyzer._json(
        stack.enter_context(
            replay.analyzer._bound_file(replay.ROOT / replay.analyzer.RECEIPT_RELATIVE, replay.RECEIPT_SHA256)
        )
    )
    return replay.analyzer._json(
        stack.enter_context(replay.analyzer._bound_file(Path(receipt["config"]), provenance["config_sha256"]))
    )


def run(helpers: Path, hashes: dict, index: int, output_path: Path, report_path: Path, require_owner: Callable) -> dict:
    """Execute the complete protocol only after a separate caller authenticates its exclusive CUDA owner."""
    owner = require_owner()
    if (
        owner["device"] != "cuda:0"
        or owner["worker_pid"] != os.getpid()
        or not owner["physical_uuid"].startswith("GPU-")
        or output_path.exists()
        or report_path.exists()
        or output_path != report_path.with_suffix(".pt")
        or not report_path.parent.is_dir()
    ):
        raise RuntimeError("NS budget execution requires caller-owned CUDA and unused canonical output paths")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != owner["physical_uuid"]
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or os.environ.get("NVIDIA_TF32_OVERRIDE") == "1"
    ):
        raise RuntimeError("NS budget execution requires its UUID-bound deterministic CUDA environment")
    coordinates(index)
    with ExitStack() as stack:
        bind_helpers(stack, helpers, hashes)
        groups, provenance = replay.load_saved_group(stack)
        source_snapshots(groups)
        if any(tuple(pair["source"].shape) != replay.SAVED_SHAPE for pair in groups.values()):
            raise ValueError("NS budget CUDA execution requires the original full saved source batches")
        definitions = pinned_definitions(stack, provenance)
        config = saved_config(stack, provenance)
        if require_owner() != owner:
            raise RuntimeError("NS budget ownership changed before CUDA initialization")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(config["runtime"]["deterministic"])
        if torch.cuda.device_count() != 1:
            raise RuntimeError("NS budget execution requires one explicitly UUID-bound CUDA device")
        actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if UUID(actual_uuid.removeprefix("GPU-")) != UUID(owner["physical_uuid"].removeprefix("GPU-")):
            raise RuntimeError("NS budget execution actual physical device differs")
        torch.cuda.set_device(0)
        resident = {
            source: pair["source"].to(device="cuda:0", copy=True, memory_format=torch.preserve_format)
            for source, pair in groups.items()
        }
        for source, value in resident.items():
            original = groups[source]["source"]
            if value.stride() != original.stride() or value.storage_offset() != original.storage_offset():
                raise ValueError("NS budget upload changed the full saved source layout")
        raw = {
            arm: partial(definitions["newton_schulz_batched"], steps=arm[1], eps=replay.NS_EPS, mode=arm[0])
            for arm in ARMS
        }
        compiled = {arm: torch.compile(numerical, dynamic=False) for arm, numerical in raw.items()}
        rng = {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(0).clone()}

        def retain(mode, steps, source, coordinate):
            value = resident[coordinate["source"]]
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                eager = raw[(mode, steps)](value)
            torch.cuda.synchronize()
            phases = [{"phase": PHASES[0], "output": tracing.cpu_tree(eager)}]
            snapshots, execution = trial.retain_kernel(value, compiled[(mode, steps)])
            for name, uploaded in resident.items():
                if not torch.equal(tracing.cpu_tree(uploaded), groups[name]["source"]):
                    raise ValueError("NS budget execution changed another resident input")
            execution.update(
                source_shape=list(source.shape),
                source_stride=list(source.stride()),
                source_storage_offset=source.storage_offset(),
            )
            return phases + snapshots, execution

        retained, executions = retain_protocol(groups, index, retain)
        torch.cuda.synchronize()
        if not (
            torch.equal(rng["cpu"], torch.get_rng_state()) and torch.equal(rng["cuda"], torch.cuda.get_rng_state(0))
        ):
            raise ValueError("NS budget complete protocol changed CPU or CUDA RNG")
        analysis = summarize(retained, groups, definitions["ABC"], index)
        timings = reconstruct_lifecycle(retained, executions, groups, index)
        if require_owner() != owner:
            raise RuntimeError("NS budget ownership changed during numerical execution")
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
            "helper_sources_sha256": hashes,
            "coordinates": coordinates(index),
            "lifecycle": executions,
            "analysis": analysis,
            "timings": timings,
            "rng_exact": True,
            "compilation_preserved": True,
            "graphs_preserved": True,
            "compile_options": {"dynamic": False},
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "scope": LIMITATION,
        }
        with report_path.open("x") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
    return report


def reconstruct(
    report_path: Path, report_hash: str, expected_owner: dict, hashes: dict, index: int
) -> tuple[dict, list]:
    """Recompute authenticated retained arithmetic and event summaries with CUDA hidden."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("NS budget retained reconstruction requires hidden CUDA")
    with ExitStack() as stack:
        bind_helpers(stack, Path(__file__).resolve().parent, hashes)
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
            or Path(report["outputs"]) != report_path.with_suffix(".pt")
        ):
            raise ValueError("NS budget retained report differs from declared execution/source/device")
        groups, provenance = replay.load_saved_group(stack)
        definitions = pinned_definitions(stack, provenance)
        config = saved_config(stack, provenance)
        if report["provenance"] != provenance or report["runtime"] != config["runtime"]:
            raise ValueError("NS budget retained source/configuration/data provenance differs")
        stream = stack.enter_context(replay.analyzer._bound_file(Path(report["outputs"]), report["outputs_sha256"]))
        retained = torch.load(f"/proc/self/fd/{stream.fileno()}", map_location="cpu", weights_only=True, mmap=True)
        analysis = summarize(retained, groups, definitions["ABC"], index)
        timings = reconstruct_lifecycle(retained, report["lifecycle"], groups, index)
        if report["analysis"] != analysis or report["timings"] != timings:
            raise ValueError("NS budget retained arithmetic or timing differs from report")
        independent = tracing.cpu_tree(retained)
    return {
        "status": "retained_ns_budget_outputs_reconstructed_not_owner_or_production_acceptance",
        "worker_index": index,
        "report_sha256": report_hash,
        "outputs_sha256": report["outputs_sha256"],
        "owner": expected_owner,
        "analysis": analysis,
        "timings": timings,
        "scope": LIMITATION,
    }, independent
