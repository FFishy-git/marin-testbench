# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain source-bound eager/capture/replay root stages under a separate accepted owner."""

import argparse
import importlib
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import replay_saved_roots as replay
import root_stage_gpu_owner as ownership
import root_stage_trace as tracing
import torch
from benchmark_diagonal import source_definitions
from saved_ns_gpu_worker import tensor_layout
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

STATUS = "owned_root_stage_cuda_replay_not_continuation_or_production_gate"
PROTOCOLS = ("control", "trace")
GROUPS = ("d", "f")


def coordinates(index: int) -> list[dict]:
    protocols = PROTOCOLS if index == 0 else tuple(reversed(PROTOCOLS))
    order = (
        ("continuous", "eager"),
        ("continuous", "capture"),
        ("resumed", "eager"),
        ("resumed", "capture"),
        ("continuous", "replay"),
        ("resumed", "replay"),
    )
    return [
        {"protocol": protocol, "group": group, "mode": mode, "phase": phase}
        for protocol in protocols
        for group in GROUPS
        for mode, phase in order
    ]


def _exact_finite(value: dict) -> bool:
    return value["tensor_comparisons"] > 0 and value["exact"] and value["finite"]


def stages_exact(value: dict) -> bool:
    return (
        value["complete_stage_coverage"]
        and _exact_finite(value["output"])
        and all(_exact_finite(stage) for stage in value["stages"])
    )


def stage_sequence(width: int) -> tuple[str, ...]:
    """Complete call sequence of the pinned two-iteration 1/8-and-1/2 split roots."""
    product = "mm_sym" if width % 256 == 0 else "_bmm32"
    iteration = (product, "iter_mid", "_bmm32", "iter_w", product, "iter_post")
    return (
        "basis_transform",
        "start",
        *iteration,
        *iteration,
        "sqrt_start",
        product,
        "sqrt_post",
        product,
        "sqrt_post",
        "_formation",
        product,
        "square",
        "_formation",
    )


def summarize(retained: list[dict], saved: dict, index: int) -> dict:
    """Reconstruct transparency, safety and historical metrics from retained CPU evidence."""
    if [{key: item[key] for key in ("protocol", "group", "mode", "phase")} for item in retained] != coordinates(index):
        raise ValueError("Root replay retained protocol/group/mode/phase coverage differs")
    indexed = {}
    for item in retained:
        output = item["output"]
        self_comparison = tracing.comparison(output, output, "root.self")
        if self_comparison["tensor_comparisons"] != 7:
            raise ValueError("Root replay requires all seven candidate output tensors")
        replay._require_cpu_tree(output)
        key = tuple(item[name] for name in ("protocol", "group", "mode", "phase"))
        indexed[key] = item
        if item["protocol"] == "trace":
            replay._require_cpu_tree(item["stages"])
            checked = tracing.compare_root_stages(item["stages"], item["stages"])
            if not checked["complete_stage_coverage"] or checked["tensor_comparisons"] == 0:
                raise ValueError("Root replay instrumented output lacks ordered tensor stages")
            width = saved[item["group"]][item["mode"]]["saved_output"][1].shape[-1]
            if tuple(record["stage"] for record in item["stages"]["records"]) != stage_sequence(width):
                raise ValueError("Root replay lacks the complete pinned split-root stage sequence")
            if not tracing.comparison(item["stages"]["output"], output, "stage.output")["exact"]:
                raise ValueError("Root replay stage output differs from its retained candidate")
        elif item.get("stages") is not None:
            raise ValueError("Root replay control must remain uninstrumented")
    transparency, safety, historical, repeats, crossed = [], [], [], [], []
    for group in GROUPS:
        for mode in ("continuous", "resumed"):
            for phase in ("eager", "capture", "replay"):
                observed = indexed[("trace", group, mode, phase)]["output"]
                control = indexed[("control", group, mode, phase)]["output"]
                transparency.append(
                    {
                        "group": group,
                        "mode": mode,
                        "phase": phase,
                        **tracing.comparison(observed, control, f"{group}.{mode}.{phase}.trace.control"),
                    }
                )
                for protocol in PROTOCOLS:
                    candidate = indexed[(protocol, group, mode, phase)]["output"]
                    historical.append(
                        {
                            "protocol": protocol,
                            "group": group,
                            "mode": mode,
                            "phase": phase,
                            **tracing.comparison(
                                candidate, saved[group][mode]["saved_output"], f"{protocol}.{group}.{mode}.{phase}.saved"
                            ),
                        }
                    )
            for protocol in PROTOCOLS:
                eager = indexed[(protocol, group, mode, "eager")]["output"]
                capture = indexed[(protocol, group, mode, "capture")]["output"]
                repeated = indexed[(protocol, group, mode, "replay")]["output"]
                for phase, candidate, reference in (
                    ("capture/eager", capture, eager),
                    ("replay/eager", repeated, eager),
                    ("replay/capture", repeated, capture),
                ):
                    safety.append(
                        {
                            "protocol": protocol,
                            "group": group,
                            "mode": mode,
                            "comparison": phase,
                            **tracing.comparison(candidate, reference, f"{protocol}.{group}.{mode}.{phase}"),
                        }
                    )
            repeats.append(
                {
                    "group": group,
                    "mode": mode,
                    **tracing.compare_root_stages(
                        indexed[("trace", group, mode, "replay")]["stages"],
                        indexed[("trace", group, mode, "capture")]["stages"],
                    ),
                }
            )
        for phase in ("eager", "capture", "replay"):
            crossed.append(
                {
                    "group": group,
                    "phase": phase,
                    **tracing.compare_root_stages(
                        indexed[("trace", group, "resumed", phase)]["stages"],
                        indexed[("trace", group, "continuous", phase)]["stages"],
                    ),
                }
            )
    trace_finite = all(
        all(stage["finite"] for stage in tracing.compare_root_stages(item["stages"], item["stages"])["stages"])
        for item in retained
        if item["protocol"] == "trace"
    )
    return {
        "schema": 1,
        "coordinates": coordinates(index),
        "trace_transparency": transparency,
        "capture_repeat_safety": safety,
        "historical_output_comparisons": historical,
        "repeated_stages": repeats,
        "crossed_stages": crossed,
        "trace_transparency_exact": all(_exact_finite(value) for value in transparency),
        "capture_repeat_safety_exact": all(_exact_finite(value) for value in safety),
        "capture_repeat_stages_exact": all(stages_exact(value) for value in repeats),
        "trace_stages_finite": trace_finite,
        "historical_outputs_within_tolerance": all(value["within_tolerance"] for value in historical),
        "intermediates_interpretable": (
            trace_finite
            and all(_exact_finite(value) for value in transparency)
            and all(_exact_finite(value) for value in safety)
            and all(stages_exact(value) for value in repeats)
        ),
        "limitation": tracing.LIMITATION,
    }


def stage_group(values: dict, device: str) -> tuple[SimpleNamespace, dict]:
    resident = {
        name: value.to(device=device, copy=True, memory_format=torch.preserve_format) for name, value in values.items()
    }
    if any(
        value.stride() != values[name].stride() or value.storage_offset() != values[name].storage_offset()
        for name, value in resident.items()
    ):
        raise ValueError("Root replay staging changed the saved raw input layout")
    group = SimpleNamespace(
        C=resident["C"],
        tol=resident["tol"],
        basis=SimpleNamespace(SB=resident["SB"], S2=resident["S2"]),
        warm=False,
        graph=None,
        per=resident["C"].shape[0],
    )
    return group, {name: tensor_layout(value) for name, value in resident.items()}


def group_inputs(group: SimpleNamespace) -> dict:
    return {"C": group.C, "tol": group.tol, "SB": group.basis.SB, "S2": group.basis.S2}


def require_group_unchanged(group: SimpleNamespace, source: dict, layouts: dict) -> None:
    values = group_inputs(group)
    if any(tensor_layout(value) != layouts[name] for name, value in values.items()):
        raise ValueError("Root replay resident input storage or layout changed")
    compared = tracing.comparison(tracing.cpu_tree(values), source, "resident.input")
    if not _exact_finite(compared):
        raise ValueError("Root replay changed resident input values")


def run(root: Path, binding_hash: str, state_path: Path, index: int) -> dict:
    """Authenticate ownership before CUDA, then retain unchanged pinned root trajectories."""
    owner = ownership.require_gpu_owner(root, binding_hash, state_path, index)
    binding_path, helpers, canonical_state, work = ownership.locations(root)
    if state_path != canonical_state or Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Root replay worker is outside its frozen execution directory")
    frozen = owner["binding"]
    report_path, tensor_path = work / f"process-{index}.json", work / f"process-{index}.pt"
    if any(path.exists() for path in (report_path, tensor_path)):
        raise FileExistsError("Root replay output already exists; preserve one-shot evidence")
    with ExitStack() as stack:
        checked = replay.analyzer._json(stack.enter_context(replay.analyzer._bound_file(binding_path, binding_hash)))
        if checked != frozen:
            raise ValueError("Root replay binding differs from acknowledged ownership")
        for name, expected in frozen["helpers_sha256"].items():
            stack.enter_context(replay.analyzer._bound_file(helpers / name, expected))
        saved, provenance = {}, {}
        for group in GROUPS:
            saved[group], provenance[group] = replay.load_saved_roots(stack, group)
        config = provenance["d"]["config"]
        if config != provenance["f"]["config"] or any(
            os.environ.get(key) != value for key, value in frozen["worker_environment"].items()
        ):
            raise ValueError("Root replay configuration or UUID-bound environment differs")
        historical_receipt = read(root / replay.analyzer.RECEIPT_RELATIVE)
        candidate = Path(historical_receipt["candidate"])
        upstream = Path(provenance["d"]["pinned_roots"]).parents[3]
        for module in tuple(sys.modules.values()):
            name = getattr(module, "__name__", "")
            if name == "toner" or name.startswith("toner."):
                raise RuntimeError("Root replay process already imported an upstream numerical module")
        sys.path.insert(0, str(upstream / "src"))
        roots = importlib.import_module("toner.fast.roots")
        engine_type = importlib.import_module("toner.fast.engine").FastTonerCore
        if Path(roots.__file__).resolve() != Path(provenance["d"]["pinned_roots"]):
            raise ValueError("Root replay imported a different numerical root source")
        arguments_path = upstream / "reproduce/marin/optimizer.py"
        arguments_stream = stack.enter_context(
            replay.analyzer._bound_file(
                arguments_path,
                provenance["d"]["identity"]["source_sha256"][
                    str(
                        (
                            candidate / "artifacts/external-sources" / upstream.name / "reproduce/marin/optimizer.py"
                        ).relative_to(candidate)
                    )
                ],
            )
        )
        args = source_definitions(
            Path(f"/proc/self/fd/{arguments_stream.fileno()}"), ("fast_arguments",), {"SimpleNamespace": SimpleNamespace}
        )["fast_arguments"](config)
        if ownership.require_gpu_owner(root, binding_hash, state_path, index) != owner:
            raise RuntimeError("Root replay ownership changed before CUDA initialization")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(config["runtime"]["deterministic"])
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Root replay requires one explicitly UUID-bound CUDA device")
        properties = torch.cuda.get_device_properties(0)
        actual_uuid = str(properties.uuid)
        if UUID(actual_uuid.removeprefix("GPU-")) != UUID(owner["physical_uuid"].removeprefix("GPU-")):
            raise RuntimeError("Root replay physical CUDA device differs")
        torch.cuda.set_device(0)
        roots.set_powers((args.toner_alpha_in, args.toner_alpha_out))
        retained, layout_evidence, lifecycle = [], {}, []
        protocols = PROTOCOLS if index == 0 else tuple(reversed(PROTOCOLS))
        for protocol in protocols:
            for name in GROUPS:
                engine = engine_type.__new__(engine_type)
                engine.args, engine.split, engine.p_in, engine.p_out = args, True, 0.5, 0.125
                engine.froot, engine.iters = roots, args.toner_iters
                engine.side, engine.graph_pool = torch.cuda.Stream(), torch.cuda.graph_pool_handle()
                residents, captured = {}, {}
                for mode in ("continuous", "resumed"):
                    group, layouts = stage_group(saved[name][mode]["inputs"], "cuda:0")
                    engine.side.wait_stream(torch.cuda.current_stream())
                    residents[mode] = group
                    layout_evidence[f"{protocol}:{name}:{mode}"] = layouts
                    for phase in ("eager", "capture"):
                        group.warm = phase == "capture"
                        context = tracing.observe_root_stages(roots) if protocol == "trace" else ExitStack()
                        with context as trace, torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                            with torch.cuda.stream(engine.side):
                                output = engine._roots(group) if phase == "eager" else engine._roots_graphed(group)
                            engine.side.synchronize()
                            copied = tracing.cpu_tree(output)
                            stages = trace.snapshot(output) if protocol == "trace" else None
                        if phase == "capture" and protocol == "trace":
                            captured[mode] = trace
                        if phase == "capture" and group.graph is None:
                            raise RuntimeError("Root replay did not capture its existing compiled graph path")
                        retained.append(
                            {
                                "protocol": protocol,
                                "group": name,
                                "mode": mode,
                                "phase": phase,
                                "output": copied,
                                "stages": stages,
                            }
                        )
                        lifecycle.append(
                            {
                                "protocol": protocol,
                                "group": name,
                                "mode": mode,
                                "phase": phase,
                                "graph_active": group.graph is not None,
                            }
                        )
                        require_group_unchanged(group, saved[name][mode]["inputs"], layouts)
                for mode in ("continuous", "resumed"):
                    group = residents[mode]
                    with (
                        torch.no_grad(),
                        torch.autocast(device_type="cuda", enabled=False),
                        torch.cuda.stream(engine.side),
                    ):
                        output = engine._roots_graphed(group)
                    engine.side.synchronize()
                    retained.append(
                        {
                            "protocol": protocol,
                            "group": name,
                            "mode": mode,
                            "phase": "replay",
                            "output": tracing.cpu_tree(output),
                            "stages": captured[mode].snapshot(output) if protocol == "trace" else None,
                        }
                    )
                    lifecycle.append(
                        {
                            "protocol": protocol,
                            "group": name,
                            "mode": mode,
                            "phase": "replay",
                            "graph_active": group.graph is not None,
                        }
                    )
                    require_group_unchanged(
                        group, saved[name][mode]["inputs"], layout_evidence[f"{protocol}:{name}:{mode}"]
                    )
                del residents, captured, engine, group, output, trace, context
                torch.cuda.synchronize()
        analysis = summarize(retained, saved, index)
        if ownership.require_gpu_owner(root, binding_hash, state_path, index) != owner:
            raise RuntimeError("Root replay ownership changed during numerical execution")
        with tensor_path.open("xb") as stream:
            torch.save(retained, stream)
        report = {
            "status": STATUS,
            "worker_index": index,
            "binding_sha256": binding_hash,
            "owner": owner,
            "outputs": str(tensor_path),
            "outputs_sha256": sha256(tensor_path),
            "report_path": str(report_path),
            "provenance": provenance,
            "environment": frozen["worker_environment"],
            "runtime": config["runtime"],
            "physical_uuid": owner["physical_uuid"],
            "actual_uuid": actual_uuid,
            "source_layouts": layout_evidence,
            "lifecycle": lifecycle,
            "analysis": analysis,
            "tolerance": {"rtol": 1e-5, "atol": 1e-6},
            "helper_sources_sha256": frozen["helpers_sha256"],
            "compilation_preserved": True,
            "graphs_preserved": True,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "scope": (
                "Bounded saved-root diagnostic only. Failed transparency prevents intermediate interpretation; "
                "no continuation or production gate."
            ),
        }
        write(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--worker-index", required=True, type=int)
    args = parser.parse_args()
    result = run(args.root.resolve(strict=True), args.binding_sha256, args.state.resolve(), args.worker_index)
    print(result["status"])


if __name__ == "__main__":
    main()
