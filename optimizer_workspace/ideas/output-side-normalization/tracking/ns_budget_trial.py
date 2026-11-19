# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compiled NS graph-trial component; a separate source-bound caller must own CUDA."""

import math
import statistics
from collections.abc import Callable

import replay_saved_ns as replay
import root_stage_trace as tracing
import torch

PHASES = ("compiled_call", "capture_executed", "replay", "replay_repeat", "timed_final")
TIMING_REPLAYS = 50
LIMITATION = (
    "No standalone CUDA entry point or allocation owner. A separately accepted worker must bind the "
    "source/config/data, physical GPU, changed step count and full saved input, run opposite orders in "
    "two fresh processes and independently reconstruct historical controls, cleanup and restoration. "
    "CUDA-event replay timing excludes compile, graph capture, input/output transfers and full training."
)


def summarize_kernel(retained: list[dict], elapsed: list[float]) -> dict:
    """Recompute complete repeat comparisons and timing statistics from retained CPU evidence."""
    if [item["phase"] for item in retained] != list(PHASES):
        raise ValueError("NS trial requires its complete compiled/capture/replay sequence")
    values = [item["output"] for item in retained]
    for value in values:
        replay._require_cpu_tensor(value)
        if value.ndim != 3 or value.shape != values[0].shape:
            raise ValueError("NS trial outputs must cover one full finite-map batch shape")
    if len(elapsed) != TIMING_REPLAYS or any(not math.isfinite(value) or value <= 0 for value in elapsed):
        raise ValueError("NS trial timing requires all finite positive CUDA-event intervals")
    comparisons = [
        {"phase": phase, **replay.compare_outputs(output, values[0])}
        for phase, output in zip(PHASES[1:], values[1:], strict=True)
    ]
    ordered = sorted(elapsed)
    return {
        "phases": list(PHASES),
        "repeat_comparisons": comparisons,
        "all_phase_outputs_exact_finite": all(value["exact"] and value["finite"] for value in comparisons),
        "cuda_event_milliseconds": elapsed,
        "median_milliseconds": statistics.median(elapsed),
        "p95_milliseconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "timing_replays": len(elapsed),
        "scope": LIMITATION,
    }


def retain_kernel(source: torch.Tensor, compiled: Callable) -> tuple[list[dict], dict]:
    """Retain and time the caller's compiled NS map on a resident, caller-owned CUDA source."""
    if source.device.type != "cuda":
        raise RuntimeError("NS graph trial requires caller-owned resident CUDA input")
    if source.dtype != torch.float32 or source.ndim != 3 or not source.numel():
        raise ValueError("NS graph trial requires a complete nonempty FP32 batch")
    if not callable(compiled) or not callable(getattr(compiled, "_torchdynamo_orig_callable", None)):
        raise ValueError("NS graph trial requires an existing compiled NS callable")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("NS graph trial cannot enter an existing CUDA capture")
    torch.cuda.synchronize(source.device)
    original = tracing.cpu_tree(source)
    if not torch.isfinite(original).all():
        raise ValueError("NS graph trial source is nonfinite")
    rng = {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(source.device).clone()}
    layout = (source.shape, source.stride(), source.storage_offset(), source.data_ptr())
    current, side = torch.cuda.current_stream(source.device), torch.cuda.Stream(device=source.device)
    side.wait_stream(current)
    retained, elapsed = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        with torch.cuda.stream(side):
            output = compiled(source)
        side.synchronize()
        if output.device != source.device or output.dtype != torch.float32 or output.shape != source.shape:
            raise ValueError("NS compiled call returned different shape/dtype/device")
        retained.append({"phase": PHASES[0], "output": tracing.cpu_tree(output)})
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            captured = compiled(source)
        if captured.device != source.device or captured.dtype != torch.float32 or captured.shape != source.shape:
            raise ValueError("NS captured call returned different shape/dtype/device")
        with torch.cuda.stream(side):
            graph.replay()
        side.synchronize()
        retained.append({"phase": PHASES[1], "output": tracing.cpu_tree(captured)})
        for phase in PHASES[2:4]:
            with torch.cuda.stream(side):
                graph.replay()
            side.synchronize()
            retained.append({"phase": phase, "output": tracing.cpu_tree(captured)})
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(TIMING_REPLAYS)
        ]
        with torch.cuda.stream(side):
            for start, end in events:
                start.record(side)
                graph.replay()
                end.record(side)
        side.synchronize()
        elapsed = [start.elapsed_time(end) for start, end in events]
        retained.append({"phase": PHASES[-1], "output": tracing.cpu_tree(captured)})
    current.wait_stream(side)
    torch.cuda.synchronize(source.device)
    unchanged = tracing.comparison(tracing.cpu_tree(source), original, "ns.source")
    if (
        layout != (source.shape, source.stride(), source.storage_offset(), source.data_ptr())
        or not unchanged["exact"]
        or not unchanged["finite"]
    ):
        raise ValueError("NS graph trial changed resident source values or storage")
    if not (
        torch.equal(rng["cpu"], torch.get_rng_state())
        and torch.equal(rng["cuda"], torch.cuda.get_rng_state(source.device))
    ):
        raise ValueError("NS graph trial changed CPU or CUDA RNG")
    summary = summarize_kernel(retained, elapsed)
    summary.update(
        input_exact_finite=unchanged,
        rng_exact=True,
        compilation_preserved=True,
        graphs_preserved=True,
        graph_replays=3 + TIMING_REPLAYS,
    )
    return retained, summary
