# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical start-trial component; a separately accepted caller must own CUDA execution."""

from collections.abc import Callable

import root_stage_trace as tracing
import root_start_accuracy as accuracy
import torch

LIMITATION = (
    "No standalone CUDA entry point or allocation owner is supplied. A future separately accepted worker must "
    "authenticate its source/config/data and exact physical GPU, retain provenance, run opposite variant orders "
    "in two fresh processes, independently reconstruct results and verify exact cleanup/restoration. This "
    "component uses an already resident input and the caller's compiled start callable; it changes no trainer."
)


def retain_kernel(B: torch.Tensor, freeze: float, compiled: Callable) -> tuple[list[dict], dict]:
    """Retain eager, capture and two graph replays of the caller's compiled start.

    The caller must establish exclusive source-bound CUDA ownership before
    entering this component. No kernel is replaced or compilation disabled.
    """
    if B.device.type != "cuda":
        raise RuntimeError("Start graph component requires caller-owned resident CUDA input")
    if B.ndim != 3 or B.shape[-1] != B.shape[-2] or B.dtype != torch.float32 or not B.numel():
        raise ValueError("Start graph component requires the full FP32 square batch")
    if not callable(compiled) or not callable(getattr(compiled, "_torchdynamo_orig_callable", None)):
        raise ValueError("Start graph component requires an existing compiled start callable")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Start trial cannot enter an existing CUDA graph capture")
    torch.cuda.synchronize(B.device)
    original = tracing.cpu_tree(B)
    if not torch.isfinite(original).all():
        raise ValueError("Start graph component input is nonfinite")
    rng = {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(B.device).clone()}
    layout = (B.shape, B.stride(), B.storage_offset(), B.data_ptr())
    current, side = torch.cuda.current_stream(B.device), torch.cuda.Stream(device=B.device)
    side.wait_stream(current)
    retained = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        with torch.cuda.stream(side):
            eager = compiled(B, freeze)
        side.synchronize()
        copied = tracing.cpu_tree(eager)
        accuracy.named_output(copied, B.shape)
        retained.append({"phase": "eager", "output": copied})
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            captured = compiled(B, freeze)
        with torch.cuda.stream(side):
            graph.replay()
        side.synchronize()
        copied = tracing.cpu_tree(captured)
        accuracy.named_output(copied, B.shape)
        retained.append({"phase": "capture", "output": copied})
        for phase in accuracy.PHASES[2:]:
            with torch.cuda.stream(side):
                graph.replay()
            side.synchronize()
            copied = tracing.cpu_tree(captured)
            accuracy.named_output(copied, B.shape)
            retained.append({"phase": phase, "output": copied})
    current.wait_stream(side)
    torch.cuda.synchronize(B.device)
    values = tracing.comparison(tracing.cpu_tree(B), original, "start.input")
    preserved_rng = torch.equal(rng["cpu"], torch.get_rng_state()) and torch.equal(
        rng["cuda"], torch.cuda.get_rng_state(B.device)
    )
    if layout != (B.shape, B.stride(), B.storage_offset(), B.data_ptr()) or not accuracy.exact_finite(values):
        raise ValueError("Start graph component changed resident input values or storage")
    if not preserved_rng:
        raise ValueError("Start graph component changed CPU or CUDA RNG")
    return retained, {
        "shape": list(B.shape),
        "stride": list(B.stride()),
        "storage_offset": B.storage_offset(),
        "input_exact_finite": values,
        "rng_exact": preserved_rng,
        "compilation_preserved": True,
        "graphs_preserved": True,
        "captured_graph_replays": 3,
        "limitation": LIMITATION,
    }
