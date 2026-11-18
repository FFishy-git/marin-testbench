# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unissued root-kernel preparation; caller ownership and CUDA validation remain required."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch


def clone_input(value):
    result = value.detach().clone(memory_format=torch.preserve_format)
    if result.stride() != value.stride() or result.storage_offset() != value.storage_offset():
        raise ValueError("Root preparation would change an input's raw layout")
    return result


def scratch_group(group):
    """Stage only root inputs without exposing restored numerical storage to preparation."""
    return SimpleNamespace(
        C=clone_input(group.C),
        tol=clone_input(group.tol),
        basis=SimpleNamespace(SB=clone_input(group.basis.SB), S2=clone_input(group.basis.S2)),
    )


def prepare_restored_roots(engine):
    """Prepare existing root kernels before fresh capture, discarding all scratch outputs.

    This helper is not installed in training or an issued diagnostic candidate. A CUDA
    caller must separately establish execution ownership before calling it. Readiness
    changes only after every eligible group and its side stream complete successfully.
    """
    if not engine.args.toner_graphs:
        return []
    if engine.begun:
        raise RuntimeError("Root preparation requires an engine without an active update")
    eligible = {
        name: group
        for name, group in engine.groups.items()
        if group.per and group.basis is not None and engine.steps >= engine.args.toner_inline
    }
    if any(group.graph is not None for group in eligible.values()):
        raise RuntimeError("Restored root preparation requires fresh process-local graphs")
    for group in eligible.values():
        group.warm = False
    if not eligible:
        return []
    cuda = torch.device(engine.device).type == "cuda"
    if cuda:
        engine.side.wait_stream(torch.cuda.current_stream(engine.device))
    context = torch.cuda.stream(engine.side) if cuda else nullcontext()
    with torch.no_grad(), context:
        for group in eligible.values():
            scratch = scratch_group(group)
            output = engine._roots(scratch)
            if cuda:
                engine.side.synchronize()
            del output, scratch
    if cuda:
        engine.side.synchronize()
    for group in eligible.values():
        group.warm = True
    return list(eligible)
