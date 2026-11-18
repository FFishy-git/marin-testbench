# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify restored numerical/RNG preservation around scratch-only kernel preparation."""

import random

import torch

if __package__:
    from .prepare_restored_roots import prepare_restored_roots
else:
    from prepare_restored_roots import prepare_restored_roots


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return value


def exact_tree(actual, expected, name="restored"):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(f"Root preparation changed a tensor contract: {name}")
        if not torch.equal(actual.cpu(), expected):
            raise ValueError(f"Root preparation changed numerical or RNG state: {name}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Root preparation changed state keys: {name}")
        for key, value in expected.items():
            exact_tree(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"Root preparation changed state structure: {name}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            exact_tree(left, right, f"{name}.{index}")
    elif actual != expected:
        raise ValueError(f"Root preparation changed state scalar: {name}")


def without_readiness(state):
    """Exclude only the two process-local group descriptors intentionally changed by preparation."""
    result = clone_tree(state)
    for name in ("input", "output"):
        if result.get(name) is not None:
            for group in result[name]["groups"].values():
                group.pop("warm")
                group.pop("cached_root_graph_active")
    return result


def checked_preparation(fast, *, rank, step):
    if fast.input is None or fast.output is not None or step != 22 or fast.input.steps != step:
        raise ValueError("Root diagnostic requires the restored fused input-only boundary engine")
    if torch.device(fast.input.device).type == "cuda":
        torch.cuda.synchronize(fast.input.device)
    before = without_readiness(fast.runtime_state())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = (
        torch.cuda.get_rng_state(fast.input.device).clone() if torch.device(fast.input.device).type == "cuda" else None
    )
    python_rng = random.getstate()
    try:
        groups = prepare_restored_roots(fast.input)
        exact_tree(without_readiness(fast.runtime_state()), before)
        exact_tree(torch.get_rng_state(), cpu_rng, "torch_rng")
        if cuda_rng is not None:
            exact_tree(torch.cuda.get_rng_state(fast.input.device), cuda_rng, "cuda_rng")
        exact_tree(random.getstate(), python_rng, "python_rng")
        if not groups or any(
            not fast.input.groups[name].warm or fast.input.groups[name].graph is not None for name in groups
        ):
            raise ValueError("Root preparation did not commit readiness with fresh process-local graphs")
    except BaseException:
        for group in fast.input.groups.values():
            if group.per and group.basis is not None and fast.input.steps >= fast.input.args.toner_inline:
                group.warm = False
        raise
    return {
        "status": "restored_root_scratch_preparation_state_rng_exact_not_continuation_acceptance",
        "rank": rank,
        "checkpoint_step": step,
        "groups": groups,
        "numerical_state_exact": True,
        "torch_rng_exact": True,
        "cuda_rng_exact": True if cuda_rng is not None else None,
        "python_rng_exact": True,
        "fresh_graphs": True,
        "scratch_outputs_consumed": False,
    }
