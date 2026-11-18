# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require numerical gates to detect broken actual updates and replay restoration."""

import sys
from dataclasses import replace

import pytest
import torch
from fused_attribution_checks import run_checks

checks = sys.modules[run_checks.__module__]


def test_actual_steps_match_input_only_null_and_outer_equations_with_exact_continuation():
    result = checks.run_checks()
    assert result["status"] == "fused_attribution_cpu_behavior_passed"
    assert result["momentum_max_error"] == result["null_output_invariance_max_error"] == 0.0
    for case in result["cases"].values():
        assert case["maximum_tolerance_fraction"] <= 1.0
        assert case["minimum_output_null_difference"] > 1e-5
        assert case["checkpoint_exact"]


def test_gate_rejects_null_that_applies_observed_output_factors(monkeypatch):
    step = checks._CpuStepHolder.step

    def apply_output(holder):
        original = holder.arm
        holder.arm = replace(original, identity_left_outer=False)
        try:
            step(holder)
        finally:
            holder.arm = original

    monkeypatch.setattr(checks._CpuStepHolder, "step", apply_output)
    with pytest.raises(AssertionError):
        checks.run_checks()


def test_gate_rejects_numerically_correct_null_that_bypasses_outer_work(monkeypatch):
    step = checks._CpuStepHolder.step

    def bypass_output(holder):
        original = holder.arm
        if original.identity_left_outer:
            holder.arm = replace(original, apply_left=False, identity_left_outer=False)
        try:
            step(holder)
        finally:
            holder.arm = original

    monkeypatch.setattr(checks._CpuStepHolder, "step", bypass_output)
    with pytest.raises(AssertionError):
        checks.run_checks()


@pytest.mark.parametrize("corruption", ["momentum", "decay"])
def test_gate_rejects_corrupt_momentum_or_unshaped_decay(monkeypatch, corruption):
    update = checks.block_toner.block_update_stack

    def corrupt_update(*args, **kwargs):
        if corruption == "decay":
            kwargs["weight_decay"] = 0.0
        momentum, delta = update(*args, **kwargs)
        if corruption == "momentum":
            momentum = momentum + 0.001
        return momentum, delta

    monkeypatch.setattr(checks.block_toner, "block_update_stack", corrupt_update)
    with pytest.raises(AssertionError):
        checks.run_checks()


def test_gate_rejects_null_that_skips_output_moment_collection(monkeypatch):
    factors = checks._CpuStepHolder._factors

    def skip_output(holder, engine, diagonal):
        if holder.arm.identity_left_outer and diagonal is holder.output_diagonal:
            return {}
        return factors(holder, engine, diagonal)

    monkeypatch.setattr(checks._CpuStepHolder, "_factors", skip_output)
    with pytest.raises(AssertionError):
        checks.run_checks()


def test_gate_rejects_identity_roots_returned_instead_of_computed_output_factors(monkeypatch):
    factors = checks.DiagonalMoments.factors

    def identity_output(diagonal):
        return {key: tuple(torch.ones_like(value) for value in pair) for key, pair in factors(diagonal).items()}

    monkeypatch.setattr(checks.DiagonalMoments, "factors", identity_output)
    with pytest.raises(AssertionError):
        checks.run_checks()


@pytest.mark.parametrize("corruption", ["weights", "momentum", "input_roots", "output_moments", "steps", "rng"])
def test_gate_rejects_checkpoint_with_lost_numerical_or_replay_state(monkeypatch, corruption):
    restore = checks._restore_checkpoint

    def corrupt_restore(holder, payload):
        restore(holder, payload)
        with torch.no_grad():
            if corruption == "weights":
                holder.params[0].add_(0.001)
            elif corruption == "momentum":
                next(iter(holder.momentum.values())).zero_()
            elif corruption == "input_roots":
                next(iter(holder.input.groups.values())).C.zero_()
            elif corruption == "output_moments":
                next(iter(holder.output_diagonal.moment.values())).zero_()
            elif corruption == "steps":
                holder.steps = 0
            else:
                holder.generator.manual_seed(99)

    monkeypatch.setattr(checks, "_restore_checkpoint", corrupt_restore)
    with pytest.raises(AssertionError):
        checks.run_checks()
