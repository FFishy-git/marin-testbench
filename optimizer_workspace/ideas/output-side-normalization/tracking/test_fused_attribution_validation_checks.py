# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict finite-NS and retained-work gates on synthetic CPU stacks."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from experiment import block_toner

if Path(__file__).parent.name == "tests":
    from experiment import fused_attribution_validation_checks as checks
else:
    spec = importlib.util.spec_from_file_location(
        "experiment.fused_attribution_validation_checks",
        Path(__file__).with_name("fused_attribution_validation_checks.py"),
    )
    checks = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = checks
    spec.loader.exec_module(checks)


@pytest.mark.parametrize("width", (4, 512, 2048))
def test_applied_identity_retains_packing_normalization_row_work(width):
    observer = checks.AppliedOuterTrace()
    direction = torch.randn(2, width, 4)
    with observer:
        applied = torch.stack([torch.ones(width), torch.ones(width)])
        normalized = block_toner.stack_unit_rms(applied)
        result = direction * normalized[:, :, None]
    observer.enabled = False
    observer.require_applied(applied, result)


@pytest.mark.parametrize("fault", ("packing", "normalization", "application", "wrong_factor"))
def test_numerically_correct_null_cannot_bypass_retained_work(fault):
    observer = checks.AppliedOuterTrace()
    applied, direction = torch.ones(2, 4), torch.randn(2, 4, 4)
    with observer:
        packed = applied if fault == "packing" else torch.stack(list(applied.unbind(0)))
        normalized = packed if fault == "normalization" else block_toner.stack_unit_rms(packed)
        result = direction if fault == "application" else direction * normalized[:, :, None]
    if fault == "wrong_factor":
        applied = applied * 2
    with pytest.raises(ValueError):
        observer.require_applied(applied, result)


@pytest.mark.parametrize("null", (False, True))
@pytest.mark.parametrize("fault", ("none", "delta", "zero_outer_at_zero_lr", "scaled_outer_at_zero_lr"))
def test_actual_stack_requires_unrelaxed_update_and_applied_factor(null, fault):
    rows = torch.tensor([[0.8, 1.1, 1.2, 0.9], [1.2, 0.9, 0.8, 1.1]], dtype=torch.bfloat16)
    diagonal = block_toner.DiagonalMoments(
        {(0, "q"): rows},
        decay=0.938,
        damping=0.001,
        inner_power=0.5,
        outer_power=0.125,
        rows_total=2,
        world=1,
        device="cpu",
        backend="torch",
    )
    # CPU stand-in at the reduction boundary; all moment/root/update checks remain real.
    diagonal.backend = "triton"
    diagonal.reducers = {4: lambda: rows.float().square().sum(0)[None, :]}
    fast = SimpleNamespace(
        arm=SimpleNamespace(identity_left_outer=null, apply_left_inner=False, lag_left=False),
        args=SimpleNamespace(
            muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32"
        ),
        output_diagonal=diagonal,
        input=object(),
        output=None,
        device="cpu",
        rank=0,
        shape_groups=[((4, 4), [[(0, "q", None)]])],
    )
    verifier = checks.FusedAttributionVerifier(fast)
    verifier.begin()
    mine = [(0, "q", None)]
    weights, gradients, momentum = torch.randn(1, 4, 4), torch.randn(1, 4, 4), torch.randn(1, 4, 4)
    right = (torch.eye(4)[None, :], torch.eye(4)[None, :])
    trace = {}
    lr = 0.0 if fault in ("zero_outer_at_zero_lr", "scaled_outer_at_zero_lr") else 0.064
    with verifier.observe():
        diagonal.update()
        factors = diagonal.factors()
        required = torch.ones(4) if null else factors[0, "q"][1]
        if fault in ("zero_outer_at_zero_lr", "scaled_outer_at_zero_lr"):
            torch.stack([required])
        applied_vector = (
            torch.zeros(4)
            if fault == "zero_outer_at_zero_lr"
            else required * 2 if fault == "scaled_outer_at_zero_lr" else required
        )
        applied = torch.stack([applied_vector])
        actual_momentum, delta = block_toner.block_update_stack(
            weights,
            gradients,
            momentum,
            (None, applied),
            right,
            args=fast.args,
            lr=lr,
            weight_decay=0.0,
            trace=trace,
        )
    if fault == "delta":
        delta = delta + 0.0001
    if fault == "delta":
        with pytest.raises(AssertionError):
            verifier(mine, weights, gradients, momentum, actual_momentum, delta, right, trace, lr, 0.0)
    elif fault == "zero_outer_at_zero_lr":
        assert not torch.isfinite(delta).all()
        with pytest.raises(AssertionError):
            verifier(mine, weights, gradients, momentum, actual_momentum, delta, right, trace, lr, 0.0)
    elif fault == "scaled_outer_at_zero_lr":
        assert torch.isfinite(delta).all()
        assert torch.count_nonzero(delta).item() == 0
        with pytest.raises(ValueError):
            verifier(mine, weights, gradients, momentum, actual_momentum, delta, right, trace, lr, 0.0)
    else:
        verifier(mine, weights, gradients, momentum, actual_momentum, delta, right, trace, lr, 0.0)
        result = verifier.finish()
        assert result["output_work_retained_passed"]
        if null:
            assert result["null_identity_outer_passed"]
