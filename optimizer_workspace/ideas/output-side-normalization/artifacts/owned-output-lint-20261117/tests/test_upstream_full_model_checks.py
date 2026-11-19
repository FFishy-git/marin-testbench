# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual-path observations with independent CPU kernels, not CUDA emulation."""

import importlib
import importlib.util
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
CANDIDATE_TEST = Path(__file__).parent.name == "tests"


def verifier_module():
    if CANDIDATE_TEST:
        return importlib.import_module("experiment.upstream_full_model_checks")
    candidate = Path(
        "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
        "/candidates/upstream-outer-fixed-age-0p5m-20261117"
    )
    package_name = "_upstream_full_model_cpu"
    if package_name not in sys.modules:
        package_spec = importlib.util.spec_from_file_location(
            package_name,
            candidate / "experiment/__init__.py",
            submodule_search_locations=[str(candidate / "experiment")],
        )
        package = importlib.util.module_from_spec(package_spec)
        sys.modules[package_name] = package
        package_spec.loader.exec_module(package)
    name = f"{package_name}.upstream_full_model_checks"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("upstream_full_model_checks.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def kernels(monkeypatch):
    module = verifier_module()

    def reference_input(gradients, momentum, factors, mu, nesterov):
        gradient = torch.stack(gradients).float()
        updated = mu * momentum + gradient
        source = gradient + mu * updated if nesterov else updated
        return updated, torch.bmm(source, factors)

    def reference_ns(source, steps, epsilon, mode):
        return source / source.norm(dim=(-2, -1), keepdim=True).clamp_min(epsilon)

    def reference_output(weights, direction, factors, norm2, lr, decay, scale):
        weight = torch.stack(weights)
        normalized = factors * (factors.shape[-1] / norm2).sqrt()[:, None, None]
        return -lr * (
            scale * torch.bmm(direction, normalized) + decay * torch.bmm(torch.bmm(weight, normalized), normalized)
        )

    monkeypatch.setattr(module.PR, "rows_lo", lambda values: values)
    monkeypatch.setattr(module.MU, "newton_schulz_batched", reference_ns)
    monkeypatch.setattr(module, "muont_in", reference_input)
    monkeypatch.setattr(module, "muont_out", reference_output)
    prototype = importlib.import_module(f"{module.__package__}.upstream_outer_products")
    monkeypatch.setattr(prototype, "muont_in", reference_input)
    monkeypatch.setattr(prototype, "muont_out", reference_output)
    return module


def fixture(shape):
    generator = torch.Generator().manual_seed(71)
    count, rows, columns = shape
    weights, gradients, momentum = (torch.randn(shape, generator=generator) * 0.01 for _ in range(3))
    matrix = torch.randn(count, columns, columns, generator=generator)
    right = (matrix @ matrix.mT / columns + torch.eye(columns), torch.eye(columns).repeat(count, 1, 1))
    outer = torch.rand(count, rows, generator=generator) + 0.5
    return weights, gradients, momentum, right, outer


def cpu_row_squares(buffers):
    return torch.stack([buffer.float().square().sum(0) for buffer in buffers])


def captured_fast(module, shapes, apply_left):
    """Use real moments with CPU reducers behind the production observer interface."""
    block = importlib.import_module(f"{module.__package__}.block_toner")
    generator = torch.Generator().manual_seed(73)
    bufs, shape_groups = {}, []
    for shape in shapes:
        _, rows, columns = shape
        kind = "q" if rows == columns else ("gate" if rows > columns else "down")
        weights, _, _, _, _ = fixture(shape)
        mine = [(index, kind, weights[index]) for index in range(shape[0])]
        shape_groups.append((tuple(shape[-2:]), [mine]))
        for index, _, _ in mine:
            bufs[index, kind] = torch.randn(8, rows, generator=generator) * torch.linspace(0.5, 1.5, rows)
    actual = block.DiagonalMoments(
        bufs,
        decay=0.9,
        damping=0.1,
        inner_power=0.25,
        outer_power=0.5,
        rows_total=8,
        world=1,
        device=torch.device("cpu"),
        backend="torch",
    )
    actual.update()
    actual.reducers = {
        width: partial(cpu_row_squares, [bufs[key] for key in keys]) for width, keys in actual.keys.items()
    }
    # No Triton/CUDA kernel is emulated; the observer requires this backend identity.
    actual.backend = "triton"
    arm = "outer_leftdiag_upstream_fused" if apply_left else "null_leftdiag_upstream_fused"
    return SimpleNamespace(
        arm=block.BLOCK_ARMS[arm],
        update_scalars=torch.tensor([0.02, 0.1]),
        args=SimpleNamespace(
            muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32"
        ),
        input=object(),
        output=None,
        output_diagonal=actual,
        device=torch.device("cpu"),
        shape_groups=shape_groups,
        rank=0,
    )


def observed_arguments(module, fast, group):
    shape, per_rank = group
    mine = per_rank[0]
    weights, gradients, momentum, right, _ = fixture((len(mine), *shape))
    outer = torch.stack([fast.output_diagonal.factors()[index, kind][1] for index, kind, _ in mine])
    trace = {}
    updated, delta = module.upstream_outer_update_stack(
        weights,
        gradients,
        momentum,
        (None, outer if fast.arm.apply_left else None),
        right,
        args=fast.args,
        lr=fast.update_scalars[0],
        weight_decay=fast.update_scalars[1],
        trace=trace,
    )
    return (mine, weights, gradients, momentum, updated, delta, right, trace, 0.02, 0.1)


@pytest.mark.parametrize("shape", [(2, 4, 4), (2, 8, 4), (2, 4, 8)])
@pytest.mark.parametrize("apply_left", [False, True])
def test_actual_upstream_stack_observer_checks_null_geometry_and_scheduled_scalars(kernels, shape, apply_left):
    module = kernels
    fast = captured_fast(module, [shape], apply_left)
    verifier = module.UpstreamProductVerifier(fast)
    verifier.begin()
    fast.output_diagonal.update()
    arguments = observed_arguments(module, fast, fast.shape_groups[0])
    verifier(*arguments)
    result = verifier.finish()
    assert result["upstream_null_exact"] is (not apply_left)
    assert result["upstream_output_equation_max_error"] < 1e-6
    fast.update_scalars[0] = 0.01
    with pytest.raises(AssertionError):
        verifier(*arguments)
    fast.update_scalars[0] = 0.02
    broken = list(arguments)
    broken[5] = arguments[5] + 0.01
    with pytest.raises(AssertionError):
        verifier(*broken)


@pytest.mark.parametrize("apply_left", [False, True])
def test_actual_stack_observer_resets_owned_coverage_between_updates(kernels, apply_left):
    fast = captured_fast(kernels, [(2, 4, 4), (2, 8, 4), (2, 4, 8)], apply_left)
    verifier = kernels.UpstreamProductVerifier(fast)
    verifier.begin()
    fast.output_diagonal.update()
    for group in fast.shape_groups:
        verifier(*observed_arguments(kernels, fast, group))
    assert verifier.finish()["upstream_owned_stacks_checked"] == 3
    for buffer in fast.output_diagonal.bufs.values():
        buffer.mul_(1.5)
    verifier.begin()
    fast.output_diagonal.update()
    verifier(*observed_arguments(kernels, fast, fast.shape_groups[0]))
    with pytest.raises(ValueError, match="every owned shape"):
        verifier.finish()
    for group in fast.shape_groups[1:]:
        verifier(*observed_arguments(kernels, fast, group))
    assert verifier.finish()["upstream_owned_stacks_checked"] == 3


@pytest.mark.parametrize("corruption", ["moment", "root"])
@pytest.mark.parametrize("apply_left", [False, True])
def test_actual_stack_observer_rejects_corrupt_captured_moments_and_roots(kernels, corruption, apply_left):
    fast = captured_fast(kernels, [(2, 4, 4)], apply_left)
    verifier = kernels.UpstreamProductVerifier(fast)
    verifier.begin()
    fast.output_diagonal.update()
    if corruption == "moment":
        fast.output_diagonal.moment[4][0, 0] += 0.01
    else:
        fast.output_diagonal.outer_power = 0.0
    with pytest.raises(AssertionError):
        verifier(*observed_arguments(kernels, fast, fast.shape_groups[0]))


def test_changing_scalar_probe_runs_actual_step_with_reduced_and_zero_lr(kernels, monkeypatch):
    module = kernels
    real_configuration = module.configuration

    def small_configuration(*args):
        config = real_configuration(*args)
        config["model"].update(hidden=4, mlp=8, layers=8)
        return config

    monkeypatch.setattr(module, "configuration", small_configuration)
    module.test_upstream_scalar_step(torch.device("cpu"))
    fast = module.ScalarStepProbe(torch.device("cpu"), apply_left=False)
    fast.step()
    previous = [parameter.detach().clone() for _, per_rank in fast.shape_groups for _, _, parameter in per_rank[0]]
    fast.param_groups[0]["lr"] = 0.0
    fast.step()
    for parameter, expected in zip(
        (parameter for _, per_rank in fast.shape_groups for _, _, parameter in per_rank[0]), previous, strict=True
    ):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    torch.testing.assert_close(fast.update_scalars, torch.tensor([0.0, 0.1]), rtol=0, atol=0)
