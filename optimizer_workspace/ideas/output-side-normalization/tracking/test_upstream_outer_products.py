# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the upstream null path and outer equation without substituting CUDA evidence."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from benchmark_diagonal import source_definitions

TASK = Path(__file__).resolve().parents[1]
UPSTREAM = TASK / "artifacts/external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c"
CANDIDATE = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/candidates/upstream-outer-fixed-age-0p5m-20261117"
)


@pytest.fixture
def equations():
    # FP64 direct products are an algebra oracle, not an emulation of the CUDA split kernels.
    roots = SimpleNamespace(rows_lo=lambda value: value, apply16_raw=torch.bmm)
    muon = SimpleNamespace(
        **source_definitions(
            UPSTREAM / "src/toner/fast/muon.py",
            ("ABC", "_bmm", "newton_schulz_batched"),
            {"torch": torch},
        )
    )
    pinned = source_definitions(
        UPSTREAM / "src/toner/fast/engine.py",
        ("_muont_in", "_muont_out"),
        {"torch": torch, "PR": roots},
    )
    prototype = source_definitions(
        Path(__file__).with_name("upstream_outer_products.py"),
        ("upstream_outer_update_stack",),
        {"torch": torch, "PR": roots, "MU": muon, "muont_in": pinned["_muont_in"], "muont_out": pinned["_muont_out"]},
    )
    return pinned, prototype["upstream_outer_update_stack"], muon, roots


def fixture(shape, dtype):
    generator = torch.Generator().manual_seed(91)
    weights, gradients, momenta = (torch.randn(shape, generator=generator, dtype=dtype) for _ in range(3))
    count, rows, columns = shape
    matrix = torch.randn(count, columns, columns, generator=generator, dtype=dtype)
    inner = matrix @ matrix.mT / columns + torch.eye(columns, dtype=dtype)
    outer = matrix.mT @ matrix / columns + torch.eye(columns, dtype=dtype)
    left = torch.rand(count, rows, generator=generator, dtype=dtype) + 0.25
    return weights, gradients, momenta, left, (inner, outer)


@pytest.mark.parametrize("shape", [(2, 4, 4), (2, 8, 4), (2, 4, 8)])
@pytest.mark.parametrize("nesterov", [False, True])
def test_null_prototype_exactly_retains_pinned_momentum_ns_and_outer_delta(equations, shape, nesterov):
    pinned, prototype, muon, roots = equations
    weights, gradients, momenta, _, right = fixture(shape, torch.float32)
    args = SimpleNamespace(
        muon_momentum=0.95, muon_nesterov=nesterov, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32"
    )
    lr, decay = torch.tensor(0.02262741696), torch.tensor(0.1)
    momentum, source = pinned["_muont_in"](list(gradients.unbind()), momenta, right[0], 0.95, nesterov)
    direction = muon.newton_schulz_batched(source, 10, 1e-5, "f32")
    expected = pinned["_muont_out"](
        list(weights.unbind()),
        direction,
        roots.rows_lo(right[1]),
        right[1].square().sum((-2, -1)),
        lr,
        decay,
        max(1.0, shape[-2] / shape[-1]) ** 0.5,
    )
    trace = {}
    actual_momentum, delta = prototype(
        weights, gradients, momenta, (None, None), right, args=args, lr=lr, weight_decay=decay, trace=trace
    )
    torch.testing.assert_close(actual_momentum, momentum, rtol=0, atol=0)
    torch.testing.assert_close(trace["ns_source"], source, rtol=0, atol=0)
    torch.testing.assert_close(trace["ns_output"], direction, rtol=0, atol=0)
    torch.testing.assert_close(delta, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 4, 4), (2, 8, 4), (2, 4, 8)])
@pytest.mark.parametrize("lr", [0.0, 0.02262741696, 0.00070710678])
def test_outer_prototype_matches_independent_diagonal_direction_and_input_decay(equations, shape, lr):
    _, prototype, _, _ = equations
    weights, gradients, momenta, left, right = fixture(shape, torch.float64)
    args = SimpleNamespace(muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32")
    trace = {}
    _, actual = prototype(
        weights, gradients, momenta, (None, left), right, args=args, lr=lr, weight_decay=0.1, trace=trace
    )
    expected = []
    for weight, direction, output, factor in zip(weights, trace["ns_output"], left, right[1], strict=True):
        normalized_right = factor / (factor.square().sum() / factor.shape[-1]).sqrt()
        normalized_left = torch.diag(output / output.square().mean().sqrt())
        scale = max(1.0, weight.shape[0] / weight.shape[1]) ** 0.5
        expected.append(
            -lr
            * (
                scale * normalized_left @ direction.double() @ normalized_right
                + 0.1 * weight @ normalized_right @ normalized_right
            )
        )
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-12, atol=1e-12)
    if lr == 0:
        assert torch.count_nonzero(actual).item() == 0


def test_output_scaling_does_not_change_input_momentum_or_ns(equations):
    _, prototype, _, _ = equations
    weights, gradients, momenta, left, right = fixture((2, 4, 8), torch.float32)
    args = SimpleNamespace(muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32")
    traces, deltas, buffers = [], [], []
    for output in (None, left):
        trace = {}
        buffer, delta = prototype(
            weights, gradients, momenta, (None, output), right, args=args, lr=0.02, weight_decay=0.1, trace=trace
        )
        traces.append(trace)
        deltas.append(delta)
        buffers.append(buffer)
    torch.testing.assert_close(buffers[0], buffers[1], rtol=0, atol=0)
    for field in ("ns_source", "ns_output"):
        torch.testing.assert_close(traces[0][field], traces[1][field], rtol=0, atol=0)
    assert torch.count_nonzero(deltas[0] - deltas[1]).item() > 0


@pytest.mark.parametrize("apply_left", [False, True])
def test_candidate_step_uses_pinned_products_and_preserves_null_collection(equations, apply_left):
    _, prototype, _, _ = equations
    path = CANDIDATE / "experiment/block_toner.py"
    if not path.exists():
        pytest.skip("The separately prepared candidate is not available in this environment")
    namespace = source_definitions(path, ("stack_factors", "INPUT_KIND"), {"torch": torch, "KINDS_D": ()})
    tree = ast.parse(path.read_text())
    block = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BlockToner")
    step = next(node for node in block.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    namespace.update(torch=torch, upstream_outer_update_stack=prototype)
    exec(compile(ast.Module(body=[step], type_ignores=[]), str(path), "exec"), namespace)
    state = SimpleNamespace(
        arm=SimpleNamespace(
            upstream_product_kernels=True, lag_left=False, apply_left=apply_left, apply_left_inner=False
        ),
        args=SimpleNamespace(
            muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32"
        ),
        update_scalars=torch.empty(2),
        world=1,
        rank=0,
        device="cpu",
        input=object(),
        output=object(),
        input_diagonal=None,
        output_diagonal=None,
        update_verifier=None,
        engines=[],
        steps=0,
        momentum={},
        shape_groups=[],
        param_groups=[{"lr": 0.02, "weight_decay": 0.1}],
    )
    factors, expected = {}, []
    for index, shape in enumerate(((4, 4), (8, 4), (4, 8))):
        weights, gradients, momenta, left, right = fixture((1, *shape), torch.float32)
        parameter = torch.nn.Parameter(weights[0].clone())
        parameter.grad = gradients[0].clone()
        state.shape_groups.append((shape, [[(index, "q", parameter)]]))
        state.momentum[shape] = momenta.clone()
        factors[index] = (left, right)
        _, delta = prototype(
            weights,
            gradients,
            momenta,
            (None, left if apply_left else None),
            right,
            args=state.args,
            lr=torch.tensor(0.02),
            weight_decay=torch.tensor(0.1),
        )
        expected.append(weights[0] + delta[0])
    collected = []

    def collect(engine, diagonal):
        collected.append(engine)
        return {
            (index, "attn" if engine is state.input else "q"): (
                (right[0][0], right[1][0]) if engine is state.input else (left[0], left[0])
            )
            for index, (left, right) in factors.items()
        }

    state._factors = collect
    state._engine_metrics = lambda: {}
    namespace["step"](state)
    assert collected == [state.input, state.output]
    assert state.steps == 1
    for (_, entries), reference in zip(state.shape_groups, expected, strict=True):
        torch.testing.assert_close(entries[0][0][2], reference, rtol=0, atol=0)
