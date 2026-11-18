# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from experiment.block_toner import (
    BLOCK_ARMS,
    BLOCK_KINDS,
    ROW_SCALE,
    BlockToner,
    DiagonalMoments,
    MatrixFactors,
    RowObservedLinear,
    Side,
    block_update,
    block_update_stack,
    diagonal_roots,
    stack_factors,
)
from experiment.factors import OutputSettings, inverse_factors
from toner.fast.muon import newton_schulz_batched
from torch import nn

ARGS = SimpleNamespace(muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32")


def unit(factor):
    return factor * (factor.shape[-1] ** 0.5 / factor.norm())


def random_factors(rows, columns, generator):
    left = torch.randn(rows, rows + 2, generator=generator)
    right = torch.randn(columns, columns + 2, generator=generator)
    left_inner, left_outer = inverse_factors(left @ left.T, OutputSettings())
    right_inner, right_outer = inverse_factors(right @ right.T, OutputSettings())
    return left_inner, left_outer, right_inner, right_outer


def test_arm_output_sides_partition_block_kinds():
    expected = {
        "inner_outer_all": (("q", "k", "v", "o", "gate", "up", "down"), ()),
        "null_all": (("q", "k", "v", "o", "gate", "up", "down"), ()),
        "left_all": (("q", "k", "v", "o", "gate", "up", "down"), ()),
        "inner_outer_all_diag": ((), ("q", "k", "v", "o", "gate", "up", "down")),
        "inner_outer_hidden": (("q", "k", "v", "o", "down"), ()),
        "inner_outer_mlpdiag": (("q", "k", "v", "o", "down"), ("gate", "up")),
        "inner_outer_leftdiag": ((), ("q", "k", "v", "o", "gate", "up", "down")),
        "outer_left_all": (("q", "k", "v", "o", "gate", "up", "down"), ()),
        "inner_outer_all_lag": (("q", "k", "v", "o", "gate", "up", "down"), ()),
        "null_mlpdiag": (("q", "k", "v", "o", "down"), ("gate", "up")),
        "inner_outer_mlpdiag_lag": (("q", "k", "v", "o", "down"), ("gate", "up")),
    }
    assert set(expected) == set(BLOCK_ARMS)
    for arm, (full, diagonal) in expected.items():
        assert BLOCK_ARMS[arm].output_kinds(Side.FULL) == full
        assert BLOCK_ARMS[arm].output_kinds(Side.DIAGONAL) == diagonal
        assert set(full) | set(diagonal) <= set(BLOCK_KINDS)
        assert not set(full) & set(diagonal)
    # Only the null control withholds left factors; only outer_left_all drops the inner left factor.
    assert [n for n, a in BLOCK_ARMS.items() if not a.apply_left] == ["null_all", "null_mlpdiag"]
    assert [n for n, a in BLOCK_ARMS.items() if not a.apply_left_inner] == ["outer_left_all"]
    # Lag is defined only against a dense output engine, since a diagonal side has no launch to hide.
    lagged = [n for n, a in BLOCK_ARMS.items() if a.lag_left]
    assert lagged == ["inner_outer_all_lag", "inner_outer_mlpdiag_lag"]
    for name in lagged:
        assert BLOCK_ARMS[name].apply_left


class ReplayingEngine:
    """Stands in for the cached-root engine: each launch rewrites the previous one's memory.

    The real engine replays a CUDA graph whose output buffers are never freed, so a factor
    that outlives its update has to be copied out. Rewriting the buffer here is what makes
    a missing copy observable.
    """

    def __init__(self, keys, width):
        self.groups = {"out": SimpleNamespace(own=list(keys))}
        self.begun = False
        self.launches = 0
        self.buffer = torch.zeros(2, len(keys), width)

    def begin(self):
        self.launches += 1
        self.buffer.fill_(float(self.launches))
        self.begun = True

    def _finish_roots(self):
        return {"out": (self.buffer[0], self.buffer[1])}


class LaggedToner:
    """BlockToner's lag bookkeeping, exercised without a model or a CUDA engine."""

    finish_backward = BlockToner.finish_backward
    _collect_lagged = BlockToner._collect_lagged
    _engine_factors = BlockToner._engine_factors

    def __init__(self, keys, width, arm="inner_outer_all_lag", diagonal=None):
        self.arm = BLOCK_ARMS[arm]
        self.output = ReplayingEngine(keys, width)
        self.output_diagonal = diagonal
        self.lagged = {}
        self.pending_diagonal = {}


def test_lagged_arm_applies_the_factors_of_the_previous_update():
    keys = [(0, "q"), (1, "down")]
    toner = LaggedToner(keys, 3)
    # The first update has no earlier launch, so it applies input factors only.
    toner.finish_backward()
    assert toner.output.launches == 1
    assert toner.lagged == {}
    # Each later update collects the previous launch before this one overwrites its memory.
    for expected in (1.0, 2.0, 3.0):
        toner.finish_backward()
        assert toner.output.launches == expected + 1
        assert set(toner.lagged) == set(keys)
        for inner, outer in toner.lagged.values():
            torch.testing.assert_close(inner, torch.full((3,), expected))
            torch.testing.assert_close(outer, torch.full((3,), expected))


class SteppingDiagonal:
    """Stands in for DiagonalMoments; its factors report which update produced them."""

    def __init__(self, keys, width):
        self.keys, self.width, self.updates = list(keys), width, 0

    def update(self):
        self.updates += 1

    def factors(self):
        value = torch.full((self.width,), float(self.updates))
        return {key: (value.clone(), value.clone()) for key in self.keys}


def test_lag_delays_diagonal_and_dense_left_factors_by_the_same_update():
    """A mixed arm must apply one update's lag to every left factor, not just the dense ones."""
    dense_keys, diagonal_keys = [(0, "q")], [(0, "gate")]
    diagonal = SteppingDiagonal(diagonal_keys, 2)
    toner = LaggedToner(dense_keys, 3, arm="inner_outer_mlpdiag_lag", diagonal=diagonal)
    toner.finish_backward()
    # Nothing precedes the first update, so it applies input factors only.
    assert toner.lagged == {}
    assert diagonal.updates == 1
    for expected in (1.0, 2.0, 3.0):
        toner.finish_backward()
        assert set(toner.lagged) == {*dense_keys, *diagonal_keys}
        torch.testing.assert_close(toner.lagged[dense_keys[0]][0], torch.full((3,), expected))
        torch.testing.assert_close(toner.lagged[diagonal_keys[0]][0], torch.full((2,), expected))
        # The diagonal advances once per update, in step with the dense launches.
        assert diagonal.updates == toner.output.launches


def test_collecting_lagged_factors_twice_leaves_the_next_update_unchanged():
    """A checkpoint collects the in-flight launch; the next update must not collect it again."""
    keys = [(0, "q")]
    toner = LaggedToner(keys, 2)
    toner.finish_backward()
    toner.finish_backward()
    collected = toner._collect_lagged()
    torch.testing.assert_close(collected[keys[0]][0], torch.full((2,), 2.0))
    assert not toner.output.begun
    # Collecting again without a launch in flight returns the same factors.
    toner.lagged = collected
    assert toner._collect_lagged() is collected
    assert toner.output.launches == 2


class CpuStepToner:
    """BlockToner.step with no engines, so the update, finite check and apply run on CPU."""

    step = BlockToner.step
    _factors = BlockToner._factors
    _engine_factors = BlockToner._engine_factors
    _engine_metrics = BlockToner._engine_metrics
    engines = BlockToner.engines

    def __init__(self, shapes, lr=0.02, weight_decay=0.1):
        self.arm = BLOCK_ARMS["inner_outer_all"]
        self.input = self.output = self.input_diagonal = self.output_diagonal = None
        self.args = ARGS
        self.world, self.rank, self.device, self.pg = 1, 0, "cpu", None
        self.lagged, self.last_metrics, self.steps = {}, {}, 0
        generator = torch.Generator().manual_seed(4)
        self.shape_groups, self.momentum, self.params = [], {}, []
        for index, shape in enumerate(shapes):
            parameter = nn.Parameter(torch.randn(*shape, generator=generator))
            parameter.grad = torch.randn(*shape, generator=generator)
            self.params.append(parameter)
            self.shape_groups.append((shape, [[(index, "q", parameter)]]))
            self.momentum[shape] = torch.zeros(1, *shape)
        self.param_groups = [{"params": self.params, "lr": lr, "weight_decay": weight_decay}]


def test_step_applies_the_per_matrix_update_to_every_shape_group():
    shapes = [(4, 6), (6, 4)]
    toner = CpuStepToner(shapes)
    before = [parameter.detach().clone() for parameter in toner.params]
    expected = [
        block_update(
            parameter.detach(),
            parameter.grad,
            torch.zeros(*shape),
            MatrixFactors(),
            args=ARGS,
            lr=0.02,
            weight_decay=0.1,
        )[1]
        for parameter, (shape, _) in zip(toner.params, toner.shape_groups, strict=True)
    ]
    toner.step()
    assert toner.steps == 1
    for parameter, start, delta in zip(toner.params, before, expected, strict=True):
        torch.testing.assert_close(parameter.detach(), start + delta, rtol=1e-4, atol=1e-6)


def test_step_leaves_every_parameter_unchanged_when_any_shape_group_is_nonfinite():
    """The finite check covers all groups at once, so one bad group must block the whole update."""
    toner = CpuStepToner([(4, 6), (6, 4)])
    toner.params[1].grad[0, 0] = float("inf")
    before = [parameter.detach().clone() for parameter in toner.params]
    with pytest.raises(FloatingPointError, match=r"6, 4"):
        toner.step()
    for parameter, start in zip(toner.params, before, strict=True):
        torch.testing.assert_close(parameter.detach(), start, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(6, 6), (3, 7), (8, 4)])
def test_two_sided_update_matches_independent_finite_ns_equations(shape):
    generator = torch.Generator().manual_seed(5)
    weight, gradient, momentum = (torch.randn(*shape, generator=generator) for _ in range(3))
    li, lo, ri, ro = random_factors(*shape, generator)
    actual_momentum, actual = block_update(
        weight, gradient, momentum, MatrixFactors(li, lo, ri, ro), args=ARGS, lr=0.02, weight_decay=0.1
    )
    expected_momentum = 0.95 * momentum + gradient
    source = li @ (gradient + 0.95 * expected_momentum) @ ri
    direction = unit(lo) @ newton_schulz_batched(source[None], 10, 1e-5, "f32")[0] @ unit(ro)
    decay = weight @ unit(ro) @ unit(ro)
    scale = max(1.0, shape[0] / shape[1]) ** 0.5
    torch.testing.assert_close(actual_momentum, expected_momentum, rtol=0, atol=0)
    torch.testing.assert_close(actual, -0.02 * (scale * direction + 0.1 * decay), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "sides",
    [("dense", "dense"), ("dense", "diagonal"), ("none", "dense"), ("none", "diagonal"), ("diagonal", "none")],
)
def test_batched_update_matches_per_matrix_update(sides):
    """One batched call over a shape group must equal the per-matrix equations exactly."""
    left_kind, right_kind = sides
    count, shape = 5, (6, 3)
    generator = torch.Generator().manual_seed(31)
    weight, gradient, momentum = (torch.randn(count, *shape, generator=generator) for _ in range(3))
    common = {"args": ARGS, "lr": 0.02, "weight_decay": 0.1}

    def side(kind, width):
        """Per-matrix (inner, outer) factor lists for one side."""
        if kind == "none":
            return [None] * count, [None] * count
        if kind == "diagonal":
            vectors = [torch.rand(width, generator=generator) + 0.2 for _ in range(count)]
            return vectors, [torch.rand(width, generator=generator) + 0.2 for _ in range(count)]
        pairs = []
        for _ in range(count):
            sample = torch.randn(width, width + 2, generator=generator)
            pairs.append(inverse_factors(sample @ sample.T, OutputSettings()))
        return [inner for inner, _ in pairs], [outer for _, outer in pairs]

    left_inner, left_outer = side(left_kind, shape[0])
    right_inner, right_outer = side(right_kind, shape[1])
    expected_momenta, expected = [], []
    for index in range(count):
        factors = MatrixFactors(left_inner[index], left_outer[index], right_inner[index], right_outer[index])
        moment, delta = block_update(weight[index], gradient[index], momentum[index], factors, **common)
        expected_momenta.append(moment)
        expected.append(delta)
    actual_momenta, actual = block_update_stack(
        weight,
        gradient,
        momentum,
        (
            stack_factors(left_inner, leading=count, width=shape[0]),
            stack_factors(left_outer, leading=count, width=shape[0]),
        ),
        (
            stack_factors(right_inner, leading=count, width=shape[1]),
            stack_factors(right_outer, leading=count, width=shape[1]),
        ),
        **common,
    )
    torch.testing.assert_close(actual_momenta, torch.stack(expected_momenta), rtol=0, atol=0)
    # Batched GEMMs reassociate FP32 sums; only that reduction order differs from the per-matrix path.
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("shape", [(5, 9), (9, 5)])
def test_left_only_update_uses_left_shaped_decay(shape):
    generator = torch.Generator().manual_seed(8)
    weight, gradient, momentum = (torch.randn(*shape, generator=generator) for _ in range(3))
    li, lo, _, _ = random_factors(*shape, generator)
    _, actual = block_update(
        weight, gradient, momentum, MatrixFactors(left_inner=li, left_outer=lo), args=ARGS, lr=0.03, weight_decay=0.1
    )
    source = li @ (gradient + 0.95 * (0.95 * momentum + gradient))
    direction = unit(lo) @ newton_schulz_batched(source[None], 10, 1e-5, "f32")[0]
    decay = unit(lo) @ unit(lo) @ weight
    scale = max(1.0, shape[0] / shape[1]) ** 0.5
    torch.testing.assert_close(actual, -0.03 * (scale * direction + 0.1 * decay), rtol=1e-5, atol=1e-6)


def test_identity_left_factors_reduce_two_sided_update_to_input_muont():
    generator = torch.Generator().manual_seed(13)
    weight, gradient, momentum = (torch.randn(4, 6, generator=generator) for _ in range(3))
    _, _, ri, ro = random_factors(4, 6, generator)
    eye = torch.eye(4)
    common = {"args": ARGS, "lr": 0.02, "weight_decay": 0.1}
    _, identity = block_update(weight, gradient, momentum, MatrixFactors(eye, eye, ri, ro), **common)
    _, null = block_update(weight, gradient, momentum, MatrixFactors(right_inner=ri, right_outer=ro), **common)
    source = (gradient + 0.95 * (0.95 * momentum + gradient)) @ ri
    muont = newton_schulz_batched(source[None], 10, 1e-5, "f32")[0] @ unit(ro)
    torch.testing.assert_close(identity, null, rtol=0, atol=0)
    torch.testing.assert_close(null, -0.02 * (muont + 0.1 * weight @ unit(ro) @ unit(ro)), rtol=1e-5, atol=1e-6)


def test_diagonal_update_equals_dense_update_with_diagonal_factors():
    generator = torch.Generator().manual_seed(21)
    weight, gradient, momentum = (torch.randn(8, 4, generator=generator) for _ in range(3))
    vectors = [torch.rand(n, generator=generator) + 0.1 for n in (8, 8, 4, 4)]
    common = {"args": ARGS, "lr": 0.02, "weight_decay": 0.1}
    _, diagonal = block_update(weight, gradient, momentum, MatrixFactors(*vectors), **common)
    _, dense = block_update(weight, gradient, momentum, MatrixFactors(*(torch.diag(v) for v in vectors)), **common)
    torch.testing.assert_close(diagonal, dense, rtol=1e-5, atol=1e-7)


def test_diagonal_roots_match_dense_roots_of_the_diagonal_moment():
    values = torch.tensor([[0.0, 0.2, 1.0, 7.0], [3.0, 3.0, 3.0, 3.0]], dtype=torch.float64)
    inner, outer = diagonal_roots(values, 0.001, 0.5, 0.125)
    for index in range(2):
        dense_inner, dense_outer = inverse_factors(torch.diag(values[index]), OutputSettings())
        torch.testing.assert_close(inner[index], dense_inner.diagonal(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(unit(outer[index]), dense_outer.diagonal(), rtol=1e-12, atol=1e-12)
    scaled_inner, scaled_outer = diagonal_roots(values * 1e-9, 0.001, 0.5, 0.125)
    torch.testing.assert_close(scaled_inner, inner, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(scaled_outer, outer, rtol=1e-10, atol=1e-12)
    zero_inner, zero_outer = diagonal_roots(torch.zeros(1, 3, dtype=torch.float64), 0.001, 0.5, 0.125)
    torch.testing.assert_close(zero_inner, torch.ones(1, 3, dtype=torch.float64))
    torch.testing.assert_close(zero_outer, torch.ones(1, 3, dtype=torch.float64))


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "inductor"])
def test_row_observer_stores_exact_scaled_cotangent_rows_per_microbatch(backend):
    generator = torch.Generator().manual_seed(3)
    linear = nn.Linear(4, 6, bias=False).to(torch.bfloat16)
    reference = copy.deepcopy(linear)
    positions = torch.tensor([0, 2])
    rows = torch.zeros(8, 6, dtype=torch.bfloat16)
    at = torch.zeros(4, dtype=torch.int64)
    observed = torch.compile(RowObservedLinear(linear, rows, at, positions), backend=backend, fullgraph=True)
    expected = torch.zeros_like(rows)
    for index in range(2):
        at.copy_(torch.arange(index * 4, index * 4 + 4))
        x = torch.randn(2, 3, 4, generator=generator).to(torch.bfloat16)
        cotangent = torch.randn(2, 3, 6, generator=generator).to(torch.bfloat16)
        output = observed(x)
        output.backward(cotangent)
        reference(x).backward(cotangent)
        expected[index * 4 : index * 4 + 4] = cotangent[:, positions].reshape(4, 6) * ROW_SCALE
    torch.testing.assert_close(rows, expected, rtol=0, atol=0)
    torch.testing.assert_close(linear.weight.grad, reference.weight.grad, rtol=0, atol=0)
    # Evaluation forwards never write rows.
    observed(torch.zeros(2, 3, 4, dtype=torch.bfloat16)).detach()
    torch.testing.assert_close(rows, expected, rtol=0, atol=0)


def test_diagonal_moments_pool_mean_squares_before_ema():
    rows = {"a": torch.tensor([[1.0, 2.0], [3.0, 0.0]]).to(torch.bfloat16)}
    moments = DiagonalMoments(
        rows, decay=0.5, damping=0.001, inner_power=0.5, outer_power=0.125, rows_total=2, world=1, device="cpu"
    )
    moments.update()
    rows["a"].mul_(2)
    moments.update()
    first = torch.tensor([5.0, 2.0])
    torch.testing.assert_close(moments.moment[2][0], 0.25 * first + 0.5 * 4 * first)
    inner, outer = moments.factors()["a"]
    expected_inner, expected_outer = diagonal_roots(moments.moment[2], 0.001, 0.5, 0.125)
    torch.testing.assert_close(inner, expected_inner[0])
    torch.testing.assert_close(outer, expected_outer[0])


def distributed_diagonal_worker(rank, rendezvous, directory):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    rows = {"hidden": torch.full((2, 3), float(rank + 1)).to(torch.bfloat16), "wide": torch.ones(2, 4)}
    rows["wide"] = (rows["wide"] * (rank + 2)).to(torch.bfloat16)
    moments = DiagonalMoments(
        rows, decay=0.5, damping=0.001, inner_power=0.5, outer_power=0.125, rows_total=4, world=2, device="cpu"
    )
    moments.update()
    torch.save(moments.state_dict(), directory / f"diagonal_{rank}.pt")
    dist.destroy_process_group()


def test_distributed_diagonal_moments_pool_all_rank_rows(tmp_path):
    mp.spawn(distributed_diagonal_worker, args=(f"file://{tmp_path / 'rendezvous'}", tmp_path), nprocs=2, join=True)
    for rank in range(2):
        state = torch.load(tmp_path / f"diagonal_{rank}.pt", weights_only=True)
        torch.testing.assert_close(state["moment"][3], torch.full((1, 3), 0.5 * (2 * 1 + 2 * 4) / 4))
        torch.testing.assert_close(state["moment"][4], torch.full((1, 4), 0.5 * (2 * 4 + 2 * 9) / 4))
