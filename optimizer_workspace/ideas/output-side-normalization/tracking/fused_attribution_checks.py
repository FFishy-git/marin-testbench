# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU replay gates for the candidate's actual fused output/null optimizer step."""

import io
from types import SimpleNamespace

import torch
from experiment import block_toner
from experiment.block_toner import BLOCK_ARMS, INPUT_KIND, BlockArm, BlockToner, DiagonalMoments, MatrixFactors, Side
from torch.utils._python_dispatch import TorchDispatchMode

NULL_ARM = "null_leftdiag_fused"
OUTPUT_ARM = "outer_leftdiag_fused"
RTOL = 1e-5
ATOL = 1e-6
RATES = (0.06399999989260598, 0.048, 0.032, 0.016, 0.002, 0.0)
CHECKPOINT_UPDATE = 3
KINDS_BY_SHAPE = (((4, 4), ("q", "k", "v", "o")), ((8, 4), ("gate", "up")), ((4, 8), ("down",)))


class _CpuStream:
    def synchronize(self) -> None:
        pass


class _ReplayInputRoots:
    """Replay dense, nonidentity input factors through the real root/state interfaces."""

    def __init__(self, keys: list[tuple[int, str]], generator: torch.Generator):
        self.groups = {}
        for width in (4, 8):
            own = [key for key in keys if (8 if key[1] == "down" else 4) == width]
            sample = torch.randn(len(own), width, width, generator=generator)
            covariance = sample @ sample.mT / width + torch.eye(width)
            self.groups[str(width)] = SimpleNamespace(
                own=own,
                C=covariance,
                tol=torch.full((len(own), 3), 1e-4),
                basis_step=[0] * len(own),
                pending=[None] * len(own),
                warm=False,
                graph=None,
                basis=None,
            )
        self.side = _CpuStream()
        self.state = {}
        self.weight = 0.0
        self.steps = self.t = self.fallbacks = self.extra = 0
        self.refresh_install_age = 1
        self.refresh_installations = []
        self.begun = False
        self.device = torch.device("cpu")

    def begin(self) -> None:
        self.begun = True
        self.weight = 0.938 * self.weight + 1 - 0.938

    def replay_factors(self) -> dict:
        result = {}
        for group in self.groups.values():
            width = group.C.shape[-1]
            inner = group.C + (0.01 * self.steps) * torch.eye(width)
            outer = 0.6 * group.C + 0.4 * torch.eye(width)
            result |= {key: (inner[index], outer[index]) for index, key in enumerate(group.own)}
        return result

    def _finish_roots(self) -> dict:
        replay = self.replay_factors()
        return {
            name: tuple(torch.stack([replay[key][side] for key in group.own]) for side in (0, 1))
            for name, group in self.groups.items()
        }


class _ObservedDiagonalMoments(DiagonalMoments):
    """Retain the actual returned factor values, without a second post-step root call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.computed_factors = {}

    def factors(self) -> dict:
        result = super().factors()
        self.computed_factors = {key: tuple(value.clone() for value in pair) for key, pair in result.items()}
        return result


class _OutputWorkTrace(TorchDispatchMode):
    """Observe actual packing, norm and row-product tensors without replacing kernels."""

    def __init__(self):
        super().__init__()
        self.packed, self.normalized_inputs, self.row_products = [], [], []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if function == torch.ops.aten.stack.default and result.ndim == 2:
            self.packed.append(result.detach().clone())
        elif function == torch.ops.aten.linalg_vector_norm.default and args[0].ndim == 2:
            self.normalized_inputs.append(args[0].detach().clone())
        elif function == torch.ops.aten.mul.Tensor:
            first, second = args
            if (
                isinstance(first, torch.Tensor)
                and isinstance(second, torch.Tensor)
                and first.ndim == second.ndim == 3
                and first.shape[:2] == second.shape[:2]
                and first.shape[-1] > 1
                and second.shape[-1] == 1
            ):
                self.row_products.append((tuple(first.shape), second.detach().clone()))
        return result


class _CpuStepHolder:
    step = BlockToner.step
    begin = BlockToner.begin
    finish_backward = BlockToner.finish_backward
    _factors = BlockToner._factors
    _engine_factors = BlockToner._engine_factors
    _engine_metrics = BlockToner._engine_metrics
    engines = BlockToner.engines
    runtime_state = BlockToner.runtime_state
    restore_runtime_state = BlockToner.restore_runtime_state

    def __init__(self, arm: str, *, alternate_output: bool = False, nesterov: bool = True):
        self.arm = BLOCK_ARMS[arm]
        self.generator = torch.Generator().manual_seed(871)
        self.alternate_output = alternate_output
        self.args = SimpleNamespace(
            muon_momentum=0.95, muon_nesterov=nesterov, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32"
        )
        self.world, self.rank, self.device, self.pg = 1, 0, torch.device("cpu"), None
        self.output = self.input_diagonal = None
        self.lagged, self.pending_diagonal, self.last_metrics = {}, {}, {}
        self.steps, self.update_verifier = 0, None
        self.params, self.shape_groups, self.momentum, buffers = [], [], {}, {}
        input_keys = []
        for shape, kinds in KINDS_BY_SHAPE:
            mine = []
            for layer in range(2):
                for kind in kinds:
                    parameter = torch.nn.Parameter(torch.randn(*shape, generator=self.generator) * 0.1)
                    mine.append((layer, kind, parameter))
                    self.params.append(parameter)
                    buffers[layer, kind] = torch.zeros(8, shape[0], dtype=torch.bfloat16)
                    key = layer, INPUT_KIND[kind]
                    if key not in input_keys:
                        input_keys.append(key)
            self.shape_groups.append((shape, [mine]))
            self.momentum[shape] = torch.randn(len(mine), *shape, generator=self.generator) * 0.03
        self.input = _ReplayInputRoots(input_keys, self.generator)
        self.output_diagonal = _ObservedDiagonalMoments(
            buffers,
            decay=0.938,
            damping=0.001,
            inner_power=0.5,
            outer_power=0.125,
            rows_total=8,
            world=1,
            device=self.device,
            backend="torch",
        )
        for width, moment in self.output_diagonal.moment.items():
            coordinate = torch.logspace(-3, 3, width)
            if alternate_output:
                coordinate = coordinate.flip(0)
            moment.copy_(torch.linspace(0.8, 1.2, len(moment))[:, None] * coordinate)
        self.output_diagonal.weight = 0.7
        self.param_groups = [{"params": self.params, "lr": RATES[0], "weight_decay": 0.1}]

    def feed(self, lr: float) -> None:
        self.param_groups[0]["lr"] = lr
        for parameter in self.params:
            parameter.grad = torch.randn(parameter.shape, generator=self.generator) * 0.03
        for buffer in self.output_diagonal.bufs.values():
            coordinate = torch.logspace(-1, 1, buffer.shape[-1])
            if self.alternate_output:
                coordinate = coordinate.flip(0)
            buffer.copy_(torch.randn(buffer.shape, generator=self.generator) * coordinate)
        self.output_diagonal.computed_factors = {}

    def checkpoint_state(self) -> dict:
        return {
            "weights": [parameter.detach().clone() for parameter in self.params],
            "runtime": self.runtime_state(),
            "rng": self.generator.get_state(),
            "shape_groups": [
                (shape, [(layer, kind) for layer, kind, _ in per_rank[0]]) for shape, per_rank in self.shape_groups
            ],
        }


def _restore_checkpoint(holder: _CpuStepHolder, payload: bytes) -> None:
    saved = torch.load(io.BytesIO(payload), weights_only=True)
    if saved["shape_groups"] != holder.checkpoint_state()["shape_groups"]:
        raise ValueError("CPU replay checkpoint shape ownership differs")
    with torch.no_grad():
        for parameter, value in zip(holder.params, saved["weights"], strict=True):
            parameter.copy_(value)
    holder.restore_runtime_state(saved["runtime"])
    holder.generator.set_state(saved["rng"])


def _assert_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            _assert_equal(value, reference)
    else:
        assert actual == expected


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=RTOL, atol=ATOL)
    return (actual.double() - expected.double()).abs().max().item()


def _tolerance_fraction(actual: torch.Tensor, expected: torch.Tensor) -> float:
    reference = expected.to(actual.dtype).double()
    return ((actual.double() - reference).abs() / (ATOL + RTOL * reference.abs())).max().item()


def _expected_output(holder: _CpuStepHolder) -> tuple[dict, dict]:
    diagonal = holder.output_diagonal
    moments, factors = {}, {}
    for width, old in diagonal.moment.items():
        rows = torch.stack([diagonal.bufs[key].double().square().sum(0) for key in diagonal.keys[width]])
        moments[width] = old.double() * diagonal.decay + rows * ((1 - diagonal.decay) / diagonal.rows_total)
        for index, key in enumerate(diagonal.keys[width]):
            values = moments[width][index]
            normalized = values / values.mean() + diagonal.damping
            factors[key] = (normalized.pow(-diagonal.inner_power), normalized.pow(-diagonal.outer_power))
    return moments, factors


def _checked_step(holder: _CpuStepHolder) -> dict:
    moments, factors = _expected_output(holder)
    old_moments = {width: value.clone() for width, value in holder.output_diagonal.moment.items()}
    right = holder.input.replay_factors()
    expected, expected_momentum = [], {}
    group = holder.param_groups[0]
    for shape, per_rank in holder.shape_groups:
        momenta = []
        for index, (layer, kind, parameter) in enumerate(per_rank[0]):
            ri, ro = right[layer, INPUT_KIND[kind]]
            outer = None if holder.arm.identity_left_outer else factors[layer, kind][1].float()
            matrix_factors = MatrixFactors(left_outer=outer, right_inner=ri, right_outer=ro)
            momentum, delta = block_toner.block_update(
                parameter.detach(),
                parameter.grad,
                holder.momentum[shape][index],
                matrix_factors,
                args=holder.args,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
            )
            normalized = ro / (ro.square().sum() / ro.shape[-1]).sqrt()
            _, without_decay = block_toner.block_update(
                parameter.detach(),
                parameter.grad,
                holder.momentum[shape][index],
                matrix_factors,
                args=holder.args,
                lr=group["lr"],
                weight_decay=0.0,
            )
            decay = -group["lr"] * group["weight_decay"] * (parameter.detach() @ normalized @ normalized)
            torch.testing.assert_close(delta - without_decay, decay, rtol=RTOL, atol=ATOL)
            expected.append(parameter.detach().clone() + delta)
            momenta.append(momentum)
        expected_momentum[shape] = torch.stack(momenta)
    previous_steps = holder.steps
    previous_weight = holder.output_diagonal.weight
    holder.begin()
    holder.finish_backward()
    work = _OutputWorkTrace()
    with work:
        holder.step()
    assert holder.steps == previous_steps + 1
    assert holder.input.steps == holder.input.t == holder.steps
    assert holder.output_diagonal.weight == (
        holder.output_diagonal.decay * previous_weight + 1 - holder.output_diagonal.decay
    )
    errors = {
        "parameter_max_error": 0.0,
        "moment_max_error": 0.0,
        "factor_max_error": 0.0,
        "maximum_tolerance_fraction": 0.0,
    }
    for parameter, reference in zip(holder.params, expected, strict=True):
        errors["parameter_max_error"] = max(errors["parameter_max_error"], _max_error(parameter.detach(), reference))
        errors["maximum_tolerance_fraction"] = max(
            errors["maximum_tolerance_fraction"], _tolerance_fraction(parameter.detach(), reference)
        )
    for shape, reference in expected_momentum.items():
        torch.testing.assert_close(holder.momentum[shape], reference, rtol=0, atol=0)
    for width, reference in moments.items():
        actual = holder.output_diagonal.moment[width]
        assert not torch.equal(actual, old_moments[width])
        errors["moment_max_error"] = max(errors["moment_max_error"], _max_error(actual, reference))
        errors["maximum_tolerance_fraction"] = max(
            errors["maximum_tolerance_fraction"], _tolerance_fraction(actual, reference)
        )
    assert holder.output_diagonal.computed_factors.keys() == factors.keys()
    for key, pair in holder.output_diagonal.computed_factors.items():
        for actual, reference in zip(pair, factors[key], strict=True):
            errors["factor_max_error"] = max(errors["factor_max_error"], _max_error(actual, reference))
            errors["maximum_tolerance_fraction"] = max(
                errors["maximum_tolerance_fraction"], _tolerance_fraction(actual, reference)
            )
    for shape, per_rank in holder.shape_groups:
        mine = per_rank[0]
        outer = torch.stack([holder.output_diagonal.computed_factors[layer, kind][1] for layer, kind, _ in mine])
        applied = torch.ones_like(outer) if holder.arm.identity_left_outer else outer
        assert any(torch.equal(value, applied) for value in work.packed)
        assert any(torch.equal(value, applied) for value in work.normalized_inputs)
        products = [value for dimensions, value in work.row_products if dimensions == (len(mine), *shape)]
        assert len(products) == 1
        normalized = applied / applied.square().mean(-1, keepdim=True).sqrt()
        _max_error(products[0].squeeze(-1), normalized)
    return errors


def _run_replay(*, nesterov: bool) -> dict:
    null = _CpuStepHolder(NULL_ARM, nesterov=nesterov)
    output = _CpuStepHolder(OUTPUT_ARM, nesterov=nesterov)
    alternate = _CpuStepHolder(NULL_ARM, alternate_output=True, nesterov=nesterov)
    holders = (null, output, alternate)
    resumed = []
    errors = {
        "parameter_max_error": 0.0,
        "moment_max_error": 0.0,
        "factor_max_error": 0.0,
        "maximum_tolerance_fraction": 0.0,
    }
    minimum_difference = float("inf")
    for update, lr in enumerate(RATES, start=1):
        for holder in (*holders, *resumed):
            holder.feed(lr)
            measured = _checked_step(holder)
            for name, value in measured.items():
                errors[name] = max(errors[name], value)
        _assert_equal(null.generator.get_state(), output.generator.get_state())
        _assert_equal(null.generator.get_state(), alternate.generator.get_state())
        _assert_equal(null.input.replay_factors(), output.input.replay_factors())
        _assert_equal(null.output_diagonal.state_dict(), output.output_diagonal.state_dict())
        for first, second in zip(null.params, alternate.params, strict=True):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        _assert_equal(null.momentum, alternate.momentum)
        _assert_equal(null.momentum, output.momentum)
        for first, second in zip(null.params, output.params, strict=True):
            difference = (first.detach() - second.detach()).abs().max().item()
            assert difference > 1e-5
            minimum_difference = min(minimum_difference, difference)
        for original, restored in zip(holders, resumed, strict=False):
            _assert_equal(original.checkpoint_state(), restored.checkpoint_state())
        if update == CHECKPOINT_UPDATE:
            for original in holders:
                stream = io.BytesIO()
                torch.save(original.checkpoint_state(), stream)
                restored = _CpuStepHolder(
                    NULL_ARM if original is not output else OUTPUT_ARM,
                    alternate_output=original.alternate_output,
                    nesterov=nesterov,
                )
                _restore_checkpoint(restored, stream.getvalue())
                _assert_equal(restored.checkpoint_state(), original.checkpoint_state())
                resumed.append(restored)
    return errors | {"minimum_output_null_difference": minimum_difference, "checkpoint_exact": True}


def run_checks() -> dict:
    """Check actual null/output CPU steps and exact checkpointed replay continuation.

    Returns:
        Numerical error bounds and replay scope, after all behavioral assertions pass.
        This does not validate Triton reductions, real cached roots, or distributed CUDA.
    """
    if BLOCK_ARMS[NULL_ARM] != BlockArm(
        Side.FULL, Side.DIAGONAL, Side.DIAGONAL, True, apply_left_inner=False, identity_left_outer=True
    ):
        raise ValueError("Matched null must retain full input and diagonal output work with identity outer application")
    if BLOCK_ARMS[OUTPUT_ARM] != BlockArm(Side.FULL, Side.DIAGONAL, Side.DIAGONAL, True, apply_left_inner=False):
        raise ValueError("Output reference is not the frozen outer-only diagonal arm")
    rng = torch.random.get_rng_state().clone()
    cases = {str(nesterov): _run_replay(nesterov=nesterov) for nesterov in (True, False)}
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    return {
        "status": "fused_attribution_cpu_behavior_passed",
        "arms": [NULL_ARM, OUTPUT_ARM],
        "stack_shapes": [[2 * len(kinds), *shape] for shape, kinds in KINDS_BY_SHAPE],
        "updates": len(RATES),
        "checkpoint_update": CHECKPOINT_UPDATE,
        "momentum_max_error": 0.0,
        "null_output_invariance_max_error": 0.0,
        "rng_unchanged": True,
        "output_packing_normalization_application_verified": True,
        "rtol": RTOL,
        "atol": ATOL,
        "cases": cases,
        "scope": (
            "Actual BlockToner.step; replayed dense input roots; live Torch output moments; "
            "identity-outer null versus input-only reference; CPU finite NS."
        ),
    }
