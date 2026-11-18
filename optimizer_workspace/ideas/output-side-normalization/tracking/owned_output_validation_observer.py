# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed observations of full pooled moments and rank-owned output factors."""

import math

import torch
import torch.distributed as dist
from toner.fast.engine import layer_counts

from .block_toner import BLOCK_KINDS, DiagonalMoments, OwnedOutputMoments, block_update_stack
from .outer_diagonal_checks import fp64_diagonal_factors
from .upstream_full_model_checks import UpstreamProductVerifier, fp64_output_delta

WORLD = 8
TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
CHECKS = (
    "global_keys_complete",
    "full_moment_schema_passed",
    "full_moment_parity_passed",
    "moment_weight_exact",
    "owned_coverage_exact",
    "owned_outer_factors_exact",
    "owned_outer_factors_fp64_passed",
    "inner_factors_absent",
)
ERRORS = ("moment_torch", "moment_fp64", "factor_same_state", "factor_fp64", "weight")
UPSTREAM_CHECKS = (
    "upstream_product_parity_passed",
    "upstream_scheduled_scalars_passed",
    "upstream_pinned_input_exact",
    "upstream_output_equation_passed",
)


def _projection_shape(kind, model):
    hidden, mlp = model["hidden"], model["mlp"]
    if kind in ("gate", "up"):
        return mlp, hidden
    if kind == "down":
        return hidden, mlp
    return hidden, hidden


def owned_output_layout(model, world, rank):
    """Derive the global moment schema and local projection partition from model dimensions."""
    if (
        set(model) != {"layers", "hidden", "mlp"}
        or any(type(value) is not int or value <= 0 for value in model.values())
        or type(world) is not int
        or world != WORLD
        or model["layers"] % world
        or type(rank) is not int
        or not 0 <= rank < world
    ):
        raise ValueError("Owned output observations require a positive model and eight equal rank partitions")
    projections = {
        (layer, kind): _projection_shape(kind, model) for layer in range(model["layers"]) for kind in BLOCK_KINDS
    }
    counts = layer_counts(model["layers"], world)
    first, last = sum(counts[:rank]), sum(counts[: rank + 1])
    owned = {key: shape for key, shape in projections.items() if first <= key[0] < last}
    widths = {}
    shapes = {}
    for key, shape in projections.items():
        widths.setdefault(shape[0], []).append(key)
    for shape in owned.values():
        shapes[shape] = shapes.get(shape, 0) + 1
    return {
        "global_keys": [list(key) for key in projections],
        "global_count": len(projections),
        "global_shapes": [{"width": width, "moment_shape": [len(keys), width]} for width, keys in widths.items()],
        "owned_keys": [list(key) for key in owned],
        "owned_count": len(owned),
        "owned_shapes": [{"shape": list(shape), "count": count} for shape, count in shapes.items()],
    }


def _matches(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_matches(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_matches(a, e) for a, e in zip(actual, expected, strict=True))
    return actual == expected


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _weight(decay, step):
    if type(decay) not in (int, float) or not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("Owned output EMA decay must be finite and in [0, 1)")
    weight = 0.0
    for _ in range(step):
        weight = decay * weight + 1 - decay
    return weight


def validate_owned_output_records(records, *, model, world, step, apply_left, backend, device_type, decay=0.938):
    """Validate one update's gathered wire records against independently supplied execution context."""
    if (
        type(step) is not int
        or step < 1
        or type(apply_left) is not bool
        or (backend, device_type) not in (("triton", "cuda"), ("torch", "cpu"))
        or type(records) is not list
        or len(records) != world
    ):
        raise ValueError("Owned output gathered evidence has invalid execution context or rank coverage")
    expected_weight = _weight(decay, step)
    maxima = dict.fromkeys((*ERRORS, "upstream_output_equation"), 0.0)
    for rank, record in enumerate(records):
        expected = {
            "schema": 1,
            "rank": rank,
            "world": world,
            "step": step,
            "model": model,
            "backend": backend,
            "device_type": device_type,
            "state_schema": ["moment", "weight"],
            **owned_output_layout(model, world, rank),
        }
        if (
            type(record) is not dict
            or record.keys() != expected.keys() | {"weight", "checks", "max_errors", "upstream"}
            or any(not _matches(record.get(key), value) for key, value in expected.items())
            or not _finite(record["weight"])
            or record["weight"] != expected_weight
            or type(record["checks"]) is not dict
            or record["checks"].keys() != set(CHECKS)
            or any(record["checks"][name] is not True for name in CHECKS)
            or type(record["max_errors"]) is not dict
            or record["max_errors"].keys() != set(ERRORS)
            or any(not _finite(value) for value in record["max_errors"].values())
            or record["max_errors"]["factor_same_state"] != 0
            or record["max_errors"]["weight"] != 0
        ):
            raise ValueError(f"Owned output rank {rank} schema, inventory, weight or numerical evidence differs")
        upstream = record["upstream"]
        if (
            type(upstream) is not dict
            or upstream.keys()
            != set(UPSTREAM_CHECKS)
            | {"upstream_null_exact", "upstream_owned_stacks_checked", "upstream_output_equation_max_error"}
            or any(upstream.get(name) is not True for name in UPSTREAM_CHECKS)
            or upstream.get("upstream_null_exact") is not (not apply_left)
            or type(upstream.get("upstream_owned_stacks_checked")) is not int
            or upstream["upstream_owned_stacks_checked"] != len(expected["owned_shapes"])
            or not _finite(upstream.get("upstream_output_equation_max_error"))
        ):
            raise ValueError(f"Owned output rank {rank} lacks complete upstream update evidence")
        for name, error in record["max_errors"].items():
            maxima[name] = max(maxima[name], error)
        maxima["upstream_output_equation"] = max(
            maxima["upstream_output_equation"], upstream["upstream_output_equation_max_error"]
        )
    return {
        "owned_output_records": records,
        "owned_output_world": world,
        "owned_output_step": step,
        "owned_output_global_count": records[0]["global_count"],
        "owned_output_global_shapes": records[0]["global_shapes"],
        "owned_output_parity_passed": True,
        "owned_output_max_errors": maxima,
    }


class OwnedOutputVerifier(UpstreamProductVerifier):
    """Observe retained global statistics, exact local factors and all-rank update evidence."""

    def __init__(self, fast):
        actual = fast.output_diagonal
        device = torch.device(fast.device)
        if (
            not fast.arm.upstream_product_kernels
            or fast.arm.apply_left_inner
            or fast.arm.lag_left
            or fast.output is not None
            or not isinstance(actual, OwnedOutputMoments)
            or (actual.backend, device.type) not in (("triton", "cuda"), ("torch", "cpu"))
            or actual.world != fast.world
        ):
            raise ValueError("Owned output verification requires owned outer-only moments and a supported backend")
        self.fast, self.actual = fast, actual
        self.model = {"layers": fast.input.L, "hidden": fast.input.groups["d"].n, "mlp": fast.input.groups["f"].n}
        self.layout = owned_output_layout(self.model, fast.world, fast.rank)
        self.width_keys = {}
        for layer, kind in self.layout["global_keys"]:
            width = self.model["mlp"] if kind in ("gate", "up") else self.model["hidden"]
            self.width_keys.setdefault(width, []).append((layer, kind))
        self.owned_keys = {tuple(key) for key in self.layout["owned_keys"]}
        self.expected = {tuple(item["shape"]) for item in self.layout["owned_shapes"]}
        self._require_inventory()
        self._require_state(actual.state_dict(), fast.steps)
        settings = {
            "decay": actual.decay,
            "damping": actual.damping,
            "inner_power": actual.inner_power,
            "outer_power": actual.outer_power,
            "rows_total": actual.rows_total,
            "device": device,
            "backend": "torch",
        }
        # Independent increments use the real collective; exact factoring uses a
        # separate full reference loaded from identical pooled state, without reduction.
        if device.type == "cuda":
            super().__init__(fast)
        else:
            self.reference = DiagonalMoments(actual.bufs, world=fast.world, **settings)
            self.reference.load_state_dict(actual.state_dict())
        self.same_state = DiagonalMoments(actual.bufs, world=1, **settings)
        self.fp64_moment = {width: moment.double().clone() for width, moment in actual.moment.items()}
        self.factors, self.checked, self.checked_keys = {}, set(), set()
        self.observed_step = None
        self.factor_errors = None
        self.max_equation_error = 0.0

    def _require_inventory(self):
        if self.actual.keys != self.width_keys or set(self.actual.bufs) != {
            tuple(key) for key in self.layout["global_keys"]
        }:
            raise ValueError("Owned output buffers or keys do not retain the complete model-derived global inventory")
        for width, keys in self.width_keys.items():
            if any(
                self.actual.bufs[key].ndim != 2
                or self.actual.bufs[key].shape[-1] != width
                or self.actual.bufs[key].shape[0] < 1
                or self.actual.bufs[key].device != torch.device(self.fast.device)
                for key in keys
            ):
                raise ValueError("Owned output row shapes or devices differ from the model")
        if self.fast.layer_counts != layer_counts(self.model["layers"], self.fast.world):
            raise ValueError("Owned output layer ownership differs from the model-derived partition")
        observed = []
        for shape, per_rank in self.fast.shape_groups:
            if len(per_rank) != self.fast.world:
                raise ValueError("Owned output shape groups omit ranks")
            for rank, entries in enumerate(per_rank):
                observed.extend((rank, layer, kind, tuple(shape)) for layer, kind, _ in entries)
                if any(tuple(parameter.shape) != tuple(shape) for _, _, parameter in entries):
                    raise ValueError("Owned output projection parameters differ from their declared shape")
        expected = []
        for rank in range(self.fast.world):
            layout = owned_output_layout(self.model, self.fast.world, rank)
            for layer, kind in layout["owned_keys"]:
                expected.append((rank, layer, kind, _projection_shape(kind, self.model)))
        if sorted(observed) != sorted(expected):
            raise ValueError("Owned output shape groups do not cover exactly the model-derived rank partitions")

    def _require_state(self, state, step):
        if (
            type(step) is not int
            or step < 0
            or type(state) is not dict
            or state.keys() != {"moment", "weight"}
            or type(state["moment"]) is not dict
            or state["moment"].keys() != self.width_keys.keys()
            or self.actual.moment.keys() != self.width_keys.keys()
        ):
            raise ValueError("Owned output full moment state schema is incomplete")
        for width, keys in self.width_keys.items():
            for moment in (state["moment"][width], self.actual.moment[width]):
                if (
                    not isinstance(moment, torch.Tensor)
                    or tuple(moment.shape) != (len(keys), width)
                    or moment.dtype != torch.float32
                    or moment.device != torch.device(self.fast.device)
                    or not torch.isfinite(moment).all().item()
                    or (moment < 0).any().item()
                ):
                    raise ValueError("Owned output full moment tensor shape, dtype, device or values differ")
        if not _finite(state["weight"]) or state["weight"] != _weight(self.actual.decay, step):
            raise ValueError("Owned output moment weight differs from the exact update recurrence")

    @torch.no_grad()
    def begin(self):
        self._require_inventory()
        self._require_state(self.actual.state_dict(), self.fast.steps)
        self.observed_step = self.fast.steps + 1
        self.checked, self.checked_keys = set(), set()
        self.factor_errors = None
        self.max_equation_error = 0.0
        if torch.device(self.fast.device).type == "cuda":
            super().begin()
            return
        for width, keys in self.width_keys.items():
            expected = torch.stack([self.actual.bufs[key].float().square().sum(0) for key in keys])
            local = self.actual.reducers[width]() if self.actual.backend == "triton" else expected
            reference = torch.stack([self.actual.bufs[key].double().square().sum(0) for key in keys])
            torch.testing.assert_close(local, expected, **TOLERANCE)
            torch.testing.assert_close(local.double(), reference, **TOLERANCE)
            dist.all_reduce(reference, group=self.reference.pg)
            self.fp64_moment[width].mul_(self.actual.decay).add_(
                reference, alpha=(1 - self.actual.decay) / self.actual.rows_total
            )
        self.reference.update()
        self.factors = self.reference.factors()

    @torch.no_grad()
    def check_factors(self):
        if self.observed_step is None:
            raise ValueError("Owned output factor checks require a fresh observation boundary")
        self._require_inventory()
        state = self.actual.state_dict()
        self._require_state(state, self.observed_step)
        self.same_state.load_state_dict(state)
        original = self.same_state.factors()
        factors = self.actual.factors()
        if set(factors) != self.owned_keys:
            raise ValueError("Owned output factors do not cover exactly the locally updated projections")
        errors = dict.fromkeys(ERRORS, 0.0)
        if state["weight"] != self.reference.weight:
            raise ValueError("Owned output weight differs from the independent Torch accumulation")
        for width, keys in self.width_keys.items():
            moment = state["moment"][width]
            torch.testing.assert_close(moment, self.reference.moment[width], **TOLERANCE)
            torch.testing.assert_close(moment.double(), self.fp64_moment[width], **TOLERANCE)
            errors["moment_torch"] = max(
                errors["moment_torch"], (moment.double() - self.reference.moment[width].double()).abs().max().item()
            )
            errors["moment_fp64"] = max(
                errors["moment_fp64"], (moment.double() - self.fp64_moment[width]).abs().max().item()
            )
            exact = fp64_diagonal_factors(
                self.fp64_moment[width], self.actual.damping, self.actual.inner_power, self.actual.outer_power
            )[1]
            for index, key in enumerate(keys):
                if key not in self.owned_keys:
                    continue
                inner, outer = factors[key]
                if inner is not None:
                    raise ValueError("Owned outer-only output factors constructed an unused inner root")
                torch.testing.assert_close(outer, original[key][1], rtol=0, atol=0)
                torch.testing.assert_close(outer.double(), exact[index], **TOLERANCE)
                errors["factor_fp64"] = max(errors["factor_fp64"], (outer.double() - exact[index]).abs().max().item())
        if any(not _finite(error) for error in errors.values()):
            raise ValueError("Owned output numerical errors must be finite")
        self.factor_errors = errors

    @torch.no_grad()
    def __call__(self, mine, weights, gradients, momentum, actual_momentum, delta, right, trace, lr, weight_decay):
        shape = tuple(weights.shape[-2:])
        expected = {
            (layer, kind)
            for group_shape, per_rank in self.fast.shape_groups
            if tuple(group_shape) == shape
            for layer, kind, _ in per_rank[self.fast.rank]
        }
        keys = [(layer, kind) for layer, kind, _ in mine]
        if len(keys) != len(expected) or set(keys) != expected or self.checked_keys.intersection(keys):
            raise ValueError("Owned output captured update omits, duplicates or repeats local projections")
        stacks = (weights, gradients, momentum, actual_momentum, delta)
        if any(tuple(tensor.shape) != (len(keys), *shape) for tensor in stacks):
            raise ValueError("Owned output captured tensors do not cover the complete local projection stack")
        if weights.device.type == "cuda":
            super().__call__(mine, weights, gradients, momentum, actual_momentum, delta, right, trace, lr, weight_decay)
        else:
            self._check_cpu_update(
                mine, weights, gradients, momentum, actual_momentum, delta, right, trace, lr, weight_decay
            )
        self.checked_keys.update(keys)

    def _check_cpu_update(
        self, mine, weights, gradients, momentum, actual_momentum, delta, right, trace, lr, weight_decay
    ):
        """Use the existing Torch reference, never the CUDA-only pinned product kernels."""
        if not self.checked:
            self.check_factors()
        scheduled = torch.tensor([lr, weight_decay], dtype=self.fast.update_scalars.dtype)
        torch.testing.assert_close(self.fast.update_scalars, scheduled, rtol=0, atol=0)
        reference_trace = {}
        expected_momentum, null_delta = block_update_stack(
            weights,
            gradients,
            momentum,
            (None, None),
            right,
            args=self.fast.args,
            lr=scheduled[0],
            weight_decay=scheduled[1],
            trace=reference_trace,
        )
        torch.testing.assert_close(actual_momentum, expected_momentum, rtol=0, atol=0)
        for name in ("ns_source", "ns_output"):
            torch.testing.assert_close(trace[name], reference_trace[name], rtol=0, atol=0)
        outer = None
        if self.fast.arm.apply_left:
            outer = torch.stack([self.actual.factors()[layer, kind][1] for layer, kind, _ in mine])
        else:
            torch.testing.assert_close(delta, null_delta, rtol=0, atol=0)
        equation = fp64_output_delta(
            weights, reference_trace["ns_output"], outer, right[1], lr=scheduled[0], decay=scheduled[1]
        )
        torch.testing.assert_close(delta.double(), equation, **TOLERANCE)
        self.max_equation_error = max(self.max_equation_error, (delta.double() - equation).abs().max().item())
        self.checked.add(tuple(weights.shape[-2:]))

    def finish(self):
        upstream = super().finish()
        if self.factor_errors is None or self.checked_keys != self.owned_keys or self.fast.steps != self.observed_step:
            raise ValueError("Owned output update lacks fresh complete factor and projection evidence")
        self.check_factors()
        record = {
            "schema": 1,
            "rank": self.fast.rank,
            "world": self.fast.world,
            "step": self.observed_step,
            "model": dict(self.model),
            "backend": self.actual.backend,
            "device_type": torch.device(self.fast.device).type,
            **self.layout,
            "state_schema": ["moment", "weight"],
            "weight": self.actual.weight,
            "checks": dict.fromkeys(CHECKS, True),
            "max_errors": dict(self.factor_errors),
            "upstream": upstream,
        }
        records = [None] * self.fast.world
        dist.all_gather_object(records, record)
        summary = validate_owned_output_records(
            records,
            model=self.model,
            world=self.fast.world,
            step=self.observed_step,
            apply_left=self.fast.arm.apply_left,
            backend=self.actual.backend,
            device_type=torch.device(self.fast.device).type,
            decay=self.actual.decay,
        )
        upstream["upstream_output_equation_max_error"] = summary["owned_output_max_errors"]["upstream_output_equation"]
        return {**upstream, **summary}
