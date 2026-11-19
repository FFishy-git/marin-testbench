# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execution-only checks of diagonal outer factors, never inner-fused approval."""

import torch
import torch.distributed as dist

from .block_toner import DiagonalMoments, block_update_stack, stack_unit_rms


def fp64_diagonal_factors(moment, damping, inner_power, outer_power):
    """Independent FP64 coordinate reference, including the zero-moment identity convention."""
    inner, outer = [], []
    for values in moment.double():
        if values.sum() == 0:
            inner.append(torch.ones_like(values))
            outer.append(torch.ones_like(values))
        else:
            normalized = values / values.mean() + damping
            inner.append(normalized ** (-inner_power))
            outer.append(normalized ** (-outer_power))
    return torch.stack(inner), torch.stack(outer)


def compare_outer_update(
    weights, gradients, momentum, actual_momentum, actual_delta, right, actual_trace, outer, *, args, lr, weight_decay
):
    """Compare the actual production stack to Torch outer factors with identical NS inputs."""
    reference_trace = {}
    expected_momentum, expected_delta = block_update_stack(
        weights,
        gradients,
        momentum,
        (None, outer),
        right,
        args=args,
        lr=lr,
        weight_decay=weight_decay,
        trace=reference_trace,
    )
    torch.testing.assert_close(actual_momentum, expected_momentum, rtol=0, atol=0)
    for key in ("ns_source", "ns_output"):
        torch.testing.assert_close(actual_trace[key], reference_trace[key], rtol=0, atol=0, msg=key)
    torch.testing.assert_close(actual_delta, expected_delta, rtol=0.003, atol=5e-5)
    normalized = stack_unit_rms(outer)
    torch.testing.assert_close(
        normalized.square().sum(-1),
        torch.full((outer.shape[0],), float(outer.shape[1]), device=outer.device),
        rtol=1e-5,
        atol=1e-6,
    )
    return {
        "ns_source_exact": True,
        "ns_output_exact": True,
        "momentum_exact": True,
        "max_update_error": (actual_delta - expected_delta).abs().max().item(),
    }


class OuterDiagonalVerifier:
    """Shadow captured output rows with Torch and FP64, before applying parameter updates."""

    def __init__(self, fast):
        if fast.arm.apply_left_inner or fast.arm.lag_left or fast.output is not None:
            raise ValueError("Outer-only verification cannot approve inner or lagged output updates")
        actual = fast.output_diagonal
        if actual is None or actual.backend != "triton" or fast.input is None:
            raise ValueError("Outer-only verification requires fused diagonal output and full input factors")
        self.fast = fast
        self.actual = actual
        self.reference = DiagonalMoments(
            actual.bufs,
            decay=actual.decay,
            damping=actual.damping,
            inner_power=actual.inner_power,
            outer_power=actual.outer_power,
            rows_total=actual.rows_total,
            world=actual.world,
            device=fast.device,
            backend="torch",
        )
        self.reference.load_state_dict(actual.state_dict())
        self.fp64_moment = {width: moment.double().clone() for width, moment in actual.moment.items()}
        self.factors = {}
        self.checked = set()
        self.expected = {shape for shape, per_rank in fast.shape_groups if per_rank[fast.rank]}
        self.max_update_error = 0.0
        self.max_combined_update_error = 0.0

    @torch.no_grad()
    def begin(self):
        """Check local reductions and independently pool FP64 before the production update."""
        self.checked = set()
        self.max_update_error = 0.0
        self.max_combined_update_error = 0.0
        for width, reducer in self.actual.reducers.items():
            local = reducer()
            reference = torch.stack([self.actual.bufs[key].double().square().sum(0) for key in self.actual.keys[width]])
            expected = torch.stack([self.actual.bufs[key].float().square().sum(0) for key in self.actual.keys[width]])
            torch.testing.assert_close(local, expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(local.double(), reference, rtol=1e-5, atol=1e-6)
            if self.actual.world > 1:
                dist.all_reduce(reference, group=self.reference.pg)
            self.fp64_moment[width].mul_(self.actual.decay).add_(
                reference, alpha=(1 - self.actual.decay) / self.actual.rows_total
            )
        self.reference.update()
        self.factors = self.reference.factors()

    @torch.no_grad()
    def __call__(
        self,
        mine,
        weights,
        gradients,
        momentum,
        actual_momentum,
        delta,
        right,
        trace,
        lr,
        weight_decay,
    ):
        if not self.checked:
            self.check_factors()
        if self.fast.arm.combine_outer_products:
            actual_factors = self.actual.factors()
            actual_outer = torch.stack([actual_factors[layer, kind][1] for layer, kind, _ in mine])
            reference_trace = {}
            reference_momentum, reference_delta = block_update_stack(
                weights,
                gradients,
                momentum,
                (None, actual_outer),
                right,
                args=self.fast.args,
                lr=lr,
                weight_decay=weight_decay,
                trace=reference_trace,
            )
            torch.testing.assert_close(actual_momentum, reference_momentum, rtol=0, atol=0)
            for key in ("ns_source", "ns_output"):
                torch.testing.assert_close(trace[key], reference_trace[key], rtol=0, atol=0, msg=key)
            torch.testing.assert_close(delta, reference_delta, rtol=1e-5, atol=1e-6)
            self.max_combined_update_error = max(
                self.max_combined_update_error,
                (delta - reference_delta).abs().max().item(),
            )
        outer = torch.stack([self.factors[layer, kind][1] for layer, kind, _ in mine])
        result = compare_outer_update(
            weights,
            gradients,
            momentum,
            actual_momentum,
            delta,
            right,
            trace,
            outer,
            args=self.fast.args,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.checked.add(tuple(weights.shape[-2:]))
        self.max_update_error = max(self.max_update_error, result["max_update_error"])

    @torch.no_grad()
    def check_factors(self):
        """Check pooled moments and roots before any parameter application."""
        factors = self.actual.factors()
        for width, moment in self.actual.moment.items():
            torch.testing.assert_close(moment, self.reference.moment[width], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(moment.double(), self.fp64_moment[width], rtol=1e-5, atol=1e-6)
            exact_roots = fp64_diagonal_factors(
                self.fp64_moment[width],
                self.actual.damping,
                self.actual.inner_power,
                self.actual.outer_power,
            )
            for index, key in enumerate(self.actual.keys[width]):
                actual_roots = factors[key]
                for actual, reference, exact in zip(actual_roots, self.factors[key], exact_roots, strict=True):
                    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(actual.double(), exact[index], rtol=1e-5, atol=1e-6)

    def finish(self):
        """Require parity for every owned stack before publishing the update receipt."""
        if self.checked != self.expected:
            raise ValueError("Actual outer update parity did not cover every owned shape")
        result = {
            "outer_diagonal_update_parity_passed": True,
            "outer_diagonal_max_update_error": self.max_update_error,
        }
        if self.fast.arm.combine_outer_products:
            result.update(
                combined_outer_update_parity_passed=True,
                combined_outer_max_update_error=self.max_combined_update_error,
            )
        return result
