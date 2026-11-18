# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict actual-stack checks, keeping the candidate's finite-NS and output-work path."""

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .block_toner import block_update_stack, stack_unit_rms
from .outer_diagonal_checks import OuterDiagonalVerifier

RTOL, ATOL = 1e-5, 1e-6


class AppliedOuterTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.enabled = True
        self.packed, self.normalized, self.products = [], [], []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if not self.enabled:
            return result
        if function == torch.ops.aten.stack.default and result.ndim == 2:
            self.packed.append(result.detach().clone())
        elif function == torch.ops.aten.linalg_vector_norm.default and args[0].ndim == 2:
            self.normalized.append(args[0].detach().clone())
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
                self.products.append((first.detach().clone(), second.detach().clone(), result.detach().clone()))
        return result

    def require_applied(self, applied: torch.Tensor, direction: torch.Tensor) -> None:
        actual = [value for value in self.normalized if tuple(value.shape) == tuple(applied.shape)]
        if len(actual) != 1 or not torch.equal(actual[0], applied):
            raise ValueError("Attribution actual applied outer vector differs from its required factor")
        if not any(torch.equal(value, actual[0]) for value in self.packed):
            raise ValueError("Attribution step bypassed applied outer packing")
        normalized = stack_unit_rms(applied)
        matches = [
            (vector, product)
            for source, vector, product in self.products
            if tuple(product.shape) == tuple(direction.shape) and tuple(vector.shape) == (*applied.shape, 1)
        ]
        if len(matches) != 1:
            raise ValueError("Attribution step did not apply exactly one normalized outer row vector")
        torch.testing.assert_close(matches[0][0].squeeze(-1), normalized, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(matches[0][1], direction, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(
            normalized.square().sum(-1), torch.full_like(normalized[:, 0], applied.shape[-1]), rtol=RTOL, atol=ATOL
        )


class FusedAttributionVerifier(OuterDiagonalVerifier):
    """Reuse real capture/root checks and require strict pre-application update/work parity."""

    def __init__(self, fast):
        super().__init__(fast)
        self.null = fast.arm.identity_left_outer
        self.observation = None
        self.work_checked = set()

    def begin(self):
        super().begin()
        self.observation = AppliedOuterTrace()
        self.work_checked = set()

    def observe(self):
        if self.observation is None:
            raise RuntimeError("Attribution verifier observation began before captured rows")
        return self.observation

    @torch.no_grad()
    def __call__(self, mine, weights, gradients, momentum, actual_momentum, delta, right, trace, lr, weight_decay):
        if self.observation is None:
            raise RuntimeError("Attribution step has no work observation")
        self.observation.enabled = False
        try:
            if not self.checked:
                self.check_factors()
            outer = torch.stack([self.factors[layer, kind][1] for layer, kind, _ in mine])
            actual_factors = self.actual.factors()
            applied = (
                torch.ones_like(outer)
                if self.null
                else torch.stack([actual_factors[layer, kind][1] for layer, kind, _ in mine])
            )
            reference_trace = {}
            expected_momentum, expected_delta = block_update_stack(
                weights,
                gradients,
                momentum,
                (None, None if self.null else outer),
                right,
                args=self.fast.args,
                lr=lr,
                weight_decay=weight_decay,
                trace=reference_trace,
            )
            torch.testing.assert_close(actual_momentum, expected_momentum, rtol=0, atol=0)
            for key in ("ns_source", "ns_output"):
                torch.testing.assert_close(trace[key], reference_trace[key], rtol=0, atol=0)
            torch.testing.assert_close(delta, expected_delta, rtol=RTOL, atol=ATOL)
            direction = trace["ns_output"]
            if right[1] is not None:
                right_outer = stack_unit_rms(right[1])
                direction = (
                    torch.bmm(direction, right_outer) if right_outer.ndim == 3 else (direction * right_outer[:, None, :])
                )
            expected_direction = direction * stack_unit_rms(applied)[:, :, None]
            self.observation.require_applied(applied, expected_direction)
            shape = tuple(weights.shape[-2:])
            self.checked.add(shape)
            self.work_checked.add(shape)
            self.max_update_error = max(self.max_update_error, (delta - expected_delta).abs().max().item())
            self.observation.packed.clear()
            self.observation.normalized.clear()
            self.observation.products.clear()
        finally:
            self.observation.enabled = True

    def finish(self):
        result = super().finish()
        if self.work_checked != self.expected:
            raise ValueError("Attribution retained-work gate missed an owned matrix shape")
        result["output_work_retained_passed"] = True
        if self.null:
            result["null_identity_outer_passed"] = True
        return result
