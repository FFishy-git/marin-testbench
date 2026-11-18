# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execution-only observations of the actual pinned-product optimizer stacks."""

import torch
from toner.fast import muon as MU
from toner.fast import roots as PR
from toner.fast.engine import muont_in, muont_out

from .block_toner import BLOCK_ARMS, BlockToner, fast_arguments
from .configure import configuration
from .outer_diagonal_checks import OuterDiagonalVerifier
from .upstream_outer_products import upstream_outer_update_stack


def pinned_null_update(weights, gradients, momentum, right, *, args, lr, decay):
    """Run the pinned input, NS and outer kernels without invoking the candidate update."""
    updated, source = muont_in(
        list(gradients.unbind(0)),
        momentum,
        PR.rows_lo(right[0]),
        args.muon_momentum,
        args.muon_nesterov,
    )
    direction = MU.newton_schulz_batched(source, args.muon_ns_steps, args.muon_eps, args.muon_ns_mode)
    delta = muont_out(
        list(weights.unbind(0)),
        direction,
        PR.rows_lo(right[1]),
        right[1].square().sum((-2, -1)),
        lr,
        decay,
        max(1.0, weights.shape[-2] / weights.shape[-1]) ** 0.5,
    )
    return updated, source, direction, delta


def fp64_output_delta(weights, direction, outer, right, *, lr, decay):
    """Evaluate the output geometry and input-shaped decay using direct FP64 products."""
    rows, columns = weights.shape[-2:]
    weight, value, factor = weights.double(), direction.double(), right.double()
    normalized = factor / (factor.square().sum((-2, -1), keepdim=True) / columns).sqrt()
    if outer is not None:
        left = outer.double()
        value = value * (left / left.square().mean(-1, keepdim=True).sqrt())[:, :, None]
    scale = max(1.0, rows / columns) ** 0.5
    return -lr.double() * (
        scale * torch.bmm(value, normalized) + decay.double() * torch.bmm(torch.bmm(weight, normalized), normalized)
    )


@torch.no_grad()
def test_upstream_product_updates(device):
    """Exercise pinned null/input and FP64 outer equations at actual eight-rank owned shapes."""
    config = configuration("outer_leftdiag_upstream_fused", "0p5m", 0, 4)
    args = fast_arguments(config)
    owned = config["model"]["layers"] // config["training"]["world_size"]
    hidden, mlp = config["model"]["hidden"], config["model"]["mlp"]
    shapes = ((4 * owned, hidden, hidden), (2 * owned, mlp, hidden), (owned, hidden, mlp))
    for shape in shapes:
        count, rows, columns = shape
        for pattern in ("gaussian", "zero", "anisotropic"):
            generator = torch.Generator(device=device).manual_seed(83)
            weights, gradients, momentum = (
                torch.randn(shape, generator=generator, device=device) * 0.01 for _ in range(3)
            )
            if pattern == "zero":
                gradients.zero_()
                momentum.zero_()
            elif pattern == "anisotropic":
                gradients *= torch.logspace(-2, 2, columns, device=device)
            right = tuple(torch.eye(columns, device=device).repeat(count, 1, 1) for _ in range(2))
            right[0].diagonal(dim1=-2, dim2=-1).copy_(
                torch.linspace(0.8, 1.2, columns, device=device).expand(count, columns)
            )
            right[1].diagonal(dim1=-2, dim2=-1).copy_(
                torch.linspace(0.9, 1.1, columns, device=device).expand(count, columns)
            )
            outer = torch.linspace(0.5, 1.5, rows, device=device).expand(count, rows)
            for nesterov in (False, True):
                args.muon_nesterov = nesterov
                for lr_value in (0.0, config["optimizer"]["lr"], config["optimizer"]["lr"] / 32):
                    lr = torch.tensor(lr_value, device=device)
                    decay = torch.tensor(config["optimizer"]["weight_decay"], device=device)
                    expected = pinned_null_update(weights, gradients, momentum, right, args=args, lr=lr, decay=decay)
                    for left in (None, outer):
                        trace = {}
                        updated, delta = upstream_outer_update_stack(
                            weights,
                            gradients,
                            momentum,
                            (None, left),
                            right,
                            args=args,
                            lr=lr,
                            weight_decay=decay,
                            trace=trace,
                        )
                        torch.testing.assert_close(updated, expected[0], rtol=0, atol=0)
                        torch.testing.assert_close(trace["ns_source"], expected[1], rtol=0, atol=0)
                        torch.testing.assert_close(trace["ns_output"], expected[2], rtol=0, atol=0)
                        if left is None:
                            torch.testing.assert_close(delta, expected[3], rtol=0, atol=0)
                        equation = fp64_output_delta(weights, expected[2], left, right[1], lr=lr, decay=decay)
                        torch.testing.assert_close(delta.double(), equation, rtol=1e-5, atol=1e-6)


class ScalarStepProbe:
    """Run the actual production dispatch with fixed factors and changing device scalars."""

    step = BlockToner.step

    def __init__(self, device, *, apply_left):
        config = configuration("outer_leftdiag_upstream_fused", "0p5m", 0, 4)
        self.args = fast_arguments(config)
        arm = "outer_leftdiag_upstream_fused" if apply_left else "null_leftdiag_upstream_fused"
        self.arm = BLOCK_ARMS[arm]
        self.device, self.world, self.rank = device, 1, 0
        self.update_scalars = torch.empty(2, device=device)
        self.param_groups = [{"lr": config["optimizer"]["lr"], "weight_decay": config["optimizer"]["weight_decay"]}]
        self.input, self.output = object(), object()
        self.input_diagonal, self.output_diagonal = None, None
        self.engines, self.shape_groups = [], []
        self.steps = 0
        self.momentum, self.right, self.left = {}, {}, {}
        self.update_verifier = None
        owned = config["model"]["layers"] // config["training"]["world_size"]
        hidden, mlp = config["model"]["hidden"], config["model"]["mlp"]
        for count, rows, columns, kind, input_kind in (
            (4 * owned, hidden, hidden, "q", "attn"),
            (2 * owned, mlp, hidden, "gate", "mlp"),
            (owned, hidden, mlp, "down", "down"),
        ):
            generator = torch.Generator(device=device).manual_seed(rows + columns)
            weights = torch.randn(count, rows, columns, generator=generator, device=device) * 0.01
            gradients = torch.randn(count, rows, columns, generator=generator, device=device) * 0.01
            mine = []
            for index in range(count):
                parameter = torch.nn.Parameter(weights[index])
                parameter.grad = gradients[index]
                mine.append((index, kind, parameter))
                inner = torch.diag(torch.linspace(0.8, 1.2, columns, device=device))
                outer = torch.diag(torch.linspace(0.9, 1.1, columns, device=device))
                self.right[index, input_kind] = (inner, outer)
                self.left[index, kind] = (None, torch.linspace(0.5, 1.5, rows, device=device))
            self.shape_groups.append(((rows, columns), [mine]))
            self.momentum[rows, columns] = torch.zeros_like(weights)

    def _factors(self, full, diagonal):
        return self.right if full is self.input else self.left

    def _engine_metrics(self):
        return {}


@torch.no_grad()
def test_upstream_scalar_step(device):
    """Check changed and zero LR through BlockToner.step, including applied parameters."""
    for apply_left in (False, True):
        fast = ScalarStepProbe(device, apply_left=apply_left)
        base_lr = fast.param_groups[0]["lr"]
        for lr in (base_lr, base_lr / 32, 0.0, base_lr):
            group = fast.param_groups[0]
            group["lr"] = lr
            expected = {}
            for shape, per_rank in fast.shape_groups:
                mine = per_rank[0]
                input_kind = "attn" if shape[0] == shape[1] else ("mlp" if shape[0] > shape[1] else "down")
                weights = torch.stack([parameter.detach() for _, _, parameter in mine])
                gradients = torch.stack([parameter.grad for _, _, parameter in mine])
                right = tuple(
                    torch.stack([fast.right[index, input_kind][side] for index, _, _ in mine]) for side in (0, 1)
                )
                scalars = torch.tensor([lr, group["weight_decay"]], device=device)
                updated, _, direction, null_delta = pinned_null_update(
                    weights,
                    gradients,
                    fast.momentum[shape],
                    right,
                    args=fast.args,
                    lr=scalars[0],
                    decay=scalars[1],
                )
                outer = torch.stack([fast.left[index, kind][1] for index, kind, _ in mine]) if apply_left else None
                delta = fp64_output_delta(weights, direction, outer, right[1], lr=scalars[0], decay=scalars[1])
                expected[shape] = (weights, updated, delta, null_delta)
            fast.step()
            torch.testing.assert_close(
                fast.update_scalars, torch.tensor([lr, group["weight_decay"]], device=device), rtol=0, atol=0
            )
            for shape, per_rank in fast.shape_groups:
                previous, updated, delta, null_delta = expected[shape]
                actual = torch.stack([parameter.detach() for _, _, parameter in per_rank[0]])
                torch.testing.assert_close(fast.momentum[shape], updated, rtol=0, atol=0)
                torch.testing.assert_close(actual.double(), previous.double() + delta, rtol=1e-5, atol=1e-6)
                if not apply_left:
                    torch.testing.assert_close(actual, previous + null_delta, rtol=0, atol=0)
                if lr == 0:
                    torch.testing.assert_close(actual, previous, rtol=0, atol=0)


class UpstreamProductVerifier(OuterDiagonalVerifier):
    """Verify actual captured rows, scheduled scalars and every owned parameter stack."""

    def __init__(self, fast):
        if not fast.arm.upstream_product_kernels:
            raise ValueError("Pinned-product observations require a pinned-product arm")
        super().__init__(fast)
        self.max_equation_error = 0.0

    @torch.no_grad()
    def begin(self):
        super().begin()
        self.max_equation_error = 0.0

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
        scheduled = torch.tensor([lr, weight_decay], dtype=self.fast.update_scalars.dtype, device=weights.device)
        torch.testing.assert_close(self.fast.update_scalars, scheduled, rtol=0, atol=0)
        scalar_lr, scalar_decay = self.fast.update_scalars.unbind(0)
        expected_momentum, source, direction, null_delta = pinned_null_update(
            weights,
            gradients,
            momentum,
            right,
            args=self.fast.args,
            lr=scalar_lr,
            decay=scalar_decay,
        )
        torch.testing.assert_close(actual_momentum, expected_momentum, rtol=0, atol=0)
        torch.testing.assert_close(trace["ns_source"], source, rtol=0, atol=0)
        torch.testing.assert_close(trace["ns_output"], direction, rtol=0, atol=0)
        if self.fast.arm.apply_left:
            factors = self.actual.factors()
            outer = torch.stack([factors[layer, kind][1] for layer, kind, _ in mine])
        else:
            outer = None
            torch.testing.assert_close(delta, null_delta, rtol=0, atol=0)
        equation = fp64_output_delta(weights, direction, outer, right[1], lr=scalar_lr, decay=scalar_decay)
        torch.testing.assert_close(delta.double(), equation, rtol=1e-5, atol=1e-6)
        self.max_equation_error = max(self.max_equation_error, (delta.double() - equation).abs().max().item())
        self.checked.add(tuple(weights.shape[-2:]))

    def finish(self):
        if self.checked != self.expected:
            raise ValueError("Pinned-product observations did not cover every owned shape")
        return {
            "upstream_product_parity_passed": True,
            "upstream_scheduled_scalars_passed": True,
            "upstream_pinned_input_exact": True,
            "upstream_output_equation_passed": True,
            "upstream_output_equation_max_error": self.max_equation_error,
            "upstream_null_exact": not self.fast.arm.apply_left,
            "upstream_owned_stacks_checked": len(self.checked),
        }
