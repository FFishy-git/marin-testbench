# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Outer-diagonal prototype using the pinned compiled MuonT product kernels."""

from toner.fast import muon as MU
from toner.fast import roots as PR
from toner.fast.engine import muont_in, muont_out


def upstream_outer_update_stack(weights, gradients, momenta, left, right, *, args, lr, weight_decay, trace=None):
    """Preserve upstream input/NS/outer products; apply optional diagonal output scaling."""
    left_inner, left_outer = left
    right_inner, right_outer = right
    if left_inner is not None or right_inner is None or right_outer is None:
        raise ValueError("Upstream-product prototype requires outer-only left and dense right factors")
    if right_inner.ndim != 3 or right_outer.ndim != 3 or (left_outer is not None and left_outer.ndim != 2):
        raise ValueError("Upstream-product prototype requires batched dense right and diagonal left factors")
    splits = PR.rows_lo(right_inner)
    updated, source = muont_in(list(gradients.unbind(0)), momenta, splits, args.muon_momentum, args.muon_nesterov)
    direction = MU.newton_schulz_batched(source, args.muon_ns_steps, args.muon_eps, args.muon_ns_mode)
    if trace is not None:
        trace["ns_source"] = source
        trace["ns_output"] = direction
    if left_outer is not None:
        left = left_outer * (left_outer.shape[-1] ** 0.5 / left_outer.norm(dim=-1, keepdim=True))
        direction = left[:, :, None] * direction
    norm2 = right_outer.square().sum((-2, -1))
    delta = muont_out(
        list(weights.unbind(0)),
        direction,
        PR.rows_lo(right_outer),
        norm2,
        lr,
        weight_decay,
        max(1.0, weights.shape[-2] / weights.shape[-1]) ** 0.5,
    )
    return updated, delta
