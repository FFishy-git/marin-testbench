# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Output moment roots and explicitly oriented matrix directions."""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OutputSettings:
    momentum: float = 0.938
    damping: float = 0.001
    inner_strength: float = 0.5
    outer_strength: float = 0.125
    refresh: int = 4


def inverse_factors(moment, settings):
    """Return unnormalized inner roots and unit-RMS outer roots of raw moments."""
    n = moment.shape[-1]
    sym = (moment + moment.mT) * 0.5
    trace = sym.diagonal(dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(n, dtype=moment.dtype, device=moment.device)
    normalized = sym * (n / trace.clamp_min(torch.finfo(moment.dtype).tiny))[..., None, None]
    normalized = torch.where((trace > 0)[..., None, None], normalized, eye)
    eigenvalues, vectors = torch.linalg.eigh(normalized)
    eigenvalues = eigenvalues.clamp_min(0) + settings.damping
    inner = (vectors * eigenvalues.pow(-settings.inner_strength).unsqueeze(-2)) @ vectors.mT
    outer = (vectors * eigenvalues.pow(-settings.outer_strength).unsqueeze(-2)) @ vectors.mT
    outer = outer * (n**0.5 / outer.norm(dim=(-2, -1), keepdim=True))
    inner = torch.where((trace > 0)[..., None, None], inner, eye)
    outer = torch.where((trace > 0)[..., None, None], outer, eye)
    return inner, outer


def direction_and_decay(
    weight: torch.Tensor,
    momentum: torch.Tensor,
    *,
    arm: str,
    polar: Callable,
    right_inner: torch.Tensor | None = None,
    right_outer: torch.Tensor | None = None,
    left_inner: torch.Tensor | None = None,
    left_outer: torch.Tensor | None = None,
):
    """Compute D and Q for y=Wx; decay remains separate from the direction."""
    if arm == "muon":
        return polar(momentum), weight
    if arm == "down_hybrid":
        return left_outer @ polar(left_inner @ momentum), left_outer @ left_outer @ weight
    source = momentum @ right_inner
    if arm == "inner_outer_down":
        source = left_inner @ source
    direction = polar(source) @ right_outer
    if arm in ("outer_left_down", "inner_outer_down"):
        direction = left_outer @ direction
    return direction, weight @ right_outer @ right_outer
