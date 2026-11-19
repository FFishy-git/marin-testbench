# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-tested root-start hypothesis; integration and CUDA acceptance remain separate."""

import torch

EPS = 1e-3
QUARTER = 0.25
U32 = 2.0**-24
LOWP = torch.float16


def quarter_divided_difference(d: torch.Tensor) -> torch.Tensor:
    """Evaluate the inverse-quarter divided difference without subtracting nearby powers.

    For positive x=a^4 and y=b^4, (x^(-1/4)-y^(-1/4))/(x-y)
    equals -1/(a*b*(a+b)*(a*a+b*b)), including the equal-value limit.
    """
    root = d.sqrt().sqrt()
    left, right = root[..., :, None], root[..., None, :]
    return -1.0 / (left * right * (left + right) * (left.square() + right.square()))


def rational_start(B: torch.Tensor, freeze: float) -> tuple:
    """Preserve the pinned start's outputs except for its off-diagonal divided difference.

    The cancellation-free coefficient is used on positive, unclamped diagonal
    pairs. Other pairs retain the original clamped-power/midpoint behavior.
    This callable does not install or compile itself into an existing trainer.
    """
    B = 0.5 * (B + B.mT)
    dB = B.diagonal(dim1=-2, dim2=-1)
    d = dB + EPS
    E = B - torch.diag_embed(dB)
    tau = freeze * B.size(-1) ** 0.5 * U32 * d.amax(-1, keepdim=True)
    small = d < tau
    frozen = small[..., :, None] & small[..., None, :]
    rel = E.abs() / (d[..., :, None] + d[..., None, :])
    rho = rel.masked_fill(frozen, 0.0).amax((-2, -1))
    lam = d.clamp_min(EPS * 1e-3).pow(-QUARTER)
    dd = d[..., :, None] - d[..., None, :]
    mid = 0.5 * (d[..., :, None] + d[..., None, :])
    original = torch.where(
        dd.abs() > 1e-6 * mid,
        (lam[..., :, None] - lam[..., None, :]) / torch.where(dd == 0, 1.0, dd),
        -QUARTER * mid.pow(-QUARTER - 1),
    )
    valid = d >= EPS * 1e-3
    pairs = valid[..., :, None] & valid[..., None, :]
    coefficient = torch.where(pairs, quarter_divided_difference(d.clamp_min(EPS * 1e-3)), original)
    O = coefficient * E
    return (
        d,
        E,
        E.to(LOWP),
        lam,
        O,
        O.to(LOWP),
        d.sqrt(),
        small,
        rho,
        O.square().sum((-2, -1)).sqrt(),
    )
