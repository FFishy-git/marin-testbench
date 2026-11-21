# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mousse formula control and explicitly adapted author-numerics profile.

Source: arXiv:2603.09697v2, Algorithm 1 and Section 5; Mousse revision
d00c1bf17790fbe56424ee5567cce80d8e75f4b2, dion/dion/mousse.py.
Both profiles use gradient Grams after the runner's DDP gradient pooling, not
activation/cotangent moments. The matched inverse powers and eigen-coordinate
rotate/scale/NS/scale/rotate order follow the pinned implementation.

The controlled profile retains common momentum, NS, shape scaling, and
auxiliary routing. The paper profile uses the author's non-Triton five-stage
BF16 NS, source/output quantization, and sqrt(m/n) scaling. Architecture, data,
config-driven momentum, auxiliary optimizer, owner routing, and FP32 additive
parameter deltas remain runner adaptations, not a full paper reproduction.
Zero-trace moments use identity factors and zero updates graft to zero rather
than the author's undefined divisions. EMA buffers are never trace-normalized
in place, including when bias correction is disabled.
"""

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.distributed as dist
from toner.fast import muon as MU

from .related_base import MatrixBaseline

Eigensystem = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class MousseSettings:
    """Mousse geometry and numerical profile, separate from common optimizer settings."""

    beta: float = 0.95
    damping: float = 1e-5
    refresh: int = 10
    alpha: float = 0.125
    bias_correction: bool = True
    norm_graft: bool = True
    profile: Literal["controlled", "paper"] = "controlled"
    decay: Literal["isotropic", "input_shaped", "none"] = "isotropic"
    paper_ns_epsilon: float = 1e-8

    def __post_init__(self):
        if not 0 <= self.beta < 1 or not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("Mousse requires beta in [0, 1) and a finite nonnegative alpha")
        if not math.isfinite(self.damping) or self.damping <= 0 or self.refresh < 1:
            raise ValueError("Mousse requires positive damping and refresh")
        if self.profile not in ("controlled", "paper") or self.decay not in ("isotropic", "input_shaped", "none"):
            raise ValueError("Unknown Mousse numerical profile or decay")
        if not math.isfinite(self.paper_ns_epsilon) or self.paper_ns_epsilon <= 0:
            raise ValueError("Mousse requires a positive paper NS epsilon")


@dataclass
class GramState:
    left: torch.Tensor
    right: torch.Tensor
    eigen_left: Eigensystem
    eigen_right: Eigensystem
    refresh_step: int = 0


def eigenpairs(moment: torch.Tensor, settings: MousseSettings, step: int) -> Eigensystem:
    """Trace-normalize a Gram EMA and apply the pinned author's two damping stages."""
    corrected = moment / (1 - settings.beta**step) if settings.bias_correction else moment
    symmetric = (corrected + corrected.T) * 0.5
    trace = symmetric.trace()
    dimension = moment.shape[0]
    eye = torch.eye(dimension, device=moment.device, dtype=moment.dtype)
    if trace.item() == 0:
        return torch.ones(dimension, device=moment.device, dtype=moment.dtype), eye
    normalized = symmetric * (dimension / trace)
    values, vectors = torch.linalg.eigh(normalized + settings.damping * eye)
    # clean_eigenvalues shifts the entire spectrum, not a pointwise clamp.
    shift = (-values.min()).clamp_min(0) + settings.damping
    values = values + shift
    return values, vectors


# The five-stage helper below is adapted from the pinned source under MIT:
# Copyright (c) 2026 ShikiNatsume; Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
PAPER_NS_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def paper_newton_schulz(source: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Run the pinned author's eager BF16 helper without optional Triton dependencies."""
    matrix = source.to(torch.bfloat16)
    transpose = source.shape[-2] > source.shape[-1]
    if transpose:
        matrix = matrix.mT
    matrix = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + epsilon)
    for a, b, c in PAPER_NS_COEFFICIENTS:
        gram = matrix @ matrix.mT
        polynomial = b * gram + c * (gram @ gram)
        matrix = a * matrix + polynomial @ matrix
    return matrix.mT if transpose else matrix


def mousse_direction(
    source: torch.Tensor,
    left: Eigensystem,
    right: Eigensystem,
    settings: MousseSettings,
    *,
    ns_steps: int,
    ns_epsilon: float,
    ns_mode: str,
) -> torch.Tensor:
    """Return the matched-power direction, grafted to the actual finite-NS norm."""
    left_values, left_vectors = left
    right_values, right_vectors = right
    left_scale = left_values.abs().pow(settings.alpha)
    right_scale = right_values.abs().pow(settings.alpha)
    matrix = source.to(torch.bfloat16).float() if settings.profile == "paper" else source.float()
    matrix = left_vectors.T @ matrix
    matrix = matrix @ right_vectors
    matrix = matrix / left_scale[:, None]
    matrix = matrix / right_scale[None, :]
    if settings.profile == "paper":
        orthogonal = paper_newton_schulz(matrix, settings.paper_ns_epsilon)
    else:
        orthogonal = MU.newton_schulz(matrix, ns_steps, ns_epsilon, ns_mode)
    target_norm = orthogonal.norm()
    matrix = orthogonal / left_scale[:, None]
    matrix = matrix / right_scale[None, :]
    matrix = left_vectors @ matrix
    matrix = matrix @ right_vectors.T
    if settings.norm_graft:
        norm = matrix.norm()
        matrix = matrix * (target_norm / norm.clamp_min(torch.finfo(matrix.dtype).tiny))
    return matrix.to(torch.bfloat16).float() if settings.profile == "paper" else matrix.float()


def decay_direction(weight: torch.Tensor, right: Eigensystem, settings: MousseSettings) -> torch.Tensor:
    """Return isotropic decay or the explicitly requested normalized input-shaped adaptation."""
    if settings.decay == "isotropic":
        return weight.float()
    if settings.decay == "none":
        return torch.zeros_like(weight, dtype=torch.float32)
    values, vectors = right
    scales = values.abs().pow(-settings.alpha)
    scales = scales * (scales.numel() ** 0.5 / scales.norm())
    factor_squared = (vectors * scales.square()[None, :]) @ vectors.T
    return weight.float() @ factor_squared


class Mousse(MatrixBaseline):
    """Mousse on the shared baseline's matrix owners, without any capture hooks."""

    def __init__(
        self,
        model,
        config,
        *,
        positions,
        rows_per_microbatch,
        capacity,
        rows_total,
        world,
        rank,
        device,
    ):
        super().__init__(model, config, world=world, rank=rank, device=device)
        self.settings = MousseSettings(**config["experiment"]["related_baseline"]["settings"])
        self.grams = {}
        for key, parameter in self.owned.items():
            rows, columns = parameter.shape
            self.grams[key] = GramState(
                left=torch.zeros(rows, rows, device=device, dtype=torch.float32),
                right=torch.zeros(columns, columns, device=device, dtype=torch.float32),
                eigen_left=(torch.ones(rows, device=device), torch.eye(rows, device=device)),
                eigen_right=(torch.ones(columns, device=device), torch.eye(columns, device=device)),
            )

    @torch.no_grad()
    def step(self):
        settings = self.settings
        optimizer = self.config["optimizer"]
        group = self.param_groups[0]
        step = self.steps + 1
        refresh = step % settings.refresh == 1 or settings.refresh == 1
        finite = torch.tensor(
            all(torch.isfinite(parameter.grad).all().item() for parameter in self.owned.values()),
            device=self.device,
            dtype=torch.int32,
        )
        if self.world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("Nonfinite Mousse gradient")
        deltas = {}
        for key, parameter in self.owned.items():
            gradient = parameter.grad.float()
            state = self.grams[key]
            state.left.mul_(settings.beta).add_(gradient @ gradient.T, alpha=1 - settings.beta)
            state.right.mul_(settings.beta).add_(gradient.T @ gradient, alpha=1 - settings.beta)
            if refresh:
                state.eigen_left = eigenpairs(state.left, settings, step)
                state.eigen_right = eigenpairs(state.right, settings, step)
                state.refresh_step = step
            source = self.momentum_source(key, gradient)
            direction = mousse_direction(
                source,
                state.eigen_left,
                state.eigen_right,
                settings,
                ns_steps=optimizer["ns_steps"],
                ns_epsilon=optimizer["ns_eps"],
                ns_mode=optimizer["ns_mode"],
            )
            ratio = parameter.shape[0] / parameter.shape[1]
            shape_scale = ratio**0.5 if settings.profile == "paper" else max(1.0, ratio) ** 0.5
            decay = decay_direction(parameter, state.eigen_right, settings)
            deltas[key] = -group["lr"] * (shape_scale * direction + group["weight_decay"] * decay)
        self.apply_updates(deltas)
        self.last_metrics = {
            "related_baseline_steps": self.steps,
            "mousse_eigen_refreshed": refresh,
            "mousse_eigh_matrices": 2 * len(self.owned) if refresh else 0,
            "mousse_factor_age": max((step - state.refresh_step for state in self.grams.values()), default=0),
        }

    def runtime_state(self):
        return {
            "method": "mousse",
            "settings": asdict(self.settings),
            "base": self.base_runtime_state(),
            "grams": {
                key: {
                    "left": state.left,
                    "right": state.right,
                    "eigen_left": state.eigen_left,
                    "eigen_right": state.eigen_right,
                    "refresh_step": state.refresh_step,
                }
                for key, state in self.grams.items()
            },
        }

    def restore_runtime_state(self, state):
        if state["method"] != "mousse" or state["settings"] != asdict(self.settings):
            raise ValueError("Resume requires unchanged Mousse settings")
        if state["grams"].keys() != self.grams.keys():
            raise ValueError("Resume requires unchanged Mousse matrix ownership")
        self.restore_base_runtime_state(state["base"])
        for key, values in state["grams"].items():
            target = self.grams[key]
            target.left.copy_(values["left"])
            target.right.copy_(values["right"])
            target.eigen_left = tuple(value.to(self.device).clone() for value in values["eigen_left"])
            target.eigen_right = tuple(value.to(self.device).clone() for value in values["eigen_right"])
            target.refresh_step = values["refresh_step"]
