# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GO-MUON v1 formula controls, not an author-code reproduction.

Source: arXiv:2608.09763v1, equations (12)--(18) and (21)--(35), archived at
artifacts/papers/go-muon-2608.09763.pdf. Both profiles reuse the shared BF16 row
capture, including on paper-profile reuse steps. Those steps skip moment pooling,
OAS, and roots, but do not reproduce the paper's hook-gated collection cost.
"""

from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.distributed as dist
from toner.fast import muon as MU

from .block_toner import INPUT_KIND, INPUT_KINDS, ROW_SCALE, attach_output_rows
from .related_base import MATRIX_KINDS, MatrixBaseline

SOURCE_REVISION = "arxiv:2608.09763v1"
SOURCE_SHA256 = "dcc88c9592c99a76b1a31030ff6a1b294b30459d84dcefe66b1217f005593093"
MatrixKey = tuple[int, str]
StatisticMode = Literal["ema_per_step", "paper_sparse"]
PolarMethod = Literal["newton_schulz", "exact_svd"]


@dataclass(frozen=True)
class GoMuonSettings:
    """Select statistical and polar fidelity independently for the control.

    Shared statistics momentum, damping, and root refresh configure ema_per_step.
    paper_sparse uses root refresh as K, OAS before its backward EMA, and requires
    exact_svd. Its relative floor is FP32 roundoff; the absolute floor covers
    all-zero factors. The paper does not specify these floor constants.
    """

    statistic_mode: StatisticMode = "ema_per_step"
    polar_method: PolarMethod = "newton_schulz"
    backward_decay: float = 0.97
    eigenvalue_floor_relative: float = 1.1920928955078125e-7
    eigenvalue_floor_absolute: float = 1e-30

    def __post_init__(self):
        if self.statistic_mode not in ("ema_per_step", "paper_sparse"):
            raise ValueError(f"Unknown GO-MUON statistic mode {self.statistic_mode}")
        if self.polar_method not in ("newton_schulz", "exact_svd"):
            raise ValueError(f"Unknown GO-MUON polar method {self.polar_method}")
        if self.statistic_mode == "paper_sparse" and self.polar_method != "exact_svd":
            raise ValueError("paper_sparse requires the paper's compact-SVD support polar")
        if not 0 <= self.backward_decay < 1:
            raise ValueError("Backward decay must lie in [0, 1)")
        if self.eigenvalue_floor_relative < 0 or self.eigenvalue_floor_absolute <= 0:
            raise ValueError("Eigenvalue floors must be nonnegative relative and positive absolute")


def oas_factor(moment: torch.Tensor, sample_count: int) -> torch.Tensor:
    """Apply the N-aware, uncentered OAS formula in equations (17)--(18)."""
    if moment.ndim != 2 or moment.shape[0] != moment.shape[1] or sample_count <= 0:
        raise ValueError("OAS requires a square moment and a positive sample count")
    symmetric = (moment + moment.mT) * 0.5
    width = symmetric.shape[0]
    mean = symmetric.trace() / width
    if not torch.isfinite(symmetric).all().item():
        raise FloatingPointError("Nonfinite GO-MUON second moment")
    if mean.item() <= 0:
        return torch.zeros_like(symmetric)
    # OAS's coefficient is scale invariant. Normalize before squared traces to
    # avoid underflow for mean-loss cotangents and overflow for scaled captures.
    normalized = symmetric / mean
    trace_squared = normalized.square().sum()
    dispersion = trace_squared - width
    denominator = (sample_count + 1 - 2 / width) * dispersion
    numerator = (1 - 2 / width) * trace_squared + width**2
    shrinkage = (numerator / denominator).clamp(0, 1) if denominator.item() > 0 else moment.new_tensor(1.0)
    identity = torch.eye(width, dtype=moment.dtype, device=moment.device)
    return (1 - shrinkage) * symmetric + shrinkage * mean * identity


def inverse_quarter_factor(
    moment: torch.Tensor,
    *,
    relative_floor: float,
    absolute_floor: float,
    normalize_trace: bool = False,
    damping: float = 0.0,
) -> torch.Tensor:
    """Return an inverse fourth root, solving in FP64 and retaining the input dtype."""
    # A fixed ridge can be small compared with a dominant eigenvalue. FP32
    # eigensolver error then materially changes inverse roots near the null space.
    symmetric = (moment.double() + moment.mT.double()) * 0.5
    width = symmetric.shape[0]
    if not torch.isfinite(symmetric).all().item():
        raise FloatingPointError("Nonfinite GO-MUON effective factor")
    identity = torch.eye(width, dtype=symmetric.dtype, device=moment.device)
    if normalize_trace:
        trace = symmetric.trace()
        if trace.item() <= 0:
            return identity.to(moment.dtype)
        symmetric = symmetric * (width / trace) + damping * identity
    eigenvalues, basis = torch.linalg.eigh(symmetric)
    floor = (relative_floor * eigenvalues.abs().amax()).clamp_min(absolute_floor)
    powers = eigenvalues.clamp_min(floor).pow(-0.25)
    root = (basis * powers) @ basis.mT
    return ((root + root.mT) * 0.5).to(moment.dtype)


def compact_support_polar(source: torch.Tensor) -> torch.Tensor:
    """Compact full-SVD polar, including a semi-orthogonal null completion.

    This is a support maximizer at rank deficiency, not the canonical partial
    polar. The paper explicitly special-cases zero instead of completing it.
    """
    if not torch.isfinite(source).all().item():
        raise FloatingPointError("Nonfinite GO-MUON polar source")
    if not torch.any(source != 0).item():
        return torch.zeros_like(source)
    left, _, right = torch.linalg.svd(source, full_matrices=False)
    return left @ right


def matched_direction(
    source: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    polar_method: PolarMethod,
    ns_steps: int = 10,
    ns_eps: float = 1e-5,
    ns_mode: str = "f32",
) -> torch.Tensor:
    """Matched quarter-factor sandwich with final Frobenius graft, without LR."""
    if not torch.isfinite(source).all().item():
        raise FloatingPointError("Nonfinite GO-MUON momentum source")
    if not torch.any(source != 0).item():
        return torch.zeros_like(source)
    transformed = left @ source @ right
    if polar_method == "exact_svd":
        polar = compact_support_polar(transformed)
    elif polar_method == "newton_schulz":
        polar = MU.newton_schulz(transformed, ns_steps, ns_eps, ns_mode).to(source.dtype)
    else:
        raise ValueError(f"Unknown GO-MUON polar method {polar_method}")
    direction = left @ polar @ right
    # FP32 reductions drift enough on rectangular matrices to mis-graft the norm.
    norm = torch.linalg.matrix_norm(direction, dtype=torch.float64).to(direction.dtype)
    if not torch.isfinite(norm).item() or norm.item() <= 0:
        raise FloatingPointError("Nonzero GO-MUON source produced an invalid graft norm")
    return direction * (min(source.shape) ** 0.5 / norm)


class GoMuonStatistics:
    """Stateful statistical reference, independent of model capture and routing."""

    def __init__(self, settings: GoMuonSettings, *, decay: float, damping: float, refresh: int):
        if not 0 <= decay < 1 or damping < 0 or refresh <= 0:
            raise ValueError("Invalid GO-MUON EMA, damping, or refresh")
        self.settings, self.decay, self.damping, self.refresh = settings, decay, damping, refresh
        self.input_moments: dict[MatrixKey, torch.Tensor] = {}
        self.output_moments: dict[MatrixKey, torch.Tensor] = {}
        self.input_factors: dict[MatrixKey, torch.Tensor] = {}
        self.output_factors: dict[MatrixKey, torch.Tensor] = {}
        self.last_observation_step = self.last_root_step = -1
        self.observations = self.root_refreshes = self.sample_count = 0

    def observes(self, step: int) -> bool:
        return self.settings.statistic_mode == "ema_per_step" or step % self.refresh == 0

    @torch.no_grad()
    def update(
        self,
        step: int,
        inputs: dict[MatrixKey, torch.Tensor],
        outputs: dict[MatrixKey, torch.Tensor],
        sample_count: int,
    ) -> bool:
        """Consume pooled second-moment means; return whether roots refreshed."""
        if not self.observes(step):
            return False
        if step <= self.last_observation_step:
            raise ValueError("GO-MUON observations must advance the step counter")
        if self.observations and (
            inputs.keys() != self.input_moments.keys() or outputs.keys() != self.output_moments.keys()
        ):
            raise ValueError("GO-MUON statistic ownership changed")
        if self.settings.statistic_mode == "paper_sparse":
            self.input_moments = {key: oas_factor(value, sample_count) for key, value in inputs.items()}
            beta = self.settings.backward_decay**self.refresh
            for key, value in outputs.items():
                shrunk = oas_factor(value, sample_count)
                self.output_moments[key] = (
                    beta * self.output_moments[key] + (1 - beta) * shrunk if key in self.output_moments else shrunk
                )
        else:
            for moments, samples in ((self.input_moments, inputs), (self.output_moments, outputs)):
                for key, value in samples.items():
                    old = moments[key] if key in moments else torch.zeros_like(value)
                    moments[key] = self.decay * old + (1 - self.decay) * value
        self.observations += 1
        self.sample_count = sample_count
        self.last_observation_step = step
        if step % self.refresh != 0:
            return False
        common = self.settings.statistic_mode == "ema_per_step"
        for factors, moments in (
            (self.input_factors, self.input_moments),
            (self.output_factors, self.output_moments),
        ):
            for key, moment in moments.items():
                factors[key] = inverse_quarter_factor(
                    moment,
                    relative_floor=self.settings.eigenvalue_floor_relative,
                    absolute_floor=self.settings.eigenvalue_floor_absolute,
                    normalize_trace=common,
                    damping=self.damping if common else 0.0,
                )
        self.last_root_step = step
        self.root_refreshes += 1
        return True

    def state_dict(self):
        """Snapshot moments and cached factors without aliasing subsequent updates."""
        return {
            "input_moments": {key: value.clone() for key, value in self.input_moments.items()},
            "output_moments": {key: value.clone() for key, value in self.output_moments.items()},
            "input_factors": {key: value.clone() for key, value in self.input_factors.items()},
            "output_factors": {key: value.clone() for key, value in self.output_factors.items()},
            "last_observation_step": self.last_observation_step,
            "last_root_step": self.last_root_step,
            "observations": self.observations,
            "root_refreshes": self.root_refreshes,
            "sample_count": self.sample_count,
        }

    def load_state_dict(self, state, *, device):
        for name in ("input_moments", "output_moments", "input_factors", "output_factors"):
            setattr(self, name, {key: value.to(device=device).clone() for key, value in state[name].items()})
        for name in ("last_observation_step", "last_root_step", "observations", "root_refreshes", "sample_count"):
            setattr(self, name, state[name])


class GoMuon(MatrixBaseline):
    """All seven block projections with shared ownership and auxiliary routing.

    Call begin_step, select_microbatch before each forward, finish_backward after
    accumulation, then step. The shared head hook may call begin, which is a
    no-op: both sides are formed together after backward, before the update.
    """

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
        self.settings = GoMuonSettings(**config["experiment"]["related_baseline"]["settings"])
        if (
            rows_per_microbatch <= 0
            or capacity <= 0
            or capacity % rows_per_microbatch != 0
            or capacity * world != rows_total
        ):
            raise ValueError("GO-MUON requires complete, equally sized accumulation buffers")
        blocks = list(model.body.blocks)
        self.input_rows = {
            (layer, kind): getattr(block, f"input_rows_{kind}")
            for layer, block in enumerate(blocks)
            for kind in INPUT_KINDS
        }
        if any(buffer.shape[0] != capacity for buffer in self.input_rows.values()):
            raise ValueError("Model input row capacity differs from GO-MUON capacity")
        self.input_at = model.body.stat_at
        self.output_rows, self.output_at = attach_output_rows(
            blocks, MATRIX_KINDS, positions, rows_per_microbatch=rows_per_microbatch, capacity=capacity, device=device
        )
        self.rows_per_microbatch, self.capacity, self.rows_total = rows_per_microbatch, capacity, rows_total
        self.statistic_config = {
            "decay": config["statistics"]["momentum"],
            "damping": config["statistics"]["damping"],
            "refresh": config["root"]["refresh"],
        }
        self.statistics = GoMuonStatistics(self.settings, **self.statistic_config)
        self.selected_microbatches = 0
        self.backward_finished = False
        self.roots_refreshed = False

    def begin_step(self, step):
        super().begin_step(step)
        self.selected_microbatches = 0
        self.backward_finished = self.roots_refreshed = False

    def select_microbatch(self, index):
        if index != self.selected_microbatches or (index + 1) * self.rows_per_microbatch > self.capacity:
            raise ValueError("GO-MUON microbatches must fill the capture buffer in order")
        start = index * self.rows_per_microbatch
        at = torch.arange(start, start + self.rows_per_microbatch, device=self.device)
        self.output_at.copy_(at)
        self.input_at.copy_(at)
        self.selected_microbatches += 1

    def _require_collectively_valid(self, checks, message):
        valid = (
            torch.stack(checks).all().to(torch.int32)
            if checks
            else torch.ones((), dtype=torch.int32, device=self.device)
        )
        if self.world > 1:
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not valid.item():
            raise FloatingPointError(message)

    def _validate_gradients(self):
        checks = [
            (
                torch.isfinite(parameter.grad).all()
                if parameter.grad is not None
                else torch.zeros((), dtype=torch.bool, device=self.device)
            )
            for parameter in self.owned.values()
        ]
        self._require_collectively_valid(checks, "Invalid GO-MUON owned gradient on at least one rank")

    @torch.no_grad()
    def _pooled_moments(self, buffers, local_count, *, row_scale=1.0):
        moments = {}
        owned_layers = {layer for layer, _ in self.owned}
        for key, buffer in buffers.items():
            rows = buffer[:local_count].float() / row_scale
            total = rows.mT @ rows
            if self.world > 1:
                dist.all_reduce(total)
            if key[0] in owned_layers:
                moments[key] = total / self.rows_total
        return moments

    @torch.no_grad()
    def finish_backward(self):
        if self.backward_finished:
            return
        # Coordinate failure before any rank updates statistics or enters roots.
        self._validate_gradients()
        local_count = self.selected_microbatches * self.rows_per_microbatch
        if local_count != self.capacity:
            raise RuntimeError("GO-MUON backward must fill every microbatch row before pooling")
        if self.statistics.observes(self.steps):
            count = torch.tensor(local_count, dtype=torch.int64, device=self.device)
            if self.world > 1:
                dist.all_reduce(count)
            if count.item() != self.rows_total:
                raise RuntimeError("GO-MUON pooled sample count differs from configured rows_total")
            inputs = self._pooled_moments(self.input_rows, local_count)
            # Remove capture's exact BF16 scaling to recover actual autograd
            # cotangents. These retain the runner's mean-loss scaling.
            outputs = self._pooled_moments(self.output_rows, local_count, row_scale=ROW_SCALE)
            self._require_collectively_valid(
                [torch.isfinite(moment).all() for moment in (*inputs.values(), *outputs.values())],
                "Invalid GO-MUON pooled raw moment on at least one rank",
            )
            self.roots_refreshed = self.statistics.update(self.steps, inputs, outputs, count.item())
        self.backward_finished = True

    @torch.no_grad()
    def step(self):
        if not self.backward_finished:
            raise RuntimeError("Call GO-MUON finish_backward before step")
        # Clipping and other runner work can change gradients after finish_backward.
        self._validate_gradients()
        optimizer = self.config["optimizer"]
        group = self.param_groups[0]
        gradient_scale = 1 - optimizer["momentum"] if self.settings.statistic_mode == "paper_sparse" else 1.0
        deltas = {}
        for key, parameter in self.owned.items():
            source = self.momentum_source(key, parameter.grad, gradient_scale=gradient_scale)
            left = self.statistics.output_factors[key]
            right = self.statistics.input_factors[key[0], INPUT_KIND[key[1]]]
            direction = matched_direction(
                source,
                left,
                right,
                polar_method=self.settings.polar_method,
                ns_steps=optimizer["ns_steps"],
                ns_eps=optimizer["ns_eps"],
                ns_mode=optimizer["ns_mode"],
            )
            scale = max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
            deltas[key] = -group["lr"] * (scale * direction + group["weight_decay"] * parameter.float())
        self.apply_updates(deltas)
        self.last_metrics = {
            "go_muon_root_age": self.steps - 1 - self.statistics.last_root_step,
            "go_muon_roots_refreshed": int(self.roots_refreshed),
            "go_muon_statistic_observations": self.statistics.observations,
            "go_muon_root_refreshes": self.statistics.root_refreshes,
            "go_muon_capture_always_on": 1,
        }
        self.backward_finished = False

    def runtime_state(self):
        """Checkpoint all mathematical state; row buffers are scratch space."""
        base = self.base_runtime_state()
        return {
            "source_revision": SOURCE_REVISION,
            "source_sha256": SOURCE_SHA256,
            "implementation": "paper_formula_not_author_code",
            "capture_policy": "always_on_rows_pooling_and_roots_gated",
            "cotangent_scaling": "runner_mean_loss",
            "matrix_decay": "isotropic_controlled_adapter_not_author_specification",
            "root_compute_dtype": "float64",
            "settings": asdict(self.settings),
            "statistic_config": self.statistic_config.copy(),
            "base": {
                "steps": base["steps"],
                "momentum": {key: value.clone() for key, value in base["momentum"].items()},
            },
            "statistics": self.statistics.state_dict(),
        }

    def restore_runtime_state(self, state):
        if (
            state["source_revision"] != SOURCE_REVISION
            or state["source_sha256"] != SOURCE_SHA256
            or state["settings"] != asdict(self.settings)
            or state["statistic_config"] != self.statistic_config
        ):
            raise ValueError("GO-MUON resume requires unchanged source and factor settings")
        saved = state["statistics"]
        expected_inputs = {(layer, INPUT_KIND[kind]) for layer, kind in self.owned}
        if saved["observations"] and (
            saved["input_moments"].keys() != expected_inputs or saved["output_moments"].keys() != self.owned.keys()
        ):
            raise ValueError("GO-MUON resume requires unchanged statistic ownership")
        self.restore_base_runtime_state(state["base"])
        self.statistics.load_state_dict(saved, device=self.device)
        self.selected_microbatches = 0
        self.backward_finished = self.roots_refreshed = False
