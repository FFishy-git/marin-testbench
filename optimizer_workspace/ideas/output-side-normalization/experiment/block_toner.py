# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input, output, or two-sided Toner factors on every block matrix.

Output cotangent rows are stored during backward and pooled by Toner's own statistics,
cached-root, and host-refresh machinery, so output factors target the current update's
moment exactly as input factors do. The diagonal arm keeps only per-coordinate moments.
"""

from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace

import torch
import torch.distributed as dist
from reproduce.marin.optimizer import fast_arguments
from toner.fast import muon as MU
from toner.fast.engine import KINDS_D, FastTonerCore, StatGroup, gather_uneven, layer_counts
from torch import nn
from torch.nn import functional as F

from .toner_state import engine_state, restore_engine

BLOCK_KINDS = ("q", "k", "v", "o", "gate", "up", "down")
INPUT_KIND = {"q": "attn", "k": "attn", "v": "attn", "o": "o", "gate": "mlp", "up": "mlp", "down": "down"}
INPUT_KINDS = (*KINDS_D, "down")
HIDDEN_OUTPUTS = ("q", "k", "v", "o", "down")
MLP_OUTPUTS = ("gate", "up")
# Exact in BF16. Trace normalization cancels any constant positive cotangent scale.
ROW_SCALE = 2.0**16


class Side(StrEnum):
    NONE = "none"
    FULL = "full"
    DIAGONAL = "diagonal"


@dataclass(frozen=True)
class BlockArm:
    input: Side
    # Output factors of hidden-width outputs (q, k, v, o, down) and MLP-width outputs (gate, up).
    hidden_output: Side
    mlp_output: Side
    # The null control observes and roots output moments but leaves them out of the update.
    apply_left: bool
    # Wave 0's best arm applied only the outer output factor; this generalizes it to every block.
    apply_left_inner: bool = True
    # Apply left factors one update after the rows they were built from, hiding the
    # output root wait behind the next update's forward and backward.
    lag_left: bool = False

    def output_side(self, kind):
        return self.hidden_output if kind in HIDDEN_OUTPUTS else self.mlp_output

    def output_kinds(self, side):
        return tuple(kind for kind in BLOCK_KINDS if self.output_side(kind) == side)


BLOCK_ARMS = {
    "inner_outer_all": BlockArm(Side.FULL, Side.FULL, Side.FULL, apply_left=True),
    "null_all": BlockArm(Side.FULL, Side.FULL, Side.FULL, apply_left=False),
    "left_all": BlockArm(Side.NONE, Side.FULL, Side.FULL, apply_left=True),
    "inner_outer_all_diag": BlockArm(Side.DIAGONAL, Side.DIAGONAL, Side.DIAGONAL, apply_left=True),
    # Cheaper output sides on MuonT's full input factors: no 2048-wide output roots.
    "inner_outer_hidden": BlockArm(Side.FULL, Side.FULL, Side.NONE, apply_left=True),
    "inner_outer_mlpdiag": BlockArm(Side.FULL, Side.FULL, Side.DIAGONAL, apply_left=True),
    "inner_outer_leftdiag": BlockArm(Side.FULL, Side.DIAGONAL, Side.DIAGONAL, apply_left=True),
    "outer_left_all": BlockArm(Side.FULL, Side.FULL, Side.FULL, apply_left=True, apply_left_inner=False),
    # Same factors as inner_outer_all, applied one update late so the root wait overlaps.
    "inner_outer_all_lag": BlockArm(Side.FULL, Side.FULL, Side.FULL, apply_left=True, lag_left=True),
    # The best-performing output geometry with the lagged application; wave-4 candidate.
    "inner_outer_mlpdiag_lag": BlockArm(Side.FULL, Side.FULL, Side.DIAGONAL, apply_left=True, lag_left=True),
    # Matched control for inner_outer_mlpdiag: same path and same work, identity left factors.
    "null_mlpdiag": BlockArm(Side.FULL, Side.FULL, Side.DIAGONAL, apply_left=False),
}


@torch.library.custom_op("output_marin::store_cotangent_rows", mutates_args=("rows",))
def store_cotangent_rows(gradient: torch.Tensor, rows: torch.Tensor, at: torch.Tensor, positions: torch.Tensor) -> None:
    if gradient.dtype != torch.bfloat16:
        raise TypeError(f"Output cotangent rows require BF16 projection gradients, got {gradient.dtype}")
    sampled = gradient.index_select(1, positions).reshape(-1, gradient.shape[-1])
    rows.index_copy_(0, at, sampled * ROW_SCALE)


@store_cotangent_rows.register_fake
def _(gradient, rows, at, positions):
    return None


class ObserveRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, rows, at, positions):
        # Backward mutates the rows buffer; keep it out of saved-tensor version checks.
        ctx.rows = rows
        ctx.save_for_backward(at, positions)
        return output.view_as(output)

    @staticmethod
    def backward(ctx, gradient):
        at, positions = ctx.saved_tensors
        store_cotangent_rows(gradient, ctx.rows, at, positions)
        return gradient, None, None, None


class RowObservedLinear(nn.Module):
    """The original linear map; backward stores sampled output-cotangent rows."""

    def __init__(self, linear, rows, at, positions):
        super().__init__()
        self.weight = linear.weight
        self.register_buffer("output_rows", rows, persistent=False)
        self.register_buffer("output_at", at, persistent=False)
        self.register_buffer("output_positions", positions, persistent=False)

    def forward(self, x):
        return ObserveRows.apply(F.linear(x, self.weight), self.output_rows, self.output_at, self.output_positions)


def attach_output_rows(blocks, kinds, positions, *, rows_per_microbatch, capacity, device):
    """Wrap the given block projections; return row buffers by (layer, kind) and the shared row index."""
    at = torch.arange(rows_per_microbatch, device=device)
    index = torch.tensor(positions, dtype=torch.int64, device=device)
    rows = {}
    for layer, block in enumerate(blocks):
        for kind in kinds:
            linear = getattr(block, kind)
            # Independent storage per projection keeps compiled backward version counters separate.
            rows[layer, kind] = torch.zeros(capacity, linear.weight.shape[0], dtype=torch.bfloat16, device=device)
            setattr(block, kind, RowObservedLinear(linear, rows[layer, kind], at, index))
    return rows, at


class OutputToner(FastTonerCore):
    """Toner's pooled statistics, cached roots, and host refresh over output-cotangent rows."""

    def __init__(self, rows, kinds, *, layers, hidden, mlp, rows_total, args, world, rank, device):
        # The parent constructor builds input groups from block buffers. Give it empty one-column
        # placeholders, then install the output groups before any statistic, root, or refresh runs.
        placeholder = torch.empty(0, 1, dtype=torch.bfloat16, device=device)
        blocks = [SimpleNamespace(**{f"input_rows_{kind}": placeholder for kind in INPUT_KINDS})] * layers
        super().__init__(
            blocks, {}, "muont", hid=1, mlp=1, rows_total=rows_total, args=args, world=world, rank=rank, device=device
        )
        widths = (
            ("out_hidden", tuple(k for k in kinds if k in HIDDEN_OUTPUTS), hidden),
            ("out_mlp", tuple(k for k in kinds if k in MLP_OUTPUTS), mlp),
        )
        self.groups = {
            name: StatGroup(
                [(layer, kind) for layer in range(layers) for kind in members],
                width,
                [rows[layer, kind] for layer in range(layers) for kind in members],
                [len(members) * count for count in self.layer_counts],
                rank,
                device,
            )
            for name, members, width in widths
            if members
        }
        for group in self.groups.values():
            group.host_tol = torch.empty(group.per, 3, pin_memory=True)
            group.tol = torch.full((group.per, 3), 1e-4, device=device)
        self.flags_host = torch.zeros(sum(g.per for g in self.groups.values()), dtype=torch.uint8, pin_memory=True)


class DiagonalMoments:
    """Pooled per-coordinate EMAs of sampled rows with exact elementwise inverse roots."""

    def __init__(self, bufs, *, decay, damping, inner_power, outer_power, rows_total, world, device):
        self.bufs = bufs
        self.decay, self.damping = decay, damping
        self.inner_power, self.outer_power = inner_power, outer_power
        self.rows_total, self.world = rows_total, world
        self.keys = {}
        for key, buffer in bufs.items():
            self.keys.setdefault(buffer.shape[-1], []).append(key)
        self.moment = {width: torch.zeros(len(keys), width, device=device) for width, keys in self.keys.items()}
        self.weight = 0.0
        self.pg = dist.new_group(backend=dist.get_backend()) if world > 1 else None

    @torch.no_grad()
    def update(self):
        sums = {
            width: torch.stack([self.bufs[key].float().square().sum(0) for key in keys])
            for width, keys in self.keys.items()
        }
        if self.world > 1:
            flat = torch.cat([value.reshape(-1) for value in sums.values()])
            dist.all_reduce(flat, group=self.pg)
            sums = dict(zip(sums, flat.split([value.numel() for value in sums.values()]), strict=True))
        for width, moment in self.moment.items():
            moment.mul_(self.decay).add_(sums[width].reshape(moment.shape), alpha=(1 - self.decay) / self.rows_total)
        self.weight = self.decay * self.weight + 1 - self.decay

    @torch.no_grad()
    def factors(self):
        """Return {key: (inner, outer)} diagonal roots; outer roots are not yet unit-RMS normalized."""
        factors = {}
        for width, moment in self.moment.items():
            inner, outer = diagonal_roots(moment, self.damping, self.inner_power, self.outer_power)
            for index, key in enumerate(self.keys[width]):
                factors[key] = (inner[index], outer[index])
        return factors

    def state_dict(self):
        return {"moment": self.moment, "weight": self.weight}

    def load_state_dict(self, state):
        for width, moment in self.moment.items():
            moment.copy_(state["moment"][width])
        self.weight = state["weight"]


def diagonal_roots(moment, damping, inner_power, outer_power):
    """Trace-normalized, damped elementwise inverse powers of rows of per-coordinate moments."""
    width = moment.shape[-1]
    trace = moment.sum(-1, keepdim=True)
    positive = trace > 0
    normalized = moment * (width / trace.clamp_min(torch.finfo(moment.dtype).tiny))
    normalized = torch.where(positive, normalized.clamp_min(0), torch.ones_like(moment)) + damping
    ones = torch.ones_like(moment)
    inner = torch.where(positive, normalized.pow(-inner_power), ones)
    outer = torch.where(positive, normalized.pow(-outer_power), ones)
    return inner, outer


def left_product(factor, matrix):
    return factor[:, None] * matrix if factor.ndim == 1 else factor @ matrix


def right_product(matrix, factor):
    return matrix * factor if factor.ndim == 1 else matrix @ factor


def unit_rms(factor):
    """Scale a dense or diagonal factor to unit RMS, ||F||_F^2 / d = 1."""
    return factor * (factor.shape[-1] ** 0.5 / factor.norm())


@dataclass(frozen=True)
class MatrixFactors:
    """Root factors of one matrix; None is identity. Diagonal factors are vectors."""

    left_inner: torch.Tensor | None = None
    left_outer: torch.Tensor | None = None
    right_inner: torch.Tensor | None = None
    right_outer: torch.Tensor | None = None


def block_update(weight, gradient, momentum, factors, *, args, lr, weight_decay):
    """Return updated momentum and the finite-NS parameter change of one block matrix.

    D = L_out NS(L_in U R_in) R_out with unit-RMS outer factors. Decay is input-shaped,
    W R_out^2, when right factors exist, else left-shaped, L_out^2 W.
    """
    momentum = args.muon_momentum * momentum + gradient
    source = gradient + args.muon_momentum * momentum if args.muon_nesterov else momentum
    if factors.right_inner is not None:
        source = right_product(source, factors.right_inner)
    if factors.left_inner is not None:
        source = left_product(factors.left_inner, source)
    direction = MU.newton_schulz(source, args.muon_ns_steps, args.muon_eps, args.muon_ns_mode)
    decay = weight
    if factors.right_outer is not None:
        right = unit_rms(factors.right_outer)
        direction = right_product(direction, right)
        decay = right_product(right_product(weight, right), right)
    if factors.left_outer is not None:
        left = unit_rms(factors.left_outer)
        direction = left_product(left, direction)
        if factors.right_outer is None:
            decay = left_product(left, left_product(left, weight))
    scale = max(1.0, weight.shape[-2] / weight.shape[-1]) ** 0.5
    return momentum, -lr * (scale * direction + weight_decay * decay)


def stack_factors(factors, *, leading, width):
    """Stack per-matrix factors of one side into a batched tensor, or None when every factor is absent.

    Diagonal factors become [B, width]; dense factors become [B, width, width].
    """
    if all(factor is None for factor in factors):
        return None
    if any(factor is None for factor in factors):
        raise ValueError("A stacked side must have a factor for every matrix or none")
    diagonal = factors[0].ndim == 1
    if any((factor.ndim == 1) != diagonal for factor in factors):
        raise ValueError("A stacked side must mix neither diagonal nor dense factors")
    if any(factor.shape[-1] != width for factor in factors):
        raise ValueError("A stacked side must match the matrix width")
    return torch.stack(factors)


def stack_unit_rms(factor):
    """Per-matrix unit-RMS scaling of a stack, ||F||_F^2 / d = 1 for each entry."""
    dims = (-2, -1) if factor.ndim == 3 else (-1,)
    return factor * (factor.shape[-1] ** 0.5 / factor.norm(dim=dims, keepdim=True))


def block_update_stack(weights, gradients, momenta, left, right, *, args, lr, weight_decay):
    """The batched form of block_update over a stack of same-shape matrices.

    Newton-Schulz and every factor product run as batched GEMMs, so one launch covers the
    whole shape group instead of one per matrix. Each side is a (inner, outer) pair of
    stacked factors, either [B, width] diagonal or [B, width, width] dense, or None.
    """
    left_inner, left_outer = left
    right_inner, right_outer = right
    momenta = args.muon_momentum * momenta + gradients
    source = gradients + args.muon_momentum * momenta if args.muon_nesterov else momenta
    if right_inner is not None:
        source = torch.bmm(source, right_inner) if right_inner.ndim == 3 else source * right_inner[:, None, :]
    if left_inner is not None:
        source = torch.bmm(left_inner, source) if left_inner.ndim == 3 else source * left_inner[:, :, None]
    direction = MU.newton_schulz_batched(source, args.muon_ns_steps, args.muon_eps, args.muon_ns_mode)
    decay = weights
    if right_outer is not None:
        right = stack_unit_rms(right_outer)
        direction = torch.bmm(direction, right) if right.ndim == 3 else direction * right[:, None, :]
        decay = torch.bmm(decay, right) if right.ndim == 3 else decay * right[:, None, :]
        decay = torch.bmm(decay, right) if right.ndim == 3 else decay * right[:, None, :]
    if left_outer is not None:
        left = stack_unit_rms(left_outer)
        direction = torch.bmm(left, direction) if left.ndim == 3 else direction * left[:, :, None]
        if right_outer is None:
            decay = torch.bmm(left, decay) if left.ndim == 3 else decay * left[:, :, None]
            decay = torch.bmm(left, decay) if left.ndim == 3 else decay * left[:, :, None]
    scale = max(1.0, weights.shape[-2] / weights.shape[-1]) ** 0.5
    return momenta, -lr * (scale * direction + weight_decay * decay)


def _check_output_settings(config):
    """Output factors use Toner's input statistic and root settings; refuse silent divergence."""
    output, statistics, root = config["experiment"]["output"], config["statistics"], config["root"]
    expected = {
        "momentum": statistics["momentum"],
        "damping": statistics["damping"],
        "inner_strength": statistics["inner_strength"],
        "outer_strength": statistics["outer_strength"],
        "refresh": root["refresh"],
    }
    if output != expected:
        raise ValueError(f"Block-arm output settings {output} must equal the input settings {expected}")


class BlockToner:
    """MuonT-family updates of all 224 block matrices with the arm's input and output factors."""

    def __init__(self, model, config, positions, *, rows_per_microbatch, capacity, rows_total, world, rank, device):
        _check_output_settings(config)
        self.arm = BLOCK_ARMS[config["experiment"]["arm"]]
        self.args = fast_arguments(config)
        self.world, self.rank, self.device = world, rank, device
        blocks = list(model.body.blocks)
        layers = len(blocks)
        hidden, mlp = config["model"]["hidden"], config["model"]["mlp"]
        statistics = config["statistics"]
        observed = tuple(kind for kind in BLOCK_KINDS if self.arm.output_side(kind) != Side.NONE)
        self.output_rows, self.output_at = attach_output_rows(
            blocks, observed, positions, rows_per_microbatch=rows_per_microbatch, capacity=capacity, device=device
        )
        self.rows_per_microbatch = rows_per_microbatch
        engine = {"rows_total": rows_total, "args": self.args, "world": world, "rank": rank, "device": device}
        self.input = (
            FastTonerCore(blocks, {}, "muont", hid=hidden, mlp=mlp, **engine) if self.arm.input == Side.FULL else None
        )
        full, diagonal_kinds = self.arm.output_kinds(Side.FULL), self.arm.output_kinds(Side.DIAGONAL)
        self.output = (
            OutputToner(self.output_rows, full, layers=layers, hidden=hidden, mlp=mlp, **engine) if full else None
        )
        diagonal = {
            "decay": statistics["momentum"],
            "damping": statistics["damping"],
            "inner_power": statistics["inner_strength"],
            "outer_power": statistics["outer_strength"],
            "rows_total": rows_total,
            "world": world,
            "device": device,
        }
        self.input_diagonal = (
            DiagonalMoments(
                {(i, k): getattr(blocks[i], f"input_rows_{k}") for i in range(layers) for k in INPUT_KINDS},
                **diagonal,
            )
            if self.arm.input == Side.DIAGONAL
            else None
        )
        self.output_diagonal = (
            DiagonalMoments(
                {key: rows for key, rows in self.output_rows.items() if key[1] in diagonal_kinds},
                **diagonal,
            )
            if diagonal_kinds
            else None
        )
        self.layer_counts = layer_counts(layers, world)
        owner = [r for r, count in enumerate(self.layer_counts) for _ in range(count)]
        entries = [(layer, kind, getattr(blocks[layer], kind).weight) for layer in range(layers) for kind in BLOCK_KINDS]
        self.params = [parameter for _, _, parameter in entries]
        by_shape = {}
        for entry in entries:
            by_shape.setdefault(tuple(entry[2].shape), []).append(entry)
        self.shape_groups = [
            (shape, [[e for e in group if owner[e[0]] == r] for r in range(world)]) for shape, group in by_shape.items()
        ]
        self.momentum = {
            shape: torch.zeros((len(per_rank[rank]), *shape), device=device) for shape, per_rank in self.shape_groups
        }
        self.param_groups = [{"params": self.params, "lr": 0.0, "weight_decay": self.args.wd}]
        self.steps = 0
        # Left factors of the previous update. Empty on the first update of a lagged arm,
        # which therefore applies input factors only, and unused by every other arm.
        self.lagged = {}
        # Diagonal left factors built from this update's rows, applied by the next one.
        # The dense side has no equivalent because its launch is still in flight.
        self.pending_diagonal = {}
        self.last_metrics = {}
        self.pg = dist.new_group(backend=dist.get_backend()) if world > 1 else None

    @property
    def engines(self):
        return [engine for engine in (self.input, self.output) if engine is not None]

    def select_microbatch(self, index):
        start = index * self.rows_per_microbatch
        self.output_at.copy_(torch.arange(start, start + self.rows_per_microbatch, device=self.device))

    def begin(self):
        """Input statistics and roots after the last microbatch's forward (Toner's head hook)."""
        if self.input is not None:
            self.input.begin()

    def finish_backward(self):
        """Output statistics and roots once every output cotangent row of this update exists.

        A lagged arm first collects the previous update's left factors, whose dense roots
        have had that update plus this one's forward and backward to finish, then launches
        this update's. Collecting here rather than in step() is what moves the host wait off
        the critical path, and it must happen before the next launch rewrites the graph
        outputs. The diagonal side is exact and immediate, so it is computed here too and
        held for the next update, which keeps one meaning of lag for every left factor.
        """
        if self.arm.lag_left:
            self.lagged = self._collect_lagged()
        if self.output is not None:
            self.output.begin()
        if self.arm.lag_left and self.output_diagonal is not None:
            self.output_diagonal.update()
            self.pending_diagonal = self.output_diagonal.factors()

    def _collect_lagged(self):
        """Left factors built from the previous update's rows, safe to outlive this launch."""
        dense = {}
        if self.output is not None and self.output.begun:
            dense = self._engine_factors(self.output, copy=True)
        if not dense and not self.pending_diagonal:
            return self.lagged
        return self.pending_diagonal | dense

    def zero_grad(self, set_to_none=True):
        for parameter in self.params:
            parameter.grad = None

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def _engine_factors(self, engine, copy=False):
        # Like Toner's step: if the hook did not begin this update, its rows are complete by now.
        if not engine.begun:
            engine.begin()
        engine.begun = False
        roots = engine._finish_roots()
        factors = {}
        for name, group in engine.groups.items():
            inner, outer = roots[name]
            for index, key in enumerate(group.own):
                # Cached-root graph outputs are rewritten by the next launch, so a factor
                # that outlives this update must be copied out of the graph pool.
                factors[key] = (inner[index].clone(), outer[index].clone()) if copy else (inner[index], outer[index])
        return factors

    def _factors(self, engine, diagonal):
        factors = {} if engine is None else self._engine_factors(engine)
        if diagonal is not None:
            diagonal.update()
            factors |= diagonal.factors()
        return factors

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        right = self._factors(self.input, self.input_diagonal)
        # A lagged arm already holds finished factors from the previous update and must not
        # wait on the launch still in flight for this one.
        left = self.lagged if self.arm.lag_left else self._factors(self.output, self.output_diagonal)
        self.last_metrics = self._engine_metrics()
        if not self.arm.apply_left:
            left = {}
        elif not self.arm.apply_left_inner:
            left = {key: (None, outer) for key, (_, outer) in left.items()}
        works, finite = [], []
        for shape, per_rank in self.shape_groups:
            momentum = self.momentum[shape]
            mine = per_rank[self.rank]
            if mine:
                factorise = [
                    (*right.get((layer, INPUT_KIND[kind]), (None, None)), *left.get((layer, kind), (None, None)))
                    for layer, kind, _ in mine
                ]

                right_inner, right_outer = (
                    stack_factors([entry[0] for entry in factorise], leading=len(mine), width=shape[1]),
                    stack_factors([entry[1] for entry in factorise], leading=len(mine), width=shape[1]),
                )
                left_inner, left_outer = (
                    stack_factors([entry[2] for entry in factorise], leading=len(mine), width=shape[0]),
                    stack_factors([entry[3] for entry in factorise], leading=len(mine), width=shape[0]),
                )
                momentum[:], delta = block_update_stack(
                    torch.stack([parameter for _, _, parameter in mine]),
                    torch.stack([parameter.grad.float() for _, _, parameter in mine]),
                    momentum,
                    (left_inner, left_outer),
                    (right_inner, right_outer),
                    args=self.args,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                )
            else:
                delta = torch.empty((0, *shape), device=self.device)
            finite.append(torch.isfinite(delta).all())
            counts = [len(entries) for entries in per_rank]
            if self.world > 1:
                works.append((per_rank, *gather_uneven(delta, counts, group=self.pg, async_op=True)))
            else:
                works.append((per_rank, None, [delta]))
        # One host wait for all shape groups rather than one each, taken after the gathers
        # are in flight so it overlaps them, and still before any parameter changes.
        # Rank-local by design: ranks own different matrices, so this is an early signal and
        # not a cross-rank guarantee. train.py all-reduces the loss and checks it every
        # update, which is the collective check. Launching the gathers before raising also
        # keeps peer ranks from blocking in work.wait() when one rank fails.
        checks = torch.stack(finite).tolist()
        if not all(checks):
            bad = [shape for (shape, _), ok in zip(self.shape_groups, checks, strict=True) if not ok]
            raise FloatingPointError(f"Nonfinite block update for shapes {bad}")
        for per_rank, work, gathered in works:
            if work is not None:
                work.wait()
            for entries, updates in zip(per_rank, gathered, strict=True):
                if entries:
                    torch._foreach_add_([parameter for _, _, parameter in entries], list(updates.unbind(0)))
        for engine in self.engines:
            engine.steps += 1
            engine.t += 1
        self.steps += 1

    def _engine_metrics(self):
        metrics = {}
        for label, engine in (("input", self.input), ("output", self.output)):
            if engine is None:
                continue
            ages = [engine.steps - step for g in engine.groups.values() for step in g.basis_step]
            metrics[f"{label}_fallbacks"] = engine.fallbacks
            metrics[f"{label}_extra_roots"] = engine.extra
            metrics[f"{label}_max_basis_age"] = max(ages, default=0)
            metrics[f"{label}_pending_refreshes"] = sum(p is not None for g in engine.groups.values() for p in g.pending)
        return metrics

    def runtime_state(self):
        # Collect the launch still in flight so the saved factors are the ones the next
        # update would apply. An in-flight launch cannot be serialized, so without this a
        # resumed run would apply factors one update older than a continuous run.
        if self.arm.lag_left:
            self.lagged = self._collect_lagged()
            self.pending_diagonal = {}
        return {
            "lagged": self.lagged,
            "input": None if self.input is None else engine_state(self.input),
            "output": None if self.output is None else engine_state(self.output),
            "input_diagonal": None if self.input_diagonal is None else self.input_diagonal.state_dict(),
            "output_diagonal": None if self.output_diagonal is None else self.output_diagonal.state_dict(),
            "momentum": self.momentum,
            "steps": self.steps,
        }

    def restore_runtime_state(self, state):
        for engine, saved in ((self.input, state["input"]), (self.output, state["output"])):
            if engine is not None:
                restore_engine(engine, saved)
        for diagonal, saved in (
            (self.input_diagonal, state["input_diagonal"]),
            (self.output_diagonal, state["output_diagonal"]),
        ):
            if diagonal is not None:
                diagonal.load_state_dict(saved)
        for shape, momentum in self.momentum.items():
            momentum.copy_(state["momentum"][shape])
        # Absent or empty means no lagged factors, which is the state every arm that does
        # not lag is saved in, and the state a lagged arm's first update starts from.
        self.lagged = {
            key: tuple(tensor.to(self.device) for tensor in pair) for key, pair in state.get("lagged", {}).items()
        }
        self.steps = state["steps"]
