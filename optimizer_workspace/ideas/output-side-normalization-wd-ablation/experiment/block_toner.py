# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input, output, or two-sided Toner factors on every block matrix.

Output cotangent rows are stored during backward via grouped Triton kernels (or accumulated
inline for diagonal output moments) and pooled by Toner's own statistics, cached-root, and
host-refresh machinery. Lagged arms overlap dense root candidates, repairs, exact fallbacks,
and factor splitting asynchronously on a side stream across updates.
"""

from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace

import torch
import torch._dynamo
import torch.distributed as dist
from reproduce.marin.optimizer import fast_arguments
from toner.fast import muon as MU
from toner.fast import roots as PR
from toner.fast.engine import KINDS_D, FastTonerCore, StatGroup, gather_uneven, layer_counts
from toner.fast.syrk import list_offsets, syrk_list_packed, unpack_ema_
from torch import nn
from torch.nn import functional as F

from .fast_cotangent import (
    ROW_SCALE,
     InlineDiagonalAccumulator,
    _store_cotangent_at_kernel,
    attach_fast_block_observers,
    reduce_row_squares,
)
from .toner_state import engine_state, restore_engine

torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)

BLOCK_KINDS = ("q", "k", "v", "o", "gate", "up", "down")
INPUT_KIND = {"q": "attn", "k": "attn", "v": "attn", "o": "o", "gate": "mlp", "up": "mlp", "down": "down"}
INPUT_KINDS = (*KINDS_D, "down")
HIDDEN_OUTPUTS = ("q", "k", "v", "o", "down")
MLP_OUTPUTS = ("gate", "up")


class Side(StrEnum):
    NONE = "none"
    FULL = "full"
    DIAGONAL = "diagonal"


class DecayMode(StrEnum):
    DEFAULT = "default"
    ISOTROPIC = "isotropic"
    TWO_SIDED = "two_sided"


@dataclass(frozen=True)
class BlockArm:
    hidden_input: Side
    mlp_input: Side
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
    # Narrow down-only ablations (Abl-DownOnly, Abl-DownSwap) observe and precondition only W_down.
    down_only_output: bool = False

    def input_side(self, input_kind):
        return self.mlp_input if input_kind == "down" else self.hidden_input

    def input_kinds(self, side):
        return tuple(kind for kind in INPUT_KINDS if self.input_side(kind) == side)

    def output_side(self, kind):
        if self.down_only_output:
            return self.hidden_output if kind == "down" else Side.NONE
        return self.hidden_output if kind in HIDDEN_OUTPUTS else self.mlp_output

    def output_kinds(self, side):
        return tuple(kind for kind in BLOCK_KINDS if self.output_side(kind) == side)


BLOCK_ARMS = {
    "inner_outer_all": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.FULL, apply_left=True),
    "null_all": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.FULL, apply_left=False),
    "left_all": BlockArm(Side.NONE, Side.NONE, Side.FULL, Side.FULL, apply_left=True),
    "inner_outer_all_diag": BlockArm(Side.DIAGONAL, Side.DIAGONAL, Side.DIAGONAL, Side.DIAGONAL, apply_left=True),
    # Cheaper output sides on MuonT's full input factors: no 2048-wide output roots.
    "inner_outer_hidden": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.NONE, apply_left=True),
    "inner_outer_mlpdiag": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.DIAGONAL, apply_left=True),
    "inner_outer_leftdiag": BlockArm(Side.FULL, Side.FULL, Side.DIAGONAL, Side.DIAGONAL, apply_left=True),
    "outer_leftdiag_upstream_fused": BlockArm(
        Side.FULL, Side.FULL, Side.DIAGONAL, Side.DIAGONAL, apply_left=True, apply_left_inner=False
    ),
    "null_leftdiag_upstream_fused": BlockArm(
        Side.FULL, Side.FULL, Side.DIAGONAL, Side.DIAGONAL, apply_left=False
    ),
    "outer_left_all": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.FULL, apply_left=True, apply_left_inner=False),
    # Narrow down-only ablations unified onto the same 3-pillar BlockToner engine.
    "outer_left_down": BlockArm(
        Side.FULL, Side.FULL, Side.FULL, Side.NONE, apply_left=True, apply_left_inner=False, down_only_output=True
    ),
    "down_hybrid": BlockArm(
        Side.FULL, Side.NONE, Side.FULL, Side.NONE, apply_left=True, down_only_output=True
    ),
    # Same factors as inner_outer_all, applied one update late so the root wait overlaps.
    "inner_outer_all_lag": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.FULL, apply_left=True, lag_left=True),
    # The best-performing output geometry with the lagged application; wave-4 candidate.
    "inner_outer_mlpdiag_lag": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.DIAGONAL, apply_left=True, lag_left=True),
    # Matched control for inner_outer_mlpdiag: same path and same work, identity left factors.
    "null_mlpdiag": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.DIAGONAL, apply_left=False),
    # Symmetric width-thresholded geometry: every d=512 side dense, every d=2048 side diagonal.
    "inner_outer_sym_mlpdiag": BlockArm(Side.FULL, Side.DIAGONAL, Side.FULL, Side.DIAGONAL, apply_left=True),
    "inner_outer_sym_mlpdiag_lag": BlockArm(
        Side.FULL, Side.DIAGONAL, Side.FULL, Side.DIAGONAL, apply_left=True, lag_left=True
    ),
}


@torch.library.custom_op("output_marin::store_cotangent_rows", mutates_args=("rows",))
def store_cotangent_rows(gradient: torch.Tensor, rows: torch.Tensor, at: torch.Tensor, positions: torch.Tensor) -> None:
    if gradient.dtype != torch.bfloat16:
        raise TypeError(f"Output cotangent rows require BF16 projection gradients, got {gradient.dtype}")
    if gradient.is_cuda and rows.is_contiguous():
        g = gradient if gradient.stride(-1) == 1 else gradient.contiguous()
        P, D = positions.numel(), g.shape[-1]
        import triton

        block = min(512, triton.next_power_of_2(max(16, D)))
        _store_cotangent_at_kernel[(g.shape[0] * P,)](
            g, rows, at, positions, P, D, g.stride(0), g.stride(1), SCALE=ROW_SCALE, BLOCK=block, num_warps=4
        )
        return
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

    def __init__(self, rows, kinds, *, layers, hidden, mlp, rows_total, args, world, rank, device, lagged=False):
        # The parent constructor builds input groups from block buffers. Give it empty one-column
        # placeholders, then install the output groups before any statistic, root, or refresh runs.
        placeholder = torch.empty(0, 1, dtype=torch.bfloat16, device=device)
        blocks = [SimpleNamespace(**{f"input_rows_{kind}": placeholder for kind in INPUT_KINDS})] * layers
        super().__init__(
            blocks, {}, "muont", hid=1, mlp=1, rows_total=rows_total, args=args, world=world, rank=rank, device=device
        )
        if lagged:
            # Lagged output work has a full update (~1s) to complete; keep normal priority so it
            # never preempts high-priority input roots (-1) or critical-path forward/backward kernels.
            self.side = torch.cuda.Stream(device=device, priority=0)
        self.ev_rows_consumed = torch.cuda.Event()
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

    def _stats(self):
        """Compute packed Grams, record row-buffer consumption immediately, then reduce-scatter and EMA."""
        b = self.decay
        for g in self.groups.values():
            if g.offs is None:
                g.offs = list_offsets(g.bufs)
            syrk_list_packed(g.bufs, g.offs, g.partial, acc=self.chunks > 0)
        # All row buffers have been consumed into g.partial on self.side; next step's backward may overwrite them
        # as soon as this event completes (~1.5 ms), without waiting on reduce-scatter or root iterations.
        self.ev_rows_consumed.record(self.side)
        works = []
        for g in self.groups.values():
            if self.world == 1:
                g.summed.copy_(g.partial)
            elif len(set(g.counts)) == 1:
                works.append(dist.reduce_scatter_tensor(g.summed, g.partial, group=self.pg_stats, async_op=True))
            else:
                works.append(dist.reduce_scatter(g.summed, list(g.partial.split(g.counts)), group=self.pg_stats, async_op=True))
        for w in works:
            w.wait()
        for g in self.groups.values():
            if g.per:
                unpack_ema_(g.summed, g.C, b, (1 - b) / self.rows_total)
        self.weight = b * self.weight + (1 - b)
        self.chunks = 0

    def finish_roots_on_side(self):
        """Finish flagged root repairs and exact fallbacks purely on self.side without stalling main_stream."""
        a = self.args
        self.ev_flags.synchronize()
        flags = self.flags_host.tolist()
        roots, o, todo = {}, 0, []
        for name, g in self.groups.items():
            R, B, _, state = self.cand[name]
            bad = [j for j in range(g.per) if flags[o + j]]
            o += g.per
            roots[name] = R
            if bad:
                todo.append((g, R, B, state, bad))
        if todo:
            with torch.cuda.stream(self.side):
                for g, R, B, state, still in todo:
                    flagged = list(still)
                    cur, hopeless = None, []
                    for _ in range(a.toner_repair_rounds if state is not None else 0):
                        k = 1 << (len(still) - 1).bit_length()
                        pad = still + still[-1:] * (k - len(still))
                        idx = torch.tensor(pad, device=self.device)
                        if cur is None:
                            st = tuple(t.index_select(0, idx) for t in state)
                        else:
                            pos = torch.tensor([cur[0].index(j) for j in pad], device=self.device)
                            st = tuple(t.index_select(0, pos) for t in cur[1])
                        lam, O, rho, est, dl = self.froot.froot_more(B.index_select(0, idx), st, 2, a.toner_freeze)
                        if self.split:
                            Rp2, est_s = PR.form_powers(
                                PR.Basis.of_splits(g.basis.S2.index_select(0, idx)), lam, O
                            )
                            R2 = (Rp2[self.p_in], Rp2[self.p_out])
                            okl = (
                                (rho <= a.toner_rho_max)
                                & ~self._split_over(est, est_s, g.tol.index_select(0, idx))
                                & torch.isfinite(lam).all(-1)
                            ).tolist()
                        else:
                            R2 = PR.form(PR.Basis.of_splits(g.basis.S2.index_select(0, idx)), lam, O)
                            okl = (
                                (rho <= a.toner_rho_max)
                                & (est <= g.tol.index_select(0, idx))
                                & torch.isfinite(lam).all(-1)
                            ).tolist()
                        rhok = (rho <= a.toner_rho_max).tolist()
                        for i, j in enumerate(still):
                            if okl[i]:
                                if self.split:
                                    R[0][j].copy_(R2[0][i])
                                    if R[1] is not R[0]:
                                        R[1][j].copy_(R2[1][i])
                                else:
                                    R[j].copy_(R2[i])
                        self.extra += sum(okl[: len(still)])
                        hopeless += [j for i, j in enumerate(still) if not okl[i] and not rhok[i]]
                        cur = (pad, (lam, O, dl))
                        still = [j for i, j in enumerate(still) if not okl[i] and rhok[i]]
                        if not still:
                            break
                    still = still + hopeless
                    for j in still:
                        self.fallbacks += 1
                        if self.split:
                            (V, Rxp), lam_x = PR.exact_eigh_powers(g.C[j : j + 1], True)
                            g.tol[j : j + 1].copy_(self._tolerance(lam_x))
                            R[0][j].copy_(Rxp[self.p_in][0])
                            if R[1] is not R[0]:
                                R[1][j].copy_(Rxp[self.p_out][0])
                        else:
                            (V, Rx), lam_x = PR.exact_eigh(g.C[j : j + 1], True)
                            g.tol[j : j + 1].copy_(self._tolerance(lam_x))
                            R[j].copy_(Rx[0])
                        g.basis.install(j, V[0])
                        g.basis_step[j] = self.steps
                    if self.refresh_flagged:
                        for j in flagged:
                            if j not in still and g.pending[j] is None:
                                self._submit_refresh(g, j)
        return roots


class DiagonalMoments:
    """Pooled per-coordinate EMAs of sampled rows with exact elementwise inverse roots."""

    def __init__(
        self,
        bufs,
        *,
        decay,
        damping,
        inner_power,
        outer_power,
        rows_total,
        world,
        device,
        keys_by_width=None,
        inline: InlineDiagonalAccumulator | None = None,
    ):
        self.bufs = bufs
        self.inline = inline
        self.decay, self.damping = decay, damping
        self.inner_power, self.outer_power = inner_power, outer_power
        self.rows_total, self.world = rows_total, world
        if keys_by_width is not None:
            self.keys = {w: list(ks) for w, ks in keys_by_width.items() if ks}
        else:
            self.keys = {}
            for key, buffer in bufs.items():
                self.keys.setdefault(buffer.shape[-1], []).append(key)
        self.moment = {width: torch.zeros(len(keys), width, device=device) for width, keys in self.keys.items()}
        self._scratch = {}
        self.weight = 0.0
        self.pg = dist.new_group(backend=dist.get_backend()) if world > 1 else None

    @torch.no_grad()
    def local_sums(self):
        sums = {}
        for width, keys in self.keys.items():
            if self.inline is not None and self.inline.is_active(width):
                sums[width] = self.inline.sums(width)
            else:
                buffers = [self.bufs[key] for key in keys]
                if buffers[0].is_cuda and buffers[0].dtype == torch.bfloat16:
                    if width not in self._scratch:
                        self._scratch[width] = torch.empty(len(keys), 8, width, dtype=torch.float32, device=buffers[0].device)
                    sums[width] = reduce_row_squares(buffers, self._scratch[width])
                else:
                    sums[width] = torch.stack([b.float().square().sum(0) for b in buffers])
        return sums

    @torch.no_grad()
    def finish_update(self, sums):
        if self.world > 1:
            flat = torch.cat([value.reshape(-1) for value in sums.values()])
            dist.all_reduce(flat, group=self.pg)
            sums = dict(zip(sums, flat.split([value.numel() for value in sums.values()]), strict=True))
        for width, moment in self.moment.items():
            moment.mul_(self.decay).add_(sums[width].reshape(moment.shape), alpha=(1 - self.decay) / self.rows_total)
        self.weight = self.decay * self.weight + 1 - self.decay

    @torch.no_grad()
    def update(self):
        self.finish_update(self.local_sums())

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


def block_update(weight, gradient, momentum, factors, *, args, lr, weight_decay, decay_mode: str = DecayMode.DEFAULT):
    """Return updated momentum and the finite-NS parameter change of one block matrix.

    D = L_out NS(L_in U R_in) R_out with unit-RMS outer factors. Under decay_mode='default',
    decay is input-shaped (W R_out^2) when right factors exist, else left-shaped (L_out^2 W);
    'isotropic' uses W; 'two_sided' uses L_out^2 W R_out^2 (with missing sides = I).
    """
    mode = DecayMode(decay_mode)
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
        if mode in (DecayMode.DEFAULT, DecayMode.TWO_SIDED):
            decay = right_product(right_product(decay, right), right)
    if factors.left_outer is not None:
        left = unit_rms(factors.left_outer)
        direction = left_product(left, direction)
        if mode == DecayMode.TWO_SIDED or (mode == DecayMode.DEFAULT and factors.right_outer is None):
            decay = left_product(left, left_product(left, decay))
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
    return factor * (factor.shape[-1] ** 0.5 / factor.norm(dim=dims, keepdim=True).clamp_min(1e-30))


# ---------------------------------------------------------------------------
# Ozaki FP16-split Tensor-Core products (FP32-accurate with allow_tf32=False)
# ---------------------------------------------------------------------------
_LO = PR.LO  # 4096.0


def _hi_lo(x: torch.Tensor):
    h = x.half()
    return h, ((x - h.float()) * _LO).half()


def _cols_hi_scaled_raw(x: torch.Tensor):
    a = x.abs().amax((-2, -1), keepdim=True).clamp_min(1e-30)
    s = torch.exp2(13.0 - torch.floor(torch.log2(a)))
    h, lo = _hi_lo(x * s)
    return torch.cat([h, lo], -1), 1.0 / s


def _rows_lo_scaled_raw(x: torch.Tensor):
    a = x.abs().amax((-2, -1), keepdim=True).clamp_min(1e-30)
    s = torch.exp2(13.0 - torch.floor(torch.log2(a)))
    h, lo = _hi_lo(x * s)
    return torch.cat([lo, h], -2), 1.0 / s


def _apply16_right_raw(A: torch.Tensor, R2: torch.Tensor):
    A2, inv_A = _cols_hi_scaled_raw(A)
    k = A2.size(-1) // 2
    C = torch.bmm(A2[..., :k], R2[..., k:, :], out_dtype=torch.float32)
    return (C + torch.bmm(A2, R2, out_dtype=torch.float32) * (1.0 / _LO)) * inv_A


def _apply16_left_raw(L2: torch.Tensor, A: torch.Tensor):
    A2, inv_A = _rows_lo_scaled_raw(A)
    k = L2.size(-1) // 2
    C = torch.bmm(L2[..., :k], A2[..., k:, :], out_dtype=torch.float32)
    return (C + torch.bmm(L2, A2, out_dtype=torch.float32) * (1.0 / _LO)) * inv_A


def _prepare_right_dense_raw(factor: torch.Tensor, normalize_rms: bool):
    if normalize_rms:
        factor = stack_unit_rms(factor)
    return PR._rows_lo(factor)


def _prepare_left_dense_raw(factor: torch.Tensor, normalize_rms: bool):
    if normalize_rms:
        factor = stack_unit_rms(factor)
    return PR._cols_hi(factor)


def _block_in_raw(Gs, momenta, left_in, right_in, mu: float, nesterov: bool, right_dense: bool, left_dense: bool):
    G = Gs.float() if isinstance(Gs, torch.Tensor) else torch.stack(Gs).float()
    momenta = mu * momenta + G
    source = G + mu * momenta if nesterov else momenta
    if right_in is not None:
        source = _apply16_right_raw(source, right_in) if right_dense else source * right_in[:, None, :]
    if left_in is not None:
        source = _apply16_left_raw(left_in, source) if left_dense else source * left_in[:, :, None]
    return momenta, source


def _block_out_raw(Ps, direction, left_out, right_out, lr_t, wd_t, scale: float, right_dense: bool, left_dense: bool):
    P = Ps.float() if isinstance(Ps, torch.Tensor) else torch.stack(Ps).float()
    if right_out is not None:
        if left_out is None:
            dir_l = direction
        elif left_dense:
            dir_l = _apply16_left_raw(left_out, direction)
        else:
            dir_l = direction * left_out[:, :, None]

        if right_dense:
            WR = _apply16_right_raw(P, right_out)
            V = (lr_t * wd_t) * WR + (lr_t * scale) * dir_l
            return -_apply16_right_raw(V, right_out)
        r = right_out[:, None, :]
        return -lr_t * (scale * (dir_l * r) + wd_t * (P * (r * r)))

    if left_out is None:
        return -lr_t * (scale * direction + wd_t * P)
    if left_dense:
        LW = _apply16_left_raw(left_out, P)
        V = (lr_t * wd_t) * LW + (lr_t * scale) * direction
        return -_apply16_left_raw(left_out, V)
    l = left_out[:, :, None]
    return -lr_t * (scale * (l * direction) + wd_t * (l * l * P))


def _block_out_isotropic_raw(
    Ps, direction, left_out, right_out, lr_t, wd_t, scale: float, right_dense: bool, left_dense: bool
):
    P = Ps.float() if isinstance(Ps, torch.Tensor) else torch.stack(Ps).float()
    if left_out is None:
        dir_l = direction
    elif left_dense:
        dir_l = _apply16_left_raw(left_out, direction)
    else:
        dir_l = direction * left_out[:, :, None]

    if right_out is None:
        dir_lr = dir_l
    elif right_dense:
        dir_lr = _apply16_right_raw(dir_l, right_out)
    else:
        dir_lr = dir_l * right_out[:, None, :]

    return -lr_t * (scale * dir_lr + wd_t * P)


def _block_out_two_sided_raw(
    Ps, direction, left_out, right_out, lr_t, wd_t, scale: float, right_dense: bool, left_dense: bool
):
    P = Ps.float() if isinstance(Ps, torch.Tensor) else torch.stack(Ps).float()
    if left_out is None:
        WL = P
    elif left_dense:
        WL = _apply16_left_raw(left_out, P)
    else:
        WL = P * left_out[:, :, None]

    if right_out is None:
        WLR = WL
    elif right_dense:
        WLR = _apply16_right_raw(WL, right_out)
    else:
        WLR = WL * right_out[:, None, :]

    V = (lr_t * wd_t) * WLR + (lr_t * scale) * direction

    if right_out is None:
        VR = V
    elif right_dense:
        VR = _apply16_right_raw(V, right_out)
    else:
        VR = V * right_out[:, None, :]

    if left_out is None:
        return -VR
    if left_dense:
        return -_apply16_left_raw(left_out, VR)
    return -(VR * left_out[:, :, None])


_prepare_right_dense = torch.compile(_prepare_right_dense_raw, dynamic=False, options=PR._EXACT)
_prepare_left_dense = torch.compile(_prepare_left_dense_raw, dynamic=False, options=PR._EXACT)
_compiled_block_in = torch.compile(_block_in_raw, dynamic=False, options=PR._EXACT)
_compiled_block_out = torch.compile(_block_out_raw, dynamic=False, options=PR._EXACT)
_compiled_block_out_isotropic = torch.compile(_block_out_isotropic_raw, dynamic=False, options=PR._EXACT)
_compiled_block_out_two_sided = torch.compile(_block_out_two_sided_raw, dynamic=False, options=PR._EXACT)


def _select_block_out_fn(mode: DecayMode, *, compiled: bool):
    if mode == DecayMode.DEFAULT:
        return _compiled_block_out if compiled else _block_out_raw
    if mode == DecayMode.ISOTROPIC:
        return _compiled_block_out_isotropic if compiled else _block_out_isotropic_raw
    if mode == DecayMode.TWO_SIDED:
        return _compiled_block_out_two_sided if compiled else _block_out_two_sided_raw
    raise ValueError(f"Unsupported decay_mode: {mode}")


def _prepare_side_factor(factor: torch.Tensor | None, *, side: str, normalize_rms: bool, compiled: bool):
    if factor is None:
        return None
    if factor.ndim == 2:
        return stack_unit_rms(factor) if normalize_rms else factor
    if side == "right":
        return (_prepare_right_dense if compiled else _prepare_right_dense_raw)(factor, normalize_rms)
    return (_prepare_left_dense if compiled else _prepare_left_dense_raw)(factor, normalize_rms)


def block_update_stack(
    weights,
    gradients,
    momenta,
    left,
    right,
    *,
    args,
    lr,
    weight_decay,
    compiled=True,
    decay_mode: str = DecayMode.DEFAULT,
):
    """The batched form of block_update over a stack of same-shape matrices.

    On CUDA, dense factor products use FP32-accurate Ozaki FP16-split Tensor-Core GEMMs and
    fuse the post-NS direction and weight-decay products.
    """
    mode = DecayMode(decay_mode)
    left_inner, left_outer = left
    right_inner, right_outer = right
    if weights.is_cuda:
        use_compile = bool(compiled)
        li = _prepare_side_factor(left_inner, side="left", normalize_rms=False, compiled=use_compile)
        lo = _prepare_side_factor(left_outer, side="left", normalize_rms=True, compiled=use_compile)
        ri = _prepare_side_factor(right_inner, side="right", normalize_rms=False, compiled=use_compile)
        ro = _prepare_side_factor(right_outer, side="right", normalize_rms=True, compiled=use_compile)
        in_fn = _compiled_block_in if use_compile else _block_in_raw
        out_fn = _select_block_out_fn(mode, compiled=use_compile)
        lr_t = lr if isinstance(lr, torch.Tensor) else torch.tensor(lr, dtype=torch.float32, device=weights.device)
        wd_t = (
            weight_decay
            if isinstance(weight_decay, torch.Tensor)
            else torch.tensor(weight_decay, dtype=torch.float32, device=weights.device)
        )
        momenta, source = in_fn(
            gradients,
            momenta,
            li,
            ri,
            float(args.muon_momentum),
            bool(args.muon_nesterov),
            right_inner is not None and right_inner.ndim == 3,
            left_inner is not None and left_inner.ndim == 3,
        )
        direction = MU.newton_schulz_batched(source, args.muon_ns_steps, args.muon_eps, args.muon_ns_mode)
        scale = max(1.0, weights.shape[-2] / weights.shape[-1]) ** 0.5
        delta = out_fn(
            weights,
            direction,
            lo,
            ro,
            lr_t,
            wd_t,
            scale,
            right_outer is not None and right_outer.ndim == 3,
            left_outer is not None and left_outer.ndim == 3,
        )
        return momenta, delta

    momenta = args.muon_momentum * momenta + gradients
    source = gradients + args.muon_momentum * momenta if args.muon_nesterov else momenta
    if right_inner is not None:
        source = torch.bmm(source, right_inner) if right_inner.ndim == 3 else source * right_inner[:, None, :]
    if left_inner is not None:
        source = torch.bmm(left_inner, source) if left_inner.ndim == 3 else source * left_inner[:, :, None]
    direction = MU.newton_schulz_batched(source, args.muon_ns_steps, args.muon_eps, args.muon_ns_mode)
    decay = weights
    if right_outer is not None:
        right_norm = stack_unit_rms(right_outer)
        direction = torch.bmm(direction, right_norm) if right_norm.ndim == 3 else direction * right_norm[:, None, :]
        if mode in (DecayMode.DEFAULT, DecayMode.TWO_SIDED):
            decay = torch.bmm(decay, right_norm) if right_norm.ndim == 3 else decay * right_norm[:, None, :]
            decay = torch.bmm(decay, right_norm) if right_norm.ndim == 3 else decay * right_norm[:, None, :]
    if left_outer is not None:
        left_norm = stack_unit_rms(left_outer)
        direction = torch.bmm(left_norm, direction) if left_norm.ndim == 3 else direction * left_norm[:, :, None]
        if mode == DecayMode.TWO_SIDED or (mode == DecayMode.DEFAULT and right_outer is None):
            decay = torch.bmm(left_norm, decay) if left_norm.ndim == 3 else decay * left_norm[:, :, None]
            decay = torch.bmm(left_norm, decay) if left_norm.ndim == 3 else decay * left_norm[:, :, None]
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
        self.decay_mode = DecayMode(config["experiment"].get("decay_mode", DecayMode.DEFAULT))
        self.args = fast_arguments(config)
        self.world, self.rank, self.device = world, rank, device
        self._fast_cuda = torch.device(device).type == "cuda"
        blocks = list(model.body.blocks)
        layers = len(blocks)
        hidden, mlp = config["model"]["hidden"], config["model"]["mlp"]
        statistics = config["statistics"]
        observed = tuple(kind for kind in BLOCK_KINDS if self.arm.output_side(kind) != Side.NONE)

        fast_attached = attach_fast_block_observers(
            blocks,
            hidden_mode=str(self.arm.hidden_output),
            mlp_mode=str(self.arm.mlp_output),
            positions=positions,
            capacity=capacity,
            device=device,
            down_only=self.arm.down_only_output,
        )
        if fast_attached is not None:
            self.output_rows, self._row_off, self._idx, self._inline_diag = fast_attached
            self.output_at = None
        else:
            self.output_rows, self.output_at = attach_output_rows(
                blocks, observed, positions, rows_per_microbatch=rows_per_microbatch, capacity=capacity, device=device
            )
            self._row_off = None
            self._inline_diag = None

        self.rows_per_microbatch = rows_per_microbatch
        max_microbatches = max(1, capacity // max(1, rows_per_microbatch))
        self._host_row_offsets = (
            torch.tensor([i * rows_per_microbatch for i in range(max_microbatches)], dtype=torch.int64).pin_memory()
            if self._fast_cuda
            else None
        )
        self._step_scalars_cpu = torch.zeros(2, dtype=torch.float32).pin_memory() if self._fast_cuda else None

        engine = {"rows_total": rows_total, "args": self.args, "world": world, "rank": rank, "device": device}
        input_full_kinds = self.arm.input_kinds(Side.FULL)
        input_diag_kinds = self.arm.input_kinds(Side.DIAGONAL)
        if input_full_kinds == KINDS_D:
            placeholder = torch.empty(0, hidden, dtype=torch.bfloat16, device=device)
            input_blocks = [
                SimpleNamespace(
                    input_rows_attn=b.input_rows_attn,
                    input_rows_o=b.input_rows_o,
                    input_rows_mlp=b.input_rows_mlp,
                    input_rows_down=placeholder,
                )
                for b in blocks
            ]
            self.input = FastTonerCore(input_blocks, {}, "muont", hid=hidden, mlp=hidden, **engine)
            del self.input.groups["f"]
            self.input.flags_host = torch.zeros(
                sum(g.per for g in self.input.groups.values()), dtype=torch.uint8, pin_memory=True
            )
        elif input_full_kinds == INPUT_KINDS:
            self.input = FastTonerCore(blocks, {}, "muont", hid=hidden, mlp=mlp, **engine)
        elif not input_full_kinds:
            self.input = None
        else:
            raise ValueError(f"Unsupported dense input kinds: {input_full_kinds}")

        full, diagonal_kinds = self.arm.output_kinds(Side.FULL), self.arm.output_kinds(Side.DIAGONAL)
        self.output = (
            OutputToner(
                self.output_rows,
                full,
                layers=layers,
                hidden=hidden,
                mlp=mlp,
                lagged=self.arm.lag_left,
                **engine,
            )
            if full
            else None
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
        in_diag_keys_by_width = {}
        for layer_idx in range(layers):
            for kind in input_diag_kinds:
                buf = getattr(blocks[layer_idx], f"input_rows_{kind}")
                in_diag_keys_by_width.setdefault(buf.shape[-1], []).append((layer_idx, kind))
        self.input_diagonal = (
            DiagonalMoments(
                {(i, k): getattr(blocks[i], f"input_rows_{k}") for i in range(layers) for k in input_diag_kinds},
                keys_by_width=in_diag_keys_by_width,
                **diagonal,
            )
            if input_diag_kinds
            else None
        )
        diag_keys_by_width = {}
        for layer_idx in range(layers):
            for kind in diagonal_kinds:
                w = getattr(blocks[layer_idx], kind).weight.shape[0]
                diag_keys_by_width.setdefault(w, []).append((layer_idx, kind))
        self.output_diagonal = (
            DiagonalMoments(
                self.output_rows,
                keys_by_width=diag_keys_by_width,
                inline=self._inline_diag,
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
        self.lagged = {}
        self.pending_diagonal = {}
        self._lagged_prepared = {}
        self._input_diag_begun = False
        self._input_diag_factors = {}
        self.last_metrics = {}
        self.defer_finite_check = False
        self._last_finite_checks = None
        self.pg = dist.new_group(backend=dist.get_backend()) if world > 1 else None

        if self._fast_cuda:
            self._main_stream = torch.cuda.current_stream(device)
            self._input_side_stream = (
                self.input.side if self.input is not None else torch.cuda.Stream(device=device, priority=-1)
            )
            self._side_stream = (
                self.output.side if self.output is not None else torch.cuda.Stream(device=device, priority=0)
            )
            self._ev_input_diag_ready = torch.cuda.Event()
            self._ev_rows_consumed = torch.cuda.Event()
            self._ev_lagged_ready = torch.cuda.Event()
            self._rows_guard_active = False
            self._input_gather_spec = {}
            if self.input is not None:
                own_maps = {name: {key: i for i, key in enumerate(g.own)} for name, g in self.input.groups.items()}
                for shape, per_rank in self.shape_groups:
                    mine = per_rank[self.rank]
                    if not mine:
                        continue
                    keys = [(layer, INPUT_KIND[kind]) for layer, kind, _ in mine]
                    for g_name, g_map in own_maps.items():
                        if keys[0] in g_map:
                            if keys == list(self.input.groups[g_name].own):
                                self._input_gather_spec[shape] = (g_name, None)
                            else:
                                idx = torch.tensor([g_map[k] for k in keys], dtype=torch.int64, device=device)
                                self._input_gather_spec[shape] = (g_name, idx)
                            break
            self._output_gather_spec = {}
            if self.output is not None:
                out_maps = {name: {key: i for i, key in enumerate(g.own)} for name, g in self.output.groups.items()}
                for shape, per_rank in self.shape_groups:
                    mine = per_rank[self.rank]
                    if not mine:
                        continue
                    keys = [(layer, kind) for layer, kind, _ in mine]
                    for g_name, g_map in out_maps.items():
                        if keys[0] in g_map:
                            if keys == list(self.output.groups[g_name].own):
                                self._output_gather_spec[shape] = (g_name, None)
                            else:
                                idx = torch.tensor([g_map[k] for k in keys], dtype=torch.int64, device=device)
                                self._output_gather_spec[shape] = (g_name, idx)
                            break

    @property
    def engines(self):
        return [engine for engine in (self.input, self.output) if engine is not None]

    def select_microbatch(self, index):
        if self._fast_cuda and index == 0:
            if self._rows_guard_active:
                self._main_stream.wait_event(self._ev_rows_consumed)
                self._rows_guard_active = False
            if self._inline_diag is not None:
                self._inline_diag.zero_()
        if self._row_off is not None:
            if self._fast_cuda and index < self._host_row_offsets.numel():
                self._row_off.copy_(self._host_row_offsets[index], non_blocking=True)
            else:
                self._row_off.fill_(index * self.rows_per_microbatch)
        elif self.output_at is not None:
            start = index * self.rows_per_microbatch
            self.output_at.copy_(torch.arange(start, start + self.rows_per_microbatch, device=self.device))

        # Opportunistically finish previous step's flagged output repairs/fallbacks + factor splits on
        # _side_stream once microbatch 0's forward+backward (~235 ms GPU work) is already running on main_stream.
        if (
            self._fast_cuda
            and self.arm.lag_left
            and index >= 1
            and ((self.output is not None and self.output.begun) or self.pending_diagonal)
            and (self.output is None or not self.output.begun or self.output.ev_flags.query())
        ):
            self.lagged = self._collect_lagged()

    def _begin_input_diagonal(self):
        ev_fwd = torch.cuda.Event()
        ev_fwd.record(self._main_stream)
        with torch.cuda.stream(self._input_side_stream):
            self._input_side_stream.wait_event(ev_fwd)
            self.input_diagonal.update()
            raw = self.input_diagonal.factors()
            for pair in raw.values():
                for t in pair:
                    t.record_stream(self._main_stream)
            self._input_diag_factors = raw
            self._ev_input_diag_ready.record(self._input_side_stream)
        self._input_diag_begun = True

    def begin(self):
        """Input statistics and roots after the last microbatch's forward (Toner's head hook)."""
        if self.input is not None:
            self.input.begin()
        if getattr(self, "_fast_cuda", False) and self.input_diagonal is not None and not self._input_diag_begun:
            self._begin_input_diagonal()

    def _prepare_lagged_for_groups_on_side(self, factors: dict):
        """Pre-stack and FP16-split lagged left factors on _side_stream so step() pays 0 ms for them."""
        if not factors or not self.arm.apply_left:
            return {}
        prepared = {}
        with torch.cuda.stream(self._side_stream):
            for shape, per_rank in self.shape_groups:
                mine = per_rank[self.rank]
                if not mine:
                    continue
                pairs = [factors.get((layer, kind), (None, None)) for layer, kind, _ in mine]
                li_list = [p[0] for p in pairs] if self.arm.apply_left_inner else [None] * len(mine)
                lo_list = [p[1] for p in pairs]
                li_stack = stack_factors(li_list, leading=len(mine), width=shape[0])
                lo_stack = stack_factors(lo_list, leading=len(mine), width=shape[0])
                li_prep = _prepare_side_factor(li_stack, side="left", normalize_rms=False, compiled=True)
                lo_prep = _prepare_side_factor(lo_stack, side="left", normalize_rms=True, compiled=True)
                for item in (li_prep, lo_prep):
                    if isinstance(item, tuple):
                        for t in item:
                            t.record_stream(self._main_stream)
                    elif isinstance(item, torch.Tensor):
                        item.record_stream(self._main_stream)
                prepared[shape] = (li_prep, lo_prep)
        return prepared

    def finish_backward(self):
        """Output statistics and roots once every output cotangent row of this update exists."""
        if not getattr(self, "_fast_cuda", False):
            if self.arm.lag_left:
                self.lagged = self._collect_lagged()
            if self.output is not None:
                self.output.begin()
            if self.arm.lag_left and self.output_diagonal is not None:
                self.output_diagonal.update()
                self.pending_diagonal = self.output_diagonal.factors()
            return

        if self.arm.lag_left:
            self.lagged = self._collect_lagged()

        ev_bwd = torch.cuda.Event()
        ev_bwd.record(self._main_stream)

        diag_sums = None
        if self.arm.lag_left and self.output_diagonal is not None:
            with torch.cuda.stream(self._side_stream):
                self._side_stream.wait_event(ev_bwd)
                diag_sums = self.output_diagonal.local_sums()

        if self.output is not None:
            self.output.begin()

        with torch.cuda.stream(self._side_stream):
            if self.output is None:
                self._side_stream.wait_event(ev_bwd)
            else:
                self._side_stream.wait_event(self.output.ev_rows_consumed)
            self._ev_rows_consumed.record(self._side_stream)
            self._rows_guard_active = True

            if diag_sums is not None:
                self.output_diagonal.finish_update(diag_sums)
                self.pending_diagonal = self.output_diagonal.factors()
                for pair in self.pending_diagonal.values():
                    for t in pair:
                        t.record_stream(self._main_stream)

    def _collect_lagged(self):
        """Left factors built from the previous update's rows, safe to outlive this launch."""
        if not getattr(self, "_fast_cuda", False):
            dense = {}
            if self.output is not None and self.output.begun:
                dense = self._engine_factors(self.output, copy=True)
            if not dense and not self.pending_diagonal:
                return self.lagged
            combined = self.pending_diagonal | dense
            self.pending_diagonal = {}
            return combined

        dense = {}
        if self.output is not None and self.output.begun:
            self.output.begun = False
            roots = self.output.finish_roots_on_side()
            with torch.cuda.stream(self._side_stream):
                for name, group in self.output.groups.items():
                    inner, outer = roots[name]
                    ic, oc = inner.clone(), outer.clone()
                    ic.record_stream(self._main_stream)
                    oc.record_stream(self._main_stream)
                    for index, key in enumerate(group.own):
                        dense[key] = (ic[index], oc[index])
        if not dense and not self.pending_diagonal:
            return self.lagged
        combined = self.pending_diagonal | dense
        self.pending_diagonal = {}
        self._lagged_prepared = self._prepare_lagged_for_groups_on_side(combined)
        self._ev_lagged_ready.record(self._side_stream)
        return combined

    def zero_grad(self, set_to_none=True):
        for parameter in self.params:
            parameter.grad = None

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def _engine_factors(self, engine, copy=False):
        if not engine.begun:
            engine.begin()
        engine.begun = False
        roots = engine._finish_roots()
        factors = {}
        for name, group in engine.groups.items():
            inner, outer = roots[name]
            for index, key in enumerate(group.own):
                factors[key] = (inner[index].clone(), outer[index].clone()) if copy else (inner[index], outer[index])
        return factors

    def _factors(self, engine, diagonal):
        factors = {} if engine is None else self._engine_factors(engine)
        if diagonal is not None:
            diagonal.update()
            factors |= diagonal.factors()
        return factors

    def check_last_step_finite(self):
        """Verify finite block updates after main_stream synchronization (0 ms host stall when deferred)."""
        if self._last_finite_checks is None:
            return
        checks = self._last_finite_checks.tolist()
        self._last_finite_checks = None
        if not all(checks):
            bad = [shape for (shape, _), ok in zip(self.shape_groups, checks, strict=True) if not ok]
            raise FloatingPointError(f"Nonfinite block update for shapes {bad}")

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        fast_cuda = getattr(self, "_fast_cuda", False)
        input_roots = None
        output_roots = None
        right = {}
        left = {}
        if fast_cuda:
            if self.input is not None:
                if not self.input.begun:
                    self.input.begin()
                self.input.begun = False
                input_roots = self.input._finish_roots()
            if self.input_diagonal is not None:
                if not self._input_diag_begun:
                    self._begin_input_diagonal()
                self._input_diag_begun = False
                self._main_stream.wait_event(self._ev_input_diag_ready)
                right = self._input_diag_factors
                self._input_diag_factors = {}
            if self.arm.lag_left:
                left = self.lagged
            else:
                if self.output_diagonal is not None:
                    self.output_diagonal.update()
                    left = self.output_diagonal.factors()
                if self.output is not None:
                    if not self.output.begun:
                        self.output.begin()
                    self.output.begun = False
                    output_roots = self.output._finish_roots()
        else:
            right = self._factors(self.input, self.input_diagonal)
            left = self.lagged if self.arm.lag_left else self._factors(self.output, self.output_diagonal)

        self.last_metrics = self._engine_metrics()
        if not self.arm.apply_left:
            left = {}
            output_roots = None
        elif not self.arm.apply_left_inner:
            left = {key: (None, outer) for key, (_, outer) in left.items()}

        use_prepped_lag = fast_cuda and self.arm.lag_left
        if use_prepped_lag:
            self._main_stream.wait_event(self._ev_lagged_ready)

        if fast_cuda:
            self._step_scalars_cpu[0] = group["lr"]
            self._step_scalars_cpu[1] = group["weight_decay"]
            lr_t, wd_t = self._step_scalars_cpu.to(self.device, non_blocking=True).unbind(0)

        works, finite = [], []
        for shape, per_rank in self.shape_groups:
            momentum = self.momentum[shape]
            mine = per_rank[self.rank]
            if mine:
                if input_roots is not None and shape in self._input_gather_spec:
                    g_name, g_idx = self._input_gather_spec[shape]
                    ri_grp, ro_grp = input_roots[g_name]
                    if g_idx is None:
                        right_inner, right_outer = ri_grp, ro_grp
                    else:
                        right_inner = ri_grp.index_select(0, g_idx)
                        right_outer = ro_grp.index_select(0, g_idx)
                else:
                    right_pairs = [right.get((layer, INPUT_KIND[kind]), (None, None)) for layer, kind, _ in mine]
                    right_inner = stack_factors([p[0] for p in right_pairs], leading=len(mine), width=shape[1])
                    right_outer = stack_factors([p[1] for p in right_pairs], leading=len(mine), width=shape[1])

                if fast_cuda:
                    ri_prep = _prepare_side_factor(right_inner, side="right", normalize_rms=False, compiled=True)
                    ro_prep = _prepare_side_factor(right_outer, side="right", normalize_rms=True, compiled=True)
                    if use_prepped_lag:
                        li_prep, lo_prep = self._lagged_prepared.get(shape, (None, None))
                    elif output_roots is not None and shape in self._output_gather_spec:
                        g_name, g_idx = self._output_gather_spec[shape]
                        li_grp, lo_grp = output_roots[g_name]
                        if g_idx is None:
                            left_inner = li_grp if self.arm.apply_left_inner else None
                            left_outer = lo_grp
                        else:
                            left_inner = li_grp.index_select(0, g_idx) if self.arm.apply_left_inner else None
                            left_outer = lo_grp.index_select(0, g_idx)
                        li_prep = _prepare_side_factor(left_inner, side="left", normalize_rms=False, compiled=True)
                        lo_prep = _prepare_side_factor(left_outer, side="left", normalize_rms=True, compiled=True)
                    else:
                        left_pairs = [left.get((layer, kind), (None, None)) for layer, kind, _ in mine]
                        left_inner = stack_factors([p[0] for p in left_pairs], leading=len(mine), width=shape[0])
                        left_outer = stack_factors([p[1] for p in left_pairs], leading=len(mine), width=shape[0])
                        li_prep = _prepare_side_factor(left_inner, side="left", normalize_rms=False, compiled=True)
                        lo_prep = _prepare_side_factor(left_outer, side="left", normalize_rms=True, compiled=True)

                    ps = [parameter for _, _, parameter in mine]
                    gs = [parameter.grad for _, _, parameter in mine]
                    r_in_dense = ri_prep is not None and ri_prep.ndim == 3
                    r_out_dense = ro_prep is not None and ro_prep.ndim == 3
                    l_in_dense = li_prep is not None and li_prep.ndim == 3
                    l_out_dense = lo_prep is not None and lo_prep.ndim == 3
                    self.momentum[shape], source = _compiled_block_in(
                        gs,
                        momentum,
                        li_prep,
                        ri_prep,
                        float(self.args.muon_momentum),
                        bool(self.args.muon_nesterov),
                        r_in_dense,
                        l_in_dense,
                    )
                    direction = MU.newton_schulz_batched(
                        source, self.args.muon_ns_steps, self.args.muon_eps, self.args.muon_ns_mode
                    )
                    scale = max(1.0, shape[0] / shape[1]) ** 0.5
                    out_fn = _select_block_out_fn(
                        getattr(self, "decay_mode", DecayMode.DEFAULT), compiled=True
                    )
                    delta = out_fn(
                        ps, direction, lo_prep, ro_prep, lr_t, wd_t, scale, r_out_dense, l_out_dense
                    )
                else:
                    left_pairs = [left.get((layer, kind), (None, None)) for layer, kind, _ in mine]
                    left_inner = stack_factors([p[0] for p in left_pairs], leading=len(mine), width=shape[0])
                    left_outer = stack_factors([p[1] for p in left_pairs], leading=len(mine), width=shape[0])
                    momentum[:], delta = block_update_stack(
                        torch.stack([parameter for _, _, parameter in mine]),
                        torch.stack([parameter.grad.float() for _, _, parameter in mine]),
                        momentum,
                        (left_inner, left_outer),
                        (right_inner, right_outer),
                        args=self.args,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        decay_mode=getattr(self, "decay_mode", DecayMode.DEFAULT),
                    )
            else:
                delta = torch.empty((0, *shape), device=self.device)
            finite.append(torch.isfinite(delta).all())
            counts = [len(entries) for entries in per_rank]
            if self.world > 1:
                works.append((per_rank, *gather_uneven(delta, counts, group=self.pg, async_op=True)))
            else:
                works.append((per_rank, None, [delta]))

        finite_vec = torch.stack(finite)
        defer = bool(getattr(self, "defer_finite_check", False)) and fast_cuda
        if defer:
            self._last_finite_checks = finite_vec
            all_ok = finite_vec.all()
        else:
            checks = finite_vec.tolist()
            if not all(checks):
                bad = [shape for (shape, _), ok in zip(self.shape_groups, checks, strict=True) if not ok]
                raise FloatingPointError(f"Nonfinite block update for shapes {bad}")

        for per_rank, work, gathered in works:
            if work is not None:
                work.wait()
            for entries, updates in zip(per_rank, gathered, strict=True):
                if entries:
                    if defer:
                        updates = torch.where(all_ok, updates, torch.zeros((), dtype=updates.dtype, device=updates.device))
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
        if getattr(self, "_fast_cuda", False):
            self._input_side_stream.synchronize()
            self._side_stream.synchronize()
        if self.arm.lag_left:
            self.lagged = self._collect_lagged()
            self.pending_diagonal = {}
            if getattr(self, "_fast_cuda", False):
                self._side_stream.synchronize()
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
        self.lagged = {
            key: tuple(tensor.to(self.device) for tensor in pair) for key, pair in state.get("lagged", {}).items()
        }
        self.pending_diagonal = {}
        if getattr(self, "_fast_cuda", False) and self.arm.lag_left:
            self._lagged_prepared = self._prepare_lagged_for_groups_on_side(self.lagged)
            self._ev_lagged_ready.record(self._side_stream)
        self.steps = state["steps"]
