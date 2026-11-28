# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input, output, or two-sided Toner factors on every block matrix, including Lazy, HeadBlock, and MetricHyperball."""

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
from .lazy_roots import BatchExactDenseEngine, HeadBlockMoments, LazySchedule
from .toner_state import engine_state, restore_engine

torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)

BLOCK_KINDS = ("q", "k", "v", "o", "gate", "up", "down")
INPUT_KIND = {"q": "attn", "k": "attn", "v": "attn", "o": "o", "gate": "mlp", "up": "mlp", "down": "down"}
INPUT_KINDS = (*KINDS_D, "down")
HIDDEN_OUTPUTS = ("q", "k", "v", "o", "down")
MLP_OUTPUTS = ("gate", "up")
HYPERBALL_NORM_FLOOR = 1e-10
QUINTIC_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

# Side mode integer codes for static torch.compile specialization
SIDE_NONE = 0
SIDE_DIAG = 1
SIDE_DENSE = 2
SIDE_HEAD_BLOCK = 3


def _newton_schulz(u: torch.Tensor, steps: int, eps: float, mode: str = "tpu", coeffs: str = "simple") -> torch.Tensor:
    if coeffs == "simple":
        return MU.newton_schulz(u, steps, eps, mode)
    if coeffs != "quintic":
        raise ValueError(f"Unsupported ns_coeffs: {coeffs!r}")
    x = u.float()
    x = x / (torch.linalg.matrix_norm(x) + eps)
    if mode == "bf16":
        x = x.to(torch.bfloat16)
    tr = x.size(0) > x.size(1)
    if tr:
        x = x.T
    for i in range(steps):
        a, b, c = QUINTIC_NS_COEFFS[i % len(QUINTIC_NS_COEFFS)]
        A = MU._mm(x, x.T, mode)
        x = a * x + MU._mm(b * A + c * MU._mm(A, A, mode), x, mode)
    return (x.T if tr else x).float()


def _newton_schulz_batched_impl(
    U: torch.Tensor, steps: int, eps: float, mode: str = "tpu", coeffs: str = "simple"
) -> torch.Tensor:
    x = U.float()
    x = x / (torch.linalg.matrix_norm(x, dim=(-2, -1), keepdim=True) + eps)
    if mode == "bf16":
        x = x.to(torch.bfloat16)
    tr = x.size(-2) > x.size(-1)
    if tr:
        x = x.transpose(-2, -1)
    if coeffs == "simple":
        a, b, c = MU.ABC
        for _ in range(steps):
            A = MU._bmm(x, x.transpose(-2, -1), mode)
            x = a * x + MU._bmm(b * A + c * MU._bmm(A, A, mode), x, mode)
    elif coeffs == "quintic":
        for i in range(steps):
            a, b, c = QUINTIC_NS_COEFFS[i % len(QUINTIC_NS_COEFFS)]
            A = MU._bmm(x, x.transpose(-2, -1), mode)
            x = a * x + MU._bmm(b * A + c * MU._bmm(A, A, mode), x, mode)
    else:
        raise ValueError(f"Unsupported ns_coeffs: {coeffs!r}")
    return (x.transpose(-2, -1) if tr else x).float()


_compiled_ns_batched = torch.compile(_newton_schulz_batched_impl, dynamic=False)


def _newton_schulz_batched(
    U: torch.Tensor,
    steps: int,
    eps: float,
    mode: str = "tpu",
    coeffs: str = "simple",
    compiled_ns: bool = False,
) -> torch.Tensor:
    if compiled_ns and U.is_cuda:
        return _compiled_ns_batched(U, steps, eps, mode, coeffs)
    if coeffs == "simple":
        return MU.newton_schulz_batched(U, steps, eps, mode)
    return _newton_schulz_batched_impl(U, steps, eps, mode, coeffs)


class Side(StrEnum):
    NONE = "none"
    FULL = "full"
    DIAGONAL = "diagonal"
    HEAD_BLOCK = "head_block"


class DecayMode(StrEnum):
    DEFAULT = "default"
    ISOTROPIC = "isotropic"
    TWO_SIDED = "two_sided"
    HYPERBALL = "hyperball"
    METRIC_HYPERBALL = "metric_hyperball"


def hyperball_delta(weight: torch.Tensor, direction: torch.Tensor, lr: torch.Tensor | float) -> torch.Tensor:
    """Frobenius norm-preserving Hyperball step (Levanter norm_preserving_update / MuonH)."""
    work_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    w = weight.to(work_dtype)
    u = direction.to(work_dtype)
    dims = (-2, -1)
    p_norm = w.square().sum(dim=dims, keepdim=True).sqrt()
    u_norm = u.square().sum(dim=dims, keepdim=True).sqrt().clamp_min(HYPERBALL_NORM_FLOOR)
    p_mid = w - lr * u * (p_norm / u_norm)
    p_mid_norm = p_mid.square().sum(dim=dims, keepdim=True).sqrt().clamp_min(HYPERBALL_NORM_FLOOR)
    return (p_mid * (p_norm / p_mid_norm) - w).to(weight.dtype)


def metric_hyperball_delta(
    weight: torch.Tensor,
    direction: torch.Tensor,
    metric_decay: torch.Tensor | None,
    lr: torch.Tensor | float,
    weight_decay: torch.Tensor | float,
) -> torch.Tensor:
    """Right-metric tangential Hyperball step conserving ||W + delta||_F == ||W||_F."""
    work_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    w = weight.to(work_dtype)
    u = direction.to(work_dtype)
    dims = (-2, -1)
    p_sq = w.square().sum(dim=dims, keepdim=True)
    p_norm = p_sq.sqrt()
    u_norm = u.square().sum(dim=dims, keepdim=True).sqrt().clamp_min(HYPERBALL_NORM_FLOOR)
    step_vec = u * (p_norm / u_norm)
    if metric_decay is not None:
        m = metric_decay.to(work_dtype)
        proj = (w * m).sum(dim=dims, keepdim=True) / p_sq.clamp_min(HYPERBALL_NORM_FLOOR**2)
        step_vec = step_vec + weight_decay * (m - proj * w)
    p_mid = w - lr * step_vec
    p_mid_norm = p_mid.square().sum(dim=dims, keepdim=True).sqrt().clamp_min(HYPERBALL_NORM_FLOOR)
    return (p_mid * (p_norm / p_mid_norm) - w).to(weight.dtype)


@dataclass(frozen=True)
class BlockArm:
    hidden_input: Side
    mlp_input: Side
    hidden_output: Side
    mlp_output: Side
    apply_left: bool
    apply_left_inner: bool = True
    lag_left: bool = False
    down_only_output: bool = False
    qk_output: Side | None = None
    v_output: Side | None = None
    res_output: Side | None = None

    def input_side(self, input_kind):
        return self.mlp_input if input_kind == "down" else self.hidden_input

    def input_kinds(self, side):
        return tuple(kind for kind in INPUT_KINDS if self.input_side(kind) == side)

    def output_side(self, kind):
        if self.down_only_output:
            return self.hidden_output if kind == "down" else Side.NONE
        if kind in ("q", "k"):
            return self.qk_output if self.qk_output is not None else self.hidden_output
        if kind == "v":
            return self.v_output if self.v_output is not None else self.hidden_output
        if kind in ("o", "down"):
            return self.res_output if self.res_output is not None else self.hidden_output
        return self.mlp_output

    def output_kinds(self, side):
        return tuple(kind for kind in BLOCK_KINDS if self.output_side(kind) == side)

    @property
    def has_role_overrides(self) -> bool:
        return self.qk_output is not None or self.v_output is not None or self.res_output is not None


BLOCK_ARMS = {
    "none_all": BlockArm(Side.NONE, Side.NONE, Side.NONE, Side.NONE, apply_left=False),
    "input_only": BlockArm(Side.FULL, Side.FULL, Side.NONE, Side.NONE, apply_left=False),
    "inner_outer_all": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.FULL, apply_left=True),
    "null_all": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.FULL, apply_left=False),
    "left_all": BlockArm(Side.NONE, Side.NONE, Side.FULL, Side.FULL, apply_left=True),
    "inner_outer_all_diag": BlockArm(Side.DIAGONAL, Side.DIAGONAL, Side.DIAGONAL, Side.DIAGONAL, apply_left=True),
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
    "outer_left_down": BlockArm(
        Side.FULL, Side.FULL, Side.FULL, Side.NONE, apply_left=True, apply_left_inner=False, down_only_output=True
    ),
    "down_hybrid": BlockArm(
        Side.FULL, Side.NONE, Side.FULL, Side.NONE, apply_left=True, down_only_output=True
    ),
    "inner_outer_all_lag": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.FULL, apply_left=True, lag_left=True),
    "inner_outer_mlpdiag_lag": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.DIAGONAL, apply_left=True, lag_left=True),
    "null_mlpdiag": BlockArm(Side.FULL, Side.FULL, Side.FULL, Side.DIAGONAL, apply_left=False),
    "inner_outer_sym_mlpdiag": BlockArm(Side.FULL, Side.DIAGONAL, Side.FULL, Side.DIAGONAL, apply_left=True),
    "inner_outer_sym_mlpdiag_lag": BlockArm(
        Side.FULL, Side.DIAGONAL, Side.FULL, Side.DIAGONAL, apply_left=True, lag_left=True
    ),
    # Structure-aware output role placements:
    # 1. Residual-Dense, Internal-Diagonal: dense (o, down), diagonal (q, k, v, gate, up)
    "resdense_alldiag_lag": BlockArm(
        Side.FULL,
        Side.FULL,
        Side.FULL,
        Side.DIAGONAL,
        apply_left=True,
        lag_left=True,
        qk_output=Side.DIAGONAL,
        v_output=Side.DIAGONAL,
        res_output=Side.FULL,
    ),
    # 2. Head-Block-Diagonal on (q, k) [99.8%/98.7% fidelity], Dense on (v, o, down), Diagonal on (gate, up)
    "headdiag_qk_mlpdiag_lag": BlockArm(
        Side.FULL,
        Side.FULL,
        Side.FULL,
        Side.DIAGONAL,
        apply_left=True,
        lag_left=True,
        qk_output=Side.HEAD_BLOCK,
        v_output=Side.FULL,
        res_output=Side.FULL,
    ),
    # 3. Head-Block-Diagonal on (q, k, v), Dense on (o, down), Diagonal on (gate, up)
    "headdiag_qkv_mlpdiag_lag": BlockArm(
        Side.FULL,
        Side.FULL,
        Side.FULL,
        Side.DIAGONAL,
        apply_left=True,
        lag_left=True,
        qk_output=Side.HEAD_BLOCK,
        v_output=Side.HEAD_BLOCK,
        res_output=Side.FULL,
    ),
    # 4. Head-Block-Diagonal on (q, k), Diagonal on (v, gate, up), Dense on (o, down)
    "headdiag_qk_vdiag_mlpdiag_lag": BlockArm(
        Side.FULL,
        Side.FULL,
        Side.FULL,
        Side.DIAGONAL,
        apply_left=True,
        lag_left=True,
        qk_output=Side.HEAD_BLOCK,
        v_output=Side.DIAGONAL,
        res_output=Side.FULL,
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
            rows[layer, kind] = torch.zeros(capacity, linear.weight.shape[0], dtype=torch.bfloat16, device=device)
            setattr(block, kind, RowObservedLinear(linear, rows[layer, kind], at, index))
    return rows, at


class OutputToner(FastTonerCore):
    """Toner's pooled statistics, cached roots, and host refresh over output-cotangent rows."""

    def __init__(self, rows, kinds, *, layers, hidden, mlp, rows_total, args, world, rank, device, lagged=False):
        placeholder = torch.empty(0, 1, dtype=torch.bfloat16, device=device)
        blocks = [SimpleNamespace(**{f"input_rows_{kind}": placeholder for kind in INPUT_KINDS})] * layers
        super().__init__(
            blocks, {}, "muont", hid=1, mlp=1, rows_total=rows_total, args=args, world=world, rank=rank, device=device
        )
        if lagged:
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
    if factor.ndim == 1:
        return factor[:, None] * matrix
    if factor.ndim == 3:
        H, hd, _ = factor.shape
        return torch.bmm(factor, matrix.view(H, hd, -1)).reshape_as(matrix)
    return factor @ matrix


def right_product(matrix, factor):
    if factor.ndim == 1:
        return matrix * factor
    if factor.ndim == 3:
        H, hd, _ = factor.shape
        m = matrix.shape[0]
        return torch.bmm(matrix.view(m, H, hd).permute(1, 0, 2), factor).permute(1, 0, 2).reshape_as(matrix)
    return matrix @ factor


def unit_rms(factor):
    """Scale a dense, diagonal, or head-block factor to unit RMS, ||F||_F^2 / d = 1."""
    if factor.ndim == 3:
        d = factor.shape[0] * factor.shape[1]
        return factor * (d**0.5 / torch.linalg.vector_norm(factor).clamp_min(1e-30))
    return factor * (factor.shape[-1] ** 0.5 / torch.linalg.vector_norm(factor).clamp_min(1e-30))


@dataclass(frozen=True)
class MatrixFactors:
    """Root factors of one matrix; None is identity. Diagonal=1D, Dense=2D, HeadBlock=3D."""

    left_inner: torch.Tensor | None = None
    left_outer: torch.Tensor | None = None
    right_inner: torch.Tensor | None = None
    right_outer: torch.Tensor | None = None


def block_update(weight, gradient, momentum, factors, *, args, lr, weight_decay, decay_mode: str = DecayMode.DEFAULT):
    """Return updated momentum and the finite-NS parameter change of one block matrix."""
    mode = DecayMode(decay_mode)
    momentum = args.muon_momentum * momentum + gradient
    source = gradient + args.muon_momentum * momentum if args.muon_nesterov else momentum
    if factors.right_inner is not None:
        source = right_product(source, factors.right_inner)
    if factors.left_inner is not None:
        source = left_product(factors.left_inner, source)
    direction = _newton_schulz(
        source, args.muon_ns_steps, args.muon_eps, args.muon_ns_mode, getattr(args, "muon_ns_coeffs", "simple")
    )
    decay = weight
    metric_decay = None
    if factors.right_outer is not None:
        right = unit_rms(factors.right_outer)
        direction = right_product(direction, right)
        if mode in (DecayMode.DEFAULT, DecayMode.TWO_SIDED):
            decay = right_product(right_product(decay, right), right)
        elif mode == DecayMode.METRIC_HYPERBALL:
            metric_decay = right_product(right_product(weight, right), right)
    if factors.left_outer is not None:
        left = unit_rms(factors.left_outer)
        direction = left_product(left, direction)
        if mode == DecayMode.TWO_SIDED or (mode == DecayMode.DEFAULT and factors.right_outer is None):
            decay = left_product(left, left_product(left, decay))
    if mode == DecayMode.HYPERBALL:
        return momentum, hyperball_delta(weight, direction, lr)
    if mode == DecayMode.METRIC_HYPERBALL:
        return momentum, metric_hyperball_delta(weight, direction, metric_decay, lr, weight_decay)
    scale = max(1.0, weight.shape[-2] / weight.shape[-1]) ** 0.5
    return momentum, -lr * (scale * direction + weight_decay * decay)


def stack_factors(factors, *, leading, width):
    """Stack per-matrix factors of one side into a batched tensor, or None when every factor is absent."""
    if all(factor is None for factor in factors):
        return None
    if any(factor is None for factor in factors):
        raise ValueError("A stacked side must have a factor for every matrix or none")
    nd = factors[0].ndim
    if any(factor.ndim != nd for factor in factors):
        raise ValueError("A stacked side must not mix factor ranks")
    if nd in (1, 2):
        if any(factor.shape[-1] != width for factor in factors):
            raise ValueError("A stacked side must match the matrix width")
    elif nd == 3:
        if any(factor.shape[0] * factor.shape[1] != width for factor in factors):
            raise ValueError("A stacked head-block side must match the matrix width")
    else:
        raise ValueError(f"Unsupported per-matrix factor rank {nd}")
    return torch.stack(factors)


def stack_unit_rms(factor):
    """Per-matrix unit-RMS scaling of a stack, ||F||_F^2 / d = 1 for each entry."""
    if factor.ndim == 4:
        d = factor.shape[1] * factor.shape[2]
        return factor * (d**0.5 / torch.linalg.vector_norm(factor, dim=(-3, -2, -1), keepdim=True).clamp_min(1e-30))
    dims = (-2, -1) if factor.ndim == 3 else (-1,)
    return factor * (factor.shape[-1] ** 0.5 / torch.linalg.vector_norm(factor, dim=dims, keepdim=True).clamp_min(1e-30))


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


def _apply16_left_headblock_raw(L2: torch.Tensor, A: torch.Tensor):
    """Apply Ozaki-split head-block L2 [B, H, hd, 2*hd] on the left of A [B, H*hd, N]."""
    B, H, hd, two_hd = L2.shape
    N = A.size(-1)
    A_bh = A.reshape(B * H, hd, N)
    A2, inv_A = _rows_lo_scaled_raw(A_bh)
    L_bh = L2.reshape(B * H, hd, two_hd)
    C = torch.bmm(L_bh[..., :hd], A2[..., hd:, :], out_dtype=torch.float32)
    out = (C + torch.bmm(L_bh, A2, out_dtype=torch.float32) * (1.0 / _LO)) * inv_A
    return out.reshape(B, H * hd, N)


def _apply_left_mode_raw(left_factor: torch.Tensor | None, X: torch.Tensor, left_mode: int) -> torch.Tensor:
    if left_mode == SIDE_NONE or left_factor is None:
        return X
    if left_mode == SIDE_DENSE:
        return _apply16_left_raw(left_factor, X)
    if left_mode == SIDE_HEAD_BLOCK:
        return _apply16_left_headblock_raw(left_factor, X)
    return X * left_factor[:, :, None]


def _apply_right_mode_raw(X: torch.Tensor, right_factor: torch.Tensor | None, right_mode: int) -> torch.Tensor:
    if right_mode == SIDE_NONE or right_factor is None:
        return X
    if right_mode == SIDE_DENSE:
        return _apply16_right_raw(X, right_factor)
    return X * right_factor[:, None, :]


def _prepare_right_dense_raw(factor: torch.Tensor, normalize_rms: bool):
    if normalize_rms:
        factor = stack_unit_rms(factor)
    return PR._rows_lo(factor)


def _prepare_left_dense_raw(factor: torch.Tensor, normalize_rms: bool):
    if normalize_rms:
        factor = stack_unit_rms(factor)
    return PR._cols_hi(factor)


def _prepare_left_headblock_raw(factor: torch.Tensor, normalize_rms: bool):
    if normalize_rms:
        factor = stack_unit_rms(factor)
    return PR._cols_hi(factor)


def _block_in_raw(Gs, momenta, left_in, right_in, mu: float, nesterov: bool, right_mode: int, left_mode: int):
    G = Gs.float() if isinstance(Gs, torch.Tensor) else torch.stack(Gs).float()
    momenta = mu * momenta + G
    source = G + mu * momenta if nesterov else momenta
    if right_mode != SIDE_NONE and right_in is not None:
        source = _apply_right_mode_raw(source, right_in, right_mode)
    if left_mode != SIDE_NONE and left_in is not None:
        source = _apply_left_mode_raw(left_in, source, left_mode)
    return momenta, source


def _block_out_raw(Ps, direction, left_out, right_out, lr_t, wd_t, scale: float, right_mode: int, left_mode: int):
    P = Ps.float() if isinstance(Ps, torch.Tensor) else torch.stack(Ps).float()
    if right_mode != SIDE_NONE and right_out is not None:
        dir_l = _apply_left_mode_raw(left_out, direction, left_mode)
        if right_mode == SIDE_DENSE:
            WR = _apply16_right_raw(P, right_out)
            V = (lr_t * wd_t) * WR + (lr_t * scale) * dir_l
            return -_apply16_right_raw(V, right_out)
        r = right_out[:, None, :]
        return -lr_t * (scale * (dir_l * r) + wd_t * (P * (r * r)))

    if left_mode == SIDE_NONE or left_out is None:
        return -lr_t * (scale * direction + wd_t * P)
    if left_mode == SIDE_DENSE:
        LW = _apply16_left_raw(left_out, P)
        V = (lr_t * wd_t) * LW + (lr_t * scale) * direction
        return -_apply16_left_raw(left_out, V)
    if left_mode == SIDE_HEAD_BLOCK:
        LW = _apply16_left_headblock_raw(left_out, P)
        V = (lr_t * wd_t) * LW + (lr_t * scale) * direction
        return -_apply16_left_headblock_raw(left_out, V)
    l = left_out[:, :, None]
    return -lr_t * (scale * (l * direction) + wd_t * (l * l * P))


def _block_out_isotropic_raw(
    Ps, direction, left_out, right_out, lr_t, wd_t, scale: float, right_mode: int, left_mode: int
):
    P = Ps.float() if isinstance(Ps, torch.Tensor) else torch.stack(Ps).float()
    dir_l = _apply_left_mode_raw(left_out, direction, left_mode)
    dir_lr = _apply_right_mode_raw(dir_l, right_out, right_mode)
    return -lr_t * (scale * dir_lr + wd_t * P)


def _block_out_two_sided_raw(
    Ps, direction, left_out, right_out, lr_t, wd_t, scale: float, right_mode: int, left_mode: int
):
    P = Ps.float() if isinstance(Ps, torch.Tensor) else torch.stack(Ps).float()
    WL = _apply_left_mode_raw(left_out, P, left_mode)
    WLR = _apply_right_mode_raw(WL, right_out, right_mode)
    V = (lr_t * wd_t) * WLR + (lr_t * scale) * direction
    VR = _apply_right_mode_raw(V, right_out, right_mode)
    return -_apply_left_mode_raw(left_out, VR, left_mode)


def _block_out_hyperball_raw(
    Ps, direction, left_out, right_out, lr_t, wd_t, scale: float, right_mode: int, left_mode: int
):
    del wd_t, scale
    P = Ps.float() if isinstance(Ps, torch.Tensor) else torch.stack(Ps).float()
    dir_l = _apply_left_mode_raw(left_out, direction, left_mode)
    dir_lr = _apply_right_mode_raw(dir_l, right_out, right_mode)
    dims = (-2, -1)
    p_norm = P.square().sum(dim=dims, keepdim=True).sqrt()
    u_norm = dir_lr.square().sum(dim=dims, keepdim=True).sqrt().clamp_min(HYPERBALL_NORM_FLOOR)
    p_mid = P - lr_t * dir_lr * (p_norm / u_norm)
    p_mid_norm = p_mid.square().sum(dim=dims, keepdim=True).sqrt().clamp_min(HYPERBALL_NORM_FLOOR)
    return p_mid * (p_norm / p_mid_norm) - P


def _block_out_metric_hyperball_raw(
    Ps, direction, left_out, right_out, lr_t, wd_t, scale: float, right_mode: int, left_mode: int
):
    del scale
    P = Ps.float() if isinstance(Ps, torch.Tensor) else torch.stack(Ps).float()
    dir_l = _apply_left_mode_raw(left_out, direction, left_mode)
    dir_lr = _apply_right_mode_raw(dir_l, right_out, right_mode)
    dims = (-2, -1)
    p_sq = P.square().sum(dim=dims, keepdim=True)
    p_norm = p_sq.sqrt()
    u_norm = dir_lr.square().sum(dim=dims, keepdim=True).sqrt().clamp_min(HYPERBALL_NORM_FLOOR)
    step_vec = dir_lr * (p_norm / u_norm)
    if right_mode != SIDE_NONE and right_out is not None:
        wr = _apply_right_mode_raw(P, right_out, right_mode)
        wr2 = _apply_right_mode_raw(wr, right_out, right_mode)
        proj = (P * wr2).sum(dim=dims, keepdim=True) / p_sq.clamp_min(HYPERBALL_NORM_FLOOR**2)
        step_vec = step_vec + wd_t * (wr2 - proj * P)
    p_mid = P - lr_t * step_vec
    p_mid_norm = p_mid.square().sum(dim=dims, keepdim=True).sqrt().clamp_min(HYPERBALL_NORM_FLOOR)
    return p_mid * (p_norm / p_mid_norm) - P


_prepare_right_dense = torch.compile(_prepare_right_dense_raw, dynamic=False, options=PR._EXACT)
_prepare_left_dense = torch.compile(_prepare_left_dense_raw, dynamic=False, options=PR._EXACT)
_prepare_left_headblock = torch.compile(_prepare_left_headblock_raw, dynamic=False, options=PR._EXACT)
_compiled_block_in = torch.compile(_block_in_raw, dynamic=False, options=PR._EXACT)
_compiled_block_out = torch.compile(_block_out_raw, dynamic=False, options=PR._EXACT)
_compiled_block_out_isotropic = torch.compile(_block_out_isotropic_raw, dynamic=False, options=PR._EXACT)
_compiled_block_out_two_sided = torch.compile(_block_out_two_sided_raw, dynamic=False, options=PR._EXACT)
_compiled_block_out_hyperball = torch.compile(_block_out_hyperball_raw, dynamic=False, options=PR._EXACT)
_compiled_block_out_metric_hyperball = torch.compile(_block_out_metric_hyperball_raw, dynamic=False, options=PR._EXACT)


def _select_block_out_fn(mode: DecayMode, *, compiled: bool):
    if mode == DecayMode.DEFAULT:
        return _compiled_block_out if compiled else _block_out_raw
    if mode == DecayMode.ISOTROPIC:
        return _compiled_block_out_isotropic if compiled else _block_out_isotropic_raw
    if mode == DecayMode.TWO_SIDED:
        return _compiled_block_out_two_sided if compiled else _block_out_two_sided_raw
    if mode == DecayMode.HYPERBALL:
        return _compiled_block_out_hyperball if compiled else _block_out_hyperball_raw
    if mode == DecayMode.METRIC_HYPERBALL:
        return _compiled_block_out_metric_hyperball if compiled else _block_out_metric_hyperball_raw
    raise ValueError(f"Unsupported decay_mode: {mode}")


def _factor_mode_code(factor: torch.Tensor | None) -> int:
    if factor is None:
        return SIDE_NONE
    if factor.ndim == 2:
        return SIDE_DIAG
    if factor.ndim == 3:
        return SIDE_DENSE
    if factor.ndim == 4:
        return SIDE_HEAD_BLOCK
    raise ValueError(f"Unexpected stacked factor rank {factor.ndim}")


def _prepare_side_factor(factor: torch.Tensor | None, *, side: str, normalize_rms: bool, compiled: bool):
    if factor is None:
        return None
    if factor.ndim == 2:
        return stack_unit_rms(factor) if normalize_rms else factor
    if factor.ndim == 4:
        if side != "left":
            raise ValueError("Head-block factors are supported on the left (output) side")
        return (_prepare_left_headblock if compiled else _prepare_left_headblock_raw)(factor, normalize_rms)
    if side == "right":
        return (_prepare_right_dense if compiled else _prepare_right_dense_raw)(factor, normalize_rms)
    return (_prepare_left_dense if compiled else _prepare_left_dense_raw)(factor, normalize_rms)


def _stack_apply_left_cpu(factor: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    if factor.ndim == 2:
        return X * factor[:, :, None]
    if factor.ndim == 3:
        return torch.bmm(factor, X)
    if factor.ndim == 4:
        B, H, hd, _ = factor.shape
        N = X.size(-1)
        return torch.bmm(factor.reshape(B * H, hd, hd), X.reshape(B * H, hd, N)).reshape(B, H * hd, N)
    raise ValueError(f"Unsupported factor rank {factor.ndim}")


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
    """The batched form of block_update over a stack of same-shape matrices."""
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
            _factor_mode_code(ri),
            _factor_mode_code(li),
        )
        direction = _newton_schulz_batched(
            source,
            args.muon_ns_steps,
            args.muon_eps,
            args.muon_ns_mode,
            getattr(args, "muon_ns_coeffs", "simple"),
            bool(getattr(args, "muon_compile_ns", False)),
        )
        scale = max(1.0, weights.shape[-2] / weights.shape[-1]) ** 0.5
        delta = out_fn(
            weights,
            direction,
            lo,
            ro,
            lr_t,
            wd_t,
            scale,
            _factor_mode_code(ro),
            _factor_mode_code(lo),
        )
        return momenta, delta

    momenta = args.muon_momentum * momenta + gradients
    source = gradients + args.muon_momentum * momenta if args.muon_nesterov else momenta
    if right_inner is not None:
        source = torch.bmm(source, right_inner) if right_inner.ndim == 3 else source * right_inner[:, None, :]
    if left_inner is not None:
        source = _stack_apply_left_cpu(left_inner, source)
    direction = _newton_schulz_batched(
        source, args.muon_ns_steps, args.muon_eps, args.muon_ns_mode, getattr(args, "muon_ns_coeffs", "simple")
    )
    decay = weights
    metric_decay = None
    if right_outer is not None:
        right_norm = stack_unit_rms(right_outer)
        direction = torch.bmm(direction, right_norm) if right_norm.ndim == 3 else direction * right_norm[:, None, :]
        if mode in (DecayMode.DEFAULT, DecayMode.TWO_SIDED):
            decay = torch.bmm(decay, right_norm) if right_norm.ndim == 3 else decay * right_norm[:, None, :]
            decay = torch.bmm(decay, right_norm) if right_norm.ndim == 3 else decay * right_norm[:, None, :]
        elif mode == DecayMode.METRIC_HYPERBALL:
            metric_decay = torch.bmm(weights, right_norm) if right_norm.ndim == 3 else weights * right_norm[:, None, :]
            metric_decay = (
                torch.bmm(metric_decay, right_norm) if right_norm.ndim == 3 else metric_decay * right_norm[:, None, :]
            )
    if left_outer is not None:
        left_norm = stack_unit_rms(left_outer)
        direction = _stack_apply_left_cpu(left_norm, direction)
        if mode == DecayMode.TWO_SIDED or (mode == DecayMode.DEFAULT and right_outer is None):
            decay = _stack_apply_left_cpu(left_norm, _stack_apply_left_cpu(left_norm, decay))
    if mode == DecayMode.HYPERBALL:
        return momenta, hyperball_delta(weights, direction, lr)
    if mode == DecayMode.METRIC_HYPERBALL:
        return momenta, metric_hyperball_delta(weights, direction, metric_decay, lr, weight_decay)
    scale = max(1.0, weights.shape[-2] / weights.shape[-1]) ** 0.5
    return momenta, -lr * (scale * direction + weight_decay * decay)


def _check_output_settings(config):
    """Output factors default to Toner's input settings unless allow_custom_output=True."""
    if config.get("experiment", {}).get("allow_custom_output", False):
        return
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
    """MuonT-family updates of all 224 block matrices with optional lazy schedules and role-aware factors."""

    def __init__(self, model, config, positions, *, rows_per_microbatch, capacity, rows_total, world, rank, device):
        _check_output_settings(config)
        self.arm = BLOCK_ARMS[config["experiment"]["arm"]]
        self.decay_mode = DecayMode(config["experiment"].get("decay_mode", DecayMode.DEFAULT))
        self.lazy = LazySchedule.from_config(config)
        self.args = fast_arguments(config)
        self.args.muon_ns_coeffs = config["optimizer"].get("ns_coeffs", "simple")
        self.args.muon_compile_ns = bool(config["optimizer"].get("compile_ns", False))
        self.world, self.rank, self.device = world, rank, device
        self._fast_cuda = torch.device(device).type == "cuda"
        blocks = list(model.body.blocks)
        layers = len(blocks)
        hidden, mlp = config["model"]["hidden"], config["model"]["mlp"]
        heads = config["model"]["heads"]
        statistics = config["statistics"]
        output_cfg = config["experiment"]["output"]
        observed = tuple(kind for kind in BLOCK_KINDS if self.arm.output_side(kind) != Side.NONE)
        self.observes_output = bool(observed)

        if not observed:
            self.output_rows = {}
            self.output_at = None
            self._row_off = None
            self._inline_diag = None
        else:
            kind_modes = (
                {k: str(self.arm.output_side(k)) for k in BLOCK_KINDS} if self.arm.has_role_overrides else None
            )
            fast_attached = attach_fast_block_observers(
                blocks,
                hidden_mode=str(self.arm.hidden_output),
                mlp_mode=str(self.arm.mlp_output),
                positions=positions,
                capacity=capacity,
                device=device,
                down_only=self.arm.down_only_output,
                kind_modes=kind_modes,
            )
            if fast_attached is not None:
                self.output_rows, self._row_off, self._idx, self._inline_diag = fast_attached
                self.output_at = None
            else:
                self.output_rows, self.output_at = attach_output_rows(
                    blocks,
                    observed,
                    positions,
                    rows_per_microbatch=rows_per_microbatch,
                    capacity=capacity,
                    device=device,
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

        self.layer_counts = layer_counts(layers, world)
        in_decay = self.lazy.input_ema_decay if self.lazy.enabled else statistics["momentum"]
        out_decay = self.lazy.output_ema_decay if self.lazy.enabled else output_cfg["momentum"]
        use_batch_in_engine = self.lazy.enabled and self.lazy.root_backend in ("gpu_eigh", "async_host_both")
        use_batch_out_engine = self.lazy.enabled and self.lazy.root_backend in ("gpu_eigh", "async_host", "async_host_both")
        async_in_host = self.lazy.enabled and self.lazy.root_backend == "async_host_both" and self._fast_cuda
        async_out_host = self.lazy.enabled and self.lazy.root_backend in ("async_host", "async_host_both") and self._fast_cuda

        if self.lazy.enabled and self.lazy.root_backend == "async_host" and self._fast_cuda:
            in_args = fast_arguments(config)
            in_args.toner_decay = in_decay
            in_args.toner_inline = self.lazy.input_inline_steps
            in_args.toner_refresh = self.lazy.input_refresh_every
        else:
            in_args = self.args

        engine = {"rows_total": rows_total, "args": in_args, "world": world, "rank": rank, "device": device}
        out_engine_kw = {"rows_total": rows_total, "args": self.args, "world": world, "rank": rank, "device": device}
        input_full_kinds = self.arm.input_kinds(Side.FULL)
        input_diag_kinds = self.arm.input_kinds(Side.DIAGONAL)

        if use_batch_in_engine:
            in_specs = []
            if any(k in KINDS_D for k in input_full_kinds):
                d_members = tuple(k for k in KINDS_D if k in input_full_kinds)
                in_specs.append(
                    (
                        "d",
                        [(i, k) for i in range(layers) for k in d_members],
                        hidden,
                        [getattr(blocks[i], f"input_rows_{k}") for i in range(layers) for k in d_members],
                        [len(d_members) * c for c in self.layer_counts],
                    )
                )
            if "down" in input_full_kinds:
                in_specs.append(
                    (
                        "f",
                        [(i, "down") for i in range(layers)],
                        mlp,
                        [blocks[i].input_rows_down for i in range(layers)],
                        list(self.layer_counts),
                    )
                )
            self.input = (
                BatchExactDenseEngine(
                    in_specs,
                    decay=in_decay,
                    damping=statistics["damping"],
                    inner_power=statistics["inner_strength"],
                    outer_power=statistics["outer_strength"],
                    rows_total=rows_total,
                    world=world,
                    rank=rank,
                    device=device,
                    stream_priority=-1,
                    async_host=async_in_host,
                )
                if in_specs
                else None
            )
        elif input_full_kinds == KINDS_D:
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

        full = self.arm.output_kinds(Side.FULL)
        diagonal_kinds = self.arm.output_kinds(Side.DIAGONAL)
        headblock_kinds = self.arm.output_kinds(Side.HEAD_BLOCK)

        if full and use_batch_out_engine:
            out_specs = []
            for g_name, members, width in (
                ("out_hidden", tuple(k for k in full if k in HIDDEN_OUTPUTS), hidden),
                ("out_mlp", tuple(k for k in full if k in MLP_OUTPUTS), mlp),
            ):
                if members:
                    out_specs.append(
                        (
                            g_name,
                            [(layer, kind) for layer in range(layers) for kind in members],
                            width,
                            [self.output_rows[layer, kind] for layer in range(layers) for kind in members],
                            [len(members) * c for c in self.layer_counts],
                        )
                    )
            self.output = BatchExactDenseEngine(
                out_specs,
                decay=out_decay,
                damping=output_cfg["damping"],
                inner_power=output_cfg["inner_strength"],
                outer_power=output_cfg["outer_strength"],
                rows_total=rows_total,
                world=world,
                rank=rank,
                device=device,
                stream_priority=0,
                async_host=async_out_host,
            )
        elif full:
            self.output = OutputToner(
                self.output_rows,
                full,
                layers=layers,
                hidden=hidden,
                mlp=mlp,
                lagged=self.arm.lag_left,
                **out_engine_kw,
            )
        else:
            self.output = None

        self.output_headblock = (
            HeadBlockMoments(
                self.output_rows,
                headblock_kinds,
                layers=layers,
                hidden=hidden,
                heads=heads,
                decay=out_decay,
                damping=output_cfg["damping"],
                inner_power=output_cfg["inner_strength"],
                outer_power=output_cfg["outer_strength"],
                rows_total=rows_total,
                world=world,
                rank=rank,
                device=device,
                async_host=async_out_host,
            )
            if headblock_kinds
            else None
        )

        in_diag_keys_by_width = {}
        for layer_idx in range(layers):
            for kind in input_diag_kinds:
                buf = getattr(blocks[layer_idx], f"input_rows_{kind}")
                in_diag_keys_by_width.setdefault(buf.shape[-1], []).append((layer_idx, kind))
        self.input_diagonal = (
            DiagonalMoments(
                {(i, k): getattr(blocks[i], f"input_rows_{k}") for i in range(layers) for k in input_diag_kinds},
                keys_by_width=in_diag_keys_by_width,
                decay=in_decay,
                damping=statistics["damping"],
                inner_power=statistics["inner_strength"],
                outer_power=statistics["outer_strength"],
                rows_total=rows_total,
                world=world,
                device=device,
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
                decay=out_decay,
                damping=output_cfg["damping"],
                inner_power=output_cfg["inner_strength"],
                outer_power=output_cfg["outer_strength"],
                rows_total=rows_total,
                world=world,
                device=device,
            )
            if diagonal_kinds
            else None
        )

        owner = [r for r, count in enumerate(self.layer_counts) for _ in range(count)]
        entries = [(layer, kind, getattr(blocks[layer], kind).weight) for layer in range(layers) for kind in BLOCK_KINDS]
        self.params = [parameter for _, _, parameter in entries]

        raw_by_shape: dict[tuple[int, int], list] = {}
        for entry in entries:
            raw_by_shape.setdefault(tuple(entry[2].shape), []).append(entry)

        # Keep group_key == shape when homogeneous (preserving exact legacy checkpoint keys),
        # and split into homogeneous (shape, out_side, in_side) groups when roles within a shape differ.
        self.shape_groups = []
        self.group_shapes = {}
        for shape, group in raw_by_shape.items():
            modes = {(self.arm.output_side(k), self.arm.input_side(INPUT_KIND[k])) for _, k, _ in group}
            if len(modes) == 1:
                self.shape_groups.append((shape, [[e for e in group if owner[e[0]] == r] for r in range(world)]))
                self.group_shapes[shape] = shape
            else:
                sub_groups: dict[tuple, list] = {}
                for e in group:
                    m_key = (shape, str(self.arm.output_side(e[1])), str(self.arm.input_side(INPUT_KIND[e[1]])))
                    sub_groups.setdefault(m_key, []).append(e)
                for g_key, sub in sub_groups.items():
                    self.shape_groups.append((g_key, [[e for e in sub if owner[e[0]] == r] for r in range(world)]))
                    self.group_shapes[g_key] = shape

        self.momentum = {
            g_key: torch.zeros((len(per_rank[rank]), *self.group_shapes[g_key]), device=device)
            for g_key, per_rank in self.shape_groups
        }
        self.param_groups = [{"params": self.params, "lr": 0.0, "weight_decay": self.args.wd}]
        self.steps = 0
        self.lagged = {}
        self.pending_diagonal = {}
        self.pending_headblock = {}
        self._headblock_in_flight = False
        self._lagged_prepared = {}
        self._cached_right = {}
        self._cached_right_prep = {}
        self._input_diag_begun = False
        self._input_diag_factors = {}
        self._do_in_this_step = True
        self._do_out_this_step = True
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
            self._ev_in_rows_consumed = torch.cuda.Event()
            self._ev_input_lagged_ready = torch.cuda.Event()
            self._in_rows_guard_active = False
            self._ev_rows_consumed = torch.cuda.Event()
            self._ev_lagged_ready = torch.cuda.Event()
            self._rows_guard_active = False
            self._input_gather_spec = {}
            if self.input is not None:
                own_maps = {name: {key: i for i, key in enumerate(g.own)} for name, g in self.input.groups.items()}
                for g_key, per_rank in self.shape_groups:
                    mine = per_rank[self.rank]
                    if not mine:
                        continue
                    keys = [(layer, INPUT_KIND[kind]) for layer, kind, _ in mine]
                    for g_name, g_map in own_maps.items():
                        if keys[0] in g_map:
                            if keys == list(self.input.groups[g_name].own):
                                self._input_gather_spec[g_key] = (g_name, None)
                            else:
                                idx = torch.tensor([g_map[k] for k in keys], dtype=torch.int64, device=device)
                                self._input_gather_spec[g_key] = (g_name, idx)
                            break
            self._output_gather_spec = {}
            if self.output is not None:
                out_maps = {name: {key: i for i, key in enumerate(g.own)} for name, g in self.output.groups.items()}
                for g_key, per_rank in self.shape_groups:
                    mine = per_rank[self.rank]
                    if not mine:
                        continue
                    keys = [(layer, kind) for layer, kind, _ in mine]
                    for g_name, g_map in out_maps.items():
                        if keys[0] in g_map:
                            if keys == list(self.output.groups[g_name].own):
                                self._output_gather_spec[g_key] = (g_name, None)
                            else:
                                idx = torch.tensor([g_map[k] for k in keys], dtype=torch.int64, device=device)
                                self._output_gather_spec[g_key] = (g_name, idx)
                            break

    @property
    def engines(self):
        return [engine for engine in (self.input, self.output) if engine is not None]

    def set_step_capture(self, *, capture_in: bool, capture_out: bool) -> None:
        self._do_in_this_step = bool(capture_in)
        self._do_out_this_step = bool(capture_out)

    def _output_ready_for_collect(self) -> bool:
        if self.output is not None and self.output.begun:
            ready = self.output.hosts_ready() if isinstance(self.output, BatchExactDenseEngine) else self.output.ev_flags.query()
            if not ready:
                return False
        if self._headblock_in_flight and self.output_headblock is not None and not self.output_headblock.hosts_ready():
            return False
        return True

    def _input_ready_for_collect(self) -> bool:
        if self.input is not None and self.input.begun:
            if isinstance(self.input, BatchExactDenseEngine) and not self.input.hosts_ready():
                return False
        if self._input_diag_begun and self._fast_cuda and not self._ev_input_diag_ready.query():
            return False
        return True

    def _has_in_flight_input(self) -> bool:
        return (self.input is not None and self.input.begun) or self._input_diag_begun

    def _collect_in_flight_input(self) -> None:
        if not getattr(self, "_fast_cuda", False):
            right = {}
            if self.input is not None and self.input.begun:
                right |= self._engine_factors(self.input, copy=True)
            if self._input_diag_begun:
                self._input_diag_begun = False
                right |= {k: (a.clone(), b.clone()) for k, (a, b) in self._input_diag_factors.items()}
                self._input_diag_factors = {}
            elif self.input_diagonal is not None and self._do_in_this_step and not self._cached_right:
                self.input_diagonal.update()
                right |= {k: (a.clone(), b.clone()) for k, (a, b) in self.input_diagonal.factors().items()}
            if right:
                self._cached_right = right
                self._rebuild_cached_right_prep(fast_cuda=False)
            return

        new_right = {}
        if self._input_diag_begun:
            self._input_diag_begun = False
            self._input_side_stream.wait_event(self._ev_input_diag_ready)
            new_right.update(self._input_diag_factors)
            self._input_diag_factors = {}
        if self.input is not None and self.input.begun:
            if isinstance(self.input, BatchExactDenseEngine):
                roots = self.input.finish_roots_on_side(self._input_side_stream)
            else:
                roots = self._finish_engine(self.input)
            with torch.cuda.stream(self._input_side_stream):
                for name, grp in self.input.groups.items():
                    ri_g, ro_g = roots[name]
                    ic, oc = ri_g.clone(), ro_g.clone()
                    ic.record_stream(self._main_stream)
                    oc.record_stream(self._main_stream)
                    for idx_k, key in enumerate(grp.own):
                        new_right[key] = (ic[idx_k], oc[idx_k])
        if not new_right:
            return
        self._cached_right = new_right
        with torch.cuda.stream(self._input_side_stream):
            self._rebuild_cached_right_prep(fast_cuda=True, record_main=True)
            self._ev_input_lagged_ready.record(self._input_side_stream)

    def select_microbatch(self, index, *, capturing: bool = True):
        if self._fast_cuda and index == 0 and self._do_in_this_step and self._in_rows_guard_active:
            self._main_stream.wait_event(self._ev_in_rows_consumed)
            self._in_rows_guard_active = False
        if self._fast_cuda and index == 0 and capturing:
            if self._rows_guard_active:
                self._main_stream.wait_event(self._ev_rows_consumed)
                self._rows_guard_active = False
            if self._inline_diag is not None:
                self._inline_diag.zero_()
        if capturing:
            if self._row_off is not None:
                if self._fast_cuda and index < self._host_row_offsets.numel():
                    self._row_off.copy_(self._host_row_offsets[index], non_blocking=True)
                else:
                    self._row_off.fill_(index * self.rows_per_microbatch)
            elif self.output_at is not None:
                start = index * self.rows_per_microbatch
                self.output_at.copy_(torch.arange(start, start + self.rows_per_microbatch, device=self.device))

        if (
            self._fast_cuda
            and self.lazy.enabled
            and self.lazy.lag_input
            and index >= 1
            and self._has_in_flight_input()
            and self._input_ready_for_collect()
        ):
            self._collect_in_flight_input()

        if (
            self._fast_cuda
            and self.arm.lag_left
            and index >= 1
            and (
                (self.output is not None and self.output.begun)
                or self.pending_diagonal
                or self.pending_headblock
                or self._headblock_in_flight
            )
            and self._output_ready_for_collect()
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
        """Input statistics and roots after the last captured microbatch's forward."""
        if not self._do_in_this_step:
            return
        if self.lazy.enabled and self.lazy.lag_input and self._has_in_flight_input():
            self._collect_in_flight_input()
        if self.input is not None:
            self.input.begin()
            if getattr(self, "_fast_cuda", False) and isinstance(self.input, BatchExactDenseEngine):
                with torch.cuda.stream(self._input_side_stream):
                    self._input_side_stream.wait_event(self.input.ev_rows_consumed)
                    self._ev_in_rows_consumed.record(self._input_side_stream)
                self._in_rows_guard_active = True
        if self.input_diagonal is not None and not self._input_diag_begun:
            if getattr(self, "_fast_cuda", False):
                self._begin_input_diagonal()
            elif self.lazy.enabled and self.lazy.lag_input:
                self.input_diagonal.update()
                self._input_diag_factors = self.input_diagonal.factors()
                self._input_diag_begun = True

    def _prepare_lagged_for_groups_on_side(self, factors: dict):
        """Pre-stack and FP16-split lagged left factors on _side_stream so step() pays 0 ms for them."""
        if not factors or not self.arm.apply_left:
            return {}
        prepared = {}
        with torch.cuda.stream(self._side_stream):
            for g_key, per_rank in self.shape_groups:
                mine = per_rank[self.rank]
                if not mine:
                    continue
                shape = self.group_shapes[g_key]
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
                prepared[g_key] = (li_prep, lo_prep)
        return prepared

    def finish_backward(self):
        """Output statistics and roots once every output cotangent row of this update exists."""
        if (
            self.output is None
            and self.output_diagonal is None
            and self.output_headblock is None
            and not self.arm.lag_left
        ):
            return

        if not getattr(self, "_fast_cuda", False):
            if self.arm.lag_left:
                self.lagged = self._collect_lagged()
            if not self._do_out_this_step:
                return
            if self.output is not None:
                self.output.begin()
            if self.arm.lag_left and self.output_diagonal is not None:
                self.output_diagonal.update()
                self.pending_diagonal = self.output_diagonal.factors()
            if self.arm.lag_left and self.output_headblock is not None:
                self.pending_headblock = self.output_headblock.update_and_factors()
            return

        if self.arm.lag_left and (
            (self.output is not None and self.output.begun)
            or self.pending_diagonal
            or self.pending_headblock
            or self._headblock_in_flight
        ):
            self.lagged = self._collect_lagged()

        if not self._do_out_this_step:
            return

        ev_bwd = torch.cuda.Event()
        ev_bwd.record(self._main_stream)

        diag_sums = None
        if self.arm.lag_left and (self.output_diagonal is not None or self.output_headblock is not None):
            with torch.cuda.stream(self._side_stream):
                self._side_stream.wait_event(ev_bwd)
                if self.output_diagonal is not None:
                    diag_sums = self.output_diagonal.local_sums()
                if self.output_headblock is not None:
                    self.output_headblock.compute_local_grams()

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
            if self.arm.lag_left and self.output_headblock is not None:
                self.output_headblock.start_ema_and_async_roots(self._side_stream)
                self._headblock_in_flight = True

    def _collect_lagged(self):
        """Left factors built from the previous refresh's rows, safe to outlive subsequent launches."""
        if not getattr(self, "_fast_cuda", False):
            dense = {}
            if self.output is not None and self.output.begun:
                dense = self._engine_factors(self.output, copy=True)
            if not dense and not self.pending_diagonal and not self.pending_headblock:
                return self.lagged
            combined = self.pending_diagonal | self.pending_headblock | dense
            self.pending_diagonal = {}
            self.pending_headblock = {}
            return combined

        dense = {}
        if self.output is not None and self.output.begun:
            if isinstance(self.output, BatchExactDenseEngine):
                roots = self.output.finish_roots_on_side(self._side_stream)
            else:
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
        if self._headblock_in_flight and self.output_headblock is not None:
            self.pending_headblock = self.output_headblock.collect_async_roots(self._side_stream)
            with torch.cuda.stream(self._side_stream):
                for pair in self.pending_headblock.values():
                    for t in pair:
                        t.record_stream(self._main_stream)
            self._headblock_in_flight = False
        if not dense and not self.pending_diagonal and not self.pending_headblock:
            return self.lagged
        combined = self.pending_diagonal | self.pending_headblock | dense
        self.pending_diagonal = {}
        self.pending_headblock = {}
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

    def _finish_engine(self, engine):
        if isinstance(engine, BatchExactDenseEngine):
            return engine.finish_roots_on_main()
        if not engine.begun:
            engine.begin()
        engine.begun = False
        return engine._finish_roots()

    def _engine_factors(self, engine, copy=False):
        roots = self._finish_engine(engine)
        factors = {}
        for name, group in engine.groups.items():
            inner, outer = roots[name]
            for index, key in enumerate(group.own):
                factors[key] = (inner[index].clone(), outer[index].clone()) if copy else (inner[index], outer[index])
        return factors

    def _factors(self, engine, diagonal, headblock=None):
        factors = {} if engine is None else self._engine_factors(engine)
        if diagonal is not None:
            diagonal.update()
            factors |= diagonal.factors()
        if headblock is not None:
            factors |= headblock.update_and_factors()
        return factors

    def _rebuild_cached_right_prep(self, fast_cuda: bool, record_main: bool = False) -> None:
        self._cached_right_prep = {}
        for g_key, per_rank in self.shape_groups:
            mine = per_rank[self.rank]
            if not mine:
                continue
            shape = self.group_shapes[g_key]
            right_pairs = [self._cached_right.get((layer, INPUT_KIND[kind]), (None, None)) for layer, kind, _ in mine]
            right_inner = stack_factors([p[0] for p in right_pairs], leading=len(mine), width=shape[1])
            right_outer = stack_factors([p[1] for p in right_pairs], leading=len(mine), width=shape[1])
            if fast_cuda:
                ri_prep = _prepare_side_factor(right_inner, side="right", normalize_rms=False, compiled=True)
                ro_prep = _prepare_side_factor(right_outer, side="right", normalize_rms=True, compiled=True)
                if record_main:
                    for item in (ri_prep, ro_prep):
                        if isinstance(item, tuple):
                            for t in item:
                                t.record_stream(self._main_stream)
                        elif isinstance(item, torch.Tensor):
                            item.record_stream(self._main_stream)
                self._cached_right_prep[g_key] = (ri_prep, ro_prep)
            else:
                self._cached_right_prep[g_key] = (right_inner, right_outer)

    def check_last_step_finite(self):
        """Verify finite block updates after main_stream synchronization (0 ms host stall when deferred)."""
        if self._last_finite_checks is None:
            return
        checks = self._last_finite_checks.tolist()
        self._last_finite_checks = None
        if not all(checks):
            bad = [g_key for (g_key, _), ok in zip(self.shape_groups, checks, strict=True) if not ok]
            raise FloatingPointError(f"Nonfinite block update for shapes {bad}")

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        fast_cuda = getattr(self, "_fast_cuda", False)
        input_roots = None
        output_roots = None
        right = {}
        left = {}
        use_lag_in = self.lazy.enabled and self.lazy.lag_input
        need_initial_in = not self._cached_right and (self.input is not None or self.input_diagonal is not None)
        refresh_in_now = (not use_lag_in and self._do_in_this_step) or need_initial_in

        if fast_cuda:
            if use_lag_in:
                if need_initial_in:
                    if self.input is not None and not self.input.begun:
                        self.input.begin()
                    if self.input_diagonal is not None and not self._input_diag_begun:
                        self._begin_input_diagonal()
                    self._collect_in_flight_input()
                elif not self._do_in_this_step and self._has_in_flight_input():
                    self._collect_in_flight_input()
                if self._cached_right_prep:
                    self._main_stream.wait_event(self._ev_input_lagged_ready)
            elif refresh_in_now:
                if self.input is not None:
                    input_roots = self._finish_engine(self.input)
                if self.input_diagonal is not None:
                    if not self._input_diag_begun:
                        self._begin_input_diagonal()
                    self._input_diag_begun = False
                    self._main_stream.wait_event(self._ev_input_diag_ready)
                    right = self._input_diag_factors
                    self._input_diag_factors = {}
                if self.lazy.enabled:
                    new_right = dict(right)
                    if input_roots is not None:
                        for name, grp in self.input.groups.items():
                            ri_g, ro_g = input_roots[name]
                            for idx_k, key in enumerate(grp.own):
                                new_right[key] = (ri_g[idx_k].clone(), ro_g[idx_k].clone())
                    self._cached_right = new_right
                    self._rebuild_cached_right_prep(fast_cuda=True)

            if self.arm.lag_left:
                left = self.lagged
            elif self._do_out_this_step:
                if self.output_diagonal is not None:
                    self.output_diagonal.update()
                    left |= self.output_diagonal.factors()
                if self.output_headblock is not None:
                    left |= self.output_headblock.update_and_factors()
                if self.output is not None:
                    output_roots = self._finish_engine(self.output)
        else:
            if use_lag_in:
                if need_initial_in:
                    if self.input is not None and not self.input.begun:
                        self.input.begin()
                    self._collect_in_flight_input()
                elif not self._do_in_this_step and self._has_in_flight_input():
                    self._collect_in_flight_input()
                right = self._cached_right
            elif refresh_in_now:
                right = self._factors(self.input, self.input_diagonal)
                if self.lazy.enabled:
                    self._cached_right = {k: (a.clone(), b.clone()) for k, (a, b) in right.items()}
            elif self.lazy.enabled:
                right = self._cached_right
            left = self.lagged if self.arm.lag_left else self._factors(self.output, self.output_diagonal, self.output_headblock)

        self.last_metrics = self._engine_metrics()
        if not self.arm.apply_left:
            left = {}
            output_roots = None
        elif not self.arm.apply_left_inner:
            left = {key: (None, outer) for key, (_, outer) in left.items()}

        use_prepped_lag = fast_cuda and self.arm.lag_left
        if use_prepped_lag and self._lagged_prepared:
            self._main_stream.wait_event(self._ev_lagged_ready)

        if fast_cuda:
            self._step_scalars_cpu[0] = group["lr"]
            self._step_scalars_cpu[1] = group["weight_decay"]
            lr_t, wd_t = self._step_scalars_cpu.to(self.device, non_blocking=True).unbind(0)

        works, finite = [], []
        for g_key, per_rank in self.shape_groups:
            shape = self.group_shapes[g_key]
            momentum = self.momentum[g_key]
            mine = per_rank[self.rank]
            if mine:
                if fast_cuda:
                    if self.lazy.enabled and g_key in self._cached_right_prep:
                        ri_prep, ro_prep = self._cached_right_prep[g_key]
                    else:
                        if input_roots is not None and g_key in self._input_gather_spec:
                            g_name, g_idx = self._input_gather_spec[g_key]
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
                        ri_prep = _prepare_side_factor(right_inner, side="right", normalize_rms=False, compiled=True)
                        ro_prep = _prepare_side_factor(right_outer, side="right", normalize_rms=True, compiled=True)

                    if use_prepped_lag:
                        li_prep, lo_prep = self._lagged_prepared.get(g_key, (None, None))
                    elif output_roots is not None and g_key in self._output_gather_spec:
                        g_name, g_idx = self._output_gather_spec[g_key]
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
                    r_in_mode = _factor_mode_code(ri_prep)
                    r_out_mode = _factor_mode_code(ro_prep)
                    l_in_mode = _factor_mode_code(li_prep)
                    l_out_mode = _factor_mode_code(lo_prep)
                    self.momentum[g_key], source = _compiled_block_in(
                        gs,
                        momentum,
                        li_prep,
                        ri_prep,
                        float(self.args.muon_momentum),
                        bool(self.args.muon_nesterov),
                        r_in_mode,
                        l_in_mode,
                    )
                    direction = _newton_schulz_batched(
                        source,
                        self.args.muon_ns_steps,
                        self.args.muon_eps,
                        self.args.muon_ns_mode,
                        getattr(self.args, "muon_ns_coeffs", "simple"),
                        bool(getattr(self.args, "muon_compile_ns", False)),
                    )
                    scale = max(1.0, shape[0] / shape[1]) ** 0.5
                    out_fn = _select_block_out_fn(
                        getattr(self, "decay_mode", DecayMode.DEFAULT), compiled=True
                    )
                    delta = out_fn(
                        ps, direction, lo_prep, ro_prep, lr_t, wd_t, scale, r_out_mode, l_out_mode
                    )
                else:
                    right_pairs = [right.get((layer, INPUT_KIND[kind]), (None, None)) for layer, kind, _ in mine]
                    right_inner = stack_factors([p[0] for p in right_pairs], leading=len(mine), width=shape[1])
                    right_outer = stack_factors([p[1] for p in right_pairs], leading=len(mine), width=shape[1])
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
                bad = [g_key for (g_key, _), ok in zip(self.shape_groups, checks, strict=True) if not ok]
                raise FloatingPointError(f"Nonfinite block update for shapes {bad}")

        for per_rank, work, gathered in works:
            if work is not None:
                work.wait()
            for entries, updates in zip(per_rank, gathered, strict=True):
                if entries:
                    if defer:
                        updates = torch.where(all_ok, updates, torch.zeros((), dtype=updates.dtype, device=updates.device))
                    torch._foreach_add_([parameter for _, _, parameter in entries], list(updates.unbind(0)))
        for label, engine in (("in", self.input), ("out", self.output)):
            if engine is None:
                continue
            if (label == "in" and (self._do_in_this_step or refresh_in_now)) or (label == "out" and self._do_out_this_step):
                engine.steps += 1
                engine.t += 1
        self.steps += 1

    def _engine_metrics(self):
        metrics = {}
        for label, engine in (("input", self.input), ("output", self.output)):
            if engine is None:
                continue
            metrics[f"{label}_fallbacks"] = engine.fallbacks
            metrics[f"{label}_extra_roots"] = engine.extra
            if hasattr(engine, "pool"):
                ages = [engine.steps - step for g in engine.groups.values() for step in g.basis_step]
                metrics[f"{label}_max_basis_age"] = max(ages, default=0)
                metrics[f"{label}_pending_refreshes"] = sum(
                    p is not None for g in engine.groups.values() for p in g.pending
                )
            else:
                metrics[f"{label}_refreshes"] = engine.steps
        return metrics

    def runtime_state(self):
        if getattr(self, "_fast_cuda", False):
            self._input_side_stream.synchronize()
            self._side_stream.synchronize()
        if self.lazy.enabled and self.lazy.lag_input and self._has_in_flight_input():
            self._collect_in_flight_input()
            if getattr(self, "_fast_cuda", False):
                self._input_side_stream.synchronize()
        if self.arm.lag_left:
            self.lagged = self._collect_lagged()
            self.pending_diagonal = {}
            self.pending_headblock = {}
            if getattr(self, "_fast_cuda", False):
                self._side_stream.synchronize()
        state = {
            "lagged": self.lagged,
            "input": None if self.input is None else engine_state(self.input),
            "output": None if self.output is None else engine_state(self.output),
            "input_diagonal": None if self.input_diagonal is None else self.input_diagonal.state_dict(),
            "output_diagonal": None if self.output_diagonal is None else self.output_diagonal.state_dict(),
            "momentum": self.momentum,
            "steps": self.steps,
        }
        if self.output_headblock is not None:
            state["output_headblock"] = self.output_headblock.state_dict()
        if self.lazy.enabled:
            state["cached_right"] = self._cached_right
        return state

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
        if self.output_headblock is not None and "output_headblock" in state:
            self.output_headblock.load_state_dict(state["output_headblock"])
        for g_key, momentum in self.momentum.items():
            momentum.copy_(state["momentum"][g_key])
        self.lagged = {
            key: tuple(tensor.to(self.device) for tensor in pair) for key, pair in state.get("lagged", {}).items()
        }
        self.pending_diagonal = {}
        self.pending_headblock = {}
        if self.lazy.enabled and "cached_right" in state:
            self._cached_right = {
                key: tuple(tensor.to(self.device) for tensor in pair)
                for key, pair in state["cached_right"].items()
            }
            self._rebuild_cached_right_prep(fast_cuda=getattr(self, "_fast_cuda", False))
            if getattr(self, "_fast_cuda", False):
                self._ev_input_lagged_ready.record(self._main_stream)
        if getattr(self, "_fast_cuda", False) and self.arm.lag_left:
            self._lagged_prepared = self._prepare_lagged_for_groups_on_side(self.lagged)
            self._ev_lagged_ready.record(self._side_stream)
        self.steps = state["steps"]
