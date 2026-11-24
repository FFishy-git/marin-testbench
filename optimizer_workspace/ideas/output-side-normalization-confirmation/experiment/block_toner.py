# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input, output, or two-sided Toner factors on every block matrix.

Output cotangent rows are stored during backward via grouped Triton kernels (or accumulated
inline for diagonal output moments) and pooled by Toner's own statistics, cached-root, and
host-refresh machinery. Lagged arms overlap dense root candidates, repairs, exact fallbacks,
and factor splitting asynchronously on a side stream across updates.
"""

import contextlib
import copy
import os
import types
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
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


def _exact_root_eager(lam: torch.Tensor, V: torch.Tensor, n: int, p: float, B2: torch.Tensor) -> torch.Tensor:
    x = lam.clamp_min(0)
    x = x / x.mean(-1, keepdim=True)
    f = (x + PR.EPS).pow(-p)
    A2 = PR._cols_hi(V * f[..., None, :])
    if PR._sym_ok(n) and V.dim() == 3:
        return PR.mm16_sym(A2, B2, False)
    R = PR.mm16(A2, B2, False)
    return 0.5 * (R + R.mT)


def _batched_exact_fallback(engine, g, R, still: list[int]) -> None:
    """Solve all fallback statistics in one unpadded batched eigendecomposition with zero dynamo recompiles."""
    s = len(still)
    if not s:
        return
    still_idx = torch.tensor(still, dtype=torch.int64, device=engine.device)
    C_sub = g.C.index_select(0, still_idx)
    lam_x, V = torch.linalg.eigh(0.5 * (C_sub + C_sub.mT))
    g.tol.index_copy_(0, still_idx, engine._tolerance(lam_x))

    B2 = PR._rows_lo(V.mT.contiguous())
    if engine.split:
        R[0].index_copy_(0, still_idx, _exact_root_eager(lam_x, V, g.n, engine.p_in, B2))
        if R[1] is not R[0]:
            R[1].index_copy_(0, still_idx, _exact_root_eager(lam_x, V, g.n, engine.p_out, B2))
    else:
        R.index_copy_(0, still_idx, _exact_root_eager(lam_x, V, g.n, PR.ALPHA, B2))

    Vc = V.contiguous()
    g.basis.SB.index_copy_(0, still_idx, PR._rows_lo(Vc))
    g.basis.S2.index_copy_(0, still_idx, PR._cols_hi(Vc))


@torch.no_grad()
def prewarm_repair_shapes(engine) -> None:
    """Pre-compile power-of-two repair batch shapes on all ranks at startup to prevent mid-run DDP stalls."""
    if torch.device(engine.device).type != "cuda" or engine.args.toner_repair_rounds <= 0:
        return
    seen: set[tuple[int, int]] = set()
    for g in engine.groups.values():
        if not g.per:
            continue
        max_k = 1 << (g.per - 1).bit_length()
        k = 1
        while k <= max_k:
            if (k, g.n) not in seen:
                seen.add((k, g.n))
                B_d = torch.eye(g.n, dtype=torch.float32, device=engine.device).unsqueeze(0).expand(k, -1, -1).contiguous()
                lam_d = torch.ones(k, g.n, dtype=torch.float32, device=engine.device)
                O_d = torch.zeros(k, g.n, g.n, dtype=torch.float32, device=engine.device)
                dl_d = torch.full((k,), 1e-3, dtype=torch.float32, device=engine.device)
                S2_d = torch.zeros(k, g.n, 2 * g.n, dtype=torch.float16, device=engine.device)
                lam_o, O_o, _, _, _ = engine.froot.froot_more(B_d, (lam_d, O_d, dl_d), 2, engine.args.toner_freeze)
                sub_b = PR.Basis.of_splits(S2_d)
                if engine.split:
                    PR.form_powers(sub_b, lam_o, O_o)
                else:
                    PR.form(sub_b, lam_o, O_o)
                del B_d, lam_d, O_d, dl_d, S2_d, lam_o, O_o
            k <<= 1
    torch.cuda.synchronize(engine.device)


def _launch_engine_repairs(engine) -> bool:
    """Stage 1: inspect flags_host and enqueue masked repair rounds on engine.side with zero .tolist() syncs."""
    a = engine.args
    if getattr(engine, "ev_flags", None) is not None:
        engine.ev_flags.synchronize()
    flags = engine.flags_host.tolist()
    roots, o, todo = {}, 0, []
    for name, g in engine.groups.items():
        R, B, _, state = engine.cand[name]
        # Flag encoding: 0 = converged, 1 = repairable (rho <= rho_max), 2 = hopeless (rho > rho_max).
        bad_all = [j for j in range(g.per) if flags[o + j]]
        repairable = [j for j in range(g.per) if flags[o + j] == 1]
        hopeless = [j for j in range(g.per) if flags[o + j] >= 2]
        roots[name] = R
        if bad_all:
            todo.append((g, R, B, state, bad_all, repairable, hopeless))
        o += g.per

    engine._pending_roots = roots
    if not todo:
        engine._repair_todo = []
        engine.ev_repairs = None
        return False

    is_cuda = torch.device(engine.device).type == "cuda"
    ctx = torch.cuda.stream(engine.side) if is_cuda else contextlib.nullcontext()
    total_rep = sum(len(rep) for _, _, _, _, _, rep, _ in todo)
    if total_rep > 0 and (
        getattr(engine, "repair_done_host", None) is None or engine.repair_done_host.numel() < total_rep
    ):
        engine.repair_done_host = torch.zeros(max(1, total_rep), dtype=torch.uint8, pin_memory=is_cuda)

    done_chunks = []
    rep_slices = []
    rep_offset = 0
    rounds = a.toner_repair_rounds
    with ctx:
        for g, R, B, state, bad_all, repairable, hopeless in todo:
            m = len(repairable)
            if m == 0 or state is None or rounds <= 0:
                rep_slices.append((g, R, bad_all, repairable, hopeless, rep_offset, 0))
                continue
            k = 1 << (m - 1).bit_length()
            pad = repairable + repairable[-1:] * (k - m)
            idx = torch.tensor(pad, dtype=torch.int64, device=engine.device)
            idx_u = idx[:m]
            B_sub = B.index_select(0, idx)
            sub_basis = PR.Basis.of_splits(g.basis.S2.index_select(0, idx))
            tol_sub = g.tol.index_select(0, idx)
            st = tuple(t.index_select(0, idx) for t in state)
            valid = torch.arange(k, device=engine.device) < m
            active = valid.clone()
            done = torch.zeros(k, dtype=torch.bool, device=engine.device)

            for _ in range(rounds):
                lam, O, rho, est, dl = engine.froot.froot_more(B_sub, st, 2, a.toner_freeze)
                if engine.split:
                    Rp2, est_s = PR.form_powers(sub_basis, lam, O)
                    R2_in, R2_out = Rp2[engine.p_in], Rp2[engine.p_out]
                    ok = (
                        (rho <= a.toner_rho_max)
                        & ~engine._split_over(est, est_s, tol_sub)
                        & torch.isfinite(lam).all(-1)
                    )
                else:
                    R2_in = PR.form(sub_basis, lam, O)
                    R2_out = R2_in
                    ok = (rho <= a.toner_rho_max) & (est <= tol_sub) & torch.isfinite(lam).all(-1)
                rhok = rho <= a.toner_rho_max
                newly_ok = active & ok
                m_ok = newly_ok[:m, None, None]
                if engine.split:
                    R[0].index_copy_(0, idx_u, torch.where(m_ok, R2_in[:m], R[0].index_select(0, idx_u)))
                    if R[1] is not R[0]:
                        R[1].index_copy_(0, idx_u, torch.where(m_ok, R2_out[:m], R[1].index_select(0, idx_u)))
                else:
                    R.index_copy_(0, idx_u, torch.where(m_ok, R2_in[:m], R.index_select(0, idx_u)))
                done |= newly_ok
                active = active & ~ok & rhok
                st = (lam, O, dl)

            done_chunks.append(done[:m].to(torch.uint8))
            rep_slices.append((g, R, bad_all, repairable, hopeless, rep_offset, m))
            rep_offset += m

        if done_chunks:
            cat_done = torch.cat(done_chunks)
            engine.repair_done_host[: cat_done.numel()].copy_(cat_done, non_blocking=is_cuda)
            if is_cuda:
                engine.ev_repairs = torch.cuda.Event()
                engine.ev_repairs.record(engine.side)
            else:
                engine.ev_repairs = None
        else:
            engine.ev_repairs = None

    engine._repair_todo = rep_slices
    return True


def _finish_engine_fallbacks(engine):
    """Stage 2: read repair outcomes and solve all remaining fallbacks via one batched eigendecomposition."""
    if getattr(engine, "ev_repairs", None) is not None:
        engine.ev_repairs.synchronize()
        engine.ev_repairs = None

    roots = engine._pending_roots
    rep_slices = engine._repair_todo
    engine._pending_roots = None
    engine._repair_todo = None
    if not rep_slices:
        return roots

    total_m = sum(m for _, _, _, _, _, _, m in rep_slices)
    done_list = engine.repair_done_host[:total_m].tolist() if total_m > 0 else []
    is_cuda = torch.device(engine.device).type == "cuda"
    ctx = torch.cuda.stream(engine.side) if is_cuda else contextlib.nullcontext()
    with ctx:
        for g, R, bad_all, repairable, hopeless, off, m in rep_slices:
            if m > 0:
                sub_done = done_list[off : off + m]
                engine.extra += sum(1 for d in sub_done if d)
                failed_rep = [repairable[i] for i in range(m) if not sub_done[i]]
            else:
                failed_rep = list(repairable)
            still = failed_rep + list(hopeless)
            if still:
                engine.fallbacks += len(still)
                _batched_exact_fallback(engine, g, R, still)
                for j in still:
                    g.basis_step[j] = engine.steps
            if engine.refresh_flagged:
                still_set = set(still)
                for j in bad_all:
                    if j not in still_set and g.pending[j] is None:
                        engine._submit_refresh(g, j)
    return roots


def finish_roots_batched(engine):
    """Batched, zero-roundtrip replacement for FastTonerCore._finish_roots."""
    a = engine.args
    if getattr(engine, "_repair_todo", None) is None:
        had_todo = _launch_engine_repairs(engine)
    else:
        had_todo = bool(engine._repair_todo)
    roots = _finish_engine_fallbacks(engine)
    if torch.device(engine.device).type == "cuda":
        if had_todo:
            engine.main_stream.wait_stream(engine.side)
        else:
            engine.main_stream.wait_event(engine.ev_done)
    if a.toner_check_every and engine.steps % a.toner_check_every == 0:
        for name, Rr in roots.items():
            g = engine.groups[name]
            if not g.per:
                continue
            if engine.split:
                _, Rxp = PR.exact_eigh_powers(g.C)
                pairs = [(name, Rr[0], Rxp[engine.p_in]), (name + "_out", Rr[1], Rxp[engine.p_out])]
            else:
                _, Rx = PR.exact_eigh(g.C)
                pairs = [(name, Rr, Rx)]
            for key, R, Rx in pairs:
                rel = (R - Rx).norm(dim=(-2, -1)) / Rx.norm(dim=(-2, -1))
                Cn = g.C / g.C.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
                d = R - Rx
                ec = (
                    ((d @ Cn @ d.mT).diagonal(dim1=-2, dim2=-1).sum(-1) / (Rx @ Cn @ Rx.mT).diagonal(dim1=-2, dim2=-1).sum(-1))
                    .clamp_min(0)
                    .sqrt()
                )
                engine.check[key] = (rel.max().item(), rel.median().item(), ec.max().item())
    return roots


def roots_2bit(engine, g):
    """Root candidates with 2-bit flag encoding (0 = converged, 1 = repairable, 2 = hopeless rho > rho_max)."""
    a = engine.args
    B, _ = PR.basis_transform(g.C, g.basis)
    lam, O, rho, est, state = engine.froot.froot_state(B, engine.iters, a.toner_freeze)
    hopeless = rho > a.toner_rho_max
    if engine.split:
        Rp, est_s = PR.form_powers(g.basis, lam, O)
        bad = hopeless | engine._split_over(est, est_s, g.tol) | ~torch.isfinite(lam).all(-1)
        status = bad.to(torch.uint8) + (bad & hopeless).to(torch.uint8)
        return (Rp[engine.p_in], Rp[engine.p_out]), B, status, state
    bad = hopeless | ~(est <= g.tol) | ~torch.isfinite(lam).all(-1)
    status = bad.to(torch.uint8) + (bad & hopeless).to(torch.uint8)
    return PR.form(g.basis, lam, O), B, status, state


def _parse_cpulist(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


try:
    _INITIAL_AFFINITY: set[int] | None = set(os.sched_getaffinity(0))
except OSError:
    _INITIAL_AFFINITY = None


def configure_numa_host_args(args, device, *, role: str = "input"):
    """Isolate background host eigendecomposition workers onto NUMA-local cores away from main/NCCL threads."""
    out = copy.copy(args)
    if torch.device(device).type != "cuda" or getattr(out, "toner_host_cores", ""):
        if role == "output":
            out.toner_host_workers = min(3, max(1, int(out.toner_host_workers)))
            out.toner_host_threads = min(2, max(1, int(out.toner_host_threads)))
        return out
    dev_idx = torch.device(device).index
    if dev_idx is None:
        dev_idx = torch.cuda.current_device()
    cpulist_path = Path(f"/sys/devices/system/node/node{dev_idx}/cpulist")
    try:
        raw_cpus = _parse_cpulist(cpulist_path.read_text())
        numa_cpus = [c for c in raw_cpus if _INITIAL_AFFINITY is None or c in _INITIAL_AFFINITY]
    except OSError:
        numa_cpus = []
    if len(numa_cpus) >= 16:
        half = len(numa_cpus) // 2
        phys = numa_cpus[:half]
        sibs = numa_cpus[half:]
        # Reserve phys[:4] (+ sibs[:4]) exclusively for the rank's PyTorch main thread & NCCL proxy threads.
        try:
            main_cores = set(phys[:4] + sibs[:4])
            if main_cores:
                os.sched_setaffinity(0, main_cores)
        except OSError:
            pass
        bg_pool = phys[4:] + sibs[4:]
        if role == "input":
            in_pool = bg_pool[: min(12, max(6, len(bg_pool) * 2 // 3))]
            out.toner_host_cores = ",".join(str(c) for c in in_pool)
            out.toner_host_workers = min(2, max(1, int(out.toner_host_workers)))
            out.toner_host_threads = max(2, min(6, len(in_pool) // max(1, out.toner_host_workers)))
        else:
            split_idx = min(12, max(6, len(bg_pool) * 2 // 3))
            out_pool = bg_pool[split_idx:] or bg_pool[-6:]
            out.toner_host_cores = ",".join(str(c) for c in out_pool)
            out.toner_host_workers = min(3, max(2, len(out_pool) // 2))
            out.toner_host_threads = max(1, min(2, len(out_pool) // max(1, out.toner_host_workers)))
    elif role == "output":
        out.toner_host_workers = min(3, max(1, int(out.toner_host_workers)))
        out.toner_host_threads = min(2, max(1, int(out.toner_host_threads)))
    return out


class OutputToner(FastTonerCore):
    """Toner's pooled statistics, cached roots, and host refresh over output-cotangent rows."""

    def __init__(self, rows, kinds, *, layers, hidden, mlp, rows_total, args, world, rank, device, lagged=False):
        out_args = configure_numa_host_args(args, device, role="output")
        placeholder = torch.empty(0, 1, dtype=torch.bfloat16, device=device)
        blocks = [SimpleNamespace(**{f"input_rows_{kind}": placeholder for kind in INPUT_KINDS})] * layers
        super().__init__(
            blocks, {}, "muont", hid=1, mlp=1, rows_total=rows_total, args=out_args, world=world, rank=rank, device=device
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
        total_per = sum(g.per for g in self.groups.values())
        self.flags_host = torch.zeros(total_per, dtype=torch.uint8, pin_memory=True)
        self.repair_done_host = torch.zeros(max(1, total_per), dtype=torch.uint8, pin_memory=True)
        self.ev_repairs = None
        self._repair_todo = None
        self._pending_roots = None

    def _roots(self, g):
        return roots_2bit(self, g)

    def begin(self):
        self._repair_todo = None
        self._pending_roots = None
        self.ev_repairs = None
        return super().begin()

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

    def poll_repairs_nonblocking(self) -> bool:
        """Advance Stage 1 -> Stage 2 without blocking the Python main thread; return True when ready to finalize."""
        if not self.begun:
            return False
        if self._repair_todo is None:
            if not self.ev_flags.query():
                return False
            _launch_engine_repairs(self)
        return self.ev_repairs is None or self.ev_repairs.query()

    def finish_roots_on_side(self):
        """Finish flagged root repairs and batched exact fallbacks purely on self.side without stalling main_stream."""
        if self._repair_todo is None:
            _launch_engine_repairs(self)
        return _finish_engine_fallbacks(self)

    def _finish_roots(self):
        return finish_roots_batched(self)


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
        self._chunk_sums: dict[int, torch.Tensor] = {}
        self.weight = 0.0
        self.pg = dist.new_group(backend=dist.get_backend()) if world > 1 else None

    @torch.no_grad()
    def _raw_local_sums(self):
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
    def accum_chunk(self):
        """Accumulate row-buffer squares for a completed chunk prior to buffer reuse."""
        for width, cur in self._raw_local_sums().items():
            if self.inline is not None and self.inline.is_active(width):
                continue
            if width in self._chunk_sums:
                self._chunk_sums[width].add_(cur)
            else:
                self._chunk_sums[width] = cur.clone()

    @torch.no_grad()
    def local_sums(self):
        sums = self._raw_local_sums()
        if self._chunk_sums:
            for width, prev in self._chunk_sums.items():
                sums[width] = prev.add_(sums[width])
            self._chunk_sums.clear()
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


_prepare_right_dense = torch.compile(_prepare_right_dense_raw, dynamic=False, options=PR._EXACT)
_prepare_left_dense = torch.compile(_prepare_left_dense_raw, dynamic=False, options=PR._EXACT)
_compiled_block_in = torch.compile(_block_in_raw, dynamic=False, options=PR._EXACT)
_compiled_block_out = torch.compile(_block_out_raw, dynamic=False, options=PR._EXACT)


def _prepare_side_factor(factor: torch.Tensor | None, *, side: str, normalize_rms: bool, compiled: bool):
    if factor is None:
        return None
    if factor.ndim == 2:
        return stack_unit_rms(factor) if normalize_rms else factor
    if side == "right":
        return (_prepare_right_dense if compiled else _prepare_right_dense_raw)(factor, normalize_rms)
    return (_prepare_left_dense if compiled else _prepare_left_dense_raw)(factor, normalize_rms)


def block_update_stack(weights, gradients, momenta, left, right, *, args, lr, weight_decay, compiled=True):
    """The batched form of block_update over a stack of same-shape matrices.

    On CUDA, dense factor products use FP32-accurate Ozaki FP16-split Tensor-Core GEMMs and
    fuse the post-NS direction and weight-decay right products into two GEMMs instead of three.
    """
    left_inner, left_outer = left
    right_inner, right_outer = right
    if weights.is_cuda:
        use_compile = bool(compiled)
        li = _prepare_side_factor(left_inner, side="left", normalize_rms=False, compiled=use_compile)
        lo = _prepare_side_factor(left_outer, side="left", normalize_rms=True, compiled=use_compile)
        ri = _prepare_side_factor(right_inner, side="right", normalize_rms=False, compiled=use_compile)
        ro = _prepare_side_factor(right_outer, side="right", normalize_rms=True, compiled=use_compile)
        in_fn = _compiled_block_in if use_compile else _block_in_raw
        out_fn = _compiled_block_out if use_compile else _block_out_raw
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
        decay = torch.bmm(decay, right_norm) if right_norm.ndim == 3 else decay * right_norm[:, None, :]
        decay = torch.bmm(decay, right_norm) if right_norm.ndim == 3 else decay * right_norm[:, None, :]
    if left_outer is not None:
        left_norm = stack_unit_rms(left_outer)
        direction = torch.bmm(left_norm, direction) if left_norm.ndim == 3 else direction * left_norm[:, :, None]
        if right_outer is None:
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
        self.args = fast_arguments(config)
        self.args.toner_stats_ctas = int(config.get("root", {}).get("stats_ctas", 0))
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

        in_args = configure_numa_host_args(self.args, device, role="input")
        in_engine = {"rows_total": rows_total, "args": in_args, "world": world, "rank": rank, "device": device}
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
            self.input = FastTonerCore(input_blocks, {}, "muont", hid=hidden, mlp=hidden, **in_engine)
            del self.input.groups["f"]
            self.input.flags_host = torch.zeros(
                sum(g.per for g in self.input.groups.values()), dtype=torch.uint8, pin_memory=True
            )
        elif input_full_kinds == INPUT_KINDS:
            self.input = FastTonerCore(blocks, {}, "muont", hid=hidden, mlp=mlp, **in_engine)
        elif not input_full_kinds:
            self.input = None
        else:
            raise ValueError(f"Unsupported dense input kinds: {input_full_kinds}")
        if self.input is not None:
            self.input._roots = types.MethodType(roots_2bit, self.input)
            self.input._finish_roots = types.MethodType(finish_roots_batched, self.input)
            prewarm_repair_shapes(self.input)

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
        if self.output is not None:
            prewarm_repair_shapes(self.output)
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
            self._ev_in_diag_chunk = None
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

    def _poll_lagged_nonblocking(self):
        if not self._fast_cuda or not self.arm.lag_left:
            return
        has_work = (self.output is not None and self.output.begun) or bool(self.pending_diagonal)
        if not has_work:
            return
        if self.output is not None and self.output.begun:
            if not self.output.poll_repairs_nonblocking():
                return
        self.lagged = self._collect_lagged()

    def chunk_wait(self):
        """Wait for prior input chunk consumption before overwriting input row buffers."""
        if self.input is not None:
            self.input.chunk_wait()
        if self._fast_cuda and self._ev_in_diag_chunk is not None:
            self._main_stream.wait_event(self._ev_in_diag_chunk)

    def chunk_stats(self):
        """Accumulate partial input Grams / diagonal squares for a completed input chunk."""
        if self.input is not None:
            self.input.chunk_stats()
        if self.input_diagonal is not None:
            if self._fast_cuda:
                ev = torch.cuda.Event()
                ev.record(self._main_stream)
                with torch.cuda.stream(self._input_side_stream):
                    self._input_side_stream.wait_event(ev)
                    self.input_diagonal.accum_chunk()
                    self._ev_in_diag_chunk = torch.cuda.Event()
                    self._ev_in_diag_chunk.record(self._input_side_stream)
            else:
                self.input_diagonal.accum_chunk()

    def before_microbatch_backward(self, index: int, *, chunk: int | None = None):
        """Ensure output row buffers are free right before backward writes cotangent rows."""
        if not self._fast_cuda:
            return
        if index == 0 and self._rows_guard_active:
            self._main_stream.wait_event(self._ev_rows_consumed)
            self._rows_guard_active = False
        if chunk is not None and index > 0 and index % chunk == 0 and self.output is not None:
            self.output.chunk_wait()

    def chunk_output_stats(self):
        """Accumulate partial output Grams after a chunk's last microbatch backward."""
        self._poll_lagged_nonblocking()
        if self.output is not None:
            self.output.chunk_stats()

    def select_microbatch(self, index, *, chunk: int | None = None):
        if self._fast_cuda and index == 0:
            if self._inline_diag is not None:
                self._inline_diag.zero_()
        slot = index if chunk is None else (index % chunk)
        if self._row_off is not None:
            if self._fast_cuda and slot < self._host_row_offsets.numel():
                self._row_off.copy_(self._host_row_offsets[slot], non_blocking=True)
            else:
                self._row_off.fill_(slot * self.rows_per_microbatch)
        elif self.output_at is not None:
            start = slot * self.rows_per_microbatch
            self.output_at.copy_(torch.arange(start, start + self.rows_per_microbatch, device=self.device))

        if index >= 1:
            self._poll_lagged_nonblocking()

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

        prepped_input = {}
        if fast_cuda:
            self._step_scalars_cpu[0] = group["lr"]
            self._step_scalars_cpu[1] = group["weight_decay"]
            lr_t, wd_t = self._step_scalars_cpu.to(self.device, non_blocking=True).unbind(0)
            if input_roots is not None:
                for g_name, (ri_grp, ro_grp) in input_roots.items():
                    if ri_grp.numel():
                        prepped_input[g_name] = (
                            _prepare_side_factor(ri_grp, side="right", normalize_rms=False, compiled=True),
                            _prepare_side_factor(ro_grp, side="right", normalize_rms=True, compiled=True),
                        )

        works, finite = [], []
        for shape, per_rank in self.shape_groups:
            momentum = self.momentum[shape]
            mine = per_rank[self.rank]
            if mine:
                if fast_cuda and shape in self._input_gather_spec and self._input_gather_spec[shape][0] in prepped_input:
                    g_name, g_idx = self._input_gather_spec[shape]
                    ri_grp_p, ro_grp_p = prepped_input[g_name]
                    if g_idx is None:
                        ri_prep, ro_prep = ri_grp_p, ro_grp_p
                    else:
                        ri_prep = ri_grp_p.index_select(0, g_idx)
                        ro_prep = ro_grp_p.index_select(0, g_idx)
                else:
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

                if fast_cuda:
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
                    delta = _compiled_block_out(
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
        zero_scalar = None
        if defer:
            self._last_finite_checks = finite_vec
            all_ok = finite_vec.all()
            zero_scalar = torch.zeros((), dtype=torch.float32, device=self.device)
        else:
            checks = finite_vec.tolist()
            if not all(checks):
                bad = [shape for (shape, _), ok in zip(self.shape_groups, checks, strict=True) if not ok]
                raise FloatingPointError(f"Nonfinite block update for shapes {bad}")

        for idx_w in range(len(works)):
            per_rank, work, gathered = works[idx_w]
            works[idx_w] = None
            if work is not None:
                work.wait()
            for entries, updates in zip(per_rank, gathered, strict=True):
                if entries:
                    if defer:
                        z = zero_scalar if updates.dtype == torch.float32 else torch.zeros((), dtype=updates.dtype, device=updates.device)
                        torch.where(all_ok, updates, z, out=updates)
                    torch._foreach_add_([parameter for _, _, parameter in entries], list(updates.unbind(0)))
            del gathered
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
