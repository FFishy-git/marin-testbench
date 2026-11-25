# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused Triton backward cotangent capture and inline diagonal second-moment accumulation."""

import types

import torch
import triton
import triton.language as tl
from reproduce.marin import ops as marin_ops
from reproduce.marin.model import Block as MarinBlock, norm

ROW_SCALE = 2.0**16
DIAG_SPLIT_R = 8
DIAG_BLOCK_R = 16
DIAG_BLOCK_C = 64

# Pure-Python tracker recording which InlineDiagonalAccumulator partial buffers received backward updates.
_ACTIVE_PARTIALS: set[int] = set()
_PTR_TABLES: dict[tuple[int, ...], torch.Tensor] = {}


@triton.jit
def _store_cotangent_at_kernel(
    g_ptr,
    d_ptr,
    at_ptr,
    idx_ptr,
    P,
    D,
    s_b,
    s_t,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    b = r // P
    p = r - b * P
    pos = tl.load(idx_ptr + p)
    row = tl.load(at_ptr + r)
    offs = tl.arange(0, BLOCK)
    src_base = g_ptr + b * s_b + pos * s_t
    dst_base = d_ptr + row * D
    for c in range(0, D, BLOCK):
        m = c + offs < D
        val = tl.load(src_base + c + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(dst_base + c + offs, (val * SCALE).to(tl.bfloat16), mask=m)


@triton.jit
def _store_cotangent_1_kernel(
    g0_ptr,
    d0_ptr,
    row_off_ptr,
    idx_ptr,
    P,
    D,
    s0_b,
    s0_t,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    b = r // P
    p = r - b * P
    pos = tl.load(idx_ptr + p)
    row = tl.load(row_off_ptr) + r
    offs = tl.arange(0, BLOCK)
    src_base = g0_ptr + b * s0_b + pos * s0_t
    dst_base = d0_ptr + row * D
    for c in range(0, D, BLOCK):
        m = c + offs < D
        val = tl.load(src_base + c + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(dst_base + c + offs, (val * SCALE).to(tl.bfloat16), mask=m)


@triton.jit
def _store_cotangent_2_kernel(
    g0_ptr,
    g1_ptr,
    d0_ptr,
    d1_ptr,
    row_off_ptr,
    idx_ptr,
    P,
    D,
    s0_b,
    s0_t,
    s1_b,
    s1_t,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    which = tl.program_id(1)
    b = r // P
    p = r - b * P
    pos = tl.load(idx_ptr + p)
    row = tl.load(row_off_ptr) + r
    offs = tl.arange(0, BLOCK)
    if which == 0:
        src_base = g0_ptr + b * s0_b + pos * s0_t
        dst_base = d0_ptr + row * D
    else:
        src_base = g1_ptr + b * s1_b + pos * s1_t
        dst_base = d1_ptr + row * D
    for c in range(0, D, BLOCK):
        m = c + offs < D
        val = tl.load(src_base + c + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(dst_base + c + offs, (val * SCALE).to(tl.bfloat16), mask=m)


@triton.jit
def _store_cotangent_3_kernel(
    g0_ptr,
    g1_ptr,
    g2_ptr,
    d0_ptr,
    d1_ptr,
    d2_ptr,
    row_off_ptr,
    idx_ptr,
    P,
    D,
    s0_b,
    s0_t,
    s1_b,
    s1_t,
    s2_b,
    s2_t,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    which = tl.program_id(1)
    b = r // P
    p = r - b * P
    pos = tl.load(idx_ptr + p)
    row = tl.load(row_off_ptr) + r
    offs = tl.arange(0, BLOCK)
    if which == 0:
        src_base = g0_ptr + b * s0_b + pos * s0_t
        dst_base = d0_ptr + row * D
    elif which == 1:
        src_base = g1_ptr + b * s1_b + pos * s1_t
        dst_base = d1_ptr + row * D
    else:
        src_base = g2_ptr + b * s2_b + pos * s2_t
        dst_base = d2_ptr + row * D
    for c in range(0, D, BLOCK):
        m = c + offs < D
        val = tl.load(src_base + c + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(dst_base + c + offs, (val * SCALE).to(tl.bfloat16), mask=m)


@triton.jit
def _accum_diag_1_kernel(
    g0_ptr,
    partial_ptr,
    idx_ptr,
    B,
    P,
    D,
    s0_b,
    s0_t,
    slot_stride,
    split_stride,
    slot_base,
    SCALE: tl.constexpr,
    SPLIT_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_tile = tl.program_id(0)
    s_id = tl.program_id(1)
    total_r = B * P
    chunk = tl.cdiv(total_r, SPLIT_R)
    r_start = s_id * chunk
    r_end = tl.minimum(r_start + chunk, total_r)
    offs_r = tl.arange(0, BLOCK_R)
    cols = c_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < D
    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for r0 in range(r_start, r_end, BLOCK_R):
        rr = r0 + offs_r
        rmask = rr < r_end
        b = rr // P
        p = rr - b * P
        pos = tl.load(idx_ptr + p, mask=rmask, other=0)
        ptrs = g0_ptr + b[:, None] * s0_b + pos[:, None] * s0_t + cols[None, :]
        val = tl.load(ptrs, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        scaled = (val * SCALE).to(tl.bfloat16).to(tl.float32)
        acc += scaled * scaled
    col_sum = tl.sum(acc, axis=0)
    out_ptr = partial_ptr + slot_base * slot_stride + s_id * split_stride + cols
    prev = tl.load(out_ptr, mask=cmask, other=0.0)
    tl.store(out_ptr, prev + col_sum, mask=cmask)


@triton.jit
def _accum_diag_2_kernel(
    g0_ptr,
    g1_ptr,
    partial_ptr,
    idx_ptr,
    B,
    P,
    D,
    s0_b,
    s0_t,
    s1_b,
    s1_t,
    slot_stride,
    split_stride,
    slot_base,
    SCALE: tl.constexpr,
    SPLIT_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_tile = tl.program_id(0)
    s_id = tl.program_id(1)
    which = tl.program_id(2)
    total_r = B * P
    chunk = tl.cdiv(total_r, SPLIT_R)
    r_start = s_id * chunk
    r_end = tl.minimum(r_start + chunk, total_r)
    offs_r = tl.arange(0, BLOCK_R)
    cols = c_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < D
    if which == 0:
        g_ptr = g0_ptr
        s_b = s0_b
        s_t = s0_t
    else:
        g_ptr = g1_ptr
        s_b = s1_b
        s_t = s1_t
    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for r0 in range(r_start, r_end, BLOCK_R):
        rr = r0 + offs_r
        rmask = rr < r_end
        b = rr // P
        p = rr - b * P
        pos = tl.load(idx_ptr + p, mask=rmask, other=0)
        ptrs = g_ptr + b[:, None] * s_b + pos[:, None] * s_t + cols[None, :]
        val = tl.load(ptrs, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        scaled = (val * SCALE).to(tl.bfloat16).to(tl.float32)
        acc += scaled * scaled
    col_sum = tl.sum(acc, axis=0)
    out_ptr = partial_ptr + (slot_base + which) * slot_stride + s_id * split_stride + cols
    prev = tl.load(out_ptr, mask=cmask, other=0.0)
    tl.store(out_ptr, prev + col_sum, mask=cmask)


@triton.jit
def _accum_diag_3_kernel(
    g0_ptr,
    g1_ptr,
    g2_ptr,
    partial_ptr,
    idx_ptr,
    B,
    P,
    D,
    s0_b,
    s0_t,
    s1_b,
    s1_t,
    s2_b,
    s2_t,
    slot_stride,
    split_stride,
    slot_base,
    SCALE: tl.constexpr,
    SPLIT_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_tile = tl.program_id(0)
    s_id = tl.program_id(1)
    which = tl.program_id(2)
    total_r = B * P
    chunk = tl.cdiv(total_r, SPLIT_R)
    r_start = s_id * chunk
    r_end = tl.minimum(r_start + chunk, total_r)
    offs_r = tl.arange(0, BLOCK_R)
    cols = c_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < D
    if which == 0:
        g_ptr = g0_ptr
        s_b = s0_b
        s_t = s0_t
    elif which == 1:
        g_ptr = g1_ptr
        s_b = s1_b
        s_t = s1_t
    else:
        g_ptr = g2_ptr
        s_b = s2_b
        s_t = s2_t
    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for r0 in range(r_start, r_end, BLOCK_R):
        rr = r0 + offs_r
        rmask = rr < r_end
        b = rr // P
        p = rr - b * P
        pos = tl.load(idx_ptr + p, mask=rmask, other=0)
        ptrs = g_ptr + b[:, None] * s_b + pos[:, None] * s_t + cols[None, :]
        val = tl.load(ptrs, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        scaled = (val * SCALE).to(tl.bfloat16).to(tl.float32)
        acc += scaled * scaled
    col_sum = tl.sum(acc, axis=0)
    out_ptr = partial_ptr + (slot_base + which) * slot_stride + s_id * split_stride + cols
    prev = tl.load(out_ptr, mask=cmask, other=0.0)
    tl.store(out_ptr, prev + col_sum, mask=cmask)


@triton.jit
def _buf_row_sq_kernel(
    ptrs_ptr,
    partial_ptr,
    R,
    D,
    SPLIT_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    k_id = tl.program_id(0)
    s_id = tl.program_id(1)
    c_tile = tl.program_id(2)
    src_ptr = tl.load(ptrs_ptr + k_id).to(tl.pointer_type(tl.bfloat16))
    chunk = tl.cdiv(R, SPLIT_R)
    r_start = s_id * chunk
    r_end = tl.minimum(r_start + chunk, R)
    offs_r = tl.arange(0, BLOCK_R)
    cols = c_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < D
    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for r0 in range(r_start, r_end, BLOCK_R):
        rr = r0 + offs_r
        rmask = rr < r_end
        val = tl.load(src_ptr + rr[:, None] * D + cols[None, :], mask=rmask[:, None] & cmask[None, :], other=0.0).to(
            tl.float32
        )
        acc += val * val
    out_idx = (k_id * SPLIT_R + s_id) * D + cols
    tl.store(partial_ptr + out_idx, tl.sum(acc, axis=0), mask=cmask)


def reduce_row_squares(buffers: list[torch.Tensor], scratch: torch.Tensor | None = None) -> torch.Tensor:
    """Fast deterministic FP32 sum-of-squares across a list of (R, D) BF16 row buffers."""
    b0 = buffers[0]
    if not b0.is_cuda or b0.dtype != torch.bfloat16 or not all(b.is_contiguous() for b in buffers):
        return torch.stack([b.float().square().sum(0) for b in buffers])
    K, (R, D) = len(buffers), b0.shape
    if R == 0:
        return torch.zeros(K, D, dtype=torch.float32, device=b0.device)
    key = tuple(b.data_ptr() for b in buffers)
    ptrs = _PTR_TABLES.get(key)
    if ptrs is None:
        if len(_PTR_TABLES) >= 64:
            _PTR_TABLES.clear()
        ptrs = torch.tensor(key, dtype=torch.int64).pin_memory().to(b0.device, non_blocking=True)
        _PTR_TABLES[key] = ptrs
    if scratch is None or scratch.shape != (K, DIAG_SPLIT_R, D):
        scratch = torch.empty(K, DIAG_SPLIT_R, D, dtype=torch.float32, device=b0.device)
    grid = (K, DIAG_SPLIT_R, triton.cdiv(D, DIAG_BLOCK_C))
    _buf_row_sq_kernel[grid](
        ptrs, scratch, R, D, SPLIT_R=DIAG_SPLIT_R, BLOCK_R=DIAG_BLOCK_R, BLOCK_C=DIAG_BLOCK_C, num_warps=4
    )
    return scratch.sum(dim=1)


def _ensure_last_contig(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.bfloat16:
        raise TypeError(f"Output cotangent rows require BF16 projection gradients, got {t.dtype}")
    return t if t.stride(-1) == 1 else t.contiguous()


def _cpu_store_one(g: torch.Tensor, dst: torch.Tensor, row_off: torch.Tensor, idx: torch.Tensor) -> None:
    if g.dtype != torch.bfloat16:
        raise TypeError(f"Output cotangent rows require BF16 projection gradients, got {g.dtype}")
    sampled = g.index_select(1, idx).reshape(-1, g.shape[-1]) * ROW_SCALE
    start = int(row_off.item())
    dst[start : start + sampled.shape[0]] = sampled


def _cpu_accum_diag_one(g: torch.Tensor, partial: torch.Tensor, slot: int, idx: torch.Tensor) -> None:
    if g.dtype != torch.bfloat16:
        raise TypeError(f"Output cotangent rows require BF16 projection gradients, got {g.dtype}")
    sampled = (g.index_select(1, idx).reshape(-1, g.shape[-1]) * ROW_SCALE).float()
    partial[slot, 0].add_(sampled.square().sum(0))


@torch.library.custom_op("output_marin::store_cotangent_1", mutates_args=("d0",))
def store_cotangent_1(
    g0: torch.Tensor,
    d0: torch.Tensor,
    row_off: torch.Tensor,
    idx: torch.Tensor,
) -> None:
    if not g0.is_cuda:
        _cpu_store_one(g0, d0, row_off, idx)
        return
    g0 = _ensure_last_contig(g0)
    P, D = idx.numel(), g0.shape[-1]
    block = min(512, triton.next_power_of_2(max(16, D)))
    _store_cotangent_1_kernel[(g0.shape[0] * P,)](
        g0, d0, row_off, idx, P, D, g0.stride(0), g0.stride(1), SCALE=ROW_SCALE, BLOCK=block, num_warps=4
    )


@store_cotangent_1.register_fake
def _(g0, d0, row_off, idx):
    return None


@torch.library.custom_op("output_marin::store_cotangent_2", mutates_args=("d0", "d1"))
def store_cotangent_2(
    g0: torch.Tensor,
    g1: torch.Tensor,
    d0: torch.Tensor,
    d1: torch.Tensor,
    row_off: torch.Tensor,
    idx: torch.Tensor,
) -> None:
    if not g0.is_cuda:
        _cpu_store_one(g0, d0, row_off, idx)
        _cpu_store_one(g1, d1, row_off, idx)
        return
    g0, g1 = _ensure_last_contig(g0), _ensure_last_contig(g1)
    P, D = idx.numel(), g0.shape[-1]
    block = min(512, triton.next_power_of_2(max(16, D)))
    _store_cotangent_2_kernel[(g0.shape[0] * P, 2)](
        g0,
        g1,
        d0,
        d1,
        row_off,
        idx,
        P,
        D,
        g0.stride(0),
        g0.stride(1),
        g1.stride(0),
        g1.stride(1),
        SCALE=ROW_SCALE,
        BLOCK=block,
        num_warps=4,
    )


@store_cotangent_2.register_fake
def _(g0, g1, d0, d1, row_off, idx):
    return None


@torch.library.custom_op("output_marin::store_cotangent_3", mutates_args=("d0", "d1", "d2"))
def store_cotangent_3(
    g0: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    d0: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    row_off: torch.Tensor,
    idx: torch.Tensor,
) -> None:
    if not g0.is_cuda:
        _cpu_store_one(g0, d0, row_off, idx)
        _cpu_store_one(g1, d1, row_off, idx)
        _cpu_store_one(g2, d2, row_off, idx)
        return
    g0, g1, g2 = _ensure_last_contig(g0), _ensure_last_contig(g1), _ensure_last_contig(g2)
    P, D = idx.numel(), g0.shape[-1]
    block = min(512, triton.next_power_of_2(max(16, D)))
    _store_cotangent_3_kernel[(g0.shape[0] * P, 3)](
        g0,
        g1,
        g2,
        d0,
        d1,
        d2,
        row_off,
        idx,
        P,
        D,
        g0.stride(0),
        g0.stride(1),
        g1.stride(0),
        g1.stride(1),
        g2.stride(0),
        g2.stride(1),
        SCALE=ROW_SCALE,
        BLOCK=block,
        num_warps=4,
    )


@store_cotangent_3.register_fake
def _(g0, g1, g2, d0, d1, d2, row_off, idx):
    return None


@torch.library.custom_op("output_marin::accum_diag_1", mutates_args=("partial",))
def accum_diag_1(
    g0: torch.Tensor,
    partial: torch.Tensor,
    idx: torch.Tensor,
    slot_base: int,
) -> None:
    _ACTIVE_PARTIALS.add(partial.data_ptr())
    if not g0.is_cuda:
        _cpu_accum_diag_one(g0, partial, slot_base, idx)
        return
    g0 = _ensure_last_contig(g0)
    B, P, D = g0.shape[0], idx.numel(), g0.shape[-1]
    grid = (triton.cdiv(D, DIAG_BLOCK_C), DIAG_SPLIT_R)
    _accum_diag_1_kernel[grid](
        g0,
        partial,
        idx,
        B,
        P,
        D,
        g0.stride(0),
        g0.stride(1),
        partial.stride(0),
        partial.stride(1),
        slot_base,
        SCALE=ROW_SCALE,
        SPLIT_R=DIAG_SPLIT_R,
        BLOCK_R=DIAG_BLOCK_R,
        BLOCK_C=DIAG_BLOCK_C,
        num_warps=4,
    )


@accum_diag_1.register_fake
def _(g0, partial, idx, slot_base):
    return None


@torch.library.custom_op("output_marin::accum_diag_2", mutates_args=("partial",))
def accum_diag_2(
    g0: torch.Tensor,
    g1: torch.Tensor,
    partial: torch.Tensor,
    idx: torch.Tensor,
    slot_base: int,
) -> None:
    _ACTIVE_PARTIALS.add(partial.data_ptr())
    if not g0.is_cuda:
        _cpu_accum_diag_one(g0, partial, slot_base, idx)
        _cpu_accum_diag_one(g1, partial, slot_base + 1, idx)
        return
    g0, g1 = _ensure_last_contig(g0), _ensure_last_contig(g1)
    B, P, D = g0.shape[0], idx.numel(), g0.shape[-1]
    grid = (triton.cdiv(D, DIAG_BLOCK_C), DIAG_SPLIT_R, 2)
    _accum_diag_2_kernel[grid](
        g0,
        g1,
        partial,
        idx,
        B,
        P,
        D,
        g0.stride(0),
        g0.stride(1),
        g1.stride(0),
        g1.stride(1),
        partial.stride(0),
        partial.stride(1),
        slot_base,
        SCALE=ROW_SCALE,
        SPLIT_R=DIAG_SPLIT_R,
        BLOCK_R=DIAG_BLOCK_R,
        BLOCK_C=DIAG_BLOCK_C,
        num_warps=4,
    )


@accum_diag_2.register_fake
def _(g0, g1, partial, idx, slot_base):
    return None


@torch.library.custom_op("output_marin::accum_diag_3", mutates_args=("partial",))
def accum_diag_3(
    g0: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    partial: torch.Tensor,
    idx: torch.Tensor,
    slot_base: int,
) -> None:
    _ACTIVE_PARTIALS.add(partial.data_ptr())
    if not g0.is_cuda:
        _cpu_accum_diag_one(g0, partial, slot_base, idx)
        _cpu_accum_diag_one(g1, partial, slot_base + 1, idx)
        _cpu_accum_diag_one(g2, partial, slot_base + 2, idx)
        return
    g0, g1, g2 = _ensure_last_contig(g0), _ensure_last_contig(g1), _ensure_last_contig(g2)
    B, P, D = g0.shape[0], idx.numel(), g0.shape[-1]
    grid = (triton.cdiv(D, DIAG_BLOCK_C), DIAG_SPLIT_R, 3)
    _accum_diag_3_kernel[grid](
        g0,
        g1,
        g2,
        partial,
        idx,
        B,
        P,
        D,
        g0.stride(0),
        g0.stride(1),
        g1.stride(0),
        g1.stride(1),
        g2.stride(0),
        g2.stride(1),
        partial.stride(0),
        partial.stride(1),
        slot_base,
        SCALE=ROW_SCALE,
        SPLIT_R=DIAG_SPLIT_R,
        BLOCK_R=DIAG_BLOCK_R,
        BLOCK_C=DIAG_BLOCK_C,
        num_warps=4,
    )


@accum_diag_3.register_fake
def _(g0, g1, g2, partial, idx, slot_base):
    return None


class _ObserveRows1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, d0, row_off, idx):
        ctx.d0 = d0
        ctx.save_for_backward(row_off, idx)
        return y0.view_as(y0)

    @staticmethod
    def backward(ctx, g0):
        row_off, idx = ctx.saved_tensors
        store_cotangent_1(g0, ctx.d0, row_off, idx)
        return g0, None, None, None


class _ObserveRows2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, y1, d0, d1, row_off, idx):
        ctx.d0, ctx.d1 = d0, d1
        ctx.save_for_backward(row_off, idx)
        return y0.view_as(y0), y1.view_as(y1)

    @staticmethod
    def backward(ctx, g0, g1):
        row_off, idx = ctx.saved_tensors
        store_cotangent_2(g0, g1, ctx.d0, ctx.d1, row_off, idx)
        return g0, g1, None, None, None, None


class _ObserveRows3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, y1, y2, d0, d1, d2, row_off, idx):
        ctx.d0, ctx.d1, ctx.d2 = d0, d1, d2
        ctx.save_for_backward(row_off, idx)
        return y0.view_as(y0), y1.view_as(y1), y2.view_as(y2)

    @staticmethod
    def backward(ctx, g0, g1, g2):
        row_off, idx = ctx.saved_tensors
        store_cotangent_3(g0, g1, g2, ctx.d0, ctx.d1, ctx.d2, row_off, idx)
        return g0, g1, g2, None, None, None, None, None


class _ObserveDiag1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, partial, idx, slot_base):
        ctx.partial = partial
        ctx.slot_base = slot_base
        ctx.save_for_backward(idx)
        return y0.view_as(y0)

    @staticmethod
    def backward(ctx, g0):
        (idx,) = ctx.saved_tensors
        accum_diag_1(g0, ctx.partial, idx, ctx.slot_base)
        return g0, None, None, None


class _ObserveDiag2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, y1, partial, idx, slot_base):
        ctx.partial = partial
        ctx.slot_base = slot_base
        ctx.save_for_backward(idx)
        return y0.view_as(y0), y1.view_as(y1)

    @staticmethod
    def backward(ctx, g0, g1):
        (idx,) = ctx.saved_tensors
        accum_diag_2(g0, g1, ctx.partial, idx, ctx.slot_base)
        return g0, g1, None, None, None


class _ObserveDiag3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, y1, y2, partial, idx, slot_base):
        ctx.partial = partial
        ctx.slot_base = slot_base
        ctx.save_for_backward(idx)
        return y0.view_as(y0), y1.view_as(y1), y2.view_as(y2)

    @staticmethod
    def backward(ctx, g0, g1, g2):
        (idx,) = ctx.saved_tensors
        accum_diag_3(g0, g1, g2, ctx.partial, idx, ctx.slot_base)
        return g0, g1, g2, None, None, None


class InlineDiagonalAccumulator:
    """Zero-row-buffer inline FP32 cotangent-square accumulator with independent storage per observer site.

    Independent storage per block observer site prevents one DDP bucket's compiled backward partition
    from incrementing the version counter of a buffer saved by an earlier-layer DDP bucket partition.
    """

    def __init__(self, keys_by_width: dict[int, list[tuple[int, str]]], device):
        self.keys_by_width = {w: list(ks) for w, ks in keys_by_width.items() if ks}
        self.site_buffers: dict[int, list[torch.Tensor]] = {}
        self.site_of: dict[tuple[int, str], torch.Tensor] = {}
        for w, ks in self.keys_by_width.items():
            bufs = []
            idx = 0
            while idx < len(ks):
                layer, kind = ks[idx]
                if kind == "q" and idx + 2 < len(ks) and ks[idx + 1] == (layer, "k") and ks[idx + 2] == (layer, "v"):
                    buf = torch.zeros(3, DIAG_SPLIT_R, w, dtype=torch.float32, device=device)
                    self.site_of[layer, "qkv"] = buf
                    bufs.append(buf)
                    idx += 3
                elif kind == "gate" and idx + 1 < len(ks) and ks[idx + 1] == (layer, "up"):
                    buf = torch.zeros(2, DIAG_SPLIT_R, w, dtype=torch.float32, device=device)
                    self.site_of[layer, "gate_up"] = buf
                    bufs.append(buf)
                    idx += 2
                else:
                    buf = torch.zeros(1, DIAG_SPLIT_R, w, dtype=torch.float32, device=device)
                    self.site_of[layer, kind] = buf
                    bufs.append(buf)
                    idx += 1
            self.site_buffers[w] = bufs

    def zero_(self):
        for bufs in self.site_buffers.values():
            for buf in bufs:
                _ACTIVE_PARTIALS.discard(buf.data_ptr())
            if bufs:
                torch._foreach_zero_(bufs)

    def is_active(self, width: int) -> bool:
        return width in self.site_buffers and any(b.data_ptr() in _ACTIVE_PARTIALS for b in self.site_buffers[width])

    def sums(self, width: int) -> torch.Tensor:
        return torch.cat(self.site_buffers[width], dim=0).sum(dim=1)


class LazyRowDict(dict):
    """Row dictionary that keeps diagonal-only buffers unallocated in production, materializing only on direct test access."""

    def __init__(self, eager: dict, specs: dict[tuple[int, str], tuple[int, int, torch.device, object]]):
        super().__init__(eager)
        self._specs = specs

    def _materialize(self, key):
        if not super().__contains__(key) and key in self._specs:
            capacity, width, device, linear = self._specs[key]
            buf = torch.zeros(capacity, width, dtype=torch.bfloat16, device=device)
            super().__setitem__(key, buf)
            object.__setattr__(linear, "output_rows", buf)
        return super().__getitem__(key)

    def _materialize_all(self):
        for key in self._specs:
            self._materialize(key)

    def __contains__(self, key):
        return super().__contains__(key) or key in self._specs

    def __len__(self):
        return len(set(super().keys()) | set(self._specs.keys()))

    def __iter__(self):
        seen = set()
        for key in list(super().keys()) + list(self._specs.keys()):
            if key not in seen:
                seen.add(key)
                yield key

    def __getitem__(self, key):
        return self._materialize(key)

    def get(self, key, default=None):
        if key in self:
            return self._materialize(key)
        return default

    def keys(self):
        return list(iter(self))

    def items(self):
        self._materialize_all()
        return super().items()

    def values(self):
        self._materialize_all()
        return super().values()


def _make_observed_block_forward(
    hidden_mode: str,
    mlp_mode: str,
    layer_rows: dict[str, torch.Tensor],
    inline_diag: InlineDiagonalAccumulator | None,
    layer: int,
    row_off: torch.Tensor,
    idx: torch.Tensor,
    *,
    down_only: bool = False,
):
    q_buf = layer_rows.get("q")
    k_buf = layer_rows.get("k")
    v_buf = layer_rows.get("v")
    o_buf = layer_rows.get("o")
    g_buf = layer_rows.get("gate")
    u_buf = layer_rows.get("up")
    d_buf = layer_rows.get("down")

    qkv_mode = "none" if down_only else hidden_mode
    o_mode = "none" if down_only else hidden_mode
    mlp_eff_mode = "none" if down_only else mlp_mode
    down_mode = hidden_mode

    qkv_partial = o_partial = d_partial = None
    if down_mode == "diagonal" and inline_diag is not None:
        d_partial = inline_diag.site_of[layer, "down"]
    if qkv_mode == "diagonal" and inline_diag is not None:
        qkv_partial = inline_diag.site_of[layer, "qkv"]
        o_partial = inline_diag.site_of[layer, "o"]

    mlp_partial = None
    if mlp_eff_mode == "diagonal" and inline_diag is not None:
        mlp_partial = inline_diag.site_of[layer, "gate_up"]

    def forward(self, x, cos, sin, cu, meta, captures=None):
        c = self.config
        B, T, H = x.shape
        heads, hd = c["heads"], H // c["heads"]
        bf16_res = c["residual_dtype"] == "bf16"
        x_in = x
        h = norm(self.ln1, x, c["residual_dtype"])
        ql, kl, vl = self.q(h), self.k(h), self.v(h)
        if qkv_mode == "full":
            ql, kl, vl = _ObserveRows3.apply(ql, kl, vl, q_buf, k_buf, v_buf, row_off, idx)
        elif qkv_mode == "diagonal":
            ql, kl, vl = _ObserveDiag3.apply(ql, kl, vl, qkv_partial, idx, 0)
        q, k, v = ql.view(B, T, heads, hd), kl.view(B, T, heads, hd), vl.view(B, T, heads, hd)

        def rotate(t):
            return t * cos + torch.cat((-t[..., hd // 2 :], t[..., : hd // 2]), -1) * sin

        q, k, v = rotate(q).to(torch.bfloat16), rotate(k).to(torch.bfloat16), v.to(torch.bfloat16)
        a, _ = marin_ops.doc_attn(
            q.reshape(B * T, heads, hd),
            k.reshape(B * T, heads, hd),
            v.reshape(B * T, heads, hd),
            cu,
            meta,
        )
        a = a.view(B, T, H)
        yo = self.o(a)
        if o_mode == "full":
            if bf16_res:
                x = _ObserveRows1.apply(x + yo, o_buf, row_off, idx)
            else:
                x = x + _ObserveRows1.apply(yo, o_buf, row_off, idx)
        elif o_mode == "diagonal":
            if bf16_res:
                x = _ObserveDiag1.apply(x + yo, o_partial, idx, 0)
            else:
                x = x + _ObserveDiag1.apply(yo, o_partial, idx, 0)
        else:
            x = x + yo
        h2 = norm(self.ln2, x, c["residual_dtype"])
        g, u = self.gate(h2), self.up(h2)
        if mlp_eff_mode == "full":
            g, u = _ObserveRows2.apply(g, u, g_buf, u_buf, row_off, idx)
        elif mlp_eff_mode == "diagonal":
            g, u = _ObserveDiag2.apply(g, u, mlp_partial, idx, 0)
        m = torch.nn.functional.silu(g) * u
        if captures is not None:
            captures.append(tuple(t.detach() for t in (x_in, a, x, g, u)))
        yd = self.down(m)
        if down_mode == "full":
            if bf16_res:
                return _ObserveRows1.apply(x + yd, d_buf, row_off, idx)
            return x + _ObserveRows1.apply(yd, d_buf, row_off, idx)
        if down_mode == "diagonal":
            if bf16_res:
                return _ObserveDiag1.apply(x + yd, d_partial, idx, 0)
            return x + _ObserveDiag1.apply(yd, d_partial, idx, 0)
        return x + yd

    return forward


def attach_fast_block_observers(
    blocks,
    *,
    hidden_mode: str,
    mlp_mode: str,
    positions,
    capacity: int,
    device,
    down_only: bool = False,
):
    """Attach grouped Triton cotangent observers at the Block level without wrapping nn.Linear."""
    if not all(isinstance(b, MarinBlock) for b in blocks):
        return None
    idx = torch.tensor(positions, dtype=torch.int64, device=device)
    row_off = torch.zeros((), dtype=torch.int64, device=device)
    eager_rows = {}
    lazy_specs = {}
    diag_keys_by_width: dict[int, list[tuple[int, str]]] = {}

    hidden_kinds = ("down",) if down_only else ("q", "k", "v", "o", "down")
    mlp_kinds = () if down_only else ("gate", "up")

    for layer, block in enumerate(blocks):
        for kind in hidden_kinds:
            linear = getattr(block, kind)
            width = linear.weight.shape[0]
            if hidden_mode == "full":
                buf = torch.zeros(capacity, width, dtype=torch.bfloat16, device=device)
                eager_rows[layer, kind] = buf
                object.__setattr__(linear, "output_rows", buf)
            elif hidden_mode == "diagonal":
                lazy_specs[layer, kind] = (capacity, width, device, linear)
                diag_keys_by_width.setdefault(width, []).append((layer, kind))
        for kind in mlp_kinds:
            linear = getattr(block, kind)
            width = linear.weight.shape[0]
            if mlp_mode == "full":
                buf = torch.zeros(capacity, width, dtype=torch.bfloat16, device=device)
                eager_rows[layer, kind] = buf
                object.__setattr__(linear, "output_rows", buf)
            elif mlp_mode == "diagonal":
                lazy_specs[layer, kind] = (capacity, width, device, linear)
                diag_keys_by_width.setdefault(width, []).append((layer, kind))

    inline_diag = InlineDiagonalAccumulator(diag_keys_by_width, device) if diag_keys_by_width else None

    for layer, block in enumerate(blocks):
        layer_rows = {kind: eager_rows[layer, kind] for (l_idx, kind) in eager_rows if l_idx == layer}
        fn = _make_observed_block_forward(
            hidden_mode,
            mlp_mode,
            layer_rows,
            inline_diag,
            layer,
            row_off,
            idx,
            down_only=down_only,
        )
        object.__setattr__(block, "forward", types.MethodType(fn, block))

    rows = LazyRowDict(eager_rows, lazy_specs) if lazy_specs else eager_rows
    return rows, row_off, idx, inline_diag
