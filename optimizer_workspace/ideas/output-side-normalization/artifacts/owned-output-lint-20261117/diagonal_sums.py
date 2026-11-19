# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable two-stage BF16 row-square reduction for diagonal statistics."""

import torch
import triton
import triton.language as tl

ROW_TILE = 128
COLUMN_TILE = 32


@triton.jit
def partial_squares(
    pointers,
    partials,
    rows: tl.constexpr,
    width: tl.constexpr,
    row_tile: tl.constexpr,
    column_tile: tl.constexpr,
    tiles: tl.constexpr,
):
    item = tl.program_id(0)
    tile = tl.program_id(1)
    column = tl.program_id(2) * column_tile + tl.arange(0, column_tile)
    row = tile * row_tile + tl.arange(0, row_tile)
    pointer = tl.load(pointers + item).to(tl.pointer_type(tl.bfloat16))
    value = tl.load(
        pointer + row[:, None] * width + column[None, :],
        (row[:, None] < rows) & (column[None, :] < width),
        other=0,
    ).to(tl.float32)
    tl.store(
        partials + (item * tiles + tile) * width + column,
        tl.sum(value * value, axis=0),
        column < width,
    )


@triton.jit
def sum_partials(
    partials,
    output,
    width: tl.constexpr,
    tiles: tl.constexpr,
    tile_extent: tl.constexpr,
    column_tile: tl.constexpr,
):
    item = tl.program_id(0)
    column = tl.program_id(1) * column_tile + tl.arange(0, column_tile)
    tile = tl.arange(0, tile_extent)
    value = tl.load(
        partials + (item * tiles + tile[:, None]) * width + column[None, :],
        (tile[:, None] < tiles) & (column[None, :] < width),
        other=0,
    )
    tl.store(output + item * width + column, tl.sum(value, axis=0), column < width)


class RowSquares:
    """Reduce fixed contiguous BF16 buffers into reusable FP32 output on their current stream."""

    def __init__(self, buffers: list[torch.Tensor], output: torch.Tensor):
        self.buffers = tuple(buffers)
        rows, width = buffers[0].shape
        device = buffers[0].device
        if device.type != "cuda" or rows < 1 or width < 1:
            raise ValueError("Triton row squares require nonempty CUDA row matrices")
        if any(
            buffer.shape != (rows, width)
            or buffer.device != device
            or buffer.dtype != torch.bfloat16
            or not buffer.is_contiguous()
            or buffer.requires_grad
            for buffer in buffers
        ):
            raise ValueError("Row squares require same-shaped contiguous BF16 buffers on one CUDA device")
        if (
            output.shape != (len(buffers), width)
            or output.device != device
            or output.dtype != torch.float32
            or not output.is_contiguous()
            or output.requires_grad
        ):
            raise ValueError("Row squares require a contiguous FP32 output matrix on the buffer device")
        self.rows, self.width, self.output = rows, width, output
        self.pointers = torch.tensor([buffer.data_ptr() for buffer in buffers], dtype=torch.int64, device=device)
        self.tiles = triton.cdiv(rows, ROW_TILE)
        self.partials = torch.empty(len(buffers), self.tiles, width, dtype=torch.float32, device=device)

    def __call__(self) -> torch.Tensor:
        partial_squares[(len(self.buffers), self.tiles, triton.cdiv(self.width, COLUMN_TILE))](
            self.pointers,
            self.partials,
            self.rows,
            self.width,
            ROW_TILE,
            COLUMN_TILE,
            self.tiles,
            num_warps=4,
            enable_fp_fusion=False,
        )
        sum_partials[(len(self.buffers), triton.cdiv(self.width, COLUMN_TILE))](
            self.partials,
            self.output,
            self.width,
            self.tiles,
            triton.next_power_of_2(self.tiles),
            COLUMN_TILE,
            num_warps=4,
        )
        return self.output
