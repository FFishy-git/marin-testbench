# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable A100/NCCL checks on labeled synthetic tensors, before real-data smoke."""

import argparse
import copy
import gc
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from reproduce.marin import ops
from reproduce.marin.data import pack
from reproduce.marin.model import Model, initialize_model, sample_positions
from reproduce.marin.optimizer import fast_arguments
from toner.fast import muon as MU
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .block_toner import (
    BLOCK_ARMS,
    BLOCK_KINDS,
    ROW_SCALE,
    BlockToner,
    DiagonalMoments,
    MatrixFactors,
    RowObservedLinear,
    block_update,
    block_update_stack,
    combined_outer_update_stack,
)
from .bootstrap import verify_snapshot
from .capture import OutputLinear, OutputMoments, build_model
from .configure import configuration
from .diagonal_sums import RowSquares
from .execution_owner import require_synthetic_owner
from .factors import OutputSettings, inverse_factors
from .launch import WAVE
from .optimizer import (
    DownToner,
    accurate_left,
    accurate_right,
    build,
    down_update,
    fast_runtime_state,
    restore_fast,
    toner_engines,
)
from .outer_diagonal_checks import compare_outer_update, fp64_diagonal_factors
from .upstream_full_model_checks import test_upstream_product_updates, test_upstream_scalar_step
from .records import atomic_json, compare_tree, sha256, utc_now
from .reference import TASK


def test_collector(device):
    torch.manual_seed(119)
    sums = torch.zeros(256, 256, device=device)
    count = torch.zeros((), dtype=torch.int64, device=device)
    positions = torch.tensor([0, 2, 7, 15], device=device)
    scale = torch.tensor(64.0, device=device)
    linear = nn.Linear(256, 256, bias=False).to(device)
    plain = copy.deepcopy(linear)
    observed = torch.compile(
        OutputLinear(linear, sums, count, positions, scale),
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
    )
    # Match compiled BF16 weight-gradient accumulation on the unobserved path.
    unobserved = torch.compile(plain, fullgraph=True, mode="max-autotune-no-cudagraphs")
    expected_sum = torch.zeros_like(sums)
    for _ in range(2):
        x = torch.randn(2, 16, 256, device=device)
        target = torch.randn(2, 16, 256, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = observed(x)
            reference = unobserved(x)
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-5)
        ((value.float() - target).square().sum() / scale).backward()
        ((reference.float() - target).square().sum() / scale).backward()
        # Gradients through BF16 output are rounded before the collector sees them.
        delta = (2 * (reference.detach().float() - target) / scale).to(torch.bfloat16).float() * scale
        rows = delta[:, positions].reshape(-1, 256)
        expected_sum += rows.T @ rows
    torch.testing.assert_close(linear.weight.grad, plain.weight.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sums, expected_sum, rtol=1e-5, atol=1e-3)
    assert count.item() == 16
    observed(torch.zeros(2, 16, 256, device=device)).detach()
    assert count.item() == 16


def test_distributed_moments(device, world, rank):
    model = nn.Module()
    model.body = nn.Module()
    model.body.blocks = nn.ModuleList([nn.Module() for _ in range(world)])
    for block in model.body.blocks:
        block.down = nn.Linear(8, 8, bias=False).to(device)
        with torch.no_grad():
            block.down.weight.fill_(0.125)
    output = OutputMoments(
        model,
        [0, 2],
        OutputSettings(momentum=0.5),
        world=world,
        rank=rank,
        device=device,
    )
    output.begin(32)
    x = torch.ones(rank + 1, 3, 8, device=device) * (rank + 1)
    for block in model.body.blocks:
        (block.down(x).square().sum() / 32).backward()
    output.update(2 * (rank + 1))
    numerator = sum(2 * (r + 1) * (2 * (r + 1)) ** 2 for r in range(world))
    denominator = sum(2 * (r + 1) for r in range(world))
    expected = torch.full_like(output.moment, 0.5 * numerator / denominator)
    torch.testing.assert_close(output.moment, expected, rtol=1e-6, atol=1e-6)


def test_attention(device):
    torch.manual_seed(22)
    q, k, v = [torch.randn(16, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    cu = torch.tensor([0, 6, 16, 16, 16], dtype=torch.int32, device=device)
    meta = torch.tensor([2, 10], dtype=torch.int64)
    actual, _ = ops.doc_attn(q, k, v, cu, meta)
    chunks = []
    with sdpa_kernel(SDPBackend.MATH):
        for start, end in ((0, 6), (6, 16)):
            chunks.append(
                F.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1).unsqueeze(0),
                    k[start:end].transpose(0, 1).unsqueeze(0),
                    v[start:end].transpose(0, 1).unsqueeze(0),
                    is_causal=True,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
    expected = torch.cat(chunks)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    gradient = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, (q, k, v), gradient, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (q, k, v), gradient)
    for actual_value, expected_value in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=0.02, atol=0.02)


def test_microbatch_partition(device):
    torch.manual_seed(43)
    x = torch.randn(16, 16, 256, device=device)
    cotangent = torch.randn_like(x).to(torch.bfloat16)
    weight = torch.randn(256, 256, device=device) * 0.05
    results = []
    for micro in (8, 4):
        linear = nn.Linear(256, 256, bias=False).to(device)
        with torch.no_grad():
            linear.weight.copy_(weight)
        sums = torch.zeros(256, 256, device=device)
        count = torch.zeros((), dtype=torch.int64, device=device)
        positions = torch.tensor([0, 2, 7, 15], device=device)
        accumulation = 16 // micro
        scale = torch.tensor(float(16 * 16), device=device)
        observed = torch.compile(
            OutputLinear(linear, sums, count, positions, scale),
            fullgraph=True,
            mode="max-autotune-no-cudagraphs",
        )
        for start in range(0, 16, micro):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = observed(x[start : start + micro])
            loss = (value.float() * cotangent[start : start + micro].float()).sum() / (micro * 16)
            (loss / accumulation).backward()
        results.append((linear.weight.grad.clone(), sums.clone(), count.clone()))
    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-3)
    rows = cotangent[:, positions].reshape(-1, 256).float()
    torch.testing.assert_close(results[1][1], rows.mT @ rows, rtol=1e-5, atol=1e-3)
    assert results[1][2].item() == 64


def test_down_matrix_updates(device):
    torch.manual_seed(31)
    config = configuration("down_hybrid", "0p5m", 0, 8)
    args = fast_arguments(config)
    weight = torch.randn(2, 256, 512, device=device) * 0.05
    gradient = torch.randn_like(weight)
    momentum = torch.randn_like(weight)
    left = torch.randn(2, 256, 64, device=device)
    right = torch.randn(2, 512, 64, device=device)
    li, lo = inverse_factors(left @ left.mT, OutputSettings(damping=0.01))
    ri, ro = inverse_factors(right @ right.mT, OutputSettings(damping=0.01))
    for mode in ("f32", config["optimizer"]["ns_mode"]):
        args.muon_ns_mode = mode
        for arm in ("outer_left_down", "inner_outer_down", "down_hybrid"):
            actual_m, delta = down_update(
                weight,
                gradient,
                momentum,
                arm=arm,
                left_inner=li,
                left_outer=lo,
                right_inner=ri,
                right_outer=ro,
                args=args,
                lr=0.02,
                weight_decay=0.1,
                right_product=accurate_right,
                left_product=accurate_left,
            )
            expected_m = args.muon_momentum * momentum + gradient
            source = gradient + args.muon_momentum * expected_m
            if arm == "down_hybrid":
                source = li @ source
                decay = lo @ lo @ weight
            else:
                source = source @ ri
                if arm == "inner_outer_down":
                    source = li @ source
                ron = ro * (512**0.5 / ro.norm(dim=(-2, -1), keepdim=True))
                decay = weight @ ron @ ron
            # Use the upstream per-matrix routine, not the candidate's batched NS.
            orthogonal = torch.stack([MU.newton_schulz(matrix, 10, 1e-5, mode) for matrix in source])
            direction = lo @ orthogonal
            if arm != "down_hybrid":
                direction = direction @ ron
            torch.testing.assert_close(actual_m, expected_m, rtol=0, atol=0)
            torch.testing.assert_close(
                delta,
                -0.02 * (direction + 0.1 * decay),
                rtol=0.003,
                atol=5e-5,
                msg=f"{arm}.{mode}",
            )


def test_compiled_body(device):
    config = configuration("down_hybrid", "0p5m", 0, 1)
    config["model"].update(hidden=256, mlp=1024, heads=4, layers=2, sequence_length=64, vocab_size=256)
    positions = sample_positions(64, 8, 8)
    rows = len(positions)
    candidate = build_model(
        config,
        positions=positions,
        row_capacity=rows,
        rows_per_microbatch=rows,
        device=device,
    )
    initialize_model(candidate, 0)
    reference = Model(config["model"]).to(device)
    reference.load_state_dict(candidate.state_dict())
    output = OutputMoments(candidate, positions, OutputSettings(), world=1, rank=0, device=device)
    output.begin(64)
    compiled = torch.compile(candidate.body, mode="max-autotune-no-cudagraphs", fullgraph=True)
    pinned_body = torch.compile(reference.body, mode="max-autotune-no-cudagraphs", fullgraph=True)
    tokens = np.arange(64, dtype=np.uint32).reshape(1, 64)
    tok, _, cu, meta = pack(tokens, 34, EOS=128001)
    tok, cu = tok.to(device), cu.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = compiled(tok, cu, meta)
        expected = pinned_body(tok, cu, meta)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    for (name, parameter), (_, original) in zip(candidate.named_parameters(), reference.named_parameters(), strict=True):
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, original.grad, rtol=0.05, atol=0.01, msg=name)
    assert torch.all(output.counts == rows).item()
    assert all("input_rows_down" not in dict(block.named_buffers()) for block in candidate.body.blocks)
    fast = DownToner(candidate, config, output, rows_total=rows, world=1, rank=0, device=device)
    assert set(fast.groups) == {"d"}
    assert all(group.n == 256 for group in fast.groups.values())
    fast.pool.shutdown(wait=True)


def test_output_rows(device):
    torch.manual_seed(5)
    positions = torch.tensor([0, 3, 9, 15], device=device)
    rows = torch.zeros(16, 256, dtype=torch.bfloat16, device=device)
    at = torch.zeros(8, dtype=torch.int64, device=device)
    linear = nn.Linear(512, 256, bias=False).to(device)
    plain = copy.deepcopy(linear)
    observed = torch.compile(
        RowObservedLinear(linear, rows, at, positions),
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
    )
    unobserved = torch.compile(plain, fullgraph=True, mode="max-autotune-no-cudagraphs")
    expected = torch.zeros_like(rows)
    for index in range(2):
        at.copy_(torch.arange(index * 8, index * 8 + 8, device=device))
        x = torch.randn(2, 16, 512, device=device)
        target = torch.randn(2, 16, 256, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = observed(x)
            reference = unobserved(x)
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-5)
        (value.float() - target).square().sum().backward()
        (reference.float() - target).square().sum().backward()
        # The projection gradient is BF16 before the observer stores it; the power-of-two scale is exact.
        cotangent = (2 * (reference.detach().float() - target)).to(torch.bfloat16)
        expected[index * 8 : index * 8 + 8] = cotangent[:, positions].reshape(8, 256) * ROW_SCALE
    torch.testing.assert_close(linear.weight.grad, plain.weight.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rows, expected, rtol=0, atol=0)


def test_block_matrix_updates(device):
    torch.manual_seed(37)
    args = fast_arguments(configuration("inner_outer_all", "0p5m", 0, 8))

    def reference(weight, gradient, momentum, factors, mode):
        momentum = 0.95 * momentum + gradient
        source = gradient + 0.95 * momentum
        dense = {
            name: value if value is None or value.ndim == 2 else torch.diag(value)
            for name, value in vars(factors).items()
        }
        if dense["right_inner"] is not None:
            source = source @ dense["right_inner"]
        if dense["left_inner"] is not None:
            source = dense["left_inner"] @ source
        direction = MU.newton_schulz(source, 10, 1e-5, mode)
        decay = weight
        if dense["right_outer"] is not None:
            right = dense["right_outer"] * (weight.shape[1] ** 0.5 / dense["right_outer"].norm())
            direction, decay = direction @ right, weight @ right @ right
        if dense["left_outer"] is not None:
            left = dense["left_outer"] * (weight.shape[0] ** 0.5 / dense["left_outer"].norm())
            direction = left @ direction
            if dense["right_outer"] is None:
                decay = left @ left @ weight
        scale = max(1.0, weight.shape[0] / weight.shape[1]) ** 0.5
        return momentum, -0.02 * (scale * direction + 0.1 * decay)

    for shape in ((256, 256), (1024, 256), (256, 1024)):
        weight = torch.randn(shape, device=device) * 0.05
        gradient, momentum = torch.randn_like(weight), torch.randn_like(weight)
        left = torch.randn(shape[0], 64, device=device)
        right = torch.randn(shape[1], 64, device=device)
        li, lo = inverse_factors(left @ left.T, OutputSettings(damping=0.01))
        ri, ro = inverse_factors(right @ right.T, OutputSettings(damping=0.01))
        cases = {
            "two_sided": MatrixFactors(li, lo, ri, ro),
            "null": MatrixFactors(right_inner=ri, right_outer=ro),
            "left_only": MatrixFactors(left_inner=li, left_outer=lo),
            "diagonal": MatrixFactors(
                *(torch.rand(n, device=device) + 0.2 for n in (shape[0], shape[0], shape[1], shape[1]))
            ),
        }
        for mode in ("f32", args.muon_ns_mode):
            args.muon_ns_mode = mode
            for name, factors in cases.items():
                actual_m, delta = block_update(
                    weight,
                    gradient,
                    momentum,
                    factors,
                    args=args,
                    lr=0.02,
                    weight_decay=0.1,
                )
                expected_m, expected = reference(weight, gradient, momentum, factors, mode)
                torch.testing.assert_close(actual_m, expected_m, rtol=0, atol=0)
                torch.testing.assert_close(delta, expected, rtol=0.003, atol=5e-5, msg=f"{name}.{shape}.{mode}")


def test_combined_outer_updates(device):
    """Check production owned stacks against unchanged NS and post-NS equations."""
    args = fast_arguments(configuration("outer_leftdiag_combined_fused", "0p5m", 0, 4))
    generator = torch.Generator(device=device).manual_seed(73)
    for count, rows, columns in ((16, 512, 512), (8, 2048, 512), (4, 512, 2048)):
        shape = (count, rows, columns)
        weights = torch.randn(shape, generator=generator, device=device) * 0.02
        gradient = torch.randn(shape, generator=generator, device=device)
        momentum = torch.randn(shape, generator=generator, device=device) * 0.1
        matrix = torch.randn(count, columns, columns, generator=generator, device=device) / columns**0.5
        identity = torch.eye(columns, device=device)[None]
        right = (matrix @ matrix.mT + identity, matrix.mT @ matrix + identity)
        left = torch.rand(count, rows, generator=generator, device=device) + 0.25
        for pattern in ("gaussian", "zero", "anisotropic"):
            gradients, momenta, outer = gradient.clone(), momentum.clone(), left.clone()
            if pattern == "zero":
                gradients.zero_()
                momenta.zero_()
            elif pattern == "anisotropic":
                outer *= torch.logspace(-2, 2, rows, device=device)
            for lr in (0.0, 0.02, 0.02 / 32):
                actual_trace, reference_trace = {}, {}
                reference_momentum, reference_delta = block_update_stack(
                    weights,
                    gradients,
                    momenta,
                    (None, outer),
                    right,
                    args=args,
                    lr=lr,
                    weight_decay=0.1,
                    trace=reference_trace,
                )
                actual_momentum, actual_delta = combined_outer_update_stack(
                    weights,
                    gradients,
                    momenta,
                    (None, outer),
                    right,
                    args=args,
                    lr=lr,
                    weight_decay=0.1,
                    trace=actual_trace,
                )
                torch.testing.assert_close(actual_momentum, reference_momentum, rtol=0, atol=0)
                for key in ("ns_source", "ns_output"):
                    torch.testing.assert_close(actual_trace[key], reference_trace[key], rtol=0, atol=0)
                torch.testing.assert_close(actual_delta, reference_delta, rtol=1e-5, atol=1e-6)


def small_block_config(arm, layers):
    config = configuration(arm, "0p5m", 0, 1)
    config["model"].update(hidden=256, mlp=1024, heads=4, layers=layers, sequence_length=16, vocab_size=256)
    return config


def synthetic_rows(rank, layer, offset, shape, step=0):
    grid = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
    return torch.sin(grid * 0.013 + rank * 0.7 + layer * 0.1 + offset * 0.01 + step * 0.0001).to(torch.bfloat16)


def test_fused_diagonal_pipeline(device, world, rank):
    """Value/root/update parity, pooled EMA and restored scratch after 24 diagonal updates."""
    for rows, width in (
        (1, 1),
        (17, 33),
        (129, 511),
        (513, 2049),
        (4576, 512),
        (9152, 2048),
    ):
        buffers = [synthetic_rows(rank, index, 3, (rows, width)).to(device) for index in range(3)]
        output = torch.empty(3, width, device=device)
        reducer = RowSquares(buffers, output)
        actual = reducer()
        expected = torch.stack([buffer.float().square().sum(0) for buffer in buffers])
        reference = torch.stack([buffer.double().square().sum(0) for buffer in buffers])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual.double(), reference, rtol=1e-5, atol=1e-6)
        for buffer in buffers:
            buffer.zero_()
        torch.testing.assert_close(reducer(), torch.zeros_like(output), rtol=0, atol=0)
    rows = {
        (layer, kind): torch.zeros(129, width, dtype=torch.bfloat16, device=device)
        for layer in range(2)
        for kind, width in (("q", 33), ("gate", 65))
    }
    settings = {
        "decay": 0.95,
        "damping": 0.001,
        "inner_power": 0.5,
        "outer_power": 0.125,
        "rows_total": 129 * world,
        "world": world,
        "device": device,
    }
    fused = DiagonalMoments(rows, backend="triton", **settings)
    original = DiagonalMoments(rows, backend="torch", **settings)
    restored = DiagonalMoments(rows, backend="triton", **settings)
    reference_moments = {width: torch.zeros_like(moment, device="cpu") for width, moment in fused.moment.items()}
    for step in range(24):
        for (layer, kind), buffer in rows.items():
            buffer.copy_(synthetic_rows(rank, layer, BLOCK_KINDS.index(kind), buffer.shape, step).to(device))
        fused.update()
        original.update()
        for width, keys in fused.keys.items():
            values = [
                sum(
                    synthetic_rows(
                        other,
                        layer,
                        BLOCK_KINDS.index(kind),
                        rows[layer, kind].shape,
                        step,
                    )
                    .float()
                    .square()
                    .sum(0)
                    for other in range(world)
                )
                for layer, kind in keys
            ]
            reference_moments[width].mul_(0.95).add_(torch.stack(values), alpha=0.05 / (129 * world))
            torch.testing.assert_close(fused.moment[width], original.moment[width], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(
                fused.moment[width].cpu(),
                reference_moments[width],
                rtol=1e-5,
                atol=1e-6,
            )
        for key, factors in fused.factors().items():
            for actual, expected in zip(factors, original.factors()[key], strict=True):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        if step == 21:
            restored.load_state_dict(serialized_state(fused.state_dict()))
        elif step > 21:
            restored.update()
            for width, moment in fused.moment.items():
                torch.testing.assert_close(restored.moment[width], moment, rtol=0, atol=0)
    args = fast_arguments(configuration("inner_outer_mlpdiag", "0p5m", 0, 4))
    for key in rows:
        width = rows[key].shape[1]
        weight = synthetic_rows(rank, 1, 0, (width, 47)).to(device).float()
        gradient = synthetic_rows(rank, 1, 0, (width, 47), 99).to(device).float() * 0.01
        momentum = torch.zeros_like(weight)
        actual = block_update(
            weight,
            gradient,
            momentum,
            MatrixFactors(*fused.factors()[key]),
            args=args,
            lr=0.02,
            weight_decay=0.1,
        )[1]
        expected = block_update(
            weight,
            gradient,
            momentum,
            MatrixFactors(*original.factors()[key]),
            args=args,
            lr=0.02,
            weight_decay=0.1,
        )[1]
        torch.testing.assert_close(actual, expected, rtol=0.003, atol=5e-5)


def outer_pipeline_updates(rows, fused_factors, torch_factors, args, device, rank):
    """Exercise all production orientations and the original fixture, with full right factors."""
    generator = torch.Generator(device=device).manual_seed(711 + rank)
    patterns = (
        ("structured", "gaussian", "original")
        if any(buffer.shape[1] == 512 for buffer in rows.values())
        else ("original",)
    )
    for pattern in patterns:
        shapes = ((33, 47), (65, 47)) if pattern == "original" else ((512, 512), (2048, 512), (512, 2048))
        for height, width in shapes:
            keys = [key for key, buffer in rows.items() if buffer.shape[1] == height]
            if pattern == "original":
                weights = torch.stack([synthetic_rows(rank, 1, 0, (height, width)).to(device).float() for _ in keys])
                gradients = torch.stack(
                    [synthetic_rows(rank, 1, 0, (height, width), 99).to(device).float() * 0.01 for _ in keys]
                )
                momentum = torch.zeros_like(weights)
            elif pattern == "structured":
                grid = torch.arange(2 * height * width, device=device).reshape(2, height, width).float()
                weights = torch.sin(grid * 0.013 + rank * 0.7) * 0.05
                gradients = torch.cos(grid * 0.017 + 0.3) * 0.01
                momentum = torch.sin(grid * 0.019 + 0.7) * 0.005
            else:
                weights = torch.randn(2, height, width, device=device, generator=generator) * 0.05
                gradients = torch.randn(2, height, width, device=device, generator=generator) * 0.01
                momentum = torch.randn(2, height, width, device=device, generator=generator) * 0.005
            vectors = torch.randn(2, width, 8, device=device, generator=generator) / width**0.5
            eye = torch.eye(width, device=device)[None].expand(2, -1, -1)
            right = (eye + vectors @ vectors.mT, eye + 0.1 * (vectors @ vectors.mT))
            outer = torch.stack([fused_factors[key][1] for key in keys])
            reference_outer = torch.stack([torch_factors[key][1] for key in keys])
            trace = {}
            actual_momentum, actual_delta = block_update_stack(
                weights,
                gradients,
                momentum,
                (None, outer),
                right,
                args=args,
                lr=0.02,
                weight_decay=0.1,
                trace=trace,
            )
            result = compare_outer_update(
                weights,
                gradients,
                momentum,
                actual_momentum,
                actual_delta,
                right,
                trace,
                reference_outer,
                args=args,
                lr=0.02,
                weight_decay=0.1,
            )
            restored_trace = {}
            restored_momentum, restored_delta = block_update_stack(
                weights,
                gradients,
                serialized_state(momentum).to(device),
                (None, outer),
                right,
                args=args,
                lr=0.02,
                weight_decay=0.1,
                trace=restored_trace,
            )
            torch.testing.assert_close(restored_momentum, actual_momentum, rtol=0, atol=0)
            torch.testing.assert_close(restored_delta, actual_delta, rtol=0, atol=0)
            for key in trace:
                torch.testing.assert_close(restored_trace[key], trace[key], rtol=0, atol=0)
            # The original failed fixture also remains covered without right factors.
            if pattern == "original":
                trace = {}
                actual_momentum, actual_delta = block_update_stack(
                    weights,
                    gradients,
                    momentum,
                    (None, outer),
                    (None, None),
                    args=args,
                    lr=0.02,
                    weight_decay=0.1,
                    trace=trace,
                )
                compare_outer_update(
                    weights,
                    gradients,
                    momentum,
                    actual_momentum,
                    actual_delta,
                    (None, None),
                    trace,
                    reference_outer,
                    args=args,
                    lr=0.02,
                    weight_decay=0.1,
                )
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "outer_diagonal_case": pattern,
                            "shape": [height, width],
                            **result,
                        }
                    ),
                    flush=True,
                )


def test_outer_diagonal_pipeline(device, world, rank):
    """Outer-only reduction/EMA/root/stack parity and exact restoration at update 22."""
    for count, width in (
        (1, 1),
        (17, 33),
        (129, 511),
        (513, 2049),
        (4576, 512),
        (9152, 2048),
    ):
        buffers = [synthetic_rows(rank, index, 3, (count, width)).to(device) for index in range(3)]
        output = torch.empty(3, width, device=device)
        reducer = RowSquares(buffers, output)
        torch.testing.assert_close(
            reducer(),
            torch.stack([buffer.float().square().sum(0) for buffer in buffers]),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            reducer().double(),
            torch.stack([buffer.double().square().sum(0) for buffer in buffers]),
            rtol=1e-5,
            atol=1e-6,
        )
        for buffer in buffers:
            buffer.zero_()
        torch.testing.assert_close(reducer(), torch.zeros_like(output), rtol=0, atol=0)
    outer_diagonal_sequence(device, world, rank, decay=0.938)
    outer_diagonal_sequence(device, world, rank, decay=0.95)


def outer_diagonal_sequence(device, world, rank, *, decay):
    """Retain the original fixture's 0.95 EMA and separately gate the actual 0.938 recipe."""
    widths = (("q", 33), ("gate", 65))
    if decay == 0.938:
        widths += (("q512", 512), ("gate2048", 2048))
    rows = {
        (layer, kind): torch.zeros(129, width, dtype=torch.bfloat16, device=device)
        for layer in range(2)
        for kind, width in widths
    }
    # Production widths use standard projection keys for the update-stack comparisons.
    rows = {
        (
            layer + (2 if kind in ("q512", "gate2048") else 0),
            "gate" if "gate" in kind else "q",
        ): buffer
        for (layer, kind), buffer in rows.items()
    }
    settings = {
        "decay": decay,
        "damping": 0.001,
        "inner_power": 0.5,
        "outer_power": 0.125,
        "rows_total": 129 * world,
        "world": world,
        "device": device,
    }
    fused = DiagonalMoments(rows, backend="triton", **settings)
    original = DiagonalMoments(rows, backend="torch", **settings)
    restored = DiagonalMoments(rows, backend="triton", **settings)
    reference = {
        width: torch.zeros_like(moment, device="cpu", dtype=torch.float64) for width, moment in fused.moment.items()
    }
    args = fast_arguments(configuration("outer_leftdiag_fused", "0p5m", 0, 4))
    for step in range(24):
        for (layer, kind), buffer in rows.items():
            buffer.copy_(synthetic_rows(rank, layer, BLOCK_KINDS.index(kind), buffer.shape, step).to(device))
        fused.update()
        original.update()
        fused_factors, original_factors = fused.factors(), original.factors()
        for width, keys in fused.keys.items():
            sums = torch.stack(
                [
                    sum(
                        synthetic_rows(
                            other,
                            layer,
                            BLOCK_KINDS.index(kind),
                            rows[layer, kind].shape,
                            step,
                        )
                        .double()
                        .square()
                        .sum(0)
                        for other in range(world)
                    )
                    for layer, kind in keys
                ]
            )
            reference[width].mul_(settings["decay"]).add_(sums, alpha=(1 - settings["decay"]) / settings["rows_total"])
            torch.testing.assert_close(fused.moment[width], original.moment[width], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(
                fused.moment[width].cpu().double(),
                reference[width],
                rtol=1e-5,
                atol=1e-6,
            )
            exact = fp64_diagonal_factors(reference[width], 0.001, 0.5, 0.125)
            for index, key in enumerate(keys):
                for actual, expected, fp64 in zip(fused_factors[key], original_factors[key], exact, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(actual.cpu().double(), fp64[index], rtol=1e-5, atol=1e-6)
        if step == 21:
            restored.load_state_dict(serialized_state(fused.state_dict()))
            compare_tree(restored.state_dict(), fused.state_dict(), "outer.restored22", [])
        elif step > 21:
            restored.update()
            for width, moment in fused.moment.items():
                torch.testing.assert_close(restored.moment[width], moment, rtol=0, atol=0)
            assert restored.weight == fused.weight
            for key, factors in restored.factors().items():
                for actual, expected in zip(factors, fused_factors[key], strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if step in (21, 23):
            outer_pipeline_updates(rows, fused_factors, original_factors, args, device, rank)


def test_block_statistics(device, world, rank):
    positions, capacity = [0, 2, 4, 6], 16
    for arm in ("inner_outer_all", "inner_outer_all_diag", "inner_outer_mlpdiag"):
        config = small_block_config(arm, world)
        network = build_model(
            config,
            positions=positions,
            row_capacity=capacity,
            rows_per_microbatch=4,
            device=device,
        )
        initialize_model(network, 0)
        fast = BlockToner(
            network,
            config,
            positions,
            rows_per_microbatch=4,
            capacity=capacity,
            rows_total=capacity * world,
            world=world,
            rank=rank,
            device=device,
        )
        try:
            for (layer, kind), buffer in fast.output_rows.items():
                buffer.copy_(synthetic_rows(rank, layer, BLOCK_KINDS.index(kind), buffer.shape).to(device))
            beta = config["statistics"]["momentum"]

            def pooled(layer, offset, shape):
                return [synthetic_rows(r, layer, offset, shape).float() for r in range(world)]

            if fast.output is not None:
                fast.finish_backward()
                fast.output.side.synchronize()
                for group in fast.output.groups.values():
                    for index, (layer, kind) in enumerate(group.own):
                        rows = pooled(layer, BLOCK_KINDS.index(kind), (capacity, group.n))
                        expected = (1 - beta) * sum(r.T @ r for r in rows) / (capacity * world)
                        torch.testing.assert_close(group.C[index].cpu(), expected, rtol=1e-5, atol=1e-7)
            if fast.output_diagonal is not None:
                fast.output_diagonal.update()
                for width, keys in fast.output_diagonal.keys.items():
                    for index, (layer, kind) in enumerate(keys):
                        rows = pooled(layer, BLOCK_KINDS.index(kind), (capacity, width))
                        expected = (1 - beta) * sum(r.square().sum(0) for r in rows) / (capacity * world)
                        torch.testing.assert_close(
                            fast.output_diagonal.moment[width][index].cpu(),
                            expected,
                            rtol=1e-5,
                            atol=1e-7,
                        )
        finally:
            for engine in toner_engines(fast):
                engine.side.synchronize()
                engine.pool.shutdown(wait=True)


def test_block_compiled_body(device):
    config = small_block_config("inner_outer_all", 2)
    config["model"]["sequence_length"] = 64
    positions = sample_positions(64, 8, 8)
    rows = len(positions)
    candidate = build_model(
        config,
        positions=positions,
        row_capacity=rows,
        rows_per_microbatch=rows,
        device=device,
    )
    initialize_model(candidate, 0)
    reference = Model(config["model"]).to(device)
    reference.load_state_dict(candidate.state_dict())
    eager = Model(config["model"]).to(device)
    eager.load_state_dict(candidate.state_dict())
    fast = BlockToner(
        candidate,
        config,
        positions,
        rows_per_microbatch=rows,
        capacity=rows,
        rows_total=rows,
        world=1,
        rank=0,
        device=device,
    )
    try:
        observed_gradients = {}
        index = torch.tensor(positions, device=device)
        for layer, block in enumerate(eager.body.blocks):
            for kind in BLOCK_KINDS:

                def keep(module, inputs, output, key=(layer, kind)):
                    output.register_hook(
                        lambda g: observed_gradients.__setitem__(key, g[:, index].reshape(-1, g.shape[-1]))
                    )

                getattr(block, kind).register_forward_hook(keep)
        compiled = torch.compile(candidate.body, mode="max-autotune-no-cudagraphs", fullgraph=True)
        pinned_body = torch.compile(reference.body, mode="max-autotune-no-cudagraphs", fullgraph=True)
        tokens = np.arange(64, dtype=np.uint32).reshape(1, 64)
        tok, _, cu, meta = pack(tokens, 34, EOS=128001)
        tok, cu = tok.to(device), cu.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = compiled(tok, cu, meta)
            expected = pinned_body(tok, cu, meta)
            hooked = eager.body(tok, cu, meta)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        # A fixed random readout: mean(h^2) of an RMSNorm output is nearly constant, so its
        # gradients are rounding noise that differs between compiled and eager kernels.
        target = torch.randn(
            actual.shape,
            generator=torch.Generator(device).manual_seed(11),
            device=device,
        )
        for value in (actual, expected, hooked):
            (value.float() * target).sum().backward()

        def relative(value, reference_value):
            return ((value.float() - reference_value.float()).norm() / reference_value.float().norm()).item()

        for (name, parameter), (_, original) in zip(
            candidate.named_parameters(), reference.named_parameters(), strict=True
        ):
            if parameter.grad is not None and relative(parameter.grad, original.grad) > 0.05:
                raise AssertionError(f"Compiled observed gradient differs for {name}")
        for layer, block in enumerate(candidate.body.blocks):
            for kind in BLOCK_KINDS:
                stored = getattr(block, kind).output_rows.float() / ROW_SCALE
                # Compiled and eager BF16 backward accumulate differently; rows must agree in aggregate.
                error = relative(stored, observed_gradients[layer, kind])
                if not error <= 0.05:
                    raise AssertionError(f"Compiled output rows differ for layer {layer} {kind}: {error}")
    finally:
        for engine in toner_engines(fast):
            engine.side.synchronize()
            engine.pool.shutdown(wait=True)


def serialized_state(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def test_cached_root_continuation(device, world, rank):
    """Continuation parity for the arms this wave launches, plus the three references.

    Every engine opens NCCL communicators that are never all released, and one is created
    without a handle inside the pinned snapshot, so they cannot be destroyed here. Covering
    every arm ever defined therefore exhausts communicators in this one process: at thirteen
    arms it fails deterministically inside the statistics reduce-scatter. Arms from earlier
    waves were gated by this same check when they ran.
    """
    for arm in dict.fromkeys(("muont", "outer_left_down", "down_hybrid", *(a for a, _, _ in WAVE))):
        config = configuration(arm, "0p5m", 0, 1)
        config["model"].update(
            hidden=256,
            mlp=1024,
            heads=4,
            layers=world,
            sequence_length=16,
            vocab_size=256,
        )
        positions, rows = [0, 2, 4, 6], 16

        def construct(config, positions, rows, arm):
            network = build_model(
                config,
                positions=positions,
                row_capacity=rows,
                rows_per_microbatch=len(positions),
                device=device,
            )
            initialize_model(network, 0)
            output = (
                None
                if arm == "muont" or arm in BLOCK_ARMS
                else OutputMoments(
                    network,
                    positions,
                    OutputSettings(),
                    world=world,
                    rank=rank,
                    device=device,
                )
            )
            optimizers, fast, _ = build(
                network,
                config,
                output,
                positions=positions,
                rows_per_microbatch=len(positions),
                capacity=rows,
                rows_total=rows * world,
                world=world,
                rank=rank,
                device=device,
            )
            fast.param_groups[0]["lr"] = config["optimizer"]["lr"]
            return network, output, fast, optimizers[-1]

        def update(network, output, fast, step, positions):
            for block in network.body.blocks:
                for name, value in block.named_buffers():
                    if name.startswith("input_rows_"):
                        grid = torch.arange(value.numel(), device=device).reshape(value.shape).float()
                        value.copy_(torch.sin(grid * 0.017 + step * 0.0001 + rank * 0.1))
            for index, parameter in enumerate(fast.params):
                grid = torch.arange(parameter.numel(), device=device).reshape(parameter.shape)
                parameter.grad = torch.cos(grid * 0.01 + index * 0.1 + step * 0.001).float() * 0.01
            if output is not None:
                output.begin(16)
                for block in network.body.blocks:
                    x = torch.ones(1, 8, 1024, device=device) * (1 + step * 0.001)
                    (block.down(x).square().mean() / 16).backward()
                output.update(len(positions))
            fast.begin()
            if isinstance(fast, BlockToner):
                for (layer, kind), buffer in fast.output_rows.items():
                    buffer.copy_(synthetic_rows(rank, layer, BLOCK_KINDS.index(kind), buffer.shape, step).to(device))
                fast.finish_backward()
            fast.step()
            # Control host completion in this numerical probe, not in production.
            for engine in toner_engines(fast):
                engine.side.synchronize()
                for group in engine.groups.values():
                    for pending in group.pending:
                        if pending is not None:
                            pending[0].result()

        network, output, fast, auxiliary = construct(config, positions, rows, arm)
        try:
            for step in range(22):
                update(network, output, fast, step, positions)
            if not all(g.graph is not None for e in toner_engines(fast) for g in e.groups.values() if g.per):
                raise AssertionError("Continuation probe did not reach cached-root CUDA graphs")
            saved = serialized_state(
                {
                    "model": network.state_dict(),
                    "fast": fast_runtime_state(fast),
                    "output": None if output is None else output.state_dict(),
                }
            )
            for step in range(22, 24):
                update(network, output, fast, step, positions)
            expected = serialized_state(
                {
                    "model": network.state_dict(),
                    "fast": fast_runtime_state(fast),
                    "output": None if output is None else output.state_dict(),
                }
            )
        finally:
            for engine in toner_engines(fast):
                engine.side.synchronize()
                engine.pool.shutdown(wait=True)
        del network, output, fast, auxiliary
        restored, output, fast, _auxiliary = construct(config, positions, rows, arm)
        try:
            restored.load_state_dict(saved["model"])
            restore_fast(fast, saved["fast"])
            if output is not None:
                output.load_state_dict(saved["output"])
            for step in range(22, 24):
                update(restored, output, fast, step, positions)
            actual = serialized_state(
                {
                    "model": restored.state_dict(),
                    "fast": fast_runtime_state(fast),
                    "output": None if output is None else output.state_dict(),
                }
            )
            compare_tree(actual, expected, f"cached_root_resume.{arm}", [])
        finally:
            for engine in toner_engines(fast):
                engine.side.synchronize()
                engine.pool.shutdown(wait=True)
        # Every arm builds its own engines, each holding a CUDA-graph private pool that is
        # only released with the graph object. Release them between arms so a long arm list
        # does not accumulate graph pools for the whole run.
        del restored, output, fast, _auxiliary
        gc.collect()
        torch.cuda.empty_cache()


def main():
    require_synthetic_owner()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    verify_snapshot()
    local = int(os.environ["LOCAL_RANK"])
    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError("GPU checks require A100 SM80")
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", device_id=device)
    results = {}
    try:
        for name, check in (
            ("gpu_collector", lambda: test_collector(device)),
            ("microbatch_partition", lambda: test_microbatch_partition(device)),
            (
                "distributed_statistics",
                lambda: test_distributed_moments(device, world, rank),
            ),
            ("fa2", lambda: test_attention(device)),
            ("gpu_down_updates", lambda: test_down_matrix_updates(device)),
            ("synthetic_compiled_body", lambda: test_compiled_body(device)),
            ("gpu_output_rows", lambda: test_output_rows(device)),
            ("gpu_block_updates", lambda: test_block_matrix_updates(device)),
            (
                "outer_diagonal_pipeline",
                lambda: test_outer_diagonal_pipeline(device, world, rank),
            ),
            ("gpu_upstream_product_updates", lambda: test_upstream_product_updates(device)),
            ("gpu_upstream_scalar_step", lambda: test_upstream_scalar_step(device)),
            (
                "block_distributed_statistics",
                lambda: test_block_statistics(device, world, rank),
            ),
            ("block_compiled_body", lambda: test_block_compiled_body(device)),
            (
                "cached_root_continuation",
                lambda: test_cached_root_continuation(device, world, rank),
            ),
        ):
            check()
            dist.barrier()
            results[name] = True
            if rank == 0:
                print(json.dumps({"check": name, "passed": True}), flush=True)
        if rank == 0:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(
                args.receipt,
                {
                    "time": utc_now(),
                    "checks": results,
                    "world": world,
                    "kind": "synthetic_gpu_correctness",
                    "source_sha256": {
                        str(p.relative_to(TASK)): sha256(p) for p in sorted((TASK / "experiment").rglob("*.py"))
                    },
                    "evidence": {
                        "command": sys.argv,
                        "gpu": torch.cuda.get_device_name(device),
                        "down_update_ns_modes": [
                            "f32",
                            configuration("down_hybrid", "0p5m", 0, 8)["optimizer"]["ns_mode"],
                        ],
                    },
                    "limitation": "Synthetic GPU correctness is not real-data Marin training evidence.",
                },
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
