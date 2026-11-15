# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable A100/NCCL checks on labeled synthetic tensors, before real-data smoke."""

import argparse
import copy
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

from .bootstrap import verify_snapshot
from .capture import OutputLinear, OutputMoments, build_model
from .configure import configuration
from .factors import OutputSettings, inverse_factors
from .optimizer import DownToner, accurate_left, accurate_right, build, down_update, fast_runtime_state, restore_fast
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


def serialized_state(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def test_cached_root_continuation(device, world, rank):
    for arm in ("muont", "outer_left_down", "down_hybrid"):
        config = configuration(arm, "0p5m", 0, 1)
        config["model"].update(hidden=256, mlp=1024, heads=4, layers=world, sequence_length=16, vocab_size=256)
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
                if arm == "muont"
                else OutputMoments(network, positions, OutputSettings(), world=world, rank=rank, device=device)
            )
            optimizers, fast, _ = build(
                network, config, output, rows_total=rows * world, world=world, rank=rank, device=device
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
            fast.step()
            # Control host completion in this numerical probe, not in production.
            fast.side.synchronize()
            for group in fast.groups.values():
                for pending in group.pending:
                    if pending is not None:
                        pending[0].result()

        network, output, fast, auxiliary = construct(config, positions, rows, arm)
        try:
            for step in range(22):
                update(network, output, fast, step, positions)
            if not all(group.graph is not None for group in fast.groups.values() if group.per):
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
            fast.side.synchronize()
            fast.pool.shutdown(wait=True)
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
            fast.side.synchronize()
            fast.pool.shutdown(wait=True)


def main():
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
            ("distributed_statistics", lambda: test_distributed_moments(device, world, rank)),
            ("fa2", lambda: test_attention(device)),
            ("gpu_down_updates", lambda: test_down_matrix_updates(device)),
            ("synthetic_compiled_body", lambda: test_compiled_body(device)),
            ("cached_root_continuation", lambda: test_cached_root_continuation(device, world, rank)),
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
