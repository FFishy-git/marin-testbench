# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic GO-MUON checks for the parent-owned, two-rank A100/NCCL gate.

Importing this module does not initialize a process group or launch GPU work.
The caller owns devices, deterministic settings, and the final evidence receipt.
"""

import copy
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from reproduce.marin.data import pack
from reproduce.marin.model import Model, initialize_model
from torch import nn

from .block_toner import INPUT_KINDS, ROW_SCALE, RowObservedLinear
from .go_muon import GoMuon, GoMuonSettings, matched_direction
from .related_base import MATRIX_KINDS

UPDATE_RTOL = 0.003
UPDATE_ATOL = 5e-5
TRANSPARENCY_RTOL = 1e-5
TRANSPARENCY_ATOL = 1e-5
COMPILE_OPTIONS = {
    "max_autotune": True,
    "triton.cudagraphs": False,
    "max_autotune_gemm_backends": "ATEN",
    "emulate_precision_casts": True,
}


def reference_finite_direction(source, left, right):
    """Independent per-matrix arithmetic for the production BF16/FP32 NS map."""
    transformed = left @ source @ right
    x = transformed.float() / (torch.linalg.matrix_norm(transformed.float()) + 1e-5)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.mT
    for _ in range(10):
        gram = torch.mm(x.to(torch.bfloat16), x.mT.to(torch.bfloat16), out_dtype=torch.float32)
        square = torch.mm(gram.to(torch.bfloat16), gram.to(torch.bfloat16), out_dtype=torch.float32)
        polynomial = -4.7750 * gram + 2.0315 * square
        x = 3.4445 * x + torch.mm(polynomial.to(torch.bfloat16), x.to(torch.bfloat16), out_dtype=torch.float32)
    polar = x.mT if transpose else x
    raw = left @ polar @ right
    return raw * (min(source.shape) ** 0.5 / torch.linalg.matrix_norm(raw))


def reference_exact_direction(source, left, right):
    u, _, vh = torch.linalg.svd(left.double() @ source.double() @ right.double(), full_matrices=False)
    raw = left.double() @ (u @ vh) @ right.double()
    return (raw * (min(source.shape) ** 0.5 / raw.norm())).float()


@torch.no_grad()
def check_exact_numerics(device):
    """Accept the paper polar without comparing nonunique null-space bases."""
    generator = torch.Generator(device=device).manual_seed(953)
    max_absolute, max_mean = 0.0, 0.0
    max_completion_error, max_alignment_error = 0.0, 0.0
    shapes = ((256, 256), (1024, 256), (256, 1024))
    for shape in shapes:
        source = torch.randn(shape, generator=generator, device=device)
        left = torch.diag(torch.rand(shape[0], generator=generator, device=device) + 0.5)
        right = torch.diag(torch.rand(shape[1], generator=generator, device=device) + 0.5)
        actual = matched_direction(source, left, right, polar_method="exact_svd")
        expected = reference_exact_direction(source, left, right)
        torch.testing.assert_close(actual, expected, rtol=UPDATE_RTOL, atol=UPDATE_ATOL)
        error = (actual - expected).abs()
        max_absolute, max_mean = max(max_absolute, error.max().item()), max(max_mean, error.mean().item())
        torch.testing.assert_close(
            actual.double().norm(),
            torch.tensor(min(shape) ** 0.5, dtype=torch.float64, device=device),
            rtol=1e-6,
            atol=1e-6,
        )
        zero = matched_direction(torch.zeros_like(source), left, right, polar_method="exact_svd")
        torch.testing.assert_close(zero, torch.zeros_like(source), rtol=0, atol=0)

    rank_deficient_shapes = ((64, 128), (128, 64))
    for shape in rank_deficient_shapes:
        source = torch.zeros(shape, device=device)
        support = torch.arange(16, device=device)
        singular_values = torch.linspace(1.0, 4.0, 16, device=device)
        signs = torch.where(support % 2 == 0, 1.0, -1.0)
        source[support, support] = signs * singular_values
        # Scalar factors force the final graft to remove a sixfold map-back
        # scale. Null completions may differ, but their norm and supported map
        # are fixed independently of either backend's SVD basis choice.
        left, right = 2 * torch.eye(shape[0], device=device), 3 * torch.eye(shape[1], device=device)
        actual = matched_direction(source, left, right, polar_method="exact_svd")
        known_support = torch.zeros_like(source)
        known_support[support, support] = signs
        actual_support = actual[:16, :] if shape[0] < shape[1] else actual[:, :16]
        expected_support = known_support[:16, :] if shape[0] < shape[1] else known_support[:, :16]
        torch.testing.assert_close(actual_support, expected_support, rtol=UPDATE_RTOL, atol=UPDATE_ATOL)
        gram = actual @ actual.mT if shape[0] < shape[1] else actual.mT @ actual
        identity = torch.eye(min(shape), device=device)
        torch.testing.assert_close(gram, identity, rtol=UPDATE_RTOL, atol=UPDATE_ATOL)
        transformed = left.double() @ source.double() @ right.double()
        alignment = (transformed * actual.double()).sum()
        optimum = torch.linalg.svdvals(transformed).sum()
        torch.testing.assert_close(alignment, optimum, rtol=UPDATE_RTOL, atol=UPDATE_ATOL)
        torch.testing.assert_close(
            actual.double().norm(),
            torch.tensor(min(shape) ** 0.5, dtype=torch.float64, device=device),
            rtol=1e-6,
            atol=1e-6,
        )
        max_completion_error = max(max_completion_error, (gram - identity).abs().max().item())
        max_alignment_error = max(max_alignment_error, (alignment - optimum).abs().item())
    return {
        "go_muon_gpu_exact_svd": True,
        "go_muon_gpu_exact_svd_full_rank_shapes": len(shapes),
        "go_muon_gpu_exact_svd_max_absolute_error": max_absolute,
        "go_muon_gpu_exact_svd_max_mean_absolute_error": max_mean,
        "go_muon_gpu_exact_svd_rank_deficient_shapes": len(rank_deficient_shapes),
        "go_muon_gpu_exact_svd_completion_max_absolute_error": max_completion_error,
        "go_muon_gpu_exact_svd_alignment_max_absolute_error": max_alignment_error,
        "go_muon_gpu_exact_svd_null_convention": "semi_orthogonal_support_maximizer_not_unique_basis_parity",
    }


@torch.no_grad()
def check_numerics(device):
    exact_diagnostics = check_exact_numerics(device)
    generator = torch.Generator(device=device).manual_seed(937)
    max_absolute, max_mean, max_oracle_error = 0.0, 0.0, 0.0
    for shape in ((256, 256), (1024, 256), (256, 1024)):
        source = torch.randn(shape, generator=generator, device=device)
        left = torch.diag(torch.rand(shape[0], generator=generator, device=device) + 0.5)
        right = torch.diag(torch.rand(shape[1], generator=generator, device=device) + 0.5)
        actual = matched_direction(source, left, right, polar_method="newton_schulz", ns_mode="tpu")
        expected = reference_finite_direction(source, left, right)
        torch.testing.assert_close(actual, expected, rtol=UPDATE_RTOL, atol=UPDATE_ATOL)
        error = (actual - expected).abs()
        max_absolute, max_mean = max(max_absolute, error.max().item()), max(max_mean, error.mean().item())
        torch.testing.assert_close(
            actual.double().norm(),
            torch.tensor(min(shape) ** 0.5, dtype=torch.float64, device=device),
            rtol=1e-6,
            atol=1e-6,
        )
        oracle = reference_exact_direction(source, left, right)
        max_oracle_error = max(max_oracle_error, ((actual - oracle).norm() / oracle.norm()).item())
        zero = matched_direction(torch.zeros_like(source), left, right, polar_method="newton_schulz", ns_mode="tpu")
        torch.testing.assert_close(zero, torch.zeros_like(source), rtol=0, atol=0)
    # For an isotropic orthonormal source the final graft cancels the finite
    # polynomial's scalar singular-value response, so exact polar is an oracle.
    isotropic = torch.eye(64, 128, device=device)
    actual = matched_direction(
        isotropic,
        torch.eye(64, device=device),
        torch.eye(128, device=device),
        polar_method="newton_schulz",
        ns_mode="tpu",
    )
    torch.testing.assert_close(actual, isotropic, rtol=UPDATE_RTOL, atol=UPDATE_ATOL)
    return {
        **exact_diagnostics,
        "go_muon_gpu_ns_max_absolute_error": max_absolute,
        "go_muon_gpu_ns_max_mean_absolute_error": max_mean,
        "go_muon_gpu_ns_exact_polar_relative_error": max_oracle_error,
        "go_muon_gpu_ns_mode": "tpu",
    }


def small_config(settings, layers):
    return {
        "model": {
            "hidden": 32,
            "mlp": 128,
            "heads": 2,
            "layers": layers,
            "sequence_length": 64,
            "vocab_size": 256,
            "rope_theta": 10000.0,
            "norm_eps": 1e-5,
            "residual_dtype": "bf16",
        },
        "optimizer": {
            "lr": 0.02,
            "weight_decay": 0.1,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 10,
            "ns_eps": 1e-5,
            "ns_mode": "tpu",
        },
        "statistics": {"momentum": 0.5, "damping": 0.001},
        "root": {"refresh": 2},
        "experiment": {"related_baseline": {"settings": asdict(settings)}},
    }


def synthetic_rows(rank, layer, kind_index, width, *, step):
    grid = torch.arange(16 * width, dtype=torch.float32).reshape(16, width)
    return torch.sin(grid * 0.037 + rank * 0.7 + layer * 0.13 + kind_index * 0.23 + step * 0.11).to(torch.bfloat16)


def reference_shrink(moment, count):
    width = moment.shape[0]
    trace = moment.trace()
    squared = (moment @ moment).trace()
    denominator = (count + 1 - 2 / width) * (squared - trace.square() / width)
    numerator = (1 - 2 / width) * squared + trace.square()
    coefficient = (numerator / denominator).clamp(0, 1) if denominator.item() > 0 else moment.new_tensor(1.0)
    return (1 - coefficient) * moment + coefficient * trace / width * torch.eye(width, dtype=moment.dtype)


def reference_moment(layer, kind_index, width, world, step):
    rows = torch.cat([synthetic_rows(rank, layer, kind_index, width, step=step).double() for rank in range(world)])
    return rows.mT @ rows / (16 * world)


@torch.no_grad()
def check_statistics(device, world, rank):
    max_error, comparisons = 0.0, 0
    for mode in ("ema_per_step", "paper_sparse"):
        settings = GoMuonSettings(statistic_mode=mode, polar_method="exact_svd")
        snapshots = []
        for rows_per_microbatch in (4, 8):
            config = small_config(settings, world)
            model = Model(
                config["model"], positions=[0, 2, 4, 6], row_capacity=16, rows_per_microbatch=rows_per_microbatch
            ).to(device)
            optimizer = GoMuon(
                model,
                config,
                positions=[0, 2, 4, 6],
                rows_per_microbatch=rows_per_microbatch,
                capacity=16,
                rows_total=16 * world,
                world=world,
                rank=rank,
                device=device,
            )
            for step in range(3):
                optimizer.begin_step(step)
                for parameter in optimizer.matrices.values():
                    parameter.grad = torch.zeros_like(parameter)
                for index in range(16 // rows_per_microbatch):
                    optimizer.select_microbatch(index)
                    start, end = index * rows_per_microbatch, (index + 1) * rows_per_microbatch
                    for (layer, kind), buffer in optimizer.input_rows.items():
                        rows = synthetic_rows(rank, layer, INPUT_KINDS.index(kind), buffer.shape[1], step=step)
                        buffer[start:end].copy_(rows[start:end].to(device))
                    for (layer, kind), buffer in optimizer.output_rows.items():
                        rows = synthetic_rows(rank, layer, MATRIX_KINDS.index(kind), buffer.shape[1], step=step)
                        buffer[start:end].copy_(rows[start:end].to(device) * ROW_SCALE)
                optimizer.finish_backward()
                state = optimizer.runtime_state()["statistics"]
                for name, kinds in (("input_moments", INPUT_KINDS), ("output_moments", MATRIX_KINDS)):
                    for (layer, kind), moment in state[name].items():
                        if layer != rank:
                            raise AssertionError("GO-MUON statistic ownership differs from contiguous two-rank routing")
                        width = moment.shape[0]
                        observations = [
                            reference_moment(layer, kinds.index(kind), width, world, t) for t in range(step + 1)
                        ]
                        if mode == "ema_per_step":
                            expected = sum(0.5 ** (step + 1 - t) * value for t, value in enumerate(observations))
                        elif name == "input_moments":
                            expected = reference_shrink(observations[step - step % 2], 16 * world)
                        else:
                            first = reference_shrink(observations[0], 16 * world)
                            expected = (
                                0.97**2 * first + (1 - 0.97**2) * reference_shrink(observations[2], 16 * world)
                                if step == 2
                                else first
                            )
                        torch.testing.assert_close(moment.cpu().double(), expected, rtol=1e-5, atol=1e-7)
                        max_error = max(max_error, (moment.cpu().double() - expected).abs().max().item())
                        comparisons += 1
                optimizer.step()
            snapshots.append(optimizer.runtime_state()["statistics"])
        for name in ("input_moments", "output_moments", "input_factors", "output_factors"):
            for key, moment in snapshots[0][name].items():
                torch.testing.assert_close(snapshots[1][name][key], moment, rtol=1e-5, atol=1e-7)
    return {
        "go_muon_gpu_statistics_max_absolute_error": max_error,
        "go_muon_gpu_statistics_comparisons": comparisons,
        "go_muon_gpu_statistics_profiles": "ema_per_step,paper_sparse",
    }


def relative_error(value, reference):
    denominator = reference.float().norm()
    return ((value.float() - reference.float()).norm() / denominator.clamp_min(1e-30)).item()


def capture_projection(module, inputs, output, *, key, positions, activations, cotangents):
    activations[key] = inputs[0][:, positions].reshape(-1, inputs[0].shape[-1]).detach()
    if output.requires_grad:
        output.register_hook(
            lambda gradient: cotangents.__setitem__(key, gradient[:, positions].reshape(-1, gradient.shape[-1]).detach())
        )


def check_compiled_row_observer(device):
    """Accept transparency against compiled plain linear maps and supplied cotangents."""
    generator = torch.Generator(device=device).manual_seed(811)
    positions = torch.tensor([0, 3, 9, 15], device=device)
    max_forward, max_gradient, max_sum = 0.0, 0.0, 0.0
    for outputs, inputs in ((32, 32), (128, 32), (32, 128)):
        linear = nn.Linear(inputs, outputs, bias=False).to(device)
        plain = copy.deepcopy(linear)
        rows = torch.zeros(16, outputs, dtype=torch.bfloat16, device=device)
        at = torch.arange(8, device=device)
        observed = torch.compile(
            RowObservedLinear(linear, rows, at, positions),
            options=COMPILE_OPTIONS,
            fullgraph=True,
            dynamic=False,
        )
        unobserved = torch.compile(plain, options=COMPILE_OPTIONS, fullgraph=True, dynamic=False)
        expected_rows = torch.zeros_like(rows)
        for microbatch in range(2):
            at.copy_(torch.arange(microbatch * 8, (microbatch + 1) * 8, device=device))
            x = torch.randn(2, 16, inputs, generator=generator, device=device)
            cotangent = torch.randn(2, 16, outputs, generator=generator, device=device, dtype=torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                actual, expected = observed(x), unobserved(x)
            torch.testing.assert_close(actual, expected, rtol=TRANSPARENCY_RTOL, atol=TRANSPARENCY_ATOL)
            actual.backward(cotangent)
            expected.backward(cotangent)
            torch.testing.assert_close(
                linear.weight.grad, plain.weight.grad, rtol=TRANSPARENCY_RTOL, atol=TRANSPARENCY_ATOL
            )
            expected_rows[microbatch * 8 : (microbatch + 1) * 8] = (
                cotangent[:, positions].reshape(8, outputs) * ROW_SCALE
            )
            max_forward = max(max_forward, (actual.detach().float() - expected.detach().float()).abs().max().item())
            max_gradient = max(max_gradient, (linear.weight.grad - plain.weight.grad).abs().max().item())
        torch.testing.assert_close(rows, expected_rows, rtol=0, atol=0)
        actual_cotangents, expected_cotangents = rows.float() / ROW_SCALE, expected_rows.float() / ROW_SCALE
        actual_sum = actual_cotangents.mT @ actual_cotangents
        expected_sum = expected_cotangents.mT @ expected_cotangents
        torch.testing.assert_close(actual_sum, expected_sum, rtol=TRANSPARENCY_RTOL, atol=TRANSPARENCY_ATOL)
        max_sum = max(max_sum, (actual_sum - expected_sum).abs().max().item())
        saved_rows, saved_gradient = rows.clone(), linear.weight.grad.clone()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual, expected = observed(x), unobserved(x)
        torch.testing.assert_close(actual, expected, rtol=TRANSPARENCY_RTOL, atol=TRANSPARENCY_ATOL)
        torch.testing.assert_close(rows, saved_rows, rtol=0, atol=0)
        torch.testing.assert_close(linear.weight.grad, saved_gradient, rtol=0, atol=0)
    return {
        "go_muon_gpu_compiled_row_observer": True,
        "go_muon_gpu_compiled_row_observer_shapes": 3,
        "go_muon_gpu_compiled_row_observer_forward_max_absolute_error": max_forward,
        "go_muon_gpu_compiled_row_observer_gradient_max_absolute_error": max_gradient,
        "go_muon_gpu_compiled_row_observer_sum_max_absolute_error": max_sum,
        "go_muon_gpu_compiled_row_observer_rows_exact": True,
    }


def check_compiled_capture(device, world, rank):
    observer_diagnostics = check_compiled_row_observer(device)
    config = small_config(GoMuonSettings(), world)
    positions = [0, 3, 9, 15, 31, 47, 63]
    rows_per_microbatch, capacity = len(positions), 2 * len(positions)
    candidate = Model(
        config["model"], positions=positions, row_capacity=capacity, rows_per_microbatch=rows_per_microbatch
    ).to(device)
    initialize_model(candidate, 43)
    reference, eager = Model(config["model"]).to(device), Model(config["model"]).to(device)
    reference.load_state_dict(candidate.state_dict())
    eager.load_state_dict(candidate.state_dict())
    optimizer = GoMuon(
        candidate,
        config,
        positions=positions,
        rows_per_microbatch=rows_per_microbatch,
        capacity=capacity,
        rows_total=capacity * world,
        world=world,
        rank=rank,
        device=device,
    )
    activation_rows, cotangent_rows = {}, {}
    index = torch.tensor(positions, device=device)
    handles = []
    for layer, block in enumerate(eager.body.blocks):
        for kind in MATRIX_KINDS:
            handles.append(
                getattr(block, kind).register_forward_hook(
                    lambda module, inputs, output, key=(layer, kind): capture_projection(
                        module,
                        inputs,
                        output,
                        key=key,
                        positions=index,
                        activations=activation_rows,
                        cotangents=cotangent_rows,
                    )
                )
            )
    compiled = torch.compile(candidate.body, options=COMPILE_OPTIONS, fullgraph=True, dynamic=False)
    unobserved = torch.compile(reference.body, options=COMPILE_OPTIONS, fullgraph=True, dynamic=False)
    max_gradient, max_gradient_absolute, max_forward_absolute = 0.0, 0.0, 0.0
    max_row_error, max_input_error = 0.0, 0.0
    try:
        optimizer.begin_step(0)
        generator = torch.Generator(device=device).manual_seed(942 + rank)
        for microbatch in range(2):
            optimizer.select_microbatch(microbatch)
            tokens = ((np.arange(64) + rank * 17 + microbatch * 3) % 256).astype(np.uint32).reshape(1, 64)
            tok, _, cu, meta = pack(tokens, 34, EOS=128001)
            tok, cu = tok.to(device), cu.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                actual, expected, hooked = compiled(tok, cu, meta), unobserved(tok, cu, meta), eager.body(tok, cu, meta)
            torch.testing.assert_close(actual, expected, rtol=TRANSPARENCY_RTOL, atol=TRANSPARENCY_ATOL)
            max_forward_absolute = max(max_forward_absolute, (actual.detach() - expected.detach()).abs().max().item())
            target = torch.randn(actual.shape, generator=generator, device=device)
            for value in (actual, expected, hooked):
                (value.float() * target).sum().div(2).backward()
            for (name, parameter), (_, original) in zip(
                candidate.named_parameters(), reference.named_parameters(), strict=True
            ):
                if parameter.grad is not None or original.grad is not None:
                    torch.testing.assert_close(
                        parameter.grad, original.grad, rtol=TRANSPARENCY_RTOL, atol=TRANSPARENCY_ATOL, msg=name
                    )
                    max_gradient = max(max_gradient, relative_error(parameter.grad, original.grad))
                    max_gradient_absolute = max(
                        max_gradient_absolute, (parameter.grad - original.grad).abs().max().item()
                    )
            start, end = microbatch * rows_per_microbatch, (microbatch + 1) * rows_per_microbatch
            # Eager and compiled kernels can accumulate BF16 differently. These
            # comparisons are diagnostics only; the supplied-cotangent observer
            # check above and compiled-to-compiled parity define acceptance.
            for key, buffer in optimizer.output_rows.items():
                error = relative_error(buffer[start:end].float() / ROW_SCALE, cotangent_rows[key])
                max_row_error = max(max_row_error, error)
            for (layer, kind), buffer in optimizer.input_rows.items():
                projection = {"attn": "q", "o": "o", "mlp": "gate", "down": "down"}[kind]
                error = relative_error(buffer[start:end], activation_rows[layer, projection])
                max_input_error = max(max_input_error, error)
        optimizer.finish_backward()
        saved = optimizer.runtime_state()
        saved_rows = {key: buffer.clone() for key, buffer in optimizer.output_rows.items()}
        gradients = {
            name: parameter.grad.clone()
            for name, parameter in candidate.named_parameters()
            if parameter.grad is not None
        }
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual, expected = compiled(tok, cu, meta), unobserved(tok, cu, meta)
        torch.testing.assert_close(actual, expected, rtol=TRANSPARENCY_RTOL, atol=TRANSPARENCY_ATOL)
        after = optimizer.runtime_state()
        for name in ("input_moments", "output_moments", "input_factors", "output_factors"):
            for key, moment in saved["statistics"][name].items():
                torch.testing.assert_close(after["statistics"][name][key], moment, rtol=0, atol=0)
        for key, buffer in saved_rows.items():
            torch.testing.assert_close(optimizer.output_rows[key], buffer, rtol=0, atol=0)
        for name, parameter in candidate.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
        optimizer.step()
    finally:
        for handle in handles:
            handle.remove()
    return {
        **observer_diagnostics,
        "go_muon_gpu_compiled_gradient_relative_error": max_gradient,
        "go_muon_gpu_compiled_gradient_max_absolute_error": max_gradient_absolute,
        "go_muon_gpu_compiled_forward_max_absolute_error": max_forward_absolute,
        "go_muon_gpu_eager_cotangent_relative_error_diagnostic_only": max_row_error,
        "go_muon_gpu_eager_input_relative_error_diagnostic_only": max_input_error,
        "go_muon_gpu_eager_rows_acceptance": "diagnostic_only_no_pass_threshold",
        "go_muon_gpu_compiled_transparency_rtol": TRANSPARENCY_RTOL,
        "go_muon_gpu_compiled_transparency_atol": TRANSPARENCY_ATOL,
        "go_muon_gpu_compiled_capture_model": "pinned_Model_body",
    }


def run_checks(device) -> dict:
    """Run checks on both ranks of an already-initialized A100/NCCL runner."""
    if not dist.is_initialized() or dist.get_world_size() != 2 or dist.get_backend() != "nccl":
        raise ValueError("GO-MUON GPU checks require an initialized two-rank NCCL runner")
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError("GO-MUON GPU checks require A100 SM80")
    world, rank = dist.get_world_size(), dist.get_rank()
    results = {}
    for key, check in (
        ("go_muon_gpu_numerics", lambda: check_numerics(device)),
        ("go_muon_gpu_statistics", lambda: check_statistics(device, world, rank)),
        ("go_muon_gpu_compiled_capture", lambda: check_compiled_capture(device, world, rank)),
    ):
        results.update(check())
        dist.barrier()
        results[key] = True
    return results
