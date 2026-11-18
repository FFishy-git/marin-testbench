# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the frozen diagonal backend's direction divergence without approving a launch."""

import argparse
import importlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from prepare_tuning import sha256

MOMENT_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
UPDATE_TOLERANCE = {"rtol": 0.003, "atol": 5e-5}


def differences(actual: torch.Tensor, expected: torch.Tensor, tolerance: dict) -> dict:
    """Preserve pointwise and norm errors, including the original gate's mismatch count."""
    comparison_dtype = torch.promote_types(actual.dtype, expected.dtype)
    close = torch.isclose(actual.to(comparison_dtype), expected.to(comparison_dtype), **tolerance)
    finite = torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item()
    if not finite:
        return {
            "elements": expected.numel(),
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "relative_frobenius_error": None,
            "reference_norm": None,
            "mismatched_elements": (~close).sum().item(),
            "within_tolerance": False,
            "finite": False,
            "actual_nonfinite_elements": (~torch.isfinite(actual)).sum().item(),
            "reference_nonfinite_elements": (~torch.isfinite(expected)).sum().item(),
        }
    delta = actual.double() - expected.double()
    norm = expected.double().norm().item()
    return {
        "elements": expected.numel(),
        "max_absolute_error": delta.abs().max().item(),
        "mean_absolute_error": delta.abs().mean().item(),
        "relative_frobenius_error": delta.norm().item() / norm if norm else None,
        "reference_norm": norm,
        "mismatched_elements": (~close).sum().item(),
        "within_tolerance": close.all().item(),
        "finite": finite,
    }


def structured_rows(rank: int, layer: int, offset: int, shape: tuple, step: int = 0) -> torch.Tensor:
    grid = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
    return torch.sin(grid * 0.013 + rank * 0.7 + layer * 0.1 + offset * 0.01 + step * 0.0001).to(torch.bfloat16)


def candidate_modules(binding: Path) -> tuple:
    frozen = json.loads(binding.read_text())
    source = Path(frozen["candidate"])
    if any(sha256(source / name) != digest for name, digest in frozen["files_sha256"].items()):
        raise ValueError("Frozen source changed; diagnostic identity is invalid")
    sys.path.insert(0, str(source))
    block = importlib.import_module("experiment.block_toner")
    configure = importlib.import_module("experiment.configure")
    factors = importlib.import_module("experiment.factors")
    optimizer = importlib.import_module("reproduce.marin.optimizer")
    for module in (block, configure, factors):
        if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            raise ValueError("Diagnostic imported a different experiment source")
    return source, block, configure, factors, optimizer


def source_and_trace(weights, gradients, momentum, left, right, args) -> tuple:
    source = gradients + args.muon_momentum * (args.muon_momentum * momentum + gradients)
    if not args.muon_nesterov:
        source = args.muon_momentum * momentum + gradients
    if right[0] is not None:
        source = torch.bmm(source, right[0])
    source = source * left[0][:, :, None]
    singular = torch.linalg.svdvals(source[0].double()).cpu()
    return source, {
        "first_matrix_largest_singular_value": singular[0].item(),
        "first_matrix_smallest_singular_value": singular[-1].item(),
        "first_matrix_condition_number": (singular[0] / singular[-1]).item() if singular[-1] > 0 else None,
        "weights_shape": list(weights.shape),
    }


def compare_stack(block, weight, gradient, left, reference_left, right, args) -> dict:
    momentum = torch.zeros_like(weight)
    actual_source, condition = source_and_trace(weight, gradient, momentum, left, right, args)
    reference_source, _ = source_and_trace(weight, gradient, momentum, reference_left, right, args)
    modes = {}
    for mode in (args.muon_ns_mode, "f32"):
        settings = SimpleNamespace(**(vars(args) | {"muon_ns_mode": mode}))
        _, actual = block.block_update_stack(
            weight, gradient, momentum, left, right, args=settings, lr=0.02, weight_decay=0.1
        )
        _, expected = block.block_update_stack(
            weight, gradient, momentum, reference_left, right, args=settings, lr=0.02, weight_decay=0.1
        )
        iterations = []
        for step in range(1, args.muon_ns_steps + 1):
            actual_ns = block.MU.newton_schulz_batched(actual_source, step, args.muon_eps, mode)
            expected_ns = block.MU.newton_schulz_batched(reference_source, step, args.muon_eps, mode)
            iterations.append({"iteration": step, **differences(actual_ns, expected_ns, UPDATE_TOLERANCE)})
        modes[mode] = {"parameter_change": differences(actual, expected, UPDATE_TOLERANCE), "iterations": iterations}
    return {
        "condition": condition,
        "left_inner": differences(left[0], reference_left[0], MOMENT_TOLERANCE),
        "left_outer": differences(left[1], reference_left[1], MOMENT_TOLERANCE),
        "preconditioned_source": differences(actual_source, reference_source, MOMENT_TOLERANCE),
        "preconditioned_bf16_mismatches": (actual_source.bfloat16() != reference_source.bfloat16()).sum().item(),
        "modes": modes,
    }


def fp64_diagonal_factors(moment: torch.Tensor, damping: float, inner: float, outer: float) -> tuple:
    """Independent FP64 per-coordinate inverse powers, before outer RMS normalization."""
    total = moment.double().sum(-1, keepdim=True)
    normalized = torch.where(total > 0, moment.double() / total.clamp_min(torch.finfo(torch.float64).tiny), 0)
    spectrum = normalized * moment.shape[-1] + damping
    return tuple(torch.where(total > 0, spectrum.pow(-power), 1) for power in (inner, outer))


def operand_error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    return {
        **differences(actual, expected, UPDATE_TOLERANCE),
        "bf16_mismatched_elements": (actual.bfloat16() != expected.bfloat16()).sum().item(),
    }


def trace_sources(muon, actual_source: torch.Tensor, reference_source: torch.Tensor, args, mode: str) -> dict:
    """Inspect upstream NS prefixes and their exact GEMM operands without changing the iteration."""
    transposed = actual_source.shape[-2] > actual_source.shape[-1]
    normalized = (
        muon.newton_schulz_batched(actual_source, 0, args.muon_eps, mode),
        muon.newton_schulz_batched(reference_source, 0, args.muon_eps, mode),
    )
    iterations = []
    for step in range(1, args.muon_ns_steps + 1):
        inputs = [
            muon.newton_schulz_batched(source, step - 1, args.muon_eps, mode)
            for source in (actual_source, reference_source)
        ]
        if transposed:
            inputs = [value.transpose(-2, -1) for value in inputs]
        grams = [muon._bmm(value, value.transpose(-2, -1), mode) for value in inputs]
        _, b, c = muon.ABC
        polynomials = [b * gram + c * muon._bmm(gram, gram, mode) for gram in grams]
        outputs = [
            muon.newton_schulz_batched(source, step, args.muon_eps, mode) for source in (actual_source, reference_source)
        ]
        iterations.append(
            {
                "iteration": step,
                "input": operand_error(*inputs),
                "gram": operand_error(*grams),
                "polynomial": operand_error(*polynomials),
                "output": operand_error(*outputs),
            }
        )
    return {"normalized_source": operand_error(*normalized), "iterations": iterations}


def outer_direction(block, direction: torch.Tensor, left: tuple, right: tuple) -> torch.Tensor:
    if right[1] is not None:
        factor = block.stack_unit_rms(right[1])
        direction = torch.bmm(direction, factor) if factor.ndim == 3 else direction * factor[:, None, :]
    if left[1] is not None:
        factor = block.stack_unit_rms(left[1])
        direction = torch.bmm(factor, direction) if factor.ndim == 3 else direction * factor[:, :, None]
    return direction


def sensitivity_stack(block, weight, gradient, left, reference_left, right, args) -> dict:
    momentum = torch.zeros_like(weight)
    reference_source, _ = source_and_trace(weight, gradient, momentum, reference_left, right, args)
    swaps = {
        "identical_repeat": reference_left,
        "inner_only": (left[0], reference_left[1]),
        "outer_only": (reference_left[0], left[1]),
        "both": left,
    }
    modes = {}
    for mode in (args.muon_ns_mode, "f32"):
        settings = SimpleNamespace(**(vars(args) | {"muon_ns_mode": mode}))
        baseline = block.block_update_stack(
            weight, gradient, momentum, reference_left, right, args=settings, lr=0.02, weight_decay=0.1
        )[1]
        controls = {}
        for name, factors in swaps.items():
            actual = block.block_update_stack(
                weight, gradient, momentum, factors, right, args=settings, lr=0.02, weight_decay=0.1
            )[1]
            source, _ = source_and_trace(weight, gradient, momentum, factors, right, args)
            controls[name] = {
                "source": differences(source, reference_source, MOMENT_TOLERANCE),
                "parameter_change": differences(actual, baseline, UPDATE_TOLERANCE),
            }
            if name in ("inner_only", "both"):
                controls[name]["trace"] = trace_sources(block.MU, source, reference_source, args, mode)
        perturbed = torch.nextafter(reference_source, torch.full_like(reference_source, float("inf")))
        original_ns = block.MU.newton_schulz_batched(reference_source, args.muon_ns_steps, args.muon_eps, mode)
        changed_ns = block.MU.newton_schulz_batched(perturbed, args.muon_ns_steps, args.muon_eps, mode)
        change = outer_direction(block, changed_ns, reference_left, right) - outer_direction(
            block, original_ns, reference_left, right
        )
        scale = max(1.0, weight.shape[-2] / weight.shape[-1]) ** 0.5
        perturbed_update = baseline - 0.02 * scale * change
        controls["one_ulp_source"] = {
            "source": differences(perturbed, reference_source, MOMENT_TOLERANCE),
            "parameter_change": differences(perturbed_update, baseline, UPDATE_TOLERANCE),
            "trace": trace_sources(block.MU, perturbed, reference_source, args, mode),
        }
        modes[mode] = controls
    return {
        "interpretation": (
            "Controls only; identical repeat, separate inner/outer differences and one FP32 ULP toward +inf."
        ),
        "modes": modes,
    }


def diagnostic(binding: Path, device: torch.device, rank: int, world: int, profile: str = "parity") -> dict:
    source, block, configure, factor_module, optimizer = candidate_modules(binding)
    config = configure.configuration("inner_outer_leftdiag", "0p5m", 0, 4)
    args = optimizer.fast_arguments(config)
    cases = []
    for pattern, widths, capacity in (
        ("failed_fixture", (33, 65), 129),
        ("structured_production_shapes", (512, 2048), 4576),
        ("gaussian_production_shapes", (512, 2048), 4576),
    ):
        rows = {
            (layer, kind): torch.zeros(capacity, width, dtype=torch.bfloat16, device=device)
            for layer in range(2)
            for kind, width in zip(("q", "gate"), widths, strict=True)
        }
        settings = {
            "decay": 0.95 if pattern == "failed_fixture" else config["statistics"]["momentum"],
            "damping": config["statistics"]["damping"],
            "inner_power": config["statistics"]["inner_strength"],
            "outer_power": config["statistics"]["outer_strength"],
            "rows_total": capacity * world,
            "world": world,
            "device": device,
        }
        fused = block.DiagonalMoments(rows, backend="triton", **settings)
        reference = block.DiagonalMoments(rows, backend="torch", **settings)
        observation_errors = []
        oracle_moments = {width: torch.zeros_like(moment, dtype=torch.float64) for width, moment in fused.moment.items()}
        generator = torch.Generator().manual_seed(9127 + rank)
        for step in range(24):
            for (layer, kind), buffer in rows.items():
                if pattern.startswith("gaussian"):
                    values = torch.randn(buffer.shape, generator=generator).bfloat16()
                else:
                    values = structured_rows(rank, layer, block.BLOCK_KINDS.index(kind), buffer.shape, step)
                buffer.copy_(values.to(device))
            if profile == "sensitivity":
                for width, keys in fused.keys.items():
                    sums = torch.stack([rows[key].double().square().sum(0) for key in keys])
                    dist.all_reduce(sums)
                    oracle_moments[width] = (
                        settings["decay"] * oracle_moments[width]
                        + (1 - settings["decay"]) * sums / settings["rows_total"]
                    )
            fused.update()
            reference.update()
            if step in (0, 21, 23):
                observed = {
                    "update": step + 1,
                    "widths": {
                        str(width): differences(fused.moment[width], reference.moment[width], MOMENT_TOLERANCE)
                        for width in widths
                    },
                }
                if profile == "sensitivity":
                    observed["fp64_reference"] = {
                        str(width): {
                            "fused": differences(fused.moment[width].double(), oracle_moments[width], MOMENT_TOLERANCE),
                            "torch": differences(
                                reference.moment[width].double(), oracle_moments[width], MOMENT_TOLERANCE
                            ),
                        }
                        for width in widths
                    }
                observation_errors.append(observed)
        actual_factors, expected_factors = fused.factors(), reference.factors()
        if pattern == "failed_fixture":
            per_matrix = []
            for key, buffer in rows.items():
                width = buffer.shape[1]
                weight = structured_rows(rank, 1, 0, (width, 47)).to(device).float()
                gradient = structured_rows(rank, 1, 0, (width, 47), 99).to(device).float() * 0.01
                for mode in (args.muon_ns_mode, "f32"):
                    settings_args = SimpleNamespace(**(vars(args) | {"muon_ns_mode": mode}))
                    common = {"args": settings_args, "lr": 0.02, "weight_decay": 0.1}
                    actual = block.block_update(
                        weight,
                        gradient,
                        torch.zeros_like(weight),
                        block.MatrixFactors(*actual_factors[key]),
                        **common,
                    )[1]
                    expected = block.block_update(
                        weight,
                        gradient,
                        torch.zeros_like(weight),
                        block.MatrixFactors(*expected_factors[key]),
                        **common,
                    )[1]
                    per_matrix.append(
                        {"key": list(key), "mode": mode, **differences(actual, expected, UPDATE_TOLERANCE)}
                    )
        else:
            per_matrix = []
        stacks = []
        for width in widths:
            keys = [key for key in rows if rows[key].shape[1] == width]
            columns = 47 if pattern == "failed_fixture" else config["model"]["hidden"]
            left = tuple(torch.stack([actual_factors[key][index] for key in keys]) for index in range(2))
            reference_left = tuple(torch.stack([expected_factors[key][index] for key in keys]) for index in range(2))
            shape = (len(keys), width, columns)
            if pattern.startswith("gaussian"):
                weight = torch.randn(shape, generator=generator).to(device)
                gradient = torch.randn(shape, generator=generator).to(device) * 0.01
            else:
                weight = (
                    torch.stack([structured_rows(rank, key[0], 0, (width, columns)) for key in keys]).to(device).float()
                )
                gradient = (
                    torch.stack([structured_rows(rank, key[0], 0, (width, columns), 99) for key in keys])
                    .to(device)
                    .float()
                    * 0.01
                )
            if pattern == "failed_fixture":
                right = (None, None)
            else:
                pool = torch.randn(2 * columns, columns, generator=generator).to(device)
                moment = pool.T @ pool / len(pool)
                right_factors = factor_module.inverse_factors(moment, factor_module.OutputSettings())
                right = tuple(factor.expand(len(keys), -1, -1).contiguous() for factor in right_factors)
            stack = {
                "width": width,
                "right_factor_scope": "identity" if right[0] is None else "fixed dense SPD input factors",
                **compare_stack(block, weight, gradient, left, reference_left, right, args),
            }
            if profile == "sensitivity":
                oracle_left = fp64_diagonal_factors(
                    oracle_moments[width], settings["damping"], settings["inner_power"], settings["outer_power"]
                )
                stack["fp64_factor_reference"] = {
                    "fused": [
                        differences(actual.double(), expected, MOMENT_TOLERANCE)
                        for actual, expected in zip(left, oracle_left, strict=True)
                    ],
                    "torch": [
                        differences(actual.double(), expected, MOMENT_TOLERANCE)
                        for actual, expected in zip(reference_left, oracle_left, strict=True)
                    ],
                }
                stack["sensitivity"] = sensitivity_stack(block, weight, gradient, left, reference_left, right, args)
            stacks.append(stack)
        cases.append(
            {
                "pattern": pattern,
                "capacity_per_rank": capacity,
                "moment_settings": {key: value for key, value in settings.items() if key != "device"},
                "matrices_per_width_per_rank": 2,
                "observation_errors": observation_errors,
                "failed_fixture_per_matrix_replay": per_matrix,
                "stacks": stacks,
            }
        )
    return {
        "time": datetime.now(UTC).isoformat(),
        "rank": rank,
        "profile": profile,
        "world": world,
        "gpu": torch.cuda.get_device_name(device),
        "source": str(source),
        "binding_sha256": sha256(binding),
        "diagnostic_sha256": sha256(Path(__file__)),
        "production_mode": args.muon_ns_mode,
        "moment_tolerance": MOMENT_TOLERANCE,
        "update_tolerance": UPDATE_TOLERANCE,
        "scope": "Synthetic replay and production-shaped stack math; no compiled backward or training.",
        "interpretation": "Diagnostic only. Direction failures remain failures; FP32 controls do not approve launch.",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("parity", "sensitivity"), default="parity")
    args = parser.parse_args()
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 2 or args.out.exists():
        raise ValueError("Diagnostic requires a new two-rank output identity")
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("nccl")
    try:
        result = diagnostic(args.binding, device, rank, world, args.profile)
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        if rank == 0:
            payload = {
                "time": datetime.now(UTC).isoformat(),
                "status": "measured_diagnostic_not_launch_gate",
                "ranks": gathered,
            }
            with args.out.open("x") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(args.out)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
