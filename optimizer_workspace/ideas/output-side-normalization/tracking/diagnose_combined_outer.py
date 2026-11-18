# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only diagnostic of three-product outer-diagonal updates; never launch approval."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from benchmark_diagonal import source_definitions
from launch_wave5 import read, sha256
from prepare_fixed_age_tuning import require_source
from watch_wave5 import write

TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}


def combined_outer_delta(weights, ns_output, left_outer, right_outer, *, lr, weight_decay, unit_rms):
    """Combine decay and diagonal-left direction before the final normalized right product."""
    right = unit_rms(right_outer)
    left = unit_rms(left_outer)
    scale = max(1.0, weights.shape[-2] / weights.shape[-1]) ** 0.5
    mixed = (lr * weight_decay) * torch.bmm(weights, right) + (lr * scale) * (left[:, :, None] * ns_output)
    return -torch.bmm(mixed, right)


def combined_outer_update(
    weights, gradients, momenta, left, right, *, args, lr, weight_decay, muon, unit_rms, trace=None
):
    """Prototype the combined outer product while retaining the reference's NS operands."""
    left_inner, left_outer = left
    right_inner, right_outer = right
    if left_outer.ndim != 2 or right_outer.ndim != 3:
        raise ValueError("The combined diagnostic requires diagonal left and dense right outer factors")
    momenta = args.muon_momentum * momenta + gradients
    source = gradients + args.muon_momentum * momenta if args.muon_nesterov else momenta
    if right_inner is not None:
        source = torch.bmm(source, right_inner)
    if left_inner is not None:
        source = source * left_inner[:, :, None]
    direction = muon.newton_schulz_batched(source, args.muon_ns_steps, args.muon_eps, args.muon_ns_mode)
    if trace is not None:
        trace["ns_source"] = source
        trace["ns_output"] = direction
    delta = combined_outer_delta(
        weights, direction, left_outer, right_outer, lr=lr, weight_decay=weight_decay, unit_rms=unit_rms
    )
    return momenta, delta


def diagnostic(root: Path) -> dict:
    """Reuse the frozen equations on CPU fixtures, retaining source and hypothetical cost evidence."""
    release = root / "run-preparation/wave7-release-binding.json"
    frozen = require_source(root, release)
    candidate = Path(frozen["candidate"])
    config_path = candidate / "experiment/configs/outer_leftdiag_fused_0p5m_seed0.json"
    config = read(config_path)
    upstream = read(candidate / "experiment/upstream_manifest.json")
    muon_path = candidate / f"artifacts/external-sources/Toner-{upstream['revision']}/src/toner/fast/muon.py"
    block_path = candidate / "experiment/block_toner.py"
    muon = source_definitions(muon_path, ("ABC", "_bmm", "newton_schulz_batched"), {"torch": torch})
    block = source_definitions(
        block_path,
        ("stack_unit_rms", "block_update_stack"),
        {"torch": torch, "MU": SimpleNamespace(**muon)},
    )
    optimizer = config["optimizer"]
    args = SimpleNamespace(
        muon_momentum=optimizer["momentum"],
        muon_nesterov=optimizer["nesterov"],
        muon_ns_steps=optimizer["ns_steps"],
        muon_eps=optimizer["ns_eps"],
        muon_ns_mode="f32",
    )
    generator = torch.Generator().manual_seed(27)
    cases = []
    for rows, columns in ((4, 4), (8, 4), (4, 8)):
        for pattern in ("gaussian", "zero_gradient", "anisotropic"):
            for lr in (0.0, optimizer["lr"], optimizer["lr"] / 32):
                shape = (3, rows, columns)
                weights = torch.randn(shape, generator=generator) * 0.02
                gradients = torch.randn(shape, generator=generator)
                momentum = torch.randn(shape, generator=generator) * 0.1
                inner = torch.randn(3, columns, columns, generator=generator)
                outer = torch.randn(3, columns, columns, generator=generator)
                identity = torch.eye(columns)[None]
                inner = inner @ inner.mT / columns + identity
                outer = outer @ outer.mT / columns + identity
                left = torch.rand(3, rows, generator=generator) + 0.25
                if pattern == "zero_gradient":
                    gradients.zero_()
                    momentum.zero_()
                elif pattern == "anisotropic":
                    left *= torch.logspace(-2, 2, rows)
                    outer *= torch.logspace(-2, 2, columns)[None, :, None]
                    outer = outer @ outer.mT + identity
                trace = {}
                _, expected = block["block_update_stack"](
                    weights,
                    gradients,
                    momentum,
                    (None, left),
                    (inner, outer),
                    args=args,
                    lr=lr,
                    weight_decay=optimizer["weight_decay"],
                    trace=trace,
                )
                actual = combined_outer_delta(
                    weights,
                    trace["ns_output"],
                    left,
                    outer,
                    lr=lr,
                    weight_decay=optimizer["weight_decay"],
                    unit_rms=block["stack_unit_rms"],
                )
                torch.testing.assert_close(actual, expected, **TOLERANCE)
                difference = actual - expected
                cases.append(
                    {
                        "shape": list(shape),
                        "pattern": pattern,
                        "lr": lr,
                        "max_absolute_error": difference.abs().max().item(),
                        "relative_frobenius_error": (
                            (difference.norm() / expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)).item()
                        ),
                        "post_ns_comparison_passed": True,
                    }
                )
    model, training = config["model"], config["training"]
    hidden, mlp = model["hidden"], model["mlp"]
    owned_layers, remainder = divmod(model["layers"], training["world_size"])
    if remainder:
        raise ValueError("Cost description requires the executed equal layer ownership")
    groups = [(4 * owned_layers, hidden, hidden), (2 * owned_layers, mlp, hidden), (owned_layers, hidden, mlp)]
    return {
        "status": "cpu_combined_outer_diagnostic_not_launch_gate",
        "source_release_sha256": sha256(release),
        "sources_sha256": {str(path): sha256(path) for path in (config_path, block_path, muon_path, Path(__file__))},
        "device": "cpu",
        "ns_mode": "f32",
        "production_ns_mode": optimizer["ns_mode"],
        "tolerance": TOLERANCE,
        "cases": cases,
        "cost_hypothesis": {
            "owned_matrix_shapes": [list(group) for group in groups],
            "right_dense_products_per_matrix": {"executed_block_path": 4, "proposed_combination": 3},
            "one_product_flops_per_rank_update": sum(2 * count * rows * columns**2 for count, rows, columns in groups),
            "scope": "Algebraic multiply count only; not measured latency, memory or total optimizer speedup.",
        },
        "scope": (
            "Only CPU FP32 post-NS arithmetic fixtures. Uses the reference's actual NS output unchanged; "
            "no full trainer, production tpu parity, distributed statistics, cached-root continuation or GPU timing. "
            "No frozen candidate, helper, configuration or tolerance is changed. Not numerical or launch approval."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve previous diagnostic evidence; use a new observation path")
    result = diagnostic(args.root)
    write(args.out, result)
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]), "receipt": str(args.out)}))


if __name__ == "__main__":
    main()
