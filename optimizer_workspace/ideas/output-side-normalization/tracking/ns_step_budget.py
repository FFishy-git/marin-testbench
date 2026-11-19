# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only NS step-budget study; changed finite maps require separate GPU and training gates."""

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path

import ns_operand_precision as precision
import replay_saved_ns as replay
import torch
from benchmark_diagonal import source_definitions
from diagnose_fused_diagonal import MOMENT_TOLERANCE, differences
from watch_wave5 import write

STEPS = (3, 5, 7, 10)
MODES = ("f32", "bf16_1")
LIMITATION = (
    "CPU study of changed finite NS maps on one authenticated historical rank0/update23/occurrence0 pair. "
    "BF16 operands use CPU FP32 accumulation; CUDA reduction and execution differ. SVD is a geometric "
    "reference, not the optimizer target. Product counts are arithmetic estimates, not measured speed. "
    "No CUDA owner, causal correction, continuation, quality, efficiency, SwanLab or production acceptance."
)


def spectral_references(source: torch.Tensor, coefficients: tuple, eps: float) -> tuple[dict, dict]:
    """Use FP64 SVD and scalar polynomial recurrence as an independent finite-map reference.

    The partial polar sets numerically null singular directions to zero at
    machine epsilon times matrix dimension times largest singular value.
    Every finite-polynomial reference retains all computed singular values.
    """
    precision.require_cpu((source,))
    if not torch.isfinite(source).all():
        raise ValueError("NS step reference requires finite full CPU batches")
    a, b, c = coefficients
    collected = {step: [] for step in STEPS}
    polars, spectra = [], []
    for matrix in source:
        matrix = matrix.double()
        left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        threshold = torch.finfo(torch.float64).eps * max(matrix.shape) * singular.amax()
        active = singular > threshold
        polars.append((left * active) @ right)
        normalized = singular / (matrix.norm() + eps)
        value = normalized.clone()
        record = {
            "singular_values": singular.tolist(),
            "normalized_singular_values": normalized.tolist(),
            "numerical_rank": int(active.sum()),
            "partial_polar_threshold": threshold.item(),
            "polynomial_singular_values": {},
        }
        for step in range(1, max(STEPS) + 1):
            value = a * value + b * value.pow(3) + c * value.pow(5)
            if step in STEPS:
                collected[step].append((left * value) @ right)
                record["polynomial_singular_values"][str(step)] = value.tolist()
        spectra.append(record)
    return (
        {"partial_polar": torch.stack(polars), **{str(step): torch.stack(value) for step, value in collected.items()}},
        {
            "matrices": spectra,
            "rank_convention": "sigma > eps_fp64 * max(rows, columns) * sigma_max; null polar directions are zero",
        },
    )


def geometry(output: torch.Tensor, source: torch.Tensor, polar: torch.Tensor) -> dict:
    """Report direction length, cosine and descent alignment against an SVD-based partial polar."""
    metrics = []
    for matrix, original, ideal in zip(output, source, polar, strict=True):
        actual, original, ideal = matrix.double(), original.double(), ideal.double()
        actual_norm, ideal_norm = actual.norm(), ideal.norm()
        denominator = actual_norm * ideal_norm
        optimal = (original * ideal).sum()
        scaled = actual * ideal_norm / actual_norm if actual_norm > 0 else actual
        metrics.append(
            {
                "output_norm": actual_norm.item(),
                "partial_polar_norm": ideal_norm.item(),
                "cosine_to_partial_polar": ((actual * ideal).sum() / denominator).item() if denominator > 0 else None,
                "equal_norm_descent_ratio": ((original * scaled).sum() / optimal).item() if optimal > 0 else None,
                "equal_norm_relative_polar_error": (
                    ((scaled - ideal).norm() / ideal_norm).item() if ideal_norm > 0 else None
                ),
                "raw_relative_polar_error": ((actual - ideal).norm() / ideal_norm).item() if ideal_norm > 0 else None,
            }
        )
    return {"per_matrix": metrics, "norm_grafting": "diagnostic metric only; outputs and optimizer remain ungrafted"}


def run(out: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or out.exists():
        raise RuntimeError("NS budget study requires hidden CUDA and an unused output directory")
    out.mkdir(parents=True)
    source_paths = {
        Path(__file__).resolve(),
        Path(precision.__file__).resolve(),
        Path(replay.__file__).resolve(),
        Path(replay.analyzer.__file__).resolve(),
        Path(replay.analyzer.compare_saved.__code__.co_filename).resolve(),
        Path(source_definitions.__code__.co_filename).resolve(),
        Path(differences.__code__.co_filename).resolve(),
    }
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    result = {
        "status": "cpu_ns_step_budget_running",
        "steps": list(STEPS),
        "modes": list(MODES),
        "eps": replay.NS_EPS,
        "tolerance": MOMENT_TOLERANCE,
        "source_sha256": source_hashes,
        "trials": [],
        "scope": LIMITATION,
    }
    receipt = out / "result.json"
    write(receipt, result)
    try:
        torch.set_num_threads(8)
        with torch.no_grad(), ExitStack() as stack:
            for path, digest in source_hashes.items():
                stack.enter_context(replay.analyzer._bound_file(Path(path), digest))
            groups, provenance = replay.load_saved_group(stack)
            stream = stack.enter_context(
                replay.analyzer._bound_file(Path(provenance["pinned_muon"]), provenance["pinned_muon_sha256"])
            )
            path = Path(f"/proc/self/fd/{stream.fileno()}")
            coefficients = source_definitions(path, ("ABC",), {})["ABC"]
            references, spectra = {}, {}
            for source, group in groups.items():
                references[source], spectra[source] = spectral_references(group["source"], coefficients, replay.NS_EPS)
            reference_path = out / "spectral-references.pt"
            with reference_path.open("xb") as output:
                torch.save(references, output)
            result.update(
                provenance=provenance,
                coefficients=list(coefficients),
                reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                spectra=spectra,
                fp64_pairs={
                    str(step): differences(
                        references["resumed"][str(step)], references["continuous"][str(step)], MOMENT_TOLERANCE
                    )
                    for step in STEPS
                },
            )
            write(receipt, result)
            for mode in MODES:
                baseline = precision.pinned_numerical(path, mode, steps=replay.NS_STEPS, eps=replay.NS_EPS)
                baseline_outputs = {source: baseline(group["source"]) for source, group in groups.items()}
                for step in STEPS:
                    numerical = precision.pinned_numerical(path, mode, steps=step, eps=replay.NS_EPS)
                    trial = replay.replay_pair(groups, numerical)
                    output_path = out / f"{mode}-q{step}.pt"
                    with output_path.open("xb") as output:
                        torch.save(trial.outputs, output)
                    result["trials"].append(
                        {
                            "mode": mode,
                            "steps": step,
                            "products": 3 * step,
                            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                            "replay": trial.report,
                            "vs_svd_polynomial": {
                                source: differences(
                                    trial.outputs[index], references[source][str(step)], MOMENT_TOLERANCE
                                )
                                for index, source in enumerate(replay.ORDER[:2])
                            },
                            "vs_same_mode_ten_steps": {
                                source: replay.compare_outputs(trial.outputs[index], baseline_outputs[source])
                                for index, source in enumerate(replay.ORDER[:2])
                            },
                            "geometry": {
                                source: geometry(
                                    trial.outputs[index], groups[source]["source"], references[source]["partial_polar"]
                                )
                                for index, source in enumerate(replay.ORDER[:2])
                            },
                        }
                    )
                    write(receipt, result)
                    pair = trial.report["cross_outputs"][0]
                    print(
                        f"{mode} q={step}: pair_error={pair['relative_frobenius_error']} "
                        f"mismatches={pair['mismatched_elements']}",
                        flush=True,
                    )
        result["status"] = "completed_cpu_ns_step_budget_not_cuda_or_optimizer_acceptance"
        write(receipt, result)
        return result
    except BaseException as error:
        result.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        write(receipt, result)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.out.resolve())["status"]}), flush=True)


if __name__ == "__main__":
    main()
