# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only operand-splitting study of the source-bound pinned NS polynomial."""

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import replay_saved_ns as replay
import torch
from benchmark_diagonal import source_definitions
from diagnose_fused_diagonal import MOMENT_TOLERANCE, differences
from watch_wave5 import write

MODES = ("f32", "bf16_1", "bf16_3", "bf16_6")
PRODUCTS = {"f32": 1, "bf16_1": 1, "bf16_3": 3, "bf16_6": 6}
REFERENCE = "Henry, Tang and Heinecke, arXiv:1904.06376v1 (2019), Sections II-A and II-G."


def require_cpu(values: tuple[torch.Tensor, ...]) -> None:
    if any(value.device.type != "cpu" for value in values):
        raise RuntimeError("Operand precision study is CPU-only; no CUDA execution is issued")
    if any(value.dtype != torch.float32 or value.ndim != 3 for value in values):
        raise ValueError("Operand precision study requires batched FP32 inputs")


def parts(value: torch.Tensor, count: int) -> tuple[torch.Tensor, ...]:
    """Represent an FP32 CPU tensor by consecutively rounded BF16 residuals."""
    require_cpu((value,))
    residual, result = value, []
    for _ in range(count):
        component = residual.to(torch.bfloat16).float()
        result.append(component)
        residual = residual - component
    return tuple(result)


def operand_bmm(left: torch.Tensor, right: torch.Tensor, mode: str) -> torch.Tensor:
    """Emulate BF16 operand products with CPU FP32 accumulation, retaining low-order terms."""
    require_cpu((left, right))
    if mode == "f32":
        return torch.bmm(left, right)
    count = {"bf16_1": 1, "bf16_3": 2, "bf16_6": 3}[mode]
    x, y = parts(left, count), parts(right, count)
    leading = torch.bmm(x[0], y[0])
    if mode == "bf16_1":
        return leading
    first = torch.bmm(x[0], y[1]) + torch.bmm(x[1], y[0])
    if mode == "bf16_3":
        return leading + first
    second = torch.bmm(x[0], y[2]) + (torch.bmm(x[1], y[1]) + torch.bmm(x[2], y[0]))
    return leading + (first + second)


def pinned_numerical(path: Path, mode: str, *, steps: int, eps: float):
    """Keep the pinned normalization/polynomial and replace only its batched product primitive."""
    namespace = {"torch": torch}
    definitions = source_definitions(path, ("ABC", "_bmm", "newton_schulz_batched"), namespace)
    namespace["_bmm"] = operand_bmm
    return partial(definitions["newton_schulz_batched"], steps=steps, eps=eps, mode=mode)


def reference_fp64(source: torch.Tensor, coefficients: tuple, *, steps: int, eps: float) -> torch.Tensor:
    """Evaluate each matrix independently in FP64; this is the finite polynomial, not exact polar."""
    require_cpu((source,))
    outputs = []
    a, b, c = coefficients
    for matrix in source:
        x = matrix.double()
        x = x / (x.norm() + eps)
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.T
        for _ in range(steps):
            gram = x @ x.T
            correction = b * gram + c * (gram @ gram)
            x = a * x + correction @ x
        outputs.append(x.T if transposed else x)
    return torch.stack(outputs)


def run(out: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or out.exists():
        raise RuntimeError("CPU operand study requires hidden CUDA and an unused output directory")
    out.mkdir(parents=True)
    source_paths = {
        Path(__file__).resolve(),
        Path(replay.__file__).resolve(),
        Path(replay.analyzer.__file__).resolve(),
        Path(replay.analyzer.compare_saved.__code__.co_filename).resolve(),
        Path(source_definitions.__code__.co_filename).resolve(),
        Path(differences.__code__.co_filename).resolve(),
    }
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    result = {
        "status": "cpu_operand_precision_study_running",
        "literature_metadata": REFERENCE,
        "literature_validation": "Full paper text unavailable; no paper-specific implementation or accuracy claim.",
        "source_sha256": source_hashes,
        "tolerance": MOMENT_TOLERANCE,
        "modes": {},
        "scope": (
            "CPU BF16 operand emulation and FP64 finite-polynomial reference on authenticated historical "
            "rank0/update23/occurrence0. CPU accumulation/reduction differ from CUDA. No causal, continuation, "
            "runtime, optimizer-quality, efficiency, SwanLab or production acceptance."
        ),
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
            references = {
                mode: reference_fp64(group["source"], coefficients, steps=replay.NS_STEPS, eps=replay.NS_EPS)
                for mode, group in groups.items()
            }
            result.update(provenance=provenance, coefficients=list(coefficients))
            reference_path = out / "fp64-reference.pt"
            with reference_path.open("xb") as output:
                torch.save(references, output)
            result["reference_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            result["fp64_source_pair"] = differences(references["resumed"], references["continuous"], MOMENT_TOLERANCE)
            write(receipt, result)
            for mode in MODES:
                numerical = pinned_numerical(path, mode, steps=replay.NS_STEPS, eps=replay.NS_EPS)
                trial = replay.replay_pair(groups, numerical)
                outputs_path = out / f"{mode}.pt"
                with outputs_path.open("xb") as output:
                    torch.save(trial.outputs, output)
                result["modes"][mode] = {
                    "replay": trial.report,
                    "output_sha256": hashlib.sha256(outputs_path.read_bytes()).hexdigest(),
                    "vs_fp64": {
                        source: differences(trial.outputs[index], references[source], MOMENT_TOLERANCE)
                        for index, source in enumerate(replay.ORDER[:2])
                    },
                    "products_per_ns_iteration": 3 * PRODUCTS[mode],
                    "omitted_product_levels": (
                        "none"
                        if mode == "f32"
                        else (
                            "all residuals"
                            if mode == "bf16_1"
                            else (
                                "two-part low-low and remaining residuals"
                                if mode == "bf16_3"
                                else "three-part products with i+j >= 3"
                            )
                        )
                    ),
                }
                write(receipt, result)
                print(f"{mode}: output pair {trial.report['cross_outputs'][0]['relative_frobenius_error']}", flush=True)
        result["status"] = "completed_cpu_operand_precision_study_not_cuda_acceptance"
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
