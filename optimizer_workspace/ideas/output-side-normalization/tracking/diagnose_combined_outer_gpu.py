# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure separately bound combined-product production updates; never training approval."""

import argparse
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from benchmark_diagonal import source_definitions
from diagnose_combined_outer import TOLERANCE, combined_outer_update
from diagnose_fused_diagonal import differences
from launch_wave5 import read, sha256
from prepare_fixed_age_tuning import require_source
from watch_wave5 import process, write
from wave7_release import PROCESS_MARKER

WARMUP = 5
REPEATS = 15


def timing(function, *, graphs: bool) -> dict:
    """Time eager calls or replayed graphs using CUDA events, excluding setup."""
    for _ in range(WARMUP):
        function()
    torch.cuda.synchronize()
    graph = None
    if graphs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = function()
        function = graph.replay
    samples = []
    for _ in range(REPEATS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    if graph is not None:
        del output, graph
    return {"median_ms": statistics.median(samples), "samples_ms": samples, "warmup": WARMUP, "repeats": REPEATS}


def require_owner(out: Path) -> dict:
    """Require the exact controller generation and marked child before initializing CUDA."""
    deadline = time.monotonic() + 10
    while True:
        state = read(out / "state.json")
        if "child_pid" in state:
            break
        owner = process(state["controller_pid"])
        if owner is None or owner[0] == "Z" or time.monotonic() >= deadline:
            raise RuntimeError("Combined-product GPU controller has no child acknowledgment")
        time.sleep(0.05)
    owner = process(state["controller_pid"])
    argv = (Path("/proc") / str(state["controller_pid"]) / "cmdline").read_bytes().split(b"\0")
    expected = [
        state["python"],
        str(out / "helpers/run_combined_outer_gpu.py"),
        "--root",
        state["root"],
        "--out",
        str(out),
        "--phase",
        "run",
    ]
    if (
        owner is None
        or owner[0] == "Z"
        or owner[1] != state["controller_start_ticks"]
        or [item.decode() for item in argv if item] != expected
        or os.getppid() != state["controller_pid"]
        or state["child_pid"] != os.getpid()
        or os.environ.get(PROCESS_MARKER) != state["child_marker"]
        or state["status"] != "running"
    ):
        raise RuntimeError("Combined-product GPU worker has no live bound controller ownership")
    if Path(__file__).resolve() != (out / "helpers/diagnose_combined_outer_gpu.py").resolve():
        raise ValueError("Combined-product GPU worker must execute the exact bound copy")
    for name, digest in state["helpers_sha256"].items():
        if sha256(out / "helpers" / name) != digest:
            raise ValueError("Combined-product GPU helper identity changed")
    return state


def frozen_equations(root: Path) -> tuple:
    release = root / "run-preparation/wave7-release-binding.json"
    frozen = require_source(root, release)
    source = Path(frozen["candidate"])
    config_path = source / "experiment/configs/outer_leftdiag_fused_0p5m_seed0.json"
    config = read(config_path)
    upstream = read(source / "experiment/upstream_manifest.json")
    ns = source / f"artifacts/external-sources/Toner-{upstream['revision']}/src/toner/fast/muon.py"
    block = source / "experiment/block_toner.py"
    muon = SimpleNamespace(**source_definitions(ns, ("ABC", "_bmm", "newton_schulz_batched"), {"torch": torch}))
    equations = source_definitions(block, ("stack_unit_rms", "block_update_stack"), {"torch": torch, "MU": muon})
    return config, muon, equations, {str(path): sha256(path) for path in (release, config_path, ns, block)}


def fixture(shape: tuple, pattern: str, device: torch.device) -> tuple:
    generator = torch.Generator(device=device).manual_seed(73)
    count, rows, columns = shape
    weights = torch.randn(shape, generator=generator, device=device) * 0.02
    gradients = torch.randn(shape, generator=generator, device=device)
    momenta = torch.randn(shape, generator=generator, device=device) * 0.1
    matrix = torch.randn(count, columns, columns, generator=generator, device=device) / columns**0.5
    identity = torch.eye(columns, device=device)[None]
    right_inner = matrix @ matrix.mT + identity
    right_outer = matrix.mT @ matrix + identity
    left = torch.rand(count, rows, generator=generator, device=device) + 0.25
    if pattern == "zero":
        gradients.zero_()
        momenta.zero_()
    elif pattern == "anisotropic":
        left *= torch.logspace(-2, 2, rows, device=device)
        scale = torch.logspace(-1, 1, columns, device=device)
        right_outer *= scale[None, :, None] * scale[None, None, :]
    return weights, gradients, momenta, left, (right_inner, right_outer)


def diagnostic(root: Path, out: Path) -> dict:
    owner = require_owner(out)
    config, muon, equations, sources = frozen_equations(root)
    optimizer = config["optimizer"]
    args = SimpleNamespace(
        muon_momentum=optimizer["momentum"],
        muon_nesterov=optimizer["nesterov"],
        muon_ns_steps=optimizer["ns_steps"],
        muon_eps=optimizer["ns_eps"],
        muon_ns_mode=optimizer["ns_mode"],
    )
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", 0)
    model, training = config["model"], config["training"]
    owned, remainder = divmod(model["layers"], training["world_size"])
    if remainder:
        raise ValueError("Unequal layer ownership needs a separate workload")
    shapes = (
        (4 * owned, model["hidden"], model["hidden"]),
        (2 * owned, model["mlp"], model["hidden"]),
        (owned, model["hidden"], model["mlp"]),
    )
    result = {
        "status": "production_combined_outer_diagnostic_running",
        "owner": {"controller_pid": owner["controller_pid"], "controller_start_ticks": owner["controller_start_ticks"]},
        "sources_sha256": sources,
        "helpers_sha256": owner["helpers_sha256"],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ns_mode": args.muon_ns_mode,
        "parameter_delta_tolerance": TOLERANCE,
        "exact_fields": ["momentum", "ns_source", "ns_output"],
        "cases": [],
        "timing": [],
        "scope": (
            "Synthetic owned production-shape stacks on one A100. Exact reference NS operands/output, "
            "unchanged parameter-delta tolerance; eager and CUDA-graph GPU event timings only. "
            "Excludes capture, statistics, communication, factor computation, full model, continuation and loss. "
            "A passed diagnostic does not approve training or establish whole-training efficiency."
        ),
    }
    path = out / "result.json"
    write(path, result)
    for shape in shapes:
        for pattern in ("gaussian", "zero", "anisotropic"):
            weights, gradients, momenta, left, right = fixture(shape, pattern, device)
            for inner in (False, True):
                factors = (left if inner else None, left)
                for lr in (0.0, optimizer["lr"], optimizer["lr"] / 32):
                    reference_trace, actual_trace = {}, {}
                    reference_momentum, reference = equations["block_update_stack"](
                        weights,
                        gradients,
                        momenta,
                        factors,
                        right,
                        args=args,
                        lr=lr,
                        weight_decay=optimizer["weight_decay"],
                        trace=reference_trace,
                    )
                    actual_momentum, actual = combined_outer_update(
                        weights,
                        gradients,
                        momenta,
                        factors,
                        right,
                        args=args,
                        lr=lr,
                        weight_decay=optimizer["weight_decay"],
                        muon=muon,
                        unit_rms=equations["stack_unit_rms"],
                        trace=actual_trace,
                    )
                    exact = {
                        "momentum": torch.equal(actual_momentum, reference_momentum),
                        **{
                            name: torch.equal(actual_trace[name], reference_trace[name])
                            for name in ("ns_source", "ns_output")
                        },
                    }
                    error = differences(actual, reference, TOLERANCE)
                    result["cases"].append(
                        {
                            "shape": list(shape),
                            "pattern": pattern,
                            "left_inner": inner,
                            "lr": lr,
                            "exact": exact,
                            "parameter_delta": error,
                        }
                    )
                    write(path, result)
                    if not all(exact.values()) or not error["within_tolerance"]:
                        result["status"] = "failed_production_parity_not_launch_gate"
                        write(path, result)
                        return result
            del weights, gradients, momenta, left, right, reference_trace, actual_trace
            del reference_momentum, actual_momentum, reference, actual
        weights, gradients, momenta, left, right = fixture(shape, "gaussian", device)
        factors = (None, left)
        reference = partial(
            equations["block_update_stack"],
            weights,
            gradients,
            momenta,
            factors,
            right,
            args=args,
            lr=optimizer["lr"],
            weight_decay=optimizer["weight_decay"],
        )
        combined = partial(
            combined_outer_update,
            weights,
            gradients,
            momenta,
            factors,
            right,
            args=args,
            lr=optimizer["lr"],
            weight_decay=optimizer["weight_decay"],
            muon=muon,
            unit_rms=equations["stack_unit_rms"],
        )
        for graphs in (False, True):
            measures = {}
            # Reverse paired order to expose order-dependent measurements.
            for index, order in enumerate((("reference", "combined"), ("combined", "reference"))):
                for name in order:
                    measures[f"{name}_{index}"] = timing(reference if name == "reference" else combined, graphs=graphs)
            base = statistics.mean(measures[f"reference_{index}"]["median_ms"] for index in range(2))
            actual = statistics.mean(measures[f"combined_{index}"]["median_ms"] for index in range(2))
            result["timing"].append(
                {
                    "shape": list(shape),
                    "graphs": graphs,
                    "left_inner": False,
                    "measures": measures,
                    "reference_ms": base,
                    "combined_ms": actual,
                    "ratio": actual / base,
                }
            )
            write(path, result)
        del reference, combined, weights, gradients, momenta, left, right
        torch.cuda.empty_cache()
    result["status"] = "passed_production_combined_outer_diagnostic_not_launch_gate"
    write(path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = diagnostic(args.root, args.out)
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]), "timing": len(result["timing"])}))
    if result["status"].startswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
