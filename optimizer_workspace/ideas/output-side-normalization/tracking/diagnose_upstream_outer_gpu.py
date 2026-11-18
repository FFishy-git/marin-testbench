# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test pinned split-product null/output updates under a separately owned GPU diagnostic."""

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from diagnose_combined_outer_gpu import fixture, timing
from diagnose_fused_diagonal import differences
from launch_wave5 import read, sha256
from watch_wave5 import process, write
from wave7_release import PROCESS_MARKER, candidate_inventory

TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
PATTERNS = ("gaussian", "zero", "anisotropic")
ARMS = ("outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused")


def cpu_source(receipt_path: Path) -> tuple[Path, dict]:
    """Require the current candidate inventory and successful recorded CPU command logs."""
    receipt = read(receipt_path)
    candidate = Path(receipt["candidate"])
    initial = Path(receipt["initial_preparation"])
    if (
        receipt["status"] != "cpu_verified_gpu_execution_denied"
        or receipt["cuda_visible_devices"] != ""
        or tuple(receipt["arms"]) != ARMS
        or sha256(initial) != receipt["initial_preparation_sha256"]
        or candidate_inventory(candidate) != receipt["files_sha256"]
        or sha256(Path(receipt["archive"])) != receipt["archive_sha256"]
        or len(receipt["commands"]) != 2
        or any(
            type(command["exit_code"]) is not int
            or command["exit_code"] != 0
            or sha256(Path(command["log"])) != command["log_sha256"]
            for command in receipt["commands"]
        )
    ):
        raise ValueError("Upstream-product CPU source or recorded test evidence differs")
    root = Path(receipt["initial_preparation"]).parents[2]
    if (
        sha256(root / "run-preparation/wave8-launch-binding.json") != receipt["predecessor_release_sha256"]
        or sha256(root / "run-preparation/wave8-candidate.json") != receipt["predecessor_candidate_sha256"]
    ):
        raise ValueError("Upstream-product predecessor identity changed")
    return candidate, receipt


def describe(receipt_path: Path) -> dict:
    """Derive workloads without importing candidate kernels or initializing CUDA."""
    candidate, receipt = cpu_source(receipt_path)
    config = read(candidate / f"experiment/configs/{ARMS[0]}_0p5m_seed0.json")
    null = read(candidate / f"experiment/configs/{ARMS[1]}_0p5m_seed0.json")
    left_config, null_config = json.loads(json.dumps(config)), json.loads(json.dumps(null))
    left_config["experiment"].pop("arm")
    null_config["experiment"].pop("arm")
    if left_config != null_config:
        raise ValueError("Upstream output and null configurations differ outside application")
    model, training, optimizer = config["model"], config["training"], config["optimizer"]
    owned, remainder = divmod(model["layers"], training["world_size"])
    if remainder or training["world_size"] != 8 or optimizer["ns_mode"] != "tpu":
        raise ValueError("Upstream diagnostic requires the actual equally owned eight-rank production shapes")
    return {
        "status": "described_gpu_not_executed",
        "cpu_receipt": str(receipt_path.resolve()),
        "cpu_receipt_sha256": sha256(receipt_path),
        "candidate": str(candidate),
        "candidate_files": len(receipt["files_sha256"]),
        "config": config,
        "shapes": [
            [4 * owned, model["hidden"], model["hidden"]],
            [2 * owned, model["mlp"], model["hidden"]],
            [owned, model["hidden"], model["mlp"]],
        ],
        "patterns": list(PATTERNS),
        "learning_rates": [0.0, optimizer["lr"], optimizer["lr"] / 32],
        "nesterov": [False, True],
        "cases": 3 * len(PATTERNS) * 3 * 2,
        "outer_delta_tolerance": TOLERANCE,
        "exact_null_fields": ["momentum", "ns_source", "ns_output", "parameter_delta"],
        "scope": (
            "Synthetic single-A100 owned production-shape stacks. Exact pinned null, same-NS output equation, "
            "direct-vs-split sensitivity and eager/graph timing. Excludes capture, distributed statistics, "
            "full model, continuation, loss and whole-training efficiency. No training approval."
        ),
    }


def require_owner(out: Path, receipt_path: Path) -> dict:
    """Require a live exact controller generation and acknowledged marked child before CUDA use."""
    deadline = time.monotonic() + 10
    while True:
        state = read(out / "state.json")
        parent = process(state["controller_pid"])
        if "child_pid" in state:
            break
        if parent is None or parent[0] == "Z" or time.monotonic() >= deadline:
            raise RuntimeError("Upstream GPU diagnostic has no live child acknowledgment")
        time.sleep(0.05)
    argv = (Path("/proc") / str(state["controller_pid"]) / "cmdline").read_bytes().split(b"\0")
    expected = [
        state["python"],
        str(out / "helpers/run_upstream_outer_gpu.py"),
        "--root",
        state["root"],
        "--cpu-receipt",
        str(receipt_path),
        "--out",
        str(out),
        "--phase",
        "run",
    ]
    if (
        parent is None
        or parent[0] == "Z"
        or parent[1] != state["controller_start_ticks"]
        or [item.decode() for item in argv if item] != expected
        or os.getppid() != state["controller_pid"]
        or state["child_pid"] != os.getpid()
        or os.environ.get(PROCESS_MARKER) != state["child_marker"]
        or state["status"] != "running"
        or state["cpu_receipt_sha256"] != sha256(receipt_path)
        or Path(__file__).resolve() != (out / "helpers/diagnose_upstream_outer_gpu.py").resolve()
        or sha256(out / "binding.json") != state["binding_sha256"]
        or any(sha256(out / "helpers" / name) != digest for name, digest in state["helpers_sha256"].items())
    ):
        raise RuntimeError("Upstream GPU diagnostic lacks exact live bound ownership")
    return state


def independent_outer_delta(weights, direction, left, right, *, lr, decay):
    """Use a direct FP64 matrix equation with an already computed NS output."""
    rows, columns = weights.shape[-2:]
    weight, value, factor = weights.double(), direction.double(), right.double()
    normalized_right = factor / (factor.square().sum((-2, -1), keepdim=True) / columns).sqrt()
    if left is not None:
        vector = left.double()
        value = value * (vector / vector.square().mean(-1, keepdim=True).sqrt())[:, :, None]
    scale = max(1.0, rows / columns) ** 0.5
    return -lr * (
        scale * torch.bmm(value, normalized_right)
        + decay * torch.bmm(torch.bmm(weight, normalized_right), normalized_right)
    )


def load_kernels(candidate: Path, inventory: dict) -> tuple:
    sys.path.insert(0, str(candidate))
    prototype = importlib.import_module("experiment.upstream_outer_products")
    engine = importlib.import_module("toner.fast.engine")
    roots = importlib.import_module("toner.fast.roots")
    muon = importlib.import_module("toner.fast.muon")
    block = importlib.import_module("experiment.block_toner")
    for module in (prototype, engine, roots, muon, block):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(candidate.resolve()) or sha256(path) != inventory[str(path.relative_to(candidate))]:
            raise ValueError("Upstream diagnostic imported different candidate or pinned kernel source")
    return prototype.upstream_outer_update_stack, engine, roots, muon, block


def pinned_null(weights, gradients, momenta, right, *, args, lr, decay, engine, roots, muon):
    buffer, source = engine.muont_in(
        list(gradients.unbind()), momenta, roots.rows_lo(right[0]), args.muon_momentum, args.muon_nesterov
    )
    direction = muon.newton_schulz_batched(source, args.muon_ns_steps, args.muon_eps, args.muon_ns_mode)
    delta = engine.muont_out(
        list(weights.unbind()),
        direction,
        roots.rows_lo(right[1]),
        right[1].square().sum((-2, -1)),
        lr,
        decay,
        max(1.0, weights.shape[-2] / weights.shape[-1]) ** 0.5,
    )
    return buffer, source, direction, delta


def diagnostic(receipt_path: Path, out: Path) -> dict:
    owner = require_owner(out, receipt_path)
    plan = describe(receipt_path)
    candidate, receipt = cpu_source(receipt_path)
    prototype, engine, roots, muon, block = load_kernels(candidate, receipt["files_sha256"])
    config = plan["config"]
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
    result = {
        **plan,
        "status": "upstream_production_diagnostic_running",
        "owner": {"controller_pid": owner["controller_pid"], "controller_start_ticks": owner["controller_start_ticks"]},
        "binding_sha256": owner["binding_sha256"],
        "helpers_sha256": owner["helpers_sha256"],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cases": [],
        "timing": [],
    }
    path = out / "result.json"
    write(path, result)
    for shape in plan["shapes"]:
        for pattern in PATTERNS:
            weights, gradients, momenta, left, right = fixture(tuple(shape), pattern, device)
            for nesterov in plan["nesterov"]:
                args.muon_nesterov = nesterov
                for lr_value in plan["learning_rates"]:
                    lr, decay = torch.tensor(lr_value, device=device), torch.tensor(
                        optimizer["weight_decay"], device=device
                    )
                    expected = pinned_null(
                        weights,
                        gradients,
                        momenta,
                        right,
                        args=args,
                        lr=lr,
                        decay=decay,
                        engine=engine,
                        roots=roots,
                        muon=muon,
                    )
                    traces, outputs = [], []
                    for diagonal in (None, left):
                        trace = {}
                        outputs.append(
                            prototype(
                                weights,
                                gradients,
                                momenta,
                                (None, diagonal),
                                right,
                                args=args,
                                lr=lr,
                                weight_decay=decay,
                                trace=trace,
                            )
                        )
                        traces.append(trace)
                    exact = {
                        "momentum": torch.equal(outputs[0][0], expected[0]),
                        "ns_source": torch.equal(traces[0]["ns_source"], expected[1]),
                        "ns_output": torch.equal(traces[0]["ns_output"], expected[2]),
                        "parameter_delta": torch.equal(outputs[0][1], expected[3]),
                        "outer_momentum": torch.equal(outputs[1][0], outputs[0][0]),
                        "outer_ns_source": torch.equal(traces[1]["ns_source"], traces[0]["ns_source"]),
                        "outer_ns_output": torch.equal(traces[1]["ns_output"], traces[0]["ns_output"]),
                    }
                    equation = independent_outer_delta(
                        weights,
                        traces[1]["ns_output"],
                        left,
                        right[1],
                        lr=lr.item(),
                        decay=decay.item(),
                    )
                    outer_error = differences(outputs[1][1], equation, TOLERANCE)
                    legacy_trace = {}
                    _, legacy = block.block_update_stack(
                        weights,
                        gradients,
                        momenta,
                        (None, left),
                        right,
                        args=args,
                        lr=lr_value,
                        weight_decay=optimizer["weight_decay"],
                        trace=legacy_trace,
                    )
                    drift = {
                        "ns_source": differences(traces[1]["ns_source"], legacy_trace["ns_source"], TOLERANCE),
                        "ns_output": differences(traces[1]["ns_output"], legacy_trace["ns_output"], TOLERANCE),
                        "parameter_delta": differences(outputs[1][1], legacy, TOLERANCE),
                        "scope": "Diagnostic-only direct-FP32 versus pinned split-product numerical drift.",
                    }
                    result["cases"].append(
                        {
                            "shape": shape,
                            "pattern": pattern,
                            "nesterov": nesterov,
                            "lr": lr_value,
                            "exact_null_and_unchanged_input": exact,
                            "same_ns_outer_equation": outer_error,
                            "direct_product_sensitivity": drift,
                        }
                    )
                    write(path, result)
                    if not all(exact.values()) or not outer_error["within_tolerance"]:
                        result["status"] = "failed_upstream_production_parity_not_launch_gate"
                        write(path, result)
                        return result
            del weights, gradients, momenta, left, right, outputs, traces, expected, equation, legacy, legacy_trace
        args.muon_nesterov = optimizer["nesterov"]
        weights, gradients, momenta, left, right = fixture(tuple(shape), "gaussian", device)
        lr, decay = torch.tensor(optimizer["lr"], device=device), torch.tensor(optimizer["weight_decay"], device=device)
        functions = {
            "legacy_direct": partial(
                block.block_update_stack,
                weights,
                gradients,
                momenta,
                (None, left),
                right,
                args=args,
                lr=optimizer["lr"],
                weight_decay=optimizer["weight_decay"],
            ),
            "upstream_outer": partial(
                prototype,
                weights,
                gradients,
                momenta,
                (None, left),
                right,
                args=args,
                lr=lr,
                weight_decay=decay,
            ),
            "upstream_null": partial(
                prototype,
                weights,
                gradients,
                momenta,
                (None, None),
                right,
                args=args,
                lr=lr,
                weight_decay=decay,
            ),
        }
        for graphs in (False, True):
            samples = {}
            for order_index, order in enumerate((tuple(functions), tuple(reversed(functions)))):
                for name in order:
                    samples[f"{name}_{order_index}"] = timing(functions[name], graphs=graphs)
            medians = {
                name: statistics.mean(samples[f"{name}_{index}"]["median_ms"] for index in range(2))
                for name in functions
            }
            result["timing"].append(
                {
                    "shape": shape,
                    "graphs": graphs,
                    "samples": samples,
                    "paired_median_ms": medians,
                    "upstream_outer_over_legacy": medians["upstream_outer"] / medians["legacy_direct"],
                    "upstream_outer_over_null": medians["upstream_outer"] / medians["upstream_null"],
                    "scope": (
                        "Synthetic GPU events; excludes capture, factor computation, scalar transfer and communication."
                    ),
                }
            )
            write(path, result)
        del functions, weights, gradients, momenta, left, right
        torch.cuda.empty_cache()
    if len(result["cases"]) != plan["cases"] or len(result["timing"]) != 6:
        raise ValueError("Upstream diagnostic coverage differs from its bound plan")
    result["status"] = "passed_upstream_production_diagnostic_not_launch_gate"
    write(path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if not args.describe and args.out is None:
        parser.error("--out is required for owned GPU execution")
    result = describe(args.cpu_receipt) if args.describe else diagnostic(args.cpu_receipt, args.out)
    print(json.dumps(result if args.describe else {"status": result["status"], "out": str(args.out)}))
    if not args.describe and result["status"] != "passed_upstream_production_diagnostic_not_launch_gate":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
