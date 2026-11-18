# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated A100 benchmark of BF16 row-square sums; does not alter training.

Workloads come from the checked-in 0p5m/1m configs and capture definitions.
--describe prints their geometry without initializing CUDA or running kernels.
GPU execution checks FP32 production parity and an independent FP64 reference
before timing eager calls; results are not a whole-training speed claim.
"""

import argparse
import ast
import hashlib
import json
import math
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
import triton
import triton.language as tl

TASK = Path(__file__).resolve().parents[1]
ROW_TILE = 128
COLUMN_TILE = 32
TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
PATTERNS = ("gaussian", "zero", "constant", "anisotropic", "sparse_outliers", "scaled_cotangent")


def source_definitions(path: Path, names: tuple[str, ...], namespace: dict) -> dict:
    """Load only requested constants/functions, avoiding trainer import side effects."""
    parsed = ast.parse(path.read_text(), filename=str(path))
    definitions = []
    found = set()
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            definitions.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & set(names):
                definitions.append(node)
                found.update(targets & set(names))
    if found != set(names):
        raise ValueError(f"{path}: missing geometry definitions {set(names) - found}")
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def workload_geometry(batches: list[str]) -> dict:
    """Derive full per-rank row capacities and unsharded diagonal group counts."""
    manifest_path = TASK / "experiment/upstream_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    upstream = TASK / "artifacts/external-sources" / f"Toner-{manifest['revision']}"
    model_path = upstream / "reproduce/marin/model.py"
    engine_path = upstream / "src/toner/fast/engine.py"
    for path in (model_path, engine_path):
        relative = str(path.relative_to(upstream))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][relative]:
            raise ValueError(f"Modified pinned geometry source: {path}")
    sampling = source_definitions(model_path, ("sample_positions",), {})
    inputs = source_definitions(engine_path, ("KINDS_D",), {})
    block_path = TASK / "experiment/block_toner.py"
    kinds = source_definitions(block_path, ("INPUT_KINDS", "HIDDEN_OUTPUTS", "MLP_OUTPUTS", "ROW_SCALE"), inputs)
    source_paths = [manifest_path, model_path, engine_path, block_path, TASK / "experiment/capture.py"]
    source_paths.append(TASK / "experiment/train.py")
    workloads = []
    configs = []
    for batch in batches:
        geometries = []
        for arm in ("inner_outer_all_diag", "inner_outer_mlpdiag"):
            path = TASK / "experiment/configs" / f"{arm}_{batch}_seed0.json"
            config = json.loads(path.read_text())
            source_paths.append(path)
            training, model, settings = config["training"], config["model"], config["statistics"]
            micro, world, global_batch = (
                training["micro_batch_size"],
                training["world_size"],
                training["global_batch_sequences"],
            )
            accumulation, remainder = divmod(global_batch, world * micro)
            if remainder or accumulation < 1 or config["root"]["chunk"] != 0:
                raise ValueError(f"{path}: benchmark requires full accumulation buffers")
            if training["batch_tokens"] != global_batch * model["sequence_length"]:
                raise ValueError(f"{path}: inconsistent global batch tokens")
            if config["experiment"]["upstream_revision"] != manifest["revision"]:
                raise ValueError(f"{path}: different pinned upstream revision")
            positions = sampling["sample_positions"](model["sequence_length"], settings["head"], settings["stride"])
            if not positions or len(set(positions)) != len(positions):
                raise ValueError(f"{path}: empty or duplicate sampled positions")
            if min(positions) < 0 or max(positions) >= model["sequence_length"]:
                raise ValueError(f"{path}: sampled position outside sequence")
            if model["layers"] % world:
                raise ValueError(f"{path}: unequal layer ownership requires separate geometry")
            geometry = {
                "batch": batch,
                "batch_tokens": training["batch_tokens"],
                "global_batch_sequences": global_batch,
                "micro_batch_sequences": micro,
                "world_size": world,
                "accumulation": accumulation,
                "sequence_length": model["sequence_length"],
                "sample_head": settings["head"],
                "sample_stride": settings["stride"],
                "sampled_positions": len(positions),
                "first_sampled_position": positions[0],
                "last_sampled_position": positions[-1],
                "rows_per_microbatch": micro * len(positions),
                "row_capacity_per_rank": accumulation * micro * len(positions),
                "rows_total_across_ranks": global_batch * len(positions),
                "layers": model["layers"],
                "owned_layers_per_rank": model["layers"] // world,
                "hidden": model["hidden"],
                "mlp": model["mlp"],
            }
            geometries.append(geometry)
            configs.append({"path": str(path.relative_to(TASK)), "arm": arm, "geometry": geometry})
        if geometries[0] != geometries[1]:
            raise ValueError(f"{batch}: input/output diagonal configs have different capture geometry")
        geometry = geometries[0]
        input_hidden = tuple(kind for kind in kinds["INPUT_KINDS"] if kind != "down")
        groups = (
            ("input_hidden", input_hidden, geometry["hidden"], "inner_outer_all_diag"),
            ("input_mlp", ("down",), geometry["mlp"], "inner_outer_all_diag"),
            ("output_hidden", kinds["HIDDEN_OUTPUTS"], geometry["hidden"], "inner_outer_all_diag"),
            ("output_mlp", kinds["MLP_OUTPUTS"], geometry["mlp"], "inner_outer_mlpdiag"),
        )
        for name, members, width, arm in groups:
            workloads.append(
                {
                    "batch": batch,
                    "group": name,
                    "representative_arm": arm,
                    "kinds": list(members),
                    "rows": geometry["row_capacity_per_rank"],
                    "width": width,
                    "buffers": geometry["layers"] * len(members),
                    "geometry": geometry,
                }
            )
    return {
        "configs": configs,
        "workloads": workloads,
        "row_scale": kinds["ROW_SCALE"],
        "group_scope": "Every rank reduces all layers' local rows, not only its owned layers.",
        "sources_sha256": {
            str(path.relative_to(TASK)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
        },
    }


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
    summed = tl.sum(value * value, axis=0)
    tl.store(partials + (item * tiles + tile) * width + column, summed, column < width)


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


def fused_squares(buffers: list[torch.Tensor]) -> tuple[Callable[[], torch.Tensor], dict]:
    """Return a callable with fixed buffers and reusable FP32 scratch/output."""
    if not buffers or buffers[0].ndim != 2:
        raise ValueError("Expected a nonempty list of row matrices")
    rows, width = buffers[0].shape
    device = buffers[0].device
    if rows < 1 or width < 1 or device.type != "cuda":
        raise ValueError("Expected nonempty CUDA row matrices")
    if any(
        buffer.shape != (rows, width)
        or not buffer.is_contiguous()
        or buffer.dtype != torch.bfloat16
        or buffer.device != device
        or buffer.requires_grad
        for buffer in buffers
    ):
        raise ValueError("Buffers must be contiguous BF16 matrices on one CUDA device without gradients")
    allocated = torch.cuda.memory_allocated(device)
    pointers = torch.tensor([buffer.data_ptr() for buffer in buffers], dtype=torch.int64, device=device)
    tiles = triton.cdiv(rows, ROW_TILE)
    partials = torch.empty(len(buffers), tiles, width, dtype=torch.float32, device=device)
    output = torch.empty(len(buffers), width, dtype=torch.float32, device=device)

    def compute():
        partial_squares[(len(buffers), tiles, triton.cdiv(width, COLUMN_TILE))](
            pointers, partials, rows, width, ROW_TILE, COLUMN_TILE, tiles, num_warps=4, enable_fp_fusion=False
        )
        sum_partials[(len(buffers), triton.cdiv(width, COLUMN_TILE))](
            partials, output, width, tiles, triton.next_power_of_2(tiles), COLUMN_TILE, num_warps=4
        )
        return output

    memory = {
        "pointer_table_bytes": pointers.numel() * pointers.element_size(),
        "partials_bytes": partials.numel() * partials.element_size(),
        "output_bytes": output.numel() * output.element_size(),
    }
    memory["scratch_bytes"] = memory["pointer_table_bytes"] + memory["partials_bytes"]
    memory["persistent_bytes"] = memory["scratch_bytes"] + memory["output_bytes"]
    memory["persistent_allocated_bytes"] = torch.cuda.memory_allocated(device) - allocated
    return compute, memory


def torch_squares(buffers: list[torch.Tensor]) -> torch.Tensor:
    """The exact eager FP32 expression used by DiagonalMoments.update."""
    return torch.stack([buffer.float().square().sum(0) for buffer in buffers])


def torch_squares_reuse(buffers: list[torch.Tensor]) -> Callable[[], torch.Tensor]:
    """Match reusable output storage while retaining Torch's eager reductions."""
    output = torch.empty(len(buffers), buffers[0].shape[1], dtype=torch.float32, device=buffers[0].device)

    def compute():
        return torch.stack([buffer.float().square().sum(0) for buffer in buffers], out=output)

    return compute


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual.double() - expected.double()).abs()
    return {
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "max_relative_error": (difference / expected.double().abs().clamp_min(1e-30)).max().item(),
    }


def verify_squares(buffers: list[torch.Tensor], compute: Callable[[], torch.Tensor]) -> dict:
    """Check the consumed BF16 values against both production FP32 and FP64 sums."""
    expected = torch_squares(buffers)
    reference = torch.stack([buffer.double().square().sum(0) for buffer in buffers])
    actual = compute()
    torch.testing.assert_close(actual, expected, **TOLERANCE)
    torch.testing.assert_close(actual.double(), reference, **TOLERANCE)
    torch.testing.assert_close(expected.double(), reference, **TOLERANCE)
    return {
        "fused_vs_production": error_metrics(actual, expected),
        "fused_vs_fp64": error_metrics(actual, reference),
        "production_vs_fp64": error_metrics(expected, reference),
    }


def make_buffers(
    rows: int, width: int, count: int, pattern: str, device, generator, row_scale: float
) -> list[torch.Tensor]:
    """Synthetic BF16 inputs, including unequal scales and masked all-zero columns."""
    buffers = []
    for index in range(count):
        if pattern == "zero":
            buffer = torch.zeros(rows, width, device=device, dtype=torch.bfloat16)
        elif pattern == "constant":
            buffer = torch.full((rows, width), (-1) ** index * (index + 1) / 8, device=device, dtype=torch.bfloat16)
        elif pattern == "sparse_outliers":
            buffer = torch.zeros(rows, width, device=device, dtype=torch.bfloat16)
            for row in sorted({0, min(ROW_TILE - 1, rows - 1), min(ROW_TILE, rows - 1), rows - 1}):
                buffer[row, ::3] = (-1) ** row * (index + 1) * 64
            buffer[-1, -1] = (index + 1) / 64
        else:
            values = torch.randn(rows, width, device=device, generator=generator, dtype=torch.float32)
            if pattern == "anisotropic":
                scales = torch.exp2((torch.arange(width, device=device) % 21 - 10).float())
                values *= scales
            elif pattern == "scaled_cotangent":
                values *= 2.0**-24
                buffer = values.to(torch.bfloat16)
                buffer.mul_(row_scale)
                buffers.append(buffer)
                continue
            elif pattern != "gaussian":
                raise ValueError(f"Unknown parity pattern {pattern}")
            buffer = values.to(torch.bfloat16)
        buffers.append(buffer)
    return buffers


def measure(
    computes: dict[str, Callable[[], torch.Tensor]], repeats: int, warmup: int, calls_per_sample: int, device
) -> dict:
    """Rotate/reverse method order with primed events and equal warmup/output lifetimes."""
    stream = torch.cuda.current_stream(device)
    events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for name in computes}
    for begin, end in events.values():
        begin.record(stream)
        end.record(stream)
        end.synchronize()
    names = list(computes)
    for iteration in range(warmup):
        offset = iteration % len(names)
        order = names[offset:] + names[:offset]
        if iteration % 2:
            order.reverse()
        for name in order:
            output = computes[name]()
            del output
    torch.cuda.synchronize(device)
    measurements = {name: {"samples_ms": [], "wall_samples_ms": [], "peak_temporary_bytes": 0} for name in names}
    for iteration in range(repeats):
        offset = iteration % len(names)
        order = names[offset:] + names[:offset]
        if iteration % 2:
            order.reverse()
        for name in order:
            allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            begin, end = events[name]
            start = perf_counter()
            begin.record(stream)
            for _ in range(calls_per_sample):
                output = computes[name]()
                del output
            end.record(stream)
            end.synchronize()
            record = measurements[name]
            record["wall_samples_ms"].append((perf_counter() - start) * 1000 / calls_per_sample)
            record["samples_ms"].append(begin.elapsed_time(end) / calls_per_sample)
            record["peak_temporary_bytes"] = max(
                record["peak_temporary_bytes"], torch.cuda.max_memory_allocated(device) - allocated
            )
    for record in measurements.values():
        timings = sorted(record["samples_ms"])
        record.update(
            median_ms=statistics.median(timings),
            p95_ms=timings[math.ceil(0.95 * repeats) - 1],
            median_wall_ms=statistics.median(record["wall_samples_ms"]),
            repeats=repeats,
            warmup_calls=warmup,
            calls_per_sample=calls_per_sample,
        )
    return measurements


@torch.inference_mode()
def run(geometry: dict, repeats: int, warmup: int, calls_per_sample: int, device_index: int) -> dict:
    """Validate and time each real local geometry on one explicitly selected A100."""
    if repeats < 1 or warmup < 1 or calls_per_sample < 1:
        raise ValueError("Repeats, warmup and calls per sample must be positive")
    device = torch.device("cuda", device_index)
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError("Probe requires A100 SM80")
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(26)
    workload_shapes = {(workload["rows"], workload["width"]) for workload in geometry["workloads"]}
    shapes = [(1, 1), (17, 33), (129, 511), (513, 2049)]
    shapes.extend(sorted(workload_shapes))
    checks, cases = [], []
    for rows, width in shapes:
        for pattern in PATTERNS:
            buffers = make_buffers(rows, width, 3, pattern, device, generator, geometry["row_scale"])
            compute, _ = fused_squares(buffers)
            errors = verify_squares(buffers, compute)
            for buffer in buffers:
                buffer.zero_()
            del buffer
            if torch.count_nonzero(compute()).item() != 0:
                raise AssertionError("Reused scratch/output retained a previous square sum")
            checks.append(
                {
                    "rows": rows,
                    "width": width,
                    "buffers": 3,
                    "pattern": pattern,
                    "zero_overwrite_verified": True,
                    **errors,
                }
            )
            del buffers, compute
    for workload in geometry["workloads"]:
        rows, width, count = workload["rows"], workload["width"], workload["buffers"]
        buffers = make_buffers(rows, width, count, "gaussian", device, generator, geometry["row_scale"])
        compute, memory = fused_squares(buffers)
        errors = verify_squares(buffers, compute)
        allocated = torch.cuda.memory_allocated(device)
        reuse = torch_squares_reuse(buffers)
        reuse_allocated = torch.cuda.memory_allocated(device) - allocated
        torch.testing.assert_close(reuse(), compute(), **TOLERANCE)
        computes = {"baseline": partial(torch_squares, buffers), "baseline_reuse_output": reuse, "fused": compute}
        timings = measure(computes, repeats, warmup, calls_per_sample, device)
        for name, persistent, resident in (
            ("baseline", 0, 0),
            ("baseline_reuse_output", memory["output_bytes"], reuse_allocated),
            ("fused", memory["persistent_bytes"], memory["persistent_allocated_bytes"]),
        ):
            timings[name]["persistent_bytes"] = persistent
            timings[name]["persistent_allocated_bytes"] = resident
            timings[name]["peak_working_allocated_bytes"] = resident + timings[name]["peak_temporary_bytes"]
        cases.append(
            {
                **workload,
                "input_dtype": "bfloat16",
                "accumulation_dtype": "float32",
                "input_buffer_bytes": sum(buffer.numel() * buffer.element_size() for buffer in buffers),
                "fused_memory": memory,
                "baseline_persistent_output_bytes": 0,
                "baseline_reuse_persistent_output_bytes": memory["output_bytes"],
                **timings,
                **errors,
                "speedup": timings["baseline"]["median_ms"] / timings["fused"]["median_ms"],
                "speedup_vs_reuse_output": timings["baseline_reuse_output"]["median_ms"] / timings["fused"]["median_ms"],
            }
        )
        del buffers, compute, reuse, computes
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return {
        "schema_version": 1,
        "time": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(device),
        "device": str(device),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "geometry": geometry,
        "scope": "Uncentered per-coordinate sums only; no communication, EMA, capture, or whole-training speed claim.",
        "timing_scope": "Eager same-stream calls, rotated/reversed method order, compilation and setup outside timing.",
        "event_scope": "Events include eager host-dispatch gaps; wall intervals also include synchronization.",
        "cache_scope": "Repeated fixed synthetic buffers; no CUDA graphs or cache-flush claim.",
        "memory_scope": "Logical tensor sizes and allocator-rounded bytes are separate; working bytes exclude inputs.",
        "reference": "Squares and sums in FP64 of the same already-quantized BF16 buffers.",
        "seed": 26,
        "tolerance": TOLERANCE,
        "row_tile": ROW_TILE,
        "column_tile": COLUMN_TILE,
        "checks": checks,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Required for GPU execution; atomic JSON receipt")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--calls-per-sample", type=int, default=3)
    parser.add_argument("--device", type=int, default=0, help="Local CUDA device index")
    parser.add_argument("--batches", nargs="+", choices=("0p5m", "1m"), default=["0p5m", "1m"])
    parser.add_argument("--describe", action="store_true", help="Print source/config geometry without initializing CUDA")
    args = parser.parse_args()
    if not args.describe and args.out is None:
        parser.error("--out is required for GPU execution")
    geometry = workload_geometry(list(dict.fromkeys(args.batches)))
    result = geometry if args.describe else run(geometry, args.repeats, args.warmup, args.calls_per_sample, args.device)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(args.out)
    print(payload, end="")


if __name__ == "__main__":
    main()
