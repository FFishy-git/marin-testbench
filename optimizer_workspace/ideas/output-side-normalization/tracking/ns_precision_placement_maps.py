# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-derived precision-placement maps without CUDA execution or ownership authority."""

import ast
import copy
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import ns_compiler_localization as localization
import replay_saved_ns as replay
import torch

TensorMap = Callable[[torch.Tensor], torch.Tensor]
Product = Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor]
SCHEDULES: dict[str, tuple[str, ...]] = {
    "q5_tpu": ("tpu",) * 5,
    "q5_f32": ("f32",) * 5,
    "q10_tpu": ("tpu",) * 10,
    "q5_tpu2_f32suffix3": ("tpu", "tpu", "f32", "f32", "f32"),
    "q5_f32prefix3_tpu2": ("f32", "f32", "f32", "tpu", "tpu"),
}
SCOPE = (
    "Numerical precision-placement component only. Default products use the authenticated pinned "
    "_bmm, including CUDA BF16 operands with FP32 out_dtype in tpu mode. All three products in each "
    "iteration share its mode; mixed schedules change executed numerics. Optional cpu_test_product "
    "injection is CPU-only test emulation, not CUDA evidence. No standalone CUDA entry, device "
    "initialization, ownership, continuation, training-quality, efficiency or production authority."
)


@dataclass(frozen=True)
class PrecisionPlacementMaps:
    """Retain callable maps and the source, schedule and product implementation they execute."""

    eager: TensorMap
    compiled: TensorMap
    source_sha256: str
    schedule_name: str
    schedule: tuple[str, ...]
    canonical_filename: str
    eps: float
    product_implementation: str


def _cpu_only(numerical: TensorMap) -> TensorMap:
    def evaluate(source: torch.Tensor) -> torch.Tensor:
        if source.device.type != "cpu":
            raise RuntimeError("cpu_test_product maps require CPU input; no CUDA emulation is permitted")
        return numerical(source)

    return evaluate


def maps(
    path: Path,
    name: str,
    compiler: Callable[[TensorMap], TensorMap],
    *,
    eps: float = replay.NS_EPS,
    cpu_test_product: Product | None = None,
) -> PrecisionPlacementMaps:
    """Derive a named schedule, retaining the pinned products unless a CPU test callback is supplied."""
    if name not in SCHEDULES:
        raise ValueError("Precision placement requires one of its five explicit schedules")
    schedule = SCHEDULES[name]
    namespace, original, _, _, _, _ = localization.source_namespace(path)
    if cpu_test_product is not None:
        namespace["_bmm"] = cpu_test_product
    if len(set(schedule)) == 1:
        eager = partial(namespace["newton_schulz_batched"], steps=len(schedule), eps=eps, mode=schedule[0])
    else:
        function = copy.deepcopy(original)
        function.name = "_ns_precision_placement_map"
        loop = next(node for node in function.body if isinstance(node, ast.For))
        loop.iter = ast.Tuple(elts=[ast.Constant(value=mode) for mode in schedule], ctx=ast.Load())
        loop.target = ast.Name(id="_ns_iteration_mode", ctx=ast.Store())
        products = [
            node
            for node in ast.walk(loop)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_bmm"
        ]
        if len(products) != 3 or any(
            len(node.args) != 3 or node.keywords or not isinstance(node.args[2], ast.Name) or node.args[2].id != "mode"
            for node in products
        ):
            raise ValueError("Pinned NS iteration product coverage or mode differs")
        for node in products:
            node.args[2] = ast.Name(id="_ns_iteration_mode", ctx=ast.Load())
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
        # Neither supported mode casts the iterate before the recurrence; modes apply only to products.
        eager = partial(namespace[function.name], steps=len(schedule), eps=eps, mode="f32")
    if cpu_test_product is not None:
        eager = _cpu_only(eager)
    compiled = compiler(eager)
    if cpu_test_product is not None:
        compiled = _cpu_only(compiled)
    return PrecisionPlacementMaps(
        eager=eager,
        compiled=compiled,
        source_sha256=replay.MUON_SHA256,
        schedule_name=name,
        schedule=schedule,
        canonical_filename=str(path),
        eps=eps,
        product_implementation="pinned_bmm" if cpu_test_product is None else "cpu_test_product",
    )
