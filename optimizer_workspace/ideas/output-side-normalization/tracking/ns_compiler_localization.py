# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-derived NS compiler boundaries and guarded retained-stage comparisons."""

import ast
import copy
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import replay_saved_ns as replay
import root_stage_trace as tracing
import torch
from benchmark_diagonal import source_definitions
from ns_budget_execution import source_snapshots
from root_preparation_probe import finite_tree

BOUNDARIES = ("eager", "full_compiled", "normalization_compiled", "recurrence_compiled")
ARMS = (("tpu", 10), ("f32", 10))
PHASES = ("call", "capture", "replay", "replay_repeat")
PRODUCT_STAGES = ("gram", "gram_square", "correction")
NORMALIZATION_FIELDS = ("input", "norm", "denominator", "normalized")
PRODUCT_FIELDS = ("left_raw", "right_raw", "left_operand", "right_operand", "product")
SCOPE = (
    "Source-derived NS compiler-boundary localization component only. Full eager and full-compiled "
    "controls retain the unmodified pinned routine. Tracing retains references and changes tensor "
    "lifetimes/compiler outputs; exact finite uninstrumented/instrumented parity at every phase is "
    "required before interpreting stages. Two opposite-order fresh CUDA processes, original whole "
    "saved batches, compiled controls, CUDA graphs, independent retained reconstruction and separately "
    "accepted exclusive ownership remain required. No standalone CUDA entry, timing, continuation, "
    "production, optimizer-quality, efficiency or SwanLab authority."
)


@dataclass(frozen=True)
class CompilerMaps:
    """Hold corresponding numerical and reference-retaining maps for each compiler boundary."""

    controls: dict[str, Callable]
    traces: dict[str, Callable]
    source_sha256: str
    mode: str
    steps: int
    eps: float


def _function(name: str, arguments: str, body: list[ast.stmt]) -> ast.FunctionDef:
    node = ast.parse(f"def {name}({arguments}):\n    pass\n").body[0]
    node.body = copy.deepcopy(body)
    return node


def _statements(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


def _record(stage: str, iteration: str, fields: dict[str, str]) -> ast.stmt:
    values = ", ".join(f"{name!r}: {value}" for name, value in fields.items())
    return _statements(f"_ns_records.append({{'stage': {stage!r}, 'iteration': {iteration}, 'values': {{{values}}}}})")[
        0
    ]


def _prefix_trace(prefix: list[ast.stmt]) -> list[ast.stmt]:
    body, found = _statements("_ns_records = []"), False
    for statement in copy.deepcopy(prefix):
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.BinOp):
            value = statement.value
            if isinstance(value.op, ast.Div):
                if found or not isinstance(value.right, ast.BinOp) or not isinstance(value.right.op, ast.Add):
                    raise ValueError("Pinned NS normalization expression differs")
                body.extend(_statements("_ns_normalization_input = x\n_ns_norm = 0\n_ns_denominator = 0"))
                body[-2].value = copy.deepcopy(value.right.left)
                body[-1].value = ast.BinOp(
                    left=ast.Name(id="_ns_norm", ctx=ast.Load()),
                    op=ast.Add(),
                    right=copy.deepcopy(value.right.right),
                )
                statement.value.right = ast.Name(id="_ns_denominator", ctx=ast.Load())
                body.append(statement)
                body.append(
                    _record(
                        "normalization",
                        "-1",
                        {
                            "input": "_ns_normalization_input",
                            "norm": "_ns_norm",
                            "denominator": "_ns_denominator",
                            "normalized": "x",
                        },
                    )
                )
                found = True
                continue
        body.append(statement)
    if not found:
        raise ValueError("Pinned NS normalization was not retained")
    body.append(_record("orientation", "-1", {"value": "x"}))
    return body


def _product_trace(original: ast.FunctionDef) -> ast.FunctionDef:
    function = copy.deepcopy(original)
    function.name = "_ns_trace_bmm"
    function.args = ast.parse("def f(x, y, mode, _ns_records, _ns_stage, _ns_iteration):\n    pass").body[0].args

    class RetainReturn(ast.NodeTransformer):
        def visit_Return(self, node):
            product = copy.deepcopy(node.value)
            if not isinstance(product, ast.Call) or len(product.args) != 2:
                raise ValueError("Pinned NS product expression differs")
            body = _statements("_ns_left_operand = 0\n_ns_right_operand = 0\n_ns_product = 0")
            body[0].value, body[1].value = product.args
            product.args = [ast.Name(id=name, ctx=ast.Load()) for name in ("_ns_left_operand", "_ns_right_operand")]
            body[2].value = product
            body.extend(
                _statements(
                    "_ns_records.append({'stage': _ns_stage, 'iteration': _ns_iteration, "
                    "'values': {'left_raw': x, 'right_raw': y, 'left_operand': _ns_left_operand, "
                    "'right_operand': _ns_right_operand, 'product': _ns_product}})\nreturn _ns_product"
                )
            )
            return body

    return RetainReturn().visit(function)


def _recurrence_trace(loop: ast.For) -> ast.For:
    result = copy.deepcopy(loop)
    index = result.target.id

    class RetainProduct(ast.NodeTransformer):
        def __init__(self):
            self.products = []

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "_bmm":
                if len(self.products) >= len(PRODUCT_STAGES) or len(node.args) != 3 or node.keywords:
                    raise ValueError("Pinned NS recurrence product coverage differs")
                stage = PRODUCT_STAGES[len(self.products)]
                self.products.append(stage)
                node.func.id = "_ns_trace_bmm"
                node.args.extend(
                    (
                        ast.Name(id="_ns_records", ctx=ast.Load()),
                        ast.Constant(value=stage),
                        ast.Name(id=index, ctx=ast.Load()),
                    )
                )
            return node

    transformer = RetainProduct()
    result = transformer.visit(result)
    if transformer.products != list(PRODUCT_STAGES):
        raise ValueError("Pinned NS recurrence does not contain all three products")
    result.body.insert(0, _record("iteration_input", index, {"value": "x"}))
    result.body.append(_record("iteration_output", index, {"value": "x"}))
    return result


def source_namespace(path: Path) -> tuple[dict, ast.FunctionDef, list[ast.stmt], ast.For, ast.Return, ast.FunctionDef]:
    """Load the authenticated pinned routine without importing a trainer or optimizer."""
    with replay.analyzer._bound_file(path, replay.MUON_SHA256) as stream:
        bound = Path(f"/proc/self/fd/{stream.fileno()}")
        parsed = ast.parse(bound.read_text(), filename=str(path))
        namespace = {"torch": torch}
        source_definitions(path, ("ABC", "_bmm", "newton_schulz_batched"), namespace)
    original = next(
        node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "newton_schulz_batched"
    )
    loops = [(index, node) for index, node in enumerate(original.body) if isinstance(node, ast.For)]
    if len(loops) != 1 or len(original.body) != loops[0][0] + 2 or not isinstance(original.body[-1], ast.Return):
        raise ValueError("Pinned NS prefix/recurrence/output boundary differs")
    index, loop = loops[0]
    if not isinstance(loop.target, ast.Name):
        raise ValueError("Pinned NS iteration coordinate differs")
    product = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "_bmm")
    return namespace, original, original.body[:index], loop, original.body[-1], product


def compiler_maps(path: Path, compiler: Callable, *, mode: str, steps: int, eps: float) -> CompilerMaps:
    """Derive split and traced functions from pinned AST; the caller supplies compilation."""
    if mode not in ("tpu", "f32") or steps <= 0:
        raise ValueError("Compiler localization requires an explicit supported NS arm")
    namespace, original, prefix, loop, ending, product = source_namespace(path)
    prefix_traced, loop_traced = _prefix_trace(prefix), _recurrence_trace(loop)
    full_traced = copy.deepcopy(original)
    full_traced.name = "_ns_full_trace"
    full_traced.body = [*prefix_traced, loop_traced, *_statements("return {'output': x, 'records': _ns_records}")]
    full_traced.body[-1].value.values[0] = copy.deepcopy(ending.value)
    functions = [
        _function("_ns_prefix", "U, eps, mode", prefix + _statements("return x, tr")),
        _function("_ns_recurrence", "x, steps, mode", [*prefix[:2], loop, *_statements("return x")]),
        _function("_ns_finish", "x, tr", [ending]),
        _function("_ns_prefix_trace", "U, eps, mode", prefix_traced + _statements("return x, tr, _ns_records")),
        _function(
            "_ns_recurrence_trace",
            "x, steps, mode",
            prefix[:2] + _statements("_ns_records = []") + [loop_traced] + _statements("return x, _ns_records"),
        ),
        _product_trace(product),
        full_traced,
    ]
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    eager = partial(namespace["newton_schulz_batched"], steps=steps, eps=eps, mode=mode)
    traced = partial(namespace["_ns_full_trace"], steps=steps, eps=eps, mode=mode)
    normalize, recurrence, finish = (namespace[name] for name in ("_ns_prefix", "_ns_recurrence", "_ns_finish"))
    trace_normalize, trace_recurrence = (namespace[name] for name in ("_ns_prefix_trace", "_ns_recurrence_trace"))

    def split(normalizer: Callable, iterator: Callable) -> Callable:
        def numerical(source):
            x, transposed = normalizer(source, eps, mode)
            return finish(iterator(x, steps, mode), transposed)

        return numerical

    def split_trace(normalizer: Callable, iterator: Callable) -> Callable:
        def numerical(source):
            x, transposed, records = normalizer(source, eps, mode)
            x, iteration_records = iterator(x, steps, mode)
            return {"output": finish(x, transposed), "records": records + iteration_records}

        return numerical

    return CompilerMaps(
        controls={
            "eager": eager,
            "full_compiled": compiler(eager),
            "normalization_compiled": split(compiler(normalize), recurrence),
            "recurrence_compiled": split(normalize, compiler(recurrence)),
        },
        traces={
            "eager": traced,
            "full_compiled": compiler(traced),
            "normalization_compiled": split_trace(compiler(trace_normalize), trace_recurrence),
            "recurrence_compiled": split_trace(trace_normalize, compiler(trace_recurrence)),
        },
        source_sha256=replay.MUON_SHA256,
        mode=mode,
        steps=steps,
        eps=eps,
    )


def coordinates(steps: int) -> list[tuple[str, int]]:
    return [("normalization", -1), ("orientation", -1)] + [
        (stage, index) for index in range(steps) for stage in ("iteration_input", *PRODUCT_STAGES, "iteration_output")
    ]


def require_trace(trace: dict, steps: int, mode: str) -> None:
    """Require full ordered CPU stages, operand dtypes and source-shape-compatible products."""
    if set(trace) != {"output", "records"}:
        raise ValueError("NS trace lacks its complete output and stage records")
    replay._require_cpu_tensor(trace["output"])
    if trace["output"].ndim != 3:
        raise ValueError("NS trace output is not a complete batched matrix")
    records = trace["records"]
    if [(item["stage"], item["iteration"]) for item in records] != coordinates(steps):
        raise ValueError("NS trace stage coverage, order or iteration differs")
    source_shape = trace["output"].shape
    oriented_shape = (
        (source_shape[0], source_shape[2], source_shape[1]) if source_shape[-2] > source_shape[-1] else source_shape
    )
    for record in records:
        stage, values = record["stage"], record["values"]
        fields = (
            NORMALIZATION_FIELDS
            if stage == "normalization"
            else PRODUCT_FIELDS if stage in PRODUCT_STAGES else ("value",)
        )
        if set(record) != {"stage", "iteration", "values"} or set(values) != set(fields):
            raise ValueError("NS trace lacks its complete named intermediates")
        for name, value in values.items():
            operand = stage in PRODUCT_STAGES and name in ("left_operand", "right_operand")
            dtype = torch.bfloat16 if operand and mode == "tpu" else torch.float32
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != dtype
                or value.ndim != 3
            ):
                raise ValueError("NS trace intermediate device, dtype or matrix batch differs")
        if stage == "normalization" and (
            values["input"].shape != source_shape
            or values["normalized"].shape != source_shape
            or values["norm"].shape != (source_shape[0], 1, 1)
            or values["denominator"].shape != (source_shape[0], 1, 1)
        ):
            raise ValueError("NS normalization trace does not retain the complete source batch")
        if stage in ("orientation", "iteration_input", "iteration_output") and values["value"].shape != oriented_shape:
            raise ValueError("NS trace iteration does not retain the complete oriented batch")
        if stage in PRODUCT_STAGES:
            left, right = values["left_raw"], values["right_raw"]
            if (
                values["left_operand"].shape != left.shape
                or values["right_operand"].shape != right.shape
                or left.shape[0] != right.shape[0]
                or left.shape[-1] != right.shape[-2]
                or values["product"].shape != (left.shape[0], left.shape[-2], right.shape[-1])
            ):
                raise ValueError("NS trace product operand or output shape differs")


def compare_traces(actual: dict, expected: dict, *, steps: int, mode: str) -> dict:
    """Compare complete stages and separately locate the first changed BF16 product operand."""
    for value in (actual, expected):
        require_trace(value, steps, mode)
    stages, first, first_operand = [], None, None
    for observed, reference in zip(actual["records"], expected["records"], strict=True):
        values = {
            name: tracing.comparison(
                observed["values"][name], reference["values"][name], f"ns.{reference['stage']}.{name}"
            )
            for name in reference["values"]
        }
        stage = {"stage": reference["stage"], "iteration": reference["iteration"], "values": values}
        stages.append(stage)
        if first is None and any(not value["exact"] for value in values.values()):
            first = stage
        if mode == "tpu" and first_operand is None and reference["stage"] in PRODUCT_STAGES:
            for name in ("left_operand", "right_operand"):
                if not values[name]["exact"]:
                    first_operand = {"stage": reference["stage"], "iteration": reference["iteration"], "operand": name}
                    break
    return {
        "stages": stages,
        "first_changed_stage": first,
        "first_changed_bf16_operand": first_operand,
        "output": replay.compare_outputs(actual["output"], expected["output"]),
    }


def guarded_comparison(
    actual_control: torch.Tensor,
    actual_trace: dict,
    expected_control: torch.Tensor,
    expected_trace: dict,
    *,
    steps: int,
    mode: str,
) -> dict:
    """Withhold stage interpretation when either trace changes its uninstrumented control."""
    for value in (actual_trace, expected_trace):
        require_trace(value, steps, mode)
    guards = {
        "actual": replay.compare_outputs(actual_trace["output"], actual_control),
        "expected": replay.compare_outputs(expected_trace["output"], expected_control),
    }
    eligible = all(value["exact"] and value["finite"] for value in guards.values())
    eligible = eligible and finite_tree(actual_trace) and finite_tree(expected_trace)
    return {
        "trace_transparency": guards,
        "stage_interpretation_permitted": eligible,
        "controls": replay.compare_outputs(actual_control, expected_control),
        "stage_comparison": compare_traces(actual_trace, expected_trace, steps=steps, mode=mode) if eligible else None,
        "scope": SCOPE,
    }


def call_coordinates(index: int) -> list[dict]:
    """Declare two complete opposite arm, compiler-boundary and repeated-input orders."""
    if index not in (0, 1):
        raise ValueError("Compiler localization requires one of two fresh-process indices")
    arms, boundaries, order = ARMS, BOUNDARIES, replay.ORDER
    if index:
        arms, boundaries, order = arms[::-1], boundaries[::-1], order[::-1]
    return [
        {"mode": mode, "steps": steps, "boundary": boundary, "source": source, "call_index": call}
        for mode, steps in arms
        for boundary in boundaries
        for call, source in enumerate(order)
    ]


def retain_protocol(groups: dict, index: int, retain: Callable) -> list[dict]:
    """Retain independent CPU phases; the callback owns all numerical and graph execution."""
    snapshots, layouts = source_snapshots(groups)
    records = []
    for coordinate in call_coordinates(index):
        source = groups[coordinate["source"]]["source"]
        phases = retain(coordinate, source)
        if [phase["phase"] for phase in phases] != list(PHASES):
            raise ValueError("Compiler localization lacks its complete call/capture/replay phases")
        for phase in phases:
            if set(phase) != {"phase", "control", "trace"}:
                raise ValueError("Compiler localization phase lacks control and traced outputs")
            replay._require_cpu_tensor(phase["control"])
            require_trace(phase["trace"], coordinate["steps"], coordinate["mode"])
            if phase["control"].shape != source.shape or phase["trace"]["output"].shape != source.shape:
                raise ValueError("Compiler localization retained only part of its saved source batch")
            records.append({**coordinate, **tracing.cpu_tree(phase)})
        for key, expected in snapshots.items():
            value = groups[key[0]][key[1]]
            layout = (value.shape, value.stride(), value.storage_offset(), value.data_ptr())
            if layout != layouts[key] or not torch.equal(value, expected):
                raise ValueError("Compiler localization changed original source/output values or storage")
    return records


def summarize_protocol(records: list[dict], index: int) -> dict:
    """Independently compare retained phases, repeated inputs and eager/compiler boundaries."""
    expected = [{**coordinate, "phase": phase} for coordinate in call_coordinates(index) for phase in PHASES]
    if [{name: row[name] for name in expected[0]} for row in records] != expected:
        raise ValueError("Compiler localization retained coordinates are missing, duplicated or reordered")
    outputs = {}
    for row in records:
        replay._require_cpu_tensor(row["control"])
        require_trace(row["trace"], row["steps"], row["mode"])
        key = row["mode"], row["steps"], row["boundary"], row["source"], row["call_index"], row["phase"]
        outputs[key] = row
    comparisons, repeats, phases = [], [], []
    for row in records:
        mode, steps, boundary, source, call, phase = (
            row[name] for name in ("mode", "steps", "boundary", "source", "call_index", "phase")
        )
        baseline = outputs[mode, steps, "eager", source, call, phase]
        report = guarded_comparison(
            row["control"], row["trace"], baseline["control"], baseline["trace"], steps=steps, mode=mode
        )
        comparisons.append({name: row[name] for name in expected[0]} | report)
        first = outputs[mode, steps, boundary, source, call, PHASES[0]]
        phases.append(
            {name: row[name] for name in expected[0]}
            | {"control": replay.compare_outputs(row["control"], first["control"])}
        )
        if call >= 2:
            previous = outputs[mode, steps, boundary, source, call - 2, phase]
            repeats.append(
                {name: row[name] for name in expected[0]}
                | {
                    "control": replay.compare_outputs(row["control"], previous["control"]),
                    "trace": tracing.comparison(row["trace"], previous["trace"], "ns.same_input"),
                }
            )
    return {
        "comparisons_to_eager": comparisons,
        "same_input_repeats": repeats,
        "phases_vs_call": phases,
        "all_traces_transparent_exact_finite": all(row["stage_interpretation_permitted"] for row in comparisons),
        "all_control_phases_exact_finite": all(row["control"]["exact"] and row["control"]["finite"] for row in phases),
        "all_same_input_repeats_exact_finite": all(
            row["control"]["exact"] and row["control"]["finite"] and row["trace"]["exact"] and row["trace"]["finite"]
            for row in repeats
        ),
        "scope": SCOPE,
    }
