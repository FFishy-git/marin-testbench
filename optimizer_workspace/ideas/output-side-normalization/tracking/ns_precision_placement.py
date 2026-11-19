# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU screen of higher-precision iteration placement in the pinned finite NS map."""

import argparse
import ast
import copy
import os
import sys
from contextlib import ExitStack
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import ns_compiler_localization as localization
import ns_operand_precision as precision
import ns_step_budget as budget
import replay_saved_ns as replay
import torch
from diagnose_fused_diagonal import MOMENT_TOLERANCE, differences
from ns_training_path_predecessor import accepted_archive
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

REFERENCE_RESULT = "run-preparation/ns-step-budget-saved-pair-20261118T2300/result.json"
REFERENCE_SHA256 = "f7b0269ba62531b908039c6b6cb16851730efe03afa298c54f3f31aaebcbd9ed"
RANKING = (
    "Compare prefix/suffix at fixed steps and FP32 product count. Rank by C/R relative "
    "Frobenius error, report worst-matrix error and own-reference accuracy alongside it. "
    "A smaller pair difference is not a continuation fix or training-quality criterion."
)
SCOPE = (
    "Exploratory CPU per-iteration precision placement on the authenticated historical "
    "rank0/update23/occurrence0 full saved pair. All three products in an iteration use "
    "FP32 or CPU BF16-operand/FP32-accumulation emulation. Normalization, coefficients, "
    "eps and original tolerances are unchanged. Mixed schedules change executed numerics. "
    "Product counts are arithmetic inventory, not measured CUDA or training speed. "
    "No CUDA reproduction, continuation correction, optimizer quality, efficiency, "
    "SwanLab or production acceptance."
)


def schedules() -> dict[str, tuple[str, ...]]:
    """Compare matched prefix/suffix placements against homogeneous five/seven/ten controls."""
    result = {}
    for steps in (5, 7, 10):
        result[f"q{steps}_bf16"] = ("bf16_1",) * steps
        result[f"q{steps}_f32"] = ("f32",) * steps
        if steps == 10:
            continue
        for count in (1, 2, 3):
            result[f"q{steps}_f32_prefix{count}"] = ("f32",) * count + ("bf16_1",) * (steps - count)
            result[f"q{steps}_f32_suffix{count}"] = ("bf16_1",) * (steps - count) + ("f32",) * count
    return result


def numerical(path: Path, schedule: tuple[str, ...], *, eps: float):
    """Derive the pinned AST, changing only the product mode for each complete iteration."""
    if not schedule or set(schedule) - {"f32", "bf16_1"}:
        raise ValueError("Precision placement requires an explicit FP32/BF16 iteration schedule")
    namespace, original, _, _, _, _ = localization.source_namespace(path)
    function = copy.deepcopy(original)
    function.name = "_ns_precision_placement"
    loop = next(node for node in function.body if isinstance(node, ast.For))
    loop.iter = ast.parse("enumerate(_ns_schedule)", mode="eval").body
    loop.target = ast.Tuple(
        elts=[copy.deepcopy(loop.target), ast.Name(id="_ns_iteration_mode", ctx=ast.Store())],
        ctx=ast.Store(),
    )
    products = [node for node in ast.walk(loop) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    products = [node for node in products if node.func.id == "_bmm"]
    if len(products) != 3 or any(len(node.args) != 3 or node.keywords for node in products):
        raise ValueError("Pinned NS iteration product coverage differs")
    for node in products:
        node.args[2] = ast.Name(id="_ns_iteration_mode", ctx=ast.Load())
    namespace.update(_bmm=precision.operand_bmm, _ns_schedule=schedule)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    evaluate = partial(namespace[function.name], steps=len(schedule), eps=eps, mode="f32")

    def cpu_map(source):
        precision.require_cpu((source,))
        return evaluate(source)

    return cpu_map


def spectral_reference(stack: ExitStack) -> tuple[dict, dict]:
    """Load the independently evaluated, unchanged historical SVD-polynomial reference."""
    result_path = replay.ROOT / REFERENCE_RESULT
    result = replay.analyzer._json(stack.enter_context(replay.analyzer._bound_file(result_path, REFERENCE_SHA256)))
    path = result_path.parent / "spectral-references.pt"
    stream = stack.enter_context(replay.analyzer._bound_file(path, result["reference_sha256"]))
    references = torch.load(stream, map_location="cpu", weights_only=True)
    if set(references) != {"continuous", "resumed"} or any(
        set(group) != {"partial_polar", "3", "5", "7", "10"} for group in references.values()
    ):
        raise ValueError("Historical polynomial reference inventory differs")
    if any(
        value.device.type != "cpu"
        or value.dtype != torch.float64
        or tuple(value.shape) != replay.SAVED_SHAPE
        or not torch.isfinite(value).all()
        for group in references.values()
        for value in group.values()
    ):
        raise ValueError("Historical polynomial reference tensors differ")
    return references, {
        "result": str(result_path),
        "result_sha256": REFERENCE_SHA256,
        "reference": str(path),
        "reference_sha256": result["reference_sha256"],
        "scope": "Retained independent FP64 SVD-polynomial reference; arithmetic is not rerun here.",
    }


def historical_control(stack: ExitStack, name: str, outputs: tuple) -> dict | None:
    """Require exact saved homogeneous output reproduction before interpreting placements."""
    if name not in {f"q{steps}_{mode}" for steps in (5, 7, 10) for mode in ("f32", "bf16")}:
        return None
    steps, mode = name.split("_")
    reference_path = replay.ROOT / REFERENCE_RESULT
    reference = replay.analyzer._json(stack.enter_context(replay.analyzer._bound_file(reference_path, REFERENCE_SHA256)))
    mode = "bf16_1" if mode == "bf16" else mode
    record = next(entry for entry in reference["trials"] if entry["mode"] == mode and entry["steps"] == int(steps[1:]))
    path = reference_path.parent / f"{mode}-{steps}.pt"
    stream = stack.enter_context(replay.analyzer._bound_file(path, record["output_sha256"]))
    expected = torch.load(stream, map_location="cpu", weights_only=True)
    comparisons = [replay.compare_outputs(actual, saved) for actual, saved in zip(outputs, expected, strict=True)]
    return {
        "output": str(path),
        "output_sha256": record["output_sha256"],
        "comparisons": comparisons,
        "all_exact_finite": all(entry["exact"] and entry["finite"] for entry in comparisons),
    }


def summaries(outputs: tuple, groups: dict, references: dict, steps: int) -> dict:
    """Reconstruct saved-pair, repeat, finite-polynomial and geometric comparisons."""
    return {
        "source_pair": replay.compare_outputs(groups["resumed"]["source"], groups["continuous"]["source"]),
        "output_pair": replay.compare_outputs(outputs[1], outputs[0]),
        "repeat_pair": replay.compare_outputs(outputs[3], outputs[2]),
        "same_input_repeats": {
            "continuous": replay.compare_outputs(outputs[2], outputs[0]),
            "resumed": replay.compare_outputs(outputs[3], outputs[1]),
        },
        "per_matrix_output_pairs": [
            replay.compare_outputs(actual, expected) for actual, expected in zip(outputs[1], outputs[0], strict=True)
        ],
        "vs_fp64_finite_polynomial": {
            source: differences(outputs[index], references[source][str(steps)], MOMENT_TOLERANCE)
            for index, source in enumerate(replay.ORDER[:2])
        },
        "geometry": {
            source: budget.geometry(outputs[index], groups[source]["source"], references[source]["partial_polar"])
            for index, source in enumerate(replay.ORDER[:2])
        },
    }


def require_origins(cpu: dict) -> None:
    """Deny a mutable entry or helper import before accepted evidence or output is accessed."""
    helpers = Path(cpu["helpers"])
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Precision placement requires the accepted archived entry")
    for name in cpu["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem)
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError("Precision placement imported mutable helper bytes")


def retained_replay(outputs: tuple, groups: dict) -> dict:
    """Reconstruct every published replay comparison from independently retained output tensors."""
    return {
        "schema": 1,
        "status": "cpu_numerical_replay_not_acceptance",
        "gpu_status": replay.GPU_STATUS,
        "owner_context": None,
        "order": list(replay.ORDER),
        "tolerance": dict(MOMENT_TOLERANCE),
        "source_comparison": replay.compare_outputs(groups["resumed"]["source"], groups["continuous"]["source"]),
        "repeats": {
            "continuous": replay.compare_outputs(outputs[2], outputs[0]),
            "resumed": replay.compare_outputs(outputs[3], outputs[1]),
        },
        "cross_outputs": [
            replay.compare_outputs(outputs[1], outputs[0]),
            replay.compare_outputs(outputs[3], outputs[2]),
        ],
        "saved_comparisons": [
            {
                "call_index": index,
                "source_mode": mode,
                "own_saved_output": replay.compare_outputs(output, groups[mode]["output"]),
                "other_saved_output": replay.compare_outputs(
                    output, groups["resumed" if mode == "continuous" else "continuous"]["output"]
                ),
            }
            for index, (mode, output) in enumerate(zip(replay.ORDER, outputs, strict=True))
        ],
    }


def run(out: Path, cpu_receipt: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or out.exists():
        raise RuntimeError("Precision placement screen requires hidden CUDA and unused external output")
    cpu = read(cpu_receipt)
    if cpu["status"] != "cpu_precision_placement_sources_tests_verified_no_gpu_authority":
        raise ValueError("Precision placement CPU/source archive is not accepted")
    require_origins(cpu)
    accepted_archive(
        replay.ROOT,
        (str(cpu_receipt.relative_to(replay.ROOT)), sha256(cpu_receipt), cpu["status"], cpu["test_count"]),
    )
    helpers = Path(cpu["helpers"])
    for name, digest in cpu["helpers_sha256"].items():
        if sha256(helpers / name) != digest:
            raise ValueError("Precision placement accepted helper changed")
    source_hashes = {str(helpers / name): digest for name, digest in cpu["helpers_sha256"].items()}
    out.mkdir(parents=True)
    receipt = out / "result.json"
    result = {
        "status": "cpu_precision_placement_running",
        "time": datetime.now(UTC).isoformat(),
        "cpu_receipt": str(cpu_receipt),
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "source_sha256": source_hashes,
        "tolerance": MOMENT_TOLERANCE,
        "scope": SCOPE,
        "ranking": RANKING,
        "trials": [],
    }
    write(receipt, result)
    try:
        torch.set_num_threads(8)
        with torch.no_grad(), ExitStack() as stack:
            for path, digest in source_hashes.items():
                stack.enter_context(replay.analyzer._bound_file(Path(path), digest))
            groups, provenance = replay.load_saved_group(stack)
            references, reference_provenance = spectral_reference(stack)
            result.update(provenance=provenance, reference_provenance=reference_provenance)
            path = Path(provenance["pinned_muon"])
            for name, schedule in schedules().items():
                trial = replay.replay_pair(groups, numerical(path, schedule, eps=replay.NS_EPS))
                output = out / f"{name}.pt"
                with output.open("xb") as stream:
                    torch.save(trial.outputs, stream)
                comparison = summaries(trial.outputs, groups, references, len(schedule))
                result["trials"].append(
                    {
                        "name": name,
                        "schedule": list(schedule),
                        "steps": len(schedule),
                        "fp32_products": 3 * schedule.count("f32"),
                        "bf16_operand_products": 3 * schedule.count("bf16_1"),
                        "output": str(output),
                        "output_sha256": sha256(output),
                        "replay": trial.report,
                        "comparisons": comparison,
                        "historical_homogeneous_control": historical_control(stack, name, trial.outputs),
                    }
                )
                result["time"] = datetime.now(UTC).isoformat()
                write(receipt, result)
                pair = comparison["output_pair"]
                print(f"{name}: pair_error={pair['relative_frobenius_error']}", flush=True)
        result.update(
            status="completed_cpu_precision_placement_not_cuda_or_optimizer_acceptance",
            homogeneous_controls_reproduced_exact_finite=all(
                entry["historical_homogeneous_control"]["all_exact_finite"]
                for entry in result["trials"]
                if entry["historical_homogeneous_control"] is not None
            ),
        )
        result["time"] = datetime.now(UTC).isoformat()
        write(receipt, result)
        return result
    except BaseException as error:
        result.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        result["time"] = datetime.now(UTC).isoformat()
        write(receipt, result)
        raise


def verify(out: Path) -> dict:
    """Verify retained outputs and rederive all reported numerical summaries without rerunning maps."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Precision placement retained verification requires hidden CUDA")
    path = out / "result.json"
    digest, result = sha256(path), read(path)
    if result["status"] != "completed_cpu_precision_placement_not_cuda_or_optimizer_acceptance":
        raise ValueError("Precision placement result is incomplete")
    if [entry["name"] for entry in result["trials"]] != list(schedules()):
        raise ValueError("Precision placement schedule coverage differs")
    cpu_receipt = Path(result["cpu_receipt"])
    require_origins(read(cpu_receipt))
    cpu = accepted_archive(
        replay.ROOT,
        (
            str(cpu_receipt.relative_to(replay.ROOT)),
            result["cpu_receipt_sha256"],
            "cpu_precision_placement_sources_tests_verified_no_gpu_authority",
            read(cpu_receipt)["test_count"],
        ),
    )
    if (
        result["source_sha256"]
        != {str(Path(cpu["helpers"]) / name): expected for name, expected in cpu["helpers_sha256"].items()}
        or result["ranking"] != RANKING
        or result["scope"] != SCOPE
        or result["tolerance"] != MOMENT_TOLERANCE
    ):
        raise ValueError("Precision placement accepted sources or ranking differ")
    torch.set_num_threads(8)
    reports = []
    with torch.no_grad(), ExitStack() as stack:
        stack.enter_context(replay.analyzer._bound_file(path, digest))
        for source, expected in result["source_sha256"].items():
            stack.enter_context(replay.analyzer._bound_file(Path(source), expected))
        groups, provenance = replay.load_saved_group(stack)
        references, reference_provenance = spectral_reference(stack)
        if provenance != result["provenance"] or reference_provenance != result["reference_provenance"]:
            raise ValueError("Precision placement retained provenance differs")
        for entry in result["trials"]:
            output = out / f"{entry['name']}.pt"
            stream = stack.enter_context(replay.analyzer._bound_file(output, entry["output_sha256"]))
            outputs = torch.load(stream, map_location="cpu", weights_only=True)
            if (
                entry["output"] != str(output)
                or tuple(entry["schedule"]) != schedules()[entry["name"]]
                or entry["steps"] != len(entry["schedule"])
                or entry["fp32_products"] != 3 * entry["schedule"].count("f32")
                or entry["bf16_operand_products"] != 3 * entry["schedule"].count("bf16_1")
                or len(outputs) != len(replay.ORDER)
                or any(tuple(value.shape) != replay.SAVED_SHAPE for value in outputs)
            ):
                raise ValueError("Precision placement retained output inventory differs")
            comparison = summaries(outputs, groups, references, entry["steps"])
            control = historical_control(stack, entry["name"], outputs)
            replay_report = retained_replay(outputs, groups)
            if (
                comparison != entry["comparisons"]
                or control != entry["historical_homogeneous_control"]
                or replay_report != entry["replay"]
            ):
                raise ValueError("Precision placement retained comparison arithmetic differs")
            reports.append(
                {
                    "name": entry["name"],
                    "output_sha256": entry["output_sha256"],
                    "historical_homogeneous_control": control,
                    "replay": replay_report,
                    **comparison,
                }
            )
    reproduced = all(
        entry["historical_homogeneous_control"]["all_exact_finite"]
        for entry in reports
        if entry["historical_homogeneous_control"] is not None
    )
    if result["homogeneous_controls_reproduced_exact_finite"] != reproduced:
        raise ValueError("Precision placement homogeneous interpretation predicate differs")
    verification = {
        "status": "cpu_precision_placement_retained_comparisons_verified_no_gpu_authority",
        "time": datetime.now(UTC).isoformat(),
        "result_sha256": digest,
        "homogeneous_controls_reproduced_exact_finite": reproduced,
        "scope": SCOPE,
        "trials": reports,
    }
    write(out / "retained-verification.json", verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("run", "verify"))
    parser.add_argument("--cpu-receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "run" and args.cpu_receipt is None:
        parser.error("--cpu-receipt is required for run")
    result = run(args.out.resolve(), args.cpu_receipt) if args.phase == "run" else verify(args.out.resolve())
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
