# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-bound root replay primitives; CUDA requires a separately accepted execution owner."""

import hashlib
import json
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import analyze_fused_continuation_stages as analyzer
import root_stage_trace as tracing
import torch
from benchmark_diagonal import source_definitions

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
RECEIPT_SHA256 = "b09b6276162b6811aef15ab0b729361c45a0ed1202c3c30fc133894f13837a82"
BINDING_SHA256 = "3eb36cfe0b1f2387bd030a3d0b419f36c9abaf6b87bd5767ff96b0aa9360e277"
ROOTS_SHA256 = "6a61bdefa6fd3fca6365e571baac4208c53d73e9187978be5d75a6e1951f331c"
ORDER = ("continuous", "resumed", "continuous", "resumed")
FIELDS = ("C", "tol", "SB", "S2")
SHAPES = {"d": (12, 512, 512), "f": (4, 2048, 2048)}
STATUS = "saved_root_stage_replay_gpu_unissued"


@dataclass(frozen=True)
class RootReplayResult:
    report: dict
    snapshots: tuple[dict, ...]


def _require_cpu_tree(value: object) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise RuntimeError("Root replay requires CPU evidence; CUDA execution owner is unissued")
    elif isinstance(value, dict):
        for item in value.values():
            _require_cpu_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _require_cpu_tree(item)


def clone_inputs(inputs: dict) -> dict:
    """Clone finite root-input storage without normalizing its strides or offsets."""
    if set(inputs) != set(FIELDS):
        raise ValueError("Root replay requires C/tol/SB/S2")
    cloned = {}
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("Root replay inputs must be tensors")
        _require_cpu_tree(value)
        if not value.numel() or not torch.isfinite(value).all():
            raise ValueError("Root replay inputs must be nonempty and finite")
        result = value.detach().clone(memory_format=torch.preserve_format)
        if result.stride() != value.stride() or result.storage_offset() != value.storage_offset():
            raise ValueError("Root replay clone changed the saved raw layout")
        cloned[name] = result
    return cloned


def replay_root_pair(groups: dict, numerical: Callable[[str, dict], dict]) -> RootReplayResult:
    """Retain four full root-stage snapshots in C,R,C,R order.

    Args:
        groups: Continuous/resumed root inputs and their historical candidate outputs.
        numerical: Caller-owned numerical function returning a completed CPU stage
            snapshot. A CUDA caller needs its separate accepted owner and must
            synchronize before returning any CPU evidence.

    Returns:
        Independent retained snapshots, same-input repeats, crossed historical-output
        metrics and complete ordered stage comparisons. This function authenticates
        no files and cannot authorize CUDA, continuation or production.
    """
    if set(groups) != {"continuous", "resumed"} or any(
        set(group) != {"inputs", "saved_output"} for group in groups.values()
    ):
        raise ValueError("Root replay requires both exact input/historical-output pairs")
    before = {}
    for mode, group in groups.items():
        before[mode] = clone_inputs(group["inputs"])
        _require_cpu_tree(group["saved_output"])
        if not tracing.comparison(group["saved_output"], group["saved_output"], "saved")["finite"]:
            raise ValueError("Root replay historical candidate output must be finite")
    shapes = [group["inputs"]["C"].shape for group in groups.values()]
    if shapes[0] != shapes[1] or len(shapes[0]) != 3 or shapes[0][-2] != shapes[0][-1]:
        raise ValueError("Root replay requires the same complete square batch shape")
    snapshots, historical = [], []
    historical_before = {mode: tracing.cpu_tree(group["saved_output"]) for mode, group in groups.items()}
    for index, mode in enumerate(ORDER):
        snapshot = numerical(mode, groups[mode]["inputs"])
        _require_cpu_tree(snapshot)
        for name, group in groups.items():
            unchanged = tracing.comparison(group["inputs"], before[name], "resident.inputs")
            saved_unchanged = tracing.comparison(group["saved_output"], historical_before[name], "historical")
            if not unchanged["exact"] or not saved_unchanged["exact"]:
                raise ValueError("Root replay mutated a source input or historical output")
        snapshot = tracing.cpu_tree(snapshot)
        self_comparison = tracing.compare_root_stages(snapshot, snapshot)
        if (
            not self_comparison["complete_stage_coverage"]
            or self_comparison["tensor_comparisons"] == 0
            or self_comparison["output"]["tensor_comparisons"] == 0
        ):
            raise ValueError("Root replay must retain tensor stages and its complete candidate output")
        other = "resumed" if mode == "continuous" else "continuous"
        historical.append(
            {
                "call_index": index,
                "source_mode": mode,
                "own_saved_output": tracing.comparison(snapshot["output"], groups[mode]["saved_output"], "own.saved"),
                "other_saved_output": tracing.comparison(
                    snapshot["output"], groups[other]["saved_output"], "other.saved"
                ),
            }
        )
        snapshots.append(snapshot)
    report = {
        "status": STATUS,
        "owner_context": None,
        "order": list(ORDER),
        "input_comparison": tracing.comparison(
            groups["resumed"]["inputs"], groups["continuous"]["inputs"], "root.inputs"
        ),
        "repeats": {
            "continuous": tracing.compare_root_stages(snapshots[2], snapshots[0]),
            "resumed": tracing.compare_root_stages(snapshots[3], snapshots[1]),
        },
        "crossed_stages": [
            tracing.compare_root_stages(snapshots[1], snapshots[0]),
            tracing.compare_root_stages(snapshots[3], snapshots[2]),
        ],
        "historical_comparisons": historical,
        "limitation": tracing.LIMITATION,
    }
    return RootReplayResult(report, tuple(snapshots))


def _require_saved_group(inputs: dict, output: object, shape: tuple[int, ...]) -> None:
    batch, width, _ = shape
    input_shapes = {"C": shape, "tol": (batch, 3), "SB": (batch, 2 * width, width), "S2": (batch, width, 2 * width)}

    def require_tensor(value: object, expected: tuple[int, ...], dtype: torch.dtype) -> None:
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != dtype
            or tuple(value.shape) != expected
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Historical root group lacks finite full-shaped input/output tensor coverage")

    for name, expected in input_shapes.items():
        require_tensor(inputs[name], expected, torch.float16 if name in ("SB", "S2") else torch.float32)
    if (
        not isinstance(output, tuple)
        or len(output) != 4
        or not isinstance(output[0], tuple)
        or len(output[0]) != 2
        or not isinstance(output[3], tuple)
        or len(output[3]) != 3
    ):
        raise ValueError("Historical root candidate output structure differs")
    for value in (*output[0], output[1], output[3][1]):
        require_tensor(value, shape, torch.float32)
    require_tensor(output[2], (batch,), torch.bool)
    require_tensor(output[3][0], (batch, width), torch.float32)
    require_tensor(output[3][2], (batch,), torch.float32)


def load_saved_roots(stack: ExitStack, group: str) -> tuple[dict, dict]:
    """Authenticate one complete historical rank-0/update-23 input-root group."""
    if group not in SHAPES:
        raise ValueError("Select the historical d or f root group")
    receipt_path = ROOT / analyzer.RECEIPT_RELATIVE
    binding_path = ROOT / "run-preparation/fused-continuation-binding-20261118.json"
    receipt = analyzer._json(stack.enter_context(analyzer._bound_file(receipt_path, RECEIPT_SHA256)))
    analyzer._require_saved_gates(receipt)
    binding = analyzer._json(stack.enter_context(analyzer._bound_file(binding_path, BINDING_SHA256)))
    candidate = Path(receipt["candidate"])
    if (
        receipt["root"] != str(ROOT)
        or binding["candidate"] != str(candidate)
        or receipt["execution_binding_sha256"] != BINDING_SHA256
    ):
        raise ValueError("Saved root input diagnostic identity differs")
    sources = binding["files_sha256"]
    manifest_path = candidate / "experiment/upstream_manifest.json"
    manifest = analyzer._json(
        stack.enter_context(analyzer._bound_file(manifest_path, sources["experiment/upstream_manifest.json"]))
    )
    upstream = candidate / f"artifacts/pinned-sources-materialized/Toner-{manifest['revision']}"
    roots_path = upstream / "src/toner/fast/roots.py"
    if (
        sources[str(roots_path.relative_to(candidate))] != ROOTS_SHA256
        or manifest["files"]["src/toner/fast/roots.py"] != ROOTS_SHA256
    ):
        raise ValueError("Historical root numerical source differs")
    # Keep source inodes authenticated throughout the caller's replay, not only at load time.
    for name, expected in sources.items():
        path = candidate / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Historical root source inventory escapes its candidate")
        stack.enter_context(analyzer._bound_file(path, expected))
    config_path = Path(receipt["config"])
    config_hash = sources[str(config_path.relative_to(candidate))]
    config = analyzer._json(stack.enter_context(analyzer._bound_file(config_path, config_hash)))
    if (
        config["root"]["iterations"] != 2
        or config["root"]["graphs"] is not True
        or config["statistics"]["inner_strength"] != 0.5
        or config["statistics"]["outer_strength"] != 0.125
        or config["statistics"]["damping"] != 1e-3
    ):
        raise ValueError("Historical split-root configuration differs")
    data_path = ROOT / "data/experiment_receipt.json"
    data_hash = binding["data_receipt_sha256"]
    stack.enter_context(analyzer._bound_file(data_path, data_hash))
    records_path = candidate / "experiment/records.py"
    records_stream = stack.enter_context(analyzer._bound_file(records_path, sources["experiment/records.py"]))
    alias = candidate / "artifacts/external-sources"
    if (
        not alias.is_symlink()
        or alias.readlink() != Path("pinned-sources-materialized")
        or alias.resolve() != upstream.parent
    ):
        raise ValueError("Historical root runtime upstream alias differs")
    records = source_definitions(
        Path(f"/proc/self/fd/{records_stream.fileno()}"),
        ("sha256", "checkpoint_identity", "CHECKPOINT_SCHEMA"),
        {"TASK": candidate, "UPSTREAM": alias / upstream.name, "Path": Path, "hashlib": hashlib, "json": json},
    )
    identity = records["checkpoint_identity"](config, data_path)
    if identity["data_receipt_sha256"] != data_hash:
        raise ValueError("Historical root data identity differs from its binding")
    arguments_path = upstream / "reproduce/marin/optimizer.py"
    arguments_stream = stack.enter_context(
        analyzer._bound_file(arguments_path, sources[str(arguments_path.relative_to(candidate))])
    )
    arguments = source_definitions(
        Path(f"/proc/self/fd/{arguments_stream.fileno()}"),
        ("fast_arguments",),
        {"SimpleNamespace": SimpleNamespace},
    )["fast_arguments"](config)
    if arguments.toner_root != "ref" or arguments.toner_split is not True or arguments.toner_graphs is not True:
        raise ValueError("Historical root execution did not select reference split-root graphs")
    pairs, evidence, identities = {}, {}, []
    for mode in ("continuous", "resumed"):
        inventory = analyzer._trace_inventory(receipt, ROOT, mode)
        path = analyzer._trace_path(ROOT, mode, 0, 23)
        trace = analyzer._load_trace(
            stack, path, inventory, rank=0, update=23, start_step=22 if mode == "resumed" else 0
        )
        before = [record["values"] for record in trace["records"] if record["stage"] == f"input/{group}/before_root"]
        after = [record["values"] for record in trace["records"] if record["stage"] == f"input/{group}/candidate"]
        if len(before) != 1 or len(after) != 1 or tuple(before[0]["C"].shape) != SHAPES[group]:
            raise ValueError("Historical root group must contain its full input batch and candidate")
        inputs = {name: before[0][name] for name in FIELDS}
        _require_saved_group(inputs, after[0], SHAPES[group])
        pairs[mode] = {"inputs": clone_inputs(inputs), "saved_output": tracing.cpu_tree(after[0])}
        identities.append(trace["provenance"]["identity"])
        evidence[mode] = {
            "trace": str(path),
            "trace_sha256": inventory[str(path)],
            "sidecar_sha256": inventory[str(path.with_suffix(".json"))],
            "input_layouts": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "stride": list(value.stride()),
                    "storage_offset": value.storage_offset(),
                }
                for name, value in inputs.items()
            },
        }
    if any(saved != identity for saved in identities):
        raise ValueError("Historical root trace identity differs from executed source/config/data")
    return pairs, {
        "receipt": str(receipt_path),
        "receipt_sha256": RECEIPT_SHA256,
        "binding_sha256": BINDING_SHA256,
        "pinned_roots": str(roots_path),
        "pinned_roots_sha256": ROOTS_SHA256,
        "trace_pair": evidence,
        "group": group,
        "rank": 0,
        "update": 23,
        "config": config,
        "config_sha256": config_hash,
        "data_receipt_sha256": data_hash,
        "identity": identity,
        "root_execution": {
            "mode": arguments.toner_root,
            "split": arguments.toner_split,
            "graphs": arguments.toner_graphs,
            "iterations": arguments.toner_iters,
            "inner_power": arguments.toner_alpha_in,
            "outer_power": arguments.toner_alpha_out,
        },
        "saved_gate_dependency": {
            "independently_reconstructed": False,
            **{gate: receipt["analysis"][gate] for gate in analyzer.GATES},
        },
        "limitation": analyzer.GATE_LIMITATION,
    }
