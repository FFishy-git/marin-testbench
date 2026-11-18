# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-bound saved NS replay primitives; CUDA is unissued without an owner adapter."""

import hashlib
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import analyze_fused_continuation_stages as analyzer
import torch
from benchmark_diagonal import source_definitions
from diagnose_fused_diagonal import MOMENT_TOLERANCE, differences

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
RECEIPT_SHA256 = "b09b6276162b6811aef15ab0b729361c45a0ed1202c3c30fc133894f13837a82"
BINDING_SHA256 = "3eb36cfe0b1f2387bd030a3d0b419f36c9abaf6b87bd5767ff96b0aa9360e277"
MUON_SHA256 = "b03f10c80ce46c61d4db788dda481bc8407273e809b252bd99132c49d665bb42"
SAVED_SHAPE = (16, 512, 512)
NS_STEPS = 10
NS_EPS = 1e-5
ORDER = ("continuous", "resumed", "continuous", "resumed")
GPU_STATUS = "unissued_owner_adapter_absent"


@dataclass(frozen=True)
class ReplayResult:
    report: dict
    outputs: tuple[torch.Tensor, ...]


def _require_cpu_tensor(value: torch.Tensor) -> None:
    if value.device.type != "cpu":
        raise RuntimeError("CUDA replay denied: accepted owner adapter is absent")
    if value.dtype != torch.float32 or value.numel() == 0:
        raise ValueError("Replay requires nonempty FP32 tensors")


def compare_outputs(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Compare retained outputs at the original tolerance, without granting acceptance."""
    for value in (actual, expected):
        _require_cpu_tensor(value)
    if actual.shape != expected.shape:
        raise ValueError("Replay output shape differs from its reference")
    return {
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "exact": torch.equal(actual, expected),
        "changed_elements": int(torch.count_nonzero(actual != expected)),
        **differences(actual, expected, MOMENT_TOLERANCE),
    }


def replay_pair(groups: dict, numerical: Callable[[torch.Tensor], torch.Tensor]) -> ReplayResult:
    """Replay full resident C/R batches in C,R,C,R order on CPU and retain independent clones.

    Args:
        groups: Continuous/resumed source and saved-output tensors; synthetic inputs are allowed.
        numerical: CPU numerical callable accepting a whole source batch without mutating inputs.

    Returns:
        Comparison metrics and detached output clones for later cross-process comparison.
        This helper authenticates no evidence and supplies no numerical or production acceptance.
    """
    if set(groups) != set(analyzer.MODES) or any(set(group) != {"source", "output"} for group in groups.values()):
        raise ValueError("Replay requires exactly the continuous/resumed source/output pairs")
    tensors = {(mode, field): value for mode, group in groups.items() for field, value in group.items()}
    shape = groups["continuous"]["source"].shape
    for value in tensors.values():
        _require_cpu_tensor(value)
        if value.shape != shape or value.ndim != 3 or not torch.isfinite(value).all():
            raise ValueError("Saved replay tensors require one finite, complete batch shape")
    snapshots = {key: value.detach().clone() for key, value in tensors.items()}
    outputs = []
    with torch.no_grad():
        for mode in ORDER:
            output = numerical(groups[mode]["source"])
            if any(not torch.equal(value, snapshots[key]) for key, value in tensors.items()):
                raise ValueError("Numerical replay mutated a source or saved output")
            _require_cpu_tensor(output)
            if output.shape != shape:
                raise ValueError("Numerical replay did not return the full source batch")
            outputs.append(output.detach().clone())
    saved = []
    for index, (mode, output) in enumerate(zip(ORDER, outputs, strict=True)):
        other = "resumed" if mode == "continuous" else "continuous"
        saved.append(
            {
                "call_index": index,
                "source_mode": mode,
                "own_saved_output": compare_outputs(output, groups[mode]["output"]),
                "other_saved_output": compare_outputs(output, groups[other]["output"]),
            }
        )
    report = {
        "schema": 1,
        "status": "cpu_numerical_replay_not_acceptance",
        "gpu_status": GPU_STATUS,
        "owner_context": None,
        "order": list(ORDER),
        "tolerance": dict(MOMENT_TOLERANCE),
        "source_comparison": compare_outputs(groups["resumed"]["source"], groups["continuous"]["source"]),
        "repeats": {
            "continuous": compare_outputs(outputs[2], outputs[0]),
            "resumed": compare_outputs(outputs[3], outputs[1]),
        },
        "cross_outputs": [compare_outputs(outputs[1], outputs[0]), compare_outputs(outputs[3], outputs[2])],
        "saved_comparisons": saved,
    }
    return ReplayResult(report, tuple(outputs))


def load_saved_group(stack: ExitStack) -> tuple[dict, dict]:
    """Authenticate the fixed rank-0/update-23/occurrence-0 pair through local analyzer helpers."""
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
        raise ValueError("Saved diagnostic root, candidate or execution binding differs")
    files = binding["files_sha256"]
    upstream = analyzer._json(
        stack.enter_context(
            analyzer._bound_file(
                candidate / "experiment/upstream_manifest.json", files["experiment/upstream_manifest.json"]
            )
        )
    )
    muon = candidate / "artifacts/pinned-sources-materialized" / f"Toner-{upstream['revision']}/src/toner/fast/muon.py"
    if (
        files[str(muon.relative_to(candidate))] != MUON_SHA256
        or upstream["files"]["src/toner/fast/muon.py"] != MUON_SHA256
    ):
        raise ValueError("Saved diagnostic pinned NS source differs")
    stack.enter_context(analyzer._bound_file(muon, MUON_SHA256))
    config_path = Path(receipt["config"])
    config_sha256 = files[str(config_path.relative_to(candidate))]
    config = analyzer._json(stack.enter_context(analyzer._bound_file(config_path, config_sha256)))
    optimizer = config["optimizer"]
    if (optimizer["ns_mode"], optimizer["ns_steps"], optimizer["ns_eps"]) != ("tpu", NS_STEPS, NS_EPS):
        raise ValueError("Saved diagnostic mixed-BF16 NS configuration differs")
    groups, evidence, identities, raw_layouts = {}, {}, [], {}
    for mode in analyzer.MODES:
        inventory = analyzer._trace_inventory(receipt, ROOT, mode)
        path = analyzer._trace_path(ROOT, mode, 0, 23)
        trace = analyzer._load_trace(
            stack, path, inventory, rank=0, update=23, start_step=22 if mode == "resumed" else 0
        )
        after = [record["values"] for record in trace["records"] if record["stage"] == "matrix/after_update"]
        if len(after) != 3:
            raise ValueError("Saved diagnostic shape-group count differs")
        selected = after[0]["trace"]
        group = {"source": selected["ns_source"], "output": selected["ns_output"]}
        groups[mode], raw_layouts[mode] = {}, {}
        for name, value in group.items():
            _require_cpu_tensor(value)
            if tuple(value.shape) != SAVED_SHAPE or not torch.isfinite(value).all():
                raise ValueError("Saved replay requires the finite full [16,512,512] batch")
            raw_layouts[mode][name] = {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "storage_offset": value.storage_offset(),
            }
            cloned = value.detach().clone(memory_format=torch.preserve_format)
            if (
                cloned.shape != value.shape
                or cloned.stride() != value.stride()
                or cloned.storage_offset() != value.storage_offset()
            ):
                raise ValueError("Saved source/output clone changed the raw trace layout")
            groups[mode][name] = cloned
        identities.append(trace["provenance"]["identity"])
        evidence[mode] = {
            "trace": str(path),
            "trace_sha256": inventory[str(path)],
            "sidecar_sha256": inventory[str(path.with_suffix(".json"))],
        }
    if identities[0] != identities[1]:
        raise ValueError("Saved trace source/config/data identities differ")
    return groups, {
        "receipt": str(receipt_path),
        "receipt_sha256": RECEIPT_SHA256,
        "binding_sha256": BINDING_SHA256,
        "trace_pair": evidence,
        "raw_layouts": raw_layouts,
        "rank": 0,
        "update": 23,
        "occurrence": 0,
        "pinned_muon": str(muon),
        "pinned_muon_sha256": MUON_SHA256,
        "config_sha256": config_sha256,
        "optimizer": optimizer,
        "saved_gate_dependency": {
            "independently_reconstructed": False,
            **{gate: receipt["analysis"][gate] for gate in analyzer.GATES},
        },
        "limitation": analyzer.GATE_LIMITATION,
    }


def replay_saved_ns(*, device: str) -> ReplayResult:
    """Run only a CPU-f32 control; CUDA is denied before evidence I/O until ownership exists."""
    if torch.device(device).type != "cpu":
        raise RuntimeError("CUDA replay denied: accepted owner adapter is absent")
    source_paths = {
        Path(__file__).resolve(),
        Path(analyzer.__file__).resolve(),
        Path(analyzer.compare_saved.__code__.co_filename).resolve(),
        Path(source_definitions.__code__.co_filename).resolve(),
        Path(differences.__code__.co_filename).resolve(),
    }
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    with ExitStack() as stack:
        for path, expected in sources.items():
            stack.enter_context(analyzer._bound_file(Path(path), expected))
        groups, provenance = load_saved_group(stack)
        stream = stack.enter_context(
            analyzer._bound_file(Path(provenance["pinned_muon"]), provenance["pinned_muon_sha256"])
        )
        # Read the authenticated inode, not a replaceable source pathname.
        definitions = source_definitions(
            Path(f"/proc/self/fd/{stream.fileno()}"),
            ("ABC", "_bmm", "newton_schulz_batched"),
            {"torch": torch},
        )
        numerical = partial(definitions["newton_schulz_batched"], steps=NS_STEPS, eps=NS_EPS, mode="f32")
        result = replay_pair(groups, numerical)
        result.report.update(
            provenance=provenance,
            worker_sources_sha256=sources,
            replay_mode="f32",
            saved_mode="tpu",
            mode_limitation="CPU-f32 control is not the saved mixed-BF16 CUDA execution.",
        )
    return result
