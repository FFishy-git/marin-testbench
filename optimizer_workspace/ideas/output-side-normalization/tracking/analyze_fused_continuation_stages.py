# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-aligned CPU inspection dependent on saved continuation transparency gates."""

import argparse
import hashlib
import json
import os
import stat
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch
from diagnose_continuation import compare_saved, digest
from diagnose_fused_diagonal import MOMENT_TOLERANCE, differences

COMPLETE_STATUS = "completed_fused_continuation_diagnostic_not_production_gate"
RECEIPT_RELATIVE = "run-preparation/diagnostic-fused-continuation-gpu-20261118.json"
WORK_RELATIVE = "validation/fused_continuation_20261118"
UPDATES = (22, 23, 24)
WORLD = 8
MODES = ("continuous", "resumed")
GATES = ("trace_transparent", "extra_checkpoint_transparent")
MATRIX_STAGES = ("matrix/before_update", "matrix/after_update")
SINGLE_STAGES = (
    "input/statistics",
    "input/before_finish",
    "input/finished",
    "model/gradients",
    "model/after_update",
)
ROOT_STAGES = tuple(f"input/{group}/{phase}" for group in ("d", "f") for phase in ("before_root", "candidate"))
BRANCH_STAGES = frozenset(
    f"input/{branch}/{phase}" for branch in ("refinement", "exact_fallback") for phase in ("input", "output")
)
JSON_BYTE_LIMIT = 32 << 20
ALIGNMENT_LIMITATION = (
    "Matrix occurrence assumes the frozen runtime's deterministic shape-group order; it is not an independent "
    "matrix identity. Unequal matrix occurrence counts are not prefix-paired. Refinement/fallback occurrence "
    "does not identify a group, flagged subset or repair round: branch tensors are not paired or compared. "
    "Their record indices and value structures refer to the retained input traces."
)
GATE_LIMITATION = (
    "Depends on the supplied completed diagnostic receipt's saved transparency gates. Trace bytes and selected "
    "provenance are authenticated here; transparency, commands, checkpoints and source closure are not "
    "independently reconstructed. Stage differences are not causal proof or numerical/production acceptance."
)


@dataclass(frozen=True)
class StageRecord:
    stage: str
    index: int
    occurrence: int
    values: object


def _indexed(records: list[dict]) -> list[StageRecord]:
    counts = Counter()
    indexed = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"stage", "values"} or not isinstance(record["stage"], str):
            raise ValueError("Trace record lacks its exact stage/value structure")
        label = record["stage"]
        occurrence = counts[label]
        counts[label] += 1
        indexed.append(StageRecord(label, index, occurrence, record["values"]))
    return indexed


def _value_structure(value):
    if isinstance(value, torch.Tensor):
        return {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): _value_structure(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value_structure(item) for item in value]
    return value


def _descriptor(record: StageRecord) -> dict:
    return {
        "stage": record.stage,
        "record_index": record.index,
        "occurrence": record.occurrence,
        "value_structure": _value_structure(record.values),
    }


def _nonfinite_tensors(value: object, name: str) -> list[dict]:
    if isinstance(value, torch.Tensor):
        count = int(torch.count_nonzero(~torch.isfinite(value)).item())
        return (
            [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype), "nonfinite_elements": count}]
            if count
            else []
        )
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in _nonfinite_tensors(child, f"{name}.{key}")]
    if isinstance(value, (list, tuple)):
        return [item for index, child in enumerate(value) for item in _nonfinite_tensors(child, f"{name}.{index}")]
    return []


def compare_stages(actual: list[dict], expected: list[dict], *, name: str = "trace") -> dict:
    """Compare stable stage values without pairing branch events by occurrence.

    Args:
        actual: Already loaded resumed stage records.
        expected: Already loaded continuous stage records.
        name: Prefix identifying this rank/update in tensor difference names.

    Returns:
        Stable comparisons, tensor finiteness/coverage, and unpaired branch descriptors.
        This pure function does not authenticate evidence or establish transparency.
    """
    indexed = {"actual": _indexed(actual), "expected": _indexed(expected)}
    labels = SINGLE_STAGES + ROOT_STAGES + MATRIX_STAGES
    stable = {
        side: {(record.stage, record.occurrence): record for record in records if record.stage in labels}
        for side, records in indexed.items()
    }
    counts = {side: Counter(record.stage for record in records) for side, records in indexed.items()}
    for side in counts:
        if any(counts[side][label] > 1 for label in SINGLE_STAGES + ROOT_STAGES):
            raise ValueError("A singleton stable stage occurs more than once")
    missing = {side: [label for label in SINGLE_STAGES + MATRIX_STAGES if counts[side][label] == 0] for side in counts}
    ambiguous = [label for label in MATRIX_STAGES if counts["actual"][label] != counts["expected"][label]]
    paired_keys = stable["actual"].keys() & stable["expected"].keys()
    paired_keys = {key for key in paired_keys if key[0] not in ambiguous}
    pairs = []
    for key in sorted(paired_keys, key=lambda key: stable["expected"][key].index):
        observed, reference = stable["actual"][key], stable["expected"][key]
        pair_name = f"{name}.{reference.stage}[{reference.occurrence}]"
        changed, scalars = [], []
        count = compare_saved(observed.values, reference.values, pair_name, changed, scalars)
        nonfinite = {
            "actual": _nonfinite_tensors(observed.values, pair_name),
            "expected": _nonfinite_tensors(reference.values, pair_name),
        }
        nonfinite_count = sum(item["nonfinite_elements"] for tensors in nonfinite.values() for item in tensors)
        pairs.append(
            {
                "stage": reference.stage,
                "occurrence": reference.occurrence,
                "actual_index": observed.index,
                "expected_index": reference.index,
                "tensor_comparisons": count,
                "tensor_coverage_sufficient": count > 0,
                "nonfinite_tensors": nonfinite,
                "nonfinite_element_count": nonfinite_count,
                "changed_tensors": changed,
                "scalar_or_structure_differences": scalars,
                "within_tolerance": (
                    count > 0
                    and nonfinite_count == 0
                    and not scalars
                    and all(item["within_tolerance"] for item in changed)
                ),
            }
        )
    unmatched = {
        side: [
            _descriptor(record)
            for record in records
            if record.stage not in BRANCH_STAGES and (record.stage, record.occurrence) not in paired_keys
        ]
        for side, records in indexed.items()
    }
    branches = {
        side: [_descriptor(record) for record in records if record.stage in BRANCH_STAGES]
        for side, records in indexed.items()
    }
    matrix_counts = {side: {label: counts[side][label] for label in MATRIX_STAGES} for side in counts}
    complete = (
        not any(missing.values())
        and not any(unmatched.values())
        and all(values[MATRIX_STAGES[0]] == values[MATRIX_STAGES[1]] for values in matrix_counts.values())
    )
    insufficient = [
        {"stage": pair["stage"], "occurrence": pair["occurrence"]}
        for pair in pairs
        if not pair["tensor_coverage_sufficient"]
    ]
    return {
        "matched_stages": pairs,
        "tensor_comparisons": sum(pair["tensor_comparisons"] for pair in pairs),
        "missing_required_stages": missing,
        "unmatched_records": unmatched,
        "ambiguous_matrix_stages": ambiguous,
        "matrix_stage_counts": matrix_counts,
        "complete_stable_coverage": complete,
        "sufficient_tensor_coverage": bool(pairs) and not insufficient,
        "insufficient_tensor_coverage": insufficient,
        "nonfinite_element_count": sum(pair["nonfinite_element_count"] for pair in pairs),
        "stable_stages_within_tolerance": complete and all(pair["within_tolerance"] for pair in pairs),
        "branch_records_not_compared": branches,
        "alignment_limitation": ALIGNMENT_LIMITATION,
    }


def _identity(info: os.stat_result) -> tuple:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink


@contextmanager
def _bound_file(path: Path, expected_sha256: str) -> Iterator[BinaryIO]:
    if (
        not path.is_absolute()
        or path != path.resolve(strict=True)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("Evidence requires a canonical path and exact SHA256")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        original = os.fstat(stream.fileno())
        if not stat.S_ISREG(original.st_mode) or original.st_nlink != 1:
            raise ValueError("Evidence must be a singly linked regular file")
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
            raise ValueError(f"Evidence digest differs: {path}")
        stream.seek(0)
        yield stream
        stream.seek(0)
        if (
            hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256
            or _identity(os.fstat(stream.fileno())) != _identity(original)
            or _identity(path.stat(follow_symlinks=False)) != _identity(original)
            or path != path.resolve(strict=True)
        ):
            raise ValueError(f"Evidence changed during stage analysis: {path}")


def _json(stream: BinaryIO) -> dict:
    raw = stream.read(JSON_BYTE_LIMIT + 1)
    if len(raw) > JSON_BYTE_LIMIT:
        raise ValueError("Evidence JSON exceeds the bounded reader limit")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Evidence JSON must be an object")
    return result


def _require_saved_gates(receipt: dict) -> None:
    analysis = receipt.get("analysis", {})
    if (
        receipt.get("status") != COMPLETE_STATUS
        or "error" in receipt
        or not isinstance(analysis, dict)
        or any(analysis.get(gate) is not True for gate in GATES)
        or receipt.get("tolerance") != MOMENT_TOLERANCE
    ):
        raise ValueError(
            "Completed diagnostic, both saved transparency predicates and exact tolerance precede trace reads"
        )
    executions = receipt.get("gpu_commands", [])
    commands = receipt.get("commands", [])
    identities = [
        (protocol, mode) for protocol in ("control", "trace") for mode in ("continuous", "interrupted", "resumed")
    ]
    if (
        len(executions) != len(identities)
        or len(commands) != len(identities)
        or any(
            (execution["protocol"], execution["mode"]) != identity
            or type(execution["exit_code"]) is not int
            or execution["exit_code"] != 0
            or execution["command"] != command
            or "trajectory" not in execution
            for execution, command, identity in zip(executions, commands, identities, strict=True)
        )
        or receipt.get("world") != WORLD
        or (receipt.get("boundary"), receipt.get("first_resumed"), receipt.get("endpoint")) != UPDATES
    ):
        raise ValueError("Saved diagnostic command completion is incomplete")


def _trace_path(root: Path, mode: str, rank: int, update: int) -> Path:
    return root / WORK_RELATIVE / f"trace_{mode}/continuation-trace/rank_{rank}/update_{update:06d}.pt"


def _trace_inventory(receipt: dict, root: Path, mode: str) -> dict:
    inventory = receipt["analysis"]["traces"][mode]["files_sha256"]
    paths = [_trace_path(root, mode, rank, update) for rank in range(WORLD) for update in UPDATES]
    expected = {str(path) for trace in paths for path in (trace, trace.with_suffix(".json"))}
    if inventory.keys() != expected:
        raise ValueError("Saved trace inventory differs from the bound all-rank/update paths")
    return inventory


def _load_trace(stack: ExitStack, path: Path, inventory: dict, *, rank: int, update: int, start_step: int) -> dict:
    receipt_path = path.with_suffix(".json")
    receipt = _json(stack.enter_context(_bound_file(receipt_path, inventory[str(receipt_path)])))
    if receipt["trace_sha256"] != inventory[str(path)]:
        raise ValueError("Trace sidecar digest differs from the diagnostic binding")
    stream = stack.enter_context(_bound_file(path, inventory[str(path)]))
    # mmap the authenticated open inode, not a pathname that could be replaced before torch.load.
    trace = torch.load(f"/proc/self/fd/{stream.fileno()}", map_location="cpu", weights_only=True, mmap=True)
    stages = [record["stage"] for record in trace["records"]]
    provenance = trace["provenance"]
    if (
        trace["schema"] != 1
        or trace["rank"] != rank
        or trace["update"] != update
        or receipt["rank"] != rank
        or receipt["update"] != update
        or receipt["status"] != "saved_continuation_intermediates_not_numerical_acceptance"
        or receipt["provenance"] != provenance
        or set(provenance) != {"identity", "start_step"}
        or provenance["start_step"] != start_step
        or receipt["stages"] != stages
        or receipt["records"] != len(stages)
    ):
        raise ValueError("Trace rank/update, stage list or provenance differs")
    return trace


def analyze_pair(receipt_path: Path, receipt_sha256: str, *, rank: int, update: int) -> dict:
    """Authenticate and inspect one saved trace pair without reconstructing its gates."""
    if type(rank) is not int or not 0 <= rank < WORLD or type(update) is not int or update not in UPDATES:
        raise ValueError("Select an original rank and traced update")
    source_paths = (
        Path(__file__).resolve(),
        Path(compare_saved.__code__.co_filename),
        Path(differences.__code__.co_filename),
    )
    sources = {str(path): digest(path) for path in source_paths}
    with ExitStack() as stack:
        receipt = _json(stack.enter_context(_bound_file(receipt_path, receipt_sha256)))
        _require_saved_gates(receipt)
        root = Path(receipt["root"])
        if not root.is_absolute() or root != root.resolve(strict=True) or receipt_path != root / RECEIPT_RELATIVE:
            raise ValueError("Diagnostic receipt is not at its canonical bound root")
        inventories = {mode: _trace_inventory(receipt, root, mode) for mode in MODES}
        traces, evidence = {}, {}
        for mode in MODES:
            path = _trace_path(root, mode, rank, update)
            traces[mode] = _load_trace(
                stack,
                path,
                inventories[mode],
                rank=rank,
                update=update,
                start_step=22 if mode == "resumed" and update > 22 else 0,
            )
            evidence[mode] = {
                "path": str(path),
                "sha256": inventories[mode][str(path)],
                "sidecar": str(path.with_suffix(".json")),
                "sidecar_sha256": inventories[mode][str(path.with_suffix(".json"))],
            }
        if traces["continuous"]["provenance"]["identity"] != traces["resumed"]["provenance"]["identity"]:
            raise ValueError("Selected trace pair has different source/config/data identities")
        result = {
            "schema": 1,
            "status": "saved_gate_dependent_stage_analysis_not_causal_proof",
            "rank": rank,
            "update": update,
            "saved_gate_dependency": {
                "receipt": str(receipt_path),
                "receipt_sha256": receipt_sha256,
                "status": receipt["status"],
                **{gate: receipt["analysis"][gate] for gate in GATES},
                "independently_reconstructed": False,
            },
            "trace_pair": evidence,
            "analyzer_sources_sha256": sources,
            "tolerance": MOMENT_TOLERANCE,
            "comparison": compare_stages(
                traces["resumed"]["records"], traces["continuous"]["records"], name=f"rank_{rank}.update_{update}"
            ),
            "limitation": GATE_LIMITATION,
        }
        if any(digest(Path(path)) != expected for path, expected in sources.items()):
            raise ValueError("Mutable analyzer sources changed during comparison")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True, help="Externally pinned completed diagnostic receipt SHA256.")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--update", type=int, choices=UPDATES, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(
        json.dumps(analyze_pair(args.receipt, args.receipt_sha256, rank=args.rank, update=args.update), allow_nan=False)
    )


if __name__ == "__main__":
    main()
