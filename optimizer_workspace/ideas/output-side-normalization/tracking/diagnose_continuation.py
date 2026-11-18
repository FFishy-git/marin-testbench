# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect preserved wave-6 reference checkpoints on CPU without approving a launch."""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from diagnose_fused_diagonal import MOMENT_TOLERANCE, differences

SECTIONS = ("model", "optimizers", "fast", "output", "torch_rng", "cuda_rng", "python_rng")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_saved(actual, expected, name: str, changed: list, scalars: list) -> int:
    """Count tensor comparisons while retaining every changed tensor and structural mismatch."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            scalars.append({"name": name, "kind": "tensor_structure"})
            return 0
        if not torch.equal(actual, expected):
            changed.append(
                {
                    "name": name,
                    "shape": list(expected.shape),
                    "dtype": str(expected.dtype),
                    "changed_elements": (actual != expected).sum().item(),
                    **differences(actual, expected, MOMENT_TOLERANCE),
                }
            )
        return 1
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            scalars.append({"name": name, "kind": "mapping_structure"})
            return 0
        return sum(
            compare_saved(actual[key], value, f"{name}.{key}", changed, scalars) for key, value in expected.items()
        )
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            scalars.append({"name": name, "kind": "sequence_structure"})
            return 0
        return sum(
            compare_saved(value, reference, f"{name}.{index}", changed, scalars)
            for index, (value, reference) in enumerate(zip(actual, expected, strict=True))
        )
    if type(actual) is not type(expected):
        scalars.append({"name": name, "kind": "value_structure"})
        return 0
    if actual != expected:
        scalars.append({"name": name, "kind": "scalar", "actual": actual, "expected": expected})
    return 0


def inspect_pair(actual_path: Path, expected_path: Path, rank: int, step: int) -> dict:
    actual = torch.load(actual_path, map_location="cpu", weights_only=False, mmap=True)
    expected = torch.load(expected_path, map_location="cpu", weights_only=False, mmap=True)
    if actual["identity"] != expected["identity"] or actual["config"] != expected["config"]:
        raise ValueError("Checkpoint source/configuration/data identity differs")
    for name, value in (("rank", rank), ("world", 8), ("step", step)):
        if actual[name] != value or expected[name] != value:
            raise ValueError(f"Checkpoint {name} differs from the requested pair")
    if actual["next_global_sequence"] != expected["next_global_sequence"]:
        raise ValueError("Checkpoint data positions differ")
    changed, scalars = [], []
    count = 0
    for name in SECTIONS:
        if name not in expected:
            if name in actual:
                scalars.append({"name": name, "kind": "unexpected_section"})
            continue
        if name not in actual:
            scalars.append({"name": name, "kind": "missing_section"})
            continue
        count += compare_saved(actual[name], expected[name], f"rank_{rank}.{name}", changed, scalars)
    groups = {}
    for name, reference in expected["fast"]["groups"].items():
        saved = actual["fast"]["groups"][name]
        groups[name] = {
            "moment_exact": torch.equal(saved["C"], reference["C"]),
            "basis_step_actual": saved["basis_step"],
            "basis_step_expected": reference["basis_step"],
            "pending_steps_actual": [None if pending is None else pending["step"] for pending in saved["pending"]],
            "pending_steps_expected": [None if pending is None else pending["step"] for pending in reference["pending"]],
        }
    return {
        "rank": rank,
        "step": step,
        "tensor_comparisons": count,
        "changed_tensor_count": len(changed),
        "failed_tensor_count": sum(not row["within_tolerance"] for row in changed),
        "changed_tensors": changed,
        "scalar_or_structure_differences": scalars,
        "root_groups": groups,
        "actual_checkpoint": str(actual_path),
        "actual_checkpoint_sha256": digest(actual_path),
        "expected_checkpoint": str(expected_path),
        "expected_checkpoint_sha256": digest(expected_path),
    }


def diagnose(root: Path) -> dict:
    """Verify the preserved release evidence and inspect every saved reference rank at both boundaries."""
    preparation = root / "run-preparation"
    frozen = json.loads((preparation / "wave6-candidate.json").read_text())
    candidate = Path(frozen["candidate"])
    helper = Path(frozen["helpers"])
    if any(digest(candidate / name) != value for name, value in frozen["files_sha256"].items()):
        raise ValueError("Frozen candidate inventory changed")
    if any(digest(helper / name) != value for name, value in frozen["helpers_sha256"].items()):
        raise ValueError("Frozen controller inventory changed")
    if digest(Path(frozen["archive"])) != frozen["archive_sha256"]:
        raise ValueError("Frozen source archive changed")
    state = json.loads((preparation / "wave6-validation-state.json").read_text())
    if state["status"] != "needs_attention" or state["exit_code"] != 1 or state["keeper_restore"] != 0:
        raise ValueError("Expected the preserved failed/restored validation")
    work = root / "validation/wave6_outer_diagonal"
    for directory in ("muont_continuous", "muont_resumed"):
        manifest = json.loads((work / directory / "run.json").read_text())
        config = json.loads((candidate / "experiment/configs/muont_0p5m_seed0.json").read_text())
        if (
            manifest["status"] != "execution_check_complete"
            or manifest["completed_steps"] != 24
            or manifest["config"] != config
            or manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
        ):
            raise ValueError("Reference trajectory manifest differs")
    pairs = []
    for step in (22, 24):
        for rank in range(8):
            filename = f"checkpoints/step_{step:06d}/rank_{rank}.pt"
            pairs.append(
                inspect_pair(work / "muont_resumed" / filename, work / "muont_continuous" / filename, rank, step)
            )
    return {
        "status": "cpu_checkpoint_diagnostic_not_launch_gate",
        "time": datetime.now(UTC).isoformat(),
        "source_sha256": {
            Path(__file__).name: digest(Path(__file__)),
            "diagnose_fused_diagonal.py": digest(Path(differences.__code__.co_filename)),
        },
        "validation_receipts_sha256": {
            name: digest(preparation / name)
            for name in ("wave6-candidate.json", "wave6-validation-state.json", "validation-wave6-gpu.json")
        },
        "tolerance": MOMENT_TOLERANCE,
        "pairs": pairs,
        "limitation": "Read-only saved-state diagnosis, not a new GPU gate, causal intervention or training result.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Diagnostic evidence already exists")
    started = time.monotonic()
    torch.set_num_threads(8)
    record = diagnose(args.root)
    record["diagnostic_seconds"] = time.monotonic() - started
    with args.out.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.out)


if __name__ == "__main__":
    main()
