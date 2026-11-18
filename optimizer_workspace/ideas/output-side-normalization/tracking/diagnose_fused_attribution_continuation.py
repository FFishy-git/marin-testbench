# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quantify preserved attribution continuation on CPU without accepting a numerical or launch gate."""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch
from diagnose_continuation import SECTIONS, compare_saved, digest
from diagnose_fused_diagonal import MOMENT_TOLERANCE
from fused_attribution_validation_contract import BOUNDARY, ENDPOINT, WORLD, locations


def inspect_pair(actual_path: Path, expected_path: Path, rank: int, step: int) -> dict:
    """Compare saved block-arm tensors and nested root state at an exact continuation boundary."""
    original = {str(path): digest(path) for path in (actual_path, expected_path)}
    actual = torch.load(actual_path, map_location="cpu", weights_only=False, mmap=True)
    expected = torch.load(expected_path, map_location="cpu", weights_only=False, mmap=True)
    if (
        actual["identity"] != expected["identity"]
        or actual["config"] != expected["config"]
        or actual["next_global_sequence"] != expected["next_global_sequence"]
        or any(
            actual[name] != value or expected[name] != value
            for name, value in (("rank", rank), ("world", WORLD), ("step", step))
        )
    ):
        raise ValueError("Attribution checkpoint identity or requested rank/boundary differs")
    changed, scalars, count = [], [], 0
    for name in SECTIONS:
        if name not in expected:
            if name in actual:
                scalars.append({"name": name, "kind": "unexpected_section"})
        elif name not in actual:
            scalars.append({"name": name, "kind": "missing_section"})
        else:
            count += compare_saved(actual[name], expected[name], f"rank_{rank}.{name}", changed, scalars)
    groups = {}
    for side in ("input", "output"):
        reference_engine, saved_engine = expected["fast"][side], actual["fast"][side]
        if reference_engine is None:
            continue
        for name, reference in reference_engine["groups"].items():
            saved = saved_engine["groups"][name]
            groups[f"{side}.{name}"] = {
                "moment_exact": torch.equal(saved["C"], reference["C"]),
                "basis_step_actual": saved["basis_step"],
                "basis_step_expected": reference["basis_step"],
                "pending_steps_actual": [None if item is None else item["step"] for item in saved["pending"]],
                "pending_steps_expected": [None if item is None else item["step"] for item in reference["pending"]],
            }
    if any(digest(Path(path)) != value for path, value in original.items()):
        raise ValueError("Attribution diagnostic checkpoint changed while being read")
    return {
        "rank": rank,
        "step": step,
        "tensor_comparisons": count,
        "changed_tensor_count": len(changed),
        "failed_tensor_count": sum(not item["within_tolerance"] for item in changed),
        "failures_by_section": dict(
            Counter(item["name"].split(".")[1] for item in changed if not item["within_tolerance"])
        ),
        "changed_tensors": changed,
        "scalar_or_structure_differences": scalars,
        "root_groups": groups,
        "actual_checkpoint": str(actual_path),
        "actual_checkpoint_sha256": original[str(actual_path)],
        "expected_checkpoint": str(expected_path),
        "expected_checkpoint_sha256": original[str(expected_path)],
    }


def diagnose(root: Path) -> dict:
    binding_path, helpers, state_path, work, gpu_path = locations(root)
    frozen, state, gpu = (json.loads(path.read_bytes()) for path in (binding_path, state_path, gpu_path))
    candidate = Path(frozen["candidate"])
    if (
        state["status"] != "needs_attention"
        or gpu["status"] != "needs_attention"
        or state["keeper_restore"] != 0
        or state["binding_sha256"] != digest(binding_path)
        or gpu["execution_binding_sha256"] != digest(binding_path)
        or gpu["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or len(gpu["gpu_commands"]) != 6
        or "rank_0.model.body.emb.weight" not in gpu["error"]
    ):
        raise ValueError("Expected the original failed and restored six-command attribution attempt")
    evidence = {
        str(path): value
        for path, value in (
            (binding_path, digest(binding_path)),
            (state_path, digest(state_path)),
            (gpu_path, digest(gpu_path)),
            (Path(state["log"]), state["log_sha256"]),
            (Path(frozen["archive"]), frozen["archive_sha256"]),
            *((candidate / name, value) for name, value in frozen["files_sha256"].items()),
            *((helpers / name, value) for name, value in frozen["helpers_sha256"].items()),
            *((Path(item["log"]), item["log_sha256"]) for item in gpu["gpu_commands"]),
        )
    }
    for path, value in evidence.items():
        if digest(Path(path)) != value:
            raise ValueError("Original attribution source or failed execution evidence changed")
    trajectories = {}
    prefix = "outer_leftdiag_fused_full_lr_lower"
    for mode in ("continuous", "resumed"):
        directory = work / f"{prefix}_{mode}"
        manifest = json.loads((directory / "run.json").read_bytes())
        if manifest["completed_steps"] != ENDPOINT or manifest["status"] != "execution_check_complete":
            raise ValueError("Attribution diagnostic trajectory is not the completed original bounded execution")
        trajectories[mode] = directory
    pairs = []
    for step in (BOUNDARY, ENDPOINT):
        for rank in range(WORLD):
            filename = f"checkpoints/step_{step:06d}/rank_{rank}.pt"
            pair = inspect_pair(trajectories["resumed"] / filename, trajectories["continuous"] / filename, rank, step)
            pairs.append(pair)
            print(
                json.dumps({key: pair[key] for key in ("rank", "step", "tensor_comparisons", "failed_tensor_count")}),
                flush=True,
            )
    if any(digest(Path(path)) != value for path, value in evidence.items()):
        raise ValueError("Original attribution evidence changed during CPU diagnosis")
    return {
        "status": "cpu_saved_attribution_all_rank_diagnostic_not_launch_gate",
        "time": datetime.now(UTC).isoformat(),
        "source_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "diagnose_continuation.py",
                "diagnose_fused_diagonal.py",
                "fused_attribution_validation_contract.py",
            )
        },
        "evidence_sha256": evidence,
        "tolerance": MOMENT_TOLERANCE,
        "pairs": pairs,
        "scope": (
            "Read-only original eight-rank checkpoint22/24 comparison; "
            "no causal intervention, GPU acceptance or optimizer result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Attribution diagnostic needs an unused external receipt")
    torch.set_num_threads(8)
    start = time.monotonic()
    result = diagnose(args.root.resolve())
    result["diagnostic_seconds"] = time.monotonic() - start
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.out)


if __name__ == "__main__":
    main()
