# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a separate equal-budget LR study of the four fixed-age Wave-7 families."""

import argparse
import json
from pathlib import Path

from prepare_tuning import prepare as prepare_trials
from prepare_tuning import sha256
from wave7_release import ARMS, GATES, candidate_inventory, read

STUDY = "wave7_fixed_age_matrix_lr"
BATCH = "0p5m"
LAUNCH_CONDITION = (
    "Wait for all four Wave-7 full-budget histories, five-window analysis, verified SwanLab import "
    "and keeper restoration. Review fixed-age numerical/loss/systems evidence; validate every actual "
    "short-schedule arm/LR configuration, freeze separate launch helpers and acquire fresh allocations. "
    "This preparation launches no GPU job and installs no queue."
)


def require_source(root: Path, source_receipt: Path) -> dict:
    """Check the immutable numerical evidence without inspecting or changing active allocations."""
    preparation = root / "run-preparation"
    if source_receipt.resolve() != (preparation / "wave7-release-binding.json").resolve():
        raise ValueError("Fixed-age tuning must use the canonical Wave-7 logging release")
    frozen = read(source_receipt)
    candidate = Path(frozen["candidate"])
    original_path = preparation / "wave7-candidate.json"
    gpu_path = preparation / "validation-wave7-gpu.json"
    state_path = preparation / "wave7-validation-state.json"
    original = read(original_path)
    gpu = read(gpu_path)
    state = read(state_path)
    if (
        frozen["status"] != "frozen_validated_logging_release"
        or tuple(frozen["arms"]) != ARMS
        or tuple(frozen["required_gates"]) != GATES
        or type(frozen["refresh_install_age"]) is not int
        or frozen["refresh_install_age"] != 1
        or frozen["original_validation_binding_sha256"] != sha256(original_path)
        or frozen["original_gpu_validation_sha256"] != sha256(gpu_path)
        or frozen["original_validation_state_sha256"] != sha256(state_path)
        or frozen["files_sha256"] != original["files_sha256"]
        or candidate.resolve() != Path(original["candidate"]).resolve()
        or candidate_inventory(candidate) != frozen["files_sha256"]
        or state["status"] != "passed"
        or type(state.get("keeper_restore")) is not int
        or state["keeper_restore"] != 0
        or state["validation_sha256"] != sha256(gpu_path)
        or gpu["source_sha256"] != frozen["source_sha256"]
        or not all(gpu["checks"][gate] is True for gate in GATES)
    ):
        raise ValueError("Wave-7 fixed-age source or successful numerical evidence changed")
    for record in (original, frozen):
        if sha256(Path(record["archive"])) != record["archive_sha256"] or any(
            sha256(Path(record["helpers"]) / name) != digest for name, digest in record["helpers_sha256"].items()
        ):
            raise ValueError("Wave-7 archived source or bound helpers changed")
    for arm in ARMS:
        config = read(candidate / f"experiment/configs/{arm}_{BATCH}_seed0.json")
        if (
            type(config["experiment"]["refresh_install_age"]) is not int
            or config["experiment"]["refresh_install_age"] != 1
        ):
            raise ValueError("All fixed-age tuning references must retain installation age 1")
    return frozen


def prepare(root: Path, source_receipt: Path, out: Path) -> dict:
    """Write twelve externally bound trials while preserving the running candidate."""
    frozen = require_source(root, source_receipt)
    result = prepare_trials(root, source_receipt, out, BATCH, families=ARMS)
    result.update(
        study=STUDY,
        root=str(root.resolve()),
        preparation_helper_sha256=sha256(Path(__file__)),
        original_gpu_validation_sha256=frozen["original_gpu_validation_sha256"],
        original_validation_state_sha256=frozen["original_validation_state_sha256"],
        refresh_install_age=1,
        launch_condition=LAUNCH_CONDITION,
    )
    path = out / "campaign.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.root, args.root / "run-preparation/wave7-release-binding.json", args.out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "study": result["study"],
                "families": result["families"],
                "trials": len(result["trials"]),
                "initial_tokens_per_family": result["initial_tokens_per_family"],
                "receipt": str(args.out / "campaign.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
