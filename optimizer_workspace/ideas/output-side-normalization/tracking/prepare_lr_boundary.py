# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a separate, equal-budget expansion above the completed fixed-age LR grid."""

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path

from analyze_fixed_age_tuning import analyze as analyze_initial
from prepare_fixed_age_tuning import BATCH, require_source
from prepare_tuning import ATTEMPTS_PER_FAMILY, TRIAL_TOKENS, sha256
from prepare_tuning import prepare as prepare_trials
from wave7_release import ARMS, read

STUDY = "wave7_fixed_age_matrix_lr_upper_boundary"
MULTIPLIERS = (("lr_expand_2x", 2.0), ("lr_expand_2sqrt2x", 2 * math.sqrt(2)))
LAUNCH_CONDITION = (
    "Wait for Wave-8 complete histories, five-window analysis, full-resolution verified SwanLab "
    "import, marked-child cleanup and keeper restoration. Freeze separate boundary-study numerical "
    "validation/execution helpers and validate all eight actual configurations through update-22 "
    "and update-24 continuation and fused capture/update gates at unchanged tolerances. Acquire "
    "fresh verified allocations. No GPU job or queue is created by this preparation."
)


def require_parent(parent_path: Path) -> tuple[dict, dict, dict]:
    """Rederive the completed grid and check locally recorded full-resolution cloud provenance."""
    parent = read(parent_path)
    analysis_path = parent_path.parent / "analysis.json"
    upload_path = parent_path.parent / "swanlab-upload.json"
    saved, upload = read(analysis_path), read(upload_path)
    actual = analyze_initial(parent_path)
    if (
        actual["status"] != "complete"
        or {key: value for key, value in saved.items() if key != "time"}
        != {key: value for key, value in actual.items() if key != "time"}
        or upload["status"] != "verified"
        or upload["campaign_sha256"] != sha256(parent_path)
        or upload["execution_binding_sha256"] != sha256(parent_path.parent / "execution-binding.json")
        or set(upload["runs"]) != {trial["name"] for trial in parent["trials"]}
    ):
        raise ValueError("Initial LR grid is not completely analyzed and cloud-verified under its recorded identity")
    for trial in parent["trials"]:
        directory = parent_path.parent / "runs" / trial["name"]
        checkpoint = directory / f"checkpoints/step_{trial['steps']:06d}"
        entry = upload["runs"][trial["name"]]
        counts = entry["verification"]["point_counts"]
        if (
            entry["status"] != "verified"
            or entry["verification"]["state"] != "FINISHED"
            or entry["campaign_sha256"] != sha256(parent_path)
            or entry["execution_binding_sha256"] != upload["execution_binding_sha256"]
            or entry["trial_config_sha256"] != trial["config_sha256"]
            or entry["manifest_sha256"] != sha256(directory / "run.json")
            or entry["metrics_sha256"] != sha256(directory / "metrics_from_000000.jsonl")
            or entry["checkpoint_receipt_sha256"] != sha256(checkpoint / "complete.json")
            or entry["checkpoint_rank_bytes"] != [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)]
            or counts["tokens"] != 320
            or counts["train/loss"] != 320
            or counts["eval/selection_loss"] != 9
            or counts["eval/c4_loss"] != 9
        ):
            raise ValueError(f"Initial LR cloud receipt no longer matches complete local history: {trial['name']}")
    for family in ARMS:
        rows = sorted(
            (trial for trial in actual["trials"] if trial["family"] == family), key=lambda trial: trial["matrix_lr"]
        )
        if actual["completed_family_lr_choices"][family]["trial"] != f"{family}_lr_up" or not all(
            left["final_selection_loss"] > right["final_selection_loss"] for left, right in pairwise(rows)
        ):
            raise ValueError("Upper-boundary expansion requires monotonic held-out gains and upper-edge choices")
    return parent, actual, upload


def prepare(root: Path, parent_path: Path, out: Path) -> dict:
    """Write eight new trial identities without changing any existing campaign or frozen source."""
    parent_path = parent_path.resolve()
    root = root.resolve()
    if parent_path.parent.parent.resolve() != (root / "campaigns").resolve():
        raise ValueError("Boundary study must extend a campaign under the same output root")
    parent, analysis, upload = require_parent(parent_path)
    if Path(parent["root"]).resolve() != root:
        raise ValueError("Initial grid belongs to a different output root")
    require_source(root, Path(parent["source_receipt"]))
    result = prepare_trials(root, Path(parent["source_receipt"]), out, BATCH, families=ARMS, multipliers=MULTIPLIERS)
    result.update(
        study=STUDY,
        root=str(root),
        refresh_install_age=1,
        preparation_helper_sha256=sha256(Path(__file__)),
        original_gpu_validation_sha256=parent["original_gpu_validation_sha256"],
        original_validation_state_sha256=parent["original_validation_state_sha256"],
        parent_campaign=str(parent_path),
        parent_campaign_sha256=sha256(parent_path),
        parent_analysis_sha256=sha256(parent_path.parent / "analysis.json"),
        parent_upload_sha256=sha256(parent_path.parent / "swanlab-upload.json"),
        parent_completed_family_lr_choices=analysis["completed_family_lr_choices"],
        parent_cloud_verified_at=upload["verified_at"],
        new_grid=[{"label": label, "multiplier": multiplier} for label, multiplier in MULTIPLIERS],
        cumulative_trials_per_family=parent["initial_trials_per_family"] + len(MULTIPLIERS),
        cumulative_tokens_per_family=parent["initial_tokens_per_family"] + len(MULTIPLIERS) * TRIAL_TOKENS,
        max_attempts_per_family=ATTEMPTS_PER_FAMILY - parent["initial_trials_per_family"],
        max_tokens_per_family=(ATTEMPTS_PER_FAMILY - parent["initial_trials_per_family"]) * TRIAL_TOKENS,
        remaining_attempts_per_family=ATTEMPTS_PER_FAMILY - parent["initial_trials_per_family"] - len(MULTIPLIERS),
        launch_condition=LAUNCH_CONDITION,
    )
    result["selection"]["confirmation"] = (
        "Rank all five completed points per family by final held-out mixture loss. Freeze settings "
        "before independently selected full-budget paired fresh seeds; short trials are not confirmation."
    )
    (out / "campaign.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.root, args.parent, args.out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "study": result["study"],
                "trials": len(result["trials"]),
                "initial_tokens_per_family": result["initial_tokens_per_family"],
                "cumulative_tokens_per_family": result["cumulative_tokens_per_family"],
                "receipt": str(args.out / "campaign.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
