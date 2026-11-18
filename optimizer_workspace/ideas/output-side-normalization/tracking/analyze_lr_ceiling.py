# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rank seven equal-budget LR points only after the new per-family ceiling sweep completes."""

import argparse
import json
import math
from pathlib import Path

from analyze_tuning import analyze as analyze_trials
from compare_runs import history
from prepare_fixed_age_tuning import BATCH
from prepare_lr_boundary import require_parent as require_initial_parent
from prepare_lr_ceiling import LAUNCH_CONDITION, MULTIPLIERS, STUDY, require_parent, require_recovery
from prepare_tuning import ATTEMPTS_PER_FAMILY, TRIAL_TOKENS, sha256
from wave7_release import ARMS, read


def analyze(campaign_path: Path) -> dict:
    campaign = read(campaign_path)
    if "recovery" in campaign:
        require_recovery(campaign)
    parent_path = Path(campaign["parent_campaign"])
    parent, parent_analysis, _ = require_parent(parent_path)
    initial_path = Path(parent["parent_campaign"])
    _, initial_analysis, _ = require_initial_parent(initial_path)
    root = Path(parent["root"])
    if (
        campaign["study"] != STUDY
        or campaign["batch"] != BATCH
        or campaign["root"] != parent["root"]
        or campaign["source_receipt_sha256"] != parent["source_receipt_sha256"]
        or campaign["parent_campaign_sha256"] != sha256(parent_path)
        or campaign["parent_analysis_sha256"] != sha256(parent_path.parent / "analysis.json")
        or campaign["parent_upload_sha256"] != sha256(parent_path.parent / "swanlab-upload.json")
        or campaign["parent_completed_family_lr_choices"] != parent_analysis["completed_cumulative_family_lr_choices"]
        or type(campaign["refresh_install_age"]) is not int
        or campaign["refresh_install_age"] != 1
        or campaign["original_gpu_validation_sha256"] != parent["original_gpu_validation_sha256"]
        or campaign["original_validation_state_sha256"] != parent["original_validation_state_sha256"]
        or campaign["predecessor_wave"] != "wave9"
        or campaign["predecessor_release_sha256"] != sha256(root / "run-preparation/wave9-launch-binding.json")
        or campaign["new_grid"] != [{"label": label, "multiplier": multiplier} for label, multiplier in MULTIPLIERS]
        or campaign["cumulative_trials_per_family"] != 7
        or campaign["cumulative_tokens_per_family"] != 7 * TRIAL_TOKENS
        or campaign["max_attempts_per_family"] != ATTEMPTS_PER_FAMILY - 5
        or campaign["max_tokens_per_family"] != (ATTEMPTS_PER_FAMILY - 5) * TRIAL_TOKENS
        or campaign["remaining_attempts_per_family"] != ATTEMPTS_PER_FAMILY - 7
        or campaign["launch_condition"] != LAUNCH_CONDITION
    ):
        raise ValueError("LR ceiling study differs from its bound parents, Wave-9 predecessor or fixed protocol")
    result = analyze_trials(campaign_path, families=ARMS, multipliers=MULTIPLIERS)
    choices = {}
    for family in ARMS:
        new = [trial for trial in result["trials"] if trial["family"] == family]
        if not all(trial["complete"] for trial in new):
            continue
        previous = [
            (trial, path)
            for path, analysis in ((initial_path, initial_analysis), (parent_path, parent_analysis))
            for trial in analysis["trials"]
            if trial["family"] == family
        ]
        combined = [(trial, campaign_path) for trial in new] + previous
        best, path = min(combined, key=lambda item: (item[0]["final_selection_loss"], item[0]["name"]))
        choices[family] = {
            "trial": best["name"],
            "campaign": str(path),
            "matrix_lr": best["matrix_lr"],
            "final_selection_loss": best["final_selection_loss"],
            "upper_boundary_selected": best["matrix_lr"] == max(trial["matrix_lr"] for trial, _ in combined),
        }
    for trial in result["trials"]:
        trial["late_evaluation_tokens"] = {}
        if not trial["complete"]:
            continue
        run = history(campaign_path.parent / "runs" / trial["name"])
        for metric, target in (
            ("selection_loss", "mean_late_selection_loss"),
            ("c4_loss", "mean_late_c4_loss_reporting_only"),
        ):
            late = [row for row in run.rows if row["tokens"] > TRIAL_TOKENS // 2 and metric in row]
            trial["late_evaluation_tokens"][metric] = [row["tokens"] for row in late]
            trial[target] = math.fsum(row[metric] for row in late) / len(late) if late else None
    result.update(
        study=STUDY,
        parent_campaign_sha256=campaign["parent_campaign_sha256"],
        completed_cumulative_family_lr_choices=choices,
        cumulative_observed_tokens_by_family={
            family: (
                parent_analysis["cumulative_observed_tokens_by_family"][family]
                + result["observed_tokens_by_family"][family]
            )
            for family in ARMS
        },
        interpretation=(
            "Two new equal-budget seed-0 trials extend five completed points; failed prefixes remain charged. "
            "Only completed seven-point families get cumulative choices. C4 is reporting only, short-schedule "
            "LR selection does not establish a full-budget optimum, and independently tuned fresh-seed "
            "quality/efficiency confirmation remains required."
        ),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.campaign)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
