# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare two equal-budget LR points above the completed five-point boundary study."""

import argparse
import json
import math
from pathlib import Path

from analyze_lr_boundary import analyze as analyze_boundary
from prepare_fixed_age_tuning import BATCH, require_source
from prepare_lr_boundary import STUDY as PARENT_STUDY
from prepare_tuning import ATTEMPTS_PER_FAMILY, TRIAL_TOKENS, sha256
from prepare_tuning import prepare as prepare_trials
from wave7_release import ARMS, read

STUDY = "wave7_fixed_age_matrix_lr_ceiling"
MULTIPLIERS = (("lr_ceiling_4x", 4.0), ("lr_ceiling_4sqrt2x", 4 * math.sqrt(2)))
GPU_VISIBILITY = "0,1,2,3,4,5,6,7"
LAUNCH_CONDITION = (
    "Wait for all four Wave-9 full-budget histories, five complete shared broad-window analyses, "
    "full-resolution private SwanLab verification, fresh original-worker absence, marked-child cleanup "
    "and keeper restoration. Freeze separate ceiling-study validation/execution helpers; validate all "
    "eight actual LR configurations through update-22/update-24 continuation and fused capture/update "
    "gates at unchanged tolerances before acquiring fresh original-node allocations. Preparation "
    "creates no GPU job or queue; no candidate or completed campaign is modified."
)


def require_parent(parent_path: Path) -> tuple[dict, dict, dict]:
    """Rederive the completed expansion and its exact full-resolution cloud provenance."""
    parent = read(parent_path)
    if parent["study"] != PARENT_STUDY:
        raise ValueError("LR ceiling study requires the completed five-point upper-boundary protocol")
    analysis_path = parent_path.parent / "analysis.json"
    upload_path = parent_path.parent / "swanlab-upload.json"
    saved, upload = read(analysis_path), read(upload_path)
    actual = analyze_boundary(parent_path)
    if (
        actual["status"] != "complete"
        or {key: value for key, value in saved.items() if key != "time"}
        != {key: value for key, value in actual.items() if key != "time"}
        or upload["status"] != "verified"
        or upload["campaign_sha256"] != sha256(parent_path)
        or upload["execution_binding_sha256"] != sha256(parent_path.parent / "execution-binding.json")
        or set(upload["runs"]) != {trial["name"] for trial in parent["trials"]}
        or parent["cumulative_trials_per_family"] != 5
        or parent["remaining_attempts_per_family"] != ATTEMPTS_PER_FAMILY - 5
    ):
        raise ValueError("Previous LR expansion is incomplete or differs from its analyzed/cloud-verified identity")
    for trial in parent["trials"]:
        directory = parent_path.parent / "runs" / trial["name"]
        checkpoint = directory / f"checkpoints/step_{trial['steps']:06d}"
        entry = upload["runs"][trial["name"]]
        verification = entry["verification"]
        if (
            entry["status"] != "verified"
            or verification["state"] != "FINISHED"
            or entry["campaign_sha256"] != sha256(parent_path)
            or entry["execution_binding_sha256"] != upload["execution_binding_sha256"]
            or entry["trial_config_sha256"] != trial["config_sha256"]
            or entry["manifest_sha256"] != sha256(directory / "run.json")
            or entry["metrics_sha256"] != sha256(directory / "metrics_from_000000.jsonl")
            or entry["checkpoint_receipt_sha256"] != sha256(checkpoint / "complete.json")
            or entry["checkpoint_rank_bytes"] != [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)]
            or any(
                verification["point_counts"][key] != count
                for key, count in (
                    ("tokens", 320),
                    ("train/loss", 320),
                    ("eval/selection_loss", 9),
                    ("eval/c4_loss", 9),
                )
            )
        ):
            raise ValueError(f"Previous LR expansion cloud receipt no longer matches local history: {trial['name']}")
    for family in ARMS:
        rows = sorted(
            (trial for trial in actual["trials"] if trial["family"] == family), key=lambda trial: trial["matrix_lr"]
        )
        choice = actual["completed_cumulative_family_lr_choices"][family]
        if (
            len(rows) != 2
            or choice["upper_boundary_selected"] is not True
            or choice["trial"] != rows[-1]["name"]
            or rows[-1]["final_selection_loss"] >= rows[0]["final_selection_loss"]
        ):
            raise ValueError("LR ceiling expansion requires unresolved upper-edge choices and improving endpoints")
    return parent, actual, upload


def recovery_evidence(failed_path: Path, observation_path: Path) -> dict:
    """Bind the zero-update CUDA-mask failure without treating it as an LR trial."""
    failed_path, observation_path = failed_path.resolve(), observation_path.resolve()
    failed, observation = read(failed_path), read(observation_path)
    directory = failed_path.parent
    state = read(directory / "validation-state.json")
    gpu = read(directory / "validation-gpu.json")
    binding = read(directory / "validation-binding.json")
    command = gpu["gpu_commands"][0] if gpu["gpu_commands"] else {}
    log = directory / "validation/muont_lr_ceiling_4x_continuous.log"
    required_files = {
        "campaign.json",
        "validation-binding.json",
        "execution-binding.json",
        "validation-state.json",
        "validation-gpu.json",
        "queue/binding.json",
        "queue/installation.json",
        "queue/validate.json",
        "queue/state.json",
        "queue/stopped_at",
        "validation/muont_lr_ceiling_4x_continuous.log",
    }
    if (
        failed["study"] != STUDY
        or "recovery" in failed
        or observation["status"] != "terminal_zero_update_cuda_mask_failure_keeper_verified"
        or observation["campaign"] != str(failed_path)
        or observation["queue_process"] != "absent"
        or observation["controller_process"] != "exited"
        or observation["marked_children"] != {}
        or set(observation["files_sha256"]) != required_files
        or any(sha256(directory / name) != digest for name, digest in observation["files_sha256"].items())
        or read(directory / "queue/state.json")["status"] != "needs_attention"
        or state["status"] != "needs_attention"
        or state["exit_code"] != 1
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or state["binding_sha256"] != sha256(directory / "validation-binding.json")
        or gpu["status"] != "needs_attention"
        or gpu["campaign_sha256"] != sha256(failed_path)
        or len(gpu["gpu_commands"]) != 1
        or len(gpu["trials"]) != 8
        or any(value is not False for trial in gpu["trials"] for value in trial["checks"].values())
        or command.get("command") != binding["plan"]["trials"][0]["commands"][0]
        or command.get("trial") != "muont_lr_ceiling_4x"
        or command.get("exit_code") != 1
        or command.get("log") != str(log)
        or command.get("log_sha256") != sha256(log)
        or log.read_text().count("RuntimeError: No CUDA GPUs are available") != 8
        or any(
            (directory / name).exists()
            for name in ("launch-state.json", "workers", "runs", "monitor", "queue/launch.json")
        )
        or any((directory / "validation").rglob("run.json"))
        or any((directory / "validation").rglob("metrics_from_*.jsonl"))
        or any((directory / "validation").rglob("rank_*.pt"))
    ):
        raise ValueError("Ceiling recovery requires the preserved terminal, zero-update CUDA-mask failure")
    return {
        "failed_campaign": str(failed_path),
        "failed_campaign_sha256": sha256(failed_path),
        "terminal_observation": str(observation_path),
        "terminal_observation_sha256": sha256(observation_path),
        "failed_files_sha256": observation["files_sha256"],
        "cuda_visible_devices": GPU_VISIBILITY,
        "failed_validation_commands": 1,
        "failed_validation_updates": 0,
        "failed_validation_tokens": 0,
        "failed_tuning_attempts_by_family": dict.fromkeys(ARMS, 0),
        "failed_tuning_tokens_by_family": dict.fromkeys(ARMS, 0),
        "policy": (
            "Distinct execution of the same eight configs after a pre-update environment failure. "
            "Preserve failed intents/logs; add no search points or allowances. All 24 actual-config "
            "commands, fresh original-node inspection, cleanup and restoration remain required."
        ),
    }


def require_recovery(campaign: dict) -> None:
    """Reject changed failure provenance, trial bytes or recovery environment."""
    recovery = campaign["recovery"]
    failed_path = Path(recovery["failed_campaign"])
    if recovery != recovery_evidence(failed_path, Path(recovery["terminal_observation"])):
        raise ValueError("Ceiling recovery failure evidence or accounting changed")
    failed = read(failed_path)
    for key in (
        "study",
        "root",
        "source",
        "source_receipt_sha256",
        "data_receipt_sha256",
        "parent_campaign_sha256",
        "new_grid",
        "max_attempts_per_family",
        "max_tokens_per_family",
        "remaining_attempts_per_family",
    ):
        if campaign[key] != failed[key]:
            raise ValueError("Ceiling recovery must preserve the original source, grid and allowances")
    if len(campaign["trials"]) != len(failed["trials"]) or any(
        {key: value for key, value in actual.items() if key != "config"}
        != {key: value for key, value in original.items() if key != "config"}
        or sha256(Path(actual["config"])) != original["config_sha256"]
        for actual, original in zip(campaign["trials"], failed["trials"], strict=True)
    ):
        raise ValueError("Ceiling recovery actual trial configurations changed")


def prepare(
    root: Path,
    parent_path: Path,
    out: Path,
    *,
    failed_campaign: Path | None = None,
    terminal_observation: Path | None = None,
) -> dict:
    """Prepare eight new trial identities while leaving all executed sources unchanged."""
    root, parent_path, out = root.resolve(), parent_path.resolve(), out.resolve()
    if parent_path.parent.parent != root / "campaigns" or out.parent != root / "campaigns":
        raise ValueError("LR ceiling campaign and parent must remain under the same output root")
    parent, analysis, upload = require_parent(parent_path)
    if Path(parent["root"]).resolve() != root:
        raise ValueError("Previous LR expansion belongs to another output root")
    require_source(root, Path(parent["source_receipt"]))
    predecessor = root / "run-preparation/wave9-launch-binding.json"
    release = read(predecessor)
    if release["status"] != "frozen_validated_launch_release" or release["root"] != str(root):
        raise ValueError("Ceiling preparation requires the separately identified Wave-9 predecessor release")
    recovery = None
    if failed_campaign is not None or terminal_observation is not None:
        if failed_campaign is None or terminal_observation is None:
            raise ValueError("Recovery requires both failed campaign and terminal observation")
        recovery = recovery_evidence(failed_campaign, terminal_observation)
        failed = read(failed_campaign)
        if (
            failed_campaign.resolve().parent.parent != root / "campaigns"
            or failed["root"] != str(root)
            or failed["parent_campaign"] != str(parent_path)
        ):
            raise ValueError("Recovery must preserve the original ceiling parent and output root")
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
        parent_completed_family_lr_choices=analysis["completed_cumulative_family_lr_choices"],
        parent_cloud_verified_at=upload["verified_at"],
        predecessor_wave="wave9",
        predecessor_release_sha256=sha256(predecessor),
        new_grid=[{"label": label, "multiplier": multiplier} for label, multiplier in MULTIPLIERS],
        cumulative_trials_per_family=7,
        cumulative_tokens_per_family=7 * TRIAL_TOKENS,
        max_attempts_per_family=ATTEMPTS_PER_FAMILY - 5,
        max_tokens_per_family=(ATTEMPTS_PER_FAMILY - 5) * TRIAL_TOKENS,
        remaining_attempts_per_family=ATTEMPTS_PER_FAMILY - 7,
        launch_condition=LAUNCH_CONDITION,
    )
    result["selection"]["confirmation"] = (
        "Rank seven completed equal-budget LR points per family using final reserved-mixture selection loss. "
        "A remaining upper-edge winner is unresolved. Keep two remaining refinements per family, then "
        "independently validate full-budget LR settings and freeze fresh paired-seed confirmation."
    )
    if recovery is not None:
        result["recovery"] = recovery
        require_recovery(result)
    (out / "campaign.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--failed-campaign", type=Path)
    parser.add_argument("--terminal-observation", type=Path)
    args = parser.parse_args()
    result = prepare(
        args.root,
        args.parent,
        args.out,
        failed_campaign=args.failed_campaign,
        terminal_observation=args.terminal_observation,
    )
    print(json.dumps({"status": result["status"], "study": STUDY, "trials": len(result["trials"])}))


if __name__ == "__main__":
    main()
