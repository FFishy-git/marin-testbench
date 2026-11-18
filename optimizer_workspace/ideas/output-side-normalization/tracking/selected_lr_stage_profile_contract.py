# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library-only execution contract for bounded selected-LR profiling."""

from pathlib import Path

ARMS = ("muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused")
NODE = "28.216.17.9"
WORLD = 8
BOUNDARY = 200
ENDPOINT = 220
PROFILE_UPDATES = (201, 220)
MODES = ("continuous", "off_resumed", "on_resumed")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
MASK = "0,1,2,3,4,5,6,7"
ENVIRONMENT = {
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONFAULTHANDLER": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": MASK,
}
PREPARATION = "selected-lr-stage-profile-validation-preparation.json"
PREPARED_STATUS = "selected_lr_stage_profile_validation_prepared_no_gpu_binding"
CPU_STATUS = "selected_lr_stage_profile_validation_cpu_verified_no_gpu_binding"
FROZEN_STATUS = "frozen_selected_lr_stage_profile_validation_pending_completed_wave12"
CONTROLLER_MARKER = "OUTPUT_MARIN_STAGE_PROFILE_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_DIAGNOSTIC_ID"
STATUS = "planned_selected_lr_stage_profile_validation_gpu_denied"


def locations(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        root / "run-preparation/selected-lr-stage-profile-validation-detached-binding-20261118.json",
        root / "controllers/selected-lr-stage-profile-validation-detached-20261118/tracking",
        root / "run-preparation/selected-lr-stage-profile-validation-detached-state-20261118.json",
        root / "validation/selected_lr_stage_profile_detached_20261118",
        root / "run-preparation/validation-selected-lr-stage-profile-detached-gpu-20261118.json",
    )


def controller_command(root: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "run_selected_lr_stage_profile.py"),
        "--root",
        str(root),
        "--phase",
        "validate",
    ]


def validator_command(root: Path, candidate: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "validate_selected_lr_stage_profile.py"),
        "--root",
        str(root),
        "--candidate",
        str(candidate),
        "--work",
        str(locations(root)[3]),
        "--receipt",
        str(locations(root)[4]),
    ]


def validation_commands(candidate: Path, root: Path, work: Path) -> list[list[str]]:
    commands = []
    for arm in ARMS:
        for mode in MODES:
            output = work / f"{arm}_{mode}"
            command = [
                PYTHON,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                "-m",
                "experiment.train",
                "--config",
                str(candidate / f"experiment/configs/{arm}_0p5m_seed0.json"),
                "--data",
                str(root / "data"),
                "--data-receipt",
                str(root / "data/experiment_receipt.json"),
                "--launch-release",
                str(locations(root)[0]),
                "--execution-boundary",
                str(BOUNDARY),
                "--out",
                str(output),
                "--max-steps",
                str(ENDPOINT),
            ]
            if mode != "continuous":
                command.extend(("--resume", str(output / f"checkpoints/step_{BOUNDARY:06d}")))
            if mode == "on_resumed":
                command.extend(("--profile-updates", *(str(update) for update in PROFILE_UPDATES)))
            commands.append(command)
    return commands
