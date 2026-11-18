# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actual full-budget configuration and bounded trajectory contracts for Wave 11."""

import json
from pathlib import Path

ARMS = ("muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused")
NODE = "28.216.17.9"
MASK = "0,1,2,3,4,5,6,7"
WORLD = 8
BOUNDARY = 22
ENDPOINT = 24
PYTHON = "/opt/conda/envs/torch-base/bin/python"
ENVIRONMENT = {
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONFAULTHANDLER": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": MASK,
}
STATUS = "frozen_pending_completed_wave10_and_selected_lr_gpu_validation"
CONTROLLER_MARKER = "OUTPUT_MARIN_SELECTED_LR_VALIDATION_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_DIAGNOSTIC_ID"
PREPARATION = "selected-lr-validation-preparation.json"
GATES = (
    "actual_full_budget_config",
    "compiled_model",
    "cached_root_checkpoint",
    "boundary_continuation",
    "post_cached_root_continuation",
    "fixed_age_history",
    "all_rank_checkpoint_identity",
)
FUSED_GATES = ("captured_diagonal_rows", "outer_diagonal_update_parity")


def locations(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    preparation = root / "run-preparation"
    return (
        preparation / "wave11-selected-lr-validation-binding.json",
        root / "controllers/wave11-selected-lr-validation-20261117/tracking",
        preparation / "wave11-selected-lr-validation-state.json",
        root / "validation/wave11_selected_lr_full_budget",
        preparation / "validation-wave11-selected-lr-gpu.json",
    )


def controller_command(root: Path) -> list[str]:
    return [
        PYTHON,
        str(locations(root)[1] / "run_selected_lr_validation.py"),
        "--root",
        str(root),
        "--phase",
        "validate",
    ]


def validator_command(root: Path, candidate: Path) -> list[str]:
    _, helpers, _, work, receipt = locations(root)
    return [
        PYTHON,
        "-P",
        str(helpers / "validate_selected_lr_full_budget.py"),
        "--root",
        str(root),
        "--candidate",
        str(candidate),
        "--work",
        str(work),
        "--receipt",
        str(receipt),
    ]


def validation_commands(python: str, candidate: Path, root: Path, work: Path) -> list[list[str]]:
    """Retain each full schedule while limiting execution to matched 24-update trajectories."""
    commands = []
    for arm in ARMS:
        continuous, resumed = work / f"{arm}_continuous", work / f"{arm}_resumed"
        common = [
            python,
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
            "--execution-boundary",
            str(BOUNDARY),
        ]
        if arm == "outer_leftdiag_fused":
            common.extend(("--verify-diagonal-capture", "--verify-outer-diagonal-updates"))
        commands.extend(
            (
                [*common, "--out", str(continuous), "--max-steps", str(ENDPOINT)],
                [*common, "--out", str(resumed), "--max-steps", str(BOUNDARY)],
                [
                    *common,
                    "--out",
                    str(resumed),
                    "--max-steps",
                    str(ENDPOINT),
                    "--resume",
                    str(resumed / f"checkpoints/step_{BOUNDARY:06d}"),
                ],
            )
        )
    return commands


def require_config(config: dict, arm: str) -> None:
    training = config["training"]
    if (
        arm not in ARMS
        or config["experiment"]["arm"] != arm
        or training["steps"] != 5120
        or training["tokens"] != 2684354560
        or training["batch_tokens"] != 524288
        or training["world_size"] != WORLD
        or training["seed"] != 0
        or config["experiment"]["refresh_install_age"] != 1
        or config["runtime"]["compile"] == "eager"
        or config["runtime"]["attention"] != "FA2"
        or config["runtime"]["head"] != "compiled"
    ):
        raise ValueError("Selected-LR validation requires the actual full-budget compiled fixed-age configuration")


def plan(candidate: Path, root: Path, work: Path) -> dict:
    """Describe validation coverage without granting execution or changing candidate files."""
    if work.resolve() != locations(root.resolve())[3] or work.resolve().is_relative_to(candidate.resolve()):
        raise ValueError("Selected-LR validation requires its canonical external work directory")
    commands = validation_commands(PYTHON, candidate, root, work)
    trials = []
    for index, arm in enumerate(ARMS):
        path = candidate / f"experiment/configs/{arm}_0p5m_seed0.json"
        config = json.loads(path.read_text())
        require_config(config, arm)
        trials.append(
            {
                "name": arm,
                "family": arm,
                "config": str(path),
                "continuous": str(work / f"{arm}_continuous"),
                "resumed": str(work / f"{arm}_resumed"),
                "commands": commands[index * 3 : (index + 1) * 3],
                "checks": dict.fromkeys((*GATES, *FUSED_GATES) if arm == "outer_leftdiag_fused" else GATES, False),
            }
        )
    return {
        "status": "planned_not_executed",
        "root": str(root),
        "candidate": str(candidate),
        "environment": ENVIRONMENT,
        "world": WORLD,
        "continuous_steps": ENDPOINT,
        "interrupted_steps": BOUNDARY,
        "resumed_total_steps": ENDPOINT,
        "continuation_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "planned_validation_tokens": len(ARMS) * 2 * ENDPOINT * 524288,
        "trials": trials,
        "gpu_commands": [],
        "scope": (
            "Actual selected-LR 5120-update configurations retain the full schedule during bounded "
            "24 versus 22+resume-to24 checks. Original numerical source gates remain separately bound. "
            "No full-budget loss, long-horizon equivalence or efficiency result."
        ),
    }
