# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate actual-LR protocol; lower-LR screening grants no authority here."""

from pathlib import Path

import ns_budget_study as study
from ns_full_model_contract import BATCH_TOKENS as BATCH_TOKENS
from ns_full_model_contract import CHECKPOINTS as CHECKPOINTS
from ns_full_model_contract import ENVIRONMENT as ENVIRONMENT
from ns_full_model_contract import FAMILIES as FAMILIES
from ns_full_model_contract import MODES as MODES
from ns_full_model_contract import PYTHON as PYTHON
from ns_full_model_contract import TOLERANCE as TOLERANCE
from ns_full_model_contract import WORLD as WORLD

BOUNDARY, ENDPOINT = 22, 24
FROZEN_STATUS = "frozen_ns_actual_lr_full_model_validation_not_production"
CONTROLLER_MARKER = "OUTPUT_MARIN_NS_ACTUAL_LR_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_NS_ACTUAL_LR_VALIDATOR"
COMMAND_MARKER = "OUTPUT_MARIN_NS_ACTUAL_LR_COMMAND"
SCOPE = (
    "Fifty-four actual family/operand/budget/LR configurations and 162 bounded eight-rank commands. "
    "Preserve original5120schedule, continuous24/interrupted22/resumed24, all-rank boundary22/"
    "post-cached-root24/scalar-loss and fused/null captured-row/update/work gates, original tolerances. "
    "Update23 is retained diagnostic only. Lower-LR screening does not replace any per-LR gate. "
    "Distinct source-bound ownership and accepted original NS terminal/restoration are required. "
    "No production release, full-budget quality, whole-training efficiency or SwanLab acceptance."
)


def locations(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        root / "run-preparation/ns-actual-lr-binding-20261119.json",
        root / "controllers/ns-actual-lr-20261119/tracking",
        root / "run-preparation/ns-actual-lr-state-20261119.json",
        root / "validation/ns_actual_lr_20261119",
        root / "run-preparation/validation-ns-actual-lr-gpu-20261119.json",
    )


def specs() -> tuple[dict, ...]:
    return study.trial_specs()


def require_config(candidate: Path, spec: dict) -> dict:
    return study.actual_config(candidate, spec)


def commands(candidate: Path, root: Path, work: Path) -> list[list[str]]:
    if (
        any(path != path.resolve() or path.is_symlink() for path in (root, candidate, work))
        or work != locations(root)[3]
        or work.is_relative_to(candidate)
        or candidate.is_relative_to(work)
    ):
        raise ValueError("NS actual-LR validation requires canonical distinct external output")
    result = []
    for spec in specs():
        require_config(candidate, spec)
        continuous, resumed = (work / f"{spec['name']}_{mode}" for mode in ("continuous", "resumed"))
        common = [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "-m",
            "experiment.train",
            "--config",
            str(candidate / spec["config_name"]),
            "--data",
            str(root / "data"),
            "--data-receipt",
            str(root / "data/experiment_receipt.json"),
            "--launch-release",
            str(locations(root)[0]),
            "--execution-boundary",
            str(BOUNDARY),
        ]
        if spec["family"] != "muont":
            common.extend(("--verify-diagonal-capture", "--verify-outer-diagonal-updates"))
        result.extend(
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
    return result
