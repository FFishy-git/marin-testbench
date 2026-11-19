# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact actual-config NS-budget validation protocol; no execution ownership."""

import copy
import json
from pathlib import Path

from fused_attribution_validation_contract import (
    BATCH_TOKENS as BATCH_TOKENS,
)
from fused_attribution_validation_contract import (
    BOUNDARY,
    ENDPOINT,
    FAMILIES,
    LR_GRID,
    PYTHON,
)
from fused_attribution_validation_contract import (
    ENVIRONMENT as ENVIRONMENT,
)
from fused_attribution_validation_contract import (
    WORLD as WORLD,
)
from fused_attribution_validation_contract import require_config as require_reference
from fused_attribution_validation_contract import trial_specs as reference_specs

BUDGETS = (5, 7, 10)
OPERANDS = ("tpu", "f32")
MODES = ("continuous", "interrupted", "resumed")
CHECKPOINTS = (BOUNDARY, 23, ENDPOINT)
TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
FROZEN_STATUS = "frozen_ns_full_model_validation_not_production"
CONTROLLER_MARKER = "OUTPUT_MARIN_NS_FULL_MODEL_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_NS_FULL_MODEL_VALIDATOR"
COMMAND_MARKER = "OUTPUT_MARIN_NS_FULL_MODEL_COMMAND"
SCOPE = (
    "Eighteen lower-LR actual-config finite-map cases, unchanged full 5120-update schedule. "
    "Continuous24 and independent interrupted22/resumed24, complete all-rank checkpoints "
    "22/23/24, original tolerances and fused/null captured-row/update/work gates. "
    "Update23 is diagnostic, including process-local graph rewarming; acceptance retains "
    "the original boundary22/post-cached-root24 and scalar-loss gates. "
    "Bounded loss is continuation evidence, not optimizer quality. No production release, "
    "whole-training efficiency or SwanLab acceptance."
)


def locations(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        root / "run-preparation/ns-full-model-binding-20261119.json",
        root / "controllers/ns-full-model-20261119/tracking",
        root / "run-preparation/ns-full-model-state-20261119.json",
        root / "validation/ns_full_model_20261119",
        root / "run-preparation/validation-ns-full-model-gpu-20261119.json",
    )


def specs() -> tuple[dict, ...]:
    return tuple(
        {
            "name": f"{family}_ns_{operand}_q{budget}_full_lr_lower",
            "family": family,
            "operand": operand,
            "budget": budget,
            "config_name": f"experiment/configs/{family}_ns_{operand}_q{budget}_full_lr_lower_0p5m_seed0.json",
            "matrix_lr": LR_GRID[0][1],
        }
        for family in FAMILIES
        for operand in OPERANDS
        for budget in BUDGETS
    )


def require_config(candidate: Path, spec: dict) -> dict:
    if spec not in specs():
        raise ValueError("NS full-model case is outside the exact lower-LR diagnostic protocol")
    paths = (
        candidate / f"experiment/configs/{spec['family']}_full_lr_lower_0p5m_seed0.json",
        candidate / spec["config_name"],
    )
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("NS full-model reference or actual configuration is absent or aliased")
    reference, actual = (json.loads(path.read_bytes()) for path in paths)
    original = next(
        item for item in reference_specs() if item["family"] == spec["family"] and item["lr_label"] == "lower"
    )
    require_reference(reference, original)
    expected = copy.deepcopy(reference)
    expected["optimizer"].update(ns_steps=spec["budget"], ns_mode=spec["operand"])
    if (
        actual != expected
        or type(actual["optimizer"]["ns_steps"]) is not int
        or actual["optimizer"]["lr"] != spec["matrix_lr"]
    ):
        raise ValueError("NS full-model actual configuration changed outside its finite-map budget/operand mode")
    return actual


def commands(candidate: Path, root: Path, work: Path) -> list[list[str]]:
    if (
        any(path != path.resolve() or path.is_symlink() for path in (candidate, root, work))
        or work != locations(root)[3]
        or work.is_relative_to(candidate)
        or candidate.is_relative_to(work)
    ):
        raise ValueError("NS full-model validation requires canonical distinct external output")
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
