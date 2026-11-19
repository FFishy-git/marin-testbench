# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actual-LR protocol under the original training boundary; no execution authority."""

from pathlib import Path

import ns_budget_study as study
from ns_training_path_contract import BATCH_TOKENS as BATCH_TOKENS
from ns_training_path_contract import BOUNDARY as BOUNDARY
from ns_training_path_contract import CHECKPOINTS as CHECKPOINTS
from ns_training_path_contract import ENDPOINT as ENDPOINT
from ns_training_path_contract import ENVIRONMENT as ENVIRONMENT
from ns_training_path_contract import MODES as MODES
from ns_training_path_contract import PYTHON as PYTHON
from ns_training_path_contract import TRAINING_FILES as TRAINING_FILES
from ns_training_path_contract import WORLD as WORLD

FROZEN_STATUS = "frozen_ns_training_path_actual_lr_validation_not_production"
CONTROLLER_MARKER = "OUTPUT_MARIN_NS_TRAINING_PATH_LR_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_NS_TRAINING_PATH_LR_VALIDATOR"
COMMAND_MARKER = "OUTPUT_MARIN_NS_TRAINING_PATH_LR_COMMAND"
TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
NUMERICAL_READERS = {
    "ns_full_model_contract.py": "d51190cd8e39fc66882b756ccf3013ff8ef9c89a9d8f160dcde6868f640721f7",
    "ns_full_model_validation.py": "2d80cf8b52318d0ba9e52ccb7beb2d77095fc4be15199a2cff4afb4ea7a6a9be",
    "ns_budget_study.py": "5db74d0a711325847890dcb772bab8c6cb5068ae80d8ffe6709655d939fa0a3e",
    "ns_lr_full_model_contract.py": "d87836d173b7a09a535c5c755fca028af716eb9a735b1219b4dd6b19b8fc6549",
    "ns_lr_full_model_validation.py": "1ca81d72e664a2cd43acc6fdaf78c1206bd44f08a7d5aba778da53e548386f1c",
}
SCOPE = (
    "Distinct original-training-path actual-LR correctness component: 54 family/operand/"
    "iteration/LR configurations, 162 continuous24/interrupted22/resumed24 commands. "
    "Original trainer and NS compiler boundary and accepted numerical readers are byte-pinned. "
    "Every actual LR independently requires boundary22/post-cached-root24, scalar-loss and "
    "fused/null captured-row/update/work evidence at unchanged tolerances. Update23 remains "
    "diagnostic. Preserve original5120schedule, compilation, cached-root CUDA graphs and all "
    "rejections. The original complete training-path diagnostic, canonical independent terminal, "
    "owner/waiter/descendant absence and exact restoration remain future ownership gates. "
    "No predecessor or execution owner is issued here; no GPU allocation, production release, "
    "quality, whole-training efficiency, private SwanLab or optimizer-superiority acceptance."
)


def specs() -> tuple[dict, ...]:
    return study.trial_specs()


def require_config(candidate: Path, spec: dict) -> dict:
    return study.actual_config(candidate, spec)


def locations(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        root / "run-preparation/ns-training-path-actual-lr-binding-20261119.json",
        root / "controllers/ns-training-path-actual-lr-20261119/tracking",
        root / "run-preparation/ns-training-path-actual-lr-state-20261119.json",
        root / "validation/ns_training_path_actual_lr_20261119",
        root / "run-preparation/validation-ns-training-path-actual-lr-gpu-20261119.json",
    )


def commands(candidate: Path, root: Path, work: Path) -> list[list[str]]:
    """Build per-configuration commands in the distinct one-shot output namespace."""
    if (
        any(path != path.resolve() or path.is_symlink() for path in (root, candidate, work))
        or work != locations(root)[3]
        or work.is_relative_to(candidate)
        or candidate.is_relative_to(work)
    ):
        raise ValueError("Training-path actual-LR commands require canonical distinct external output")
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
