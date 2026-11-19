# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distinct original-training-path NS diagnostic protocol; no execution authority."""

from pathlib import Path

from ns_full_model_contract import BATCH_TOKENS as BATCH_TOKENS
from ns_full_model_contract import BOUNDARY as BOUNDARY
from ns_full_model_contract import CHECKPOINTS as CHECKPOINTS
from ns_full_model_contract import ENDPOINT as ENDPOINT
from ns_full_model_contract import ENVIRONMENT as ENVIRONMENT
from ns_full_model_contract import MODES as MODES
from ns_full_model_contract import PYTHON as PYTHON
from ns_full_model_contract import TOLERANCE as TOLERANCE
from ns_full_model_contract import WORLD as WORLD
from ns_full_model_contract import require_config as require_config
from ns_full_model_contract import specs as specs

FROZEN_STATUS = "frozen_ns_training_path_diagnostic_not_production"
CONTROLLER_MARKER = "OUTPUT_MARIN_NS_TRAINING_PATH_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_NS_TRAINING_PATH_VALIDATOR"
COMMAND_MARKER = "OUTPUT_MARIN_NS_TRAINING_PATH_COMMAND"
MUON_ROOT = "artifacts/pinned-sources-materialized/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/src/toner/fast"
TRAINING_FILES = {
    "experiment/train.py": "7df444033a0630a22ab99708cc1971ee3feb770c8c1154b96794f92cc1576980",
    "experiment/optimizer.py": "a5ec7893960b08e729da6f9fea4825d4af625685c786ee9ad1d4c6729647a635",
    "experiment/block_toner.py": "9ff7a2b2690736ec94d8f647a735476ad90fb56b7d29296143621a3dd252142c",
    "experiment/refresh_toner.py": "3d3ef4f01f206ef29fbb3a40847764711d271e197e525edec76ef17eb563acf5",
    f"{MUON_ROOT}/muon.py": "b03f10c80ce46c61d4db788dda481bc8407273e809b252bd99132c49d665bb42",
    f"{MUON_ROOT}/engine.py": "1887e09312bb7276a2e3057cf9a41c644c353aa1b1c53dbaa1359c32f43892cb",
}
NUMERICAL_READERS = {
    "ns_full_model_contract.py": "d51190cd8e39fc66882b756ccf3013ff8ef9c89a9d8f160dcde6868f640721f7",
    "ns_full_model_validation.py": "2d80cf8b52318d0ba9e52ccb7beb2d77095fc4be15199a2cff4afb4ea7a6a9be",
}
SCOPE = (
    "Distinct original-training-path diagnostic: eighteen lower-LR family/operand/budget cases "
    "and 54 bounded eight-rank commands. Original trainer and NS boundary are byte-pinned; "
    "no NS compilation wrapper or changed numerical reader. Preserve the full 5120-update LR "
    "schedule, continuous24/interrupted22/resumed24, all-rank checkpoints22/23/24, scalar-loss "
    "and fused/null captured-row/update/work gates and original tolerances. Update23 remains "
    "diagnostic only, including process-local graph rewarming. Retain every rejected case. "
    "A separately issued training-path owner is required; this component grants no GPU or "
    "production authority, causal proof, full-budget quality, efficiency or SwanLab acceptance."
)


def locations(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        root / "run-preparation/ns-training-path-binding-20261119.json",
        root / "controllers/ns-training-path-20261119/tracking",
        root / "run-preparation/ns-training-path-state-20261119.json",
        root / "validation/ns_training_path_20261119",
        root / "run-preparation/validation-ns-training-path-gpu-20261119.json",
    )


def commands(candidate: Path, root: Path, work: Path) -> list[list[str]]:
    """Build original actual-config commands bound only to the new diagnostic namespace."""
    if (
        any(path != path.resolve() or path.is_symlink() for path in (candidate, root, work))
        or work != locations(root)[3]
        or work.is_relative_to(candidate)
        or candidate.is_relative_to(work)
    ):
        raise ValueError("NS training-path diagnostic requires canonical distinct external output")
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
