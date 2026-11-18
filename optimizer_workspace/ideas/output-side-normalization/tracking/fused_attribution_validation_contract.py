# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library-only contract for nine separately owned bounded validations."""

import json
import math
from pathlib import Path

FAMILIES = ("muont", "outer_leftdiag_fused", "null_leftdiag_fused")
LR_GRID = (("lower", 0.04525483392), ("center", 0.06399999989260598), ("upper", 0.09050966784))
WORLD, BOUNDARY, ENDPOINT = 8, 22, 24
STEPS, BATCH_TOKENS, TOKENS = 5120, 524288, 2684354560
MODES = ("continuous", "interrupted", "resumed")
NODE = "28.216.17.9"
PYTHON = "/opt/conda/envs/torch-base/bin/python"
MASK = "0,1,2,3,4,5,6,7"
ENVIRONMENT = {
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONFAULTHANDLER": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": MASK,
}
PREPARATION = "fused-attribution-validation-preparation.json"
PREPARED_STATUS = "fused_attribution_validation_prepared_no_gpu_binding"
FROZEN_STATUS = "frozen_fused_attribution_validation_pending_gpu_gates"
CONTROLLER_MARKER = "OUTPUT_MARIN_FUSED_ATTRIBUTION_VALIDATION_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_FUSED_ATTRIBUTION_VALIDATOR"
CONTINUATION_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
SCHEDULE = {
    "warmup_steps": 0,
    "stable_steps": 1024,
    "decay_steps": 4096,
    "kind": "linear",
    "min_lr_ratio": 0.0,
}
EXPANSION_POLICY = {
    "trigger": "any family winner is lower or upper on the initial three-point grid",
    "factor": math.sqrt(2),
    "additional_matrix_lrs": [LR_GRID[0][1] / math.sqrt(2), LR_GRID[-1][1] * math.sqrt(2)],
    "families": list(FAMILIES),
    "steps": STEPS,
    "tokens_per_run": TOKENS,
    "scope": "Add both new endpoints equally for every family before tuned-superiority claims; no authorization.",
}


def locations(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        root / "run-preparation/fused-attribution-validation-binding-20261118.json",
        root / "controllers/fused-attribution-validation-20261118/tracking",
        root / "run-preparation/fused-attribution-validation-state-20261118.json",
        root / "validation/fused_attribution_20261118",
        root / "run-preparation/validation-fused-attribution-gpu-20261118.json",
    )


def predecessor_locations(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / "run-preparation/selected-lr-stage-profile-validation-detached-binding-20261118.json",
        root / "controllers/selected-lr-stage-profile-validation-detached-20261118/tracking",
        root / "run-preparation/selected-lr-stage-profile-validation-detached-state-20261118.json",
        root / "run-preparation/validation-selected-lr-stage-profile-detached-gpu-20261118.json",
    )


def config_name(family: str, label: str) -> str:
    return f"experiment/configs/{family}_full_lr_{label}_0p5m_seed0.json"


def trial_specs() -> tuple[dict, ...]:
    return tuple(
        {
            "name": f"{family}_full_lr_{label}",
            "family": family,
            "lr_label": label,
            "matrix_lr": lr,
            "config_name": config_name(family, label),
        }
        for label, lr in LR_GRID
        for family in FAMILIES
    )


def require_config(config: dict, trial: dict) -> None:
    training, runtime, optimizer = config["training"], config["runtime"], config["optimizer"]
    family = trial["family"]
    if (
        trial not in trial_specs()
        or config["experiment"]["arm"] != family
        or any(type(training[key]) is not int for key in ("steps", "tokens", "batch_tokens", "world_size", "seed"))
        or training["steps"] != STEPS
        or training["tokens"] != TOKENS
        or training["batch_tokens"] != BATCH_TOKENS
        or training["world_size"] != WORLD
        or training["seed"] != 0
        or training["global_batch_sequences"] * config["model"]["sequence_length"] != BATCH_TOKENS
        or config["schedule"] != SCHEDULE
        or optimizer["lr"] != trial["matrix_lr"]
        or optimizer["ns_mode"] != "tpu"
        or optimizer["ns_steps"] != 10
        or config["experiment"]["refresh_install_age"] != 1
        or runtime["compile"] == "eager"
        or runtime["attention"] != "FA2"
        or runtime["head"] != "compiled"
        or config["root"]["graphs"] is not True
        or (family != "muont" and config["experiment"]["diagonal_backend"] != "triton")
    ):
        raise ValueError("Attribution validation requires its exact actual full-schedule grid configuration")


def controller_command(root: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "fused_attribution_validation_controller.py"),
        "--root",
        str(root),
        "--phase",
        "validate",
    ]


def validator_command(root: Path, candidate: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "fused_attribution_validation.py"),
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
    """Plan independent continuous24 and interrupted22/resumed24 executions, never launches."""
    if candidate != candidate.resolve() or root != root.resolve() or work != locations(root)[3]:
        raise ValueError("Attribution validation paths must be canonical")
    if work.resolve().is_relative_to(candidate):
        raise ValueError("Attribution validation work must be external to its candidate")
    commands = []
    for trial in trial_specs():
        path = candidate / trial["config_name"]
        if path.is_symlink():
            raise ValueError("Attribution validation configuration is aliased")
        require_config(json.loads(path.read_bytes()), trial)
        continuous, resumed = (work / f"{trial['name']}_{mode}" for mode in ("continuous", "resumed"))
        common = [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "-m",
            "experiment.train",
            "--config",
            str(path),
            "--data",
            str(root / "data"),
            "--data-receipt",
            str(root / "data/experiment_receipt.json"),
            "--launch-release",
            str(locations(root)[0]),
            "--execution-boundary",
            str(BOUNDARY),
        ]
        if trial["family"] != "muont":
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
