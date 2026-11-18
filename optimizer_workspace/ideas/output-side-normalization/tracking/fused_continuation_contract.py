# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate execution identity for first-update fused continuation diagnosis."""

import json
from pathlib import Path

if __package__:
    from .fused_attribution_validation_contract import (
        BATCH_TOKENS,
        CONTINUATION_TOLERANCE,
        PYTHON,
        WORLD,
        config_name,
        require_config,
        trial_specs,
    )
    from .fused_attribution_validation_contract import (
        ENVIRONMENT as ENVIRONMENT,
    )
    from .fused_attribution_validation_contract import (
        NODE as NODE,
    )
    from .fused_attribution_validation_contract import (
        SCHEDULE as SCHEDULE,
    )
else:
    from fused_attribution_validation_contract import (
        BATCH_TOKENS,
        CONTINUATION_TOLERANCE,
        PYTHON,
        WORLD,
        config_name,
        require_config,
        trial_specs,
    )
    from fused_attribution_validation_contract import (
        ENVIRONMENT as ENVIRONMENT,
    )
    from fused_attribution_validation_contract import (
        NODE as NODE,
    )
    from fused_attribution_validation_contract import (
        SCHEDULE as SCHEDULE,
    )

PREPARATION = "fused-continuation-preparation.json"
PREPARED_STATUS = "fused_continuation_prepared_gpu_denied"
CPU_STATUS = "fused_continuation_cpu_archive_verified_gpu_denied"
FROZEN_STATUS = "frozen_fused_continuation_diagnostic_only"
CONTROLLER_MARKER = "OUTPUT_MARIN_FUSED_CONTINUATION_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_FUSED_CONTINUATION_CHILD"
BOUNDARY, FIRST_RESUMED, ENDPOINT = 22, 23, 24
PROTOCOLS = ("control", "trace")
MODES = ("continuous", "interrupted", "resumed")
FAMILY, LR_LABEL = "outer_leftdiag_fused", "lower"
CONFIG = config_name(FAMILY, LR_LABEL)
TOLERANCE = CONTINUATION_TOLERANCE


def locations(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        root / "run-preparation/fused-continuation-binding-20261118.json",
        root / "controllers/fused-continuation-20261118/tracking",
        root / "run-preparation/fused-continuation-state-20261118.json",
        root / "validation/fused_continuation_20261118",
        root / "run-preparation/diagnostic-fused-continuation-gpu-20261118.json",
    )


def require_actual_config(candidate: Path) -> None:
    spec = next(item for item in trial_specs() if item["family"] == FAMILY and item["lr_label"] == LR_LABEL)
    require_config(json.loads((candidate / CONFIG).read_bytes()), spec)


def controller_command(root: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "fused_continuation_controller.py"),
        "--root",
        str(root),
        "--phase",
        "validate",
    ]


def validator_command(root: Path, candidate: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "fused_continuation_validation.py"),
        "--root",
        str(root),
        "--candidate",
        str(candidate),
        "--receipt",
        str(locations(root)[4]),
    ]


def validation_commands(candidate: Path, root: Path, work: Path) -> list[list[str]]:
    """Retain the lower-LR full schedule in paired control and traced trajectories."""
    if candidate != candidate.resolve() or root != root.resolve() or work != locations(root)[3]:
        raise ValueError("Continuation diagnostic requires canonical separate paths")
    if work.resolve().is_relative_to(candidate):
        raise ValueError("Continuation diagnostic output cannot modify the candidate")
    require_actual_config(candidate)
    commands = []
    for protocol in PROTOCOLS:
        continuous, resumed = (work / f"{protocol}_{mode}" for mode in ("continuous", "resumed"))
        common = [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "-m",
            "experiment.train",
            "--config",
            str(candidate / CONFIG),
            "--data",
            str(root / "data"),
            "--data-receipt",
            str(root / "data/experiment_receipt.json"),
            "--launch-release",
            str(locations(root)[0]),
            "--execution-boundary",
            str(BOUNDARY),
            "--verify-diagonal-capture",
            "--verify-outer-diagonal-updates",
        ]
        if protocol == "trace":
            common.append("--trace-continuation")
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


def plan(candidate: Path, root: Path) -> dict:
    return {
        "status": "fused_continuation_planned_not_executed",
        "root": str(root),
        "candidate": str(candidate),
        "commands": validation_commands(candidate, root, locations(root)[3]),
        "config": str(candidate / CONFIG),
        "world": WORLD,
        "boundary": BOUNDARY,
        "first_resumed": FIRST_RESUMED,
        "endpoint": ENDPOINT,
        "planned_tokens": len(PROTOCOLS) * 2 * ENDPOINT * BATCH_TOKENS,
        "tolerance": TOLERANCE,
        "protocols": list(PROTOCOLS),
        "scope": (
            "Validation-only control and intermediate-copy trajectories at the original lower LR. "
            "New update23 checkpoint on both protocols; no production authorization, optimizer "
            "quality or throughput result. Interpret traced intermediates only after trace-off/on parity."
        ),
    }
