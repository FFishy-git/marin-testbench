# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A separately owned full-model test of the accepted rational start."""

from pathlib import Path

if __package__:
    from .fused_continuation_contract import CONFIG, TOLERANCE, require_actual_config
    from .fused_continuation_contract import SCHEDULE as SCHEDULE
    from .fused_continuation_contract import config_name as config_name
    from .fused_continuation_contract import trial_specs as trial_specs
else:
    from fused_continuation_contract import CONFIG, TOLERANCE, require_actual_config
    from fused_continuation_contract import SCHEDULE as SCHEDULE
    from fused_continuation_contract import config_name as config_name
    from fused_continuation_contract import trial_specs as trial_specs

PYTHON = "/opt/conda/envs/torch-base/bin/python"
NODE = "28.216.17.9"
WORLD, BOUNDARY, FIRST_RESUMED, ENDPOINT = 8, 22, 23, 24
BATCH_TOKENS = 524288
PREPARATION = "rational-continuation-preparation.json"
PREPARED_STATUS = "rational_start_full_model_candidate_gpu_denied"
CPU_STATUS = "rational_start_full_model_cpu_archive_verified_gpu_unissued"
FROZEN_STATUS = "frozen_rational_start_full_model_diagnostic_not_production"
CONTROLLER_MARKER = "OUTPUT_MARIN_RATIONAL_CONTINUATION_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_RATIONAL_CONTINUATION_CHILD"
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONFAULTHANDLER": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TONER_SYM": "1",
}
VARIANTS = ("pinned", "rational")
PROTOCOLS = ("control", "trace")
MODES = ("continuous", "interrupted", "resumed")


def locations(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        root / "run-preparation/rational-continuation-binding-20261118.json",
        root / "controllers/rational-continuation-20261118/tracking",
        root / "run-preparation/rational-continuation-state-20261118.json",
        root / "validation/rational_continuation_20261118",
        root / "run-preparation/diagnostic-rational-continuation-gpu-20261118.json",
    )


def controller_command(root: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "rational_continuation_controller.py"),
        "--root",
        str(root),
        "--phase",
        "execute",
    ]


def validator_command(root: Path, candidate: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "rational_continuation_validation.py"),
        "--root",
        str(root),
        "--candidate",
        str(candidate),
        "--phase",
        "execute",
    ]


def coordinates() -> list[tuple[str, str, str]]:
    return [(variant, protocol, mode) for variant in VARIANTS for protocol in PROTOCOLS for mode in MODES]


def directory(root: Path, variant: str, protocol: str, mode: str) -> Path:
    return locations(root)[3] / f"{variant}_{protocol}_{'continuous' if mode == 'continuous' else 'resumed'}"


def validation_commands(candidate: Path, root: Path, work: Path) -> list[list[str]]:
    if candidate != candidate.resolve() or root != root.resolve() or work != locations(root)[3]:
        raise ValueError("Rational continuation requires separate canonical source and output paths")
    if work.is_relative_to(candidate):
        raise ValueError("Rational continuation cannot write inside its candidate")
    require_actual_config(candidate)
    commands = []
    for variant, protocol, mode in coordinates():
        out = directory(root, variant, protocol, mode)
        command = [
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
            "--root-start-variant",
            variant,
            "--out",
            str(out),
            "--max-steps",
            str(BOUNDARY if mode == "interrupted" else ENDPOINT),
        ]
        if protocol == "trace":
            command.append("--trace-continuation")
        if mode == "resumed":
            command.extend(("--resume", str(out / f"checkpoints/step_{BOUNDARY:06d}")))
        commands.append(command)
    return commands


def plan(candidate: Path, root: Path) -> dict:
    return {
        "status": "rational_start_full_model_planned_not_executed",
        "root": str(root),
        "candidate": str(candidate),
        "commands": validation_commands(candidate, root, locations(root)[3]),
        "variants": list(VARIANTS),
        "protocols": list(PROTOCOLS),
        "modes": list(MODES),
        "world": WORLD,
        "boundary": BOUNDARY,
        "first_resumed": FIRST_RESUMED,
        "endpoint": ENDPOINT,
        "tolerance": TOLERANCE,
        "planned_tokens": len(VARIANTS) * len(PROTOCOLS) * 2 * ENDPOINT * BATCH_TOKENS,
        "scope": (
            "Full-model original warm-up/capture context with the complete 5120-update schedule and bounded "
            "22/23/24 trajectories. Rational start installed before any warm-up; no readiness intervention. "
            "Trace transparency and pinned historical reproduction precede causal interpretation. "
            "No optimizer quality, efficiency or production authorization."
        ),
    }
