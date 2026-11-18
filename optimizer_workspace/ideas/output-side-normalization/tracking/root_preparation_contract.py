# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distinct, production-denied full-model test of restored-root kernel readiness."""

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
WORLD = 8
BOUNDARY, FIRST_RESUMED, ENDPOINT = 22, 23, 24
BATCH_TOKENS = 524288
PREPARATION = "root-preparation-diagnostic-preparation.json"
PREPARED_STATUS = "root_preparation_diagnostic_prepared_gpu_denied"
CPU_STATUS = "root_preparation_diagnostic_candidate_cpu_archive_verified_gpu_denied"
FROZEN_STATUS = "frozen_root_preparation_diagnostic_only"
CONTROLLER_MARKER = "OUTPUT_MARIN_ROOT_PREPARATION_CONTROLLER"
CHILD_MARKER = "OUTPUT_MARIN_ROOT_PREPARATION_CHILD"
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_INSPECTOR_ENABLE": "0",
}
REGIMES = ("unprepared", "prepared")
PROTOCOLS = ("control", "trace")
MODES = ("continuous", "interrupted", "resumed")


def locations(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        root / "run-preparation/root-preparation-binding-20261118.json",
        root / "controllers/root-preparation-20261118/tracking",
        root / "run-preparation/root-preparation-state-20261118.json",
        root / "validation/root_preparation_20261118",
        root / "run-preparation/diagnostic-root-preparation-gpu-20261118.json",
    )


def controller_command(root: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "root_preparation_controller.py"),
        "--root",
        str(root),
        "--phase",
        "validate",
    ]


def validator_command(root: Path, candidate: Path) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "root_preparation_validation.py"),
        "--root",
        str(root),
        "--candidate",
        str(candidate),
        "--receipt",
        str(locations(root)[4]),
    ]


def validation_commands(candidate: Path, root: Path, work: Path) -> list[list[str]]:
    if (
        root != root.resolve()
        or candidate != candidate.resolve()
        or work != locations(root)[3]
        or work.resolve().is_relative_to(candidate)
    ):
        raise ValueError("Root diagnostic requires distinct canonical source and output paths")
    require_actual_config(candidate)
    commands = []
    for regime in REGIMES:
        for protocol in PROTOCOLS:
            continuous = work / f"{regime}_{protocol}_continuous"
            resumed = work / f"{regime}_{protocol}_resumed"
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
            for mode in MODES:
                if mode == "continuous":
                    command = [*common, "--out", str(continuous), "--max-steps", str(ENDPOINT)]
                elif mode == "interrupted":
                    command = [*common, "--out", str(resumed), "--max-steps", str(BOUNDARY)]
                else:
                    command = [
                        *common,
                        "--out",
                        str(resumed),
                        "--max-steps",
                        str(ENDPOINT),
                        "--resume",
                        str(resumed / f"checkpoints/step_{BOUNDARY:06d}"),
                    ]
                    if regime == "prepared":
                        command.append("--prepare-restored-roots")
                commands.append(command)
    return commands


def plan(candidate: Path, root: Path) -> dict:
    return {
        "status": "root_preparation_diagnostic_planned_not_executed",
        "candidate": str(candidate),
        "root": str(root),
        "commands": validation_commands(candidate, root, locations(root)[3]),
        "regimes": list(REGIMES),
        "protocols": list(PROTOCOLS),
        "world": WORLD,
        "boundary": BOUNDARY,
        "first_resumed": FIRST_RESUMED,
        "endpoint": ENDPOINT,
        "tolerance": TOLERANCE,
        "planned_tokens": len(REGIMES) * len(PROTOCOLS) * 2 * ENDPOINT * BATCH_TOKENS,
        "scope": (
            "Diagnostic-only fresh unprepared/prepared control and trace trajectories. Only prepared "
            "resumes stage scratch root kernels after checkpoint restoration. Full original lower-LR "
            "schedule, model/data/optimizer/NS/root arithmetic unchanged; no production acceptance."
        ),
        "required_gates": [
            "exact_restored_numerical_state_and_all_rng_preservation",
            "scratch_root_inputs_and_outputs_not_consumed_by_training",
            "first_prepared_update_fresh_capture_then_next_update_replay",
            "trace_off_on_checkpoint_and_loss_transparency_each_regime",
            "unprepared_control_reproduces_original_diagnostic",
            "prepared_uninterrupted_control_equals_unprepared_control",
            "all_rank_continuation_at_22_23_24_original_tolerance",
            "root_capture_repeat_parity_and_cold_process_safety",
            "exact_acknowledged_owner_cleanup_and_healthy_keeper_restoration",
        ],
    }
