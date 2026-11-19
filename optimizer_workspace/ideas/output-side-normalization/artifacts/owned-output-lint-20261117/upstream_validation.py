# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared exact command and history contracts for bounded upstream-product validation."""

import json
import math
from pathlib import Path

ARMS = ("muont", "outer_leftdiag_fused", "outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused")
SYNTHETIC_GATES = (
    "gpu_collector",
    "microbatch_partition",
    "distributed_statistics",
    "fa2",
    "gpu_down_updates",
    "synthetic_compiled_body",
    "gpu_output_rows",
    "gpu_block_updates",
    "outer_diagonal_pipeline",
    "gpu_upstream_product_updates",
    "gpu_upstream_scalar_step",
    "block_distributed_statistics",
    "block_compiled_body",
    "cached_root_continuation",
)
GATES = (
    "cpu_numerics",
    "cpu_distributed_statistics",
    "model_parameter_count",
    *SYNTHETIC_GATES,
    "compiled_model",
    "real_data_four_arm_smoke",
    "checkpoint_continuation",
    "real_data_cached_root_continuation",
    "real_data_fused_diagonal_capture",
    "real_data_outer_diagonal_update_parity",
    "real_data_upstream_product_parity",
)


def validation_commands(python: str, candidate: Path, root: Path, work: Path) -> list[list[str]]:
    """Return one two-rank synthetic launch and twelve exact eight-rank trajectory launches."""
    launcher = [python, "-m", "torch.distributed.run", "--standalone"]
    commands = [
        [*launcher, "--nproc_per_node=2", "-m", "experiment.gpu_checks", "--receipt", str(work / "gpu_checks.json")]
    ]
    for arm in ARMS:
        common = [
            *launcher,
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
            "22",
        ]
        if arm.endswith("_fused"):
            common.extend(("--verify-diagonal-capture", "--verify-outer-diagonal-updates"))
        full, resumed = work / f"{arm}_continuous", work / f"{arm}_resumed"
        commands.extend(
            [
                [*common, "--out", str(full), "--max-steps", "24"],
                [*common, "--out", str(resumed), "--max-steps", "22"],
                [
                    *common,
                    "--out",
                    str(resumed),
                    "--max-steps",
                    "24",
                    "--resume",
                    str(resumed / "checkpoints/step_000022"),
                ],
            ]
        )
    return commands


def require_upstream_capture_history(directory: Path, arm: str) -> dict:
    """Require all logged captured updates, exact pinned input and finite same-NS output checks."""
    rows = [
        json.loads(line)
        for path in sorted(directory.glob("metrics_from_*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    if [row["step"] for row in rows] != list(range(1, 25)):
        raise ValueError("Pinned-product capture requires all 24 consecutive updates")
    if arm not in ARMS[2:]:
        raise ValueError("Pinned-product capture history requires an upstream outer or null arm")
    maximum = 0.0
    for row in rows:
        error = row.get("upstream_output_equation_max_error")
        if (
            any(
                row.get(key) is not True
                for key in (
                    "diagonal_capture_parity_passed",
                    "upstream_product_parity_passed",
                    "upstream_scheduled_scalars_passed",
                    "upstream_pinned_input_exact",
                    "upstream_output_equation_passed",
                )
            )
            or row.get("upstream_null_exact") is not (arm == ARMS[3])
            or type(row.get("upstream_owned_stacks_checked")) is not int
            or row["upstream_owned_stacks_checked"] != 3
            or type(error) not in (int, float)
            or not math.isfinite(error)
            or error < 0
        ):
            raise ValueError("Actual pinned-product captured update evidence is incomplete")
        maximum = max(maximum, error)
    return {"updates": 24, "owned_shapes_per_update": 3, "null_exact": arm == ARMS[3], "max_equation_error": maximum}
