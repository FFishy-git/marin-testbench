# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact command and all-rank evidence contracts for owned-output validation."""

import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .upstream_validation import (
        ARMS as ARMS,
    )
    from .upstream_validation import (
        GATES as UPSTREAM_GATES,
    )
    from .upstream_validation import (
        SYNTHETIC_GATES as SYNTHETIC_GATES,
    )
    from .upstream_validation import (
        require_upstream_capture_history,
    )
    from .upstream_validation import (
        validation_commands as upstream_validation_commands,
    )
else:
    from upstream_validation import (
        ARMS as ARMS,
    )
    from upstream_validation import (
        GATES as UPSTREAM_GATES,
    )
    from upstream_validation import (
        SYNTHETIC_GATES as SYNTHETIC_GATES,
    )
    from upstream_validation import (
        require_upstream_capture_history,
    )
    from upstream_validation import (
        validation_commands as upstream_validation_commands,
    )

VALIDATION_ENVIRONMENT = {
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONFAULTHANDLER": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
}
GATES = (*UPSTREAM_GATES, "gpu_owned_output_factors", "real_data_owned_output_factor_parity")
WORLD = 8
LAYERS = 32
EMA_DECAY = 0.938
POOLED_MOMENT_SHAPES = {"512": [160, 512], "2048": [64, 2048]}
OWNED_PROJECTION_SHAPES = [
    {"shape": [512, 512], "count": 16},
    {"shape": [2048, 512], "count": 8},
    {"shape": [512, 2048], "count": 4},
]
PROJECTION_KINDS = ("q", "k", "v", "o", "gate", "up", "down")
RTOL, ATOL = 1e-5, 1e-6
OWNED_RANK_CHECKS = (
    "pooled_moments_torch",
    "pooled_moments_fp64",
    "owned_factors_same_moments_exact",
    "owned_factors_fp64",
    "unused_inner_absent",
)
OWNED_SYNTHETIC_CHECKS = (
    *OWNED_RANK_CHECKS,
    "original_owned_pooling_exact",
    "zero_rows",
    "nonuniform_rows",
    "checkpoint_all_moments_exact",
    "state_22_to_24",
    "owned_update_stack_parity",
)
OWNED_STACK_CHECKS = (
    "same_moment_update_exact",
    "restored_update_exact",
    "pinned_input_exact",
    "pinned_ns_exact",
    "momentum_exact",
    "output_equation_fp64",
)
OWNED_CAPTURE_CHECKS = (
    "global_keys_complete",
    "full_moment_schema_passed",
    "full_moment_parity_passed",
    "moment_weight_exact",
    "owned_coverage_exact",
    "owned_outer_factors_exact",
    "owned_outer_factors_fp64_passed",
    "inner_factors_absent",
)
OWNED_CAPTURE_ERRORS = ("moment_torch", "moment_fp64", "factor_same_state", "factor_fp64", "weight")


def validation_commands(python: str, candidate: Path, root: Path, work: Path) -> list[list[str]]:
    """Retain all original launches, inserting the eight-rank synthetic check at index 1."""
    commands = upstream_validation_commands(python, candidate, root, work)
    commands.insert(
        1,
        [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "-m",
            "experiment.owned_output_gpu_checks",
            "--receipt",
            str(work / "owned_output_gpu_checks.json"),
        ],
    )
    return commands


def owned_config_sha256(config: dict) -> str:
    """Use the candidate checkpoint convention for a parsed configuration identity."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def require_owned_config(config: dict) -> None:
    """Require the actual model dimensions and physical world used by this release."""
    values = (
        config["model"]["layers"],
        config["model"]["hidden"],
        config["model"]["mlp"],
        config["training"]["world_size"],
    )
    if any(type(value) is not int for value in values) or values != (LAYERS, 512, 2048, WORLD):
        raise ValueError("Owned-output evidence requires the actual 32-layer 512/2048 eight-rank configuration")


def owned_rank_coverage(rank: int) -> dict:
    """Return the fixed contiguous four-layer, 28-projection physical-rank partition."""
    if type(rank) is not int or rank not in range(WORLD):
        raise ValueError("Owned-output coverage requires an explicit physical rank")
    layers = list(range(4 * rank, 4 * (rank + 1)))
    return {"layers": layers, "keys": [[layer, kind] for layer in layers for kind in PROJECTION_KINDS]}


def require_rank_records(records: list, step: int, decay: float = EMA_DECAY) -> dict:
    """Reject missing, duplicated, generic or stale per-rank numerical evidence."""
    if not isinstance(records, list) or len(records) != WORLD:
        raise ValueError("Owned-output parity requires eight explicit physical-rank records")
    ranks = [record.get("rank") for record in records if isinstance(record, dict)]
    if len(ranks) != WORLD or any(type(rank) is not int for rank in ranks) or set(ranks) != set(range(WORLD)):
        raise ValueError("Owned-output physical-rank records are incomplete or duplicated")
    maxima = {"max_moment_error": 0.0, "max_factor_error": 0.0}
    for record in records:
        coverage = owned_rank_coverage(record["rank"])
        keys = record.get("keys")
        if (
            type(record.get("step")) is not int
            or record["step"] != step
            or record.get("layers") != coverage["layers"]
            or any(type(layer) is not int for layer in record["layers"])
            or not isinstance(keys, list)
            or len(keys) != 28
        ):
            raise ValueError("Owned-output rank coverage or update identity differs")
        if any(
            not isinstance(key, list) or len(key) != 2 or type(key[0]) is not int or not isinstance(key[1], str)
            for key in keys
        ) or {tuple(key) for key in keys} != {tuple(key) for key in coverage["keys"]}:
            raise ValueError("Owned-output rank projection coverage is incomplete or duplicated")
        checks = record.get("checks")
        if not isinstance(checks, dict) or any(checks.get(name) is not True for name in OWNED_RANK_CHECKS):
            raise ValueError("Owned-output per-rank numerical parity evidence is incomplete")
        state_keys = record.get("pooled_state_keys")
        shapes = record.get("pooled_moment_shapes")
        weight = record.get("ema_weight")
        if (
            not isinstance(state_keys, list)
            or sorted(state_keys) != ["moment", "weight"]
            or shapes != POOLED_MOMENT_SHAPES
            or any(type(value) is not int for shape in shapes.values() for value in shape)
            or type(weight) not in (int, float)
            or not math.isfinite(weight)
            or abs(weight - (1 - decay**step)) > 1e-12
        ):
            raise ValueError("Owned-output full pooled state schema, moment coverage or EMA weight differs")
        for name in maxima:
            error = record.get(name)
            if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
                raise ValueError("Owned-output per-rank numerical errors must be finite and nonnegative")
            maxima[name] = max(maxima[name], error)
    return maxima


def require_owned_synthetic_receipt(record: dict, source_sha256: dict, config: dict) -> dict:
    """Bind successful eight-device synthetic evidence to exact sources and actual config."""
    require_owned_config(config)
    if (
        not source_sha256
        or record.get("source_sha256") != source_sha256
        or record.get("config_sha256") != owned_config_sha256(config)
        or record.get("kind") != "owned_output_synthetic_gpu_correctness"
        or type(record.get("world")) is not int
        or record["world"] != WORLD
        or type(record.get("updates")) is not int
        or record["updates"] != 24
        or record.get("rtol") != RTOL
        or record.get("atol") != ATOL
        or record.get("resume") != {"checkpoint_step": 22, "final_step": 24}
        or record.get("environment") != VALIDATION_ENVIRONMENT
    ):
        raise ValueError("Owned-output synthetic receipt source, config or execution identity differs")
    checks = record.get("checks")
    if not isinstance(checks, dict) or any(checks.get(name) is not True for name in OWNED_SYNTHETIC_CHECKS):
        raise ValueError("Owned-output synthetic numerical checks are incomplete")
    records = record.get("rank_records")
    maxima = require_rank_records(records, 24, config["statistics"]["momentum"])
    uuids = []
    max_update_equation_error = 0.0
    for rank_record in records:
        if (
            type(rank_record.get("local_rank")) is not int
            or rank_record["local_rank"] != rank_record["rank"]
            or not isinstance(rank_record.get("gpu_uuid"), str)
            or not rank_record["gpu_uuid"].strip()
            or not isinstance(rank_record.get("gpu"), str)
            or "A100" not in rank_record["gpu"]
            or rank_record.get("capability") != [8, 0]
            or rank_record.get("owned_shapes") != OWNED_PROJECTION_SHAPES
            or any(rank_record["checks"].get(name) is not True for name in OWNED_SYNTHETIC_CHECKS)
        ):
            raise ValueError("Owned-output synthetic receipt lacks complete physical-device checks")
        checked = require_owned_update_stack_records(
            rank_record.get("update_stack_records"), rank_record["rank"], config
        )
        max_update_equation_error = max(max_update_equation_error, checked["max_update_equation_error"])
        uuids.append(rank_record["gpu_uuid"])
    if len(set(uuids)) != WORLD:
        raise ValueError("Owned-output synthetic receipt reuses a physical GPU")
    return {
        "world": WORLD,
        "updates": 24,
        "owned_layers_per_rank": 4,
        "owned_keys_per_rank": 28,
        "pooled_keys_per_rank": LAYERS * len(PROJECTION_KINDS),
        "pooled_moment_shapes": POOLED_MOMENT_SHAPES.copy(),
        "owned_shapes": OWNED_PROJECTION_SHAPES,
        "checkpoint_step": 22,
        "final_step": 24,
        "config_sha256": record["config_sha256"],
        "update_stacks_per_rank": 6,
        "update_stacks_all_ranks": 48,
        "max_update_equation_error": max_update_equation_error,
        **maxima,
    }


def require_owned_update_stack_records(records: list, rank: int, config: dict) -> dict:
    """Require actual outer and null updates for all three full-size rank-owned stacks."""
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("Owned-output synthetic evidence requires six executed update stacks per rank")
    keys = owned_rank_coverage(rank)["keys"]
    expected = {}
    for arm in ARMS[2:]:
        for item in OWNED_PROJECTION_SHAPES:
            rows, columns = item["shape"]
            members = ("q", "k", "v", "o") if rows == columns else ("gate", "up") if rows > columns else ("down",)
            expected[arm, tuple((item["count"], rows, columns))] = [key for key in keys if key[1] in members]
    seen = set()
    maximum = 0.0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Owned-output executed stack record is missing")
        shape = record.get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or any(type(value) is not int for value in shape):
            raise ValueError("Owned-output executed stack dimensions differ from the full model")
        identity = record.get("arm"), tuple(shape)
        if identity not in expected or identity in seen:
            raise ValueError("Owned-output executed stack shape or arm is missing or duplicated")
        seen.add(identity)
        checks = record.get("checks")
        if (
            type(record.get("step")) is not int
            or record["step"] != 24
            or record.get("keys") != expected[identity]
            or record.get("ns_mode") != config["optimizer"]["ns_mode"]
            or record.get("lr") != config["optimizer"]["lr"]
            or record.get("weight_decay") != config["optimizer"]["weight_decay"]
            or not isinstance(checks, dict)
            or any(checks.get(name) is not True for name in OWNED_STACK_CHECKS)
            or checks.get("null_update_exact") is not (identity[0] == ARMS[3])
        ):
            raise ValueError("Owned-output executed stack keys, settings or numerical checks differ")
        errors = record.get("max_errors")
        if (
            not isinstance(errors, dict)
            or set(errors) != {"same_moment_update", "restored_update", "output_equation"}
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in errors.values())
            or errors["same_moment_update"] != 0
            or errors["restored_update"] != 0
        ):
            raise ValueError("Owned-output executed stack numerical errors are incomplete or nonfinite")
        maximum = max(maximum, errors["output_equation"])
    return {"max_update_equation_error": maximum}


def require_owned_capture_history(directory: Path, arm: str) -> dict:
    """Require original upstream gates plus exact all-rank owned parity on all 24 updates."""
    upstream = require_upstream_capture_history(directory, arm)
    rows = [
        json.loads(line)
        for path in sorted(directory.glob("metrics_from_*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    if [row.get("step") for row in rows] != list(range(1, 25)):
        raise ValueError("Owned-output capture requires all 24 consecutive updates")
    maxima = {"max_moment_error": 0.0, "max_factor_error": 0.0}
    for row in rows:
        if (
            type(row["step"]) is not int
            or row.get("owned_output_parity_passed") is not True
            or type(row.get("owned_output_world")) is not int
            or row["owned_output_world"] != WORLD
            or type(row.get("owned_output_step")) is not int
            or row["owned_output_step"] != row["step"]
            or type(row.get("owned_output_global_count")) is not int
            or row["owned_output_global_count"] != 224
            or row.get("owned_output_global_shapes")
            != [{"width": 512, "moment_shape": [160, 512]}, {"width": 2048, "moment_shape": [64, 2048]}]
        ):
            raise ValueError("Owned-output history lacks all-rank parity evidence")
        records = row.get("owned_output_records")
        if not isinstance(records, list) or len(records) != WORLD:
            raise ValueError("Owned-output capture requires eight explicit physical-rank records")
        maximum_errors = dict.fromkeys((*OWNED_CAPTURE_ERRORS, "upstream_output_equation"), 0.0)
        weight = 0.0
        for _ in range(row["step"]):
            weight = EMA_DECAY * weight + 1 - EMA_DECAY
        for rank, record in enumerate(records):
            coverage = owned_rank_coverage(rank)
            expected = {
                "schema": 1,
                "rank": rank,
                "world": WORLD,
                "step": row["step"],
                "model": {"layers": LAYERS, "hidden": 512, "mlp": 2048},
                "backend": "triton",
                "device_type": "cuda",
                "state_schema": ["moment", "weight"],
                "global_keys": [[layer, kind] for layer in range(LAYERS) for kind in PROJECTION_KINDS],
                "global_count": 224,
                "global_shapes": [
                    {"width": 512, "moment_shape": [160, 512]},
                    {"width": 2048, "moment_shape": [64, 2048]},
                ],
                "owned_keys": coverage["keys"],
                "owned_count": 28,
                "owned_shapes": OWNED_PROJECTION_SHAPES,
            }
            if (
                not isinstance(record, dict)
                or set(record) != set(expected) | {"weight", "checks", "max_errors", "upstream"}
                or any(
                    json.dumps(record.get(name), sort_keys=True) != json.dumps(value, sort_keys=True)
                    for name, value in expected.items()
                )
                or type(record.get("weight")) not in (int, float)
                or record["weight"] != weight
                or not isinstance(record.get("checks"), dict)
                or set(record["checks"]) != set(OWNED_CAPTURE_CHECKS)
                or any(record["checks"][name] is not True for name in OWNED_CAPTURE_CHECKS)
            ):
                raise ValueError("Owned-output captured rank inventory, state schema, weight or checks differ")
            errors = record.get("max_errors")
            if (
                not isinstance(errors, dict)
                or set(errors) != set(OWNED_CAPTURE_ERRORS)
                or any(
                    type(error) not in (int, float) or not math.isfinite(error) or error < 0 for error in errors.values()
                )
                or errors["factor_same_state"] != 0
                or errors["weight"] != 0
            ):
                raise ValueError("Owned-output captured rank numerical errors are incomplete or nonfinite")
            captured = record.get("upstream")
            if not isinstance(captured, dict) or set(captured) != {
                "upstream_product_parity_passed",
                "upstream_scheduled_scalars_passed",
                "upstream_pinned_input_exact",
                "upstream_output_equation_passed",
                "upstream_null_exact",
                "upstream_owned_stacks_checked",
                "upstream_output_equation_max_error",
            }:
                raise ValueError("Owned-output captured rank lacks original upstream evidence")
            for name, value in captured.items():
                if name == "upstream_output_equation_max_error":
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError("Owned-output upstream rank error must be finite and nonnegative")
                    maximum_errors["upstream_output_equation"] = max(maximum_errors["upstream_output_equation"], value)
                elif type(value) is not type(row[name]) or value != row[name]:
                    raise ValueError("Owned-output captured rank upstream parity differs from the original gates")
            for name, error in errors.items():
                maximum_errors[name] = max(maximum_errors[name], error)
        if (
            row.get("owned_output_max_errors") != maximum_errors
            or row["upstream_output_equation_max_error"] != maximum_errors["upstream_output_equation"]
        ):
            raise ValueError("Owned-output all-rank summary differs from the captured numerical errors")
        maxima["max_moment_error"] = max(
            maxima["max_moment_error"], maximum_errors["moment_torch"], maximum_errors["moment_fp64"]
        )
        maxima["max_factor_error"] = max(maxima["max_factor_error"], maximum_errors["factor_fp64"])
    return {
        "updates": 24,
        "ranks": WORLD,
        "rank_updates": 24 * WORLD,
        "owned_layers_per_rank": 4,
        "owned_keys_per_rank": 28,
        "pooled_moment_shapes": POOLED_MOMENT_SHAPES.copy(),
        "upstream": upstream,
        **maxima,
    }
