# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regressions for canonical launch identity and all-rank owned evidence."""

import copy
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

if Path(__file__).parent.name == "tests":
    from experiment.owned_output_validation import (
        ARMS,
        OWNED_CAPTURE_CHECKS,
        OWNED_RANK_CHECKS,
        OWNED_STACK_CHECKS,
        OWNED_SYNTHETIC_CHECKS,
        VALIDATION_ENVIRONMENT,
        require_owned_capture_history,
        require_owned_synthetic_receipt,
        validation_commands,
    )
    from experiment.upstream_validation import validation_commands as upstream_validation_commands
else:
    from owned_output_validation import (
        ARMS,
        OWNED_CAPTURE_CHECKS,
        OWNED_RANK_CHECKS,
        OWNED_STACK_CHECKS,
        OWNED_SYNTHETIC_CHECKS,
        VALIDATION_ENVIRONMENT,
        require_owned_capture_history,
        require_owned_synthetic_receipt,
        validation_commands,
    )
    from upstream_validation import validation_commands as upstream_validation_commands


def stack_records(rank, config):
    keys = [
        [layer, kind] for layer in range(rank * 4, rank * 4 + 4) for kind in ("q", "k", "v", "o", "gate", "up", "down")
    ]
    return [
        {
            "arm": arm,
            "step": 24,
            "shape": shape,
            "keys": [key for key in keys if key[1] in kinds],
            "ns_mode": config["optimizer"]["ns_mode"],
            "lr": config["optimizer"]["lr"],
            "weight_decay": config["optimizer"]["weight_decay"],
            "checks": {**dict.fromkeys(OWNED_STACK_CHECKS, True), "null_update_exact": arm == ARMS[3]},
            "max_errors": {"same_moment_update": 0.0, "restored_update": 0.0, "output_equation": 4e-7},
        }
        for arm in ARMS[2:]
        for shape, kinds in (
            ([16, 512, 512], ("q", "k", "v", "o")),
            ([8, 2048, 512], ("gate", "up")),
            ([4, 512, 2048], ("down",)),
        )
    ]


def rank_records(step):
    records = []
    for rank in range(8):
        layers = list(range(rank * 4, rank * 4 + 4))
        record = {
            "rank": rank,
            "step": step,
            "layers": layers,
            "keys": [[layer, kind] for layer in layers for kind in ("q", "k", "v", "o", "gate", "up", "down")],
            "checks": dict.fromkeys(OWNED_RANK_CHECKS, True),
            "pooled_state_keys": ["moment", "weight"],
            "pooled_moment_shapes": {"512": [160, 512], "2048": [64, 2048]},
            "ema_weight": 1 - 0.938**step,
            "max_moment_error": 2e-7,
            "max_factor_error": 3e-7,
        }
        records.append(record)
    return records


def synthetic_receipt(config, sources):
    ranks = rank_records(24)
    for record in ranks:
        rank = record["rank"]
        record.update(local_rank=rank, gpu_uuid=f"GPU-fixture-{rank}", gpu="NVIDIA A100", capability=[8, 0])
        record["owned_shapes"] = [
            {"shape": [512, 512], "count": 16},
            {"shape": [2048, 512], "count": 8},
            {"shape": [512, 2048], "count": 4},
        ]
        record["checks"] = dict.fromkeys(OWNED_SYNTHETIC_CHECKS, True)
        record["update_stack_records"] = stack_records(rank, config)
    return {
        "kind": "owned_output_synthetic_gpu_correctness",
        "world": 8,
        "updates": 24,
        "source_sha256": sources,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "rtol": 1e-5,
        "atol": 1e-6,
        "resume": {"checkpoint_step": 22, "final_step": 24},
        "environment": VALIDATION_ENVIRONMENT.copy(),
        "checks": dict.fromkeys(OWNED_SYNTHETIC_CHECKS, True),
        "rank_records": ranks,
    }


def captured_rows(arm):
    rows = []
    weight = 0.0
    for step in range(1, 25):
        weight = 0.938 * weight + 1 - 0.938
        upstream = {
            "upstream_product_parity_passed": True,
            "upstream_scheduled_scalars_passed": True,
            "upstream_pinned_input_exact": True,
            "upstream_output_equation_passed": True,
            "upstream_null_exact": arm == ARMS[3],
            "upstream_owned_stacks_checked": 3,
            "upstream_output_equation_max_error": 1e-7,
        }
        global_shapes = [{"width": 512, "moment_shape": [160, 512]}, {"width": 2048, "moment_shape": [64, 2048]}]
        errors = {
            "moment_torch": 1e-7,
            "moment_fp64": 2e-7,
            "factor_same_state": 0.0,
            "factor_fp64": 3e-7,
            "weight": 0.0,
        }
        records = [
            {
                "schema": 1,
                "rank": rank,
                "world": 8,
                "step": step,
                "model": {"layers": 32, "hidden": 512, "mlp": 2048},
                "backend": "triton",
                "device_type": "cuda",
                "state_schema": ["moment", "weight"],
                "weight": weight,
                "global_count": 224,
                "global_keys": [
                    [layer, kind] for layer in range(32) for kind in ("q", "k", "v", "o", "gate", "up", "down")
                ],
                "global_shapes": copy.deepcopy(global_shapes),
                "owned_keys": [
                    [layer, kind]
                    for layer in range(rank * 4, rank * 4 + 4)
                    for kind in ("q", "k", "v", "o", "gate", "up", "down")
                ],
                "owned_count": 28,
                "owned_shapes": [
                    {"shape": [512, 512], "count": 16},
                    {"shape": [2048, 512], "count": 8},
                    {"shape": [512, 2048], "count": 4},
                ],
                "checks": dict.fromkeys(OWNED_CAPTURE_CHECKS, True),
                "max_errors": errors.copy(),
                "upstream": upstream.copy(),
            }
            for rank in range(8)
        ]
        rows.append(
            {
                "step": step,
                "diagonal_capture_parity_passed": True,
                **upstream,
                "owned_output_parity_passed": True,
                "owned_output_world": 8,
                "owned_output_step": step,
                "owned_output_global_count": 224,
                "owned_output_global_shapes": global_shapes,
                "owned_output_max_errors": errors | {"upstream_output_equation": 1e-7},
                "owned_output_records": records,
            }
        )
    return rows


def write_history(directory, rows):
    (directory / "metrics_from_000000.jsonl").write_text("\n".join(json.dumps(row) for row in rows[:22]))
    (directory / "metrics_from_000022.jsonl").write_text("\n".join(json.dumps(row) for row in rows[22:]))


def test_validation_commands_preserve_original_trajectories_and_distinct_synthetic_receipt(tmp_path):
    candidate, root, work = tmp_path / "candidate", tmp_path / "root", tmp_path / "work"
    original = upstream_validation_commands("python", candidate, root, work)
    commands = validation_commands("python", candidate, root, work)
    assert [commands[0], *commands[2:]] == original
    assert commands[1] == [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "experiment.owned_output_gpu_checks",
        "--receipt",
        str(work / "owned_output_gpu_checks.json"),
    ]
    assert len(commands) == 14


@pytest.mark.parametrize(
    "corruption",
    [
        "source",
        "config",
        "reduced_shape",
        "rank_missing",
        "rank_duplicate",
        "gpu_duplicate",
        "rank_generic",
        "resume",
        "remote_check",
        "error",
        "local_moments_only",
        "state_schema",
        "ema_weight",
        "stack_shapes",
        "stack_execution_missing",
        "stack_duplicate",
        "stack_keys",
        "stack_mode",
        "stack_null",
        "stack_restored",
        "stack_error",
    ],
)
def test_synthetic_receipt_rejects_unbound_partial_or_replayed_evidence(corruption):
    config = {
        "model": {"layers": 32, "hidden": 512, "mlp": 2048},
        "training": {"world_size": 8},
        "statistics": {"momentum": 0.938},
        "optimizer": {"ns_mode": "tpu", "lr": 0.02262741696, "weight_decay": 0.1},
    }
    sources = {"experiment/block_toner.py": "a" * 64, "experiment/owned_output_gpu_checks.py": "b" * 64}
    record = synthetic_receipt(config, sources)
    checked = require_owned_synthetic_receipt(record, sources, config)
    assert checked["owned_keys_per_rank"] == 28
    assert checked["pooled_keys_per_rank"] == 224
    assert checked["max_factor_error"] == 3e-7
    assert checked["update_stacks_all_ranks"] == 48
    assert checked["max_update_equation_error"] == 4e-7
    if corruption == "source":
        record["source_sha256"] = sources | {"experiment/block_toner.py": "c" * 64}
    elif corruption == "config":
        changed = config | {"statistics": {"damping": 0.01}}
        record = synthetic_receipt(changed, sources)
    elif corruption == "reduced_shape":
        config = copy.deepcopy(config)
        config["model"]["hidden"] = 256
        record = synthetic_receipt(config, sources)
    elif corruption == "rank_missing":
        record["rank_records"].pop()
    elif corruption == "rank_duplicate":
        record["rank_records"][-1] = copy.deepcopy(record["rank_records"][0])
    elif corruption == "gpu_duplicate":
        record["rank_records"][-1]["gpu_uuid"] = record["rank_records"][0]["gpu_uuid"]
    elif corruption == "rank_generic":
        for row in record["rank_records"]:
            row.pop("keys")
            row["keys_checked"] = 28
    elif corruption == "resume":
        record["rank_records"][-1]["step"] = 22
    elif corruption == "remote_check":
        record["rank_records"][-1]["checks"]["state_22_to_24"] = False
    elif corruption == "local_moments_only":
        record["rank_records"][-1]["pooled_moment_shapes"] = {"512": [20, 512], "2048": [8, 2048]}
    elif corruption == "state_schema":
        record["rank_records"][-1]["pooled_state_keys"] = ["moment"]
    elif corruption == "ema_weight":
        record["rank_records"][-1]["ema_weight"] = 1 - 0.938**22
    elif corruption == "stack_shapes":
        record["rank_records"][-1]["owned_shapes"][0]["count"] = 28
    elif corruption == "stack_execution_missing":
        record["rank_records"][-1].pop("update_stack_records")
    elif corruption == "stack_duplicate":
        stacks = record["rank_records"][-1]["update_stack_records"]
        stacks[-1] = copy.deepcopy(stacks[0])
    elif corruption == "stack_keys":
        record["rank_records"][-1]["update_stack_records"][-1]["keys"][0] = [0, "down"]
    elif corruption == "stack_mode":
        record["rank_records"][-1]["update_stack_records"][-1]["ns_mode"] = "f32"
    elif corruption == "stack_null":
        record["rank_records"][-1]["update_stack_records"][-1]["checks"]["null_update_exact"] = False
    elif corruption == "stack_restored":
        record["rank_records"][-1]["update_stack_records"][-1]["max_errors"]["restored_update"] = 1e-7
    elif corruption == "stack_error":
        record["rank_records"][-1]["update_stack_records"][-1]["max_errors"]["output_equation"] = float("nan")
    else:
        record["rank_records"][-1]["max_moment_error"] = float("nan")
    with pytest.raises(ValueError):
        require_owned_synthetic_receipt(record, sources, config)


@pytest.fixture
def stack_comparator():
    if Path(__file__).parent.name == "tests":
        return importlib.import_module("experiment.owned_output_gpu_checks").require_owned_stack_parity
    candidate = Path(
        "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
        "/candidates/owned-output-observed-formatted-20261117/experiment"
    )
    package_name = "_owned_stack_comparator_cpu"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name, candidate / "__init__.py", submodule_search_locations=[str(candidate)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
        for module, filename in (
            ("owned_output_validation", "owned_output_validation.py"),
            ("execution_owner", "owned_output_execution_owner.py"),
            ("owned_output_gpu_checks", "owned_output_gpu_checks.py"),
        ):
            spec = importlib.util.spec_from_file_location(f"{package_name}.{module}", Path(__file__).with_name(filename))
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded
            spec.loader.exec_module(loaded)
    return sys.modules[f"{package_name}.owned_output_gpu_checks"].require_owned_stack_parity


@pytest.mark.parametrize("shape", [(2, 4, 4), (2, 8, 4), (2, 4, 8)])
@pytest.mark.parametrize("apply_left", [False, True])
def test_executed_stack_comparator_checks_independent_cpu_equation_and_corrupted_results(
    stack_comparator, shape, apply_left
):
    # Reduced CPU tensors exercise the comparator only, never emulate CUDA kernels.
    generator = torch.Generator().manual_seed(811)
    count, rows, columns = shape
    weights, gradients, momentum = (torch.randn(shape, generator=generator) * 0.01 for _ in range(3))
    right = tuple(torch.eye(columns).repeat(count, 1, 1) for _ in range(2))
    right[0].diagonal(dim1=-2, dim2=-1).copy_(torch.linspace(0.8, 1.2, columns))
    right[1].diagonal(dim1=-2, dim2=-1).copy_(torch.linspace(0.9, 1.1, columns))
    updated = 0.95 * momentum + gradients
    source = torch.bmm(gradients + 0.95 * updated, right[0])
    direction = source / source.norm(dim=(-2, -1), keepdim=True)
    outer = torch.linspace(0.5, 1.5, rows).repeat(count, 1) if apply_left else None
    lr, decay = torch.tensor(0.02), torch.tensor(0.1)
    normalized_right = right[1].double() / (right[1].double().square().sum((-2, -1), keepdim=True) / columns).sqrt()
    transformed = direction.double()
    if outer is not None:
        transformed = transformed * (outer.double() / outer.double().square().mean(-1, keepdim=True).sqrt())[:, :, None]
    delta = -lr.double() * (
        max(1.0, rows / columns) ** 0.5 * torch.bmm(transformed, normalized_right)
        + decay.double() * torch.bmm(torch.bmm(weights.double(), normalized_right), normalized_right)
    )
    trace = {"ns_source": source, "ns_output": direction}
    actual = (updated, delta.float(), trace)
    full, restored = copy.deepcopy(actual), copy.deepcopy(actual)
    pinned = (updated, source, direction, delta.float())
    checked = stack_comparator(actual, full, restored, pinned, weights, right, outer, lr=lr, decay=decay)
    assert checked["checks"]["null_update_exact"] is (not apply_left)
    assert checked["max_errors"]["output_equation"] < 1e-6
    restored[1][0, 0, 0] += 0.01
    with pytest.raises(AssertionError):
        stack_comparator(actual, full, restored, pinned, weights, right, outer, lr=lr, decay=decay)
    incorrect = (updated, delta.float() + 0.01, trace)
    with pytest.raises(AssertionError):
        stack_comparator(incorrect, incorrect, incorrect, pinned, weights, right, outer, lr=lr, decay=decay)
    bad_trace = {"ns_source": source + 0.01, "ns_output": direction}
    incorrect = (updated, delta.float(), bad_trace)
    with pytest.raises(AssertionError):
        stack_comparator(incorrect, incorrect, incorrect, pinned, weights, right, outer, lr=lr, decay=decay)


@pytest.mark.parametrize("arm", ARMS[2:])
@pytest.mark.parametrize(
    "corruption",
    [
        "missing_history",
        "missing_update",
        "duplicate_update",
        "rank_missing",
        "rank_duplicate",
        "rank_generic",
        "projection_missing",
        "projection_duplicate",
        "wrong_owner",
        "resumed_step",
        "resumed_check",
        "resumed_error",
        "upstream",
        "flag_only",
        "local_moments_only",
        "state_schema",
        "resumed_weight",
        "remote_upstream",
        "summary",
        "stack_shapes",
    ],
)
def test_owned_capture_history_rejects_incomplete_all_rank_or_corrupted_resume_tail(tmp_path, arm, corruption):
    rows = captured_rows(arm)
    write_history(tmp_path, rows)
    checked = require_owned_capture_history(tmp_path, arm)
    assert checked["rank_updates"] == 192
    assert checked["upstream"]["null_exact"] is (arm == ARMS[3])
    assert checked["max_moment_error"] == 2e-7
    remote = rows[-1]["owned_output_records"][-1]
    if corruption == "missing_history":
        rows.clear()
    elif corruption == "missing_update":
        rows.pop()
    elif corruption == "duplicate_update":
        rows[-1]["step"] = 23
    elif corruption == "rank_missing":
        rows[-1]["owned_output_records"].pop()
    elif corruption == "rank_duplicate":
        rows[-1]["owned_output_records"][-1] = copy.deepcopy(rows[-1]["owned_output_records"][0])
    elif corruption == "rank_generic":
        for record in rows[-1]["owned_output_records"]:
            record.pop("owned_keys")
            record["keys_checked"] = 28
    elif corruption == "projection_missing":
        remote["owned_keys"].pop()
    elif corruption == "projection_duplicate":
        remote["owned_keys"][-1] = remote["owned_keys"][0]
    elif corruption == "wrong_owner":
        remote["owned_keys"][0] = [0, "q"]
    elif corruption == "resumed_step":
        remote["step"] = 22
    elif corruption == "resumed_check":
        remote["checks"]["owned_outer_factors_exact"] = False
    elif corruption == "resumed_error":
        remote["max_errors"]["factor_fp64"] = float("inf")
    elif corruption == "upstream":
        rows[-1]["upstream_pinned_input_exact"] = False
    elif corruption == "local_moments_only":
        remote["global_shapes"] = [{"width": 512, "moment_shape": [20, 512]}, {"width": 2048, "moment_shape": [8, 2048]}]
    elif corruption == "state_schema":
        remote["state_schema"] = ["moment", "weight", "local_only"]
    elif corruption == "resumed_weight":
        remote["weight"] = 1 - 0.938**22
    elif corruption == "remote_upstream":
        remote["upstream"]["upstream_null_exact"] = arm != ARMS[3]
    elif corruption == "summary":
        rows[-1]["owned_output_max_errors"]["factor_fp64"] = 0
    elif corruption == "stack_shapes":
        remote["owned_shapes"][0]["count"] = 28
    else:
        rows[-1].pop("owned_output_records")
    write_history(tmp_path, rows)
    with pytest.raises(ValueError):
        require_owned_capture_history(tmp_path, arm)
