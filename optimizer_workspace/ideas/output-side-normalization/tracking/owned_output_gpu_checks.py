# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Candidate-only eight-rank pooled-moment, owned-factor and serialized-state GPU checks."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from reproduce.marin.model import initialize_model

from .block_toner import BLOCK_KINDS, DiagonalMoments, OwnedOutputMoments, attach_output_rows, fast_arguments
from .bootstrap import verify_snapshot
from .capture import build_model
from .execution_owner import require_owned_synthetic_owner
from .gpu_checks import serialized_state, synthetic_rows
from .outer_diagonal_checks import fp64_diagonal_factors
from .owned_output_validation import (
    ARMS,
    ATOL,
    OWNED_STACK_CHECKS,
    OWNED_SYNTHETIC_CHECKS,
    RTOL,
    VALIDATION_ENVIRONMENT,
    WORLD,
    owned_config_sha256,
    owned_rank_coverage,
    require_owned_config,
    require_owned_synthetic_receipt,
)
from .records import atomic_json, sha256, utc_now
from .reference import TASK
from .upstream_full_model_checks import fp64_output_delta, pinned_null_update
from .upstream_outer_products import upstream_outer_update_stack

CAPACITY = 16
POSITIONS = (0, 2, 4, 6)


def source_hashes() -> dict:
    return {str(path.relative_to(TASK)): sha256(path) for path in sorted((TASK / "experiment").rglob("*.py"))}


def require_exact_state(actual: dict, expected: dict) -> None:
    """Check every retained pooled moment, not only those locally used to construct factors."""
    if set(actual) != {"moment", "weight"} or set(expected) != {"moment", "weight"}:
        raise ValueError("Owned-output checkpoint changed the original numerical state schema")
    if actual["weight"] != expected["weight"] or set(actual["moment"]) != set(expected["moment"]):
        raise ValueError("Owned-output checkpoint lost full pooled moment or EMA weight coverage")
    for width, moment in actual["moment"].items():
        torch.testing.assert_close(moment, expected["moment"][width].to(moment.device), rtol=0, atol=0)


@torch.no_grad()
def require_owned_stack_parity(actual, full, restored, pinned, weights, right, outer, *, lr, decay) -> dict:
    """Check executed updates against same-state roots, restored roots and pinned/FP64 references."""
    updated, delta, trace = actual
    for comparison in (full, restored):
        torch.testing.assert_close(updated, comparison[0], rtol=0, atol=0)
        torch.testing.assert_close(delta, comparison[1], rtol=0, atol=0)
        for name in ("ns_source", "ns_output"):
            torch.testing.assert_close(trace[name], comparison[2][name], rtol=0, atol=0)
    torch.testing.assert_close(updated, pinned[0], rtol=0, atol=0)
    torch.testing.assert_close(trace["ns_source"], pinned[1], rtol=0, atol=0)
    torch.testing.assert_close(trace["ns_output"], pinned[2], rtol=0, atol=0)
    if outer is None:
        torch.testing.assert_close(delta, pinned[3], rtol=0, atol=0)
    equation = fp64_output_delta(weights, pinned[2], outer, right[1], lr=lr, decay=decay)
    if any(
        not torch.isfinite(value).all().item()
        for value in (updated, delta, trace["ns_source"], trace["ns_output"], equation)
    ):
        raise ValueError("Owned-output executed stack contains nonfinite numerical evidence")
    torch.testing.assert_close(delta.double(), equation, rtol=RTOL, atol=ATOL)
    return {
        "checks": {**dict.fromkeys(OWNED_STACK_CHECKS, True), "null_update_exact": outer is None},
        "max_errors": {
            "same_moment_update": (delta - full[1]).abs().max().item(),
            "restored_update": (delta - restored[1]).abs().max().item(),
            "output_equation": (delta.double() - equation).abs().max().item(),
        },
    }


@torch.no_grad()
def check_owned_update_stacks(config, device, rank, shape_keys, factors, full_factors, restored_factors) -> list[dict]:
    """Execute both production update kernels on each physical rank's complete projection stacks."""
    args = fast_arguments(config)
    records = []
    lr = torch.tensor(config["optimizer"]["lr"], device=device)
    decay = torch.tensor(config["optimizer"]["weight_decay"], device=device)
    for shape, keys in shape_keys.items():
        count = len(keys)
        rows, columns = shape
        generator = torch.Generator(device=device).manual_seed(701 + rank * 31 + rows + columns)
        weights, gradients, momentum = (
            torch.randn((count, rows, columns), generator=generator, device=device) * 0.01 for _ in range(3)
        )
        right = tuple(torch.eye(columns, device=device).repeat(count, 1, 1) for _ in range(2))
        for factor, low, high in ((right[0], 0.8, 1.2), (right[1], 0.9, 1.1)):
            factor.diagonal(dim1=-2, dim2=-1).copy_(
                torch.linspace(low, high, columns, device=device).expand(count, columns)
            )
        pinned = pinned_null_update(weights, gradients, momentum, right, args=args, lr=lr, decay=decay)
        for arm in ARMS[2:]:
            selected = []
            for roots in (factors, full_factors, restored_factors):
                outer = torch.stack([roots[key][1] for key in keys]) if arm == ARMS[2] else None
                trace = {}
                updated, delta = upstream_outer_update_stack(
                    weights,
                    gradients,
                    momentum,
                    (None, outer),
                    right,
                    args=args,
                    lr=lr,
                    weight_decay=decay,
                    trace=trace,
                )
                selected.append((updated, delta, trace))
            actual_outer = torch.stack([factors[key][1] for key in keys]) if arm == ARMS[2] else None
            checked = require_owned_stack_parity(*selected, pinned, weights, right, actual_outer, lr=lr, decay=decay)
            records.append(
                {
                    "arm": arm,
                    "step": 24,
                    "shape": [count, rows, columns],
                    "keys": [list(key) for key in keys],
                    "ns_mode": config["optimizer"]["ns_mode"],
                    "lr": config["optimizer"]["lr"],
                    "weight_decay": config["optimizer"]["weight_decay"],
                    **checked,
                }
            )
    return records


@torch.no_grad()
def check_owned_output_factors(config: dict, device: torch.device, rank: int) -> dict:
    """Exercise actual dimensions and all pooling ranks while constructing only owned roots."""
    require_owned_config(config)
    network = build_model(
        config,
        positions=POSITIONS,
        row_capacity=CAPACITY,
        rows_per_microbatch=len(POSITIONS),
        device=device,
    )
    initialize_model(network, config["training"]["seed"])
    coverage = owned_rank_coverage(rank)
    shape_keys = {}
    for layer in coverage["layers"]:
        for kind in BLOCK_KINDS:
            shape = tuple(getattr(network.body.blocks[layer], kind).weight.shape)
            shape_keys.setdefault(shape, []).append((layer, kind))
    rows, _ = attach_output_rows(
        list(network.body.blocks),
        BLOCK_KINDS,
        POSITIONS,
        rows_per_microbatch=len(POSITIONS),
        capacity=CAPACITY,
        device=device,
    )
    del network
    if len(rows) != 224 or any(
        value.shape != (CAPACITY, config["model"]["mlp"] if kind in ("gate", "up") else config["model"]["hidden"])
        for (_, kind), value in rows.items()
    ):
        raise ValueError("Synthetic rows do not retain the actual full-model projection dimensions")
    statistics = config["statistics"]
    settings = {
        "decay": statistics["momentum"],
        "damping": statistics["damping"],
        "inner_power": statistics["inner_strength"],
        "outer_power": statistics["outer_strength"],
        "rows_total": CAPACITY * WORLD,
        "world": WORLD,
        "device": device,
    }
    full = DiagonalMoments(rows, backend="triton", **settings)
    owned = OwnedOutputMoments(rows, rank=rank, layers=32, backend="triton", **settings)
    reference = DiagonalMoments(rows, backend="torch", **settings)
    engines = [full, owned, reference]
    fp64 = {width: moment.double().clone() for width, moment in reference.moment.items()}
    expected_keys = {tuple(key) for key in coverage["keys"]}
    max_moment_error, max_factor_error = 0.0, 0.0
    restored = None
    try:
        for step in range(1, 25):
            for (layer, kind), buffer in rows.items():
                value = synthetic_rows(rank, layer, BLOCK_KINDS.index(kind), buffer.shape, step).to(device).float()
                value *= torch.linspace(0.25, 2.0, buffer.shape[-1], device=device)
                value[::4] = 0
                if layer % 4 == 0 and kind == "q":
                    value.zero_()
                buffer.copy_(value)
            sums = {
                width: torch.stack([rows[key].double().square().sum(0) for key in keys])
                for width, keys in reference.keys.items()
            }
            for width, value in sums.items():
                dist.all_reduce(value, group=reference.pg)
                fp64[width] = settings["decay"] * fp64[width] + (1 - settings["decay"]) * value / settings["rows_total"]
            full.update()
            owned.update()
            reference.update()
            require_exact_state(owned.state_dict(), full.state_dict())
            # Factor equality is evaluated from the very same retained moments,
            # independently of the separate pooling parity check above.
            full.load_state_dict(owned.state_dict())
            baseline, actual = full.factors(), owned.factors()
            if set(actual) != expected_keys:
                raise ValueError("Synthetic owned factors do not cover exactly four layers and 28 keys")
            if set(baseline) != set(rows):
                raise ValueError("Original full-factor comparator lost unowned projection coverage")
            for width, moment in owned.moment.items():
                torch.testing.assert_close(moment, reference.moment[width], rtol=RTOL, atol=ATOL)
                torch.testing.assert_close(moment.double(), fp64[width], rtol=RTOL, atol=ATOL)
                max_moment_error = max(
                    max_moment_error,
                    (moment.double() - fp64[width]).abs().max().item(),
                    (moment - reference.moment[width]).abs().max().item(),
                )
                exact = fp64_diagonal_factors(
                    fp64[width], settings["damping"], settings["inner_power"], settings["outer_power"]
                )[1]
                for index, key in enumerate(owned.keys[width]):
                    if key not in expected_keys:
                        continue
                    inner, outer = actual[key]
                    if inner is not None:
                        raise ValueError("Owned-output factors constructed an unused inner root")
                    torch.testing.assert_close(outer, baseline[key][1], rtol=0, atol=0)
                    torch.testing.assert_close(outer.double(), exact[index], rtol=RTOL, atol=ATOL)
                    max_factor_error = max(max_factor_error, (outer.double() - exact[index]).abs().max().item())
            zero_key = (coverage["layers"][0], "q")
            torch.testing.assert_close(actual[zero_key][1], torch.ones_like(actual[zero_key][1]), rtol=0, atol=0)
            if not any(bool((moment.max(-1).values > moment.min(-1).values).any()) for moment in owned.moment.values()):
                raise ValueError("Synthetic rows did not exercise nonuniform moments")
            if abs(owned.weight - (1 - settings["decay"] ** step)) > 1e-12:
                raise ValueError("Synthetic EMA weight differs from the independent equation")
            if step == 22:
                saved = serialized_state(owned.state_dict())
                require_exact_state(saved, owned.state_dict())
                restored = OwnedOutputMoments(rows, rank=rank, layers=32, backend="triton", **settings)
                engines.append(restored)
                restored.load_state_dict(saved)
                require_exact_state(restored.state_dict(), owned.state_dict())
            elif step > 22:
                restored.update()
                require_exact_state(restored.state_dict(), owned.state_dict())
                continued = restored.factors()
                if set(continued) != expected_keys:
                    raise ValueError("Restored owned factors lost projection coverage")
                for key, (inner, outer) in continued.items():
                    if inner is not None:
                        raise ValueError("Restored owned factors constructed an unused inner root")
                    torch.testing.assert_close(outer, actual[key][1], rtol=0, atol=0)
        stack_records = check_owned_update_stacks(config, device, rank, shape_keys, actual, baseline, restored.factors())
        return {
            "rank": rank,
            "step": 24,
            **coverage,
            "owned_shapes": [{"shape": list(shape), "count": len(keys)} for shape, keys in shape_keys.items()],
            "update_stack_records": stack_records,
            "pooled_state_keys": sorted(owned.state_dict()),
            "pooled_moment_shapes": {str(width): list(moment.shape) for width, moment in owned.moment.items()},
            "ema_weight": owned.weight,
            "checks": dict.fromkeys(OWNED_SYNTHETIC_CHECKS, True),
            "max_moment_error": max_moment_error,
            "max_factor_error": max_factor_error,
        }
    finally:
        for engine in engines:
            dist.destroy_process_group(engine.pg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    state = require_owned_synthetic_owner()
    owner = {name: state[name] for name in ("controller_pid", "controller_start_ticks")}
    verify_snapshot()
    local, rank, world = (int(os.environ[name]) for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
    if world != WORLD or rank not in range(WORLD) or local != rank or int(os.environ["LOCAL_WORLD_SIZE"]) != WORLD:
        raise ValueError("Owned-output GPU checks require eight physical ranks on the single owned node")
    if any(os.environ.get(name) != value for name, value in VALIDATION_ENVIRONMENT.items()):
        raise ValueError("Owned-output GPU validation environment differs from the frozen command contract")
    config_path = TASK / f"experiment/configs/{ARMS[2]}_0p5m_seed0.json"
    config = json.loads(config_path.read_text())
    require_owned_config(config)
    sources, config_hash = source_hashes(), sha256(config_path)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    capability = torch.cuda.get_device_capability(device)
    gpu = torch.cuda.get_device_name(device)
    if capability != (8, 0) or "A100" not in gpu:
        raise ValueError("Owned-output GPU checks require A100 SM80 on every physical rank")
    gpu_uuid = str(torch.cuda.get_device_properties(device).uuid)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", device_id=device)
    try:
        checked = check_owned_output_factors(config, device, rank)
        checked.update(local_rank=local, gpu_uuid=gpu_uuid, gpu=gpu, capability=list(capability))
        verify_snapshot()
        if source_hashes() != sources or sha256(config_path) != config_hash:
            raise ValueError("Owned-output synthetic source or config changed during validation")
        ranks = [None] * WORLD
        dist.all_gather_object(ranks, checked)
        record = {
            "time": utc_now(),
            "kind": "owned_output_synthetic_gpu_correctness",
            "world": WORLD,
            "updates": 24,
            "owner": owner,
            "source_sha256": sources,
            "config_sha256": owned_config_sha256(config),
            "config_path": str(config_path),
            "config_file_sha256": config_hash,
            "environment": VALIDATION_ENVIRONMENT,
            "rtol": RTOL,
            "atol": ATOL,
            "resume": {"checkpoint_step": 22, "final_step": 24},
            "checks": {name: all(row["checks"][name] is True for row in ranks) for name in OWNED_SYNTHETIC_CHECKS},
            "rank_records": ranks,
            "evidence": {"command": sys.argv, "row_capacity": CAPACITY, "positions": list(POSITIONS)},
            "limitation": "Synthetic factor correctness is not real-data training or production authorization.",
        }
        require_owned_synthetic_receipt(record, sources, config)
        dist.barrier()
        if rank == 0:
            atomic_json(args.receipt, record)
            print(json.dumps({"check": "gpu_owned_output_factors", "passed": True, "world": WORLD}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
