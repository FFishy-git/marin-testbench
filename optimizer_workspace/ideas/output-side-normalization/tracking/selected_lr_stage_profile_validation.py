# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan and reconstruct bounded selected-LR profiling without granting GPU ownership."""

import json
import math
import shutil
from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import torch
from analyze_tuning import executed_sources, normalized_sources
from compare_runs import history, validate_rows
from launch_wave5 import read, sha256
from selected_lr_stage_profile_contract import (
    ARMS,
    BOUNDARY,
    ENDPOINT,
    ENVIRONMENT,
    MODES,
    PROFILE_UPDATES,
    STATUS,
    WORLD,
    locations,
    validation_commands,
)
from selected_lr_validation import require_config
from validate_selected_lr_full_budget import checkpoint_evidence, numerical_modules
from watch_wave5 import write

PREFIX = "profile-resume-prefix.json"
PREFIX_MANIFEST = "profile-prefix-manifest.json"
STATE_FIELDS = {
    "step",
    "next_global_sequence",
    "config",
    "rank",
    "world",
    "identity",
    "optimizers",
    "fast",
    "output",
    "torch_rng",
    "cuda_rng",
    "python_rng",
}


def plan(candidate: Path, root: Path, work: Path) -> dict:
    """Keep the actual full schedule for one continuous and two matched resumed branches."""
    candidate, root, work = candidate.resolve(), root.resolve(), work.resolve()
    if work != locations(root)[3] or work.is_relative_to(candidate):
        raise ValueError("Stage-profile validation requires its distinct canonical external work directory")
    trials, commands = [], validation_commands(candidate, root, work)
    for index, arm in enumerate(ARMS):
        config_path = candidate / f"experiment/configs/{arm}_0p5m_seed0.json"
        config = read(config_path)
        require_config(config, arm)
        outputs = {mode: str(work / f"{arm}_{mode}") for mode in MODES}
        trials.append(
            {
                "arm": arm,
                "config": str(config_path),
                "outputs": outputs,
                "commands": commands[index * len(MODES) : (index + 1) * len(MODES)],
            }
        )
    return {
        "status": STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "work": str(work),
        "environment": ENVIRONMENT,
        "world": WORLD,
        "boundary": BOUNDARY,
        "endpoint": ENDPOINT,
        "profile_updates": list(PROFILE_UPDATES),
        "continuation_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "planned_executed_tokens": len(ARMS) * (ENDPOINT + 2 * (ENDPOINT - BOUNDARY)) * 524288,
        "trials": trials,
        "gpu_commands": [],
        "scope": (
            "Twelve bounded commands, with two private hash-bound copies of checkpoint/prefix 200 per arm. "
            "The 5120-update schedule, compiled model and cached-root graphs remain enabled. "
            "Resumed traces include graph rewarming; they do not measure steady-state throughput. "
            "This plan does not authorize execution, establish optimizer quality, or replace keeper handoff."
        ),
    }


def fixed_age_evidence(rows: list[dict]) -> dict:
    """Inspect actual installation boundaries throughout a complete 220-update trajectory."""
    if [row["step"] for row in rows] != list(range(1, ENDPOINT + 1)):
        raise ValueError("Stage-profile history lacks its complete bounded trajectory")
    installations = 0
    for row in rows:
        for record in row["input_refresh_installations"]:
            if (
                record["installation_step"] != row["step"] - 1
                or record["installation_step"] - record["source_step"] != 1
            ):
                raise ValueError("Stage-profile history changed the fixed installation age")
            installations += 1
    if not installations:
        raise ValueError("Stage-profile trajectory never exercised host refresh installation")
    return {"rank": 0, "updates": ENDPOINT, "installation_age": 1, "observed_installations": installations}


def complete_checkpoint_evidence(directory: Path, config: dict, candidate: Path, frozen: dict, step: int) -> dict:
    """Reject incomplete state trees even when both compared trajectories lost the same fields."""
    checkpoint = directory / f"checkpoints/step_{step:06d}"
    names = {"complete.json", *(f"rank_{rank}.pt" for rank in range(WORLD))}
    if (
        checkpoint.is_symlink()
        or {path.name for path in checkpoint.iterdir()} != names
        or any(path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 for path in checkpoint.iterdir())
    ):
        raise ValueError("Stage-profile checkpoint does not have exactly eight regular rank states")
    for rank in range(WORLD):
        path = checkpoint / f"rank_{rank}.pt"
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("Stage-profile checkpoint rank state is aliased or shared")
        state = torch.load(path, map_location="cpu", weights_only=False)
        required = STATE_FIELDS | ({"model"} if rank == 0 else set())
        if set(state) != required:
            raise ValueError("Stage-profile checkpoint is missing a numerical, optimizer or RNG state field")
    return checkpoint_evidence(directory, config, candidate, frozen, step)


def trajectory_evidence(directory: Path, config: dict, candidate: Path, frozen: dict, *, resumed: bool) -> dict:
    """Reconstruct the source/data identity and exact segment coverage independently of numerical parity."""
    run = history(directory)
    segments = sorted(directory.glob("metrics_from_*.jsonl"))
    expected_segments = ["metrics_from_000000.jsonl"]
    if resumed:
        expected_segments.append(f"metrics_from_{BOUNDARY:06d}.jsonl")
    resume_path = str(directory / f"checkpoints/step_{BOUNDARY:06d}") if resumed else None
    if (
        run.manifest["config"] != config
        or run.manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
        or normalized_sources(candidate, run.manifest["source_sha256"]) != executed_sources(candidate, frozen)
        or run.manifest["status"] != "execution_check_complete"
        or run.manifest.get("execution_check") is not True
        or type(run.manifest["completed_steps"]) is not int
        or run.manifest["completed_steps"] != ENDPOINT
        or run.manifest.get("resume") != resume_path
        or len(run.rows) != ENDPOINT
        or [path.name for path in segments] != expected_segments
        or (resumed and [len(segment) for segment in run.segments] != [BOUNDARY, ENDPOINT - BOUNDARY])
        or (directory / "run.json").is_symlink()
        or any(
            path.is_symlink() or path.stat().st_nlink != 1 or not path.read_bytes().endswith(b"\n") for path in segments
        )
    ):
        raise ValueError("Stage-profile trajectory source, manifest or segment coverage differs")
    return {
        "manifest_sha256": sha256(directory / "run.json"),
        "history_sha256": {path.name: sha256(path) for path in segments},
        "observed_updates": len(run.rows),
        "fixed_age_history": fixed_age_evidence(run.rows),
        "checkpoints": {
            str(step): complete_checkpoint_evidence(directory, config, candidate, frozen, step)
            for step in (BOUNDARY, ENDPOINT)
        },
    }


def prefix_source(directory: Path, config: dict, candidate: Path, frozen: dict) -> tuple[dict, bytes]:
    evidence = trajectory_evidence(directory, config, candidate, frozen, resumed=False)
    path = directory / "metrics_from_000000.jsonl"
    content = path.read_bytes()
    if sha256(path) != evidence["history_sha256"][path.name]:
        raise ValueError("Continuous history changed during prefix reconstruction")
    prefix = b"".join(content.splitlines(keepends=True)[:BOUNDARY])
    validate_rows([json.loads(line) for line in prefix.splitlines()], config["training"]["batch_tokens"])
    checkpoint = directory / f"checkpoints/step_{BOUNDARY:06d}"
    expected = {"complete.json", *(f"rank_{rank}.pt" for rank in range(WORLD))}
    if checkpoint.is_symlink() or {path.name for path in checkpoint.iterdir()} != expected:
        raise ValueError("Stage-profile prefix checkpoint contains an unexpected file or alias")
    if any(path.is_symlink() or not path.is_file() for path in checkpoint.iterdir()):
        raise ValueError("Stage-profile prefix checkpoint must contain regular files")
    evidence["prefix_checkpoint_sha256"] = {name: sha256(checkpoint / name) for name in sorted(expected)}
    if read(directory / "config.json") != config:
        raise ValueError("Continuous configuration file differs from the executed configuration")
    evidence["config_file_sha256"] = sha256(directory / "config.json")
    return evidence, prefix


def seed_resume_prefix(continuous: Path, destination: Path, config: dict, candidate: Path, frozen: dict) -> dict:
    """Copy only an independently verified owned prefix, preserving all original execution evidence."""
    continuous, destination, candidate = continuous.resolve(), destination.resolve(), candidate.resolve()
    if (
        destination.exists()
        or destination.is_relative_to(continuous)
        or destination.is_relative_to(candidate)
        or continuous.is_relative_to(destination)
    ):
        raise FileExistsError("Stage-profile resume prefix requires an unused independent output")
    evidence, prefix = prefix_source(continuous, config, candidate, frozen)
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint_name = f"checkpoints/step_{BOUNDARY:06d}"
    shutil.copytree(continuous / checkpoint_name, destination / checkpoint_name)
    shutil.copy2(continuous / "config.json", destination / "config.json")
    (destination / "metrics_from_000000.jsonl").write_bytes(prefix)
    seed_manifest = {
        "status": "stage_profile_owned_prefix_prepared",
        "config": config,
        "prefix_steps": BOUNDARY,
        "source_continuous": str(continuous),
        "source_manifest_sha256": evidence["manifest_sha256"],
        "scope": "Copied prefix for bounded resume; no new training execution has occurred.",
    }
    content = (json.dumps(seed_manifest, indent=2) + "\n").encode()
    (destination / PREFIX_MANIFEST).write_bytes(content)
    (destination / "run.json").write_bytes(content)
    result = {
        "status": "stage_profile_owned_prefix_seeded",
        "continuous": str(continuous),
        "destination": str(destination),
        "source_evidence": evidence,
        "prefix_manifest_sha256": sha256(destination / PREFIX_MANIFEST),
        "prefix_history_sha256": sha256(destination / "metrics_from_000000.jsonl"),
        "checkpoint_sha256": evidence["prefix_checkpoint_sha256"],
    }
    write(destination / PREFIX, result)
    require_resume_prefix(destination, config, candidate, frozen)
    return read(destination / PREFIX)


def require_resume_prefix(destination: Path, config: dict, candidate: Path, frozen: dict) -> dict:
    """Recheck the private prefix after resume has replaced its initial run manifest."""
    result = read(destination / PREFIX)
    continuous = Path(result["continuous"])
    source, prefix = prefix_source(continuous, config, candidate, frozen)
    checkpoint = destination / f"checkpoints/step_{BOUNDARY:06d}"
    expected_manifest = {
        "status": "stage_profile_owned_prefix_prepared",
        "config": config,
        "prefix_steps": BOUNDARY,
        "source_continuous": str(continuous),
        "source_manifest_sha256": source["manifest_sha256"],
        "scope": "Copied prefix for bounded resume; no new training execution has occurred.",
    }
    if (
        destination.is_symlink()
        or (destination / PREFIX).is_symlink()
        or (destination / PREFIX_MANIFEST).is_symlink()
        or (destination / "metrics_from_000000.jsonl").is_symlink()
        or (destination / "metrics_from_000000.jsonl").stat().st_nlink != 1
        or (destination / "config.json").is_symlink()
        or checkpoint.is_symlink()
        or not checkpoint.is_dir()
        or result["status"] != "stage_profile_owned_prefix_seeded"
        or result["destination"] != str(destination.resolve())
        or result["source_evidence"] != source
        or read(destination / PREFIX_MANIFEST) != expected_manifest
        or result["prefix_manifest_sha256"] != sha256(destination / PREFIX_MANIFEST)
        or (destination / "metrics_from_000000.jsonl").read_bytes() != prefix
        or result["prefix_history_sha256"] != sha256(destination / "metrics_from_000000.jsonl")
        or result["checkpoint_sha256"] != source["prefix_checkpoint_sha256"]
        or {path.name for path in checkpoint.iterdir()} != set(result["checkpoint_sha256"])
        or any(path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 for path in checkpoint.iterdir())
        or any(sha256(checkpoint / name) != digest for name, digest in result["checkpoint_sha256"].items())
        or read(destination / "config.json") != config
    ):
        raise ValueError("Stage-profile owned resume prefix or original continuous evidence changed")
    return result


def valid_trace_interval(event: dict) -> bool:
    return all(
        not isinstance(event.get(key), bool)
        and isinstance(event.get(key), (float, int))
        and math.isfinite(event[key])
        and (key != "dur" or event[key] > 0)
        for key in ("ts", "dur")
    ) and math.isfinite(event["ts"] + event["dur"])


def trace_region_device_type(region: dict) -> str:
    """Recover the device key omitted by the issued Torch key-averages summary."""
    keys = (
        "cpu_total_microseconds",
        "cpu_self_microseconds",
        "device_total_microseconds",
        "device_self_microseconds",
    )
    if any(
        isinstance(region.get(key), bool)
        or not isinstance(region.get(key), (float, int))
        or not math.isfinite(region[key])
        for key in keys
    ):
        raise ValueError("Stage-profile summary has invalid timing fields")
    if region["cpu_total_microseconds"] > 0:
        device = "CPU"
    elif (
        region["cpu_total_microseconds"] == region["cpu_self_microseconds"] == 0
        and region["device_total_microseconds"] > 0
    ):
        device = "CUDA"
    else:
        raise ValueError("Stage-profile summary has an ambiguous device type")
    if region.get("device_type", device) != device:
        raise ValueError("Stage-profile summary device type contradicts its timing fields")
    return device


def trace_annotation_evidence(trace: dict, summary: dict) -> dict:
    """Match CPU scopes to CUDA projections by category and rank-local external ID."""
    if type(trace.get("schemaVersion")) is not int or trace["schemaVersion"] != 1:
        raise ValueError("Stage-profile trace has an unsupported schema")
    scopes = defaultdict(list)
    for event in trace["traceEvents"]:
        if not event.get("name", "").startswith("osn/"):
            continue
        device = {"user_annotation": "CPU", "gpu_user_annotation": "CUDA"}.get(event.get("cat"))
        if (
            device is None
            or event.get("ph") != "X"
            or not valid_trace_interval(event)
            or any(type(event.get(key)) is not int for key in ("pid", "tid"))
            or type(event.get("args", {}).get("External id")) is not int
        ):
            raise ValueError("Stage-profile annotation category, interval or identity differs")
        scopes[event["name"], device].append(event)
    regions = {}
    for region in summary["regions"]:
        key = region["name"], trace_region_device_type(region)
        if key in regions or type(region.get("calls")) is not int or region["calls"] <= 0:
            raise ValueError("Stage-profile summary repeats a device key or has invalid calls")
        regions[key] = region
    if set(regions) != set(scopes) or any(regions[key]["calls"] != len(events) for key, events in scopes.items()):
        raise ValueError("Stage-profile summary calls differ from typed trace annotations")
    cpu = {}
    for (name, device), events in scopes.items():
        if device != "CPU":
            continue
        for event in events:
            key = name, event["args"]["External id"]
            if key in cpu:
                raise ValueError("Stage-profile CPU annotation external ID repeats")
            cpu[key] = event
    projected = set()
    for (name, device), events in scopes.items():
        if device != "CUDA":
            continue
        for event in events:
            key = name, event["args"]["External id"]
            lane_key = (*key, event["pid"], event["tid"])
            if key not in cpu or lane_key in projected:
                raise ValueError("Stage-profile CUDA projection has no unique matching CPU annotation")
            projected.add(lane_key)
    count = ENDPOINT - BOUNDARY
    updates = sorted(scopes["osn/training_update", "CPU"], key=lambda event: event["ts"])
    optimizers = sorted(scopes["osn/optimizer_step", "CPU"], key=lambda event: event["ts"])
    if (
        len(updates) != count
        or len(optimizers) != count
        or len({(event["pid"], event["tid"]) for event in updates + optimizers}) != 1
        or any(left["ts"] >= right["ts"] or left["ts"] + left["dur"] > right["ts"] for left, right in pairwise(updates))
        or any(
            optimizer["ts"] < update["ts"] or optimizer["ts"] + optimizer["dur"] > update["ts"] + update["dur"]
            for update, optimizer in zip(updates, optimizers, strict=True)
        )
    ):
        raise ValueError("Stage-profile CPU update or optimizer spans are incomplete or overlap")
    kernels = [event for event in trace["traceEvents"] if event.get("cat") == "kernel"]
    if not kernels or any(
        event.get("ph") != "X"
        or not valid_trace_interval(event)
        or any(type(event.get(key)) is not int for key in ("pid", "tid"))
        for event in kernels
    ):
        raise ValueError("Stage-profile CUDA kernel coverage or intervals differ")
    lanes = defaultdict(list)
    for event in kernels:
        lanes[event["pid"], event["tid"]].append((event["ts"], event["ts"] + event["dur"]))
    lanes = {key: sorted(intervals) for key, intervals in lanes.items()}
    starts = {key: [start for start, _ in intervals] for key, intervals in lanes.items()}
    for name, events in (("osn/training_update", updates), ("osn/optimizer_step", optimizers)):
        cuda = scopes[name, "CUDA"]
        matches = Counter(event["args"]["External id"] for event in cuda)
        if len(cuda) != count or matches != Counter(event["args"]["External id"] for event in events):
            raise ValueError("Stage-profile CUDA update or optimizer projection coverage differs")
        parents = {event["args"]["External id"]: update for event, update in zip(events, updates, strict=True)}
        for event in cuda:
            update = parents[event["args"]["External id"]]
            lane = event["pid"], event["tid"]
            index = bisect_left(starts.get(lane, []), event["ts"])
            if (
                event["ts"] < update["ts"]
                or event["ts"] + event["dur"] > update["ts"] + update["dur"]
                or index == len(lanes.get(lane, []))
                or lanes[lane][index][1] > event["ts"] + event["dur"]
            ):
                raise ValueError("Stage-profile CUDA projection escapes its update or lacks a matched-lane kernel")
    return {
        "trace_schema_version": trace["schemaVersion"],
        "training_update_events": len(updates),
        "optimizer_step_events": len(optimizers),
        "cuda_training_update_events": len(scopes["osn/training_update", "CUDA"]),
        "cuda_optimizer_step_events": len(scopes["osn/optimizer_step", "CUDA"]),
        "device_kernel_events": len(kernels),
    }


def rank_trace_evidence(directory: Path, config: dict, identity: dict) -> dict:
    """Require all eight complete CPU/CUDA traces with matching update counts and provenance."""
    profiles = directory / "profiles"
    if profiles.is_symlink() or {path.name for path in profiles.iterdir()} != {f"rank{rank}" for rank in range(WORLD)}:
        raise ValueError("Stage-profile trace rank inventory differs")
    evidence = {}
    for rank in range(WORLD):
        output = profiles / f"rank{rank}"
        summary_path, trace_path = output / "summary.json", output / "trace.json"
        if output.is_symlink() or any(path.is_symlink() or not path.is_file() for path in (summary_path, trace_path)):
            raise ValueError("Stage-profile trace contains aliased or missing evidence")
        summary, trace = read(summary_path), read(trace_path)
        provenance = {"arm": config["experiment"]["arm"], "config": config, "checkpoint_identity": identity}
        if (
            summary["status"] != "captured"
            or type(summary["rank"]) is not int
            or summary["rank"] != rank
            or summary["activities"] != ["CPU", "CUDA"]
            or summary["first_update"] != PROFILE_UPDATES[0]
            or summary["last_update"] != PROFILE_UPDATES[1]
            or summary["observed_updates"] != list(range(PROFILE_UPDATES[0], PROFILE_UPDATES[1] + 1))
            or summary["provenance"] != provenance
            or trace.get("osn_provenance") != provenance
            or summary["trace_sha256"] != sha256(trace_path)
        ):
            raise ValueError(f"Stage-profile rank {rank} trace coverage or provenance differs")
        annotations = trace_annotation_evidence(trace, summary)
        evidence[str(rank)] = {
            "summary_sha256": sha256(summary_path),
            "trace_sha256": sha256(trace_path),
            **annotations,
        }
    return evidence


def loss_continuation_evidence(outputs: dict[str, Path]) -> dict:
    """Compare numerical loss observations across all three branches, excluding timing metrics."""
    runs = {mode: history(output) for mode, output in outputs.items()}
    expected = runs["continuous"].rows
    result = {}
    for mode in ("off_resumed", "on_resumed"):
        observed = runs[mode].rows
        if len(expected) != ENDPOINT or len(observed) != ENDPOINT:
            raise ValueError("Stage-profile loss continuation has incomplete trajectory coverage")
        errors = []
        for reference, actual in zip(expected, observed, strict=True):
            for key in ("train_loss", "c4_loss", "selection_loss"):
                if (key in reference) != (key in actual):
                    raise ValueError("Stage-profile loss continuation has different evaluation coverage")
                if key not in reference:
                    continue
                if not math.isclose(actual[key], reference[key], rel_tol=1e-5, abs_tol=1e-6):
                    raise ValueError("Stage-profile loss continuation differs beyond unchanged numerical tolerances")
                errors.append(abs(actual[key] - reference[key]))
        result[mode] = {"scalar_comparisons": len(errors), "max_absolute_error": max(errors, default=0.0)}
    return result


def trial_evidence(item: dict, candidate: Path, frozen: dict) -> dict:
    """Recompute all-rank continuation and trace checks from the three owned trajectories."""
    validation, records, _ = numerical_modules(candidate)
    config = read(Path(item["config"]))
    outputs = {mode: Path(item["outputs"][mode]) for mode in MODES}
    evidence = {
        "trajectory_identity": {
            mode: trajectory_evidence(output, config, candidate, frozen, resumed=mode != "continuous")
            for mode, output in outputs.items()
        },
        "owned_prefixes": {
            mode: require_resume_prefix(outputs[mode], config, candidate, frozen)
            for mode in ("off_resumed", "on_resumed")
        },
        "cached_root_checkpoints": {
            mode: validation.require_cached_root_checkpoint(output / f"checkpoints/step_{BOUNDARY:06d}", WORLD)
            for mode, output in outputs.items()
        },
        "boundary_continuation": {
            mode: records.compare_continuation(
                outputs["continuous"] / f"checkpoints/step_{BOUNDARY:06d}",
                outputs[mode] / f"checkpoints/step_{BOUNDARY:06d}",
                WORLD,
            )
            for mode in ("off_resumed", "on_resumed")
        },
        "endpoint_continuation": {
            mode: records.compare_continuation(
                outputs["continuous"] / f"checkpoints/step_{ENDPOINT:06d}",
                outputs[mode] / f"checkpoints/step_{ENDPOINT:06d}",
                WORLD,
            )
            for mode in ("off_resumed", "on_resumed")
        },
        "profile_on_off_continuation": records.compare_continuation(
            outputs["off_resumed"] / f"checkpoints/step_{ENDPOINT:06d}",
            outputs["on_resumed"] / f"checkpoints/step_{ENDPOINT:06d}",
            WORLD,
        ),
        "loss_continuation": loss_continuation_evidence(outputs),
    }
    if any((outputs[mode] / "profiles").exists() for mode in ("continuous", "off_resumed")):
        raise ValueError("Stage-profile off trajectory unexpectedly contains profiling output")
    identity = records.checkpoint_identity(config, Path(frozen["root"]) / "data/experiment_receipt.json")
    evidence["rank_traces"] = rank_trace_evidence(outputs["on_resumed"], config, identity)
    return evidence
