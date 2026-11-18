# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise private checkpoint-prefix copies and complete rank trace reconstruction on CPU."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from analyze_tuning import executed_sources
from launch_wave5 import read, sha256
from selected_lr_stage_profile_validation import (
    ARMS,
    BOUNDARY,
    ENDPOINT,
    PREFIX,
    PREFIX_MANIFEST,
    WORLD,
    locations,
    loss_continuation_evidence,
    plan,
    rank_trace_evidence,
    require_resume_prefix,
    seed_resume_prefix,
    trajectory_evidence,
)
from wave7_release import candidate_inventory

TASK = Path(__file__).resolve().parents[1]


@pytest.fixture
def owned_trajectory(tmp_path):
    candidate = tmp_path / "candidate"
    configs = candidate / "experiment/configs"
    configs.mkdir(parents=True)
    candidate.joinpath("experiment/train.py").write_text("def fixture_training(): return 1\n")
    candidate.joinpath("experiment/upstream_manifest.json").write_text('{"revision":"fixture","files":{}}')
    pinned = candidate / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    candidate.joinpath("artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    config = read(TASK / "experiment/configs/muont_0p5m_seed0.json")
    config["experiment"]["refresh_install_age"] = 1
    for arm in ARMS:
        arm_config = json.loads(json.dumps(config))
        arm_config["experiment"]["arm"] = arm
        configs.joinpath(f"{arm}_0p5m_seed0.json").write_text(json.dumps(arm_config))
    frozen = {"files_sha256": candidate_inventory(candidate), "data_receipt_sha256": "fixture-data"}
    identity = {
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "source_sha256": executed_sources(candidate, frozen),
        "data_receipt_sha256": frozen["data_receipt_sha256"],
    }
    continuous = tmp_path / "continuous"
    continuous.mkdir()
    manifest = {
        "status": "execution_check_complete",
        "config": config,
        "source_sha256": identity["source_sha256"],
        "data_receipt_sha256": frozen["data_receipt_sha256"],
        "execution_check": True,
        "completed_steps": ENDPOINT,
        "resume": None,
    }
    continuous.joinpath("run.json").write_text(json.dumps(manifest))
    continuous.joinpath("config.json").write_text(json.dumps(config))
    rows = [
        {
            "step": step,
            "tokens": step * config["training"]["batch_tokens"],
            "train_loss": 4.0 - step / 1000,
            "input_refresh_installations": (
                [{"installation_step": step - 1, "source_step": step - 2}] if step >= 2 else []
            ),
        }
        for step in range(1, ENDPOINT + 1)
    ]
    continuous.joinpath("metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for step in (BOUNDARY, ENDPOINT):
        checkpoint = continuous / f"checkpoints/step_{step:06d}"
        checkpoint.mkdir(parents=True)
        checkpoint.joinpath("complete.json").write_text(json.dumps({"step": step, "world": WORLD, "time": "fixture"}))
        for rank in range(WORLD):
            state = {
                "identity": identity,
                "step": step,
                "rank": rank,
                "world": WORLD,
                "config": config,
                "next_global_sequence": step * config["training"]["global_batch_sequences"],
                "optimizers": [{"momentum": torch.tensor([rank, step], dtype=torch.float32)}],
                "fast": {"steps": step, "pending": None, "root": torch.eye(2)},
                "output": None,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": None,
                "python_rng": (3, (rank, step), None),
            }
            if rank == 0:
                state["model"] = {"weight": torch.eye(2)}
            torch.save(state, checkpoint / f"rank_{rank}.pt")
    return candidate, frozen, config, continuous, identity


def test_owned_prefixes_are_independent_and_survive_resume_manifest_replacement(owned_trajectory, tmp_path):
    candidate, frozen, config, continuous, _ = owned_trajectory
    original = continuous.joinpath("metrics_from_000000.jsonl").read_bytes()
    off, on = tmp_path / "off", tmp_path / "on"
    for output in (off, on):
        receipt = seed_resume_prefix(continuous, output, config, candidate, frozen)
        assert receipt["status"] == "stage_profile_owned_prefix_seeded"
        assert len(output.joinpath("metrics_from_000000.jsonl").read_text().splitlines()) == BOUNDARY
        for rank in range(WORLD):
            relative = f"checkpoints/step_{BOUNDARY:06d}/rank_{rank}.pt"
            assert sha256(output / relative) == sha256(continuous / relative)
            assert os.stat(output / relative).st_ino != os.stat(continuous / relative).st_ino
        manifest = read(continuous / "run.json")
        manifest["resume"] = str(output / f"checkpoints/step_{BOUNDARY:06d}")
        output.joinpath("run.json").write_text(json.dumps(manifest))
        output.joinpath(f"metrics_from_{BOUNDARY:06d}.jsonl").write_bytes(
            b"".join(original.splitlines(keepends=True)[BOUNDARY:])
        )
        shutil.copytree(continuous / f"checkpoints/step_{ENDPOINT:06d}", output / f"checkpoints/step_{ENDPOINT:06d}")
        assert require_resume_prefix(output, config, candidate, frozen) == receipt
        result = trajectory_evidence(output, config, candidate, frozen, resumed=True)
        assert result["observed_updates"] == ENDPOINT
        assert result["fixed_age_history"]["observed_installations"] == ENDPOINT - 1
        with pytest.raises(FileExistsError):
            seed_resume_prefix(continuous, output, config, candidate, frozen)
    assert read(off / PREFIX_MANIFEST)["status"] == "stage_profile_owned_prefix_prepared"
    assert continuous.joinpath("metrics_from_000000.jsonl").read_bytes() == original
    assert read(on / PREFIX)["checkpoint_sha256"] == read(off / PREFIX)["checkpoint_sha256"]


@pytest.mark.parametrize(
    "fault",
    (
        "rank7_position",
        "rank7_copy",
        "rank7_state_fields",
        "source_state_fields",
        "prefix_row",
        "prefix_alias",
        "prefix_hardlink",
        "source_history",
    ),
)
def test_changed_owned_checkpoint_or_prefix_cannot_be_reaccepted(owned_trajectory, tmp_path, fault):
    candidate, frozen, config, continuous, _ = owned_trajectory
    output = tmp_path / "resume"
    seed_resume_prefix(continuous, output, config, candidate, frozen)
    if fault in ("rank7_position", "rank7_copy", "rank7_state_fields", "source_state_fields"):
        parent = continuous if fault in ("rank7_position", "source_state_fields") else output
        path = parent / f"checkpoints/step_{BOUNDARY:06d}/rank_7.pt"
        state = torch.load(path, weights_only=False)
        if fault in ("rank7_state_fields", "source_state_fields"):
            del state["python_rng"]
        else:
            state["next_global_sequence"] += 1
        torch.save(state, path)
    elif fault in ("prefix_alias", "prefix_hardlink"):
        path = output / "metrics_from_000000.jsonl"
        raw = path.read_bytes()
        path.unlink()
        target = tmp_path / "aliased-history"
        target.write_bytes(raw)
        if fault == "prefix_alias":
            path.symlink_to(target)
        else:
            os.link(target, path)
    else:
        parent = continuous if fault == "source_history" else output
        path = parent / "metrics_from_000000.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[10]["train_loss"] += 0.1
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        require_resume_prefix(output, config, candidate, frozen)


def test_resumed_loss_drift_is_rejected_even_with_unchanged_checkpoint_prefix(owned_trajectory, tmp_path):
    candidate, frozen, config, continuous, _ = owned_trajectory
    outputs = {"continuous": continuous, "off_resumed": tmp_path / "off", "on_resumed": tmp_path / "on"}
    original = continuous.joinpath("metrics_from_000000.jsonl").read_bytes()
    for mode in ("off_resumed", "on_resumed"):
        output = outputs[mode]
        seed_resume_prefix(continuous, output, config, candidate, frozen)
        output.joinpath(f"metrics_from_{BOUNDARY:06d}.jsonl").write_bytes(
            b"".join(original.splitlines(keepends=True)[BOUNDARY:])
        )
    evidence = loss_continuation_evidence(outputs)
    assert evidence["on_resumed"] == {"scalar_comparisons": ENDPOINT, "max_absolute_error": 0.0}
    path = outputs["on_resumed"] / f"metrics_from_{BOUNDARY:06d}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["train_loss"] += 0.01
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        loss_continuation_evidence(outputs)
    assert require_resume_prefix(outputs["on_resumed"], config, candidate, frozen)["status"] == (
        "stage_profile_owned_prefix_seeded"
    )


def test_protocol_uses_private_resumes_and_retains_actual_schedule(owned_trajectory, tmp_path):
    candidate, _, config, _, _ = owned_trajectory
    root = tmp_path / "output-root"
    result = plan(candidate, root, locations(root)[3])
    assert len(result["trials"]) == len(ARMS)
    assert result["gpu_commands"] == []
    for item in result["trials"]:
        assert read(Path(item["config"]))["training"] == config["training"]
        for mode, command in zip(("continuous", "off_resumed", "on_resumed"), item["commands"], strict=True):
            assert command[command.index("--launch-release") + 1] == str(locations(root)[0])
            assert command[command.index("--execution-boundary") + 1] == "200"
            assert command[command.index("--max-steps") + 1] == "220"
            if mode == "continuous":
                assert "--resume" not in command and "--profile-updates" not in command
            else:
                assert command[command.index("--resume") + 1] == (f"{item['outputs'][mode]}/checkpoints/step_000200")
                assert ("--profile-updates" in command) == (mode == "on_resumed")
    assert result["planned_executed_tokens"] == 545259520


def trace_fixture(directory, config, identity):
    """Model issued CPU/CUDA duplicate labels and unordered GPU projections, not GPU execution."""
    for rank in range(WORLD):
        output = directory / "profiles" / f"rank{rank}"
        output.mkdir(parents=True)
        provenance = {"arm": config["experiment"]["arm"], "config": config, "checkpoint_identity": identity}
        events, projections = [], []
        for update in range(201, 221):
            for name, offset, duration in (("osn/training_update", 0, 80), ("osn/optimizer_step", 40, 10)):
                external = update * 2 + (name == "osn/optimizer_step")
                events.append(
                    {
                        "name": name,
                        "cat": "user_annotation",
                        "ph": "X",
                        "pid": 1000 + rank,
                        "tid": 1000 + rank,
                        "ts": update * 100.0 + offset,
                        "dur": duration,
                        "args": {"External id": external},
                    }
                )
                projections.append(
                    {
                        "name": name,
                        "cat": "gpu_user_annotation",
                        "ph": "X",
                        "pid": rank,
                        "tid": 7,
                        "ts": update * 100.0 + (1 if offset == 0 else 55),
                        "dur": 70 if offset == 0 else 10,
                        "args": {"External id": external},
                    }
                )
            events.append(
                {
                    "name": "fixture_kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "pid": rank,
                    "tid": 7,
                    "ts": update * 100.0 + 56,
                    "dur": 2,
                }
            )
        events.extend(reversed(projections))
        output.joinpath("trace.json").write_text(
            json.dumps({"schemaVersion": 1, "osn_provenance": provenance, "traceEvents": events})
        )
        summary = {
            "status": "captured",
            "rank": rank,
            "activities": ["CPU", "CUDA"],
            "first_update": 201,
            "last_update": 220,
            "observed_updates": list(range(201, 221)),
            "provenance": provenance,
            "trace_sha256": sha256(output / "trace.json"),
            "regions": [
                {
                    "name": name,
                    "calls": 20,
                    "cpu_total_microseconds": cpu,
                    "cpu_self_microseconds": -10 if cpu else 0,
                    "device_total_microseconds": device,
                    "device_self_microseconds": device if not cpu else 0,
                }
                for name, cpu, device in (
                    ("osn/training_update", 1600, 100),
                    ("osn/training_update", 0, 1400),
                    ("osn/optimizer_step", 200, 40),
                    ("osn/optimizer_step", 0, 200),
                )
            ],
        }
        output.joinpath("summary.json").write_text(json.dumps(summary))


@pytest.mark.parametrize("explicit_device_type", (False, True))
def test_trace_reconstruction_matches_duplicate_device_labels_and_unordered_projections(
    owned_trajectory, tmp_path, explicit_device_type
):
    _, _, config, _, identity = owned_trajectory
    output = tmp_path / "profiled"
    trace_fixture(output, config, identity)
    if explicit_device_type:
        for rank in range(WORLD):
            path = output / "profiles" / f"rank{rank}" / "summary.json"
            summary = read(path)
            for region in summary["regions"]:
                region["device_type"] = "CPU" if region["cpu_total_microseconds"] else "CUDA"
            path.write_text(json.dumps(summary))
    evidence = rank_trace_evidence(output, config, identity)
    assert set(evidence) == {str(rank) for rank in range(WORLD)}
    for rank, item in evidence.items():
        assert item == {
            "summary_sha256": sha256(output / "profiles" / f"rank{rank}" / "summary.json"),
            "trace_sha256": sha256(output / "profiles" / f"rank{rank}" / "trace.json"),
            "trace_schema_version": 1,
            "training_update_events": 20,
            "optimizer_step_events": 20,
            "cuda_training_update_events": 20,
            "cuda_optimizer_step_events": 20,
            "device_kernel_events": 20,
        }


@pytest.mark.parametrize(
    "fault",
    (
        "missing_rank",
        "wrong_provenance",
        "wrong_summary_provenance",
        "source_drift",
        "trace_hash_drift",
        "cpu_only",
        "missing_update",
        "missing_optimizer",
        "missing_cuda_update",
        "missing_cuda_optimizer",
        "duplicate_update",
        "duplicate_cuda_projection",
        "overlapping_updates",
        "optimizer_outside_update",
        "cuda_update_outside_update",
        "cuda_optimizer_outside_update",
        "unmatched_projection",
        "missing_kernels",
        "missing_update_kernel",
        "wrong_kernel_lane",
        "invalid_kernel_time",
        "invalid_update_time",
        "wrong_annotation_category",
        "wrong_schema",
        "wrong_cpu_calls",
        "wrong_cuda_calls",
        "duplicate_device_region",
        "contradictory_device_type",
    ),
)
def test_trace_reconstruction_rejects_incomplete_or_mismatched_rank_evidence(owned_trajectory, tmp_path, fault):
    _, _, config, _, identity = owned_trajectory
    output = tmp_path / "profiled"
    trace_fixture(output, config, identity)
    rank = output / "profiles/rank7"
    summary = read(rank / "summary.json")
    trace = read(rank / "trace.json")
    events = trace["traceEvents"]
    update = next(event for event in events if event["name"] == "osn/training_update")
    optimizer = next(event for event in events if event["name"] == "osn/optimizer_step")
    cuda_update = next(
        event for event in events if event["name"] == "osn/training_update" and event["cat"] == "gpu_user_annotation"
    )
    cuda_optimizer = next(
        event for event in events if event["name"] == "osn/optimizer_step" and event["cat"] == "gpu_user_annotation"
    )
    kernel = next(event for event in events if event["cat"] == "kernel")
    if fault == "missing_rank":
        rank.rename(output / "profiles/unexpected")
    elif fault == "cpu_only":
        summary["activities"] = ["CPU"]
    elif fault == "missing_update":
        events.remove(update)
    elif fault == "missing_optimizer":
        events.remove(optimizer)
    elif fault == "missing_cuda_update":
        events.remove(cuda_update)
    elif fault == "missing_cuda_optimizer":
        events.remove(cuda_optimizer)
    elif fault == "duplicate_update":
        events.append(dict(update))
    elif fault == "duplicate_cuda_projection":
        events.append(dict(cuda_update))
    elif fault == "overlapping_updates":
        update["dur"] = 101
    elif fault == "optimizer_outside_update":
        optimizer["dur"] = 41
    elif fault in ("cuda_update_outside_update", "cuda_optimizer_outside_update"):
        event = cuda_update if fault == "cuda_update_outside_update" else cuda_optimizer
        event["dur"] = 101
    elif fault == "unmatched_projection":
        cuda_update["args"]["External id"] = -1
    elif fault == "missing_kernels":
        events[:] = [event for event in events if event["cat"] != "kernel"]
    elif fault == "missing_update_kernel":
        events.remove(kernel)
    elif fault == "wrong_kernel_lane":
        kernel["tid"] = 8
    elif fault == "invalid_kernel_time":
        kernel["dur"] = -1
    elif fault == "invalid_update_time":
        update["ts"] = float("nan")
    elif fault == "wrong_annotation_category":
        update["cat"] = "cpu_op"
    elif fault == "wrong_schema":
        trace["schemaVersion"] = 2
    elif fault == "wrong_cpu_calls":
        summary["regions"][0]["calls"] = 19
    elif fault == "wrong_cuda_calls":
        summary["regions"][1]["calls"] = 19
    elif fault == "duplicate_device_region":
        summary["regions"].append(dict(summary["regions"][0]))
    elif fault == "contradictory_device_type":
        summary["regions"][0]["device_type"] = "CUDA"
    elif fault == "wrong_summary_provenance":
        summary["provenance"]["arm"] = "different-arm"
    elif fault == "source_drift":
        trace["osn_provenance"]["checkpoint_identity"]["source_sha256"]["experiment/train.py"] = "different-source"
    elif fault == "wrong_provenance":
        trace["osn_provenance"]["checkpoint_identity"]["data_receipt_sha256"] = "different-data"
    elif fault == "trace_hash_drift":
        kernel["name"] = "different_kernel"
    else:
        raise AssertionError(f"Unexpected trace fixture fault: {fault}")
    if fault != "missing_rank":
        rank.joinpath("trace.json").write_text(json.dumps(trace))
        if fault != "trace_hash_drift":
            summary["trace_sha256"] = sha256(rank / "trace.json")
        rank.joinpath("summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        rank_trace_evidence(output, config, identity)
