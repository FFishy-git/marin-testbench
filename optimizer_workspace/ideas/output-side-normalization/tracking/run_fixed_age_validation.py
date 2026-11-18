# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze and guard actual short-LR GPU validation after complete, verified Wave-7 results."""

import argparse
import json
import math
import os
import subprocess
import tarfile
from pathlib import Path

import boundary_handoff
from analyze_tuning import executed_sources, normalized_sources
from compare_runs import WINDOW, compare, completion, history
from fixed_age_protocol import environment, is_boundary, predecessor_wave, require_gpu_environment
from launch_wave5 import NODES, PYTHON, node_action, read, sha256
from prepare_fixed_age_tuning import require_source
from validate_fixed_age_tuning import executed_trajectory, plan
from validate_wave5 import ENVIRONMENT
from watch_wave5 import process, write
from wave7_release import ARMS, HELPERS, bounded_execution, idle_training, local_node, node_guard, release_keeper

EXTRA_HELPERS = (
    "run_fixed_age_validation.py",
    "validate_fixed_age_tuning.py",
    "analyze_fixed_age_tuning.py",
    "prepare_fixed_age_tuning.py",
    "analyze_tuning.py",
    "prepare_tuning.py",
    "fixed_age_protocol.py",
    "analyze_lr_boundary.py",
    "prepare_lr_boundary.py",
    "boundary_handoff.py",
    "prepare_lr_ceiling.py",
    "analyze_lr_ceiling.py",
    "upstream_validation.py",
)
VALIDATION_HELPERS = (*HELPERS, *EXTRA_HELPERS)
NODE = NODES[0]
COMPARISONS = (
    ("wave7/muont", tuple(f"wave7/{arm}" for arm in ARMS[1:])),
    ("wave7/inner_outer_leftdiag", ("wave7/outer_leftdiag",)),
    ("wave7/outer_leftdiag", ("wave7/outer_leftdiag_fused",)),
)


def require_original_execution(root: Path, frozen: dict) -> None:
    """Check unchanged numerical command logs and exact configuration/environment coverage."""
    actual = read(root / "run-preparation/validation-wave7-gpu.json")
    candidate = Path(frozen["candidate"])
    commands = actual["evidence"]["gpu_commands"]
    if (
        actual["data_receipt_sha256"] != frozen["data_receipt_sha256"]
        or sha256(root / "data/experiment_receipt.json") != frozen["data_receipt_sha256"]
        or actual["environment"] != ENVIRONMENT
        or actual["config_sha256"]
        != {arm: sha256(candidate / f"experiment/configs/{arm}_0p5m_seed0.json") for arm in ARMS}
        or len(commands) != 13
        or any(item["exit_code"] != 0 or sha256(Path(item["log"])) != item["log_sha256"] for item in commands)
    ):
        raise ValueError("Original numerical execution logs, configs or data/environment evidence changed")


def freeze(campaign_path: Path) -> dict:
    """Archive a new validation controller without creating a GPU job or queue."""
    campaign = read(campaign_path)
    root = Path(campaign["root"])
    frozen = require_source(root, Path(campaign["source_receipt"]))
    require_original_execution(root, frozen)
    directory = campaign_path.parent
    binding_path = directory / "validation-binding.json"
    helpers = directory / "validation-helpers"
    archive = directory / "validation-source.tar"
    if any(path.exists() for path in (binding_path, helpers, archive)):
        raise FileExistsError("Short-config validation binding already exists; inspect it instead of overwriting")
    work = directory / "validation"
    execution_plan = plan(campaign_path, work)
    wave = predecessor_wave(campaign_path)
    predecessor = (
        boundary_handoff.release(root, wave=wave)
        if wave == "wave9"
        else boundary_handoff.release(root) if is_boundary(campaign_path) else None
    )
    helpers.mkdir()
    with tarfile.open(archive, "x") as stream:
        for name in VALIDATION_HELPERS:
            source = Path(__file__).parent / name
            destination = helpers / name
            destination.write_bytes(source.read_bytes())
            stream.add(destination, arcname=f"tracking/{name}", recursive=False)
        stream.add(campaign_path, arcname="campaign.json", recursive=False)
        for trial in campaign["trials"]:
            stream.add(Path(trial["config"]), arcname=f"configs/{trial['name']}.json", recursive=False)
    result = {
        "status": (
            f"frozen_pending_completed_{wave}_and_actual_gpu_validation"
            if predecessor
            else "frozen_pending_completed_wave7_and_actual_gpu_validation"
        ),
        "campaign": str(campaign_path.resolve()),
        "campaign_sha256": sha256(campaign_path),
        "source_release_sha256": sha256(Path(campaign["source_receipt"])),
        "original_gpu_validation_sha256": frozen["original_gpu_validation_sha256"],
        "helpers": str(helpers.resolve()),
        "helpers_sha256": {name: sha256(helpers / name) for name in VALIDATION_HELPERS},
        "archive": str(archive.resolve()),
        "archive_sha256": sha256(archive),
        "work": str(work.resolve()),
        "receipt": str((directory / "validation-gpu.json").resolve()),
        "node": NODE,
        "environment": environment(campaign_path),
        "plan": execution_plan,
        "launch_condition": campaign["launch_condition"],
    }
    if predecessor:
        result["predecessor_release_sha256"] = sha256(root / f"run-preparation/{wave}-launch-binding.json")
    write(binding_path, result)
    return result


def binding(campaign_path: Path) -> dict:
    """Require the exact archived helper copy, plan, campaign and validated numerical source."""
    campaign = read(campaign_path)
    root = Path(campaign["root"])
    source = require_source(root, Path(campaign["source_receipt"]))
    require_original_execution(root, source)
    frozen = read(campaign_path.parent / "validation-binding.json")
    if is_boundary(campaign_path):
        wave = predecessor_wave(campaign_path)
        if wave == "wave9":
            boundary_handoff.release(root, wave=wave)
        else:
            boundary_handoff.release(root)
        if frozen.get("predecessor_release_sha256") != sha256(root / f"run-preparation/{wave}-launch-binding.json"):
            raise ValueError(f"Boundary validation has a different {wave.replace('wave', 'Wave-')} predecessor release")
    helpers = Path(frozen["helpers"])
    if (
        Path(__file__).resolve() != (helpers / "run_fixed_age_validation.py").resolve()
        or frozen["campaign"] != str(campaign_path.resolve())
        or frozen["campaign_sha256"] != sha256(campaign_path)
        or frozen["source_release_sha256"] != sha256(Path(campaign["source_receipt"]))
        or set(frozen["helpers_sha256"]) != set(VALIDATION_HELPERS)
        or any(sha256(helpers / name) != value for name, value in frozen["helpers_sha256"].items())
        or sha256(Path(frozen["archive"])) != frozen["archive_sha256"]
        or frozen["plan"] != plan(campaign_path, Path(frozen["work"]))
        or frozen["node"] != NODE
        or frozen["environment"] != environment(campaign_path)
        or frozen["original_gpu_validation_sha256"] != source["original_gpu_validation_sha256"]
        or frozen["work"] != str((campaign_path.parent / "validation").resolve())
        or frozen["receipt"] != str((campaign_path.parent / "validation-gpu.json").resolve())
    ):
        raise ValueError("Actual short-config validation source, helper, archive or plan identity changed")
    return frozen


def require_actual_receipt(frozen: dict) -> dict:
    """Reject missing trials, altered commands or empty continuation evidence before release."""
    actual = read(Path(frozen["receipt"]))
    expected = frozen["plan"]
    campaign = read(Path(frozen["campaign"]))
    root, source = Path(campaign["root"]), Path(campaign["source"])
    numerical = require_source(root, Path(campaign["source_receipt"]))
    keys = (
        "campaign",
        "campaign_sha256",
        "validator_sha256",
        "source_release_sha256",
        "source_sha256",
        "data_receipt_sha256",
        "environment",
        "world",
        "continuous_steps",
        "interrupted_steps",
        "resumed_steps",
        "continuation_tolerance",
        "scope",
        "planned_validation_tokens",
    )
    if (
        actual["status"] != "passed_actual_short_config_gates"
        or any(actual[key] != expected[key] for key in keys)
        or len(actual["trials"]) != len(expected["trials"])
        or len(actual["gpu_commands"]) != 3 * len(expected["trials"])
    ):
        raise ValueError("Actual short-configuration receipt has incomplete or changed execution coverage")
    for item, planned in zip(actual["trials"], expected["trials"], strict=True):
        if (
            any(
                item[key] != planned[key]
                for key in ("name", "family", "config", "config_sha256", "commands", "continuous", "resumed")
            )
            or item["checks"] != dict.fromkeys(planned["checks"], True)
            or item["cached_root_checkpoint"]["ranks"] != 8
            or item["cached_root_checkpoint"]["owned_graph_groups"] <= 0
        ):
            raise ValueError("Actual short-configuration trial checks or identity are incomplete")
        for key in ("boundary_continuation", "post_cached_root_continuation"):
            comparison = item[key]
            if (
                type(comparison["tensor_comparisons"]) is not int
                or comparison["tensor_comparisons"] <= 0
                or not math.isfinite(comparison["max_absolute_error"])
                or comparison["max_absolute_error"] < 0
            ):
                raise ValueError("All-rank numerical continuation has no valid tensor evidence")
        for trajectory in ("continuous", "resumed"):
            directory = Path(item[trajectory])
            identity = item["trajectory_identity"][trajectory]
            scheduling = item["fixed_age_history"][trajectory]
            if (
                identity != executed_trajectory(directory, read(Path(item["config"])), source, numerical)
                or scheduling["rank"] != 0
                or scheduling["updates"] != 24
                or scheduling["installation_age"] != 1
                or scheduling["observed_installations"] <= 0
            ):
                raise ValueError("Actual short-configuration trajectory or scheduling evidence changed")
            if item["family"].endswith("_fused") and item["capture_history"][trajectory] != {
                "updates": 24,
                "captured_rows_and_outer_updates": True,
            }:
                raise ValueError("Actual fused short-configuration captured update evidence is incomplete")
    commands = [
        (item["name"], index, command) for item in expected["trials"] for index, command in enumerate(item["commands"])
    ]
    for execution, (trial, index, command) in zip(actual["gpu_commands"], commands, strict=True):
        log = Path(frozen["work"]) / f"{trial}_{('continuous', 'interrupted', 'resumed')[index]}.log"
        if (
            execution["trial"] != trial
            or execution["command"] != command
            or execution["exit_code"] != 0
            or Path(execution["log"]) != log
            or execution["log_sha256"] != sha256(log)
        ):
            raise ValueError("Actual short-configuration command or log execution identity changed")
    return actual


def require_completed_wave(root: Path, *, campaign_path: Path | None = None) -> dict:
    """Reconstruct all five windows and verify local identities against complete cloud receipts."""
    if campaign_path is not None and is_boundary(campaign_path):
        if predecessor_wave(campaign_path) == "wave9":
            return boundary_handoff.require_completed(root, wave="wave9")
        return boundary_handoff.require_completed(root)
    release_path = root / "run-preparation/wave7-release-binding.json"
    frozen = require_source(root, release_path)
    candidate = Path(frozen["candidate"])
    cloud = read(root / "run-preparation/swanlab-upload-wave7.json")
    if (
        cloud["status"] != "verified"
        or set(cloud["runs"]) != set(ARMS)
        or cloud["visibility"] != "PRIVATE"
        or not cloud["project"].endswith("/output-side-normalization-marin")
    ):
        raise ValueError("All four completed Wave-7 histories must be cloud-verified before short-LR validation")
    runs = {}
    for arm in ARMS:
        directory = root / "runs/wave7" / arm
        run = history(directory)
        expected = read(candidate / f"experiment/configs/{arm}_0p5m_seed0.json")
        if (
            run.manifest["config"] != expected
            or run.manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
            or normalized_sources(candidate, run.manifest["source_sha256"]) != executed_sources(candidate, frozen)
            or completion(run)["complete"] is not True
        ):
            raise ValueError("Wave-7 predecessor is incomplete or differs from its executed source/config/data")
        checkpoint = directory / "checkpoints/step_005120"
        receipt = read(checkpoint / "complete.json")
        rank_sizes = [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)]
        imported = cloud["runs"][arm]
        verification = imported["verification"]
        if (
            receipt["step"] != 5120
            or receipt["world"] != 8
            or not all(size > 0 for size in rank_sizes)
            or imported["status"] != "verified"
            or imported["manifest_sha256"] != sha256(directory / "run.json")
            or imported["metrics_sha256"] != sha256(directory / "metrics_from_000000.jsonl")
            or imported["checkpoint_receipt_sha256"] != sha256(checkpoint / "complete.json")
            or imported["checkpoint_rank_bytes"] != rank_sizes
            or verification["state"] != "FINISHED"
            or verification["configuration_verified"] is not True
            or verification["token_axis_verified"] is not True
            or verification["integer_values_exact"] is not True
            or verification["point_counts"]["train/loss"] != 5120
            or verification["point_counts"]["tokens"] != 5120
        ):
            raise ValueError("Completed Wave-7 cloud or final checkpoint identities are stale or incomplete")
        runs[f"wave7/{arm}"] = run
    fresh = {
        "wave": "wave7",
        "window_tokens": WINDOW,
        "analyses": [
            compare({name: runs[name] for name in (baseline, *names)}, baseline, WINDOW)
            for baseline, names in COMPARISONS
        ],
        "unpaired_runs": {},
    }
    if any(
        item["final_comparison_eligible"] is not True or len(item["training_windows"]) != 5
        for pair in fresh["analyses"]
        for item in pair["comparisons"].values()
    ) or read(root / "run-preparation/analysis-wave7.json") != json.loads(json.dumps(fresh, allow_nan=False)):
        raise ValueError("Wave-7 five-window analysis is incomplete or stale for its verified histories")
    return {
        "analysis_sha256": sha256(root / "run-preparation/analysis-wave7.json"),
        "cloud_sha256": sha256(root / "run-preparation/swanlab-upload-wave7.json"),
    }


def execute(campaign_path: Path) -> dict:
    """Transfer only the verified keeper generation and restore it after bounded validation."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("CPU-only CUDA visibility prevents actual-config GPU validation")
    require_gpu_environment(campaign_path)
    frozen = binding(campaign_path)
    root = Path(read(campaign_path)["root"])
    if local_node() != NODE:
        raise RuntimeError("Actual short-LR validation must execute on the designated .9 node")
    state_path = campaign_path.parent / "validation-state.json"
    if any(path.exists() for path in (state_path, Path(frozen["receipt"]), Path(frozen["work"]))):
        raise FileExistsError("Actual validation already has execution evidence; preserve it without retry")
    with node_guard(root, NODE) as signals:
        predecessor = require_completed_wave(root, campaign_path=campaign_path)
        wave = predecessor_wave(campaign_path)
        observation = campaign_path.parent / f"{wave}-prevalidation-snapshot.json"
        if is_boundary(campaign_path):
            if wave == "wave9":
                boundary_handoff.snapshot(root, observation, wave=wave)
            else:
                boundary_handoff.snapshot(root, observation)
        else:
            source = require_source(root, root / "run-preparation/wave7-release-binding.json")
            subprocess.run(
                [
                    PYTHON,
                    str(Path(source["helpers"]) / "snapshot_wave7_release.py"),
                    "--root",
                    str(root),
                    "--out",
                    str(observation),
                ],
                check=True,
                timeout=240,
            )
            if read(observation)["status"] != "complete":
                raise RuntimeError("Wave-7 parents or keeper restoration remain live, unknown or incomplete")
        idle_training()
        devices = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()
        if len(devices) != 8 or any("A100-SXM4-40GB" not in device for device in devices):
            raise ValueError("Actual short-LR validation requires the original eight A100 40GB allocation")
        subprocess.run(
            [PYTHON, "-c", "import torch; assert torch.cuda.device_count() == 8, 'Expected eight visible CUDA GPUs'"],
            env=os.environ | frozen["environment"],
            check=True,
            timeout=60,
        )
        subprocess.run(
            [PYTHON, "-m", "experiment.bootstrap", "--check-only"], cwd=Path(read(campaign_path)["source"]), check=True
        )
        state = {
            "status": "preflight",
            "controller_pid": os.getpid(),
            "controller_start_ticks": process(os.getpid())[1],
            "binding_sha256": sha256(campaign_path.parent / "validation-binding.json"),
            "predecessor": predecessor,
            "predecessor_snapshot_sha256": sha256(observation),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        write(state_path, state)
        try:
            state["keeper"] = release_keeper(state, state_path)
            state["status"] = "validating"
            command = [
                PYTHON,
                str(Path(frozen["helpers"]) / "validate_fixed_age_tuning.py"),
                "--campaign",
                str(campaign_path),
                "--work",
                frozen["work"],
                "--receipt",
                frozen["receipt"],
            ]
            code = bounded_execution(
                command,
                Path(read(campaign_path)["source"]),
                campaign_path.parent / "validation.log",
                state,
                state_path,
                timeout=14400,
            )
            state["exit_code"] = code
            if code:
                raise RuntimeError("Actual short-configuration GPU gates did not pass")
            require_actual_receipt(frozen)
            binding(campaign_path)
            state.update(status="passed", validation_sha256=sha256(Path(frozen["receipt"])))
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if state.get("keeper_release_attempted"):
                try:
                    if "cleanup_error" in state:
                        raise RuntimeError("Unverified child cleanup prevents keeper restoration")
                    node_action(root, "restore")
                    state["keeper_restore"] = 0
                except Exception as error:
                    state.update(status="needs_attention", keeper_restore=f"refused: {error}")
            if signals:
                state["deferred_signals"] = signals
            write(state_path, state)
        if type(state.get("keeper_restore")) is not int or state["keeper_restore"] != 0:
            raise RuntimeError("Short-configuration validation keeper restoration failed")
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "check", "validate"), required=True)
    args = parser.parse_args()
    if args.phase == "freeze":
        result = freeze(args.campaign)
    elif args.phase == "check":
        result = binding(args.campaign)
    else:
        result = execute(args.campaign)
    print(json.dumps({"status": result["status"], "campaign": str(args.campaign)}, indent=2))


if __name__ == "__main__":
    main()
