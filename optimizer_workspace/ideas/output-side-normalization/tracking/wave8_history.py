# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read source-bound Wave-8 histories without importing Torch or the training package."""

from pathlib import Path

from analyze_tuning import executed_sources, normalized_sources
from compare_runs import completion, history
from launch_wave5 import NODES, PYTHON, read, sha256
from validate_wave5 import ENVIRONMENT

ARMS = ("muont", "outer_leftdiag", "outer_leftdiag_fused", "outer_leftdiag_combined_fused")
EXTRA_HELPERS = (
    "wave8_release.py",
    "wave8_history.py",
    "analyze_wave8.py",
    "swanlab_wave8.py",
    "monitor_wave8.sh",
    "time_to_target.py",
)


def paths(root: Path) -> tuple[Path, Path]:
    return root / "run-preparation/wave8-launch-binding.json", root / "controllers/wave8-launch-20261117/tracking"


def commands(root: Path, candidate: Path) -> dict[str, list[str]]:
    release_path, _ = paths(root)
    return {
        node: [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "-m",
            "experiment.train",
            "--config",
            str(candidate / f"experiment/configs/{arm}_0p5m_seed0.json"),
            "--data",
            str(root / "data"),
            "--data-receipt",
            str(root / "data/experiment_receipt.json"),
            "--out",
            str(root / "runs/wave8" / arm),
            "--launch-release",
            str(release_path),
        ]
        for node, arm in zip(NODES, ARMS, strict=True)
    }


def helper_binding(root: Path) -> dict:
    """Retain trustworthy process probes and restoration even when numerical evidence changes."""
    release_path, helpers = paths(root)
    release = read(release_path)
    candidate_path = root / "run-preparation/wave8-candidate.json"
    numerical = read(candidate_path)
    original_helpers = numerical["helpers_sha256"]
    if (
        Path(__file__).resolve() != (helpers / "wave8_history.py").resolve()
        or release["status"] != "frozen_validated_launch_release"
        or release["root"] != str(root)
        or release["candidate"] != numerical["candidate"]
        or release["helpers"] != str(helpers)
        or set(release["helpers_sha256"]) != set(original_helpers) | set(EXTRA_HELPERS)
        or any(sha256(helpers / name) != digest for name, digest in release["helpers_sha256"].items())
        or any(release["helpers_sha256"][name] != digest for name, digest in original_helpers.items())
        or release["archive"] != str(root / "run-preparation/source-wave8-launch-20261117.tar")
        or release["archive_sha256"] != sha256(Path(release["archive"]))
        or release["candidate_binding_sha256"] != sha256(candidate_path)
        or release["environment"] != ENVIRONMENT
        or tuple(release["arms"]) != ARMS
        or tuple(release["nodes"]) != NODES
        or release["commands"] != commands(root, Path(numerical["candidate"]))
        or release["monitor_interval_seconds"] != 900
        or release["training_timeout_seconds"] != 21600
    ):
        raise ValueError("Wave-8 launch helper binding changed")
    return release


def binding(root: Path) -> dict:
    """Check the separate launch archive and exact numerical evidence, without GPU execution."""
    release = helper_binding(root)
    candidate_path = root / "run-preparation/wave8-candidate.json"
    numerical = read(candidate_path)
    source = Path(numerical["candidate"])
    validation_path = root / "run-preparation/validation-wave8-gpu.json"
    state_path = root / "run-preparation/wave8-validation-state.json"
    state, receipt = read(state_path), read(validation_path)
    if (
        release["validation_sha256"] != sha256(validation_path)
        or release["validation_state_sha256"] != sha256(state_path)
        or release["data_receipt_sha256"] != numerical["data_receipt_sha256"]
        or release["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or numerical["environment"] != ENVIRONMENT
        or receipt["environment"] != ENVIRONMENT
        or numerical["archive_sha256"] != sha256(Path(numerical["archive"]))
        or any(sha256(source / name) != digest for name, digest in numerical["files_sha256"].items())
        or receipt["source_sha256"] != numerical["source_sha256"]
        or receipt["data_receipt_sha256"] != numerical["data_receipt_sha256"]
        or receipt["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or not all(receipt["checks"].get(gate) is True for gate in numerical["required_gates"])
        or state["status"] != "passed"
        or type(state.get("keeper_restore")) is not int
        or state["keeper_restore"] != 0
        or state.get("cleanup_exit_code") != 0
        or "cleanup_error" in state
        or state["validation_sha256"] != sha256(validation_path)
        or state["binding_sha256"] != sha256(candidate_path)
        or len(receipt["evidence"]["gpu_commands"]) != 13
        or any(
            item["exit_code"] != 0 or item["log_sha256"] != sha256(Path(item["log"]))
            for item in receipt["evidence"]["gpu_commands"]
        )
    ):
        raise ValueError("Wave-8 production release, source, numerical evidence or archive changed")
    return release


def measured_run(root: Path, arm: str, release: dict) -> dict:
    """Validate full source/config identity and a consecutive original scalar prefix."""
    if arm not in ARMS:
        raise ValueError("Unexpected Wave-8 arm")
    source = Path(release["candidate"])
    numerical = read(root / "run-preparation/wave8-candidate.json")
    directory = root / "runs/wave8" / arm
    run = history(directory)
    expected = read(source / f"experiment/configs/{arm}_0p5m_seed0.json")
    if (
        run.manifest["config"] != expected
        or run.manifest["data_receipt_sha256"] != release["data_receipt_sha256"]
        or normalized_sources(source, run.manifest["source_sha256"]) != executed_sources(source, numerical)
        or len(run.segments) != 1
        or expected["training"]["steps"] != 5120
        or expected["training"]["tokens"] != 2684354560
    ):
        raise ValueError("Wave-8 history differs from its exact full-budget source/config/data")
    result = {
        "arm": arm,
        "out": str(directory),
        "observed_updates": len(run.rows),
        "observed_tokens": run.rows[-1]["tokens"],
        "complete": completion(run)["complete"],
        "manifest_sha256": sha256(directory / "run.json"),
        "metrics_sha256": sha256(directory / "metrics_from_000000.jsonl"),
    }
    if not result["complete"]:
        return result
    checkpoint = directory / "checkpoints/step_005120"
    final = read(checkpoint / "complete.json")
    sizes = [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)]
    evaluations = [row for row in run.rows if "c4_loss" in row]
    if (
        final["step"] != 5120
        or final["world"] != 8
        or not all(sizes)
        or len(evaluations) != 22
        or not all("selection_loss" in row and "heldout_losses" in row for row in evaluations)
        or evaluations[-1]["step"] != 5120
    ):
        raise ValueError("Wave-8 completed history lacks final all-rank checkpoints or evaluations")
    result.update(checkpoint_receipt_sha256=sha256(checkpoint / "complete.json"), checkpoint_rank_bytes=sizes)
    return result


def completed_run(root: Path, arm: str, release: dict) -> dict:
    measured = measured_run(root, arm, release)
    if not measured["complete"]:
        raise ValueError("Wave-8 full-budget history is incomplete")
    return measured


def verify_histories(root: Path) -> dict:
    release = binding(root)
    for arm in ARMS:
        measured_run(root, arm, release)
    return release
