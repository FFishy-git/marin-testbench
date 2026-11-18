# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read Wave-10 histories against the separate owned-output production release."""

from pathlib import Path

from analyze_tuning import executed_sources, normalized_sources
from compare_runs import completion, history
from launch_wave5 import NODES, PYTHON, read, sha256
from owned_output_validation import ARMS, GATES, VALIDATION_ENVIRONMENT, validation_commands
from wave7_release import candidate_inventory

WAVE = "wave10"
STEPS = 5120
TOKENS = 2684354560
CHANGES = {
    "experiment/execution_owner.py",
    "experiment/owned_production_owner.py",
    "experiment/cluster.json",
    "experiment/tests/test_owned_output_release.py",
}
EXTRA_HELPERS = (
    "owned_output_production.py",
    "owned_output_production_owner.py",
    "owned_output_run_history.py",
    "test_owned_output_production.py",
    "test_owned_output_production_owner.py",
    "analyze_owned_output_run.py",
    "swanlab_owned_output_run.py",
    "monitor_owned_output_run.sh",
    "swanlab_refresh_upload.py",
    "refresh_history.py",
    "time_to_target.py",
)


def paths(root: Path) -> tuple[Path, Path]:
    return root / "run-preparation/wave10-launch-binding.json", root / "controllers/wave10-launch-20261117/tracking"


def commands(root: Path, candidate: Path) -> dict[str, list[str]]:
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
            str(root / f"runs/{WAVE}" / arm),
            "--launch-release",
            str(paths(root)[0]),
        ]
        for node, arm in zip(NODES, ARMS, strict=True)
    }


def helper_binding(root: Path) -> dict:
    """Keep trusted observation available when a candidate or data file changes."""
    release_path, helpers = paths(root)
    release = read(release_path)
    original = read(root / "run-preparation/owned-output-trace-binding.json")
    candidate = root / "candidates/owned-output-production-isolated-owner-formatted-20261117"
    if (
        Path(__file__).resolve() != (helpers / "owned_output_run_history.py").resolve()
        or release["status"] != "frozen_validated_launch_release"
        or release["root"] != str(root)
        or release["candidate"] != str(candidate)
        or release["helpers"] != str(helpers)
        or set(original["helpers_sha256"]) & set(EXTRA_HELPERS)
        or set(release["helpers_sha256"]) != set(original["helpers_sha256"]) | set(EXTRA_HELPERS)
        or any(sha256(helpers / name) != digest for name, digest in release["helpers_sha256"].items())
        or any(release["helpers_sha256"][name] != digest for name, digest in original["helpers_sha256"].items())
        or release["archive"] != str(root / "run-preparation/source-wave10-launch-20261117.tar")
        or release["archive_sha256"] != sha256(Path(release["archive"]))
        or release["validation_binding_sha256"] != sha256(root / "run-preparation/owned-output-trace-binding.json")
        or release["environment"] != VALIDATION_ENVIRONMENT
        or tuple(release["arms"]) != ARMS
        or tuple(release["nodes"]) != NODES
        or release["commands"] != commands(root, candidate)
        or type(release["monitor_interval_seconds"]) is not int
        or release["monitor_interval_seconds"] != 900
        or type(release["training_timeout_seconds"]) is not int
        or release["training_timeout_seconds"] != 21600
    ):
        raise ValueError("Wave-10 observation helpers or frozen release changed")
    return release


def binding(root: Path) -> dict:
    """Check the saved full validation and exact ownership-only source migration."""
    release = helper_binding(root)
    preparation = root / "run-preparation"
    original_path = preparation / "owned-output-trace-binding.json"
    state_path = preparation / "owned-output-trace-state.json"
    receipt_path = preparation / "validation-owned-output-trace-gpu.json"
    original, state, receipt = read(original_path), read(state_path), read(receipt_path)
    source, previous = Path(release["candidate"]), Path(original["candidate"])
    files = candidate_inventory(source)
    changed = {
        name
        for name in files.keys() | original["files_sha256"].keys()
        if files.get(name) != original["files_sha256"].get(name)
    }
    completed_commands = receipt["evidence"].get("gpu_commands", [])
    if (
        candidate_inventory(previous) != original["files_sha256"]
        or original["archive_sha256"] != sha256(Path(original["archive"]))
        or tuple(original["arms"]) != ARMS
        or tuple(original["required_gates"]) != GATES
        or original["environment"] != VALIDATION_ENVIRONMENT
        or files != release["files_sha256"]
        or changed != CHANGES
        or release["source_sha256"]
        != {name: digest for name, digest in files.items() if name.startswith("experiment/") and name.endswith(".py")}
        or release["validation_sha256"] != sha256(receipt_path)
        or release["state_sha256"] != sha256(state_path)
        or release["data_receipt_sha256"] != original["data_receipt_sha256"]
        or release["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or receipt["candidate_path"] != str(previous)
        or receipt["source_sha256"] != original["source_sha256"]
        or receipt["environment"] != VALIDATION_ENVIRONMENT
        or receipt["data_receipt_sha256"] != original["data_receipt_sha256"]
        or receipt["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or not all(receipt["checks"].get(gate) is True for gate in GATES)
        or set(receipt["evidence"].get("real_data_smokes", {})) != set(ARMS)
        or original["commands"] != validation_commands(PYTHON, previous, root, root / "validation/owned_output_trace")
        or len(completed_commands) != 14
        or [item["command"] for item in completed_commands] != original["commands"]
        or any(
            type(item["exit_code"]) is not int
            or item["exit_code"] != 0
            or sha256(Path(item["log"])) != item["log_sha256"]
            for item in completed_commands
        )
        or state["status"] != "passed"
        or state["binding_sha256"] != sha256(original_path)
        or state["validation_sha256"] != sha256(receipt_path)
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or "error" in state
        or "cleanup_error" in state
    ):
        raise ValueError("Wave-10 numerical source, all fourteen GPU gates or terminal validation changed")
    prepared_path = source / "production-preparation.json"
    cpu_path = Path(release["cpu_receipt"])
    prepared, cpu = read(prepared_path), read(cpu_path)
    if (
        release["preparation"] != str(prepared_path)
        or release["preparation_sha256"] != sha256(prepared_path)
        or release["cpu_receipt_sha256"] != sha256(cpu_path)
        or prepared["status"] != "prepared_owner_only_no_production_release"
        or prepared["candidate"] != str(source)
        or prepared["validation_binding_sha256"] != sha256(original_path)
        or prepared["files_sha256"] != files
        or set(prepared["changed_files"]) != CHANGES
        or prepared["owner_sha256"] != files["experiment/owned_production_owner.py"]
        or prepared["test_sha256"] != files["experiment/tests/test_owned_output_release.py"]
        or sha256(Path(release["helpers"]) / "owned_output_production_owner.py") != prepared["owner_sha256"]
        or sha256(Path(release["helpers"]) / "test_owned_output_production_owner.py") != prepared["test_sha256"]
        or cpu["status"] != "production_cpu_verified_gpu_release_pending"
        or cpu["candidate"] != str(source)
        or cpu["files_sha256"] != files
        or cpu["preparation_sha256"] != sha256(prepared_path)
        or cpu["cuda_visible_devices"] != ""
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or "error" in cpu
        or cpu["log_sha256"] != sha256(Path(cpu["log"]))
        or cpu["archive_sha256"] != sha256(Path(cpu["archive"]))
    ):
        raise ValueError("Wave-10 ownership preparation or successful CPU evidence changed")
    cluster = read(source / "experiment/cluster.json")
    previous_cluster = read(previous / "experiment/cluster.json")
    if (
        cluster["wave_name"] != WAVE
        or {key: value for key, value in cluster.items() if key not in ("wave_name", "time")}
        != {key: value for key, value in previous_cluster.items() if key not in ("wave_name", "time")}
        or any(
            (source / f"experiment/configs/{arm}_0p5m_seed0.json").read_bytes()
            != (previous / f"experiment/configs/{arm}_0p5m_seed0.json").read_bytes()
            for arm in ARMS
        )
    ):
        raise ValueError("Wave-10 cluster or original full-budget configuration changed")
    executed_sources(source, release)
    return release


def measured_run(root: Path, arm: str, release: dict) -> dict:
    """Require one uninterrupted full-budget scalar history and original provenance."""
    if arm not in ARMS:
        raise ValueError("Unexpected Wave-10 arm")
    source = Path(release["candidate"])
    directory = root / f"runs/{WAVE}" / arm
    run = history(directory)
    config = read(source / f"experiment/configs/{arm}_0p5m_seed0.json")
    training = config["training"]
    if (
        candidate_inventory(source) != release["files_sha256"]
        or run.manifest["config"] != config
        or config["experiment"]["arm"] != arm
        or run.manifest["data_receipt_sha256"] != release["data_receipt_sha256"]
        or normalized_sources(source, run.manifest["source_sha256"]) != executed_sources(source, release)
        or len(run.segments) != 1
        or not (directory / "metrics_from_000000.jsonl").is_file()
        or training["steps"] != STEPS
        or training["tokens"] != TOKENS
        or training["batch_tokens"] != 524288
        or training["world_size"] != 8
        or training["seed"] != 0
        or run.manifest.get("execution_check")
    ):
        raise ValueError("Wave-10 full-budget source, config, data or scalar history differs")
    complete = completion(run)["complete"]
    if run.manifest.get("status") == "complete" and not complete:
        raise ValueError("Wave-10 completed manifest lacks full scalar coverage")
    result = {
        "arm": arm,
        "out": str(directory),
        "observed_updates": len(run.rows),
        "observed_tokens": run.rows[-1]["tokens"],
        "complete": complete,
        "manifest_sha256": sha256(directory / "run.json"),
        "metrics_sha256": sha256(directory / "metrics_from_000000.jsonl"),
    }
    if not complete:
        return result
    checkpoint = directory / "checkpoints/step_005120"
    final = read(checkpoint / "complete.json")
    sizes = [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)]
    evaluations = [row for row in run.rows if "c4_loss" in row]
    if (
        type(final["step"]) is not int
        or final["step"] != STEPS
        or type(final["world"]) is not int
        or final["world"] != 8
        or not all(size > 0 for size in sizes)
        or len(evaluations) != 22
        or [row["step"] for row in evaluations] != [row["step"] for row in run.rows if "selection_loss" in row]
        or not all("heldout_losses" in row for row in evaluations)
        or evaluations[-1]["step"] != STEPS
    ):
        raise ValueError("Wave-10 final checkpoint or exact-shared evaluation coverage is incomplete")
    result.update(checkpoint_receipt_sha256=sha256(checkpoint / "complete.json"), checkpoint_rank_bytes=sizes)
    return result


def completed_run(root: Path, arm: str, release: dict) -> dict:
    measured = measured_run(root, arm, release)
    if not measured["complete"]:
        raise ValueError("Wave-10 full-budget training has not completed")
    return measured


def verify_histories(root: Path) -> dict:
    release = binding(root)
    for arm in ARMS:
        measured_run(root, arm, release)
    return release
