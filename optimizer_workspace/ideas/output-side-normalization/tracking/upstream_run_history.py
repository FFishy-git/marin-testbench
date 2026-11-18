# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read Wave-9 histories and immutable evidence without importing training code or Torch."""

from pathlib import Path

from analyze_tuning import executed_sources, normalized_sources
from compare_runs import completion, history
from launch_wave5 import NODES, PYTHON, read, sha256
from upstream_validation import ARMS, GATES, SYNTHETIC_GATES, validation_commands
from validate_wave5 import ENVIRONMENT
from wave7_release import candidate_inventory

CHANGES = {
    "experiment/execution_owner.py",
    "experiment/production_owner.py",
    "experiment/cluster.json",
    "experiment/tests/test_upstream_release.py",
}
EXTRA_HELPERS = (
    "upstream_production.py",
    "upstream_production_owner.py",
    "test_upstream_production_owner.py",
    "test_upstream_production.py",
    "upstream_run_history.py",
    "analyze_upstream_run.py",
    "swanlab_upstream_run.py",
    "monitor_upstream_run.sh",
    "time_to_target.py",
)
STEPS = 5120
TOKENS = 2684354560


def paths(root: Path) -> tuple[Path, Path]:
    return root / "run-preparation/wave9-launch-binding.json", root / "controllers/wave9-launch-20261117/tracking"


def commands(root: Path, candidate: Path) -> dict[str, list[str]]:
    """Return the exact eight-rank full-budget command for each designated node."""
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
            str(root / "runs/wave9" / arm),
            "--launch-release",
            str(release_path),
        ]
        for node, arm in zip(NODES, ARMS, strict=True)
    }


def helper_binding(root: Path) -> dict:
    """Keep trusted observation helpers available even if source or data identity changes."""
    release_path, helpers = paths(root)
    release = read(release_path)
    numerical_path = root / "run-preparation/upstream-full-model-corrected-binding.json"
    numerical = read(numerical_path)
    source = root / "candidates/upstream-full-model-production-20261117"
    original_helpers = numerical["helpers_sha256"]
    if (
        Path(__file__).resolve() != (helpers / "upstream_run_history.py").resolve()
        or release["status"] != "frozen_validated_launch_release"
        or release["root"] != str(root)
        or release["candidate"] != str(source)
        or release["helpers"] != str(helpers)
        or set(original_helpers) & set(EXTRA_HELPERS)
        or set(release["helpers_sha256"]) != set(original_helpers) | set(EXTRA_HELPERS)
        or any(sha256(helpers / name) != digest for name, digest in release["helpers_sha256"].items())
        or any(release["helpers_sha256"][name] != digest for name, digest in original_helpers.items())
        or release["archive"] != str(root / "run-preparation/source-wave9-launch-20261117.tar")
        or release["archive_sha256"] != sha256(Path(release["archive"]))
        or release["validation_binding_sha256"] != sha256(numerical_path)
        or release["environment"] != ENVIRONMENT
        or tuple(release["arms"]) != ARMS
        or tuple(release["nodes"]) != NODES
        or release["commands"] != commands(root, source)
        or type(release["monitor_interval_seconds"]) is not int
        or release["monitor_interval_seconds"] != 900
        or type(release["training_timeout_seconds"]) is not int
        or release["training_timeout_seconds"] != 21600
    ):
        raise ValueError("Wave-9 launch helper archive or protocol changed")
    return release


def binding(root: Path) -> dict:
    """Check saved numerical and production evidence; acceptance rederivation belongs to the owner."""
    release = helper_binding(root)
    preparation_root = root / "run-preparation"
    numerical_path = preparation_root / "upstream-full-model-corrected-binding.json"
    numerical = read(numerical_path)
    previous = Path(numerical["candidate"])
    source = Path(release["candidate"])
    state_path = preparation_root / "upstream-full-model-corrected-state.json"
    receipt_path = preparation_root / "validation-upstream-full-model-corrected-gpu.json"
    state, receipt = read(state_path), read(receipt_path)
    files = candidate_inventory(source)
    changed = {
        name
        for name in files.keys() | numerical["files_sha256"].keys()
        if files.get(name) != numerical["files_sha256"].get(name)
    }
    work = root / "validation/upstream_full_model_corrected"
    evidence = receipt["evidence"]
    gpu_commands = evidence.get("gpu_commands", [])
    expected_validation = [
        PYTHON,
        "-m",
        "experiment.validate",
        "--gpu",
        "--data",
        str(root / "data"),
        "--data-receipt",
        str(root / "data/experiment_receipt.json"),
        "--work",
        str(work),
        "--receipt",
        str(receipt_path),
    ]
    if (
        numerical["root"] != str(root)
        or previous != root / "candidates/upstream-full-model-corrected-validator-20261117"
        or tuple(numerical["arms"]) != ARMS
        or tuple(numerical["required_gates"]) != GATES
        or numerical["environment"] != ENVIRONMENT
        or numerical["archive_sha256"] != sha256(Path(numerical["archive"]))
        or any(
            sha256(Path(numerical["helpers"]) / name) != digest for name, digest in numerical["helpers_sha256"].items()
        )
        or candidate_inventory(previous) != numerical["files_sha256"]
        or files != release["files_sha256"]
        or changed != CHANGES
        or release["source_sha256"]
        != {name: digest for name, digest in files.items() if name.startswith("experiment/") and name.endswith(".py")}
        or release["validation_sha256"] != sha256(receipt_path)
        or release["state_sha256"] != sha256(state_path)
        or release["data_receipt_sha256"] != numerical["data_receipt_sha256"]
        or release["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or receipt["candidate_path"] != str(previous)
        or receipt["source_sha256"] != numerical["source_sha256"]
        or receipt["environment"] != ENVIRONMENT
        or receipt["data_receipt_sha256"] != numerical["data_receipt_sha256"]
        or receipt["config_sha256"] != {arm: files[f"experiment/configs/{arm}_0p5m_seed0.json"] for arm in ARMS}
        or receipt["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or not all(receipt["checks"].get(gate) is True for gate in GATES)
        or set(evidence.get("real_data_smokes", {})) != set(ARMS)
        or numerical["commands"] != validation_commands(PYTHON, previous, root, work)
        or [item["command"] for item in gpu_commands] != numerical["commands"]
        or any(
            type(item["exit_code"]) is not int
            or item["exit_code"] != 0
            or item["log_sha256"] != sha256(Path(item["log"]))
            for item in gpu_commands
        )
        or evidence["validation_command"] != expected_validation
        or state["command"] != expected_validation
        or state["status"] != "passed"
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or "cleanup_error" in state
        or state["validation_sha256"] != sha256(receipt_path)
        or state["binding_sha256"] != sha256(numerical_path)
    ):
        raise ValueError("Wave-9 source migration, validation gates or command-log evidence changed")
    synthetic = read(work / "gpu_checks.json")
    if (
        evidence["gpu_receipt"] != str(work / "gpu_checks.json")
        or synthetic["world"] != 2
        or synthetic["source_sha256"] != numerical["source_sha256"]
        or not all(synthetic["checks"].get(gate) is True for gate in SYNTHETIC_GATES)
    ):
        raise ValueError("Wave-9 original synthetic source or gate coverage changed")
    preparation_path = source / "production-preparation.json"
    cpu_path = Path(release["cpu_receipt"])
    if (
        release["preparation"] != str(preparation_path)
        or release["preparation_sha256"] != sha256(preparation_path)
        or release["cpu_receipt_sha256"] != sha256(cpu_path)
    ):
        raise ValueError("Wave-9 production preparation or CPU receipt changed")
    prepared, cpu = read(preparation_path), read(cpu_path)
    if (
        prepared["status"] != "prepared_owner_only_no_production_release"
        or prepared["candidate"] != str(source)
        or prepared["validation_binding_sha256"] != sha256(numerical_path)
        or prepared["files_sha256"] != files
        or set(prepared["changed_files"]) != CHANGES
        or prepared["owner_sha256"] != files["experiment/production_owner.py"]
        or prepared["test_sha256"] != files["experiment/tests/test_upstream_release.py"]
        or sha256(Path(release["helpers"]) / "upstream_production_owner.py") != prepared["owner_sha256"]
        or sha256(Path(release["helpers"]) / "test_upstream_production_owner.py") != prepared["test_sha256"]
        or cpu["status"] != "production_cpu_verified_gpu_release_pending"
        or cpu["candidate"] != str(source)
        or cpu["preparation_sha256"] != sha256(preparation_path)
        or cpu["files_sha256"] != files
        or cpu["cuda_visible_devices"] != ""
        or cpu["cwd"] != str(source)
        or cpu["command"]
        != [
            PYTHON,
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-p",
            "no:cacheprovider",
            "-q",
            "-m",
            "not slow and not integration and not data_integration and not cluster "
            "and not requires_cluster and not docker and not manual",
            "experiment/tests",
        ]
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["log_sha256"] != sha256(Path(cpu["log"]))
        or cpu["archive_sha256"] != sha256(Path(cpu["archive"]))
    ):
        raise ValueError("Wave-9 ownership-only preparation or executed CPU evidence differs")
    cluster = read(source / "experiment/cluster.json")
    if (
        cluster["wave_name"] != "wave9"
        or tuple(cluster["nodes"]) != NODES
        or cluster["python"] != PYTHON
        or cluster["output_root"] != str(root)
        or cluster["gpus_per_run"] != 8
        or cluster["environment"] != ENVIRONMENT
        or cluster["wave"] != [{"arm": arm, "seed": 0, "batch": "0p5m"} for arm in ARMS]
    ):
        raise ValueError("Wave-9 cluster arm, node or environment protocol differs")
    executed_sources(source, release)
    return release


def measured_run(root: Path, arm: str, release: dict) -> dict:
    """Require exact full-budget provenance and a consecutive, uninterrupted scalar prefix."""
    if arm not in ARMS:
        raise ValueError("Unexpected Wave-9 arm")
    source = Path(release["candidate"])
    directory = root / "runs/wave9" / arm
    run = history(directory)
    expected = read(source / f"experiment/configs/{arm}_0p5m_seed0.json")
    training = expected["training"]
    if (
        candidate_inventory(source) != release["files_sha256"]
        or run.manifest["config"] != expected
        or expected["experiment"]["arm"] != arm
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
        raise ValueError("Wave-9 history differs from its exact full-budget source/config/data")
    complete = completion(run)["complete"]
    if run.manifest.get("status") == "complete" and not complete:
        raise ValueError("Wave-9 completed manifest has incomplete scalar coverage")
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
        run.rows[-1]["tokens"] != TOKENS
        or type(final["step"]) is not int
        or final["step"] != STEPS
        or type(final["world"]) is not int
        or final["world"] != 8
        or not all(sizes)
        or len(evaluations) != 22
        or [row["step"] for row in evaluations] != [row["step"] for row in run.rows if "selection_loss" in row]
        or not all("heldout_losses" in row for row in evaluations)
        or evaluations[-1]["step"] != STEPS
    ):
        raise ValueError("Wave-9 completed history lacks shared evaluations or final eight-rank endpoints")
    result.update(checkpoint_receipt_sha256=sha256(checkpoint / "complete.json"), checkpoint_rank_bytes=sizes)
    return result


def completed_run(root: Path, arm: str, release: dict) -> dict:
    measured = measured_run(root, arm, release)
    if not measured["complete"]:
        raise ValueError("Wave-9 full-budget history is incomplete")
    return measured


def verify_histories(root: Path) -> dict:
    release = binding(root)
    for arm in ARMS:
        measured_run(root, arm, release)
    return release
