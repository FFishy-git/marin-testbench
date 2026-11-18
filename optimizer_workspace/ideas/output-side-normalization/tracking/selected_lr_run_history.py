# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read Wave11 full-budget histories against their selected-LR production release."""

import hashlib
import math
import tarfile
from pathlib import Path

from analyze_tuning import executed_sources, normalized_sources
from compare_runs import completion, history
from launch_wave5 import NODES, PYTHON, read, sha256
from prepare_selected_lr_validation import require_archive as require_controller_archive
from selected_lr_validation import ARMS, ENVIRONMENT, locations, plan
from selected_lr_validation import PREPARATION as VALIDATION_PREPARATION
from wave7_release import candidate_inventory

WAVE = "wave11"
STEPS = 5120
TOKENS = 2684354560
STATUS = "frozen_selected_lr_production_after_original_gpu_validation"
PREPARATION = "selected-lr-release-preparation.json"
WORKER_MARKER = "OUTPUT_MARIN_WAVE11_WORKER_ID"
CHANGES = {
    "experiment/train.py",
    "experiment/selected_lr_release_owner.py",
    "experiment/tests/test_selected_lr_release_owner.py",
    "experiment/cluster.json",
}
EXTRA_HELPERS = (
    "selected_lr_production.py",
    "selected_lr_release_owner.py",
    "prepare_selected_lr_release.py",
    "selected_lr_run_history.py",
    "test_selected_lr_release_owner.py",
    "test_prepare_selected_lr_release.py",
    "test_selected_lr_production.py",
    "analyze_selected_lr_run.py",
    "swanlab_selected_lr_run.py",
    "monitor_selected_lr_run.sh",
    "swanlab_refresh_upload.py",
    "refresh_history.py",
    "time_to_target.py",
)


def paths(root: Path) -> tuple[Path, Path]:
    return root / "run-preparation/wave11-launch-binding.json", root / "controllers/wave11-launch-20261117/tracking"


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


def require_archive(path: Path, expected: dict[str, str]) -> None:
    """Verify every archived regular byte and the sole allowed pinned-source alias."""
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        regular, aliases = [item for item in members if item.isfile()], [item for item in members if not item.isfile()]
        if (
            len(regular) != len(expected)
            or {item.name for item in regular} != set(expected)
            or len(aliases) != 1
            or aliases[0].name != "artifacts/external-sources"
            or not aliases[0].issym()
            or aliases[0].linkname != "pinned-sources-materialized"
        ):
            raise ValueError("Selected-LR production archive inventory differs")
        for item in regular:
            with archive.extractfile(item) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected[item.name]:
                    raise ValueError("Selected-LR archived production bytes differ")


def helper_binding(root: Path) -> dict:
    """Retain trusted observation when training source identity changes."""
    path, helpers = paths(root)
    release, original = read(path), read(locations(root)[0])
    candidate = Path(release["candidate"])
    if (
        Path(__file__).resolve() != (helpers / Path(__file__).name).resolve()
        or release["status"] != STATUS
        or release["root"] != str(root)
        or release["candidate"] != str(candidate.resolve())
        or release["helpers"] != str(helpers)
        or not set(EXTRA_HELPERS) <= set(release["helpers_sha256"])
        or not set(original["helpers_sha256"]) <= set(release["helpers_sha256"])
        or any(sha256(helpers / name) != digest for name, digest in release["helpers_sha256"].items())
        or any(release["helpers_sha256"][name] != digest for name, digest in original["helpers_sha256"].items())
        or release["archive"] != str(root / "run-preparation/source-wave11-launch-20261117.tar")
        or release["archive_sha256"] != sha256(Path(release["archive"]))
        or release["validation_binding_sha256"] != sha256(locations(root)[0])
        or release["environment"] != ENVIRONMENT
        or tuple(release["arms"]) != ARMS
        or tuple(release["nodes"]) != NODES
        or release["commands"] != commands(root, candidate)
        or type(release["monitor_interval_seconds"]) is not int
        or release["monitor_interval_seconds"] != 900
        or type(release["training_timeout_seconds"]) is not int
        or release["training_timeout_seconds"] != 21600
    ):
        raise ValueError("Wave11 frozen production helpers or protocol differ")
    return release


def binding(root: Path) -> dict:
    """Validate saved complete GPU/CPU evidence without recreating accelerator tensors."""
    release = helper_binding(root)
    original_path, original_helpers, state_path, work, receipt_path = locations(root)
    original, state, receipt = read(original_path), read(state_path), read(receipt_path)
    intent_path = original_helpers.parent / "controller.json"
    intent = read(intent_path)
    source, previous = Path(release["candidate"]), Path(original["candidate"])
    files = candidate_inventory(source)
    changed = {
        name
        for name in files.keys() | original["files_sha256"].keys()
        if files.get(name) != original["files_sha256"].get(name)
    }
    expected = plan(previous, root, work)
    if (
        files != release["files_sha256"]
        or candidate_inventory(previous) != original["files_sha256"]
        or changed != CHANGES
        or sha256(source / VALIDATION_PREPARATION) != original["preparation_sha256"]
        or original["archive_sha256"] != sha256(Path(original["archive"]))
        or any(sha256(original_helpers / name) != digest for name, digest in original["helpers_sha256"].items())
        or state["status"] != "passed"
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or any(key in state for key in ("error", "cleanup_error"))
        or receipt["status"] != "passed_actual_full_budget_config_gates"
        or "error" in receipt
        or receipt["owner"]
        != {
            "controller_pid": state["controller_pid"],
            "controller_start_ticks": state["controller_start_ticks"],
        }
        or intent["status"] != "acknowledged"
        or intent["pid"] != state["controller_pid"]
        or intent["start_ticks"] != state["controller_start_ticks"]
        or release["state_sha256"] != sha256(state_path)
        or release["validation_sha256"] != sha256(receipt_path)
        or release["intent_sha256"] != sha256(intent_path)
        or state["binding_sha256"] != sha256(original_path)
        or state["validation_sha256"] != sha256(receipt_path)
        or any(
            receipt[key] != value for key, value in expected.items() if key not in ("status", "trials", "gpu_commands")
        )
        or len(receipt["trials"]) != len(expected["trials"])
        or any(
            any(actual[key] != value for key, value in trial.items() if key != "checks")
            or set(actual["checks"]) != set(trial["checks"])
            or any(value is not True for value in actual["checks"].values())
            for actual, trial in zip(receipt["trials"], expected["trials"], strict=True)
        )
        or len(receipt["gpu_commands"]) != 12
        or [item["command"] for item in receipt["gpu_commands"]] != original["commands"]
        or any(
            type(item["exit_code"]) is not int
            or item["exit_code"] != 0
            or sha256(Path(item["log"])) != item["log_sha256"]
            for item in receipt["gpu_commands"]
        )
        or release["data_receipt_sha256"] != original["data_receipt_sha256"]
        or release["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or release["predecessor_release_sha256"] != sha256(root / "run-preparation/wave10-launch-binding.json")
    ):
        raise ValueError("Wave11 production source or original complete validation evidence differs")
    prepared_path, cpu_path = source / PREPARATION, Path(release["cpu_receipt"])
    prepared, cpu = read(prepared_path), read(cpu_path)
    if (
        release["preparation"] != str(prepared_path)
        or release["preparation_sha256"] != sha256(prepared_path)
        or prepared["status"] != "prepared_release_owner_gpu_pending"
        or prepared["files_sha256"] != files
        or prepared["candidate"] != str(source)
        or prepared["validation_binding_sha256"] != sha256(original_path)
        or set(prepared["changed_files"]) != CHANGES
        or release["cpu_receipt_sha256"] != sha256(cpu_path)
        or cpu["status"] != "selected_lr_release_cpu_verified_gpu_pending"
        or cpu["candidate"] != str(source)
        or cpu["files_sha256"] != files
        or cpu["preparation_sha256"] != sha256(prepared_path)
        or cpu["cuda_visible_devices"] != ""
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or "error" in cpu
        or cpu["log_sha256"] != sha256(Path(cpu["log"]))
        or cpu["archive_sha256"] != sha256(Path(cpu["archive"]))
        or (source / "experiment/selected_lr_release_owner.py").read_bytes()
        != (Path(release["helpers"]) / "selected_lr_release_owner.py").read_bytes()
    ):
        raise ValueError("Wave11 tested production preparation or CPU archive differs")
    cluster, old_cluster = read(source / "experiment/cluster.json"), read(previous / "experiment/cluster.json")
    if cluster != old_cluster | {"wave_name": WAVE} or any(
        (source / f"experiment/configs/{arm}_0p5m_seed0.json").read_bytes()
        != (previous / f"experiment/configs/{arm}_0p5m_seed0.json").read_bytes()
        for arm in ARMS
    ):
        raise ValueError("Wave11 original numerical configuration differs")
    executed_sources(source, release)
    controller_path = Path(release["controller_cpu_receipt"])
    controller = read(controller_path)
    if (
        release["controller_cpu_receipt_sha256"] != sha256(controller_path)
        or controller["status"] != "selected_lr_production_controller_cpu_verified"
        or controller["candidate_cpu_receipt"] != release["cpu_receipt"]
        or controller["candidate_cpu_receipt_sha256"] != release["cpu_receipt_sha256"]
        or controller["helpers_sha256"] != release["helpers_sha256"]
        or type(controller["exit_code"]) is not int
        or controller["exit_code"] != 0
        or controller["cuda_visible_devices"] != ""
        or "error" in controller
        or controller["log_sha256"] != sha256(Path(controller["log"]))
        or controller["archive_sha256"] != sha256(Path(controller["archive"]))
    ):
        raise ValueError("Wave11 production controller CPU evidence differs")
    require_controller_archive(
        Path(controller["archive"]),
        {f"tracking/{name}": digest for name, digest in controller["helpers_sha256"].items()},
        alias=False,
    )
    return release


def measured_run(root: Path, arm: str, release: dict) -> dict:
    """Require one uninterrupted scalar prefix with original config/source/data identity."""
    if arm not in ARMS:
        raise ValueError("Unexpected Wave11 arm")
    source, directory = Path(release["candidate"]), root / f"runs/{WAVE}" / arm
    run = history(directory)
    config = read(source / f"experiment/configs/{arm}_0p5m_seed0.json")
    training = config["training"]
    if (
        candidate_inventory(source) != release["files_sha256"]
        or run.manifest["config"] != config
        or run.manifest["data_receipt_sha256"] != release["data_receipt_sha256"]
        or normalized_sources(source, run.manifest["source_sha256"]) != executed_sources(source, release)
        or len(run.segments) != 1
        or sorted(path.name for path in directory.glob("metrics_from_*.jsonl")) != ["metrics_from_000000.jsonl"]
        or training["steps"] != STEPS
        or training["tokens"] != TOKENS
        or training["batch_tokens"] != 524288
        or training["world_size"] != 8
        or training["seed"] != 0
        or config["experiment"]["arm"] != arm
        or run.manifest.get("execution_check")
    ):
        raise ValueError("Wave11 uninterrupted full-budget history or provenance differs")
    complete = completion(run)["complete"]
    if run.manifest.get("status") == "complete" and not complete:
        raise ValueError("Wave11 completed manifest lacks full scalar coverage")
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
    expected_evaluations = sorted(set(range(1, STEPS + 1, config["runtime"]["eval_every"])) | {STEPS})
    weights = config["data"]["mixture_weights"]
    heldout_keys = {f"heldout/{name}" for name in weights}
    if (
        type(final["step"]) is not int
        or final["step"] != STEPS
        or type(final["world"]) is not int
        or final["world"] != 8
        or not all(size > 0 for size in sizes)
        or [row["step"] for row in evaluations] != expected_evaluations
        or [row["step"] for row in evaluations] != [row["step"] for row in run.rows if "selection_loss" in row]
        or any(
            not isinstance(row.get("heldout_losses"), dict)
            or set(row["heldout_losses"]) != heldout_keys
            or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in row["heldout_losses"].values()
            )
            for row in evaluations
        )
        or evaluations[-1]["step"] != STEPS
    ):
        raise ValueError("Wave11 final eight-rank checkpoint or evaluation coverage differs")
    for row in evaluations:
        selection = sum(weight * row["heldout_losses"][f"heldout/{name}"] for name, weight in weights.items())
        if row["selection_loss"] != selection / sum(weights.values()):
            raise ValueError("Wave11 selection loss differs from its original held-out mixture")
    result.update(checkpoint_receipt_sha256=sha256(checkpoint / "complete.json"), checkpoint_rank_bytes=sizes)
    return result


def completed_run(root: Path, arm: str, release: dict) -> dict:
    result = measured_run(root, arm, release)
    if not result["complete"]:
        raise ValueError("Wave11 full-budget training has not completed")
    return result


def verify_histories(root: Path) -> dict:
    release = binding(root)
    for arm in ARMS:
        measured_run(root, arm, release)
    return release
