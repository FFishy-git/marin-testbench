# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify full-budget attribution histories and completed-wave succession evidence."""

import math
from pathlib import Path

from analyze_tuning import executed_sources, normalized_sources
from compare_runs import WINDOW, compare, completion, history
from fused_attribution_production_contract import (
    ENVIRONMENT,
    FAMILIES,
    FROZEN_STATUS,
    MONITOR_INTERVAL,
    STEPS,
    TOKENS,
    TRAINING_TIMEOUT,
    WORLD,
    commands,
    config_name,
    output_path,
    paths,
    require_config,
    wave_spec,
    worker_command,
    worker_path,
)
from fused_attribution_production_handoff import require_acceptance
from launch_wave5 import read, sha256
from prepare_fused_attribution_production import archive_inventory, require_cpu
from prepare_selected_lr_validation import require_archive
from wave7_release import candidate_inventory


def binding(root: Path, wave: str) -> dict:
    """Authenticate saved execution provenance without loading checkpoint tensors."""
    binding_path, helpers, _, _ = paths(root, wave)
    original = binding_path.read_bytes()
    release = read(binding_path)
    candidate = Path(release["candidate"])
    spec = wave_spec(wave)
    cpu_path = Path(release["cpu_receipt"])
    cpu = require_cpu(root, cpu_path)
    controller_path = Path(release["controller_cpu_receipt"])
    controller = read(controller_path)
    acceptance_path = Path(release["validation_acceptance"])
    acceptance = require_acceptance(root, acceptance_path)
    if (
        root != root.resolve()
        or candidate != candidate.resolve()
        or Path(__file__).resolve() != helpers / Path(__file__).name
        or release["status"] != FROZEN_STATUS
        or release["root"] != str(root)
        or release["wave"] != wave
        or release["helpers"] != str(helpers)
        or release["candidate"] != cpu["candidate"]
        or release["files_sha256"] != candidate_inventory(candidate)
        or release["files_sha256"] != cpu["files_sha256"]
        or release["cpu_receipt_sha256"] != sha256(cpu_path)
        or release["controller_cpu_receipt_sha256"] != sha256(controller_path)
        or controller["status"] != "fused_attribution_production_controller_cpu_verified_gpu_denied"
        or type(controller["exit_code"]) is not int
        or controller["exit_code"] != 0
        or controller["cuda_visible_devices"] != ""
        or "error" in controller
        or controller["candidate_cpu_receipt_sha256"] != release["cpu_receipt_sha256"]
        or controller["helpers_sha256"] != release["helpers_sha256"]
        or release["environment"] != ENVIRONMENT
        or release["families"] != list(FAMILIES)
        or release["nodes"] != spec["nodes"]
        or release["commands"] != commands(root, candidate, wave)
        or release["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or release["preparation_sha256"] != cpu["preparation_sha256"]
        or release["validation_acceptance_sha256"] != sha256(acceptance_path)
        or release["validation_binding_sha256"] != acceptance["binding_sha256"]
        or release["monitor_interval_seconds"] != MONITOR_INTERVAL
        or release["training_timeout_seconds"] != TRAINING_TIMEOUT
        or set(release["allocation_by_node"]) != set(spec["nodes"])
        or sha256(Path(release["archive"])) != release["archive_sha256"]
        or sha256(Path(controller["archive"])) != controller["archive_sha256"]
        or sha256(Path(controller["log"])) != controller["log_sha256"]
        or any(
            Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != digest
            for name, digest in release["helpers_sha256"].items()
        )
    ):
        raise ValueError("Attribution production binding or original acceptance differs")
    expected = (
        archive_inventory(read(candidate / "fused-attribution-production-preparation.json"), cpu["preparation_sha256"])
        | {
            "candidate-cpu-receipt.json": sha256(cpu_path),
            "controller-cpu-receipt.json": sha256(controller_path),
            "validation-acceptance.json": sha256(acceptance_path),
        }
        | {f"tracking/{name}": digest for name, digest in release["helpers_sha256"].items()}
    )
    require_archive(Path(release["archive"]), expected, alias=True)
    controller_expected = {f"tracking/{name}": digest for name, digest in controller["helpers_sha256"].items()}
    controller_expected.update(controller.get("fixture_files_sha256", {}))
    require_archive(Path(controller["archive"]), controller_expected, alias=False)
    if binding_path.read_bytes() != original:
        raise ValueError("Attribution production binding changed during verification")
    return release


def measured_run(root: Path, wave: str, family: str, release: dict) -> dict:
    source, directory = Path(release["candidate"]), output_path(root, wave, family)
    run = history(directory)
    config = read(source / config_name(family, wave_spec(wave)["lr_label"]))
    require_config(config, wave, family)
    if (
        candidate_inventory(source) != release["files_sha256"]
        or run.manifest["config"] != config
        or run.manifest["data_receipt_sha256"] != release["data_receipt_sha256"]
        or normalized_sources(source, run.manifest["source_sha256"]) != executed_sources(source, release)
        or len(run.segments) != 1
        or sorted(path.name for path in directory.glob("metrics_from_*.jsonl")) != ["metrics_from_000000.jsonl"]
        or run.manifest.get("execution_check")
    ):
        raise ValueError("Attribution uninterrupted history or original provenance differs")
    complete = completion(run)["complete"]
    if run.manifest.get("status") == "complete" and not complete:
        raise ValueError("Completed attribution manifest lacks its full budget")
    result = {
        "family": family,
        "arm": family,
        "wave": wave,
        "out": str(directory),
        "observed_updates": len(run.rows),
        "observed_tokens": run.rows[-1]["tokens"],
        "complete": complete,
        "manifest_sha256": sha256(directory / "run.json"),
        "metrics_sha256": sha256(directory / "metrics_from_000000.jsonl"),
    }
    if not complete:
        return result
    checkpoint = directory / f"checkpoints/step_{STEPS:06d}"
    final = read(checkpoint / "complete.json")
    sizes = [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(WORLD)]
    expected_steps = sorted(set(range(1, STEPS + 1, config["runtime"]["eval_every"])) | {STEPS})
    evaluations = [row for row in run.rows if "selection_loss" in row]
    weights = config["data"]["mixture_weights"]
    if (
        type(final["step"]) is not int
        or final["step"] != STEPS
        or type(final["world"]) is not int
        or final["world"] != WORLD
        or not all(size > 0 for size in sizes)
        or [row["step"] for row in evaluations] != expected_steps
        or [row["step"] for row in run.rows if "c4_loss" in row] != expected_steps
        or len(expected_steps) != 22
        or run.rows[-1]["tokens"] != TOKENS
        or any(
            set(row.get("heldout_losses", {})) != {f"heldout/{key}" for key in weights}
            or any(not math.isfinite(value) for value in row["heldout_losses"].values())
            for row in evaluations
        )
    ):
        raise ValueError("Attribution final checkpoint or exact held-out evaluation coverage differs")
    for row in evaluations:
        expected = sum(weight * row["heldout_losses"][f"heldout/{key}"] for key, weight in weights.items())
        if row["selection_loss"] != expected / sum(weights.values()):
            raise ValueError("Attribution selection differs from the original held-out mixture")
    result.update(
        checkpoint_receipt_sha256=sha256(checkpoint / "complete.json"),
        checkpoint_rank_bytes=sizes,
        evaluation_tokens=[row["tokens"] for row in evaluations],
    )
    return result


def completed_run(root: Path, wave: str, family: str, release: dict) -> dict:
    result = measured_run(root, wave, family, release)
    if not result["complete"]:
        raise ValueError("Attribution full-budget run remains incomplete")
    return result


def verify_histories(root: Path, wave: str) -> dict:
    release = binding(root, wave)
    for family in FAMILIES:
        measured_run(root, wave, family, release)
    return release


def require_completed_snapshot(root: Path, wave: str, release: dict) -> dict:
    snapshot_path = root / f"run-preparation/monitor-{wave}/latest.json"
    snapshot = read(snapshot_path)
    digest = sha256(paths(root, wave)[0])
    if (
        snapshot["status"] != "complete"
        or snapshot["all_workers_terminal"] is not True
        or snapshot["binding_sha256"] != digest
        or len(snapshot["workers"]) != len(FAMILIES)
        or {item["family"] for item in snapshot["workers"]} != set(FAMILIES)
    ):
        raise ValueError("Attribution original workers have not completed and restored")
    for item in snapshot["workers"]:
        family, state = item["family"], item["receipt"]
        persisted = read(worker_path(root, wave, family))
        intent = read(worker_path(root, wave, family).with_suffix(".start.json"))
        node = wave_spec(wave)["nodes"][FAMILIES.index(family)]
        if (
            state != persisted
            or item["status"] != "complete"
            or state["status"] != "complete"
            or state["node"] != node
            or state["release_binding_sha256"] != digest
            or state["command"] != release["commands"][node]
            or any(
                type(state.get(key)) is not int or state[key] != 0
                for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
            )
            or any(key in state for key in ("error", "cleanup_error"))
            or state["completed_run"] != completed_run(root, wave, family, release)
            or sha256(Path(state["log"])) != state["log_sha256"]
            or intent["status"] != "acknowledged"
            or intent["pid"] != state["worker_pid"]
            or intent["start_ticks"] != state["worker_start_ticks"]
            or intent["command"] != worker_command(root, wave, family)
            or item["original_worker_absent"] is not True
            or item["original_child_absent"] is not True
        ):
            raise ValueError("Attribution original generation, completed history or terminal evidence differs")
    return snapshot


def require_analysis(root: Path, wave: str, release: dict) -> dict:
    analysis = read(root / f"run-preparation/analysis-{wave}.json")
    runs = {f"{wave}/{family}": history(output_path(root, wave, family)) for family in FAMILIES}
    expected = [compare(runs, f"{wave}/{family}", WINDOW) for family in ("muont", "null_leftdiag_fused")]
    if (
        analysis["wave"] != wave
        or analysis["window_tokens"] != WINDOW
        or analysis["source_release_sha256"] != sha256(paths(root, wave)[0])
        or analysis["matrix_lr"] != wave_spec(wave)["matrix_lr"]
        or analysis["analyses"] != expected
        or any(
            not result["final_comparison_eligible"]
            or len(result["training_windows"]) != 5
            or len(result["evaluations"]["selection_loss"]["shared_tokens"]) != 22
            for cohort in analysis["analyses"]
            for result in cohort["comparisons"].values()
        )
    ):
        raise ValueError("Attribution completed five-window and exact-evaluation analysis is absent")
    return analysis


def require_completed_wave(root: Path, wave: str) -> dict:
    """Require saved full-budget, cleanup, analysis and every cloud scalar before succession."""
    release = verify_histories(root, wave)
    require_completed_snapshot(root, wave, release)
    require_analysis(root, wave, release)
    cloud_path = root / f"run-preparation/swanlab-upload-{wave}.json"
    cloud = read(cloud_path)
    if (
        cloud["status"] != "verified"
        or cloud["visibility"] != "PRIVATE"
        or cloud["project"] != "siyuc3141/output-side-normalization-marin"
        or set(cloud["runs"]) != set(FAMILIES)
    ):
        raise ValueError("Attribution private cloud verification remains incomplete")
    for family in FAMILIES:
        original, imported = completed_run(root, wave, family, release), cloud["runs"][family]
        verification = imported["verification"]
        if (
            imported["status"] != "verified"
            or any(
                imported[key] != original[key]
                for key in ("manifest_sha256", "metrics_sha256", "checkpoint_receipt_sha256", "checkpoint_rank_bytes")
            )
            or verification["state"] != "FINISHED"
            or verification["configuration_verified"] is not True
            or verification["token_axis_verified"] is not True
            or verification["integer_values_exact"] is not True
            or verification["csv_export_max_ulp"] != 2
            or any(error["max_ulp"] > 2 for error in verification["serialization_errors"].values())
            or verification["point_counts"].get("tokens") != STEPS
            or verification["point_counts"].get("eval/selection_loss") != 22
            or verification["point_counts"].get("eval/c4_loss") != 22
        ):
            raise ValueError("Attribution cloud history differs from its original full-budget run")
    names = (
        paths(root, wave)[0],
        root / f"run-preparation/monitor-{wave}/latest.json",
        root / f"run-preparation/analysis-{wave}.json",
        cloud_path,
    )
    return {
        "status": "attribution_wave_completed_analyzed_cloud_verified",
        "wave": wave,
        "evidence_sha256": {str(path.relative_to(root)): sha256(path) for path in names},
        "tokens_per_run": TOKENS,
        "updates_per_run": STEPS,
        "window_tokens": WINDOW,
        "complete_windows": 5,
        "exact_evaluations": 22,
    }
