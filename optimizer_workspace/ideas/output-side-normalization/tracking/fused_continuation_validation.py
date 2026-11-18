# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run paired GPU diagnostics and preserve first-update differences without production acceptance."""

import argparse
import math
import os
import subprocess
from pathlib import Path

import torch
from analyze_tuning import executed_sources, normalized_sources
from compare_runs import history
from diagnose_continuation import SECTIONS, compare_saved
from fused_attribution_validation import INTERRUPTED_MANIFEST as INTERRUPTED_MANIFEST
from fused_attribution_validation import (
    numerical_modules,
    require_capture_history,
    require_evaluations,
    require_numerical_state,
    save_interrupted_manifest,
)
from fused_attribution_validation_contract import locations as parent_locations
from fused_continuation_contract import (
    BATCH_TOKENS,
    BOUNDARY,
    CONFIG,
    ENDPOINT,
    ENVIRONMENT,
    FIRST_RESUMED,
    MODES,
    PREPARATION,
    PREPARED_STATUS,
    PROTOCOLS,
    TOLERANCE,
    WORLD,
    locations,
    plan,
    require_actual_config,
)
from selected_lr_release_owner import candidate_files, read, sha256
from selected_lr_stage_profile_validation import complete_checkpoint_evidence
from watch_wave5 import write

COMPLETE_STATUS = "completed_fused_continuation_diagnostic_not_production_gate"
COMMAND_TIMEOUT = 900


def source(candidate: Path, root: Path) -> dict:
    prepared = read(candidate / PREPARATION)
    require_actual_config(candidate)
    if (
        prepared["status"] != PREPARED_STATUS
        or prepared["candidate"] != str(candidate)
        or prepared["root"] != str(root)
        or prepared["files_sha256"] != candidate_files(candidate)
        or prepared["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or any(sha256(candidate / name) != value for name, value in prepared["parent_metadata_sha256"].items())
        or any(
            sha256(root / name) != value for name, value in prepared["allocation_predecessor"]["evidence_sha256"].items()
        )
    ):
        raise ValueError("Continuation numerical source, full schedule or failed-parent evidence changed")
    return prepared


def trajectory(directory: Path, candidate: Path, prepared: dict, mode: str) -> dict:
    config = read(candidate / CONFIG)
    run = history(directory)
    endpoint = BOUNDARY if mode == "interrupted" else ENDPOINT
    rows = run.segments[0] if mode == "interrupted" else run.rows
    manifest_path = directory / INTERRUPTED_MANIFEST if mode == "interrupted" else directory / "run.json"
    manifest = read(manifest_path)
    segments = sorted(directory.glob("metrics_from_*.jsonl"))
    expected_segments = ["metrics_from_000000.jsonl"]
    if mode == "resumed":
        expected_segments.append(f"metrics_from_{BOUNDARY:06d}.jsonl")
    if (
        mode not in MODES
        or manifest["config"] != config
        or read(directory / "config.json") != config
        or manifest["data_receipt_sha256"] != prepared["data_receipt_sha256"]
        or normalized_sources(candidate, manifest["source_sha256"]) != executed_sources(candidate, prepared)
        or manifest["status"] != "execution_check_complete"
        or manifest.get("execution_check") is not True
        or manifest["completed_steps"] != endpoint
        or manifest.get("resume") != (str(directory / f"checkpoints/step_{BOUNDARY:06d}") if mode == "resumed" else None)
        or [row["step"] for row in rows] != list(range(1, endpoint + 1))
        or any(row["tokens"] != row["step"] * BATCH_TOKENS for row in rows)
        or (mode != "interrupted" and [path.name for path in segments] != expected_segments)
        or any(path.is_symlink() or path.stat().st_nlink != 1 for path in [manifest_path, *segments])
    ):
        raise ValueError("Continuation diagnostic trajectory lacks its exact source or complete independent history")
    require_evaluations(rows, config, endpoint)
    installations = [
        entry
        for row in rows
        for entry in row["input_refresh_installations"]
        if entry["installation_step"] == row["step"] - 1 and entry["installation_step"] - entry["source_step"] == 1
    ]
    if not installations or len(installations) != sum(len(row["input_refresh_installations"]) for row in rows):
        raise ValueError("Continuation diagnostic did not retain fixed refresh installation age")
    checkpoints = {}
    for step in (BOUNDARY, FIRST_RESUMED, ENDPOINT):
        if step > endpoint:
            continue
        checkpoints[str(step)] = complete_checkpoint_evidence(directory, config, candidate, prepared, step)
        for rank in range(WORLD):
            saved = torch.load(
                directory / f"checkpoints/step_{step:06d}/rank_{rank}.pt",
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            # The first resumed root update is eager warmup of the enabled process-local graph.
            require_numerical_state(
                saved, config, step, rank, require_cached_graph=not (step == FIRST_RESUMED and mode == "resumed")
            )
    if mode != "interrupted":
        require_capture_history(directory, config["experiment"]["arm"])
    return {
        "manifest_sha256": sha256(manifest_path),
        "history_sha256": {path.name: sha256(path) for path in (segments[:1] if mode == "interrupted" else segments)},
        "updates": len(rows),
        "tokens": rows[-1]["tokens"],
        "fixed_age_installations": len(installations),
        "checkpoints": checkpoints,
    }


def checkpoint_pair(actual_directory: Path, expected_directory: Path, step: int) -> dict:
    """Report every numerical and scalar mismatch, including process-local graph descriptors."""
    result = []
    for rank in range(WORLD):
        paths = [
            directory / f"checkpoints/step_{step:06d}/rank_{rank}.pt"
            for directory in (actual_directory, expected_directory)
        ]
        hashes = [sha256(path) for path in paths]
        actual, expected = (torch.load(path, map_location="cpu", weights_only=False, mmap=True) for path in paths)
        if (
            any(
                actual[key] != expected[key] or actual[key] != value
                for key, value in (("rank", rank), ("world", WORLD), ("step", step))
            )
            or actual["config"] != expected["config"]
            or actual["next_global_sequence"] != expected["next_global_sequence"]
        ):
            raise ValueError("Continuation diagnostic checkpoint pair has different rank, data position or config")
        changed, scalars, count = [], [], 0
        for name in SECTIONS:
            if name in expected:
                count += compare_saved(actual[name], expected[name], f"rank_{rank}.{name}", changed, scalars)
        if any(sha256(path) != value for path, value in zip(paths, hashes, strict=True)):
            raise ValueError("Continuation diagnostic checkpoint changed during comparison")
        result.append(
            {
                "rank": rank,
                "tensor_comparisons": count,
                "changed_tensors": changed,
                "failed_tensor_count": sum(not item["within_tolerance"] for item in changed),
                "scalar_or_structure_differences": scalars,
                "checkpoint_sha256": dict(zip((str(path) for path in paths), hashes, strict=True)),
            }
        )
    return {
        "step": step,
        "within_tolerance": all(
            item["failed_tensor_count"] == 0 and not item["scalar_or_structure_differences"] for item in result
        ),
        "ranks": result,
    }


def loss_pair(actual_directory: Path, expected_directory: Path) -> dict:
    actual, expected = (history(directory).rows for directory in (actual_directory, expected_directory))
    if len(actual) != ENDPOINT or len(expected) != ENDPOINT:
        raise ValueError("Continuation diagnostic requires both complete loss histories")
    comparisons, failures = 0, []
    for actual_row, expected_row in zip(actual, expected, strict=True):
        for key in ("train_loss", "c4_loss", "selection_loss", "heldout_losses"):
            if (key in actual_row) != (key in expected_row):
                raise ValueError("Continuation diagnostic evaluation coverage differs")
            if key not in actual_row:
                continue
            observed, reference = actual_row[key], expected_row[key]
            values = [(key, observed, reference)]
            if key == "heldout_losses":
                if observed.keys() != reference.keys():
                    raise ValueError("Continuation diagnostic held-out identities differ")
                values = [(name, observed[name], reference[name]) for name in reference]
            for name, value, baseline in values:
                comparisons += 1
                if not math.isfinite(value) or not math.isfinite(baseline):
                    raise ValueError("Continuation diagnostic has nonfinite scalar history")
                if not math.isclose(value, baseline, rel_tol=TOLERANCE["rtol"], abs_tol=TOLERANCE["atol"]):
                    failures.append({"step": actual_row["step"], "metric": name, "actual": value, "expected": baseline})
    return {"comparisons": comparisons, "within_tolerance": not failures, "failures": failures}


def traces(directory: Path, candidate: Path, prepared: dict) -> dict:
    config = read(candidate / CONFIG)
    _, records, _ = numerical_modules(candidate)
    identity = records.checkpoint_identity(config, Path(prepared["root"]) / "data/experiment_receipt.json")
    hashes = {}
    for rank in range(WORLD):
        for step in (BOUNDARY, FIRST_RESUMED, ENDPOINT):
            path = directory / f"continuation-trace/rank_{rank}/update_{step:06d}.pt"
            receipt_path = path.with_suffix(".json")
            receipt = read(receipt_path)
            value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            provenance = {
                "identity": identity,
                "start_step": BOUNDARY if step > BOUNDARY and directory.name.endswith("resumed") else 0,
            }
            stages = [item["stage"] for item in value["records"]]
            required = {
                "input/statistics",
                "input/before_finish",
                "input/finished",
                "model/gradients",
                "model/after_update",
                "matrix/before_update",
                "matrix/after_update",
            }
            if (
                path.is_symlink()
                or path.stat().st_nlink != 1
                or receipt_path.is_symlink()
                or receipt["trace_sha256"] != sha256(path)
                or value["schema"] != 1
                or value["rank"] != rank
                or value["update"] != step
                or value["provenance"] != provenance
                or receipt["provenance"] != provenance
                or receipt["rank"] != rank
                or receipt["update"] != step
                or receipt["stages"] != stages
                or receipt["records"] != len(stages)
                or not required.issubset(stages)
            ):
                raise ValueError("Continuation trace has missing stages or different source/rank/update provenance")
            hashes[str(path)] = sha256(path)
            hashes[str(receipt_path)] = sha256(receipt_path)
    return {"files_sha256": hashes}


def require_original_trajectory(directory: Path, execution: dict, candidate: Path) -> None:
    """Authenticate retained original evidence before cross-source diagnostic comparisons."""
    retained = execution["trajectory"]
    if (
        retained["manifest_sha256"] != sha256(directory / "run.json")
        or any(sha256(directory / name) != value for name, value in retained["history_sha256"].items())
        or type(execution["exit_code"]) is not int
        or execution["exit_code"] != 0
        or read(directory / "config.json") != read(candidate / CONFIG)
    ):
        raise ValueError("Original failed-attribution manifest, history or full-schedule configuration changed")
    for step in (BOUNDARY, ENDPOINT):
        saved = retained["checkpoints"][str(step)]
        receipt = directory / f"checkpoints/step_{step:06d}/complete.json"
        if sha256(receipt) != saved["receipt_sha256"]:
            raise ValueError("Original failed-attribution complete-checkpoint receipt changed")
        for rank in range(WORLD):
            path = directory / f"checkpoints/step_{step:06d}/rank_{rank}.pt"
            if sha256(path) != saved["rank_sha256"][path.name]:
                raise ValueError("Original failed-attribution checkpoint changed before diagnostic comparison")


def analysis(root: Path, candidate: Path, prepared: dict) -> dict:
    work = locations(root)[3]
    outputs = {
        f"{protocol}_{mode}": work / f"{protocol}_{mode}" for protocol in PROTOCOLS for mode in ("continuous", "resumed")
    }
    evidence = {
        "trajectories": {
            name: trajectory(path, candidate, prepared, "resumed" if name.endswith("resumed") else "continuous")
            for name, path in outputs.items()
        },
        "continuation": {
            protocol: {
                str(step): checkpoint_pair(outputs[f"{protocol}_resumed"], outputs[f"{protocol}_continuous"], step)
                for step in (BOUNDARY, FIRST_RESUMED, ENDPOINT)
            }
            for protocol in PROTOCOLS
        },
        "transparency": {
            mode: {
                "checkpoints": {
                    str(step): checkpoint_pair(outputs[f"trace_{mode}"], outputs[f"control_{mode}"], step)
                    for step in (BOUNDARY, FIRST_RESUMED, ENDPOINT)
                },
                "loss": loss_pair(outputs[f"trace_{mode}"], outputs[f"control_{mode}"]),
            }
            for mode in ("continuous", "resumed")
        },
        "traces": {mode: traces(outputs[f"trace_{mode}"], candidate, prepared) for mode in ("continuous", "resumed")},
    }
    original_work, original_gpu = parent_locations(root)[3:]
    original = read(original_gpu)
    original_name = "outer_leftdiag_fused_full_lr_lower"
    baseline = {}
    for mode in ("continuous", "resumed"):
        directory = original_work / f"{original_name}_{mode}"
        execution = next(
            item for item in original["gpu_commands"] if item["trial"] == original_name and item["mode"] == mode
        )
        require_original_trajectory(directory, execution, candidate)
        baseline[mode] = {
            "checkpoints": {
                str(step): checkpoint_pair(outputs[f"control_{mode}"], directory, step) for step in (BOUNDARY, ENDPOINT)
            },
            "loss": loss_pair(outputs[f"control_{mode}"], directory),
        }
    evidence["original_trace_off_comparison"] = baseline
    evidence["extra_checkpoint_transparent"] = all(
        item["loss"]["within_tolerance"] and all(pair["within_tolerance"] for pair in item["checkpoints"].values())
        for item in baseline.values()
    )
    evidence["trace_transparent"] = all(
        item["loss"]["within_tolerance"] and all(pair["within_tolerance"] for pair in item["checkpoints"].values())
        for item in evidence["transparency"].values()
    )
    if evidence["trace_transparent"] and evidence["extra_checkpoint_transparent"]:
        pairs = []
        for rank in range(WORLD):
            for step in (BOUNDARY, FIRST_RESUMED, ENDPOINT):
                paths = [
                    outputs[f"trace_{mode}"] / f"continuation-trace/rank_{rank}/update_{step:06d}.pt"
                    for mode in ("resumed", "continuous")
                ]
                actual, expected = (
                    torch.load(path, map_location="cpu", weights_only=False, mmap=True) for path in paths
                )
                changed, scalars = [], []
                count = compare_saved(
                    actual["records"], expected["records"], f"rank_{rank}.update_{step}", changed, scalars
                )
                pairs.append(
                    {
                        "rank": rank,
                        "update": step,
                        "tensor_comparisons": count,
                        "changed_tensors": changed,
                        "scalar_or_structure_differences": scalars,
                    }
                )
        evidence["intermediates"] = pairs
    else:
        evidence["intermediates_withheld"] = (
            "Trace-off/on or original/new trace-off trajectories differ; saved trace files do not establish causality."
        )
    return evidence


def run(root: Path, candidate: Path, receipt_path: Path) -> dict:
    work = locations(root)[3]
    if receipt_path != locations(root)[4] or work.exists() or receipt_path.exists():
        raise FileExistsError("Continuation GPU diagnostic requires unused canonical evidence")
    prepared = source(candidate, root)
    numerical, _, owner = numerical_modules(candidate)
    result = plan(candidate, root) | {
        "owner": owner.require_gpu_owner(root),
        "execution_binding_sha256": sha256(locations(root)[0]),
        "candidate_preparation_sha256": sha256(candidate / PREPARATION),
        "gpu_commands": [],
        "status": "diagnosing",
    }
    numerical.require_receipt(root / "data", root / "data/experiment_receipt.json")
    if any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()):
        raise RuntimeError("Continuation validator environment differs")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Continuation validator refused occupied GPUs")
    work.mkdir(parents=True)
    write(receipt_path, result)
    try:
        for index, command in enumerate(result["commands"]):
            protocol, mode = PROTOCOLS[index // len(MODES)], MODES[index % len(MODES)]
            if owner.require_gpu_owner(root) != result["owner"] or source(candidate, root) != prepared:
                raise RuntimeError("Continuation source or original validator ownership changed")
            state_path = locations(root)[2]
            state = read(state_path) | {"active_command": command}
            write(state_path, state)
            log = work / f"{protocol}_{mode}.log"
            with log.open("x") as stream:
                try:
                    code = subprocess.run(
                        command,
                        cwd=candidate,
                        env=os.environ | ENVIRONMENT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        timeout=COMMAND_TIMEOUT,
                    ).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            execution = {
                "protocol": protocol,
                "mode": mode,
                "command": command,
                "exit_code": code,
                "log": str(log),
                "log_sha256": sha256(log),
            }
            result["gpu_commands"].append(execution)
            write(receipt_path, result)
            if code != 0:
                raise RuntimeError(f"Continuation diagnostic command failed; preserve {log}")
            directory = work / f"{protocol}_{'continuous' if mode == 'continuous' else 'resumed'}"
            if mode == "interrupted":
                save_interrupted_manifest(directory)
            execution["trajectory"] = trajectory(directory, candidate, prepared, mode)
            write(receipt_path, result)
        result["analysis"] = analysis(root, candidate, prepared)
        result["status"] = COMPLETE_STATUS
        write(receipt_path, result)
        return reconstruct(root, candidate, receipt_path)
    except BaseException as error:
        result.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        write(receipt_path, result)
        raise


def reconstruct(root: Path, candidate: Path, receipt_path: Path) -> dict:
    raw = receipt_path.read_bytes()
    result, prepared = read(receipt_path), source(candidate, root)
    frozen, state = read(locations(root)[0]), read(locations(root)[2])
    expected = plan(candidate, root)
    if (
        receipt_path != locations(root)[4]
        or result["status"] != COMPLETE_STATUS
        or "error" in result
        or any(result[key] != value for key, value in expected.items() if key != "status")
        or result["execution_binding_sha256"] != sha256(locations(root)[0])
        or state["binding_sha256"] != result["execution_binding_sha256"]
        or frozen["preparation_sha256"] != result["candidate_preparation_sha256"]
        or result["candidate_preparation_sha256"] != sha256(candidate / PREPARATION)
        or result["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or len(result["gpu_commands"]) != len(expected["commands"])
    ):
        raise ValueError("Continuation diagnostic has a different plan, incomplete commands or owner")
    for index, (execution, command) in enumerate(zip(result["gpu_commands"], expected["commands"], strict=True)):
        protocol, mode = PROTOCOLS[index // len(MODES)], MODES[index % len(MODES)]
        directory = locations(root)[3] / f"{protocol}_{'continuous' if mode == 'continuous' else 'resumed'}"
        log = locations(root)[3] / f"{protocol}_{mode}.log"
        if (
            execution["protocol"] != protocol
            or execution["mode"] != mode
            or execution["command"] != command
            or type(execution["exit_code"]) is not int
            or execution["exit_code"] != 0
            or execution["log"] != str(log)
            or execution["log_sha256"] != sha256(log)
            or execution["trajectory"] != trajectory(directory, candidate, prepared, mode)
        ):
            raise ValueError("Continuation original command, log or checkpoint evidence changed")
    if result["analysis"] != analysis(root, candidate, prepared) or receipt_path.read_bytes() != raw:
        raise ValueError("Continuation independently reconstructed analysis differs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--reconstruct", action="store_true")
    args = parser.parse_args()
    operation = reconstruct if args.reconstruct else run
    result = operation(args.root.resolve(), args.candidate.resolve(), args.receipt.resolve())
    print(result["status"])


if __name__ == "__main__":
    main()
