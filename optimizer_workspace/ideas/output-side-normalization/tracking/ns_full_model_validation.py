# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain/reconstruct actual-config full-model gates under a separately accepted owner."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import ns_full_model_contract as contract
import torch
from analyze_tuning import executed_sources, normalized_sources
from compare_runs import history
from fused_attribution_validation import (
    INTERRUPTED_MANIFEST,
    numerical_modules,
    require_evaluations,
    require_numerical_state,
    save_interrupted_manifest,
)
from fused_attribution_validation_controller import acknowledged_process, finish_marked_child
from fused_continuation_validation import checkpoint_pair, loss_pair
from selected_lr_release_owner import candidate_files, read, sha256
from selected_lr_stage_profile_validation import STATE_FIELDS
from watch_wave5 import write

STATUS = "ns_full_model_actual_config_gates_reconstructed_not_production"
COMMAND_TIMEOUT = 900


def plan(root: Path, candidate: Path) -> dict:
    commands = contract.commands(candidate, root, contract.locations(root)[3])
    work = contract.locations(root)[3]
    return {
        "root": str(root),
        "candidate": str(candidate),
        "world": contract.WORLD,
        "environment": contract.ENVIRONMENT,
        "commands": commands,
        "trials": [
            spec
            | {
                "config_sha256": sha256(candidate / spec["config_name"]),
                "continuous": str(work / f"{spec['name']}_continuous"),
                "resumed": str(work / f"{spec['name']}_resumed"),
                "commands": commands[index * 3 : (index + 1) * 3],
            }
            for index, spec in enumerate(contract.specs())
        ],
        "planned_tokens": len(contract.specs()) * 2 * contract.ENDPOINT * contract.BATCH_TOKENS,
        "checkpoint_steps": list(contract.CHECKPOINTS),
        "tolerance": contract.TOLERANCE,
        "scope": contract.SCOPE,
    }


def source(root: Path, candidate: Path) -> dict:
    binding = contract.locations(root)[0]
    raw, frozen = binding.read_bytes(), read(binding)
    if (
        frozen["status"] != contract.FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != str(candidate)
        or frozen["files_sha256"] != candidate_files(candidate)
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["plan"] != plan(root, candidate)
        or frozen["commands"] != frozen["plan"]["commands"]
    ):
        raise ValueError("NS full-model frozen source/data or complete actual-config command set differs")
    if binding.read_bytes() != raw:
        raise ValueError("NS full-model source binding changed during checking")
    return frozen


def checkpoint(directory: Path, config: dict, candidate: Path, frozen: dict, step: int, mode: str) -> dict:
    """Validate each complete rank tree once, bracketing reads with source and checkpoint hashes."""
    folder = directory / f"checkpoints/step_{step:06d}"
    names = {"complete.json", *(f"rank_{rank}.pt" for rank in range(contract.WORLD))}
    if (
        folder.is_symlink()
        or {path.name for path in folder.iterdir()} != names
        or any(path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 for path in folder.iterdir())
    ):
        raise ValueError("NS full-model checkpoint lacks exactly eight private regular rank states")
    hashes = {name: sha256(folder / name) for name in names}
    receipt = read(folder / "complete.json")
    if (
        type(receipt["step"]) is not int
        or receipt["step"] != step
        or type(receipt["world"]) is not int
        or receipt["world"] != contract.WORLD
    ):
        raise ValueError("NS full-model checkpoint completion metadata differs")
    sources = executed_sources(candidate, frozen)
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    sizes = []
    for rank in range(contract.WORLD):
        path = folder / f"rank_{rank}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if (
            set(state) != STATE_FIELDS | ({"model"} if rank == 0 else set())
            or any(
                type(state[key]) is not int or state[key] != value
                for key, value in (("step", step), ("rank", rank), ("world", contract.WORLD))
            )
            or state["config"] != config
            or type(state["next_global_sequence"]) is not int
            or state["next_global_sequence"] != step * config["training"]["global_batch_sequences"]
            or state["identity"]["config_sha256"] != config_digest
            or state["identity"]["data_receipt_sha256"] != frozen["data_receipt_sha256"]
            or normalized_sources(candidate, state["identity"]["source_sha256"]) != sources
        ):
            raise ValueError("NS full-model checkpoint full state schema/config/source/data/rank identity differs")
        require_numerical_state(state, config, step, rank, require_cached_graph=not (step == 23 and mode == "resumed"))
        sizes.append(path.stat().st_size)
    if hashes != {name: sha256(folder / name) for name in names}:
        raise ValueError("NS full-model checkpoint changed during single-pass reconstruction")
    return {
        "receipt_sha256": hashes["complete.json"],
        "rank_bytes": sizes,
        "rank_sha256": {f"rank_{rank}.pt": hashes[f"rank_{rank}.pt"] for rank in range(contract.WORLD)},
    }


def trajectory(directory: Path, config: dict, candidate: Path, frozen: dict, mode: str) -> dict:
    if mode not in contract.MODES:
        raise ValueError("NS full-model trajectory mode is outside its protocol")
    run = history(directory)
    interrupted = mode == "interrupted"
    endpoint = contract.BOUNDARY if interrupted else contract.ENDPOINT
    paths = sorted(directory.glob("metrics_from_*.jsonl"))
    expected = ["metrics_from_000000.jsonl"]
    if mode == "resumed":
        expected.append(f"metrics_from_{contract.BOUNDARY:06d}.jsonl")
    selected = paths[:1] if interrupted else paths
    rows = run.segments[0] if interrupted else run.rows
    manifest_path = directory / (INTERRUPTED_MANIFEST if interrupted else "run.json")
    manifest = read(manifest_path)
    identities = {str(path): sha256(path) for path in [manifest_path, directory / "config.json", *selected]}
    if (
        manifest["config"] != config
        or read(directory / "config.json") != config
        or manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
        or normalized_sources(candidate, manifest["source_sha256"]) != executed_sources(candidate, frozen)
        or manifest["status"] != "execution_check_complete"
        or manifest.get("execution_check") is not True
        or type(manifest["completed_steps"]) is not int
        or manifest["completed_steps"] != endpoint
        or manifest.get("resume")
        != (str(directory / f"checkpoints/step_{contract.BOUNDARY:06d}") if mode == "resumed" else None)
        or len(rows) != endpoint
        or [path.name for path in selected] != expected
        or (mode == "resumed" and [len(segment) for segment in run.segments] != [22, 2])
        or any(
            path.is_symlink() or path.stat().st_nlink != 1 or not path.read_bytes().endswith(b"\n")
            for path in [manifest_path, directory / "config.json", *selected]
        )
    ):
        raise ValueError("NS full-model trajectory source, prefix/segment or complete manifest differs")
    require_evaluations(rows, config, endpoint)
    installations = 0
    for row in rows:
        for entry in row["input_refresh_installations"]:
            if entry["installation_step"] != row["step"] - 1 or entry["installation_step"] - entry["source_step"] != 1:
                raise ValueError("NS full-model changed the fixed-age installation boundary")
            installations += 1
    if not installations:
        raise ValueError("NS full-model trajectory never exercised host refresh installation")
    checkpoints = {}
    for step in contract.CHECKPOINTS:
        if step > endpoint:
            continue
        checkpoints[str(step)] = checkpoint(directory, config, candidate, frozen, step, mode)
    if identities != {name: sha256(Path(name)) for name in identities}:
        raise ValueError("NS full-model trajectory changed during reconstruction")
    return {
        "manifest_sha256": sha256(manifest_path),
        "history_sha256": {path.name: sha256(path) for path in selected},
        "updates": len(rows),
        "tokens": rows[-1]["tokens"],
        "fixed_age_installations": installations,
        "checkpoints": checkpoints,
    }


def capture(directory: Path, family: str) -> dict:
    keys = ["diagonal_capture_parity_passed", "outer_diagonal_update_parity_passed", "output_work_retained_passed"]
    if family == "null_leftdiag_fused":
        keys.append("null_identity_outer_passed")
    rows = history(directory).rows
    if (
        family not in ("outer_leftdiag_fused", "null_leftdiag_fused")
        or len(rows) != contract.ENDPOINT
        or any(row.get(key) is not True for row in rows for key in keys)
    ):
        raise ValueError("NS full-model captured-row/update/work gates lack complete true coverage")
    return {"updates": len(rows), "checks": keys}


def trial(item: dict, candidate: Path, frozen: dict) -> dict:
    spec = {key: item[key] for key in contract.specs()[0]}
    config = contract.require_config(candidate, spec)
    continuous, resumed = Path(item["continuous"]), Path(item["resumed"])
    trajectories = {
        "continuous": trajectory(continuous, config, candidate, frozen, "continuous"),
        "interrupted": trajectory(resumed, config, candidate, frozen, "interrupted"),
        "resumed": trajectory(resumed, config, candidate, frozen, "resumed"),
    }
    comparisons = {str(step): checkpoint_pair(resumed, continuous, step) for step in contract.CHECKPOINTS}
    loss = loss_pair(resumed, continuous)
    captures = (
        {name: capture(path, spec["family"]) for name, path in (("continuous", continuous), ("resumed", resumed))}
        if spec["family"] != "muont"
        else {}
    )
    gates = {
        "actual_config_complete_trajectories": True,
        "boundary_all_rank_continuation": comparisons["22"]["within_tolerance"],
        "post_cached_root_all_rank_continuation": comparisons["24"]["within_tolerance"],
        "scalar_loss_continuation": loss["within_tolerance"],
        "captured_row_update_work": True,
    }
    return item | {
        "trajectories": trajectories,
        "continuation": comparisons,
        "first_resumed_diagnostic_only": True,
        "loss": loss,
        "capture": captures,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def run_command(root: Path, candidate: Path, command: list[str], log: Path) -> dict:
    _, _, state_path, _, _ = contract.locations(root)
    state, marker = read(state_path), uuid4().hex
    state["active_command"] = command
    write(state_path, state)
    environment = os.environ | contract.ENVIRONMENT | {contract.COMMAND_MARKER: marker}
    entry, child, generations = {"command": command, "marker": marker, "spawn_status": "starting"}, None, {}
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(
                command,
                cwd=candidate,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            generation = acknowledged_process(
                child, command, candidate, contract.COMMAND_MARKER, marker, contract.ENVIRONMENT
            )
            entry.update(spawn_status="acknowledged", generation=generation)
            entry["exit_code"] = child.wait(timeout=COMMAND_TIMEOUT)
    except BaseException as failure:
        entry["error"] = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        try:
            if child is not None:
                finish_marked_child(child, marker, generations, contract.COMMAND_MARKER)
                entry.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
        except BaseException as failure:
            entry["cleanup_error"] = f"{type(failure).__name__}: {failure}"
            raise
        finally:
            entry.update(log=str(log), log_sha256=sha256(log) if log.is_file() else None, generations=generations)
            write(log.with_suffix(".execution.json"), entry)
    return entry


def require_entry(entry: dict, command: list[str], log: Path) -> None:
    stored = read(log.with_suffix(".execution.json"))
    generation = entry["generation"]
    if {key: value for key, value in stored.items() if key != "time"} != {
        key: value for key, value in entry.items() if key not in ("trial", "mode", "trajectory", "time")
    }:
        raise ValueError("NS full-model command receipt changed")
    if (
        entry["command"] != command
        or entry["spawn_status"] != "acknowledged"
        or generation["command"] != command
        or generation["group"] != generation["pid"]
        or generation["session"] != generation["pid"]
        or any(type(entry[key]) is not int or entry[key] != 0 for key in ("exit_code", "cleanup_exit_code"))
        or entry["marked_descendants_absent"] is not True
        or any(key in entry for key in ("error", "cleanup_error"))
        or not entry["marker"]
        or entry["log"] != str(log)
        or entry["log_sha256"] != sha256(log)
        or log.is_symlink()
        or log.stat().st_nlink != 1
    ):
        raise ValueError("NS full-model command, exact process generation, cleanup or log identity differs")
    for path in (log.with_suffix(".execution.json"),):
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError("NS full-model command receipt is aliased or shared")


def execute(root: Path, candidate: Path) -> dict:
    # The candidate owner must authenticate original controller ancestry before any output or launch.
    _, _, owner = numerical_modules(candidate)
    owned = owner.require_gpu_owner(root)
    frozen = source(root, candidate)
    binding, _, _, work, receipt = contract.locations(root)
    if work.exists() or receipt.exists():
        raise FileExistsError("NS full-model execution identity is one-shot")
    work.mkdir(parents=True)
    result = plan(root, candidate) | {
        "status": "validating",
        "owner": owned,
        "binding_sha256": sha256(binding),
        "gpu_commands": [],
        "results": [],
    }
    write(receipt, result)
    try:
        for item in result["trials"]:
            for mode, command in zip(contract.MODES, item["commands"], strict=True):
                if source(root, candidate) != frozen or owner.require_gpu_owner(root) != owned:
                    raise RuntimeError("NS full-model source or original owner changed")
                log = work / f"{item['name']}_{mode}.log"
                entry = run_command(root, candidate, command, log)
                require_entry(entry, command, log)
                directory = Path(item["continuous"] if mode == "continuous" else item["resumed"])
                if mode == "interrupted":
                    save_interrupted_manifest(directory)
                entry.update(
                    trial=item["name"],
                    mode=mode,
                    trajectory=trajectory(
                        directory,
                        contract.require_config(candidate, {key: item[key] for key in contract.specs()[0]}),
                        candidate,
                        frozen,
                        mode,
                    ),
                )
                result["gpu_commands"].append(entry)
                write(receipt, result)
            result["results"].append(trial(item, candidate, frozen))
            write(receipt, result)
        result.update(
            status=STATUS,
            accepted_cases=[item["name"] for item in result["results"] if item["accepted"]],
            rejected_cases=[item["name"] for item in result["results"] if not item["accepted"]],
            all_cases_accepted=all(item["accepted"] for item in result["results"]),
        )
    except BaseException as failure:
        result.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        raise
    finally:
        write(receipt, result)
    return result


def reconstruct(root: Path, candidate: Path) -> dict:
    frozen, expected = source(root, candidate), plan(root, candidate)
    binding, _, state_path, work, receipt = contract.locations(root)
    raw, result = receipt.read_bytes(), read(receipt)
    state = read(state_path)
    expected_owner = {
        "controller_pid": state["controller_pid"],
        "controller_start_ticks": state["controller_start_ticks"],
        "validator_pid": state["child_pid"],
        "validator_start_ticks": state["child_start_ticks"],
    }
    if (
        result["status"] != STATUS
        or "error" in result
        or result["binding_sha256"] != sha256(binding)
        or state["binding_sha256"] != sha256(binding)
        or result["owner"] != expected_owner
        or any(result[key] != value for key, value in expected.items())
        or len(result["results"]) != len(expected["trials"])
        or len(result["gpu_commands"]) != len(expected["commands"])
        or result["accepted_cases"] != [item["name"] for item in result["results"] if item["accepted"]]
        or result["rejected_cases"] != [item["name"] for item in result["results"] if not item["accepted"]]
        or type(result["all_cases_accepted"]) is not bool
        or result["all_cases_accepted"] != all(item["accepted"] for item in result["results"])
    ):
        raise ValueError("NS full-model retained protocol has incomplete cases/commands or changed source")
    rebuilt = []
    for index, item in enumerate(expected["trials"]):
        for offset, (mode, command) in enumerate(zip(contract.MODES, item["commands"], strict=True)):
            entry = result["gpu_commands"][index * 3 + offset]
            require_entry(entry, command, work / f"{item['name']}_{mode}.log")
            config = contract.require_config(candidate, {key: item[key] for key in contract.specs()[0]})
            directory = Path(item["continuous"] if mode == "continuous" else item["resumed"])
            if (
                entry["trial"] != item["name"]
                or entry["mode"] != mode
                or entry["generation"]["parent"] != result["owner"]["validator_pid"]
                or entry["trajectory"] != trajectory(directory, config, candidate, frozen, mode)
            ):
                raise ValueError("NS full-model retained command ancestry or trajectory differs")
        rebuilt.append(trial(item, candidate, frozen))
    if (
        rebuilt != result["results"]
        or result["accepted_cases"] != [item["name"] for item in rebuilt if item["accepted"]]
        or result["rejected_cases"] != [item["name"] for item in rebuilt if not item["accepted"]]
        or type(result["all_cases_accepted"]) is not bool
        or result["all_cases_accepted"] != all(item["accepted"] for item in rebuilt)
        or receipt.read_bytes() != raw
        or source(root, candidate) != frozen
    ):
        raise ValueError("NS full-model numerical/rejection evidence changed during independent reconstruction")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("plan", "execute", "reconstruct"))
    args = parser.parse_args()
    root, candidate = args.root.resolve(strict=True), args.candidate.resolve(strict=True)
    if args.phase == "plan":
        print(plan(root, candidate)["planned_tokens"], flush=True)
    elif args.phase == "execute":
        print(execute(root, candidate)["status"], flush=True)
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("NS full-model retained reconstruction requires hidden CUDA")
        print(reconstruct(root, candidate)["status"], flush=True)


if __name__ == "__main__":
    main()
