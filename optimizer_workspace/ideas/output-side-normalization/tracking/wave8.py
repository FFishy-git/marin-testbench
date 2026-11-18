# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze and validate the combined-product candidate on its original .139 allocation."""

import argparse
import ast
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import torch
from analyze_tuning import executed_sources, normalized_sources
from benchmark_diagonal import source_definitions
from launch_wave5 import PYTHON, node_action, read, sha256
from prepare_fixed_age_tuning import require_source
from run_fixed_age_validation import VALIDATION_HELPERS, require_completed_wave
from validate_wave5 import ENVIRONMENT
from watch_wave5 import process, write
from wave7_release import GATES as ORIGINAL_GATES
from wave7_release import (
    bounded_execution,
    candidate_inventory,
    idle_training,
    local_node,
    node_guard,
    release_keeper,
)

NODE = "28.216.17.139"
ARMS = (
    "muont",
    "outer_leftdiag",
    "outer_leftdiag_fused",
    "outer_leftdiag_combined_fused",
)
GATES = (
    *ORIGINAL_GATES,
    "gpu_combined_outer_updates",
    "real_data_combined_outer_update_parity",
)
HELPERS = (*VALIDATION_HELPERS, "wave8.py", "monitor_validation.sh", "benchmark_diagonal.py")
PROBE = "validation/combined_outer_gpu_20261117_0118"
CHANGED_FILES = {
    "experiment/block_toner.py",
    "experiment/configure.py",
    "experiment/cluster.json",
    "experiment/gpu_checks.py",
    "experiment/train.py",
    "experiment/validate.py",
    "experiment/launch.py",
    "experiment/outer_diagonal_checks.py",
    "experiment/execution_owner.py",
    "experiment/tests/test_block_toner.py",
    "experiment/tests/test_validation.py",
    *(f"experiment/configs/outer_leftdiag_combined_fused_{batch}_seed0.json" for batch in ("0p5m", "1m", "2m", "4m")),
}


def paths(root: Path) -> tuple[Path, Path]:
    return (
        root / "run-preparation/wave8-candidate.json",
        root / "controllers/wave8-20261117/tracking",
    )


def require_probe(root: Path) -> str:
    """Retain strict production arithmetic evidence, including controller cleanup and restoration."""
    directory = root / PROBE
    state = read(directory / "state.json")
    result = read(directory / "result.json")
    frozen = read(directory / "binding.json")
    if (
        state["status"] != "measured_production_probe_not_launch_gate"
        or type(state.get("keeper_restore")) is not int
        or state["keeper_restore"] != 0
        or state["cleanup_exit_code"] != 0
        or state["exit_code"] != 0
        or state["binding_sha256"] != sha256(directory / "binding.json")
        or state["result_sha256"] != sha256(directory / "result.json")
        or result["status"] != "passed_production_combined_outer_diagnostic_not_launch_gate"
        or len(result["cases"]) != 54
        or len(result["timing"]) != 6
        or result["parameter_delta_tolerance"] != {"rtol": 1e-5, "atol": 1e-6}
        or result["ns_mode"] != "tpu"
        or result["helpers_sha256"] != frozen["helpers_sha256"]
        or result["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or frozen["source_release_sha256"] != sha256(root / "run-preparation/wave7-release-binding.json")
        or frozen["original_guard_binding_sha256"]
        != sha256(root / "campaigns/lr-fixed-age-0p5m-20261116/validation-binding.json")
        or frozen["archive_sha256"] != sha256(directory / "source.tar")
        or any(sha256(directory / "helpers" / name) != value for name, value in frozen["helpers_sha256"].items())
        or any(
            not all(case["exact"].values()) or not case["parameter_delta"]["within_tolerance"]
            for case in result["cases"]
        )
    ):
        raise ValueError("Completed production probe source, arithmetic or restoration evidence changed")
    return sha256(directory / "state.json")


def definition(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    return ast.dump(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name))


def require_candidate(root: Path, candidate: Path) -> dict:
    """Keep reference paths and settings unchanged, with a separately inventoried combined arm."""
    original = require_source(root, root / "run-preparation/wave7-release-binding.json")
    previous = Path(original["candidate"])
    if candidate.resolve() == previous.resolve():
        raise ValueError("Wave 8 must not mutate the frozen Wave-7 trainer")
    files = candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | original["files_sha256"].keys()
        if files.get(name) != original["files_sha256"].get(name)
    }
    if changed != CHANGED_FILES:
        raise ValueError(f"Combined candidate source scope differs: {sorted(changed ^ CHANGED_FILES)}")
    for name in ("stack_unit_rms", "block_update_stack", "block_update"):
        if definition(candidate / "experiment/block_toner.py", name) != definition(
            previous / "experiment/block_toner.py", name
        ):
            raise ValueError("Uncombined reference update or normalization changed")
    for arm in ARMS:
        config = read(candidate / f"experiment/configs/{arm}_0p5m_seed0.json")
        reference_arm = "outer_leftdiag_fused" if arm == ARMS[-1] else arm
        reference = read(previous / f"experiment/configs/{reference_arm}_0p5m_seed0.json")
        reference["experiment"]["arm"] = arm
        if config != reference:
            raise ValueError("Combined-wave config differs beyond its explicit arm identity")
    cluster = read(candidate / "experiment/cluster.json")
    expected = read(previous / "experiment/cluster.json")
    expected["wave_name"] = "wave8"
    expected["wave"] = [{"arm": arm, "seed": 0, "batch": "0p5m"} for arm in ARMS]
    if cluster != expected:
        raise ValueError("Combined-wave allocation or execution environment changed")
    return files


def freeze(root: Path, candidate: Path) -> dict:
    binding_path, helpers = paths(root)
    archive = root / "run-preparation/source-wave8-combined-outer-20261117.tar"
    if any(path.exists() for path in (binding_path, helpers, archive)):
        raise FileExistsError("Wave-8 identity already has evidence; preserve it without refreezing")
    probe_sha256 = require_probe(root)
    files = require_candidate(root, candidate)
    cpu_path = root / "run-preparation/validation-wave8-cpu.json"
    cpu = read(cpu_path)
    sources = {name: digest for name, digest in files.items() if name.startswith("experiment/") and name.endswith(".py")}
    if (
        cpu["source_sha256"] != sources
        or cpu["candidate_path"] != str(candidate.resolve())
        or cpu["environment"] != ENVIRONMENT
        or cpu["cluster_sha256"] != sha256(candidate / "experiment/cluster.json")
        or cpu["upstream_manifest_sha256"] != sha256(candidate / "experiment/upstream_manifest.json")
        or cpu["config_sha256"] != {arm: sha256(candidate / f"experiment/configs/{arm}_0p5m_seed0.json") for arm in ARMS}
        or cpu["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or cpu["evidence"]["cpu_test_exit_code"] != 0
        or cpu["evidence"]["model_parameter_count"] != 265585152
        or not all(cpu["checks"][gate] is True for gate in ORIGINAL_GATES[:3])
    ):
        raise ValueError("Wave-8 actual CPU evidence is failed or stale")
    original = root / "campaigns/lr-fixed-age-0p5m-20261116/validation-helpers"
    guard = read(original.parent / "validation-binding.json")
    helpers.mkdir(parents=True)
    with tarfile.open(archive, "x") as stream:
        for name in HELPERS:
            source = original / name if name in VALIDATION_HELPERS else Path(__file__).parent / name
            if name in VALIDATION_HELPERS and sha256(source) != guard["helpers_sha256"][name]:
                raise ValueError("Original allocation helper changed before Wave-8 freeze")
            shutil.copy2(source, helpers / name)
            stream.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        for name in sorted(files):
            stream.add(candidate / name, arcname=name, recursive=False)
        stream.add(
            candidate / "artifacts/external-sources",
            arcname="artifacts/external-sources",
            recursive=False,
        )
        stream.add(
            Path(__file__).parents[1] / "ENDEAVOR.md",
            arcname="ENDEAVOR.md",
            recursive=False,
        )
    frozen = {
        "status": "frozen_pending_gpu_validation",
        "root": str(root.resolve()),
        "candidate": str(candidate.resolve()),
        "helpers": str(helpers),
        "helpers_sha256": {name: sha256(helpers / name) for name in HELPERS},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "files_sha256": files,
        "source_sha256": sources,
        "cpu_receipt_sha256": sha256(cpu_path),
        "source_release_sha256": sha256(root / "run-preparation/wave7-release-binding.json"),
        "original_guard_binding_sha256": sha256(original.parent / "validation-binding.json"),
        "production_probe_state_sha256": probe_sha256,
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "arms": ARMS,
        "required_gates": GATES,
        "node": NODE,
        "environment": ENVIRONMENT,
    }
    write(binding_path, frozen)
    return frozen


def binding(root: Path) -> dict:
    path, helpers = paths(root)
    frozen = read(path)
    if (
        Path(__file__).resolve() != (helpers / "wave8.py").resolve()
        or frozen["root"] != str(root.resolve())
        or frozen["helpers"] != str(helpers)
        or set(frozen["helpers_sha256"]) != set(HELPERS)
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or sha256(Path(frozen["archive"])) != frozen["archive_sha256"]
        or frozen["files_sha256"] != require_candidate(root, Path(frozen["candidate"]))
        or frozen["production_probe_state_sha256"] != require_probe(root)
        or frozen["cpu_receipt_sha256"] != sha256(root / "run-preparation/validation-wave8-cpu.json")
        or frozen["source_release_sha256"] != sha256(root / "run-preparation/wave7-release-binding.json")
        or frozen["original_guard_binding_sha256"]
        != sha256(root / "campaigns/lr-fixed-age-0p5m-20261116/validation-binding.json")
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or tuple(frozen["arms"]) != ARMS
        or tuple(frozen["required_gates"]) != GATES
        or frozen["node"] != NODE
        or frozen["environment"] != ENVIRONMENT
    ):
        raise ValueError("Wave-8 frozen candidate/controller/probe binding changed")
    return frozen


def require_validation(root: Path, frozen: dict) -> dict:
    receipt = read(root / "run-preparation/validation-wave8-gpu.json")
    state = read(root / "run-preparation/wave8-validation-state.json")
    source = Path(frozen["candidate"])
    work = root / "validation/wave8_combined_outer"
    commands = []
    for arm in ARMS:
        common = [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "-m",
            "experiment.train",
            "--config",
            str(source / f"experiment/configs/{arm}_0p5m_seed0.json"),
            "--data",
            str(root / "data"),
            "--data-receipt",
            str(root / "data/experiment_receipt.json"),
            "--execution-boundary",
            "22",
        ]
        if arm.endswith("_fused"):
            common.extend(("--verify-diagonal-capture", "--verify-outer-diagonal-updates"))
        full, resumed = work / f"{arm}_continuous", work / f"{arm}_resumed"
        commands.extend(
            [
                [*common, "--out", str(full), "--max-steps", "24"],
                [*common, "--out", str(resumed), "--max-steps", "22"],
                [
                    *common,
                    "--out",
                    str(resumed),
                    "--max-steps",
                    "24",
                    "--resume",
                    str(resumed / "checkpoints/step_000022"),
                ],
            ]
        )
    expected_commands = [
        [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "experiment.gpu_checks",
            "--receipt",
            str(work / "gpu_checks.json"),
        ],
        *commands,
    ]
    if (
        receipt["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or receipt["source_sha256"] != frozen["source_sha256"]
        or receipt["data_receipt_sha256"] != frozen["data_receipt_sha256"]
        or receipt["environment"] != ENVIRONMENT
        or not all(receipt["checks"].get(gate) is True for gate in GATES)
        or receipt["config_sha256"]
        != {arm: sha256(Path(frozen["candidate"]) / f"experiment/configs/{arm}_0p5m_seed0.json") for arm in ARMS}
        or set(receipt["evidence"]["real_data_smokes"]) != set(ARMS)
        or len(receipt["evidence"]["gpu_commands"]) != 13
        or [item["command"] for item in receipt["evidence"]["gpu_commands"]] != expected_commands
        or any(
            command["exit_code"] != 0 or sha256(Path(command["log"])) != command["log_sha256"]
            for command in receipt["evidence"]["gpu_commands"]
        )
    ):
        raise ValueError("Wave-8 actual GPU gates/config/log coverage is incomplete or changed")
    comparisons = source_definitions(
        source / "experiment/records.py", ("compare_tree", "compare_continuation"), {"torch": torch}
    )
    histories = source_definitions(
        source / "experiment/validate.py",
        ("require_cached_root_checkpoint", "require_fixed_age_history", "require_outer_capture_history"),
        {"torch": torch, "json": json},
    )
    for arm, smoke in receipt["evidence"]["real_data_smokes"].items():
        if (
            smoke["cached_root_checkpoint"]["ranks"] != 8
            or smoke["cached_root_checkpoint"]["owned_graph_groups"] <= 0
            or any(smoke[key]["tensor_comparisons"] <= 0 for key in ("continuation", "boundary_continuation"))
            or any(
                item["installation_age"] != 1 or item["updates"] != 24 for item in smoke["fixed_age_history"].values()
            )
        ):
            raise ValueError(f"Wave-8 full-model cached-root/continuation evidence incomplete for {arm}")
        full, resumed = work / f"{arm}_continuous", work / f"{arm}_resumed"
        if smoke["out"] != str(full):
            raise ValueError("Wave-8 smoke trajectory path differs")
        if smoke["cached_root_checkpoint"] != histories["require_cached_root_checkpoint"](
            resumed / "checkpoints/step_000022", 8
        ):
            raise ValueError("Wave-8 owned cached-root checkpoint evidence changed")
        for label, step in (("boundary_continuation", 22), ("continuation", 24)):
            fresh = comparisons["compare_continuation"](
                full / f"checkpoints/step_{step:06d}", resumed / f"checkpoints/step_{step:06d}", 8
            )
            if fresh != smoke[label]:
                raise ValueError("Wave-8 actual all-rank continuation evidence changed")
        config = read(source / f"experiment/configs/{arm}_0p5m_seed0.json")
        for directory in (full, resumed):
            trajectory = "continuous" if directory == full else "resumed"
            if smoke["fixed_age_history"][trajectory] != histories["require_fixed_age_history"](directory):
                raise ValueError("Wave-8 fixed-age installation history differs")
            manifest = read(directory / "run.json")
            rows = [
                json.loads(row)
                for path in sorted(directory.glob("metrics_from_*.jsonl"))
                for row in path.read_text().splitlines()
            ]
            if (
                manifest["status"] != "execution_check_complete"
                or manifest["completed_steps"] != 24
                or manifest["config"] != config
                or manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
                or normalized_sources(source, manifest["source_sha256"]) != executed_sources(source, frozen)
                or [row["step"] for row in rows] != list(range(1, 25))
            ):
                raise ValueError("Wave-8 actual-config trajectory history is incomplete or changed")
            if arm.endswith("_fused") and not all(
                row.get("diagonal_capture_parity_passed") is True
                and row.get("outer_diagonal_update_parity_passed") is True
                for row in rows
            ):
                raise ValueError("Wave-8 actual captured rows or outer-update history is incomplete")
            if arm.endswith("_fused"):
                histories["require_outer_capture_history"](directory)
            if arm == ARMS[-1] and not all(row.get("combined_outer_update_parity_passed") is True for row in rows):
                raise ValueError("Wave-8 actual combined-update history is incomplete")
    return receipt


def validate(root: Path) -> dict:
    frozen = binding(root)
    if local_node() != NODE:
        raise RuntimeError("Wave-8 validation must run on original .139")
    preparation = root / "run-preparation"
    state_path = preparation / "wave8-validation-state.json"
    work = root / "validation/wave8_combined_outer"
    receipt = preparation / "validation-wave8-gpu.json"
    if any(path.exists() for path in (state_path, work, receipt)):
        raise FileExistsError("Wave-8 execution evidence exists; preserve it without retry")
    with node_guard(root, NODE) as signals:
        predecessor = require_completed_wave(root)
        idle_training()
        release = read(root / "run-preparation/wave7-release-binding.json")
        observation = root / "validation/wave8-prevalidation-snapshot.json"
        if observation.exists():
            raise FileExistsError("Wave-8 prevalidation observation already exists; preserve it")
        subprocess.run(
            [
                PYTHON,
                str(Path(release["helpers"]) / "snapshot_wave7_release.py"),
                "--root",
                str(root),
                "--out",
                str(observation),
            ],
            check=True,
            timeout=240,
        )
        if read(observation)["status"] != "complete":
            raise RuntimeError("Wave-8 predecessor process/restoration state is incomplete or unknown")
        devices = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()
        if len(devices) != 8 or any("A100-SXM4-40GB" not in name for name in devices):
            raise ValueError("Wave-8 requires the original eight A100 40GB GPUs")
        state = {
            "status": "preflight",
            "node": NODE,
            "controller_pid": os.getpid(),
            "controller_start_ticks": process(os.getpid())[1],
            "candidate": frozen["candidate"],
            "binding_sha256": sha256(paths(root)[0]),
            "predecessor": predecessor,
            "predecessor_snapshot_sha256": sha256(observation),
        }
        write(state_path, state)
        try:
            state["keeper"] = release_keeper(state, state_path)
            state["status"] = "validating"
            command = [
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
                str(receipt),
            ]
            code = bounded_execution(
                command,
                Path(frozen["candidate"]),
                preparation / "validation-wave8-gpu.log",
                state,
                state_path,
                timeout=14400,
            )
            state["exit_code"] = code
            if code:
                raise RuntimeError("Wave-8 GPU/full-model validation failed; inspect preserved evidence")
            require_validation(root, frozen)
            binding(root)
            state.update(status="passed", validation_sha256=sha256(receipt))
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if state.get("keeper_release_attempted"):
                try:
                    if "cleanup_error" in state:
                        raise RuntimeError("Unverified marked-child cleanup prevents keeper restoration")
                    node_action(root, "restore")
                    state["keeper_restore"] = 0
                except Exception as error:
                    state.update(status="needs_attention", keeper_restore=f"refused: {error}")
            if signals:
                state["deferred_signals"] = signals
            write(state_path, state)
        if type(state.get("keeper_restore")) is not int or state["keeper_restore"] != 0:
            raise RuntimeError("Wave-8 validation keeper restoration failed")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--phase", choices=("freeze", "check", "validate"), required=True)
    args = parser.parse_args()
    if args.phase == "freeze":
        if args.candidate is None:
            parser.error("freeze requires --candidate")
        result = freeze(args.root, args.candidate)
    elif args.phase == "check":
        result = binding(args.root)
    else:
        result = validate(args.root)
    print(json.dumps({"status": result["status"], "root": str(args.root)}))


if __name__ == "__main__":
    main()
