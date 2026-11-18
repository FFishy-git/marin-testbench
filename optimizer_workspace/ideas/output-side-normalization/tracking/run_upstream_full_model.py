# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze and run four-arm validation only after the owned upstream synthetic probe completes."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4

import torch
from analyze_tuning import executed_sources, normalized_sources
from benchmark_diagonal import source_definitions
from launch_wave5 import PYTHON, node_action, read, sha256
from prepare_upstream_validation import CHANGES, require_cpu_source
from run_diagonal_diagnostic import PROCESS_MARKER, finish_child, tagged_processes
from upstream_validation import ARMS, GATES, SYNTHETIC_GATES, require_upstream_capture_history, validation_commands
from validate_wave5 import ENVIRONMENT
from watch_wave5 import process, write
from wave7_release import candidate_inventory, idle_training, local_node, node_guard, release_keeper

NODE = "28.216.17.139"
PROBE = "run-preparation/upstream-outer-gpu-dtype-corrected-20261117"
EXTRA_HELPERS = (
    "run_upstream_full_model.py",
    "prepare_upstream_validation.py",
    "upstream_validation.py",
    "upstream_execution_owner.py",
    "test_upstream_validation.py",
    "test_upstream_full_model_controller.py",
    "prepare_upstream_full_model.py",
    "prepare_upstream_outer.py",
)
PREPARATION_HELPERS = (
    "prepare_upstream_validation.py",
    "upstream_validation.py",
    "upstream_execution_owner.py",
    "test_upstream_validation.py",
)


def cpu_commands(candidate: Path, helpers: Path) -> list[tuple[str, Path, list[str]]]:
    prefix = [
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
    ]
    return [
        ("candidate", candidate, [*prefix, "experiment/tests"]),
        (
            "contracts",
            helpers.parent,
            [*prefix, "tracking/test_upstream_validation.py", "tracking/test_upstream_full_model_controller.py"],
        ),
    ]


def locations(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "run-preparation/upstream-full-model-corrected-binding.json",
        root / "controllers/upstream-full-model-corrected-20261117/tracking",
        root / "run-preparation/upstream-full-model-corrected-state.json",
    )


def require_preparation(candidate: Path) -> dict:
    initial = read(candidate / "validation-preparation.json")
    previous, cpu = require_cpu_source(Path(initial["predecessor_cpu_receipt"]))
    files = candidate_inventory(candidate)
    changed = {name for name in files.keys() | cpu["files_sha256"] if files.get(name) != cpu["files_sha256"].get(name)}
    if (
        initial["status"] != "validator_prepared_not_frozen_no_execution_release"
        or initial["candidate"] != str(candidate.resolve())
        or initial["predecessor_cpu_receipt_sha256"] != sha256(Path(initial["predecessor_cpu_receipt"]))
        or initial["predecessor_files_sha256"] != cpu["files_sha256"]
        or initial["files_sha256"] != files
        or changed != CHANGES
        or set(initial["helpers_sha256"]) != set(PREPARATION_HELPERS)
        or any(sha256(Path(__file__).parent / name) != digest for name, digest in initial["helpers_sha256"].items())
    ):
        raise ValueError("Upstream validator preparation or predecessor evidence changed")
    for arm in ARMS:
        name = f"experiment/configs/{arm}_0p5m_seed0.json"
        if read(candidate / name) != read(previous / name):
            raise ValueError("Upstream validation changed the original full-budget schedule or configuration")
    for name in ("upstream_validation.py", "upstream_execution_owner.py", "test_upstream_validation.py"):
        destination = (
            "experiment/execution_owner.py"
            if name == "upstream_execution_owner.py"
            else f"experiment/tests/{name}" if name.startswith("test_") else f"experiment/{name}"
        )
        if sha256(candidate / destination) != initial["helpers_sha256"][name]:
            raise ValueError("Copied validation contract differs from its preparation helper")
    return initial


def cpu(candidate: Path, out: Path) -> dict:
    """Record candidate and contract tests separately; never grant GPU permission."""
    initial = require_preparation(candidate)
    if out.exists() or out.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Validator CPU verification needs a new output directory")
    out.mkdir(parents=True, exist_ok=False)
    helpers = Path(__file__).parent.resolve()
    hashes = {name: sha256(helpers / name) for name in EXTRA_HELPERS}
    commands = []
    for label, cwd, command in cpu_commands(candidate, helpers):
        log = out / f"{label}.log"
        with log.open("x") as stream:
            code = subprocess.run(
                command,
                cwd=cwd,
                env=os.environ
                | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.pathsep.join((str(candidate), str(helpers)))},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).returncode
        commands.append(
            {"command": command, "cwd": str(cwd), "log": str(log), "log_sha256": sha256(log), "exit_code": code}
        )
        if code:
            raise RuntimeError(f"Validator CPU verification failed; preserve {log}")
    if require_preparation(candidate) != initial or hashes != {name: sha256(helpers / name) for name in hashes}:
        raise ValueError("Validator preparation or controller changed during CPU verification")
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in initial["files_sha256"]:
            bundle.add(candidate / name, arcname=f"candidate/{name}", recursive=False)
        bundle.add(
            candidate / "artifacts/external-sources", arcname="candidate/artifacts/external-sources", recursive=False
        )
        bundle.add(
            candidate / "validation-preparation.json", arcname="candidate/validation-preparation.json", recursive=False
        )
        for name in hashes:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    result = {
        "status": "validator_cpu_verified_no_execution_release",
        "candidate": str(candidate.resolve()),
        "initial_preparation": str(candidate / "validation-preparation.json"),
        "initial_preparation_sha256": sha256(candidate / "validation-preparation.json"),
        "files_sha256": initial["files_sha256"],
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "commands": commands,
        "cuda_visible_devices": "",
        "cpu_pythonpath": os.pathsep.join((str(candidate), str(helpers))),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
    }
    write(out / "receipt.json", result)
    return result


def require_cpu(receipt_path: Path) -> tuple[Path, dict]:
    record = read(receipt_path)
    candidate = Path(record["candidate"])
    initial = require_preparation(candidate)
    helpers = Path(record["helpers"])
    if (
        record["status"] != "validator_cpu_verified_no_execution_release"
        or record["cuda_visible_devices"] != ""
        or record["cpu_pythonpath"] != os.pathsep.join((str(candidate), str(helpers)))
        or record["files_sha256"] != initial["files_sha256"]
        or record["initial_preparation_sha256"] != sha256(candidate / "validation-preparation.json")
        or record["archive_sha256"] != sha256(Path(record["archive"]))
        or set(record["helpers_sha256"]) != set(EXTRA_HELPERS)
        or any(sha256(helpers / name) != digest for name, digest in record["helpers_sha256"].items())
        or len(record["commands"]) != 2
        or [(item["cwd"], item["command"]) for item in record["commands"]]
        != [(str(cwd), command) for _, cwd, command in cpu_commands(candidate, helpers)]
        or any(
            type(item["exit_code"]) is not int
            or item["exit_code"] != 0
            or sha256(Path(item["log"])) != item["log_sha256"]
            for item in record["commands"]
        )
    ):
        raise ValueError("Validator CPU source, command identity or evidence changed")
    return candidate, record


def freeze(root: Path, receipt_path: Path) -> dict:
    candidate, record = require_cpu(receipt_path)
    probe = root / PROBE
    diagnostic = read(probe / "binding.json")
    binding_path, helpers, _ = locations(root)
    archive = root / "run-preparation/source-upstream-full-model-corrected-20261117.tar"
    if any(path.exists() for path in (binding_path, helpers, archive)):
        raise FileExistsError("Full-model validation binding exists; preserve it without refreezing")
    names = tuple(diagnostic["helpers_sha256"]) + EXTRA_HELPERS
    if len(set(names)) != len(names):
        raise ValueError("Full-model helper inventory has duplicate dependencies")
    helpers.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "x") as bundle:
        for name in names:
            source = probe / "helpers" / name if name in diagnostic["helpers_sha256"] else Path(__file__).parent / name
            expected = (
                diagnostic["helpers_sha256"][name]
                if name in diagnostic["helpers_sha256"]
                else record["helpers_sha256"][name]
            )
            if sha256(source) != expected:
                raise ValueError("Full-model dependency changed before freeze")
            shutil.copy2(source, helpers / name)
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        for name in record["files_sha256"]:
            bundle.add(candidate / name, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
        bundle.add(receipt_path, arcname="cpu-receipt.json", recursive=False)
    files = record["files_sha256"]
    result = {
        "status": "frozen_pending_upper_lr_and_synthetic_probe",
        "root": str(root.resolve()),
        "candidate": str(candidate.resolve()),
        "node": NODE,
        "arms": ARMS,
        "required_gates": GATES,
        "cpu_receipt": str(receipt_path.resolve()),
        "cpu_receipt_sha256": sha256(receipt_path),
        "probe": str(probe),
        "probe_binding_sha256": sha256(probe / "binding.json"),
        "helpers": str(helpers),
        "helpers_sha256": {name: sha256(helpers / name) for name in names},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "files_sha256": files,
        "source_sha256": {
            name: digest for name, digest in files.items() if name.startswith("experiment/") and name.endswith(".py")
        },
        "environment": ENVIRONMENT,
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "commands": validation_commands(PYTHON, candidate, root, root / "validation/upstream_full_model_corrected"),
        "scope": "Correctness validation only; no production release, learning-rate choice or performance claim.",
    }
    write(binding_path, result)
    return result


def binding(root: Path) -> dict:
    binding_path, helpers, _ = locations(root)
    frozen = read(binding_path)
    candidate, record = require_cpu(Path(frozen["cpu_receipt"]))
    probe = root / PROBE
    diagnostic = read(probe / "binding.json")
    if (
        Path(__file__).resolve() != (helpers / "run_upstream_full_model.py").resolve()
        or frozen["status"] != "frozen_pending_upper_lr_and_synthetic_probe"
        or frozen["root"] != str(root.resolve())
        or frozen["candidate"] != str(candidate)
        or frozen["node"] != NODE
        or tuple(frozen["arms"]) != ARMS
        or tuple(frozen["required_gates"]) != GATES
        or frozen["helpers"] != str(helpers)
        or frozen["probe"] != str(probe)
        or frozen["cpu_receipt_sha256"] != sha256(Path(frozen["cpu_receipt"]))
        or frozen["files_sha256"] != record["files_sha256"]
        or frozen["source_sha256"]
        != {
            name: digest
            for name, digest in record["files_sha256"].items()
            if name.startswith("experiment/") and name.endswith(".py")
        }
        or frozen["probe_binding_sha256"] != sha256(probe / "binding.json")
        or set(frozen["helpers_sha256"]) != set(diagnostic["helpers_sha256"]) | set(EXTRA_HELPERS)
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or any(frozen["helpers_sha256"][name] != digest for name, digest in diagnostic["helpers_sha256"].items())
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["environment"] != ENVIRONMENT
        or frozen["commands"]
        != validation_commands(PYTHON, candidate, root, root / "validation/upstream_full_model_corrected")
    ):
        raise ValueError("Full-model validation source, plan or frozen helper identity changed")
    return frozen


def require_probe(root: Path, frozen: dict) -> dict:
    """Run the corrected frozen probe's own reconstruction and completed-study checks."""
    probe = root / PROBE
    script = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from run_upstream_outer_gpu import binding,completed_predecessor,require_result
from watch_wave5 import process
from run_diagonal_diagnostic import tagged_processes
from launch_wave5 import read,sha256
root,probe=Path(sys.argv[2]),Path(sys.argv[3])
frozen=read(probe/"binding.json")
bound=binding(root,Path(frozen["cpu_receipt"]),probe)
state=read(probe/"state.json")
actual=read(probe/"result.json")
if (state["status"]!="measured_upstream_probe_not_launch_gate"
    or type(state.get("keeper_restore")) is not int or state["keeper_restore"]!=0
    or state.get("cleanup_exit_code")!=0 or state.get("exit_code")!=0
    or state["result_sha256"]!=sha256(probe/"result.json")
    or state["binding_sha256"]!=sha256(probe/"binding.json")):
    raise ValueError("Upstream probe failed, partial or unrestored")
require_result(bound,actual,state)
parent=process(state["controller_pid"])
if parent is not None and parent[1]==state["controller_start_ticks"] and parent[0]!="Z":
    raise RuntimeError("Original upstream probe controller is still live")
if tagged_processes(state["child_marker"]):
    raise RuntimeError("Upstream probe marked children remain live")
predecessor=completed_predecessor(bound)
print(json.dumps({"probe_state_sha256":sha256(probe/"state.json"),
                 "probe_result_sha256":sha256(probe/"result.json"),"predecessor":predecessor}))
"""
    if frozen["probe_binding_sha256"] != sha256(probe / "binding.json"):
        raise ValueError("Full-model predecessor probe identity changed")
    result = subprocess.run(
        [PYTHON, "-c", script, str(probe / "helpers"), str(root), str(probe)],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return json.loads(result.stdout)


def require_validation(root: Path, frozen: dict) -> dict:
    """Rederive all eight-rank checkpoints and histories before accepting any passed receipt."""
    work = root / "validation/upstream_full_model_corrected"
    receipt = read(root / "run-preparation/validation-upstream-full-model-corrected-gpu.json")
    state = read(locations(root)[2])
    source = Path(frozen["candidate"])
    commands = receipt["evidence"].get("gpu_commands", [])
    if (
        receipt["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or receipt["source_sha256"] != frozen["source_sha256"]
        or receipt["environment"] != ENVIRONMENT
        or receipt["data_receipt_sha256"] != frozen["data_receipt_sha256"]
        or receipt["config_sha256"]
        != {arm: sha256(source / f"experiment/configs/{arm}_0p5m_seed0.json") for arm in ARMS}
        or not all(receipt["checks"].get(gate) is True for gate in GATES)
        or set(receipt["evidence"]["real_data_smokes"]) != set(ARMS)
        or len(commands) != len(frozen["commands"])
        or [item["command"] for item in commands] != frozen["commands"]
        or any(
            type(item["exit_code"]) is not int
            or item["exit_code"] != 0
            or sha256(Path(item["log"])) != item["log_sha256"]
            for item in commands
        )
    ):
        raise ValueError("Upstream validation gates, command coverage or log identity differ")
    synthetic = read(work / "gpu_checks.json")
    if (
        synthetic["world"] != 2
        or synthetic["source_sha256"] != frozen["source_sha256"]
        or not all(synthetic["checks"].get(gate) is True for gate in SYNTHETIC_GATES)
    ):
        raise ValueError("Upstream synthetic source, rank or gate coverage differs")
    comparisons = source_definitions(
        source / "experiment/records.py", ("compare_tree", "compare_continuation"), {"torch": torch}
    )
    histories = source_definitions(
        source / "experiment/validate.py",
        ("require_cached_root_checkpoint", "require_fixed_age_history", "require_outer_capture_history"),
        {"torch": torch, "json": json},
    )
    for arm in ARMS:
        smoke = receipt["evidence"]["real_data_smokes"][arm]
        full, resumed = work / f"{arm}_continuous", work / f"{arm}_resumed"
        if smoke["out"] != str(full) or smoke["cached_root_checkpoint"] != histories["require_cached_root_checkpoint"](
            resumed / "checkpoints/step_000022", 8
        ):
            raise ValueError("Upstream owned cached-root checkpoint identity differs")
        for label, step in (("boundary_continuation", 22), ("continuation", 24)):
            config = read(source / f"experiment/configs/{arm}_0p5m_seed0.json")
            for directory in (full, resumed):
                require_checkpoint_identity(directory / f"checkpoints/step_{step:06d}", config, source, frozen, step)
            actual = comparisons["compare_continuation"](
                full / f"checkpoints/step_{step:06d}", resumed / f"checkpoints/step_{step:06d}", 8
            )
            if smoke[label] != actual or actual["tensor_comparisons"] <= 0:
                raise ValueError("Upstream all-rank continuation evidence changed")
        config = read(source / f"experiment/configs/{arm}_0p5m_seed0.json")
        for label, directory in (("continuous", full), ("resumed", resumed)):
            manifest = read(directory / "run.json")
            if (
                manifest["status"] != "execution_check_complete"
                or manifest["completed_steps"] != 24
                or manifest["config"] != config
                or manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
                or normalized_sources(source, manifest["source_sha256"]) != executed_sources(source, frozen)
                or smoke["fixed_age_history"][label] != histories["require_fixed_age_history"](directory)
            ):
                raise ValueError("Upstream actual trajectory, source or fixed-age history changed")
            if arm == ARMS[1]:
                histories["require_outer_capture_history"](directory)
            elif arm in ARMS[2:]:
                require_upstream_capture_history(directory, arm)
    return receipt


def require_checkpoint_identity(directory: Path, config: dict, source: Path, frozen: dict, step: int) -> None:
    """Check expected identity independently on every rank, not just pairwise checkpoint agreement."""
    completion = read(directory / "complete.json")
    if completion["step"] != step or completion["world"] != 8:
        raise ValueError("Upstream checkpoint completion step or world differs")
    expected_config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    expected_sources = executed_sources(source, frozen)
    for rank in range(8):
        saved = torch.load(directory / f"rank_{rank}.pt", map_location="cpu", weights_only=False)
        identity = saved["identity"]
        if (
            saved["rank"] != rank
            or saved["world"] != 8
            or saved["step"] != step
            or saved["config"] != config
            or saved["next_global_sequence"] != step * config["training"]["global_batch_sequences"]
            or identity["schema"] != 1
            or identity["config_sha256"] != expected_config_hash
            or identity["data_receipt_sha256"] != frozen["data_receipt_sha256"]
            or normalized_sources(source, identity["source_sha256"]) != expected_sources
        ):
            raise ValueError(f"Upstream checkpoint expected rank/config/source identity differs for rank {rank}")


def bounded_validator(command: list[str], candidate: Path, log: Path, state: dict, state_path: Path) -> int:
    """Acknowledge the validator generation before CUDA ownership can be granted."""
    marker, generation = uuid4().hex, {}
    state.update(command=command, child_marker=marker, log=str(log), spawn_status="intent")
    write(state_path, state)
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=candidate,
            env=os.environ | ENVIRONMENT | {PROCESS_MARKER: marker},
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            observed = process(child.pid)
            if observed is None or observed[0] == "Z":
                raise RuntimeError("Upstream validator exited before generation acknowledgment")
            state.update(child_pid=child.pid, child_start_ticks=observed[1], spawn_status="acknowledged")
            write(state_path, state)
            deadline = time.monotonic() + 14400
            while True:
                generation.update(tagged_processes(marker))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, 14400)
                try:
                    return child.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            try:
                finish_child(child, marker, generation)
                state["cleanup_exit_code"] = child.returncode
            except Exception as error:
                state["cleanup_error"] = str(error)
                raise
            finally:
                state["child_generation"] = generation
                write(state_path, state)


def validate(root: Path) -> dict:
    frozen = binding(root)
    if local_node() != NODE:
        raise RuntimeError("Upstream full-model validation requires its original .139 allocation")
    _, _, state_path = locations(root)
    work = root / "validation/upstream_full_model_corrected"
    receipt = root / "run-preparation/validation-upstream-full-model-corrected-gpu.json"
    if any(path.exists() for path in (state_path, work, receipt)):
        raise FileExistsError("Full-model validation has evidence; preserve it without retry")
    with node_guard(root, NODE) as signals:
        predecessor = require_probe(root, frozen)
        idle_training()
        devices = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()
        if len(devices) != 8 or any("A100-SXM4-40GB" not in name for name in devices):
            raise ValueError("Upstream full-model validation requires the original eight A100 GPUs")
        state = {
            "status": "preflight",
            "node": NODE,
            "candidate": frozen["candidate"],
            "controller_pid": os.getpid(),
            "controller_start_ticks": process(os.getpid())[1],
            "binding_sha256": sha256(locations(root)[0]),
            "predecessor": predecessor,
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
            code = bounded_validator(
                command,
                Path(frozen["candidate"]),
                root / "run-preparation/validation-upstream-full-model-corrected-gpu.log",
                state,
                state_path,
            )
            state["exit_code"] = code
            if code:
                raise RuntimeError("Upstream actual full-model validation failed")
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
            raise RuntimeError("Upstream full-model keeper restoration failed")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("cpu", "freeze", "check", "validate"), required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "cpu":
        if args.candidate is None or args.out is None:
            parser.error("cpu requires --candidate and --out")
        result = cpu(args.candidate, args.out)
    elif args.phase == "freeze":
        if args.cpu_receipt is None:
            parser.error("freeze requires --cpu-receipt")
        result = freeze(args.root, args.cpu_receipt)
    elif args.phase == "check":
        result = binding(args.root)
    else:
        result = validate(args.root)
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
