# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorize only separately released Wave-10 owned-output worker descendants."""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .launch import ENVIRONMENT, NODES, OUTPUT_ROOT, PYTHON
from .owned_output_validation import ARMS, GATES, VALIDATION_ENVIRONMENT, validation_commands
from .records import sha256
from .reference import TASK

WORKER_MARKER = "OUTPUT_MARIN_WAVE10_WORKER_ID"
MARKER = "OUTPUT_MARIN_DIAGNOSTIC_ID"
NODE = "28.216.17.139"
MASK = "0,1,2,3,4,5,6,7"
PROC_ROOT = Path("/proc")
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
CHANGES = {
    "experiment/execution_owner.py",
    "experiment/owned_production_owner.py",
    "experiment/cluster.json",
    "experiment/tests/test_owned_output_release.py",
}


def observed(pid):
    directory = PROC_ROOT / str(pid)
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    argv = [arg.decode() for arg in (directory / "cmdline").read_bytes().split(b"\0") if arg]
    return fields, argv


def require_inherited(pid, worker_marker, child_marker=None):
    directory = PROC_ROOT / str(pid)
    environment = dict(
        entry.decode().split("=", 1) for entry in (directory / "environ").read_bytes().split(b"\0") if entry
    )
    if (
        (directory / "cwd").resolve() != TASK.resolve()
        or environment.get(WORKER_MARKER) != worker_marker
        or (child_marker is not None and environment.get(MARKER) != child_marker)
        or any(environment.get(name) != value for name, value in VALIDATION_ENVIRONMENT.items())
    ):
        raise RuntimeError("Wave-10 descendant cwd, eight-device environment or inherited markers changed")


def candidate_files(candidate):
    """Retain the owned validator's candidate-local source and pinned-source inventory."""
    materialized = candidate / "artifacts/pinned-sources-materialized"
    alias = candidate / "artifacts/external-sources"
    if (
        not alias.is_symlink()
        or alias.readlink() != Path("pinned-sources-materialized")
        or materialized.is_symlink()
        or not materialized.is_dir()
        or alias.resolve() != materialized.resolve()
    ):
        raise RuntimeError("Wave-10 pinned sources are not candidate-local")
    names = set()
    for directory in (candidate / "experiment", materialized):
        if directory.is_symlink():
            raise RuntimeError("Wave-10 candidate source directory is aliased")
        for path in directory.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise RuntimeError("Wave-10 candidate source inventory contains an alias")
            if path.is_file() and (directory == materialized or path.suffix in (".py", ".json")):
                names.add(str(path.relative_to(candidate)))
    return names


def cpu_command():
    return [
        PYTHON,
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "-p",
        "no:cacheprovider",
        "-q",
        "-m",
        SAFE_MARKERS,
        "experiment/tests",
    ]


def require_cpu_archive(path, files, preparation_sha256):
    """Require the exact tested source inventory, preparation and local pinned alias."""
    expected = files | {"production-preparation.json": preparation_sha256}
    with tarfile.open(path) as bundle:
        members = bundle.getmembers()
        regular = [member for member in members if member.isfile()]
        aliases = [member for member in members if not member.isfile()]
        if (
            len(regular) != len(expected)
            or {member.name for member in regular} != set(expected)
            or len(aliases) != 1
            or aliases[0].name != "artifacts/external-sources"
            or not aliases[0].issym()
            or aliases[0].linkname != "pinned-sources-materialized"
        ):
            raise RuntimeError("Wave-10 CPU archive inventory or candidate-local pinned alias changed")
        for member in regular:
            with bundle.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected[member.name]:
                    raise RuntimeError("Wave-10 CPU archive differs from the tested production source")


def require_release(release_path):
    """Bind an ownership-only release to the completed original owned-output trace evidence.

    The production controller must rederive all-rank checkpoints/histories and
    freshly accept exact restoration before issuing this canonical release.
    Workers recheck its immutable source, command, gate and terminal-state bindings.
    """
    preparation = OUTPUT_ROOT / "run-preparation"
    if release_path != preparation / "wave10-launch-binding.json":
        raise RuntimeError("Owned-output production release has not been issued for this noncanonical path")
    release = json.loads(release_path.read_text())
    original_path = preparation / "owned-output-trace-binding.json"
    original = json.loads(original_path.read_text())
    state_path = preparation / "owned-output-trace-state.json"
    validation_path = preparation / "validation-owned-output-trace-gpu.json"
    state = json.loads(state_path.read_text())
    validation = json.loads(validation_path.read_text())
    helpers = OUTPUT_ROOT / "controllers/wave10-launch-20261117/tracking"
    original_helpers = OUTPUT_ROOT / "controllers/owned-output-trace-validation-20261117/tracking"
    previous = OUTPUT_ROOT / "candidates/owned-output-trace-20261117"
    candidate = OUTPUT_ROOT / "candidates/owned-output-production-isolated-owner-formatted-20261117"
    work = OUTPUT_ROOT / "validation/owned_output_trace"
    files, original_files = release["files_sha256"], original["files_sha256"]
    changed = {name for name in files.keys() | original_files.keys() if files.get(name) != original_files.get(name)}
    commands = validation["evidence"].get("gpu_commands", [])
    expected_validator = [
        PYTHON,
        "-m",
        "experiment.validate",
        "--gpu",
        "--data",
        str(OUTPUT_ROOT / "data"),
        "--data-receipt",
        str(OUTPUT_ROOT / "data/experiment_receipt.json"),
        "--work",
        str(work),
        "--receipt",
        str(validation_path),
    ]
    if (
        release["status"] != "frozen_validated_launch_release"
        or release["root"] != str(OUTPUT_ROOT)
        or release["candidate"] != str(TASK.resolve())
        or release["candidate"] != str(candidate)
        or release["helpers"] != str(helpers)
        or original["status"] != "frozen_pending_completed_ceiling_and_owned_gpu_validation"
        or original["root"] != str(OUTPUT_ROOT)
        or original["candidate"] != str(previous)
        or original["helpers"] != str(original_helpers)
        or original["node"] != NODE
        or tuple(original["arms"]) != ARMS
        or tuple(original["required_gates"]) != GATES
        or changed != CHANGES
        or not files
        or set(files) != candidate_files(TASK)
        or set(original_files) != candidate_files(previous)
        or any(sha256(previous / name) != digest for name, digest in original_files.items())
        or any(sha256(TASK / name) != digest for name, digest in files.items())
        or files.get("experiment/owned_production_owner.py") != sha256(Path(__file__))
        or release["source_sha256"]
        != {name: digest for name, digest in files.items() if name.startswith("experiment/") and name.endswith(".py")}
        or original["source_sha256"]
        != {
            name: digest
            for name, digest in original_files.items()
            if name.startswith("experiment/") and name.endswith(".py")
        }
        or release["validation_binding_sha256"] != sha256(original_path)
        or release["validation_sha256"] != sha256(validation_path)
        or release["state_sha256"] != sha256(state_path)
        or release["data_receipt_sha256"] != original["data_receipt_sha256"]
        or release["data_receipt_sha256"] != sha256(OUTPUT_ROOT / "data/experiment_receipt.json")
        or tuple(release["arms"]) != ARMS
        or tuple(release["nodes"]) != NODES
        or release["environment"] != VALIDATION_ENVIRONMENT
        or original["environment"] != VALIDATION_ENVIRONMENT
        or validation["environment"] != VALIDATION_ENVIRONMENT
        or ENVIRONMENT != VALIDATION_ENVIRONMENT
        or os.environ.get("CUDA_VISIBLE_DEVICES") != MASK
        or any(os.environ.get(key) != value for key, value in VALIDATION_ENVIRONMENT.items())
        or original["archive"] != str(preparation / "source-owned-output-trace-validation-20261117.tar")
        or original["archive_sha256"] != sha256(Path(original["archive"]))
        or not original["helpers_sha256"]
        or "run_owned_output_trace.py" not in original["helpers_sha256"]
        or any(
            Path(name).name != name or sha256(original_helpers / name) != digest
            for name, digest in original["helpers_sha256"].items()
        )
        or not original["helpers_sha256"].keys() <= release["helpers_sha256"].keys()
        or any(release["helpers_sha256"][name] != digest for name, digest in original["helpers_sha256"].items())
        or not {
            "owned_output_production.py",
            "owned_output_production_owner.py",
            "test_owned_output_production_owner.py",
        }
        <= release["helpers_sha256"].keys()
        or release["helpers_sha256"]["owned_output_production_owner.py"] != files["experiment/owned_production_owner.py"]
        or release["helpers_sha256"]["test_owned_output_production_owner.py"]
        != files["experiment/tests/test_owned_output_release.py"]
        or any(
            Path(name).name != name or sha256(helpers / name) != digest
            for name, digest in release["helpers_sha256"].items()
        )
        or release["archive"] != str(preparation / "source-wave10-launch-20261117.tar")
        or release["archive_sha256"] != sha256(Path(release["archive"]))
        or validation["candidate_path"] != str(previous)
        or validation["source_sha256"] != original["source_sha256"]
        or validation["data_receipt_sha256"] != original["data_receipt_sha256"]
        or validation["config_sha256"]
        != {arm: original_files[f"experiment/configs/{arm}_0p5m_seed0.json"] for arm in ARMS}
        or validation["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or not all(validation["checks"].get(gate) is True for gate in GATES)
        or len(commands) != 14
        or original["commands"] != validation_commands(PYTHON, previous, OUTPUT_ROOT, work)
        or [entry["command"] for entry in commands] != original["commands"]
        or any(
            type(entry["exit_code"]) is not int
            or entry["exit_code"] != 0
            or sha256(Path(entry["log"])) != entry["log_sha256"]
            for entry in commands
        )
        or set(validation["evidence"].get("real_data_smokes", {})) != set(ARMS)
        or validation["evidence"]["validation_command"] != expected_validator
        or state["command"] != expected_validator
        or state["node"] != NODE
        or state["candidate"] != str(previous)
        or state["cuda_visible_devices"] != MASK
        or state["status"] != "passed"
        or state["binding_sha256"] != sha256(original_path)
        or state["validation_sha256"] != sha256(validation_path)
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or "error" in state
        or "cleanup_error" in state
        or release["preparation"] != str(TASK / "production-preparation.json")
        or release["preparation_sha256"] != sha256(Path(release["preparation"]))
        or release["cpu_receipt_sha256"] != sha256(Path(release["cpu_receipt"]))
    ):
        raise RuntimeError("Wave-10 ownership-only migration, original owned gates or release identity changed")
    prepared = json.loads(Path(release["preparation"]).read_text())
    cpu = json.loads(Path(release["cpu_receipt"]).read_text())
    reconstruction = release.get("validation_reconstruction", {})
    keeper = reconstruction.get("keeper", {})
    if (
        prepared.get("status") != "prepared_owner_only_no_production_release"
        or prepared.get("candidate") != str(candidate)
        or prepared.get("validation_binding_sha256") != sha256(original_path)
        or prepared.get("files_sha256") != files
        or prepared.get("changed_files") != sorted(CHANGES)
        or prepared.get("owner_sha256") != files["experiment/owned_production_owner.py"]
        or prepared.get("test_sha256") != files["experiment/tests/test_owned_output_release.py"]
        or "error" in prepared
        or cpu.get("status") != "production_cpu_verified_gpu_release_pending"
        or cpu.get("candidate") != str(candidate)
        or cpu.get("preparation_sha256") != release["preparation_sha256"]
        or cpu.get("files_sha256") != files
        or cpu.get("command") != cpu_command()
        or cpu.get("cwd") != str(candidate)
        or cpu.get("cuda_visible_devices") != ""
        or type(cpu.get("exit_code")) is not int
        or cpu["exit_code"] != 0
        or "error" in cpu
        or "errors" in cpu
        or cpu.get("log_sha256") != sha256(Path(cpu["log"]))
        or cpu.get("archive_sha256") != sha256(Path(cpu["archive"]))
        or reconstruction.get("status") != "owned_validation_rederived_absent_exact_keeper_verified"
        or reconstruction.get("receipt_sha256") != sha256(validation_path)
        or reconstruction.get("state_sha256") != sha256(state_path)
        or reconstruction.get("binding_sha256") != sha256(original_path)
        or type(reconstruction.get("commands")) is not int
        or reconstruction["commands"] != 14
        or type(reconstruction.get("checks")) is not int
        or reconstruction["checks"] != 26
        or "error" in reconstruction
        or len(keeper.get("generation", {})) != 9
        or len(keeper.get("devices", [])) != 8
        or len(keeper.get("occupied", [])) != 8
    ):
        raise RuntimeError("Wave-10 preparation, executed CPU evidence or owned validation reconstruction changed")
    require_cpu_archive(Path(cpu["archive"]), files, release["preparation_sha256"])
    return release


def require_production_owner(args, config):
    """Require the exact full-budget command and live, marked eight-rank worker ancestry."""
    release = require_release(args.launch_release)
    arm = config["experiment"]["arm"]
    if arm not in ARMS:
        raise RuntimeError("Wave-10 production arm is not released")
    node = NODES[ARMS.index(arm)]
    release_path = OUTPUT_ROOT / "run-preparation/wave10-launch-binding.json"
    command = [
        PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "experiment.train",
        "--config",
        str(TASK / f"experiment/configs/{arm}_0p5m_seed0.json"),
        "--data",
        str(OUTPUT_ROOT / "data"),
        "--data-receipt",
        str(OUTPUT_ROOT / "data/experiment_receipt.json"),
        "--out",
        str(OUTPUT_ROOT / "runs/wave10" / arm),
        "--launch-release",
        str(release_path),
    ]
    current = os.getpid()
    _, rank_argv = observed(current)
    rank_command = [command[0], *command[5:]]
    if (
        release["commands"][node] != command
        or [PYTHON, "-m", "experiment.train", *sys.argv[1:]] != rank_command
        or rank_argv not in (rank_command, [command[0], "-u", *command[5:]])
        or args.config != TASK / f"experiment/configs/{arm}_0p5m_seed0.json"
        or config != json.loads(args.config.read_text())
        or args.max_steps is not None
        or args.resume is not None
        or args.execution_boundary is not None
        or args.verify_diagonal_capture
        or args.verify_outer_diagonal_updates
        or Path.cwd().resolve() != TASK.resolve()
        or node not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("Wave-10 production differs from its exact full-budget allocation or command")
    state_path = OUTPUT_ROOT / "run-preparation" / f"wave10-worker-{node}.json"
    deadline = time.monotonic() + 30
    generation = None
    while True:
        state = json.loads(state_path.read_text())
        identity = (state["worker_pid"], state["worker_start_ticks"], state["worker_marker"])
        if generation is None:
            generation = identity
        fields, argv = observed(state["worker_pid"])
        expected = [
            PYTHON,
            str(Path(release["helpers"]) / "owned_output_production.py"),
            "--root",
            str(OUTPUT_ROOT),
            "--phase",
            "worker",
            "--arm",
            arm,
        ]
        if (
            fields[0] in ("Z", "X", "x")
            or fields[19] != state["worker_start_ticks"]
            or identity != generation
            or argv != expected
            or state["worker_command"] != expected
            or state["arm"] != arm
            or state["node"] != node
            or state["release_binding_sha256"] != sha256(release_path)
            or not state["worker_marker"]
            or os.environ.get(WORKER_MARKER) != state["worker_marker"]
        ):
            raise RuntimeError("Wave-10 production worker generation, command or release changed")
        require_inherited(state["worker_pid"], state["worker_marker"])
        if state["status"] == "running" and "child_start_ticks" in state:
            break
        if state["status"] not in ("starting", "running") or time.monotonic() >= deadline:
            raise RuntimeError("Wave-10 production has no matching launcher acknowledgment")
        time.sleep(0.05)
    fields, argv = observed(state["child_pid"])
    if (
        fields[0] in ("Z", "X", "x")
        or fields[19] != state["child_start_ticks"]
        or int(fields[1]) != state["worker_pid"]
        or argv != command
        or state["command"] != command
        or not state["child_marker"]
        or os.environ.get(MARKER) != state["child_marker"]
    ):
        raise RuntimeError("Wave-10 production launcher generation, command or parent changed")
    require_inherited(state["child_pid"], state["worker_marker"], state["child_marker"])
    seen = set()
    while current != state["child_pid"]:
        if current <= 1 or current in seen:
            raise RuntimeError("Wave-10 rank is not descended from its marked eight-rank launcher")
        seen.add(current)
        require_inherited(current, state["worker_marker"], state["child_marker"])
        fields, _ = observed(current)
        if fields[0] in ("Z", "X", "x"):
            raise RuntimeError("Wave-10 production descendant exited")
        current = int(fields[1])
