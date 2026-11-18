# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorize only frozen Wave11 production workers and their exact training ranks."""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
PROC_ROOT = Path("/proc")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
ARMS = ("muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused")
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")
MASK = "0,1,2,3,4,5,6,7"
ENVIRONMENT = {
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONFAULTHANDLER": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": MASK,
}
WORKER_MARKER = "OUTPUT_MARIN_WAVE11_WORKER_ID"
CHILD_MARKER = "OUTPUT_MARIN_DIAGNOSTIC_ID"
PREPARATION = "selected-lr-release-preparation.json"
PREPARED_STATUS = "prepared_release_owner_gpu_pending"
CPU_STATUS = "selected_lr_release_cpu_verified_gpu_pending"
STATUS = "frozen_selected_lr_production_after_original_gpu_validation"
RECONSTRUCTION_STATUS = "selected_lr_validation_rederived_absent_exact_keeper_verified"
CHANGES = {
    "experiment/train.py",
    "experiment/selected_lr_release_owner.py",
    "experiment/cluster.json",
    "experiment/tests/test_selected_lr_release_owner.py",
}
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
CHECKS = (
    "actual_full_budget_config",
    "compiled_model",
    "cached_root_checkpoint",
    "boundary_continuation",
    "post_cached_root_continuation",
    "fixed_age_history",
    "all_rank_checkpoint_identity",
)
FUSED_CHECKS = ("captured_diagonal_rows", "outer_diagonal_update_parity")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same_keeper(actual: dict, expected: dict) -> bool:
    before = {int(pid): process for pid, process in expected["generation"].items()}
    after = {int(pid): process for pid, process in actual["generation"].items()}
    return (
        actual["pid"] == expected["pid"]
        and actual["devices"] == expected["devices"]
        and set(before) == set(after)
        and all(
            process.get("status", "S") not in ("Z", "X", "x")
            and after[pid].get("status", "S") not in ("Z", "X", "x")
            and {key: value for key, value in process.items() if key != "status"}
            == {key: value for key, value in after[pid].items() if key != "status"}
            for pid, process in before.items()
        )
    )


def paths(root: Path) -> tuple[Path, Path]:
    return root / "run-preparation/wave11-launch-binding.json", root / "controllers/wave11-launch-20261117/tracking"


def candidate_files(candidate: Path) -> dict:
    pinned, alias = candidate / "artifacts/pinned-sources-materialized", candidate / "artifacts/external-sources"
    if (
        not alias.is_symlink()
        or alias.readlink() != Path("pinned-sources-materialized")
        or not pinned.is_dir()
        or pinned.is_symlink()
        or alias.resolve() != pinned.resolve()
    ):
        raise RuntimeError("Wave11 pinned sources must be the exact candidate-local alias")
    files = {}
    for directory in (candidate / "experiment", pinned):
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError("Wave11 source directory is missing or aliased")
        for path in directory.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise RuntimeError("Wave11 source inventory contains an alias")
            if path.is_file() and (directory == pinned or path.suffix in (".py", ".json")):
                files[str(path.relative_to(candidate))] = sha256(path)
    return files


def require_archive(path: Path, expected: dict) -> None:
    """Verify contents and exact scope, including duplicate names and the pinned alias."""
    with tarfile.open(path) as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected) + 1 or {member.name for member in members} != set(expected) | {
            "artifacts/external-sources"
        }:
            raise RuntimeError("Wave11 source archive inventory differs")
        alias = bundle.getmember("artifacts/external-sources")
        if not alias.issym() or alias.linkname != "pinned-sources-materialized":
            raise RuntimeError("Wave11 archive pinned-source alias differs")
        for name, digest in expected.items():
            if not bundle.getmember(name).isfile():
                raise RuntimeError("Wave11 archived source is aliased")
            with bundle.extractfile(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise RuntimeError("Wave11 archive bytes differ from the tested source")


def commands(root: Path, candidate: Path) -> dict:
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
            str(root / "runs/wave11" / arm),
            "--launch-release",
            str(paths(root)[0]),
        ]
        for node, arm in zip(NODES, ARMS, strict=True)
    }


def worker_command(root: Path, arm: str) -> list[str]:
    return [
        PYTHON,
        str(paths(root)[1] / "selected_lr_production.py"),
        "--root",
        str(root),
        "--phase",
        "worker",
        "--arm",
        arm,
    ]


def cpu_command() -> list[str]:
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


def transformed_train(previous: Path) -> bytes:
    text = (previous / "experiment/train.py").read_text()
    replacements = (
        (
            "from .selected_lr_owner import require_selected_lr_training",
            "from .selected_lr_release_owner import require_selected_lr_release_owner",
        ),
        ("    require_selected_lr_training(args)\n", "    require_selected_lr_release_owner(args)\n"),
        (
            '    parser.add_argument("--out", type=Path, required=True)\n',
            '    parser.add_argument("--out", type=Path, required=True)\n'
            '    parser.add_argument("--launch-release", type=Path, required=True)\n',
        ),
    )
    for before, after in replacements:
        if text.count(before) != 1:
            raise RuntimeError("Wave11 trainer differs from the frozen early ownership dispatch")
        text = text.replace(before, after, 1)
    return text.encode()


def require_release(release_path: Path) -> dict:
    """Bind exact source and CPU archives to accepted original validation, never mutable helpers."""
    prepared_path = TASK / PREPARATION
    prepared = read(prepared_path)
    root = Path(prepared["root"])
    canonical, helpers = paths(root)
    if root != root.resolve() or release_path != canonical:
        raise RuntimeError("Wave11 production requires its canonical separate release")
    release = read(canonical)
    original_path = root / "run-preparation/wave11-selected-lr-validation-binding.json"
    original = read(original_path)
    previous = Path(original["candidate"])
    original_helpers = root / "controllers/wave11-selected-lr-validation-20261117/tracking"
    state_path = root / "run-preparation/wave11-selected-lr-validation-state.json"
    receipt_path = root / "run-preparation/validation-wave11-selected-lr-gpu.json"
    intent_path = original_helpers.parent / "controller.json"
    state, receipt, intent = read(state_path), read(receipt_path), read(intent_path)
    files, previous_files = candidate_files(TASK), candidate_files(previous)
    changed = {name for name in files.keys() | previous_files.keys() if files.get(name) != previous_files.get(name)}
    helper_hashes = release["helpers_sha256"]
    if (
        release["status"] != STATUS
        or release["root"] != str(root)
        or release["candidate"] != str(TASK.resolve())
        or TASK.resolve() == previous.resolve()
        or release["helpers"] != str(helpers)
        or release["archive"] != str(root / "run-preparation/source-wave11-launch-20261117.tar")
        or release["files_sha256"] != files
        or original["files_sha256"] != previous_files
        or changed != CHANGES
        or release["source_sha256"]
        != {name: digest for name, digest in files.items() if name.startswith("experiment/") and name.endswith(".py")}
        or files["experiment/selected_lr_release_owner.py"] != sha256(Path(__file__))
        or release["validation_binding_sha256"] != sha256(original_path)
        or original["status"] != "frozen_pending_completed_wave10_and_selected_lr_gpu_validation"
        or original["root"] != str(root)
        or original["candidate"] != str(previous.resolve())
        or original["helpers"] != str(original_helpers)
        or original["node"] != NODES[0]
        or original["python"] != PYTHON
        or original["environment"] != ENVIRONMENT
        or tuple(release["arms"]) != ARMS
        or tuple(release["nodes"]) != NODES
        or release["environment"] != ENVIRONMENT
        or release["commands"] != commands(root, TASK)
        or release["data_receipt_sha256"] != original["data_receipt_sha256"]
        or release["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or original["predecessor_release_sha256"] != sha256(root / "run-preparation/wave10-launch-binding.json")
        or original["archive_sha256"] != sha256(Path(original["archive"]))
        or not {"run_selected_lr_validation.py", "validate_selected_lr_full_budget.py"}
        <= original["helpers_sha256"].keys()
        or any(
            Path(name).name != name or sha256(original_helpers / name) != digest
            for name, digest in original["helpers_sha256"].items()
        )
        or not {
            "selected_lr_production.py",
            "selected_lr_run_history.py",
            "prepare_selected_lr_release.py",
            "selected_lr_release_owner.py",
            "test_selected_lr_release_owner.py",
        }
        <= helper_hashes.keys()
        or helper_hashes["selected_lr_release_owner.py"] != files["experiment/selected_lr_release_owner.py"]
        or not original["helpers_sha256"].keys() <= helper_hashes.keys()
        or any(helper_hashes[name] != digest for name, digest in original["helpers_sha256"].items())
        or any(Path(name).name != name or sha256(helpers / name) != digest for name, digest in helper_hashes.items())
        or release["archive_sha256"] != sha256(Path(release["archive"]))
        or release["preparation"] != str(prepared_path)
        or release["preparation_sha256"] != sha256(prepared_path)
        or release["cpu_receipt_sha256"] != sha256(Path(release["cpu_receipt"]))
        or (TASK / "experiment/train.py").read_bytes() != transformed_train(previous)
    ):
        raise RuntimeError("Wave11 production release, frozen source or ownership-only scope differs")
    cluster, old_cluster = read(TASK / "experiment/cluster.json"), read(previous / "experiment/cluster.json")
    old_cluster["wave_name"] = "wave11"
    if cluster != old_cluster:
        raise RuntimeError("Wave11 release changed more than its cluster wave name")
    original_cpu, original_controller = read(Path(original["cpu_receipt"])), read(
        Path(original["controller_cpu_receipt"])
    )
    if (
        original["cpu_receipt_sha256"] != sha256(Path(original["cpu_receipt"]))
        or original["controller_cpu_receipt_sha256"] != sha256(Path(original["controller_cpu_receipt"]))
        or original["preparation_sha256"] != sha256(previous / "selected-lr-validation-preparation.json")
        or sha256(TASK / "selected-lr-validation-preparation.json") != original["preparation_sha256"]
        or original_cpu["status"] != "selected_lr_validation_cpu_verified_no_gpu_binding"
        or original_cpu["candidate"] != str(previous)
        or original_cpu["files_sha256"] != previous_files
        or original_cpu["preparation_sha256"] != original["preparation_sha256"]
        or type(original_cpu["exit_code"]) is not int
        or original_cpu["exit_code"] != 0
        or original_cpu["cuda_visible_devices"] != ""
        or original_cpu["command"] != cpu_command()
        or original_cpu["cwd"] != str(previous)
        or original_cpu["log_sha256"] != sha256(Path(original_cpu["log"]))
        or original_cpu["archive_sha256"] != sha256(Path(original_cpu["archive"]))
        or "error" in original_cpu
        or original_controller["status"] != "selected_lr_controller_cpu_verified_no_gpu_binding"
        or original_controller["candidate_cpu_receipt"] != original["cpu_receipt"]
        or original_controller["candidate_cpu_receipt_sha256"] != sha256(Path(original["cpu_receipt"]))
        or original_controller["helpers_sha256"] != original["helpers_sha256"]
        or type(original_controller["exit_code"]) is not int
        or original_controller["exit_code"] != 0
        or original_controller["cuda_visible_devices"] != ""
        or original_controller["log_sha256"] != sha256(Path(original_controller["log"]))
        or original_controller["archive_sha256"] != sha256(Path(original_controller["archive"]))
        or "error" in original_controller
    ):
        raise RuntimeError("Wave11 original candidate/controller CPU evidence differs")
    require_archive(
        Path(original["archive"]),
        previous_files
        | {
            "selected-lr-validation-preparation.json": original["preparation_sha256"],
            "candidate-cpu-receipt.json": original["cpu_receipt_sha256"],
            "controller-cpu-receipt.json": original["controller_cpu_receipt_sha256"],
        }
        | {f"tracking/{name}": digest for name, digest in original["helpers_sha256"].items()},
    )
    reconstruction = release["validation_reconstruction"]
    keeper = reconstruction["keeper"]
    controller_command = [
        PYTHON,
        str(original_helpers / "run_selected_lr_validation.py"),
        "--root",
        str(root),
        "--phase",
        "validate",
    ]
    validator_command = [
        PYTHON,
        "-P",
        str(original_helpers / "validate_selected_lr_full_budget.py"),
        "--root",
        str(root),
        "--candidate",
        str(previous),
        "--work",
        str(root / "validation/wave11_selected_lr_full_budget"),
        "--receipt",
        str(receipt_path),
    ]
    if (
        release["validation_sha256"] != sha256(receipt_path)
        or release["state_sha256"] != sha256(state_path)
        or release["intent_sha256"] != sha256(intent_path)
        or state["status"] != "passed"
        or state["candidate"] != str(previous)
        or state["node"] != NODES[0]
        or state["binding_sha256"] != sha256(original_path)
        or state["validation_sha256"] != sha256(receipt_path)
        or state["command"] != validator_command
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or any(key in state or key in receipt for key in ("error", "errors", "cleanup_error"))
        or intent["status"] != "acknowledged"
        or intent["command"] != controller_command
        or intent["binding_sha256"] != sha256(original_path)
        or intent["pid"] != state["controller_pid"]
        or intent["start_ticks"] != state["controller_start_ticks"]
        or receipt["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or receipt["status"] != "passed_actual_full_budget_config_gates"
        or receipt["candidate"] != str(previous)
        or receipt["root"] != str(root)
        or receipt["environment"] != ENVIRONMENT
        or receipt["world"] != 8
        or receipt["continuous_steps"] != 24
        or receipt["interrupted_steps"] != 22
        or receipt["resumed_total_steps"] != 24
        or receipt["continuation_tolerance"] != {"rtol": 1e-5, "atol": 1e-6}
        or reconstruction["status"] != RECONSTRUCTION_STATUS
        or reconstruction["binding_sha256"] != sha256(original_path)
        or reconstruction["state_sha256"] != sha256(state_path)
        or reconstruction["receipt_sha256"] != sha256(receipt_path)
        or reconstruction["intent_sha256"] != sha256(intent_path)
        or type(reconstruction["commands"]) is not int
        or reconstruction["commands"] != 12
        or type(reconstruction["checks"]) is not int
        or reconstruction["checks"] != 30
        or "error" in reconstruction
        or not same_keeper(keeper, state["restored_keeper"])
        or len(keeper["generation"]) != 9
        or len(keeper["devices"]) != 8
        or len(keeper["occupied"]) != 8
    ):
        raise RuntimeError("Wave11 original validation, terminal restoration or reconstruction differs")
    if len(receipt["trials"]) != 4 or len(receipt["gpu_commands"]) != 12:
        raise RuntimeError("Wave11 original validation requires all twelve commands and thirty gates")
    expected_commands = []
    for arm, trial, planned in zip(ARMS, receipt["trials"], original["plan"]["trials"], strict=True):
        gates = CHECKS + FUSED_CHECKS if arm == ARMS[-1] else CHECKS
        if (
            trial["name"] != arm
            or trial["family"] != arm
            or any(trial[key] != value for key, value in planned.items() if key != "checks")
            or set(trial["checks"]) != set(gates)
            or any(value is not True for value in trial["checks"].values())
        ):
            raise RuntimeError("Wave11 actual-config all-arm gate evidence differs")
        expected_commands.extend((arm, index, command) for index, command in enumerate(planned["commands"]))
    if original["commands"] != [command for _, _, command in expected_commands]:
        raise RuntimeError("Wave11 original command plan differs")
    work = root / "validation/wave11_selected_lr_full_budget"
    for execution, (arm, index, command) in zip(receipt["gpu_commands"], expected_commands, strict=True):
        log = work / f"{arm}_{('continuous', 'interrupted', 'resumed')[index]}.log"
        if (
            execution["trial"] != arm
            or execution["command"] != command
            or type(execution["exit_code"]) is not int
            or execution["exit_code"] != 0
            or execution["log"] != str(log)
            or execution["log_sha256"] != sha256(log)
        ):
            raise RuntimeError("Wave11 original validation command or log differs")
    cpu = read(Path(release["cpu_receipt"]))
    if (
        prepared["status"] != PREPARED_STATUS
        or prepared["root"] != str(root)
        or prepared["candidate"] != str(TASK.resolve())
        or prepared["parent_candidate"] != str(previous)
        or prepared["validation_binding"] != str(original_path)
        or prepared["parent_cpu_receipt"] != original["cpu_receipt"]
        or prepared["parent_controller_cpu_receipt"] != original["controller_cpu_receipt"]
        or prepared["parent_archive"] != original["archive"]
        or prepared["validation_binding_sha256"] != sha256(original_path)
        or prepared["files_sha256"] != files
        or prepared["changed_files"] != sorted(CHANGES)
        or prepared["parent_cpu_receipt_sha256"] != original["cpu_receipt_sha256"]
        or prepared["parent_controller_cpu_receipt_sha256"] != original["controller_cpu_receipt_sha256"]
        or prepared["parent_archive_sha256"] != original["archive_sha256"]
        or prepared["parent_preparation_sha256"] != original["preparation_sha256"]
        or prepared["data_receipt_sha256"] != original["data_receipt_sha256"]
        or any(key in prepared or key in cpu for key in ("error", "errors"))
        or set(prepared["helpers_sha256"])
        != {"prepare_selected_lr_release.py", "selected_lr_release_owner.py", "test_selected_lr_release_owner.py"}
        or any(helper_hashes.get(name) != digest for name, digest in prepared["helpers_sha256"].items())
        or cpu["status"] != CPU_STATUS
        or cpu["root"] != str(root)
        or cpu["candidate"] != str(TASK.resolve())
        or cpu["files_sha256"] != files
        or cpu["helpers_sha256"] != prepared["helpers_sha256"]
        or cpu["preparation_sha256"] != release["preparation_sha256"]
        or cpu["validation_binding_sha256"] != sha256(original_path)
        or cpu["command"] != cpu_command()
        or cpu["cwd"] != str(TASK.resolve())
        or cpu["cuda_visible_devices"] != ""
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["log_sha256"] != sha256(Path(cpu["log"]))
        or cpu["archive_sha256"] != sha256(Path(cpu["archive"]))
    ):
        raise RuntimeError("Wave11 release preparation or executed CPU evidence differs")
    require_archive(
        Path(cpu["archive"]),
        files
        | {PREPARATION: release["preparation_sha256"]}
        | {f"tracking/{name}": digest for name, digest in prepared["helpers_sha256"].items()},
    )
    require_archive(
        Path(release["archive"]),
        files
        | {"preparation.json": release["preparation_sha256"]}
        | {f"tracking/{name}": digest for name, digest in helper_hashes.items()},
    )
    return release


def observed(pid: int) -> dict:
    """Read identity twice so PID reuse, reparenting and exited generations fail closed."""
    directory = PROC_ROOT / str(pid)
    before = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    argv = [entry.decode() for entry in (directory / "cmdline").read_bytes().split(b"\0") if entry]
    environment = dict(
        entry.decode().split("=", 1) for entry in (directory / "environ").read_bytes().split(b"\0") if entry
    )
    cwd = str((directory / "cwd").resolve())
    after = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    if before[19] != after[19] or before[1:4] != after[1:4] or after[0] in ("Z", "X", "x"):
        raise RuntimeError("Wave11 process exited or changed generation during observation")
    return {
        "pid": pid,
        "parent": int(after[1]),
        "group": int(after[2]),
        "session": int(after[3]),
        "start_ticks": after[19],
        "command": argv,
        "environment": environment,
        "cwd": cwd,
    }


def require_process(pid: int, worker_marker: str, child_marker: str | None = None) -> dict:
    process = observed(pid)
    if (
        process["cwd"] != str(TASK.resolve())
        or process["environment"].get(WORKER_MARKER) != worker_marker
        or (child_marker is not None and process["environment"].get(CHILD_MARKER) != child_marker)
        or any(process["environment"].get(name) != value for name, value in ENVIRONMENT.items())
    ):
        raise RuntimeError("Wave11 descendant cwd, inherited markers or eight-GPU environment differs")
    return process


def require_selected_lr_release_owner(args: object) -> None:
    """Accept only the acknowledged one-shot worker's full-budget torchrun ranks before CUDA."""
    if args.launch_release is None:
        raise RuntimeError("Wave11 full-budget training requires a separate production release")
    release = require_release(args.launch_release)
    root, candidate = Path(release["root"]), Path(release["candidate"])
    config = read(args.config)
    arm = config["experiment"]["arm"]
    if arm not in ARMS:
        raise RuntimeError("Wave11 production arm is not released")
    node, command = NODES[ARMS.index(arm)], release["commands"][NODES[ARMS.index(arm)]]
    rank_command = [PYTHON, *command[5:]]
    if (
        args.config != candidate / f"experiment/configs/{arm}_0p5m_seed0.json"
        or args.data != root / "data"
        or args.data_receipt != root / "data/experiment_receipt.json"
        or args.out != root / "runs/wave11" / arm
        or args.max_steps is not None
        or args.resume is not None
        or args.execution_boundary is not None
        or args.verify_diagonal_capture
        or args.verify_outer_diagonal_updates
        or config["training"]["steps"] != 5120
        or config["training"]["tokens"] != 2684354560
        or config["training"]["world_size"] != 8
        or config["training"]["seed"] != 0
        or config["experiment"]["refresh_install_age"] != 1
        or [PYTHON, "-m", "experiment.train", *sys.argv[1:]] != rank_command
        or any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())
        or os.environ.get("WORLD_SIZE") != "8"
        or os.environ.get("LOCAL_WORLD_SIZE") != "8"
        or os.environ.get("RANK") not in tuple(str(rank) for rank in range(8))
        or os.environ.get("LOCAL_RANK") != os.environ.get("RANK")
        or Path.cwd().resolve() != candidate.resolve()
        or node not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("Wave11 training differs from its exact full-budget command or allocation")
    state_path = root / "run-preparation/wave11-workers" / f"{arm}.json"
    intent_path = state_path.with_suffix(".start.json")
    launch_path = root / "run-preparation/wave11-launch-state.json"
    digest = sha256(paths(root)[0])
    deadline = time.monotonic() + 30
    identity = None
    while True:
        state, intent, launch = read(state_path), read(intent_path), read(launch_path)
        current_identity = (state["worker_pid"], state["worker_start_ticks"], state["worker_marker"])
        if identity is None:
            identity = current_identity
        expected = worker_command(root, arm)
        generation = intent["generation"]
        worker = require_process(state["worker_pid"], state["worker_marker"])
        if (
            identity != current_identity
            or not state["worker_marker"]
            or intent["status"] != "acknowledged"
            or any(key in state or key in intent for key in ("error", "errors", "cleanup_error"))
            or intent["pid"] != worker["pid"]
            or intent["start_ticks"] != worker["start_ticks"]
            or any(worker[key] != generation[key] for key in ("pid", "start_ticks", "group", "session", "command"))
            or generation.get("status", "S") in ("Z", "X", "x")
            or worker["start_ticks"] != state["worker_start_ticks"]
            or worker["command"] != expected
            or worker["group"] != worker["pid"]
            or worker["session"] != worker["pid"]
            or state["worker_command"] != expected
            or intent["command"] != expected
            or state["arm"] != arm
            or intent["arm"] != arm
            or state["node"] != node
            or intent["node"] != node
            or state["release_binding_sha256"] != digest
            or intent["release_binding_sha256"] != digest
            or intent["worker_marker"] != state["worker_marker"]
            or launch["worker_markers"][arm] != state["worker_marker"]
            or len(set(launch["worker_markers"].values())) != 4
            or set(launch["worker_markers"]) != set(ARMS)
            or launch["binding_sha256"] != digest
            or launch["status"] not in ("launching", "launched")
            or os.environ.get(WORKER_MARKER) != state["worker_marker"]
        ):
            raise RuntimeError("Wave11 one-shot worker acknowledgment or release generation differs")
        if state["status"] == "running" and "child_start_ticks" in state:
            break
        if state["status"] not in ("preflight", "starting", "running") or time.monotonic() >= deadline:
            raise RuntimeError("Wave11 training has no matching launcher acknowledgment")
        time.sleep(0.05)
    child = require_process(state["child_pid"], state["worker_marker"], state["child_marker"])
    rank = require_process(os.getpid(), state["worker_marker"], state["child_marker"])
    if (
        not state["child_marker"]
        or state["command"] != command
        or child["command"] != command
        or child["start_ticks"] != state["child_start_ticks"]
        or child["parent"] != worker["pid"]
        or child["group"] != child["pid"]
        or child["session"] != child["pid"]
        or rank["command"] not in (rank_command, [PYTHON, "-u", *command[5:]])
        or rank["parent"] != child["pid"]
        or rank["group"] != rank["pid"]
        or rank["session"] != rank["pid"]
        or os.environ.get(CHILD_MARKER) != state["child_marker"]
        or any(
            rank["environment"].get(key) != os.environ.get(key)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        )
    ):
        raise RuntimeError("Wave11 training rank or torchrun child generation differs")
    if (
        observed(worker["pid"]) != worker
        or observed(child["pid"]) != child
        or observed(rank["pid"]) != rank
        or sha256(paths(root)[0]) != digest
        or read(intent_path) != intent
        or any(
            read(state_path)[key] != state[key]
            for key in (
                "status",
                "worker_pid",
                "worker_start_ticks",
                "worker_marker",
                "child_pid",
                "child_start_ticks",
                "child_marker",
                "command",
                "release_binding_sha256",
            )
        )
        or read(launch_path)["worker_markers"] != launch["worker_markers"]
        or read(launch_path)["binding_sha256"] != digest
        or read(launch_path)["status"] not in ("launching", "launched")
    ):
        raise RuntimeError("Wave11 ownership changed while checking the released ancestry")
