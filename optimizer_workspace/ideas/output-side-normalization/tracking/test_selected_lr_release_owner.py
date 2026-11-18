# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise exact released ancestry using persisted receipts and a synthetic /proc."""

import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import selected_lr_release_owner as owner


def record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def archive(path: Path, entries: dict[str, Path], candidate: Path) -> None:
    with tarfile.open(path, "w") as bundle:
        for name, source in entries.items():
            bundle.add(source, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)


def process(
    proc: Path, pid: int, parent: int, ticks: str, command: list, cwd: Path, environment: dict, group=None, session=None
):
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[0:4] = ["S", str(parent), str(group or pid), str(session or pid)]
    fields[19] = ticks
    (directory / "stat").write_text(f"{pid} (wave11 fixture) {' '.join(fields)}\n")
    (directory / "cmdline").write_bytes(("\0".join(command) + "\0").encode())
    (directory / "environ").write_bytes(
        ("\0".join(f"{key}={value}" for key, value in environment.items()) + "\0").encode()
    )
    link = directory / "cwd"
    link.unlink(missing_ok=True)
    link.symlink_to(cwd, target_is_directory=True)


@pytest.fixture
def released(tmp_path, monkeypatch):
    root, previous, candidate = tmp_path / "root", tmp_path / "original", tmp_path / "released"
    preparation = root / "run-preparation"
    preparation.mkdir(parents=True)
    experiment = previous / "experiment"
    (experiment / "tests").mkdir(parents=True)
    (experiment / "train.py").write_text(
        "from .selected_lr_owner import require_selected_lr_training\n\n"
        "def main():\n"
        '    parser.add_argument("--out", type=Path, required=True)\n'
        "    require_selected_lr_training(args)\n"
        "    numerical_body = 42\n"
    )
    (experiment / "selected_lr_owner.py").write_text("ORIGINAL_VALIDATION_OWNER = True\n")
    (experiment / "production_owner.py").write_text("PRESERVED_HISTORICAL_OWNER = True\n")
    (experiment / "optimizer.py").write_text("ORIGINAL_NUMERICS = 42\n")
    record(experiment / "cluster.json", {"wave_name": "validation", "nodes": list(owner.NODES), "gpus_per_run": 8})
    for arm in owner.ARMS:
        record(
            experiment / f"configs/{arm}_0p5m_seed0.json",
            {
                "experiment": {"arm": arm, "refresh_install_age": 1},
                "training": {"steps": 5120, "tokens": 2684354560, "world_size": 8, "seed": 0},
                "optimizer": {"lr": 0.064},
            },
        )
    pinned = previous / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    (pinned / "original.bin").write_bytes(b"original pinned numerical bytes")
    (previous / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    record(previous / "selected-lr-validation-preparation.json", {"status": "original_preparation"})
    shutil.copytree(previous, candidate, symlinks=True)
    (candidate / "experiment/train.py").write_bytes(owner.transformed_train(previous))
    record(candidate / "experiment/cluster.json", {"wave_name": "wave11", "nodes": list(owner.NODES), "gpus_per_run": 8})
    (candidate / "experiment/selected_lr_release_owner.py").write_bytes(Path(owner.__file__).read_bytes())
    (candidate / "experiment/tests/test_selected_lr_release_owner.py").write_bytes(Path(__file__).read_bytes())
    files, previous_files = owner.candidate_files(candidate), owner.candidate_files(previous)
    data = root / "data/experiment_receipt.json"
    record(data, {"status": "verified"})
    record(preparation / "wave10-launch-binding.json", {"status": "preserved_wave10"})
    original_helpers = root / "controllers/wave11-selected-lr-validation-20261117/tracking"
    original_helpers.mkdir(parents=True)
    for name in ("run_selected_lr_validation.py", "validate_selected_lr_full_budget.py"):
        (original_helpers / name).write_text("# immutable original validation helper\n")
    original_hashes = {path.name: owner.sha256(path) for path in original_helpers.iterdir()}
    cpu_log, controller_log = preparation / "original-cpu.log", preparation / "original-controller.log"
    cpu_log.write_text("original candidate CPU passed\n")
    controller_log.write_text("original controller CPU passed\n")
    cpu_path, controller_path = preparation / "original-cpu.json", preparation / "original-controller.json"
    cpu_archive, controller_archive = preparation / "original-cpu.tar", preparation / "original-controller.tar"
    archive(cpu_archive, {name: previous / name for name in previous_files}, previous)
    with tarfile.open(controller_archive, "w") as bundle:
        for name in original_hashes:
            bundle.add(original_helpers / name, arcname=f"tracking/{name}", recursive=False)
    original_cpu = {
        "status": "selected_lr_validation_cpu_verified_no_gpu_binding",
        "candidate": str(previous),
        "files_sha256": previous_files,
        "preparation_sha256": owner.sha256(previous / "selected-lr-validation-preparation.json"),
        "exit_code": 0,
        "cuda_visible_devices": "",
        "command": owner.cpu_command(),
        "cwd": str(previous),
        "log": str(cpu_log),
        "log_sha256": owner.sha256(cpu_log),
        "archive": str(cpu_archive),
        "archive_sha256": owner.sha256(cpu_archive),
    }
    record(cpu_path, original_cpu)
    record(
        controller_path,
        {
            "status": "selected_lr_controller_cpu_verified_no_gpu_binding",
            "candidate_cpu_receipt": str(cpu_path),
            "candidate_cpu_receipt_sha256": owner.sha256(cpu_path),
            "helpers_sha256": original_hashes,
            "exit_code": 0,
            "cuda_visible_devices": "",
            "log": str(controller_log),
            "log_sha256": owner.sha256(controller_log),
            "archive": str(controller_archive),
            "archive_sha256": owner.sha256(controller_archive),
        },
    )
    original_archive = preparation / "source-wave11-selected-lr-validation-20261117.tar"
    archive(
        original_archive,
        {name: previous / name for name in previous_files}
        | {
            "selected-lr-validation-preparation.json": previous / "selected-lr-validation-preparation.json",
            "candidate-cpu-receipt.json": cpu_path,
            "controller-cpu-receipt.json": controller_path,
        }
        | {f"tracking/{name}": original_helpers / name for name in original_hashes},
        previous,
    )
    trials, executed = [], []
    work = root / "validation/wave11_selected_lr_full_budget"
    work.mkdir(parents=True)
    for arm in owner.ARMS:
        gates = owner.CHECKS + owner.FUSED_CHECKS if arm == owner.ARMS[-1] else owner.CHECKS
        trial_commands = []
        for index, suffix in enumerate(("continuous", "interrupted", "resumed")):
            command = [owner.PYTHON, "-m", "original-validation", "--arm", arm, "--trajectory", suffix]
            trial_commands.append(command)
            log = work / f"{arm}_{suffix}.log"
            log.write_text(f"original {arm} trajectory {index} passed\n")
            executed.append(
                {"trial": arm, "command": command, "exit_code": 0, "log": str(log), "log_sha256": owner.sha256(log)}
            )
        trials.append({"name": arm, "family": arm, "commands": trial_commands, "checks": dict.fromkeys(gates, False)})
    original_path = preparation / "wave11-selected-lr-validation-binding.json"
    original = {
        "status": "frozen_pending_completed_wave10_and_selected_lr_gpu_validation",
        "root": str(root),
        "candidate": str(previous),
        "helpers": str(original_helpers),
        "helpers_sha256": original_hashes,
        "files_sha256": previous_files,
        "node": owner.NODES[0],
        "python": owner.PYTHON,
        "environment": owner.ENVIRONMENT,
        "data_receipt_sha256": owner.sha256(data),
        "predecessor_release_sha256": owner.sha256(preparation / "wave10-launch-binding.json"),
        "archive": str(original_archive),
        "archive_sha256": owner.sha256(original_archive),
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": owner.sha256(cpu_path),
        "controller_cpu_receipt": str(controller_path),
        "controller_cpu_receipt_sha256": owner.sha256(controller_path),
        "preparation_sha256": owner.sha256(previous / "selected-lr-validation-preparation.json"),
        "commands": [entry["command"] for entry in executed],
        "plan": {"trials": trials},
    }
    record(original_path, original)
    receipt_path = preparation / "validation-wave11-selected-lr-gpu.json"
    receipt = {
        "status": "passed_actual_full_budget_config_gates",
        "root": str(root),
        "candidate": str(previous),
        "owner": {"controller_pid": 111, "controller_start_ticks": "112"},
        "environment": owner.ENVIRONMENT,
        "world": 8,
        "continuous_steps": 24,
        "interrupted_steps": 22,
        "resumed_total_steps": 24,
        "continuation_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "trials": [trial | {"checks": dict.fromkeys(trial["checks"], True)} for trial in trials],
        "gpu_commands": executed,
    }
    record(receipt_path, receipt)
    state_path = preparation / "wave11-selected-lr-validation-state.json"
    keeper = {
        "pid": 30000,
        "generation": {str(pid): {"pid": pid, "start_ticks": str(pid), "group": pid} for pid in range(30000, 30009)},
        "devices": [[str(rank), f"GPU-{rank}", "A100-SXM4-40GB"] for rank in range(8)],
        "occupied": [[str(30001 + rank), f"GPU-{rank}"] for rank in range(8)],
    }
    state = {
        "status": "passed",
        "candidate": str(previous),
        "node": owner.NODES[0],
        "controller_pid": 111,
        "controller_start_ticks": "112",
        "binding_sha256": owner.sha256(original_path),
        "validation_sha256": owner.sha256(receipt_path),
        "command": [
            owner.PYTHON,
            "-P",
            str(original_helpers / "validate_selected_lr_full_budget.py"),
            "--root",
            str(root),
            "--candidate",
            str(previous),
            "--work",
            str(work),
            "--receipt",
            str(receipt_path),
        ],
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "restored_keeper": keeper,
    }
    record(state_path, state)
    intent_path = original_helpers.parent / "controller.json"
    record(
        intent_path,
        {
            "status": "acknowledged",
            "command": [
                owner.PYTHON,
                str(original_helpers / "run_selected_lr_validation.py"),
                "--root",
                str(root),
                "--phase",
                "validate",
            ],
            "binding_sha256": owner.sha256(original_path),
            "pid": 111,
            "start_ticks": "112",
        },
    )
    release_path, helpers = owner.paths(root)
    helpers.mkdir(parents=True)
    for name in ("selected_lr_production.py", "selected_lr_run_history.py", "prepare_selected_lr_release.py"):
        (helpers / name).write_text("# immutable production helper\n")
    for name in original_hashes:
        (helpers / name).write_bytes((original_helpers / name).read_bytes())
    (helpers / "selected_lr_release_owner.py").write_bytes(Path(owner.__file__).read_bytes())
    (helpers / "test_selected_lr_release_owner.py").write_bytes(Path(__file__).read_bytes())
    hashes = {path.name: owner.sha256(path) for path in helpers.iterdir()}
    prepared_path = candidate / owner.PREPARATION
    prepared = {
        "status": "prepared_release_owner_gpu_pending",
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "validation_binding": str(original_path),
        "validation_binding_sha256": owner.sha256(original_path),
        "files_sha256": files,
        "changed_files": sorted(owner.CHANGES),
        "parent_cpu_receipt_sha256": original["cpu_receipt_sha256"],
        "parent_cpu_receipt": original["cpu_receipt"],
        "parent_controller_cpu_receipt_sha256": original["controller_cpu_receipt_sha256"],
        "parent_controller_cpu_receipt": original["controller_cpu_receipt"],
        "parent_archive_sha256": original["archive_sha256"],
        "parent_archive": original["archive"],
        "parent_preparation_sha256": original["preparation_sha256"],
        "data_receipt_sha256": original["data_receipt_sha256"],
        "helpers_sha256": {
            name: hashes[name]
            for name in (
                "prepare_selected_lr_release.py",
                "selected_lr_release_owner.py",
                "test_selected_lr_release_owner.py",
            )
        },
    }
    record(prepared_path, prepared)
    cpu_path, cpu_archive, cpu_log = (
        preparation / "release-cpu.json",
        preparation / "release-cpu.tar",
        preparation / "release-cpu.log",
    )
    cpu_log.write_text("release candidate CPU passed\n")
    archive(
        cpu_archive,
        {name: candidate / name for name in files}
        | {owner.PREPARATION: prepared_path}
        | {f"tracking/{name}": helpers / name for name in prepared["helpers_sha256"]},
        candidate,
    )
    cpu = {
        "status": "selected_lr_release_cpu_verified_gpu_pending",
        "root": str(root),
        "candidate": str(candidate),
        "files_sha256": files,
        "helpers_sha256": prepared["helpers_sha256"],
        "preparation_sha256": owner.sha256(prepared_path),
        "validation_binding_sha256": owner.sha256(original_path),
        "command": owner.cpu_command(),
        "cwd": str(candidate),
        "cuda_visible_devices": "",
        "exit_code": 0,
        "log": str(cpu_log),
        "log_sha256": owner.sha256(cpu_log),
        "archive": str(cpu_archive),
        "archive_sha256": owner.sha256(cpu_archive),
    }
    record(cpu_path, cpu)
    release_archive = preparation / "source-wave11-launch-20261117.tar"
    archive(
        release_archive,
        {name: candidate / name for name in files}
        | {"preparation.json": prepared_path}
        | {f"tracking/{name}": helpers / name for name in hashes},
        candidate,
    )
    release = {
        "status": owner.STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "files_sha256": files,
        "source_sha256": {
            name: digest for name, digest in files.items() if name.startswith("experiment/") and name.endswith(".py")
        },
        "validation_binding_sha256": owner.sha256(original_path),
        "validation_sha256": owner.sha256(receipt_path),
        "state_sha256": owner.sha256(state_path),
        "intent_sha256": owner.sha256(intent_path),
        "validation_reconstruction": {
            "status": owner.RECONSTRUCTION_STATUS,
            "binding_sha256": owner.sha256(original_path),
            "receipt_sha256": owner.sha256(receipt_path),
            "state_sha256": owner.sha256(state_path),
            "intent_sha256": owner.sha256(intent_path),
            "commands": 12,
            "checks": 30,
            "keeper": keeper,
        },
        "preparation": str(prepared_path),
        "preparation_sha256": owner.sha256(prepared_path),
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": owner.sha256(cpu_path),
        "archive": str(release_archive),
        "archive_sha256": owner.sha256(release_archive),
        "arms": list(owner.ARMS),
        "nodes": list(owner.NODES),
        "environment": owner.ENVIRONMENT,
        "data_receipt_sha256": owner.sha256(data),
        "commands": owner.commands(root, candidate),
    }
    record(release_path, release)
    monkeypatch.setattr(owner, "TASK", candidate)
    proc = tmp_path / "proc"
    monkeypatch.setattr(owner, "PROC_ROOT", proc)
    monkeypatch.chdir(candidate)
    return SimpleNamespace(
        root=root,
        candidate=candidate,
        previous=previous,
        proc=proc,
        release=release,
        release_path=release_path,
        receipt=receipt,
        receipt_path=receipt_path,
        state=state,
        state_path=state_path,
        cpu=cpu,
        cpu_path=cpu_path,
        prepared=prepared,
        prepared_path=prepared_path,
    )


def bind_worker(evidence, arm, rank, monkeypatch):
    command = evidence.release["commands"][owner.NODES[owner.ARMS.index(arm)]]
    state_path = evidence.root / "run-preparation/wave11-workers" / f"{arm}.json"
    state = {
        "status": "running",
        "arm": arm,
        "node": owner.NODES[owner.ARMS.index(arm)],
        "worker_pid": 20001,
        "worker_start_ticks": "71",
        "worker_marker": f"worker-{arm}",
        "worker_command": owner.worker_command(evidence.root, arm),
        "release_binding_sha256": owner.sha256(evidence.release_path),
        "child_pid": 20002,
        "child_start_ticks": "72",
        "child_marker": f"child-{arm}",
        "command": command,
    }
    environment = owner.ENVIRONMENT | {
        owner.WORKER_MARKER: state["worker_marker"],
        owner.CHILD_MARKER: state["child_marker"],
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    process(evidence.proc, 20001, 1, "71", state["worker_command"], evidence.candidate, environment)
    process(evidence.proc, 20002, 20001, "72", command, evidence.candidate, environment)
    process(
        evidence.proc,
        os.getpid(),
        20002,
        "73",
        [owner.PYTHON, "-u", *command[5:]],
        evidence.candidate,
        environment,
    )
    record(state_path, state)
    intent = {
        "status": "acknowledged",
        "pid": 20001,
        "start_ticks": "71",
        "command": state["worker_command"],
        "arm": arm,
        "node": state["node"],
        "worker_marker": state["worker_marker"],
        "release_binding_sha256": state["release_binding_sha256"],
        "generation": owner.observed(20001),
    }
    record(state_path.with_suffix(".start.json"), intent)
    launch = {
        "status": "launching",
        "binding_sha256": state["release_binding_sha256"],
        "worker_markers": {name: f"worker-{name}" for name in owner.ARMS},
    }
    record(evidence.root / "run-preparation/wave11-launch-state.json", launch)
    monkeypatch.setattr(owner.subprocess, "check_output", lambda *args, **kwargs: state["node"])
    monkeypatch.setattr(owner.sys, "argv", [str(evidence.candidate / "experiment/train.py"), *command[7:]])
    args = SimpleNamespace(
        launch_release=evidence.release_path,
        config=Path(command[8]),
        data=evidence.root / "data",
        data_receipt=evidence.root / "data/experiment_receipt.json",
        out=evidence.root / "runs/wave11" / arm,
        max_steps=None,
        resume=None,
        execution_boundary=None,
        verify_diagonal_capture=False,
        verify_outer_diagonal_updates=False,
    )
    return state_path, state, intent, environment, args


@pytest.mark.parametrize(
    "arm,rank", (("muont", 0), ("inner_outer_leftdiag", 2), ("outer_leftdiag", 5), ("outer_leftdiag_fused", 7))
)
def test_release_authorizes_exact_each_node_rank_without_writing(released, monkeypatch, arm, rank):
    state_path, _, _, _, args = bind_worker(released, arm, rank, monkeypatch)
    before = {path: path.read_bytes() for path in (released.release_path, released.state_path, state_path)}
    assert owner.require_selected_lr_release_owner(args) is None
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not args.out.exists()


def test_release_accepts_restored_keeper_scheduler_status_change_without_generation_change(released):
    keeper = json.loads(json.dumps(released.release["validation_reconstruction"]["keeper"]))
    for process in keeper["generation"].values():
        process["status"] = "R"
    released.release["validation_reconstruction"]["keeper"] = keeper
    record(released.release_path, released.release)
    assert owner.require_release(released.release_path) == released.release
    assert not (released.root / "runs/wave11").exists()


@pytest.mark.parametrize(
    "fault",
    (
        "intent_ticks",
        "intent_group",
        "intent_command",
        "worker_ticks",
        "child_ticks",
        "rank_parent",
        "rank_group",
        "rank_session",
        "worker_mask",
        "child_marker",
        "rank_mask",
        "rank_world",
        "node",
        "budget",
        "resume",
        "capture",
        "output",
        "launch_marker",
    ),
)
def test_release_rejects_changed_acknowledgment_or_live_execution(released, monkeypatch, fault):
    state_path, _, intent, environment, args = bind_worker(released, "muont", 0, monkeypatch)
    if fault.startswith("intent_"):
        key = fault.removeprefix("intent_")
        intent["generation"][{"ticks": "start_ticks", "group": "group", "command": "command"}[key]] = "changed"
        record(state_path.with_suffix(".start.json"), intent)
    elif fault in (
        "worker_ticks",
        "child_ticks",
        "worker_mask",
        "child_marker",
        "rank_parent",
        "rank_group",
        "rank_session",
        "rank_mask",
    ):
        pid = 20001 if fault.startswith("worker") else 20002 if fault.startswith("child") else os.getpid()
        old = owner.observed(pid)
        parent, ticks, inherited = old["parent"], old["start_ticks"], dict(environment)
        if fault.endswith("ticks"):
            ticks = "reused"
        elif fault.endswith("mask"):
            inherited["CUDA_VISIBLE_DEVICES"] = "0"
        elif fault == "child_marker":
            inherited[owner.CHILD_MARKER] = "unowned"
        elif fault == "rank_parent":
            parent = 1
        process(
            released.proc,
            pid,
            parent,
            ticks,
            old["command"],
            released.candidate,
            inherited,
            20002 if fault == "rank_group" else old["group"],
            20002 if fault == "rank_session" else old["session"],
        )
    elif fault == "rank_world":
        monkeypatch.setenv("WORLD_SIZE", "4")
    elif fault == "node":
        monkeypatch.setattr(owner.subprocess, "check_output", lambda *args, **kwargs: owner.NODES[-1])
    elif fault == "budget":
        args.max_steps = 24
    elif fault == "resume":
        args.resume = released.root / "old-checkpoint"
    elif fault == "capture":
        args.verify_outer_diagonal_updates = True
    elif fault == "output":
        args.out = released.root / "runs/other-wave/muont"
    else:
        path = released.root / "run-preparation/wave11-launch-state.json"
        launch = owner.read(path)
        launch["worker_markers"]["muont"] = "replacement-worker"
        record(path, launch)
    with pytest.raises(RuntimeError):
        owner.require_selected_lr_release_owner(args)
    assert not (released.root / "runs/wave11").exists()


@pytest.mark.parametrize(
    "fault",
    (
        "failed_gate",
        "missing_command",
        "receipt_error",
        "state_error",
        "reconstruction",
        "keeper",
        "cpu_command",
        "cpu_error",
        "archive_payload",
    ),
)
def test_release_rejects_incomplete_or_rehashed_validation_cpu_and_archive(released, fault):
    if fault in ("failed_gate", "missing_command", "receipt_error"):
        if fault == "failed_gate":
            released.receipt["trials"][0]["checks"]["boundary_continuation"] = False
        elif fault == "missing_command":
            released.receipt["gpu_commands"].pop()
        else:
            released.receipt["error"] = "preserved numerical failure"
        record(released.receipt_path, released.receipt)
        released.state["validation_sha256"] = owner.sha256(released.receipt_path)
        record(released.state_path, released.state)
        released.release["validation_sha256"] = owner.sha256(released.receipt_path)
        released.release["state_sha256"] = owner.sha256(released.state_path)
        released.release["validation_reconstruction"].update(
            receipt_sha256=owner.sha256(released.receipt_path), state_sha256=owner.sha256(released.state_path)
        )
    elif fault == "state_error":
        released.state["error"] = "preserved cleanup failure"
        record(released.state_path, released.state)
        released.release["state_sha256"] = owner.sha256(released.state_path)
        released.release["validation_reconstruction"]["state_sha256"] = owner.sha256(released.state_path)
    elif fault == "reconstruction":
        released.release["validation_reconstruction"]["checks"] = 26
    elif fault == "keeper":
        released.release["validation_reconstruction"]["keeper"] = released.state["restored_keeper"] | {"pid": 999}
    else:
        if fault == "cpu_command":
            released.cpu["command"] = [sys.executable, "-c", "pass"]
        elif fault == "cpu_error":
            released.cpu["error"] = "preserved archive failure"
        else:
            path = Path(released.cpu["archive"])
            with tarfile.open(path) as bundle:
                members = bundle.getmembers()
                payloads = {member.name: bundle.extractfile(member).read() for member in members if member.isfile()}
            with tarfile.open(path, "w") as bundle:
                for member in members:
                    if member.isfile():
                        content = (
                            b"replaced numerical payload"
                            if member.name == "experiment/optimizer.py"
                            else payloads[member.name]
                        )
                        member.size = len(content)
                        bundle.addfile(member, io.BytesIO(content))
                    else:
                        bundle.addfile(member)
            released.cpu["archive_sha256"] = owner.sha256(path)
        record(released.cpu_path, released.cpu)
        released.release["cpu_receipt_sha256"] = owner.sha256(released.cpu_path)
    record(released.release_path, released.release)
    with pytest.raises(RuntimeError):
        owner.require_release(released.release_path)
    assert not (released.root / "runs/wave11").exists()
