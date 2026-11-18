# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise acknowledged profiler commands and actual persisted process-boundary ownership."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

if Path(__file__).parent.name == "tests":
    from experiment import selected_lr_stage_profile_owner as owner
else:
    package = ModuleType("_stage_profile_owner_cpu")
    package.__path__ = [str(Path(__file__).parent)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.selected_lr_stage_profile_owner",
        Path(__file__).with_name("selected_lr_stage_profile_execution_owner.py"),
    )
    owner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = owner
    spec.loader.exec_module(owner)


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def process(proc, pid, parent, ticks, command, cwd, environment, group=None, session=None):
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[:4] = ["S", str(parent), str(pid if group is None else group), str(pid if session is None else session)]
    fields[19] = ticks
    directory.joinpath("stat").write_text(f"{pid} (profile test rank) {' '.join(fields)}\n")
    directory.joinpath("cmdline").write_bytes(("\0".join(command) + "\0").encode())
    directory.joinpath("environ").write_bytes(
        ("\0".join(f"{key}={value}" for key, value in environment.items()) + "\0").encode()
    )
    link = directory / "cwd"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(cwd, target_is_directory=True)


@pytest.fixture
def owned_processes(tmp_path, monkeypatch):
    root, candidate, proc = tmp_path / "root", tmp_path / "candidate", tmp_path / "proc"
    candidate.mkdir()
    pinned = candidate / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    candidate.joinpath("artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    candidate.joinpath("experiment").mkdir()
    candidate.joinpath("experiment/selected_lr_stage_profile_owner.py").write_bytes(Path(owner.__file__).read_bytes())
    candidate.joinpath("experiment/train.py").write_text("# bounded trainer fixture\n")
    root.joinpath("data").mkdir(parents=True)
    record(root / "data/experiment_receipt.json", {"fixture": "prepared-data"})
    record(root / "run-preparation/wave12-launch-binding.json", {"fixture": "parent-release"})
    binding_path, helpers, state_path, work, _ = owner.locations(root)
    helpers.mkdir(parents=True)
    for name in ("run_selected_lr_stage_profile.py", "validate_selected_lr_stage_profile.py"):
        helpers.joinpath(name).write_text("# frozen helper fixture\n")
    archive = root / "controller-source.tar"
    archive.write_bytes(b"fixture-bound-controller")
    files = owner.candidate_files(candidate)
    record(
        candidate / owner.PREPARATION,
        {"status": owner.PREPARED_STATUS, "root": str(root), "candidate": str(candidate), "files_sha256": files},
    )
    frozen = {
        "status": owner.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "preparation_sha256": owner.sha256(candidate / owner.PREPARATION),
        "files_sha256": files,
        "helpers": str(helpers),
        "helpers_sha256": {p.name: owner.sha256(p) for p in helpers.iterdir()},
        "node": owner.NODE,
        "environment": owner.ENVIRONMENT,
        "commands": owner.validation_commands(candidate, root, work),
        "archive": str(archive),
        "archive_sha256": owner.sha256(archive),
        "data_receipt_sha256": owner.sha256(root / "data/experiment_receipt.json"),
        "predecessor_release_sha256": owner.sha256(root / "run-preparation/wave12-launch-binding.json"),
    }
    record(binding_path, frozen)
    intent = {
        "status": "acknowledged",
        "pid": 21001,
        "start_ticks": "101",
        "marker": "controller-profile",
        "binding_sha256": owner.sha256(binding_path),
        "command": owner.controller_command(root),
    }
    record(helpers.parent / "controller.json", intent)
    state = {
        "status": "validating",
        "spawn_status": "acknowledged",
        "binding_sha256": owner.sha256(binding_path),
        "controller_pid": 21001,
        "controller_start_ticks": "101",
        "child_pid": 21002,
        "child_start_ticks": "102",
        "child_marker": "validator-profile",
        "command": owner.validator_command(root, candidate),
        "keeper_release_complete": True,
    }
    record(state_path, state)
    monkeypatch.setattr(owner, "TASK", candidate)
    monkeypatch.setattr(sys.modules[owner.observed.__module__], "PROC_ROOT", proc)
    environment = owner.ENVIRONMENT | {owner.CHILD_MARKER: state["child_marker"]}
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    for key, value in {"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8", "RANK": "0", "LOCAL_RANK": "0"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(candidate)
    monkeypatch.setattr(owner.subprocess, "check_output", lambda *args, **kwargs: f"{owner.NODE}\n")
    process(
        proc,
        21001,
        1,
        "101",
        owner.controller_command(root),
        helpers,
        owner.ENVIRONMENT | {owner.CONTROLLER_MARKER: intent["marker"]},
    )
    process(proc, 21002, 21001, "102", owner.validator_command(root, candidate), candidate, environment)
    command = frozen["commands"][2]
    state["active_command"] = command
    record(state_path, state)
    process(proc, 21003, 21002, "103", command, candidate, environment)
    process(proc, os.getpid(), 21003, "104", [command[0], "-u", *command[5:]], candidate, environment)
    monkeypatch.setattr(owner.sys, "argv", [command[6], *command[7:]])
    return SimpleNamespace(
        root=root,
        candidate=candidate,
        proc=proc,
        frozen=frozen,
        state=state,
        state_path=state_path,
        command=command,
        environment=environment,
    )


@pytest.mark.parametrize(
    "fault",
    (
        "wrong_controller_generation",
        "unbounded_rank",
        "shared_rank_session",
        "wrong_launcher",
        "escaped_validator",
        "wrong_mask",
        "wrong_marker",
        "changed_source",
        "unreleased_keeper",
    ),
)
def test_owned_profile_rank_rejects_changed_ancestry_command_or_source(owned_processes, monkeypatch, fault):
    item = owned_processes
    args = SimpleNamespace(
        launch_release=owner.locations(item.root)[0], config=Path(item.command[item.command.index("--config") + 1])
    )
    owner.require_selected_lr_stage_profile_owner(args)
    if fault == "wrong_controller_generation":
        item.proc.joinpath("21001/stat").write_text(
            item.proc.joinpath("21001/stat").read_text().replace("101", "reused")
        )
    elif fault == "unbounded_rank":
        command = item.command.copy()
        index = command.index("--max-steps")
        del command[index : index + 2]
        monkeypatch.setattr(owner.sys, "argv", [command[6], *command[7:]])
    elif fault == "shared_rank_session":
        process(
            item.proc,
            os.getpid(),
            21003,
            "104",
            [item.command[0], "-u", *item.command[5:]],
            item.candidate,
            item.environment,
            group=21003,
            session=21003,
        )
    elif fault == "wrong_launcher":
        process(item.proc, 21003, 21002, "103", ["unbound-launcher"], item.candidate, item.environment)
    elif fault == "escaped_validator":
        process(
            item.proc,
            21002,
            1,
            "102",
            owner.validator_command(item.root, item.candidate),
            item.candidate,
            item.environment,
        )
    elif fault == "wrong_mask":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    elif fault == "wrong_marker":
        monkeypatch.setenv(owner.CHILD_MARKER, "unowned-profile")
    elif fault == "unreleased_keeper":
        record(item.state_path, item.state | {"keeper_release_complete": False})
    else:
        item.candidate.joinpath("experiment/train.py").write_text("# changed numerical trainer\n")
    with pytest.raises(RuntimeError):
        owner.require_selected_lr_stage_profile_owner(args)


def test_gpu_validator_must_be_original_direct_child(owned_processes, monkeypatch):
    item = owned_processes
    with pytest.raises(RuntimeError):
        owner.require_gpu_owner(item.root)
    monkeypatch.setattr(owner.os, "getpid", lambda: 21002)
    monkeypatch.setattr(owner.os, "getppid", lambda: 21001)
    assert owner.require_gpu_owner(item.root) == {"controller_pid": 21001, "controller_start_ticks": "101"}
