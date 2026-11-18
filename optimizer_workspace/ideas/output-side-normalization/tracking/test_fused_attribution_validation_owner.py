# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mocked /proc ownership tests; neither training nor CUDA subprocesses are launched."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

if Path(__file__).parent.name == "tests":
    from experiment import fused_attribution_owner as owner
else:
    package = ModuleType("_attribution_owner_cpu")
    package.__path__ = [str(Path(__file__).parent)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.fused_attribution_owner", Path(__file__).with_name("fused_attribution_validation_owner.py")
    )
    owner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = owner
    spec.loader.exec_module(owner)

contract = sys.modules[owner.validation_commands.__module__]


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def process(proc, pid, parent, ticks, command, cwd, environment, session=None):
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[:4] = ["S", str(parent), str(pid), str(pid if session is None else session)]
    fields[19] = ticks
    (directory / "stat").write_text(f"{pid} (owned rank fixture) {' '.join(fields)}\n")
    (directory / "cmdline").write_bytes(("\0".join(command) + "\0").encode())
    (directory / "environ").write_bytes(
        ("\0".join(f"{key}={value}" for key, value in environment.items()) + "\0").encode()
    )
    link = directory / "cwd"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(cwd, target_is_directory=True)


@pytest.fixture
def owned(tmp_path, monkeypatch):
    root, candidate, proc = tmp_path / "root", tmp_path / "candidate", tmp_path / "proc"
    (candidate / "experiment").mkdir(parents=True)
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    (candidate / "experiment/fused_attribution_owner.py").write_bytes(Path(owner.__file__).read_bytes())
    for trial in contract.trial_specs():
        config = {
            "model": {"sequence_length": 4096},
            "training": {
                "steps": 5120,
                "tokens": 2684354560,
                "batch_tokens": 524288,
                "global_batch_sequences": 128,
                "world_size": 8,
                "seed": 0,
            },
            "experiment": {"arm": trial["family"], "refresh_install_age": 1, "diagonal_backend": "triton"},
            "optimizer": {"lr": trial["matrix_lr"], "ns_mode": "tpu", "ns_steps": 10},
            "runtime": {"compile": "default", "attention": "FA2", "head": "compiled"},
            "root": {"graphs": True},
            "schedule": contract.SCHEDULE,
        }
        record(candidate / trial["config_name"], config)
    record(root / "data/experiment_receipt.json", {"fixture": "prepared-data"})
    evidence_path = root / "run-preparation/failed-profiler.json"
    record(evidence_path, {"fixture": "separate-failed-profiler"})
    predecessor = {
        "restored_keeper": {"pid": 100, "generation": {"100": {"pid": 100, "start_ticks": "900", "group": 100}}},
        "evidence_sha256": {str(evidence_path.relative_to(root)): owner.sha256(evidence_path)},
    }
    binding_path, helpers, state_path, work, _ = owner.locations(root)
    helpers.mkdir(parents=True)
    for name in ("fused_attribution_validation_controller.py", "fused_attribution_validation.py"):
        (helpers / name).write_text("# source-bound future runtime\n")
    archive = root / "source.tar"
    archive.write_bytes(b"fixture source archive")
    files = owner.candidate_files(candidate)
    record(
        candidate / owner.PREPARATION,
        {
            "status": owner.PREPARED_STATUS,
            "root": str(root),
            "candidate": str(candidate),
            "files_sha256": files,
            "allocation_predecessor": predecessor,
            "data_receipt_sha256": owner.sha256(root / "data/experiment_receipt.json"),
        },
    )
    frozen = {
        "status": owner.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "preparation_sha256": owner.sha256(candidate / owner.PREPARATION),
        "files_sha256": files,
        "helpers": str(helpers),
        "helpers_sha256": {path.name: owner.sha256(path) for path in helpers.iterdir()},
        "node": owner.NODE,
        "environment": owner.ENVIRONMENT,
        "commands": owner.validation_commands(candidate, root, work),
        "archive": str(archive),
        "archive_sha256": owner.sha256(archive),
        "data_receipt_sha256": owner.sha256(root / "data/experiment_receipt.json"),
        "allocation_predecessor": predecessor,
    }
    record(binding_path, frozen)
    intent = {
        "status": "acknowledged",
        "pid": 21001,
        "start_ticks": "101",
        "marker": "new-controller",
        "binding_sha256": owner.sha256(binding_path),
        "command": owner.controller_command(root),
    }
    record(helpers.parent / "controller.json", intent)
    command = frozen["commands"][-1]
    state = {
        "status": "validating",
        "spawn_status": "acknowledged",
        "binding_sha256": owner.sha256(binding_path),
        "controller_pid": 21001,
        "controller_start_ticks": "101",
        "child_pid": 21002,
        "child_start_ticks": "102",
        "child_marker": "new-child",
        "command": owner.validator_command(root, candidate),
        "active_command": command,
        "keeper_release_complete": True,
        "allocation_predecessor": predecessor,
        "keeper": {"pid": 100, "start_ticks": "900", "generation": [[100, "900", 100]]},
    }
    record(state_path, state)
    monkeypatch.setattr(owner, "TASK", candidate)
    monkeypatch.setattr(sys.modules[owner.observed.__module__], "PROC_ROOT", proc)
    environment = owner.ENVIRONMENT | {owner.CHILD_MARKER: state["child_marker"]}
    for key, value in (
        environment | {"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8", "RANK": "7", "LOCAL_RANK": "7"}
    ).items():
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
        "none",
        "active_command",
        "rank_session",
        "launcher_session",
        "controller_generation",
        "validator_parent",
        "world",
        "local_rank",
        "mask",
        "marker",
        "config",
        "source",
        "keeper",
        "predecessor",
    ),
)
def test_exact_acknowledged_rank_and_source_only(owned, monkeypatch, fault):
    item = owned
    args = SimpleNamespace(
        launch_release=owner.locations(item.root)[0], config=Path(item.command[item.command.index("--config") + 1])
    )
    owner.require_fused_attribution_owner(args)
    if fault == "none":
        return
    if fault == "active_command":
        record(item.state_path, item.state | {"active_command": item.frozen["commands"][0]})
    elif fault in ("rank_session", "launcher_session"):
        pid = os.getpid() if fault == "rank_session" else 21003
        argv = [item.command[0], "-u", *item.command[5:]] if pid == os.getpid() else item.command
        process(
            item.proc,
            pid,
            21003 if pid == os.getpid() else 21002,
            "104" if pid == os.getpid() else "103",
            argv,
            item.candidate,
            item.environment,
            session=21002,
        )
    elif fault == "controller_generation":
        path = item.proc / "21001/stat"
        path.write_text(path.read_text().replace("101", "reused"))
    elif fault == "validator_parent":
        process(
            item.proc,
            21002,
            1,
            "102",
            owner.validator_command(item.root, item.candidate),
            item.candidate,
            item.environment,
        )
    elif fault in ("world", "local_rank", "mask", "marker"):
        key, value = {
            "world": ("WORLD_SIZE", "1"),
            "local_rank": ("LOCAL_RANK", "0"),
            "mask": ("CUDA_VISIBLE_DEVICES", ""),
            "marker": (owner.CHILD_MARKER, "inherited-old"),
        }[fault]
        monkeypatch.setenv(key, value)
    elif fault == "config":
        args.config = Path(item.frozen["commands"][0][item.command.index("--config") + 1])
    elif fault == "source":
        (item.candidate / "experiment/changed.py").write_text("# changed candidate\n")
    elif fault == "keeper":
        record(item.state_path, item.state | {"keeper_release_complete": False})
    else:
        record(item.root / "run-preparation/failed-profiler.json", {"fixture": "changed-restoration"})
    with pytest.raises((RuntimeError, FileNotFoundError)):
        owner.require_fused_attribution_owner(args)


def test_validator_only_direct_original_child(owned, monkeypatch):
    with pytest.raises(RuntimeError):
        owner.require_gpu_owner(owned.root)
    monkeypatch.setattr(owner.os, "getpid", lambda: 21002)
    monkeypatch.setattr(owner.os, "getppid", lambda: 21001)
    assert owner.require_gpu_owner(owned.root) == {"controller_pid": 21001, "controller_start_ticks": "101"}
