# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check actual bounded GPU ownership through persisted process and filesystem boundaries."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if Path(__file__).parent.name == "tests":
    from experiment import selected_lr_owner as owner
else:
    package_name = "_selected_lr_owner_cpu"
    experiment = importlib.util.find_spec("experiment")
    package_path = Path(experiment.origin).parent if experiment else Path(__file__).resolve().parents[1] / "experiment"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path), str(Path(__file__).parent)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.selected_lr_owner", Path(__file__).with_name("selected_lr_execution_owner.py")
    )
    owner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = owner
    spec.loader.exec_module(owner)


def record(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def process(directory, pid, parent, ticks, command, cwd, environment, group=None, session=None):
    path = directory / str(pid)
    path.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[0:4] = ["S", str(parent), str(group or pid), str(session or pid)]
    fields[19] = ticks
    path.joinpath("stat").write_text(f"{pid} (selected LR fixture) {' '.join(fields)}\n")
    path.joinpath("cmdline").write_bytes(("\0".join(command) + "\0").encode())
    path.joinpath("environ").write_bytes(
        ("\0".join(f"{key}={value}" for key, value in environment.items()) + "\0").encode()
    )
    link = path / "cwd"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(cwd, target_is_directory=True)


@pytest.fixture
def selected_processes(tmp_path, monkeypatch):
    root, task, proc = tmp_path / "root", tmp_path / "candidate", tmp_path / "proc"
    binding_path, helpers, state_path, work, _ = owner.locations(root)
    helpers.mkdir(parents=True)
    for name in ("run_selected_lr_validation.py", "validate_selected_lr_full_budget.py"):
        helpers.joinpath(name).write_text("# frozen controller fixture\n")
    sources = {
        "experiment/selected_lr_owner.py": Path(owner.__file__).read_bytes(),
        "experiment/selected_lr_validation.py": (
            Path(sys.modules[owner.validation_commands.__module__].__file__).read_bytes()
        ),
        "experiment/train.py": b"# numerical trainer fixture\n",
        "artifacts/pinned-sources-materialized/toner.py": b"# pinned numerical fixture\n",
    }
    for name, content in sources.items():
        path = task / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    task.joinpath("artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    for arm in owner.validation_commands.__globals__["ARMS"]:
        record(
            task / f"experiment/configs/{arm}_0p5m_seed0.json",
            {"experiment": {"arm": arm}, "training": {"steps": 5120}},
        )
    data = root / "data/experiment_receipt.json"
    record(data, {"status": "verified"})
    archive = root / "source.tar"
    archive.write_bytes(b"bound controller archive fixture")
    monkeypatch.setattr(owner, "TASK", task)
    monkeypatch.setattr(owner, "PROC_ROOT", proc)
    files = owner.candidate_files()
    prepared_path = task / owner.PREPARATION
    record(prepared_path, {"root": str(root), "candidate": str(task), "files_sha256": files})
    frozen = {
        "status": owner.STATUS,
        "root": str(root),
        "candidate": str(task),
        "preparation_sha256": owner.sha256(prepared_path),
        "files_sha256": files,
        "node": owner.NODE,
        "helpers": str(helpers),
        "environment": owner.ENVIRONMENT,
        "python": owner.controller_command(root)[0],
        "commands": owner.validation_commands(owner.controller_command(root)[0], task, root, work),
        "archive": str(archive),
        "archive_sha256": owner.sha256(archive),
        "helpers_sha256": {path.name: owner.sha256(path) for path in helpers.iterdir()},
        "data_receipt_sha256": owner.sha256(data),
    }
    record(binding_path, frozen)
    intent = {
        "status": "acknowledged",
        "pid": 21001,
        "start_ticks": "101",
        "marker": "controller-test",
        "command": owner.controller_command(root),
        "binding_sha256": owner.sha256(binding_path),
    }
    record(helpers.parent / "controller.json", intent)
    state = {
        "status": "validating",
        "spawn_status": "acknowledged",
        "controller_pid": 21001,
        "controller_start_ticks": "101",
        "child_pid": 21002,
        "child_start_ticks": "102",
        "child_marker": "validator-test",
        "command": owner.validator_command(root, task),
        "binding_sha256": owner.sha256(binding_path),
    }
    record(state_path, state)
    environment = owner.ENVIRONMENT | {owner.CHILD_MARKER: state["child_marker"]}
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(task)
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
    process(proc, 21002, 21001, "102", owner.validator_command(root, task), task, environment)
    current = os.getpid()
    command = frozen["commands"][0]
    process(proc, 21003, 21002, "103", command, task, environment)
    process(proc, current, 21003, "104", [command[0], "-u", *command[5:]], task, environment)
    monkeypatch.setattr(owner.sys, "argv", [command[6], *command[7:]])
    return SimpleNamespace(
        root=root,
        task=task,
        proc=proc,
        frozen=frozen,
        binding_path=binding_path,
        state=state,
        state_path=state_path,
        environment=environment,
        current=current,
    )


def training_args(command):
    return SimpleNamespace(config=Path(command[command.index("--config") + 1]))


@pytest.mark.parametrize(
    "fault",
    (
        "controller_ticks",
        "validator_parent",
        "launcher_command",
        "rank_command",
        "mask",
        "marker",
        "source",
        "unbounded",
    ),
)
def test_selected_lr_training_requires_exact_bound_generation_and_command(selected_processes, monkeypatch, fault):
    evidence = selected_processes
    command = evidence.frozen["commands"][0]
    owner.require_selected_lr_training(training_args(command))
    if fault == "controller_ticks":
        original = owner.observed(21001)
        process(
            evidence.proc,
            21001,
            1,
            "reused",
            original["command"],
            Path(original["cwd"]),
            original["environment"],
        )
    elif fault == "validator_parent":
        process(
            evidence.proc,
            21002,
            1,
            "102",
            evidence.state["command"],
            evidence.task,
            evidence.environment,
        )
    elif fault == "launcher_command":
        process(evidence.proc, 21003, 21002, "103", [*command, "--unbound"], evidence.task, evidence.environment)
    elif fault == "rank_command":
        process(
            evidence.proc,
            evidence.current,
            21003,
            "104",
            [command[0], "-u", *command[5:], "--unbound"],
            evidence.task,
            evidence.environment,
        )
    elif fault == "mask":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    elif fault == "marker":
        monkeypatch.setenv(owner.CHILD_MARKER, "unowned")
    elif fault == "source":
        evidence.task.joinpath("experiment/train.py").write_text("# changed trainer\n")
    else:
        index = command.index("--max-steps")
        unbounded = command[:index] + command[index + 2 :]
        monkeypatch.setattr(owner.sys, "argv", [unbounded[6], *unbounded[7:]])
    with pytest.raises(RuntimeError):
        owner.require_selected_lr_training(training_args(command))


def test_selected_lr_owner_denies_legacy_gpu_entrypoint(selected_processes):
    with pytest.raises(RuntimeError):
        owner.require_selected_lr_owner()
    with pytest.raises(RuntimeError):
        owner.require_gpu_owner()
