# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check persisted validation ownership with fake /proc and hostname boundaries."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if Path(__file__).parent.name == "tests":
    from experiment import execution_owner as owner
else:
    package_name = "_owned_output_execution_owner_cpu"
    package_path = Path(__file__).resolve().parents[1] / "experiment"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path), str(Path(__file__).parent)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    name = f"{package_name}.execution_owner"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("owned_output_execution_owner.py"))
    owner = importlib.util.module_from_spec(spec)
    sys.modules[name] = owner
    spec.loader.exec_module(owner)


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def process(directory, pid, parent, ticks, argv, cwd, environment, status="S"):
    path = directory / str(pid)
    path.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[0], fields[1], fields[19] = status, str(parent), ticks
    (path / "stat").write_text(f"{pid} (owned fixture) {' '.join(fields)}\n")
    (path / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
    (path / "environ").write_bytes(("\0".join(f"{key}={value}" for key, value in environment.items()) + "\0").encode())
    link = path / "cwd"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(cwd, target_is_directory=True)


@pytest.fixture
def owned_processes(tmp_path, monkeypatch):
    root, task, proc = tmp_path / "root", tmp_path / "candidate", tmp_path / "proc"
    task.mkdir()
    helpers = root / "controllers/owned-output-full-model-validation-20261117/tracking"
    helpers.mkdir(parents=True)
    helper = helpers / "run_owned_output_validation.py"
    helper.write_text("bound controller fixture\n")
    sources = {
        "experiment/execution_owner.py": Path(owner.__file__).read_bytes(),
        "experiment/owned_output_validation.py": (
            Path(sys.modules[owner.validation_commands.__module__].__file__).read_bytes()
        ),
        "experiment/train.py": b"bound training fixture\n",
        "artifacts/pinned-sources-materialized/Toner-fixture.py": b"bound pinned fixture\n",
    }
    files = {}
    for name, content in sources.items():
        path = task / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files[name] = owner.sha256(path)
    (task / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    data = root / "data/experiment_receipt.json"
    record(data, {"status": "verified"})
    archive = root / "source.tar"
    archive.write_bytes(b"bound archive fixture")
    work = root / "validation/owned_output_full_model"
    commands = owner.validation_commands(owner.PYTHON, task, root, work)
    for command in commands[2:]:
        path = Path(command[command.index("--config") + 1])
        record(path, {"experiment": {"arm": path.stem}, "training": {"steps": 5120}})
        files[str(path.relative_to(task))] = owner.sha256(path)
    preparation = root / "run-preparation"
    preparation.mkdir()
    binding_path = preparation / "owned-output-full-model-binding.json"
    binding = {
        "status": "frozen_pending_completed_ceiling_and_owned_gpu_validation",
        "root": str(root),
        "candidate": str(task),
        "node": owner.NODE,
        "helpers": str(helpers),
        "files_sha256": files,
        "source_sha256": {name: files[name] for name in sources if name.startswith("experiment/")},
        "helpers_sha256": {helper.name: owner.sha256(helper)},
        "archive": str(archive),
        "archive_sha256": owner.sha256(archive),
        "data_receipt_sha256": owner.sha256(data),
        "environment": owner.VALIDATION_ENVIRONMENT,
        "commands": commands,
    }
    record(binding_path, binding)
    child_command = [
        owner.PYTHON,
        "-m",
        "experiment.validate",
        "--gpu",
        "--data",
        str(root / "data"),
        "--data-receipt",
        str(data),
        "--work",
        str(work),
        "--receipt",
        str(preparation / "validation-owned-output-full-model-gpu.json"),
    ]
    state = {
        "status": "validating",
        "controller_pid": 20001,
        "controller_start_ticks": "71",
        "child_pid": 20002,
        "child_start_ticks": "72",
        "child_marker": "owned-output-validation-test",
        "spawn_status": "acknowledged",
        "node": owner.NODE,
        "candidate": str(task),
        "binding_sha256": owner.sha256(binding_path),
        "command": child_command,
    }
    state_path = preparation / "owned-output-full-model-state.json"
    record(state_path, state)
    environment = {**owner.VALIDATION_ENVIRONMENT, owner.MARKER: state["child_marker"]}
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(owner, "TASK", task)
    monkeypatch.setattr(owner, "OUTPUT_ROOT", root)
    monkeypatch.setattr(owner, "ENVIRONMENT", dict(owner.VALIDATION_ENVIRONMENT))
    monkeypatch.setattr(owner, "PROC_ROOT", proc)
    monkeypatch.chdir(task)
    monkeypatch.setattr(owner.subprocess, "check_output", lambda *args, **kwargs: f"127.0.0.1 {owner.NODE}\n")
    controller_command = [owner.PYTHON, str(helper), "--root", str(root), "--phase", "validate"]
    process(proc, 20001, 1, "71", controller_command, helpers, {})
    process(proc, 20002, 20001, "72", child_command, task, environment)
    process(proc, 20003, 20002, "73", commands[0], task, environment)
    current = os.getpid()
    process(proc, current, 20003, "74", [owner.PYTHON, "-m", "experiment.gpu_checks"], task, environment)
    return SimpleNamespace(
        root=root,
        task=task,
        proc=proc,
        helper=helper,
        archive=archive,
        data=data,
        binding_path=binding_path,
        binding=binding,
        state_path=state_path,
        state=state,
        commands=commands,
        environment=environment,
        current=current,
    )


def bind_launcher(evidence, command, monkeypatch):
    process(evidence.proc, 20003, 20002, "73", command, evidence.task, evidence.environment)
    process(
        evidence.proc,
        evidence.current,
        20003,
        "74",
        [command[0], "-u", *command[5:]],
        evidence.task,
        evidence.environment,
    )
    monkeypatch.setattr(owner.sys, "argv", [command[6], *command[7:]])


def rebind(evidence):
    record(evidence.binding_path, evidence.binding)
    evidence.state["binding_sha256"] = owner.sha256(evidence.binding_path)
    record(evidence.state_path, evidence.state)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "controller_generation",
        "child_generation",
        "controller_command",
        "child_command",
        "child_parent",
        "marker",
        "source",
        "extra_source",
        "extra_config",
        "pinned_source",
        "extra_pinned",
        "pinned_alias",
        "helper",
        "archive",
        "data",
        "config",
        "command_plan",
        "source_map",
        "binding_digest",
        "old_status",
        "node",
        "cwd",
        "child_cwd",
        "launcher_cwd",
        "mask",
        "environment",
        "child_mask",
        "launcher_environment",
        "inherited_marker",
        "ancestry",
    ],
)
def test_controller_requires_frozen_identity_and_live_inherited_ownership(owned_processes, monkeypatch, fault):
    evidence = owned_processes
    assert owner.require_controller() == evidence.state
    if fault in ("controller_generation", "child_generation", "child_parent", "ancestry"):
        pid = {
            "controller_generation": 20001,
            "child_generation": 20002,
            "child_parent": 20002,
            "ancestry": evidence.current,
        }[fault]
        fields, argv = owner.observed(pid)
        parent = 1 if fault in ("child_parent", "ancestry") else int(fields[1])
        ticks = "reused" if fault.endswith("generation") else fields[19]
        process(evidence.proc, pid, parent, ticks, argv, evidence.task, evidence.environment)
    elif fault in ("controller_command", "child_command"):
        pid = 20001 if fault == "controller_command" else 20002
        fields, argv = owner.observed(pid)
        process(
            evidence.proc, pid, int(fields[1]), fields[19], [*argv, "--unbound"], evidence.task, evidence.environment
        )
    elif fault == "marker":
        monkeypatch.setenv(owner.MARKER, "unowned")
    elif fault in ("source", "pinned_source", "helper", "archive", "data", "config"):
        path = {
            "source": evidence.task / "experiment/execution_owner.py",
            "pinned_source": evidence.task / "artifacts/pinned-sources-materialized/Toner-fixture.py",
            "helper": evidence.helper,
            "archive": evidence.archive,
            "data": evidence.data,
            "config": Path(evidence.commands[2][evidence.commands[2].index("--config") + 1]),
        }[fault]
        path.write_text("changed bytes\n")
    elif fault in ("extra_source", "extra_config", "extra_pinned"):
        name = {
            "extra_source": "experiment/unlisted.py",
            "extra_config": "experiment/configs/unlisted.json",
            "extra_pinned": "artifacts/pinned-sources-materialized/unlisted.txt",
        }[fault]
        (evidence.task / name).write_text("unlisted bytes\n")
    elif fault == "pinned_alias":
        alias = evidence.task / "artifacts/external-sources"
        alias.unlink()
        alias.symlink_to(evidence.root, target_is_directory=True)
    elif fault in ("command_plan", "source_map", "old_status"):
        if fault == "command_plan":
            evidence.binding["commands"][-1] = evidence.binding["commands"][-2]
        elif fault == "source_map":
            evidence.binding["source_sha256"]["experiment/train.py"] = "unbound"
        else:
            evidence.binding["status"] = "frozen_pending_upper_lr_and_synthetic_probe"
        rebind(evidence)
    elif fault == "binding_digest":
        evidence.state["binding_sha256"] = "old binding"
        record(evidence.state_path, evidence.state)
    elif fault == "node":
        monkeypatch.setattr(owner.subprocess, "check_output", lambda *args, **kwargs: f"{owner.NODE}0\n")
    elif fault == "cwd":
        monkeypatch.chdir(evidence.root)
    elif fault in ("child_cwd", "launcher_cwd"):
        pid = 20002 if fault == "child_cwd" else 20003
        link = evidence.proc / str(pid) / "cwd"
        link.unlink()
        link.symlink_to(evidence.root, target_is_directory=True)
    elif fault in ("mask", "environment"):
        key = "CUDA_VISIBLE_DEVICES" if fault == "mask" else "CUBLAS_WORKSPACE_CONFIG"
        monkeypatch.setenv(key, "changed")
    elif fault in ("child_mask", "launcher_environment", "inherited_marker"):
        pid = 20002 if fault == "child_mask" else 20003
        key = {
            "child_mask": "CUDA_VISIBLE_DEVICES",
            "launcher_environment": "CUBLAS_WORKSPACE_CONFIG",
            "inherited_marker": owner.MARKER,
        }[fault]
        fields, argv = owner.observed(pid)
        process(
            evidence.proc, pid, int(fields[1]), fields[19], argv, evidence.task, {**evidence.environment, key: "changed"}
        )
    if fault is not None:
        with pytest.raises(RuntimeError):
            owner.require_controller()
    else:
        assert json.loads(evidence.state_path.read_text()) == evidence.state
        assert not (evidence.root / "run-preparation/validation-owned-output-full-model-gpu.json").exists()


def test_gpu_owner_requires_direct_acknowledged_validator(owned_processes, monkeypatch):
    with pytest.raises(RuntimeError):
        owner.require_gpu_owner()
    monkeypatch.setattr(owner.os, "getpid", lambda: 20002)
    monkeypatch.setattr(owner.os, "getppid", lambda: 20001)
    assert owner.require_gpu_owner() == {"controller_pid": 20001, "controller_start_ticks": "71"}
    monkeypatch.setattr(owner.os, "getppid", lambda: 1)
    with pytest.raises(RuntimeError):
        owner.require_gpu_owner()


@pytest.mark.parametrize("outcome", ["acknowledged", "timeout", "generation_drift"])
def test_controller_waits_for_matching_child_acknowledgment(owned_processes, monkeypatch, outcome):
    evidence = owned_processes
    evidence.state.pop("child_start_ticks")
    evidence.state["spawn_status"] = "intent"
    record(evidence.state_path, evidence.state)
    clock = [0.0]
    monkeypatch.setattr(owner.time, "monotonic", lambda: clock[0])

    def advance(delay):
        clock[0] += 31
        if outcome == "acknowledged":
            evidence.state.update(child_start_ticks="72", spawn_status="acknowledged")
            record(evidence.state_path, evidence.state)
        elif outcome == "generation_drift":
            _, argv = owner.observed(20001)
            process(evidence.proc, 20001, 1, "new", argv, evidence.task, evidence.environment)

    monkeypatch.setattr(owner.time, "sleep", advance)
    if outcome == "acknowledged":
        assert owner.require_controller()["spawn_status"] == "acknowledged"
    else:
        with pytest.raises(RuntimeError):
            owner.require_controller()
        assert json.loads(evidence.state_path.read_text())["spawn_status"] == "intent"


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "fault", ["rank_count", "launcher_receipt", "rank_receipt", "process_receipt", "module", "extra_argument"]
)
def test_synthetic_ranks_require_their_exact_launcher_and_receipt(owned_processes, monkeypatch, index, fault):
    evidence = owned_processes
    command = evidence.commands[index]
    bind_launcher(evidence, command, monkeypatch)
    require = owner.require_synthetic_owner if index == 0 else owner.require_owned_synthetic_owner
    assert require() == evidence.state
    changed = list(command)
    if fault == "rank_count":
        changed[4] = "--nproc_per_node=8" if index == 0 else "--nproc_per_node=2"
    elif fault in ("launcher_receipt", "rank_receipt", "process_receipt"):
        changed[changed.index("--receipt") + 1] = str(evidence.root / "unbound.json")
    elif fault == "module":
        changed[6] = "experiment.train"
    else:
        changed.append("--unbound")
    if fault == "rank_receipt":
        monkeypatch.setattr(owner.sys, "argv", [changed[6], *changed[7:]])
    elif fault == "process_receipt":
        process(
            evidence.proc,
            evidence.current,
            20003,
            "74",
            [changed[0], "-u", *changed[5:]],
            evidence.task,
            evidence.environment,
        )
    else:
        process(evidence.proc, 20003, 20002, "73", changed, evidence.task, evidence.environment)
    with pytest.raises(RuntimeError):
        require()


@pytest.mark.parametrize("index", range(2, 14))
def test_training_ranks_require_each_canonical_bounded_trajectory(owned_processes, monkeypatch, index):
    evidence = owned_processes
    command = evidence.commands[index]
    bind_launcher(evidence, command, monkeypatch)
    config_path = Path(command[command.index("--config") + 1])
    args = SimpleNamespace(launch_release=None, config=config_path)
    config = json.loads(config_path.read_text())
    # Training authorization returns no release and leaves all persisted evidence unchanged.
    assert owner.require_training_owner(args, config) is None
    assert json.loads(evidence.state_path.read_text()) == evidence.state
    changed = [*command[:4], "--nproc_per_node=2", *command[5:]]
    process(evidence.proc, 20003, 20002, "73", changed, evidence.task, evidence.environment)
    with pytest.raises(RuntimeError):
        owner.require_training_owner(args, config)


@pytest.mark.parametrize("fault", ["rank_budget", "memory_config", "config_path", "resume_path"])
def test_training_rejects_changed_rank_invocation_or_configuration(owned_processes, monkeypatch, fault):
    evidence = owned_processes
    command = evidence.commands[4]
    bind_launcher(evidence, command, monkeypatch)
    config_path = Path(command[command.index("--config") + 1])
    args = SimpleNamespace(launch_release=None, config=config_path)
    config = json.loads(config_path.read_text())
    assert owner.require_training_owner(args, config) is None
    if fault == "rank_budget":
        owner.sys.argv[owner.sys.argv.index("--max-steps") + 1] = "5120"
    elif fault == "memory_config":
        config["training"]["steps"] = 1
    elif fault == "config_path":
        args.config = evidence.root / "identical.json"
        record(args.config, config)
    else:
        owner.sys.argv[owner.sys.argv.index("--resume") + 1] = str(evidence.root / "old/checkpoints/step_000022")
    with pytest.raises(RuntimeError):
        owner.require_training_owner(args, config)


@pytest.mark.parametrize("old", ["upstream-full-model", "upstream-full-model-corrected", "wave9-launch"])
def test_old_binding_paths_cannot_supply_owned_output_authorization(owned_processes, old):
    evidence = owned_processes
    original = evidence.binding_path.read_bytes(), evidence.state_path.read_bytes()
    binding = evidence.binding_path.with_name(f"{old}-binding.json")
    state = evidence.state_path.with_name(f"{old}-state.json")
    evidence.binding_path.rename(binding)
    evidence.state_path.rename(state)
    with pytest.raises(FileNotFoundError):
        owner.require_controller()
    assert (binding.read_bytes(), state.read_bytes()) == original


@pytest.mark.parametrize("release", ["wave8", "wave9", "owned-output", "", False, 0])
def test_any_production_release_is_rejected_without_validation_fallback(tmp_path, monkeypatch, release):
    monkeypatch.setattr(owner, "OUTPUT_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="production release has not been issued"):
        owner.require_training_owner(SimpleNamespace(launch_release=release), {})
    assert not list(tmp_path.iterdir())
