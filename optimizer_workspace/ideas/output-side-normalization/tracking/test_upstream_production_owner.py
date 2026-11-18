# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise production authorization with persisted evidence and fake process/hostname boundaries."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def production_owner(tmp_path, monkeypatch):
    if Path(__file__).parent.name == "tests":
        return importlib.import_module("experiment.production_owner")
    package = ModuleType("_upstream_production_owner_test")
    package.__path__ = []
    modules = {
        package.__name__: package,
        f"{package.__name__}.launch": SimpleNamespace(
            ENVIRONMENT={}, NODES=("node0", "node1", "node2", "node3"), OUTPUT_ROOT=tmp_path, PYTHON=sys.executable
        ),
        f"{package.__name__}.records": SimpleNamespace(sha256=sha256),
        f"{package.__name__}.reference": SimpleNamespace(TASK=tmp_path),
        f"{package.__name__}.upstream_validation": SimpleNamespace(
            ARMS=("muont", "outer_leftdiag_fused", "outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused")
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.production_owner", Path(__file__).with_name("upstream_production_owner.py")
    )
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def owned_production(tmp_path, monkeypatch, production_owner):
    owner = production_owner
    root, candidate, previous = tmp_path / "root", tmp_path / "candidate", tmp_path / "validator"
    preparation = root / "run-preparation"
    preparation.mkdir(parents=True)
    candidate.mkdir()
    previous.mkdir()
    monkeypatch.setattr(owner, "TASK", candidate)
    monkeypatch.setattr(owner, "OUTPUT_ROOT", root)
    monkeypatch.setattr(owner, "ENVIRONMENT", {})
    files, original_files = {}, {}
    for name in owner.CHANGES:
        destination = candidate / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("production ownership fixture\n")
        files[name] = sha256(destination)
        if name == "experiment/production_owner.py":
            continue
        source = previous / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("validation ownership fixture\n")
        original_files[name] = sha256(source)
    config_path = candidate / "experiment/configs/muont_0p5m_seed0.json"
    config = {"experiment": {"arm": "muont"}, "training": {"steps": 5120, "tokens": 2684354560}}
    for directory in (candidate, previous):
        record(directory / "experiment/configs/muont_0p5m_seed0.json", config)
    files["experiment/configs/muont_0p5m_seed0.json"] = sha256(config_path)
    original_files["experiment/configs/muont_0p5m_seed0.json"] = sha256(config_path)
    data = root / "data/experiment_receipt.json"
    record(data, {"status": "verified"})
    archive = preparation / "original.tar"
    archive.write_bytes(b"original archive")
    log = preparation / "gpu.log"
    log.write_text("successful original command\n")
    original = {
        "candidate": str(previous),
        "files_sha256": original_files,
        "source_sha256": {"original.py": "frozen"},
        "data_receipt_sha256": sha256(data),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "environment": {},
        "required_gates": ["gpu"],
        "commands": [["original", "synthetic"]],
    }
    original_path = preparation / "upstream-full-model-corrected-binding.json"
    record(original_path, original)
    validation_path = preparation / "validation-upstream-full-model-corrected-gpu.json"
    record(
        validation_path,
        {
            "source_sha256": original["source_sha256"],
            "environment": {},
            "data_receipt_sha256": sha256(data),
            "owner": {"controller_pid": 111, "controller_start_ticks": "222"},
            "checks": {"gpu": True},
            "evidence": {
                "gpu_commands": [
                    {"command": original["commands"][0], "exit_code": 0, "log": str(log), "log_sha256": sha256(log)}
                ]
            },
        },
    )
    validation_state = preparation / "upstream-full-model-corrected-state.json"
    record(
        validation_state,
        {
            "status": "passed",
            "controller_pid": 111,
            "controller_start_ticks": "222",
            "binding_sha256": sha256(original_path),
            "validation_sha256": sha256(validation_path),
            "exit_code": 0,
            "cleanup_exit_code": 0,
            "keeper_restore": 0,
        },
    )
    helpers = root / "controllers/wave9-launch-20261117/tracking"
    helpers.mkdir(parents=True)
    helper = helpers / "upstream_production.py"
    helper.write_text("production controller fixture\n")
    launch_archive = preparation / "source-wave9-launch-20261117.tar"
    launch_archive.write_bytes(b"launch fixture")
    prepare = candidate / "production-preparation.json"
    record(prepare, {"files_sha256": files})
    cpu = preparation / "cpu.json"
    record(cpu, {"status": "production_cpu_verified_gpu_release_pending"})
    release_path = preparation / "wave9-launch-binding.json"
    command = [
        owner.PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "experiment.train",
        "--config",
        str(config_path),
        "--data",
        str(root / "data"),
        "--data-receipt",
        str(data),
        "--out",
        str(root / "runs/wave9/muont"),
        "--launch-release",
        str(release_path),
    ]
    node = owner.NODES[0]
    release = {
        "status": "frozen_validated_launch_release",
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "helpers_sha256": {helper.name: sha256(helper)},
        "files_sha256": files,
        "environment": {},
        "arms": owner.ARMS,
        "nodes": owner.NODES,
        "validation_binding_sha256": sha256(original_path),
        "validation_sha256": sha256(validation_path),
        "state_sha256": sha256(validation_state),
        "data_receipt_sha256": sha256(data),
        "archive": str(launch_archive),
        "archive_sha256": sha256(launch_archive),
        "preparation": str(prepare),
        "preparation_sha256": sha256(prepare),
        "cpu_receipt": str(cpu),
        "cpu_receipt_sha256": sha256(cpu),
        "commands": {node: command},
    }
    record(release_path, release)
    state = {
        "status": "running",
        "arm": "muont",
        "node": node,
        "worker_pid": 20001,
        "worker_start_ticks": "71",
        "child_pid": 20002,
        "child_start_ticks": "72",
        "command": command,
        "release_binding_sha256": sha256(release_path),
        "worker_marker": "worker-owned",
        "child_marker": "child-owned",
    }
    state_path = preparation / f"wave9-worker-{node}.json"
    record(state_path, state)
    monkeypatch.setenv(owner.WORKER_MARKER, state["worker_marker"])
    monkeypatch.setenv(owner.MARKER, state["child_marker"])
    monkeypatch.setattr(owner.subprocess, "check_output", lambda *args, **kwargs: node)
    monkeypatch.setattr(owner.sys, "argv", [str(candidate / "experiment/train.py"), *command[7:]])

    def fields(parent, ticks):
        values = ["0"] * 20
        values[0], values[1], values[19] = "S", str(parent), ticks
        return values

    processes = {
        20001: (
            fields(1, "71"),
            [owner.PYTHON, str(helper), "--root", str(root), "--phase", "worker", "--arm", "muont"],
        ),
        20002: (fields(20001, "72"), command),
        os.getpid(): (fields(20002, "73"), [owner.PYTHON, "-m", "experiment.train"]),
    }
    monkeypatch.setattr(owner, "observed", lambda pid: processes[pid])
    args = SimpleNamespace(
        launch_release=release_path,
        config=config_path,
        max_steps=None,
        resume=None,
        execution_boundary=None,
        verify_diagonal_capture=False,
        verify_outer_diagonal_updates=False,
    )
    return owner, root, candidate, release_path, release, state_path, state, processes, args, config


@pytest.mark.parametrize(
    "fault",
    [None, "worker_pid", "launcher_pid", "parent", "marker", "node", "command", "budget", "source", "restoration"],
)
def test_production_ownership_authorizes_only_exact_released_live_full_budget_children(
    owned_production, monkeypatch, fault
):
    owner, root, candidate, _, _, state_path, state, processes, args, config = owned_production
    if fault == "worker_pid":
        processes[state["worker_pid"]][0][19] = "reused"
    elif fault == "launcher_pid":
        processes[state["child_pid"]][0][19] = "reused"
    elif fault == "parent":
        processes[state["child_pid"]][0][1] = "1"
    elif fault == "marker":
        monkeypatch.setenv(owner.MARKER, "unowned")
    elif fault == "node":
        monkeypatch.setattr(owner.subprocess, "check_output", lambda *a, **k: "unrelated-node")
    elif fault == "command":
        owner.sys.argv.extend(("--max-steps", "24"))
    elif fault == "budget":
        args.max_steps = 24
    elif fault == "source":
        (candidate / "experiment/production_owner.py").write_text("changed source\n")
    elif fault == "restoration":
        state_path = root / "run-preparation/upstream-full-model-corrected-state.json"
        saved = json.loads(state_path.read_text())
        saved["keeper_restore"] = False
        record(state_path, saved)
    if fault is None:
        owner.require_production_owner(args, config)
        assert json.loads(state_path.read_text()) == state
    else:
        with pytest.raises(RuntimeError):
            owner.require_production_owner(args, config)


def test_original_or_missing_production_release_never_falls_back_to_validation(owned_production):
    owner, root, _, release_path, _, _, _, _, args, config = owned_production
    args.launch_release = root / "run-preparation/wave8-launch-binding.json"
    with pytest.raises(RuntimeError, match="canonical"):
        owner.require_production_owner(args, config)
    args.launch_release = release_path
    release_path.unlink()
    with pytest.raises(FileNotFoundError):
        owner.require_production_owner(args, config)
