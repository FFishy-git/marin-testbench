# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Wave-10 authorization using persisted evidence and a fake /proc boundary."""

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def process(directory, pid, parent, ticks, argv, cwd, environment, status="S"):
    path = directory / str(pid)
    path.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[0], fields[1], fields[19] = status, str(parent), ticks
    (path / "stat").write_text(f"{pid} (wave10 fixture) {' '.join(fields)}\n")
    (path / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
    (path / "environ").write_bytes(("\0".join(f"{key}={value}" for key, value in environment.items()) + "\0").encode())
    link = path / "cwd"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(cwd, target_is_directory=True)


@pytest.fixture
def production_owner(tmp_path, monkeypatch):
    if Path(__file__).parent.name == "tests":
        return importlib.import_module("experiment.owned_production_owner")
    package = ModuleType("_owned_output_production_owner_test")
    package.__path__ = [str(Path(__file__).parent)]
    modules = {
        package.__name__: package,
        f"{package.__name__}.launch": SimpleNamespace(
            ENVIRONMENT={
                "NCCL_INSPECTOR_ENABLE": "0",
                "PYTHONFAULTHANDLER": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            },
            NODES=("node0", "node1", "node2", "node3"),
            OUTPUT_ROOT=tmp_path,
            PYTHON=sys.executable,
        ),
        f"{package.__name__}.records": SimpleNamespace(sha256=sha256),
        f"{package.__name__}.reference": SimpleNamespace(TASK=tmp_path),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    name = f"{package.__name__}.production_owner"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("owned_output_production_owner.py"))
    owner = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, owner)
    spec.loader.exec_module(owner)
    return owner


@pytest.fixture
def owned_production(tmp_path, monkeypatch, production_owner):
    owner = production_owner
    root, proc = tmp_path / "root", tmp_path / "proc"
    candidate = root / "candidates/owned-output-production-isolated-owner-formatted-20261117"
    previous = root / "candidates/owned-output-trace-20261117"
    preparation = root / "run-preparation"
    preparation.mkdir(parents=True)
    sources = {
        "experiment/train.py": b"unchanged training fixture\n",
        "experiment/block_toner.py": b"unchanged full-shaped trace fixture\n",
        "experiment/owned_output_validation.py": b"unchanged owned contract fixture\n",
        "experiment/tests/test_upstream_release.py": b"preserved historical test fixture\n",
        "experiment/production_owner.py": b"preserved Wave9 owner fixture\n",
        "artifacts/pinned-sources-materialized/Toner-fixture.py": b"unchanged pinned fixture\n",
    }
    for directory in (candidate, previous):
        for name, content in sources.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (directory / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
        for arm in owner.ARMS:
            record(
                directory / f"experiment/configs/{arm}_0p5m_seed0.json",
                {"experiment": {"arm": arm}, "training": {"steps": 5120, "tokens": 2684354560, "world_size": 8}},
            )
        (directory / "experiment/execution_owner.py").write_text(
            "original validation deny\n" if directory == previous else "production dispatch fixture\n"
        )
        record(directory / "experiment/cluster.json", {"wave_name": "validation" if directory == previous else "wave10"})
    (candidate / "experiment/owned_production_owner.py").write_bytes(Path(owner.__file__).read_bytes())
    (candidate / "experiment/tests/test_owned_output_release.py").write_bytes(Path(__file__).read_bytes())
    files = {name: sha256(candidate / name) for name in owner.candidate_files(candidate)}
    original_files = {name: sha256(previous / name) for name in owner.candidate_files(previous)}
    data = root / "data/experiment_receipt.json"
    record(data, {"status": "verified"})
    archive = preparation / "source-owned-output-trace-validation-20261117.tar"
    archive.write_bytes(b"original trace archive fixture")
    original_helpers = root / "controllers/owned-output-trace-validation-20261117/tracking"
    original_helpers.mkdir(parents=True)
    original_helper = original_helpers / "run_owned_output_trace.py"
    original_helper.write_text("immutable trace controller fixture\n")
    original = {
        "status": "frozen_pending_completed_ceiling_and_owned_gpu_validation",
        "root": str(root),
        "candidate": str(previous),
        "node": owner.NODE,
        "arms": owner.ARMS,
        "files_sha256": original_files,
        "source_sha256": {
            name: digest
            for name, digest in original_files.items()
            if name.startswith("experiment/") and name.endswith(".py")
        },
        "data_receipt_sha256": sha256(data),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "helpers": str(original_helpers),
        "helpers_sha256": {original_helper.name: sha256(original_helper)},
        "environment": owner.VALIDATION_ENVIRONMENT,
        "required_gates": owner.GATES,
        "commands": owner.validation_commands(owner.PYTHON, previous, root, root / "validation/owned_output_trace"),
    }
    original_path = preparation / "owned-output-trace-binding.json"
    record(original_path, original)
    validation_path = preparation / "validation-owned-output-trace-gpu.json"
    validator_command = [
        owner.PYTHON,
        "-m",
        "experiment.validate",
        "--gpu",
        "--data",
        str(root / "data"),
        "--data-receipt",
        str(data),
        "--work",
        str(root / "validation/owned_output_trace"),
        "--receipt",
        str(validation_path),
    ]
    gpu_commands = []
    for index, command in enumerate(original["commands"]):
        log = preparation / f"gpu-{index}.log"
        log.write_text(f"successful original trace command {index}\n")
        gpu_commands.append({"command": command, "exit_code": 0, "log": str(log), "log_sha256": sha256(log)})
    validation = {
        "candidate_path": str(previous),
        "source_sha256": original["source_sha256"],
        "environment": owner.VALIDATION_ENVIRONMENT,
        "data_receipt_sha256": sha256(data),
        "config_sha256": {arm: original_files[f"experiment/configs/{arm}_0p5m_seed0.json"] for arm in owner.ARMS},
        "owner": {"controller_pid": 111, "controller_start_ticks": "222"},
        "checks": dict.fromkeys(owner.GATES, True),
        "evidence": {
            "gpu_commands": gpu_commands,
            "real_data_smokes": {arm: {} for arm in owner.ARMS},
            "validation_command": validator_command,
        },
    }
    record(validation_path, validation)
    validation_state_path = preparation / "owned-output-trace-state.json"
    validation_state = {
        "status": "passed",
        "node": owner.NODE,
        "candidate": str(previous),
        "cuda_visible_devices": owner.MASK,
        "controller_pid": 111,
        "controller_start_ticks": "222",
        "command": validator_command,
        "binding_sha256": sha256(original_path),
        "validation_sha256": sha256(validation_path),
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
    }
    record(validation_state_path, validation_state)
    helpers = root / "controllers/wave10-launch-20261117/tracking"
    helpers.mkdir(parents=True)
    helper = helpers / "owned_output_production.py"
    helper.write_text("production controller fixture\n")
    (helpers / original_helper.name).write_bytes(original_helper.read_bytes())
    (helpers / "owned_output_production_owner.py").write_bytes(Path(owner.__file__).read_bytes())
    (helpers / "test_owned_output_production_owner.py").write_bytes(Path(__file__).read_bytes())
    helper_files = {path.name: sha256(path) for path in helpers.iterdir()}
    launch_archive = preparation / "source-wave10-launch-20261117.tar"
    launch_archive.write_bytes(b"production archive fixture")
    prepare = candidate / "production-preparation.json"
    prepared = {
        "status": "prepared_owner_only_no_production_release",
        "candidate": str(candidate),
        "validation_binding_sha256": sha256(original_path),
        "files_sha256": files,
        "changed_files": sorted(owner.CHANGES),
        "owner_sha256": files["experiment/owned_production_owner.py"],
        "test_sha256": files["experiment/tests/test_owned_output_release.py"],
    }
    record(prepare, prepared)
    cpu = preparation / "production-cpu.json"
    cpu_log = preparation / "production-cpu.log"
    cpu_log.write_text("successful production candidate tests\n")
    cpu_archive = preparation / "production-cpu.tar"
    with tarfile.open(cpu_archive, "w") as bundle:
        for name in files:
            bundle.add(candidate / name, arcname=name, recursive=False)
        bundle.add(prepare, arcname="production-preparation.json", recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    cpu_receipt = {
        "status": "production_cpu_verified_gpu_release_pending",
        "candidate": str(candidate),
        "preparation_sha256": sha256(prepare),
        "files_sha256": files,
        "command": owner.cpu_command(),
        "cwd": str(candidate),
        "cuda_visible_devices": "",
        "exit_code": 0,
        "log": str(cpu_log),
        "log_sha256": sha256(cpu_log),
        "archive": str(cpu_archive),
        "archive_sha256": sha256(cpu_archive),
    }
    record(cpu, cpu_receipt)
    release_path = preparation / "wave10-launch-binding.json"
    commands = {
        node: [
            owner.PYTHON,
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
            str(data),
            "--out",
            str(root / "runs/wave10" / arm),
            "--launch-release",
            str(release_path),
        ]
        for node, arm in zip(owner.NODES, owner.ARMS, strict=True)
    }
    release = {
        "status": "frozen_validated_launch_release",
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "helpers_sha256": helper_files,
        "files_sha256": files,
        "source_sha256": {
            name: digest for name, digest in files.items() if name.startswith("experiment/") and name.endswith(".py")
        },
        "environment": owner.VALIDATION_ENVIRONMENT,
        "arms": owner.ARMS,
        "nodes": owner.NODES,
        "validation_binding_sha256": sha256(original_path),
        "validation_sha256": sha256(validation_path),
        "state_sha256": sha256(validation_state_path),
        "validation_reconstruction": {
            "status": "owned_validation_rederived_absent_exact_keeper_verified",
            "receipt_sha256": sha256(validation_path),
            "state_sha256": sha256(validation_state_path),
            "binding_sha256": sha256(original_path),
            "commands": 14,
            "checks": 26,
            "keeper": {
                "pid": 30000,
                "generation": {str(pid): {"start_ticks": str(pid)} for pid in range(30000, 30009)},
                "devices": [[str(rank), f"GPU-{rank}", "A100-SXM4-40GB"] for rank in range(8)],
                "occupied": [[str(30001 + rank), f"GPU-{rank}"] for rank in range(8)],
            },
        },
        "data_receipt_sha256": sha256(data),
        "archive": str(launch_archive),
        "archive_sha256": sha256(launch_archive),
        "preparation": str(prepare),
        "preparation_sha256": sha256(prepare),
        "cpu_receipt": str(cpu),
        "cpu_receipt_sha256": sha256(cpu),
        "commands": commands,
    }
    record(release_path, release)
    monkeypatch.setattr(owner, "TASK", candidate)
    monkeypatch.setattr(owner, "OUTPUT_ROOT", root)
    monkeypatch.setattr(owner, "ENVIRONMENT", dict(owner.VALIDATION_ENVIRONMENT))
    monkeypatch.setattr(owner, "PROC_ROOT", proc)
    monkeypatch.chdir(candidate)
    return SimpleNamespace(
        owner=owner,
        root=root,
        candidate=candidate,
        previous=previous,
        proc=proc,
        helpers=helpers,
        helper=helper,
        original_path=original_path,
        original=original,
        validation_path=validation_path,
        validation=validation,
        validation_state_path=validation_state_path,
        validation_state=validation_state,
        release_path=release_path,
        release=release,
        commands=commands,
        preparation_path=prepare,
        prepared=prepared,
        cpu_path=cpu,
        cpu=cpu_receipt,
    )


def bind_worker(evidence, arm, monkeypatch, *, rank=0):
    owner = evidence.owner
    node = owner.NODES[owner.ARMS.index(arm)]
    command = evidence.commands[node]
    worker_command = [
        owner.PYTHON,
        str(evidence.helper),
        "--root",
        str(evidence.root),
        "--phase",
        "worker",
        "--arm",
        arm,
    ]
    state = {
        "status": "running",
        "arm": arm,
        "node": node,
        "worker_pid": 20001,
        "worker_start_ticks": "71",
        "worker_command": worker_command,
        "child_pid": 20002,
        "child_start_ticks": "72",
        "command": command,
        "release_binding_sha256": sha256(evidence.release_path),
        "worker_marker": f"worker-{arm}",
        "child_marker": f"child-{arm}",
    }
    state_path = evidence.root / "run-preparation" / f"wave10-worker-{node}.json"
    record(state_path, state)
    environment = {
        **owner.VALIDATION_ENVIRONMENT,
        owner.WORKER_MARKER: state["worker_marker"],
        owner.MARKER: state["child_marker"],
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(owner.subprocess, "check_output", lambda *args, **kwargs: node)
    monkeypatch.setattr(owner.sys, "argv", [str(evidence.candidate / "experiment/train.py"), *command[7:]])
    process(evidence.proc, 20001, 1, "71", worker_command, evidence.candidate, environment)
    process(evidence.proc, 20002, 20001, "72", command, evidence.candidate, environment)
    process(evidence.proc, os.getpid(), 20002, "73", [owner.PYTHON, "-u", *command[5:]], evidence.candidate, environment)
    config_path = Path(command[command.index("--config") + 1])
    args = SimpleNamespace(
        launch_release=evidence.release_path,
        config=config_path,
        max_steps=None,
        resume=None,
        execution_boundary=None,
        verify_diagonal_capture=False,
        verify_outer_diagonal_updates=False,
    )
    return state_path, state, environment, args, json.loads(config_path.read_text())


def rebind_validation(evidence):
    record(evidence.original_path, evidence.original)
    record(evidence.validation_path, evidence.validation)
    evidence.validation_state["binding_sha256"] = sha256(evidence.original_path)
    evidence.validation_state["validation_sha256"] = sha256(evidence.validation_path)
    record(evidence.validation_state_path, evidence.validation_state)
    evidence.release.update(
        validation_binding_sha256=sha256(evidence.original_path),
        validation_sha256=sha256(evidence.validation_path),
        state_sha256=sha256(evidence.validation_state_path),
    )
    evidence.release["validation_reconstruction"].update(
        binding_sha256=sha256(evidence.original_path),
        receipt_sha256=sha256(evidence.validation_path),
        state_sha256=sha256(evidence.validation_state_path),
    )
    record(evidence.release_path, evidence.release)


@pytest.mark.parametrize("arm_index", range(4))
@pytest.mark.parametrize("rank", range(8))
def test_production_authorizes_each_released_arm_and_physical_rank_without_writing(
    owned_production, monkeypatch, arm_index, rank
):
    evidence = owned_production
    state_path, state, _, args, config = bind_worker(evidence, evidence.owner.ARMS[arm_index], monkeypatch, rank=rank)
    before = {path: path.read_bytes() for path in (evidence.original_path, evidence.release_path, state_path)}
    assert evidence.owner.require_production_owner(args, config) is None
    assert {path: path.read_bytes() for path in before} == before
    assert json.loads(state_path.read_text()) == state
    assert not (evidence.root / "runs/wave10").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "worker_generation",
        "launcher_generation",
        "worker_command",
        "launcher_command",
        "rank_command",
        "launcher_parent",
        "rank_parent",
        "ancestry_cycle",
        "worker_marker",
        "child_marker",
        "worker_cwd",
        "launcher_cwd",
        "rank_cwd",
        "worker_mask",
        "launcher_mask",
        "rank_mask",
        "inherited_worker_marker",
        "inherited_child_marker",
        "node",
        "budget",
        "resume",
        "capture",
        "out",
        "released_command",
        "environment",
        "mask",
    ],
)
def test_production_rejects_unowned_or_changed_live_execution(owned_production, monkeypatch, fault):
    evidence, owner = owned_production, owned_production.owner
    state_path, state, environment, args, config = bind_worker(evidence, owner.ARMS[0], monkeypatch)
    if fault in ("worker_marker", "child_marker", "environment", "mask"):
        key = {
            "worker_marker": owner.WORKER_MARKER,
            "child_marker": owner.MARKER,
            "environment": "CUBLAS_WORKSPACE_CONFIG",
            "mask": "CUDA_VISIBLE_DEVICES",
        }[fault]
        monkeypatch.setenv(key, "unowned")
    elif fault == "node":
        monkeypatch.setattr(owner.subprocess, "check_output", lambda *args, **kwargs: "unrelated-node")
    elif fault == "budget":
        args.max_steps = 24
    elif fault == "resume":
        args.resume = evidence.root / "old/checkpoint"
    elif fault == "capture":
        args.verify_outer_diagonal_updates = True
    elif fault == "out":
        owner.sys.argv[owner.sys.argv.index("--out") + 1] = str(evidence.root / "runs/wave9/muont")
    elif fault == "released_command":
        evidence.release["commands"][owner.NODES[0]][4] = "--nproc_per_node=2"
        record(evidence.release_path, evidence.release)
        state["release_binding_sha256"] = sha256(evidence.release_path)
        record(state_path, state)
    else:
        prefix = fault.split("_", 1)[0]
        pid = state["worker_pid"] if prefix == "worker" else state["child_pid"] if prefix == "launcher" else os.getpid()
        fields, argv = owner.observed(pid)
        parent, ticks, cwd = int(fields[1]), fields[19], evidence.candidate
        inherited = dict(environment)
        if fault.endswith("_generation"):
            ticks = "reused"
        elif fault.endswith("_command"):
            argv = [*argv, "--unbound"]
        elif fault.endswith("_parent"):
            parent = 1
        elif fault == "ancestry_cycle":
            parent = pid
        elif fault.endswith("_cwd"):
            cwd = evidence.root
        elif fault.endswith("_mask"):
            inherited["CUDA_VISIBLE_DEVICES"] = "0,1"
        else:
            key = owner.WORKER_MARKER if fault == "inherited_worker_marker" else owner.MARKER
            inherited[key] = "unowned"
        process(evidence.proc, pid, parent, ticks, argv, cwd, inherited)
    with pytest.raises(RuntimeError):
        owner.require_production_owner(args, config)
    assert not (evidence.root / "runs/wave10").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "partial_commands",
        "duplicate_command",
        "two_rank_owned",
        "owned_gate",
        "missing_owned_gate",
        "upstream_gates_only",
        "log",
        "boolean_command_exit",
        "failed_command",
        "validating",
        "cleanup",
        "restoration",
        "cleanup_error",
        "state_error",
        "owner",
        "original_candidate",
        "state_candidate",
        "source",
        "extra_source",
        "train_rehashed",
        "config_rehashed",
        "pinned_source",
        "pinned_alias",
        "original_helper",
        "helper",
        "helper_rehashed",
        "archive",
        "wave9_environment",
    ],
)
def test_release_requires_original_owned_trace_and_complete_terminal_evidence(owned_production, monkeypatch, fault):
    evidence, owner = owned_production, owned_production.owner
    _, _, _, args, config = bind_worker(evidence, owner.ARMS[0], monkeypatch)
    if fault == "partial_commands":
        evidence.validation["evidence"]["gpu_commands"].pop()
    elif fault == "duplicate_command":
        commands = evidence.validation["evidence"]["gpu_commands"]
        commands[-1]["command"] = commands[-2]["command"]
    elif fault == "two_rank_owned":
        evidence.original["commands"][1][4] = "--nproc_per_node=2"
        evidence.validation["evidence"]["gpu_commands"][1]["command"][4] = "--nproc_per_node=2"
    elif fault == "owned_gate":
        evidence.validation["checks"]["gpu_owned_output_factors"] = False
    elif fault == "missing_owned_gate":
        evidence.validation["checks"].pop("real_data_owned_output_factor_parity")
    elif fault == "upstream_gates_only":
        evidence.original["required_gates"] = list(owner.GATES[:-2])
        evidence.validation["checks"].pop("gpu_owned_output_factors")
        evidence.validation["checks"].pop("real_data_owned_output_factor_parity")
    elif fault == "log":
        Path(evidence.validation["evidence"]["gpu_commands"][0]["log"]).write_text("changed log\n")
    elif fault in ("boolean_command_exit", "failed_command"):
        evidence.validation["evidence"]["gpu_commands"][0]["exit_code"] = False if fault == "boolean_command_exit" else 1
    elif fault == "validating":
        evidence.validation_state["status"] = "validating"
    elif fault in ("cleanup", "restoration"):
        evidence.validation_state["cleanup_exit_code" if fault == "cleanup" else "keeper_restore"] = False
    elif fault == "cleanup_error":
        evidence.validation_state["cleanup_error"] = "unverified child"
    elif fault == "state_error":
        evidence.validation_state["error"] = "failed acceptance"
    elif fault == "owner":
        evidence.validation["owner"]["controller_start_ticks"] = "unrelated-validator"
    elif fault == "original_candidate":
        evidence.original["candidate"] = str(evidence.root / "candidates/upstream-full-model-production-20261117")
    elif fault == "state_candidate":
        evidence.validation_state["candidate"] = str(evidence.candidate)
    elif fault == "source":
        (evidence.candidate / "experiment/owned_production_owner.py").write_text("changed owner\n")
    elif fault == "extra_source":
        (evidence.candidate / "experiment/unlisted.py").write_text("unlisted source\n")
    elif fault in ("train_rehashed", "config_rehashed"):
        name = "experiment/train.py" if fault == "train_rehashed" else "experiment/configs/muont_0p5m_seed0.json"
        path = evidence.candidate / name
        path.write_text("unvalidated numerical change\n")
        evidence.release["files_sha256"][name] = sha256(path)
        if fault == "train_rehashed":
            evidence.release["source_sha256"][name] = sha256(path)
    elif fault == "pinned_source":
        (evidence.previous / "artifacts/pinned-sources-materialized/Toner-fixture.py").write_text("changed pin\n")
    elif fault == "pinned_alias":
        alias = evidence.candidate / "artifacts/external-sources"
        alias.unlink()
        alias.symlink_to(evidence.previous / "artifacts/pinned-sources-materialized", target_is_directory=True)
    elif fault == "original_helper":
        (Path(evidence.original["helpers"]) / "run_owned_output_trace.py").write_text("changed trace helper\n")
    elif fault in ("helper", "helper_rehashed"):
        helper = evidence.helpers / "owned_output_production_owner.py"
        helper.write_text("unvalidated owner helper\n")
        if fault == "helper_rehashed":
            evidence.release["helpers_sha256"][helper.name] = sha256(helper)
    elif fault == "archive":
        Path(evidence.release["archive"]).write_bytes(b"changed archive")
    elif fault == "wave9_environment":
        evidence.release["environment"] = {
            key: value for key, value in owner.VALIDATION_ENVIRONMENT.items() if key != "CUDA_VISIBLE_DEVICES"
        }
    rebind_validation(evidence)
    with pytest.raises(RuntimeError):
        owner.require_production_owner(args, config)
    assert not (evidence.root / "runs/wave10").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "preparation_status",
        "preparation_candidate",
        "preparation_binding",
        "preparation_files",
        "preparation_scope",
        "preparation_owner",
        "preparation_test",
        "cpu_status",
        "cpu_candidate",
        "cpu_preparation",
        "cpu_files",
        "cpu_command",
        "cpu_cwd",
        "cpu_cuda",
        "cpu_exit",
        "cpu_error",
        "cpu_log",
        "cpu_archive",
        "cpu_archive_rehashed",
        "reconstruction_missing",
        "reconstruction_status",
        "reconstruction_receipt",
        "reconstruction_state",
        "reconstruction_binding",
        "reconstruction_commands",
        "reconstruction_checks",
        "reconstruction_keeper",
        "candidate",
    ],
)
def test_release_rejects_rehashed_unaccepted_preparation_cpu_or_reconstruction(owned_production, monkeypatch, fault):
    evidence, owner = owned_production, owned_production.owner
    bind_worker(evidence, owner.ARMS[0], monkeypatch)
    if fault.startswith("preparation_"):
        key = {
            "preparation_status": "status",
            "preparation_candidate": "candidate",
            "preparation_binding": "validation_binding_sha256",
            "preparation_files": "files_sha256",
            "preparation_scope": "changed_files",
            "preparation_owner": "owner_sha256",
            "preparation_test": "test_sha256",
        }[fault]
        evidence.prepared[key] = [] if key == "changed_files" else {} if key == "files_sha256" else "unaccepted"
        record(evidence.preparation_path, evidence.prepared)
        evidence.release["preparation_sha256"] = sha256(evidence.preparation_path)
        evidence.cpu["preparation_sha256"] = sha256(evidence.preparation_path)
    elif fault.startswith("cpu_"):
        key = {
            "cpu_status": "status",
            "cpu_candidate": "candidate",
            "cpu_preparation": "preparation_sha256",
            "cpu_files": "files_sha256",
            "cpu_command": "command",
            "cpu_cwd": "cwd",
            "cpu_cuda": "cuda_visible_devices",
            "cpu_exit": "exit_code",
            "cpu_error": "error",
        }.get(fault)
        if key is not None:
            evidence.cpu[key] = False if key == "exit_code" else [] if key == "command" else "unaccepted"
        elif fault == "cpu_log":
            Path(evidence.cpu["log"]).write_text("changed CPU log\n")
        else:
            archive = Path(evidence.cpu["archive"])
            with tarfile.open(archive, "w") as bundle:
                bundle.add(evidence.candidate / "experiment/train.py", arcname="experiment/train.py")
            if fault == "cpu_archive_rehashed":
                evidence.cpu["archive_sha256"] = sha256(archive)
    elif fault.startswith("reconstruction_"):
        reconstruction = evidence.release["validation_reconstruction"]
        key = {
            "reconstruction_status": "status",
            "reconstruction_receipt": "receipt_sha256",
            "reconstruction_state": "state_sha256",
            "reconstruction_binding": "binding_sha256",
            "reconstruction_commands": "commands",
            "reconstruction_checks": "checks",
        }.get(fault)
        if key is not None:
            reconstruction[key] = 13 if key == "commands" else 24 if key == "checks" else "unaccepted"
        elif fault == "reconstruction_missing":
            evidence.release.pop("validation_reconstruction")
        else:
            reconstruction["keeper"]["occupied"].pop()
    else:
        evidence.release["candidate"] = str(evidence.root / "candidates/arbitrary-production")
    record(evidence.cpu_path, evidence.cpu)
    evidence.release["cpu_receipt_sha256"] = sha256(evidence.cpu_path)
    record(evidence.release_path, evidence.release)
    with pytest.raises(RuntimeError):
        owner.require_release(evidence.release_path)
    assert not (evidence.root / "runs/wave10").exists()
    assert not (evidence.root / "run-preparation/wave10-launch-state.json").exists()


def test_candidate_copy_uses_experiment_package_owner(tmp_path, production_owner):
    owner = production_owner
    candidate = tmp_path / "copied-candidate"
    experiment = candidate / "experiment"
    tests = experiment / "tests"
    tests.mkdir(parents=True)
    (experiment / "__init__.py").write_text("")
    shutil.copy2(owner.__file__, experiment / "owned_production_owner.py")
    shutil.copy2(__file__, tests / "test_owned_output_release.py")
    for name in ("owned_output_validation", "upstream_validation"):
        source = sys.modules[f"{owner.__package__}.{name}"].__file__
        shutil.copy2(source, experiment / f"{name}.py")
    record(
        experiment / "cluster.json",
        {
            "environment": owner.VALIDATION_ENVIRONMENT,
            "nodes": list(owner.NODES),
            "output_root": str(tmp_path),
            "python": sys.executable,
        },
    )
    (experiment / "launch.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "CLUSTER = json.loads(Path(__file__).with_name('cluster.json').read_text())\n"
        "ENVIRONMENT = CLUSTER['environment']\nNODES = tuple(CLUSTER['nodes'])\n"
        "OUTPUT_ROOT = Path(CLUSTER['output_root'])\nPYTHON = CLUSTER['python']\n"
    )
    (experiment / "reference.py").write_text("from pathlib import Path\nTASK = Path(__file__).resolve().parents[1]\n")
    (experiment / "records.py").write_text(
        "import hashlib\nfrom pathlib import Path\n"
        "def sha256(path):\n    return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-p",
            "no:cacheprovider",
            "-q",
            str(tests / "test_owned_output_release.py"),
            "-k",
            "test_production_authorizes_each_released_arm_and_physical_rank_without_writing",
        ],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (experiment / "owned_production_owner.py").read_bytes() == Path(owner.__file__).read_bytes()


@pytest.mark.parametrize(
    "old", ["wave9-launch", "owned-output-full-model", "owned-output-handoff", "owned-output-trace"]
)
def test_noncanonical_release_never_falls_back_to_old_validation(owned_production, monkeypatch, old):
    evidence = owned_production
    _, _, _, args, config = bind_worker(evidence, evidence.owner.ARMS[0], monkeypatch)
    old_path = evidence.root / "run-preparation" / f"{old}-binding.json"
    before = evidence.original_path.read_bytes()
    args.launch_release = old_path
    with pytest.raises(RuntimeError, match="noncanonical"):
        evidence.owner.require_production_owner(args, config)
    assert evidence.original_path.read_bytes() == before
    args.launch_release = evidence.release_path
    evidence.release_path.unlink()
    with pytest.raises(FileNotFoundError):
        evidence.owner.require_production_owner(args, config)
    assert not (evidence.root / "runs/wave10").exists()


@pytest.mark.parametrize("outcome", ["acknowledged", "timeout", "generation_drift"])
def test_worker_waits_only_for_the_original_launcher_acknowledgment(owned_production, monkeypatch, outcome):
    evidence, owner = owned_production, owned_production.owner
    state_path, state, environment, args, config = bind_worker(evidence, owner.ARMS[0], monkeypatch)
    state["status"] = "starting"
    state.pop("child_start_ticks")
    record(state_path, state)
    clock = [0.0]
    monkeypatch.setattr(owner.time, "monotonic", lambda: clock[0])

    def advance(delay):
        clock[0] += 31
        if outcome == "acknowledged":
            state.update(status="running", child_start_ticks="72")
            record(state_path, state)
        elif outcome == "generation_drift":
            state.update(worker_pid=20003, worker_start_ticks="new")
            process(evidence.proc, 20003, 1, "new", state["worker_command"], evidence.candidate, environment)
            record(state_path, state)

    monkeypatch.setattr(owner.time, "sleep", advance)
    if outcome == "acknowledged":
        assert owner.require_production_owner(args, config) is None
        assert json.loads(state_path.read_text())["child_start_ticks"] == "72"
    else:
        with pytest.raises(RuntimeError):
            owner.require_production_owner(args, config)
        assert "child_start_ticks" not in json.loads(state_path.read_text())
