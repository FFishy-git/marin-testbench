# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check real frozen-source copies, execution denial, cleanup and exact CPU archives."""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import prepare_selected_lr_stage_profile as preparation
import pytest
import torch
from launch_wave5 import NODES, PYTHON, read, sha256
from selected_lr_stage_profiler import StageProfiler
from selected_lr_validation import ARMS, ENVIRONMENT
from torch.profiler import ProfilerActivity
from wave7_release import candidate_inventory

PRODUCTION = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/candidates/selected-lr-wave12-post-failure-formatted-20261117"
)


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def source_archive(path: Path, candidate: Path, files: dict, metadata: dict, helpers: dict) -> None:
    with tarfile.open(path, "x") as bundle:
        for name in files:
            bundle.add(candidate / name, arcname=name, recursive=False)
        for name, source in metadata.items():
            bundle.add(source, arcname=name, recursive=False)
        for name, source in helpers.items():
            bundle.add(source, arcname=f"tracking/{name}", recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)


@pytest.fixture
def parent(tmp_path):
    root, previous = tmp_path / "root", tmp_path / "original"
    shutil.copytree(
        PRODUCTION / "experiment", previous / "experiment", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copytree(
        PRODUCTION / "artifacts/pinned-sources-materialized",
        previous / "artifacts/pinned-sources-materialized",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (previous / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    files = candidate_inventory(previous)
    preparation_path = previous / preparation.PARENT_PREPARATION
    record(
        preparation_path,
        {"status": "prepared_release_owner_gpu_pending", "candidate": str(previous), "files_sha256": files},
    )
    record(previous / "historical-preparation.json", {"status": "preserved-historical-evidence"})
    binding_path, helpers = preparation.paths(root)
    helpers.mkdir(parents=True)
    real_binding, real_helpers = preparation.paths(PRODUCTION.parents[1])
    for name in read(real_binding)["helpers_sha256"]:
        shutil.copy2(real_helpers / name, helpers / name)
    helper = helpers / "selected_lr_production.py"
    helper.write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "assert sys.argv[3:] == ['--phase','check']\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n"
        "assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'\n"
        "print((Path(sys.argv[2])/'run-preparation/wave12-launch-binding.json').read_text())\n"
    )
    data_path = root / "data/experiment_receipt.json"
    record(data_path, {"status": "fixture-data"})
    cpu_log = root / "parent-cpu.log"
    cpu_log.write_text("fixture CPU result\n")
    cpu_archive, release_archive = root / "parent-cpu.tar", root / "parent-release.tar"
    owner = previous / "experiment/selected_lr_release_owner.py"
    cpu_helpers = {"selected_lr_release_owner.py": owner}
    source_archive(cpu_archive, previous, files, {preparation.PARENT_PREPARATION: preparation_path}, cpu_helpers)
    frozen_helpers = {path.name: path for path in helpers.iterdir() if path.is_file()}
    source_archive(release_archive, previous, files, {"preparation.json": preparation_path}, frozen_helpers)
    cpu_path, controller_path = root / "parent-cpu.json", root / "controller-cpu.json"
    record(
        cpu_path,
        {
            "status": "selected_lr_release_cpu_verified_gpu_pending",
            "root": str(root),
            "candidate": str(previous),
            "files_sha256": files,
            "preparation_sha256": sha256(preparation_path),
            "helpers_sha256": {name: sha256(path) for name, path in cpu_helpers.items()},
            "command": preparation.cpu_command(),
            "cwd": str(previous),
            "cuda_visible_devices": "",
            "exit_code": 0,
            "log": str(cpu_log),
            "log_sha256": sha256(cpu_log),
            "archive": str(cpu_archive),
            "archive_sha256": sha256(cpu_archive),
        },
    )
    record(controller_path, {"status": "fixture-frozen-controller-cpu"})
    frozen = {
        "status": preparation.PARENT_STATUS,
        "root": str(root),
        "candidate": str(previous),
        "helpers": str(helpers),
        "helpers_sha256": {name: sha256(path) for name, path in frozen_helpers.items()},
        "environment": ENVIRONMENT,
        "arms": list(ARMS),
        "nodes": list(NODES),
        "files_sha256": files,
        "preparation": str(preparation_path),
        "preparation_sha256": sha256(preparation_path),
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": sha256(cpu_path),
        "controller_cpu_receipt": str(controller_path),
        "controller_cpu_receipt_sha256": sha256(controller_path),
        "archive": str(release_archive),
        "archive_sha256": sha256(release_archive),
        "data_receipt_sha256": sha256(data_path),
    }
    record(binding_path, frozen)
    return root, previous, frozen


@pytest.fixture
def candidate(parent, tmp_path):
    root, _, _ = parent
    path = tmp_path / "profile"
    preparation.prepare(root, path)
    return path


@pytest.fixture
def isolated_preparation(parent, tmp_path):
    _, _, frozen = parent
    directory = tmp_path / "local-tracking"
    directory.mkdir()
    for name in preparation.helper_binding(frozen)["helpers_sha256"]:
        shutil.copy2(Path(preparation.__file__).parent / name, directory / name)
    spec = importlib.util.spec_from_file_location(
        "isolated_stage_preparation", directory / "prepare_selected_lr_stage_profile.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, directory


def test_preparation_preserves_frozen_source_metadata_and_denies_actual_trainer(parent, candidate):
    root, previous, frozen = parent
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | frozen["files_sha256"] if files.get(name) != frozen["files_sha256"].get(name)
    }
    assert changed == {
        "experiment/train.py",
        "experiment/block_toner.py",
        "experiment/selected_lr_stage_profiler.py",
        "experiment/selected_lr_stage_profile_owner.py",
        "experiment/tests/test_selected_lr_stage_profiler.py",
    }
    assert candidate_inventory(previous) == frozen["files_sha256"]
    for name in frozen["files_sha256"].keys() - changed:
        assert (candidate / name).read_bytes() == (previous / name).read_bytes()
    for name in (preparation.PARENT_PREPARATION, "historical-preparation.json"):
        assert (candidate / name).read_bytes() == (previous / name).read_bytes()
    assert preparation.require_preparation(root, candidate)["status"] == preparation.STATUS
    output = candidate.parent / "not-created"
    command = [
        PYTHON,
        "-B",
        "-m",
        "experiment.train",
        "--config",
        str(candidate / "missing-config"),
        "--data",
        str(root / "missing-data"),
        "--data-receipt",
        str(root / "missing-receipt"),
        "--out",
        str(output),
        "--launch-release",
        str(preparation.paths(root)[0]),
        "--profile-updates",
        "201",
        "220",
    ]
    result = subprocess.run(
        command,
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "separate generation-bound profiling release" in result.stderr
    assert "missing-config" not in result.stderr
    assert not output.exists()
    assert not (root / "runs").exists()
    with pytest.raises(FileExistsError):
        preparation.prepare(root, candidate)


def test_preparation_rejects_rehashed_instrumentation_that_changes_numerics(parent, candidate):
    root, _, _ = parent
    path = candidate / "experiment/block_toner.py"
    text = path.read_text()
    assert 'group["lr"]' in text
    path.write_text(text.replace('lr=group["lr"]', 'lr=2 * group["lr"]'))
    prepared_path = candidate / preparation.PREPARATION
    prepared = read(prepared_path)
    prepared["files_sha256"] = candidate_inventory(candidate)
    record(prepared_path, prepared)
    with pytest.raises(ValueError, match="scope"):
        preparation.require_preparation(root, candidate)


def test_profiled_actual_block_step_preserves_weights_momentum_and_nonfinite_guard(parent, candidate, tmp_path):
    _, previous, _ = parent
    script = (
        "import importlib,importlib.util,json,sys,torch\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "from experiment import block_toner as module\n"
        "from experiment.selected_lr_stage_profiler import StageProfiler\n"
        "from torch.profiler import ProfilerActivity\n"
        "original=Path(sys.argv[2])\n"
        "spec=importlib.util.spec_from_file_location('_original_profile_reference',"
        "original/'experiment/__init__.py',submodule_search_locations=[str(original/'experiment')])\n"
        "package=importlib.util.module_from_spec(spec)\nsys.modules[spec.name]=package\n"
        "spec.loader.exec_module(package)\n"
        "baseline=importlib.import_module('_original_profile_reference.block_toner')\n"
        "generator=torch.Generator().manual_seed(73)\n"
        "weights=[torch.randn(4,4,generator=generator),torch.randn(8,4,generator=generator)]\n"
        "gradients=[torch.randn(4,4,generator=generator),torch.randn(8,4,generator=generator)]\n"
        "args=SimpleNamespace(muon_momentum=0.95,muon_nesterov=True,muon_ns_steps=10,"
        "muon_eps=1e-5,muon_ns_mode='f32')\n"
        "def engine(source):\n"
        " fast=source.BlockToner.__new__(source.BlockToner)\n"
        " fast.arm=source.BLOCK_ARMS['outer_leftdiag_fused']\n"
        " fast.input=fast.output=fast.input_diagonal=fast.output_diagonal=None\n"
        " fast.world,fast.rank,fast.device,fast.pg=1,0,torch.device('cpu'),None\n"
        " fast.args=args\n fast.lagged,fast.last_metrics,fast.steps={}, {},0\n"
        " fast.update_verifier=None\n"
        " fast.params=[torch.nn.Parameter(weight.clone()) for weight in weights]\n"
        " for parameter,gradient in zip(fast.params,gradients,strict=True): parameter.grad=gradient.clone()\n"
        " fast.shape_groups=[(tuple(weight.shape),[[(index,'q',parameter)]]) "
        "for index,(weight,parameter) in enumerate(zip(weights,fast.params,strict=True))]\n"
        " fast.momentum={shape:torch.zeros(1,*shape) for shape,_ in fast.shape_groups}\n"
        " fast.param_groups=[{'params':fast.params,'lr':0.02,'weight_decay':0.1}]\n"
        " return fast\n"
        "reference,actual=engine(baseline),engine(module)\nreference.step()\n"
        "profiler=StageProfiler(Path(sys.argv[1]),first=1,last=1,rank=0,provenance={},"
        "activities=[ProfilerActivity.CPU])\n"
        "profiler.install(actual,module)\nprofiler.begin_update(1)\nactual.step()\nprofiler.end_update(1)\n"
        "assert all(torch.equal(a,b) for a,b in zip(actual.params,reference.params,strict=True))\n"
        "assert all(torch.equal(actual.momentum[s],reference.momentum[s]) for s in actual.momentum)\n"
        "assert actual.steps == reference.steps == 1\n"
        "actual.params[-1].grad[0,0]=float('inf')\n"
        "before=[p.detach().clone() for p in actual.params]\n"
        "try: actual.step()\n"
        "except FloatingPointError: pass\n"
        "else: raise AssertionError('Nonfinite gradient accepted')\n"
        "assert all(torch.equal(a,b) for a,b in zip(actual.params,before,strict=True))\n"
        "print(json.dumps({'weights_exact':True,'momentum_exact':True,'nonfinite_refused':True}))\n"
    )
    output = tmp_path / "block-profile"
    result = subprocess.run(
        [PYTHON, "-B", "-c", script, str(output), str(previous)],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"weights_exact": True, "momentum_exact": True, "nonfinite_refused": True}
    summary = read(output / "summary.json")
    assert summary["status"] == "captured"
    regions = {region["name"] for region in summary["regions"]}
    assert {
        "osn/optimizer_step",
        "osn/block_update",
        "osn/parameter_gradient_pack",
        "osn/finite_check_wait",
        "osn/parameter_apply",
    } <= regions


def test_real_trace_export_failure_runs_prepared_engine_and_process_group_cleanup(candidate, tmp_path):
    tree = ast.parse((candidate / "experiment/train.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    training_try = next(node for node in main.body if isinstance(node, ast.Try))
    cleanup = ast.Module(body=training_try.finalbody, type_ignores=[])
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_bytes(b"file rather than output directory")
    profiler = StageProfiler(
        blocked_parent / "rank0", first=1, last=2, rank=0, provenance={}, activities=[ProfilerActivity.CPU]
    )
    profiler.begin_update(1)
    with profiler.scope("fixture_add"):
        assert torch.equal(torch.ones(2) + 1, torch.full((2,), 2.0))
    profiler.end_update(1)
    events = []
    engines = [
        SimpleNamespace(
            side=SimpleNamespace(synchronize=lambda: events.append("side")),
            pool=SimpleNamespace(shutdown=lambda wait: events.append("pool")),
        )
        for _ in range(2)
    ]
    namespace = {
        "stage_profile": profiler,
        "toner_engines": lambda fast: engines,
        "fast": object(),
        "world": 8,
        "dist": SimpleNamespace(destroy_process_group=lambda: events.append("process_group")),
    }
    with pytest.raises(NotADirectoryError):
        exec(compile(cleanup, "prepared_cleanup", "exec"), namespace)
    assert events == ["side", "pool", "side", "pool", "process_group"]
    assert not (blocked_parent / "rank0/summary.json").exists()


def test_candidate_cpu_round_trip_runs_emitted_tests_and_verifies_exact_archive(parent, candidate, tmp_path):
    root, _, frozen = parent
    output = tmp_path / "cpu"
    result = preparation.cpu(root, candidate, output)
    assert result["status"] == preparation.CPU_STATUS
    assert result["exit_code"] == 0
    assert preparation.require_cpu(root, output / "receipt.json") == result
    assert "passed" in (output / "candidate.log").read_text()
    prepared = read(candidate / preparation.PREPARATION)
    assert len(prepared["helpers_sha256"]) == 24
    assert len(prepared["new_helpers_sha256"]) == 5
    assert len(prepared["inherited_helpers_sha256"]) == 19
    assert prepared["inherited_helpers_sha256"] == {
        name: frozen["helpers_sha256"][name] for name in prepared["inherited_helpers_sha256"]
    }
    assert {"analyze_tuning.py", "compare_runs.py", "prepare_tuning.py"} <= prepared["helpers_sha256"].keys()
    expected = preparation.archive_inventory(prepared, result["preparation_sha256"])
    with tarfile.open(output / "source.tar") as archive:
        assert set(archive.getnames()) == set(expected) | {"artifacts/external-sources"}
        assert len(archive.getmembers()) == len(expected) + 1
    with pytest.raises(FileExistsError):
        preparation.cpu(root, candidate, output)
    missing_archive = output / "missing-inherited.tar"
    with tarfile.open(output / "source.tar") as source, tarfile.open(missing_archive, "x") as target:
        for member in source.getmembers():
            if member.name == "tracking/compare_runs.py":
                continue
            if member.isfile():
                with source.extractfile(member) as stream:
                    target.addfile(member, stream)
            else:
                target.addfile(member)
    record(
        output / "missing-receipt.json",
        result | {"archive": str(missing_archive), "archive_sha256": sha256(missing_archive)},
    )
    with pytest.raises(ValueError, match="inventory"):
        preparation.require_cpu(root, output / "missing-receipt.json")
    with tarfile.open(output / "source.tar", "a") as archive:
        archive.add(output / "candidate.log", arcname="unexpected.log", recursive=False)
    record(output / "receipt.json", result | {"archive_sha256": sha256(output / "source.tar")})
    with pytest.raises(ValueError, match="inventory"):
        preparation.require_cpu(root, output / "receipt.json")


@pytest.mark.parametrize("mutation", ("tampered", "missing", "unexpected"))
def test_prepare_rejects_unfrozen_transitive_helpers_before_creating_candidate(
    parent, isolated_preparation, tmp_path, mutation
):
    root, previous, frozen = parent
    local, directory = isolated_preparation
    inherited = directory / "compare_runs.py"
    if mutation == "tampered":
        inherited.write_bytes(inherited.read_bytes() + b"\nraise RuntimeError('unfrozen helper')\n")
    elif mutation == "missing":
        inherited.unlink()
    else:
        (directory / "unfrozen_stage_dependency.py").write_text("VALUE = 1\n")
        profiler = directory / "selected_lr_stage_profiler.py"
        profiler.write_bytes(
            profiler.read_bytes() + b"\nif False:\n    import unfrozen_stage_dependency as dependency\n"
        )
    path = tmp_path / "rejected-profile"
    with pytest.raises(ValueError, match="helper"):
        local.prepare(root, path)
    assert not path.exists()
    assert read(preparation.paths(root)[0]) == frozen
    assert candidate_inventory(previous) == frozen["files_sha256"]
    assert sha256(Path(frozen["archive"])) == frozen["archive_sha256"]


@pytest.mark.parametrize("changed_helper", ("compare_runs.py", "selected_lr_stage_profiler.py"))
def test_cpu_helper_drift_preserves_failed_evidence_even_with_zero_exit(
    parent, isolated_preparation, tmp_path, monkeypatch, changed_helper
):
    root, previous, frozen = parent
    local, directory = isolated_preparation
    candidate = tmp_path / "drift-profile"
    local.prepare(root, candidate)
    prepared = read(candidate / preparation.PREPARATION)
    real_run = subprocess.run

    def cpu_boundary(command, **kwargs):
        if command != preparation.cpu_command():
            return real_run(command, **kwargs)
        kwargs["stdout"].write("CPU boundary returned zero while helper source changed\n")
        source = directory / changed_helper
        source.write_bytes(source.read_bytes() + b"\n# changed during CPU verification\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", cpu_boundary)
    output = tmp_path / "drift-cpu"
    with pytest.raises(RuntimeError, match="preserved"):
        local.cpu(root, candidate, output)
    receipt = read(output / "receipt.json")
    assert receipt["status"] == "selected_lr_stage_profile_cpu_failed"
    assert receipt["exit_code"] == 0
    assert receipt["error"]
    assert receipt["log_sha256"] == sha256(output / "candidate.log")
    assert receipt["archive_sha256"] == sha256(output / "source.tar")
    with tarfile.open(output / "source.tar") as archive:
        assert {f"tracking/{name}" for name in prepared["helpers_sha256"]} <= set(archive.getnames())
    with pytest.raises(ValueError):
        local.require_cpu(root, output / "receipt.json")
    assert candidate_inventory(previous) == frozen["files_sha256"]
    assert read(preparation.paths(root)[0]) == frozen
    assert not (root / "runs").exists()


@pytest.mark.parametrize("outcome", ("exit", "timeout"))
def test_cpu_failure_preserves_log_archive_and_denies_acceptance(parent, candidate, tmp_path, monkeypatch, outcome):
    root, _, _ = parent
    real_run = subprocess.run

    def cpu_boundary(command, **kwargs):
        if command != preparation.cpu_command():
            return real_run(command, **kwargs)
        kwargs["stdout"].write("preserved failed CPU boundary\n")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 600)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", cpu_boundary)
    output = tmp_path / "failed-cpu"
    with pytest.raises(RuntimeError, match="preserved"):
        preparation.cpu(root, candidate, output)
    receipt = read(output / "receipt.json")
    assert receipt["status"] == "selected_lr_stage_profile_cpu_failed"
    assert receipt["exit_code"] == (124 if outcome == "timeout" else 1)
    assert receipt["log_sha256"] == sha256(output / "candidate.log")
    assert receipt["archive_sha256"] == sha256(output / "source.tar")
    with pytest.raises(ValueError):
        preparation.require_cpu(root, output / "receipt.json")
    assert not (root / "runs").exists()
