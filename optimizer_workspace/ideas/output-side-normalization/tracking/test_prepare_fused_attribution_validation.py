# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Temporary successor preparation preserves issued parents and denies pre-CUDA execution."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import prepare_fused_attribution_validation as preparation
import pytest
from fused_attribution_validation_contract import EXPANSION_POLICY, trial_specs
from fused_attribution_validation_test_support import record, write_failed_profiler
from launch_wave5 import read, sha256
from wave7_release import candidate_inventory

PARENT = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/candidates/fused-attribution-full-schedule-20261118T0257"
)


@pytest.fixture
def parent(tmp_path, monkeypatch):
    root, previous = tmp_path / "root", tmp_path / "parent"
    shutil.copytree(PARENT, previous, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    write_failed_profiler(root)
    receipt_path = tmp_path / "parent-cpu.json"
    metadata = read(previous / preparation.PARENT_PREPARATION)
    for trial, spec in zip(metadata["trials"], trial_specs(), strict=True):
        trial["config"] = str(previous / spec["config_name"])
    helper = tmp_path / "parent-helpers/prepare_fused_attribution.py"
    helper.parent.mkdir()
    helper.write_text("# archived external verifier fixture\n")
    metadata["helpers_sha256"] = {helper.name: sha256(helper)}
    metadata["candidate"] = str(previous)
    record(previous / preparation.PARENT_PREPARATION, metadata)
    archive = tmp_path / "parent-source.tar"
    digest = sha256(previous / preparation.PARENT_PREPARATION)
    expected = preparation.parent_archive_inventory(metadata, digest)
    with preparation.tarfile.open(archive, "x") as bundle:
        for name in expected:
            source = helper if name.startswith("tracking/") else previous / name
            bundle.add(source, arcname=name, recursive=False)
        bundle.add(previous / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    receipt = {
        "candidate": str(previous),
        "root": str(root),
        "files_sha256": candidate_inventory(previous),
        "helpers_sha256": metadata["helpers_sha256"],
        "preparation_sha256": digest,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
    }
    record(receipt_path, receipt)
    real_run = preparation.subprocess.run

    def boundary(command, **kwargs):
        if command[:3] == [preparation.PYTHON, "-P", "-c"]:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            assert sha256(Path(kwargs["cwd"]) / helper.name) == metadata["helpers_sha256"][helper.name]
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(read(receipt_path)), stderr="")
        return real_run(command, **kwargs)

    monkeypatch.setattr(preparation.subprocess, "run", boundary)
    return root, previous, receipt_path, receipt


@pytest.fixture
def candidate_cpu(parent, monkeypatch):
    inherited_run = preparation.subprocess.run
    execution = {"exit_code": 0, "failure": None, "during": None}

    def boundary(command, **kwargs):
        if command != preparation.cpu_command():
            return inherited_run(command, **kwargs)
        candidate = Path(kwargs["cwd"])
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert kwargs["env"]["PYTHONPATH"] == str(candidate)
        assert candidate != parent[1]
        kwargs["stdout"].write("fixture candidate CPU result\n")
        if execution["during"] is not None:
            execution["during"](candidate)
        if execution["failure"] is not None:
            raise execution["failure"]
        return subprocess.CompletedProcess(command, execution["exit_code"])

    monkeypatch.setattr(preparation.subprocess, "run", boundary)
    return execution


def test_successor_exact_scope_and_pre_cuda_denial(parent, tmp_path):
    root, previous, receipt_path, receipt = parent
    before = {path: path.read_bytes() for path in previous.rglob("*") if path.is_file() and not path.is_symlink()}
    candidate = tmp_path / "successor"
    prepared = preparation.prepare(root, receipt_path, candidate)
    assert set(prepared["changed_files"]) == preparation.CHANGES
    assert prepared["expansion_policy"] == EXPANSION_POLICY
    assert prepared["allocation_predecessor"]["restored_keeper"]["pid"] == 100
    assert prepared["plan"]["planned_validation_tokens"] == 226492416
    assert candidate_inventory(previous) == receipt["files_sha256"]
    assert all(path.read_bytes() == raw for path, raw in before.items())
    for spec in trial_specs():
        assert (candidate / spec["config_name"]).read_bytes() == (previous / spec["config_name"]).read_bytes()
    output = tmp_path / "never-created"
    checked = subprocess.run(
        [
            preparation.PYTHON,
            "-B",
            "-m",
            "experiment.train",
            "--config",
            "/missing-config",
            "--data",
            "/missing-data",
            "--data-receipt",
            "/missing-receipt",
            "--out",
            str(output),
            "--launch-release",
            "/missing-binding",
        ],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert checked.returncode != 0
    assert not output.exists()
    assert not preparation.locations(root)[0].exists()
    assert not preparation.locations(root)[3].exists()
    with pytest.raises(FileExistsError):
        preparation.prepare(root, receipt_path, candidate)


@pytest.mark.parametrize("fault", ("optimizer", "config", "owner", "expansion", "predecessor"))
def test_rehashed_successor_drift_is_not_accepted(parent, tmp_path, fault):
    root, _, receipt_path, _ = parent
    candidate = tmp_path / "successor"
    prepared = preparation.prepare(root, receipt_path, candidate)
    if fault == "optimizer":
        path = candidate / "experiment/block_toner.py"
        path.write_text(path.read_text().replace("torch.ones_like(outer)", "torch.zeros_like(outer)"))
    elif fault == "config":
        path = candidate / trial_specs()[0]["config_name"]
        value = read(path)
        value["optimizer"]["auxiliary_lr"] *= 2
        record(path, value)
    elif fault == "owner":
        (candidate / "experiment/fused_attribution_owner.py").write_text(
            "def require_fused_attribution_owner(args): pass\n"
        )
    elif fault == "expansion":
        prepared["expansion_policy"]["steps"] = 320
    else:
        prepared["allocation_predecessor"]["restored_keeper"]["pid"] = 999
    prepared["files_sha256"] = candidate_inventory(candidate)
    record(candidate / preparation.PREPARATION, prepared)
    with pytest.raises((ValueError, KeyError)):
        preparation.require_preparation(root, candidate)


def test_unrestored_predecessor_refused_before_successor_copy(parent, tmp_path):
    root, _, receipt_path, _ = parent
    state_path = preparation.failed_profiler_evidence(root)["evidence_sha256"]
    path = next(root / name for name in state_path if "-state-" in name)
    state = read(path)
    state["marked_descendants_absent"] = False
    record(path, state)
    candidate = tmp_path / "successor"
    with pytest.raises(ValueError):
        preparation.prepare(root, receipt_path, candidate)
    assert not candidate.exists()


def test_parent_cpu_uses_archived_helpers_with_no_gpu_visibility(tmp_path, monkeypatch):
    candidate, root = tmp_path / "parent", tmp_path / "root"
    candidate.mkdir()
    receipt_path = tmp_path / "cpu.json"
    helper = tmp_path / "prepare_fused_attribution.py"
    helper.write_text("# exact archived acceptance helper\n")
    (candidate / "experiment").mkdir()
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    prepared = {"helpers_sha256": {helper.name: sha256(helper)}, "files_sha256": {}, "parent_metadata_sha256": {}}
    record(candidate / preparation.PARENT_PREPARATION, prepared)
    archive = tmp_path / "source.tar"
    with preparation.tarfile.open(archive, "x") as bundle:
        bundle.add(helper, arcname=f"tracking/{helper.name}", recursive=False)
        bundle.add(candidate / preparation.PARENT_PREPARATION, arcname=preparation.PARENT_PREPARATION, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    receipt = {
        "candidate": str(candidate),
        "root": str(root),
        "archive": str(archive),
        "preparation_sha256": sha256(candidate / preparation.PARENT_PREPARATION),
        "helpers_sha256": prepared["helpers_sha256"],
        "files_sha256": {},
    }
    record(receipt_path, receipt)

    def accepted(command, **kwargs):
        assert command[:3] == [preparation.PYTHON, "-P", "-c"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert Path(kwargs["cwd"]) != candidate
        assert sha256(Path(kwargs["cwd"]) / helper.name) == prepared["helpers_sha256"][helper.name]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(receipt), stderr="")

    monkeypatch.setattr(preparation.subprocess, "run", accepted)
    assert preparation.parent_cpu(root, receipt_path) == (receipt, prepared)


def test_cpu_cli_archives_exact_successor_and_helpers_without_changing_parent(
    parent, candidate_cpu, tmp_path, monkeypatch, capsys
):
    root, previous, receipt_path, parent_receipt = parent
    before = {path: path.read_bytes() for path in previous.rglob("*") if path.is_file() and not path.is_symlink()}
    parent_receipt_bytes = receipt_path.read_bytes()
    candidate, out = tmp_path / "successor", tmp_path / "cpu"
    prepared = preparation.prepare(root, receipt_path, candidate)
    argv = ["prepare", "--root", str(root), "--candidate", str(candidate)]
    monkeypatch.setattr(sys, "argv", [*argv, "--phase", "check"])
    preparation.main()
    assert json.loads(capsys.readouterr().out) == prepared
    monkeypatch.setattr(sys, "argv", [*argv, "--phase", "cpu", "--out", str(out)])
    preparation.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == preparation.require_cpu(root, out / "receipt.json")
    assert receipt["status"] == "fused_attribution_validation_cpu_verified_gpu_denied"
    assert receipt["exit_code"] == 0
    assert receipt["cuda_visible_devices"] == ""
    assert read(out / "receipt.json") == receipt
    expected_files = prepared["files_sha256"] | prepared["parent_metadata_sha256"] | {preparation.PREPARATION: ""}
    with preparation.tarfile.open(receipt["archive"]) as bundle:
        members = bundle.getmembers()
        names = set(expected_files) | {f"tracking/{name}" for name in prepared["helpers_sha256"]}
        assert len(members) == len(names) + 1
        assert {member.name for member in members} == names | {"artifacts/external-sources"}
        alias = bundle.getmember("artifacts/external-sources")
        assert alias.issym() and alias.linkname == "pinned-sources-materialized"
        for name in expected_files:
            assert bundle.getmember(name).isfile()
            with bundle.extractfile(name) as stream:
                assert stream.read() == (candidate / name).read_bytes()
        for name in prepared["helpers_sha256"]:
            with bundle.extractfile(f"tracking/{name}") as stream:
                assert (
                    stream.read()
                    == (out / "tracking" / name).read_bytes()
                    == (Path(preparation.__file__).parent / name).read_bytes()
                )
    assert candidate_inventory(previous) == parent_receipt["files_sha256"]
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert receipt_path.read_bytes() == parent_receipt_bytes
    assert not preparation.locations(root)[0].exists()
    assert not preparation.locations(root)[3].exists()
    with pytest.raises(FileExistsError):
        preparation.cpu(root, candidate, out)


@pytest.mark.parametrize("failure", ("nonzero", "timeout", "spawn"))
def test_cpu_execution_failure_preserves_unaccepted_receipt_and_archive(parent, candidate_cpu, tmp_path, failure):
    root, _, receipt_path, _ = parent
    candidate, out = tmp_path / "successor", tmp_path / "cpu"
    preparation.prepare(root, receipt_path, candidate)
    if failure == "nonzero":
        candidate_cpu["exit_code"] = 7
    elif failure == "timeout":
        candidate_cpu["failure"] = subprocess.TimeoutExpired(preparation.cpu_command(), 900)
    else:
        candidate_cpu["failure"] = OSError("fixture candidate interpreter unavailable")
    with pytest.raises(RuntimeError):
        preparation.cpu(root, candidate, out)
    receipt = read(out / "receipt.json")
    assert receipt["status"] == "needs_attention"
    assert receipt["exit_code"] == {"nonzero": 7, "timeout": 124, "spawn": 127}[failure]
    assert receipt["cuda_visible_devices"] == ""
    assert (out / "source.tar").is_file()
    assert (out / "candidate.log").read_text() == "fixture candidate CPU result\n"
    with pytest.raises(ValueError):
        preparation.require_cpu(root, out / "receipt.json")
    assert not preparation.locations(root)[0].exists()
    assert not preparation.locations(root)[3].exists()


def test_cpu_archive_creation_failure_preserves_zero_exit_without_acceptance(
    parent, candidate_cpu, tmp_path, monkeypatch
):
    root, _, receipt_path, _ = parent
    candidate, out = tmp_path / "successor", tmp_path / "cpu"
    preparation.prepare(root, receipt_path, candidate)
    original_open = preparation.tarfile.open

    def boundary(name, mode="r", *args, **kwargs):
        if Path(name) == out / "source.tar" and mode == "x":
            raise OSError("fixture archive unavailable")
        return original_open(name, mode, *args, **kwargs)

    monkeypatch.setattr(preparation.tarfile, "open", boundary)
    with pytest.raises(RuntimeError):
        preparation.cpu(root, candidate, out)
    receipt = read(out / "receipt.json")
    assert receipt["status"] == "needs_attention"
    assert receipt["exit_code"] == 0
    assert receipt["archive_sha256"] is None
    assert "error" in receipt
    with pytest.raises(ValueError):
        preparation.require_cpu(root, out / "receipt.json")


def test_final_cpu_acceptance_drift_downgrades_receipt(parent, candidate_cpu, tmp_path, monkeypatch):
    root, _, receipt_path, _ = parent
    directory = tmp_path / "helpers"
    directory.mkdir()
    for name in preparation.helper_hashes():
        shutil.copy2(Path(preparation.__file__).parent / name, directory / name)
    monkeypatch.setattr(preparation, "__file__", str(directory / Path(preparation.__file__).name))
    candidate, out = tmp_path / "successor", tmp_path / "cpu"
    preparation.prepare(root, receipt_path, candidate)
    inherited_run = preparation.subprocess.run

    def boundary(command, **kwargs):
        if command[:3] == [preparation.PYTHON, "-P", "-c"] and (out / "receipt.json").exists():
            path = directory / "test_fused_attribution_validation.py"
            path.write_bytes(path.read_bytes() + b"\n")
        return inherited_run(command, **kwargs)

    monkeypatch.setattr(preparation.subprocess, "run", boundary)
    with pytest.raises(RuntimeError):
        preparation.cpu(root, candidate, out)
    receipt = read(out / "receipt.json")
    assert receipt["status"] == "needs_attention"
    assert receipt["exit_code"] == 0
    assert "error" in receipt
    assert receipt["archive_sha256"] == sha256(out / "source.tar")
    assert not preparation.locations(root)[0].exists()
    assert not preparation.locations(root)[3].exists()


@pytest.mark.parametrize("target", ("config", "metadata", "helper"))
def test_cpu_source_drift_during_tests_is_retained_but_never_accepted(parent, candidate_cpu, tmp_path, target):
    root, _, receipt_path, _ = parent
    candidate, out = tmp_path / "successor", tmp_path / "cpu"
    prepared = preparation.prepare(root, receipt_path, candidate)

    def drift(observed):
        assert observed == candidate
        if target == "config":
            path = candidate / trial_specs()[0]["config_name"]
        elif target == "metadata":
            path = candidate / preparation.PARENT_PREPARATION
        else:
            path = out / "tracking" / next(iter(prepared["helpers_sha256"]))
        path.write_bytes(path.read_bytes() + b"\n")

    candidate_cpu["during"] = drift
    with pytest.raises(RuntimeError):
        preparation.cpu(root, candidate, out)
    receipt = read(out / "receipt.json")
    assert receipt["status"] == "needs_attention"
    assert receipt["exit_code"] == 0
    assert "error" in receipt
    assert (out / "source.tar").is_file()
    with pytest.raises(ValueError):
        preparation.require_cpu(root, out / "receipt.json")


@pytest.mark.parametrize(
    "fault", ("boolean_exit", "log", "saved_helper", "own_helper", "archive_bytes", "archive_extra")
)
def test_cpu_receipt_rejects_rehashed_evidence_and_current_helper_drift(
    parent, candidate_cpu, tmp_path, monkeypatch, fault
):
    root, _, receipt_path, _ = parent
    candidate, out = tmp_path / "successor", tmp_path / "cpu"
    preparation.prepare(root, receipt_path, candidate)
    receipt = preparation.cpu(root, candidate, out)
    if fault == "boolean_exit":
        receipt["exit_code"] = False
    elif fault == "log":
        Path(receipt["log"]).write_text("different output\n")
    elif fault == "saved_helper":
        name = next(iter(receipt["helpers_sha256"]))
        path = out / "tracking" / name
        path.write_bytes(path.read_bytes() + b"\n")
        receipt["helpers_sha256"][name] = sha256(path)
    elif fault == "own_helper":
        directory = tmp_path / "mutable-helpers"
        shutil.copytree(out / "tracking", directory)
        path = directory / "fused_attribution_validation.py"
        path.write_bytes(path.read_bytes() + b"\n")
        monkeypatch.setattr(preparation, "__file__", str(directory / Path(preparation.__file__).name))
    else:
        archive = Path(receipt["archive"])
        original = out / "original.tar"
        archive.rename(original)
        with preparation.tarfile.open(original) as source, preparation.tarfile.open(archive, "x") as destination:
            for member in source.getmembers():
                if fault == "archive_bytes" and member.name == "experiment/train.py":
                    path = out / "different-train.py"
                    path.write_bytes((candidate / member.name).read_bytes() + b"\n")
                    destination.add(path, arcname=member.name, recursive=False)
                elif member.isfile():
                    with source.extractfile(member) as stream:
                        destination.addfile(member, stream)
                else:
                    destination.addfile(member)
            if fault == "archive_extra":
                destination.add(candidate / "experiment/train.py", arcname="unexpected.py", recursive=False)
        receipt["archive_sha256"] = sha256(archive)
    record(out / "receipt.json", receipt)
    with pytest.raises(ValueError):
        preparation.require_cpu(root, out / "receipt.json")
