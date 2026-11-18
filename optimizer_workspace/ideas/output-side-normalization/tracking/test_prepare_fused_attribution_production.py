# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual source copies and CPU archives, without live infrastructure."""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import prepare_fused_attribution_production as preparation
import pytest
from fused_attribution_production_contract import config_name
from fused_attribution_production_test_support import parent as parent_fixture
from fused_attribution_production_test_support import record
from prepare_fused_attribution_production import candidate_inventory, read, sha256


@pytest.fixture
def parent(tmp_path):
    return parent_fixture(tmp_path)


def test_preparation_preserves_parent_and_denies_training_before_output(parent, tmp_path):
    before = candidate_inventory(parent.previous)
    candidate = tmp_path / "production"
    prepared = preparation.prepare(parent.root, candidate)
    after = candidate_inventory(candidate)
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    assert changed == {
        "experiment/train.py",
        "experiment/fused_attribution_production_owner.py",
        "experiment/fused_attribution_production_contract.py",
        "experiment/tests/test_fused_attribution_production_owner.py",
    }
    assert candidate_inventory(parent.previous) == before
    for name in before.keys() - changed:
        assert (candidate / name).read_bytes() == (parent.previous / name).read_bytes()
    for path in parent.previous.glob("*.json"):
        assert (candidate / path.name).read_bytes() == path.read_bytes()
    assert preparation.require_preparation(parent.root, candidate) == prepared
    output = tmp_path / "not-created"
    denied = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "experiment.train",
            "--config",
            str(tmp_path / "missing-config.json"),
            "--out",
            str(output),
            "--launch-release",
            str(parent.root / "missing-production-release.json"),
        ],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert denied.returncode != 0
    assert "separate accepted production release" in denied.stderr
    assert not output.exists()
    assert not (parent.root / "runs").exists()
    assert not (parent.root / "run-preparation/wave13-launch-binding.json").exists()
    with pytest.raises(FileExistsError):
        preparation.prepare(parent.root, candidate)


@pytest.mark.parametrize("fault", ("helper", "cpu_log", "archive"))
def test_original_refuses_corrupt_parent_before_executing_check(parent, fault):
    if fault == "helper":
        helpers = Path(parent.frozen["helpers"])
        (helpers / "fused_attribution_validation_controller.py").write_text("raise SystemExit(0)\n")
    elif fault == "cpu_log":
        receipt = read(Path(parent.frozen["cpu_receipt"]))
        Path(receipt["log"]).write_text("changed original CPU log\n")
    else:
        Path(parent.frozen["archive"]).write_bytes(b"not an original source archive")
    with pytest.raises(ValueError):
        preparation.original(parent.root)
    assert not parent.marker.exists()


@pytest.mark.parametrize("fault", ("config", "train", "owner", "metadata"))
def test_preparation_rejects_repaired_hash_changes(parent, tmp_path, fault):
    candidate = tmp_path / "production"
    preparation.prepare(parent.root, candidate)
    path = {
        "config": candidate / config_name("muont", "lower"),
        "train": candidate / "experiment/train.py",
        "owner": candidate / "experiment/fused_attribution_production_owner.py",
        "metadata": candidate / preparation.PARENT_PREPARATION,
    }[fault]
    path.write_bytes(path.read_bytes() + b"\n# altered\n")
    record_path = candidate / preparation.PREPARATION
    prepared = read(record_path)
    prepared["files_sha256"] = candidate_inventory(candidate)
    record(record_path, prepared)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        preparation.require_preparation(parent.root, candidate)


@pytest.mark.parametrize("outcome", ("pass", "exit", "timeout", "source_drift", "archive_failure"))
def test_cpu_archives_exact_closure_and_preserves_failures(parent, tmp_path, monkeypatch, outcome):
    candidate, out = tmp_path / "production", tmp_path / "cpu"
    preparation.prepare(parent.root, candidate)
    original_run = subprocess.run

    def run(command, **kwargs):
        if command != preparation.cpu_command():
            return original_run(command, **kwargs)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        kwargs["stdout"].write("candidate CPU boundary\n")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 900)
        if outcome == "source_drift":
            (candidate / "experiment/numerical.py").write_text("CHANGED = True\n")
        if outcome == "archive_failure":
            (out / "source.tar").mkdir()
        return subprocess.CompletedProcess(command, 1 if outcome == "exit" else 0)

    monkeypatch.setattr(preparation.subprocess, "run", run)
    if outcome == "pass":
        result = preparation.cpu(parent.root, candidate, out)
        assert preparation.require_cpu(parent.root, out / "receipt.json") == result
    else:
        with pytest.raises(RuntimeError, match="preserved"):
            preparation.cpu(parent.root, candidate, out)
        with pytest.raises((ValueError, RuntimeError)):
            preparation.require_cpu(parent.root, out / "receipt.json")
    receipt = read(out / "receipt.json")
    assert receipt["exit_code"] == (124 if outcome == "timeout" else 1 if outcome == "exit" else 0)
    assert receipt["log_sha256"] == sha256(out / "candidate.log")
    if outcome == "archive_failure":
        assert receipt["archive_sha256"] is None
    else:
        assert receipt["archive_sha256"] == sha256(out / "source.tar")
        prepared = read(candidate / preparation.PREPARATION)
        with tarfile.open(out / "source.tar") as bundle:
            assert set(bundle.getnames()) == set(
                preparation.archive_inventory(prepared, receipt["preparation_sha256"])
            ) | {"artifacts/external-sources"}
        if outcome == "pass":
            copied = out / "tracking/selected_lr_release_owner.py"
            copied.write_bytes(copied.read_bytes() + b"\n# altered inherited helper\n")
            with pytest.raises(ValueError):
                preparation.require_cpu(parent.root, out / "receipt.json")
    assert not (parent.root / "run-preparation/wave13-launch-binding.json").exists()


def test_generated_owner_suite_runs_from_candidate_package(parent, tmp_path):
    candidate = tmp_path / "production"
    preparation.prepare(parent.root, candidate)
    result = subprocess.run(
        preparation.cpu_command(),
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (parent.root / "runs").exists()
