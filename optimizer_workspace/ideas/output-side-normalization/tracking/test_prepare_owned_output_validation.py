# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the distinct source transformation and denial at the real training boundary."""

import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import prepare_owned_output_validation as preparation
import pytest
from launch_wave5 import PYTHON, read, sha256
from wave7_release import candidate_inventory

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
PREDECESSOR = ROOT / "candidates/owned-output-observed-formatted-20261117"
RECEIPT = ROOT / "run-preparation/owned-output-observer-cpu/20261117T1218/receipt.json"


@pytest.fixture
def predecessor_receipt(tmp_path):
    previous = tmp_path / "previous"
    shutil.copytree(PREDECESSOR, previous, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    record = read(RECEIPT)
    record["candidate"] = record["cwd"] = str(previous)
    original = tmp_path / "numerical-cpu.json"
    original.write_text("{}\n")
    record["predecessor_cpu_receipt"] = str(original)
    record["predecessor_cpu_receipt_sha256"] = sha256(original)
    initial = previous / "owned-output-observer-preparation.json"
    saved = read(initial)
    saved["candidate"] = str(previous)
    saved["predecessor_cpu_receipt"] = str(original)
    saved["predecessor_cpu_receipt_sha256"] = sha256(original)
    initial.write_text(json.dumps(saved))
    record["preparation"] = str(initial)
    record["preparation_sha256"] = sha256(initial)
    path = tmp_path / "cpu.json"
    path.write_text(json.dumps(record))
    return path, previous


def test_preparation_preserves_optimizer_and_configs_with_distinct_denied_source(predecessor_receipt, tmp_path):
    receipt, previous = predecessor_receipt
    candidate = tmp_path / "successor"
    result = preparation.prepare(receipt, candidate)
    assert result["status"] == "owned_output_validation_prepared_gpu_denied"
    assert preparation.require_preparation(candidate) == result
    assert candidate_inventory(previous) == read(receipt)["files_sha256"]
    unchanged = ("block_toner.py", "train.py", "upstream_outer_products.py", "diagonal_sums.py")
    for name in unchanged:
        assert (candidate / "experiment" / name).read_bytes() == (previous / "experiment" / name).read_bytes()
    for path in (previous / "experiment/configs").glob("*.json"):
        assert (candidate / "experiment/configs" / path.name).read_bytes() == path.read_bytes()
    assert not list(candidate.glob("**/*binding.json"))
    (candidate / "experiment/block_toner.py").write_text("changed numerical source\n")
    with pytest.raises(ValueError, match="preparation or source changed"):
        preparation.require_preparation(candidate)
    assert candidate_inventory(previous) == read(receipt)["files_sha256"]


def test_changed_predecessor_evidence_creates_no_successor(predecessor_receipt, tmp_path):
    receipt, previous = predecessor_receipt
    (previous / "experiment/train.py").write_text("changed predecessor\n")
    candidate = tmp_path / "uncreated"
    with pytest.raises(ValueError, match="predecessor source"):
        preparation.prepare(receipt, candidate)
    assert not candidate.exists()


def test_prepared_training_cli_rejects_prior_production_release_before_output(predecessor_receipt, tmp_path):
    receipt, _ = predecessor_receipt
    candidate = tmp_path / "successor"
    preparation.prepare(receipt, candidate)
    output = tmp_path / "uncreated-run"
    command = [
        PYTHON,
        "-m",
        "experiment.train",
        "--config",
        str(candidate / "experiment/configs/muont_0p5m_seed0.json"),
        "--data",
        str(ROOT / "data"),
        "--data-receipt",
        str(ROOT / "data/experiment_receipt.json"),
        "--out",
        str(output),
        "--launch-release",
        str(ROOT / "run-preparation/wave9-launch-binding.json"),
    ]
    completed = subprocess.run(
        command,
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "production release has not been issued" in completed.stderr
    assert not output.exists()
    assert preparation.require_preparation(candidate)["status"] == "owned_output_validation_prepared_gpu_denied"


@pytest.mark.parametrize("failure", [None, "timeout", "archive_drift"])
def test_cpu_preserves_timeout_and_rejects_archive_bytes_that_were_not_tested(
    predecessor_receipt, tmp_path, monkeypatch, failure
):
    receipt, _ = predecessor_receipt
    candidate = tmp_path / "successor"
    initial = preparation.prepare(receipt, candidate)
    output = tmp_path / "cpu-outcome"
    source = candidate / "experiment/owned_output_validation.py"

    def run(command, **kwargs):
        kwargs["stdout"].write("original CPU output\n")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    add = tarfile.TarFile.add

    def archive_file(bundle, name, *args, **kwargs):
        if failure == "archive_drift" and Path(name) == source:
            original = source.read_bytes()
            source.write_bytes(b"untested source\n")
            try:
                return add(bundle, name, *args, **kwargs)
            finally:
                source.write_bytes(original)
        return add(bundle, name, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "add", archive_file)
    if failure is None:
        result = preparation.cpu(candidate, output)
        assert result["status"] == "owned_output_validation_cpu_verified_gpu_denied"
        preparation.require_cpu_archive(Path(result["archive"]), initial, result["preparation_sha256"])
    else:
        with pytest.raises(RuntimeError, match="failure preserved"):
            preparation.cpu(candidate, output)
        result = read(output / "receipt.json")
        assert result["status"] == "owned_output_validation_cpu_failed"
        assert result["exit_code"] == (124 if failure == "timeout" else 0)
        assert ("timeout sentinel" if failure == "timeout" else "archive bytes differ") in result["error"]
    assert result["log_sha256"] == sha256(output / "candidate.log")
    assert result["archive_sha256"] == sha256(output / "source.tar")
    assert preparation.require_preparation(candidate) == initial
    assert not list(candidate.glob("**/*binding.json"))
