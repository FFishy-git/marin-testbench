# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep the corrected ownership candidate distinct and preserve every tested byte."""

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import prepare_owned_output_trace as preparation
import pytest
from launch_wave5 import PYTHON, read, sha256
from wave7_release import candidate_inventory

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


@pytest.fixture
def original(tmp_path, monkeypatch):
    frozen = copy.deepcopy(read(ROOT / preparation.PREVIOUS))
    previous = tmp_path / "previous"
    shutil.copytree(
        Path(frozen["candidate"]), previous, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    frozen["candidate"] = str(previous)
    path = tmp_path / preparation.PREVIOUS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(frozen))
    (path.parent / "owned-output-handoff-failure-evidence-20261117T1533.json").write_text("{}")
    monkeypatch.setattr(preparation, "require_original", lambda root: frozen)
    return tmp_path, previous, frozen


def test_trace_successor_preserves_failed_source_and_changes_only_its_scoped_files(original):
    root, previous, frozen = original
    candidate = root / "candidate"
    result = preparation.prepare(root, candidate)
    assert preparation.require_preparation(candidate) == result
    assert set(result["changed_files"]) == preparation.CHANGES
    assert candidate_inventory(previous) == frozen["files_sha256"]
    for name, digest in frozen["files_sha256"].items():
        if name not in preparation.CHANGES:
            assert sha256(candidate / name) == digest
    owner = (candidate / "experiment/execution_owner.py").read_text()
    assert "owned-output-trace-binding.json" in owner
    assert "run_owned_output_trace.py" in owner
    assert "owned-output-handoff-binding.json" not in owner
    assert not list(root.glob("controllers/*handoff*"))
    block = (candidate / "experiment/block_toner.py").read_text()
    assert "self.moment[width].sum(-1, keepdim=True)[selection]" in block
    assert (candidate / "experiment/owned_output_gpu_checks.py").read_text().count(
        "torch.testing.assert_close(outer, baseline[key][1], rtol=0, atol=0)"
    ) == 1


def test_existing_candidate_and_numerical_drift_are_refused(original):
    root, _, _ = original
    candidate = root / "candidate"
    preparation.prepare(root, candidate)
    with pytest.raises(FileExistsError):
        preparation.prepare(root, candidate)
    (candidate / "experiment/block_toner.py").write_text("unverified numerics\n")
    with pytest.raises(ValueError, match="source changed"):
        preparation.require_preparation(candidate)


def test_real_training_cli_still_rejects_prior_production_release_before_output(original):
    root, _, _ = original
    candidate = root / "candidate"
    preparation.prepare(root, candidate)
    out = root / "uncreated"
    result = subprocess.run(
        [
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
            str(out),
            "--launch-release",
            str(ROOT / "run-preparation/wave9-launch-binding.json"),
        ],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "production release has not been issued" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("timeout", (False, True))
def test_cpu_outcomes_and_archive_are_preserved_without_gpu_permission(original, monkeypatch, timeout):
    root, _, _ = original
    candidate = root / "candidate"
    initial = preparation.prepare(root, candidate)

    def run(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        kwargs["stdout"].write("original CPU output\n")
        if timeout:
            raise subprocess.TimeoutExpired(command, 600)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    out = root / "cpu"
    if timeout:
        with pytest.raises(RuntimeError, match="failure preserved"):
            preparation.cpu(candidate, out)
    else:
        preparation.cpu(candidate, out)
    receipt = read(out / "receipt.json")
    assert receipt["exit_code"] == (124 if timeout else 0)
    assert receipt["archive_sha256"] == sha256(out / "source.tar")
    preparation.require_cpu_archive(out / "source.tar", initial, receipt["preparation_sha256"])
    assert not list(root.glob("controllers/*handoff*"))
