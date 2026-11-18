# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test diagnostic successor preservation and refusal before any CUDA execution."""

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import prepare_fused_continuation as preparation
import pytest
from fused_continuation_contract import CONFIG, PROTOCOLS
from selected_lr_release_owner import candidate_files, read, sha256

PARENT = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/candidates/fused-attribution-validation-strict-20261118T0418"
)


@pytest.fixture
def parent(tmp_path, monkeypatch):
    root, previous = tmp_path / "root", tmp_path / "parent"
    shutil.copytree(PARENT, previous, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    root.mkdir()
    binding_path = preparation.parent_locations(root)[0]
    binding_path.parent.mkdir()
    binding_path.write_text(json.dumps({"fixture": "failed-parent-source"}) + "\n")
    frozen = {
        "candidate": str(previous),
        "files_sha256": candidate_files(previous),
        "data_receipt_sha256": "prepared-data",
    }
    monkeypatch.setattr(preparation, "original", lambda root: frozen)
    monkeypatch.setattr(
        preparation,
        "failed_attribution",
        lambda root, original: {
            "status": "failed_attribution_exact_keeper_pending_live_handoff",
            "evidence_sha256": {str(binding_path.relative_to(root)): sha256(binding_path)},
            "restored_keeper": {"pid": 929078},
        },
    )
    return root, previous, frozen


def test_diagnostic_preserves_numerical_parent_and_denies_unowned_execution(parent, tmp_path):
    root, previous, frozen = parent
    candidate = tmp_path / "successor"
    prepared = preparation.prepare(root, candidate)
    assert preparation.require_preparation(root, candidate) == prepared
    assert candidate_files(previous) == frozen["files_sha256"]
    assert (candidate / CONFIG).read_bytes() == (previous / CONFIG).read_bytes()
    changed = {
        name
        for name in prepared["files_sha256"].keys() | frozen["files_sha256"].keys()
        if prepared["files_sha256"].get(name) != frozen["files_sha256"].get(name)
    }
    assert changed == preparation.CHANGES
    commands = prepared["plan"]["commands"]
    assert len(commands) == len(PROTOCOLS) * 3
    assert all(("--trace-continuation" in command) == (index >= 3) for index, command in enumerate(commands))
    ast.parse((candidate / "experiment/train.py").read_bytes())
    target = tmp_path / "never-created"
    denied = subprocess.run(
        [
            preparation.cpu_command()[0],
            "-B",
            "-m",
            "experiment.train",
            "--config",
            str(candidate / CONFIG),
            "--data",
            "/missing",
            "--data-receipt",
            "/missing",
            "--out",
            str(target),
            "--launch-release",
            "/missing",
            "--max-steps",
            "24",
        ],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert denied.returncode != 0
    assert not target.exists()
    assert "no separate prepared execution owner" not in denied.stderr
    assert "no canonical frozen execution binding" in denied.stderr


@pytest.mark.parametrize("fault", ("optimizer", "config", "owner"))
def test_rehashed_diagnostic_cannot_accept_changed_numerical_scope(parent, tmp_path, fault):
    root, _, _ = parent
    candidate = tmp_path / "successor"
    prepared = preparation.prepare(root, candidate)
    path = (
        candidate
        / {
            "optimizer": "experiment/block_toner.py",
            "config": CONFIG,
            "owner": "experiment/fused_attribution_owner.py",
        }[fault]
    )
    if fault == "config":
        config = read(path)
        config["optimizer"]["auxiliary_lr"] *= 2
        path.write_text(json.dumps(config) + "\n")
    else:
        path.write_bytes(path.read_bytes() + b"\n# changed evidence\n")
    prepared["files_sha256"] = candidate_files(candidate)
    (candidate / preparation.PREPARATION).write_text(json.dumps(prepared) + "\n")
    with pytest.raises(ValueError):
        preparation.require_preparation(root, candidate)
