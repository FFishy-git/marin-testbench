# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise exact-source attribution preparation and pre-CUDA execution denial."""

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import prepare_fused_attribution as preparation
import pytest
from launch_wave5 import PYTHON, read
from wave7_release import candidate_inventory

PRODUCTION = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/candidates/selected-lr-wave12-post-failure-formatted-20261117"
)


@pytest.fixture
def parent(tmp_path, monkeypatch):
    root, previous = tmp_path / "root", tmp_path / "original"
    shutil.copytree(
        PRODUCTION / "experiment",
        previous / "experiment",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        PRODUCTION / "artifacts/pinned-sources-materialized",
        previous / "artifacts/pinned-sources-materialized",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (previous / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    (previous / "historical-preparation.json").write_text('{"status":"historical-evidence"}\n')
    binding, helpers = preparation.paths(root)
    binding.parent.mkdir(parents=True)
    binding.write_text('{"status":"fixture-frozen-original"}\n')
    frozen = {
        "candidate": str(previous),
        "helpers": str(helpers),
        "files_sha256": candidate_inventory(previous),
        "data_receipt_sha256": "fixture-data",
        "archive": str(root / "original.tar"),
        "archive_sha256": "fixture-archive",
    }
    evidence = {"analysis_sha256": "fixture-final", "cloud_sha256": "fixture-verified"}
    monkeypatch.setattr(preparation, "original", lambda observed_root: frozen)
    monkeypatch.setattr(preparation, "result_evidence", lambda observed_root, observed_frozen: evidence)
    return root, previous, frozen


def test_full_grid_preserves_schedule_and_paired_settings_without_mutating_parent():
    reference = {
        family: read(PRODUCTION / f"experiment/configs/{family}_0p5m_seed0.json")
        for family in ("muont", "outer_leftdiag_fused")
    }
    before = copy.deepcopy(reference)
    configs = preparation.configurations(PRODUCTION)
    assert len(configs) == 9
    for label, lr in preparation.LR_GRID:
        for family in preparation.FAMILIES:
            config = configs[preparation.config_name(family, label)]
            expected = copy.deepcopy(reference["outer_leftdiag_fused" if family == "null_leftdiag_fused" else family])
            expected["experiment"]["arm"] = family
            expected["optimizer"]["lr"] = lr
            assert config == expected
            assert config["schedule"] == {
                "warmup_steps": 0,
                "stable_steps": 1024,
                "decay_steps": 4096,
                "kind": "linear",
                "min_lr_ratio": 0.0,
            }
    assert reference == before


def test_candidate_preserves_originals_and_refuses_trainer_before_missing_data(parent, tmp_path):
    root, previous, frozen = parent
    candidate = tmp_path / "attribution"
    record = preparation.prepare(root, candidate)
    expected_changes = {
        "experiment/block_toner.py",
        "experiment/configure.py",
        "experiment/train.py",
        "experiment/tests/test_block_toner.py",
        "experiment/fused_attribution_owner.py",
        "experiment/fused_attribution_checks.py",
        "experiment/tests/test_fused_attribution_checks.py",
        "experiment/cluster.json",
        *(preparation.config_name(family, label) for label, _ in preparation.LR_GRID for family in preparation.FAMILIES),
    }
    changed = {
        name
        for name in record["files_sha256"].keys() | frozen["files_sha256"].keys()
        if record["files_sha256"].get(name) != frozen["files_sha256"].get(name)
    }
    assert changed == expected_changes
    assert candidate_inventory(previous) == frozen["files_sha256"]
    assert (candidate / "historical-preparation.json").read_bytes() == (
        previous / "historical-preparation.json"
    ).read_bytes()
    assert len(record["trials"]) == 9
    assert record["total_training_tokens"] == 24159191040
    assert record["reserved_confirmation_seeds"] == [3, 4, 5]
    output = tmp_path / "uncreated-run"
    result = subprocess.run(
        [
            PYTHON,
            "-B",
            "-m",
            "experiment.train",
            "--config",
            str(candidate / preparation.config_name("null_leftdiag_fused", "center")),
            "--data",
            str(root / "missing-data"),
            "--data-receipt",
            str(root / "missing-receipt"),
            "--out",
            str(output),
            "--launch-release",
            str(root / "missing-release"),
        ],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "RuntimeError: Fused attribution requires separate validated execution ownership" in result.stderr
    assert not output.exists()
    assert not (root / "runs").exists()
    with pytest.raises(FileExistsError):
        preparation.prepare(root, candidate)


@pytest.mark.parametrize("target", ("lr", "numerics"))
def test_rehashed_candidate_cannot_change_declared_lr_or_null_mathematics(parent, tmp_path, target):
    root, _, _ = parent
    candidate = tmp_path / "attribution"
    record = preparation.prepare(root, candidate)
    if target == "lr":
        path = candidate / preparation.config_name("null_leftdiag_fused", "center")
        config = read(path)
        config["optimizer"]["auxiliary_lr"] *= 2
        path.write_text(json.dumps(config) + "\n")
    else:
        path = candidate / "experiment/block_toner.py"
        path.write_text(path.read_text().replace("torch.ones_like(outer)", "torch.zeros_like(outer)"))
    record["files_sha256"] = candidate_inventory(candidate)
    (candidate / preparation.PREPARATION).write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="exact transformation"):
        preparation.require_preparation(root, candidate)


def test_candidate_cpu_failure_is_retained_without_acceptance_or_launch(parent, tmp_path, monkeypatch):
    root, _, _ = parent
    candidate = tmp_path / "attribution"
    preparation.prepare(root, candidate)
    command = [PYTHON, "-c", "raise SystemExit(7)"]
    monkeypatch.setattr(preparation, "cpu_command", lambda: command)
    out = tmp_path / "cpu-evidence"
    with pytest.raises(RuntimeError, match="CPU/archive verification failed"):
        preparation.cpu(root, candidate, out)
    receipt = read(out / "receipt.json")
    assert receipt["status"] == "fused_attribution_cpu_failed"
    assert receipt["exit_code"] == 7
    assert receipt["cuda_visible_devices"] == ""
    assert (out / "source.tar").is_file()
    assert not (root / "runs").exists()
    with pytest.raises(ValueError, match="CPU outcome"):
        preparation.require_cpu(root, out / "receipt.json")


@pytest.mark.parametrize("corruption", ("none", "extra_column", "final_metric", "ulp", "tokens"))
def test_predecessor_cloud_acceptance_covers_every_scalar_and_unchanged_export(tmp_path, monkeypatch, corruption):
    source = PRODUCTION.parents[1]
    root = tmp_path / "root"
    destination = root / "run-preparation"
    destination.mkdir(parents=True)
    frozen = read(source / "run-preparation/wave12-launch-binding.json")
    frozen["helpers"] = str(tmp_path / "helpers")
    source_analysis = read(source / "run-preparation/analysis-wave12.json")
    history = {
        arm: {
            key: value
            for key, value in read(source / "run-preparation/swanlab-upload-wave12.json")["runs"][arm].items()
            if key in ("manifest_sha256", "metrics_sha256", "checkpoint_receipt_sha256", "checkpoint_rank_bytes")
        }
        for arm in frozen["arms"]
    }
    cloud = read(source / "run-preparation/swanlab-upload-wave12.json")
    columns = {
        arm: {
            "point_counts": copy.deepcopy(cloud["runs"][arm]["verification"]["point_counts"]),
            "final_metrics": copy.deepcopy(cloud["runs"][arm]["verification"]["final_metrics"]),
        }
        for arm in frozen["arms"]
    }
    shutil.copy2(source / "run-preparation/wave12-launch-binding.json", destination)
    shutil.copy2(source / "run-preparation/analysis-wave12.json", destination)
    verification = cloud["runs"]["outer_leftdiag_fused"]["verification"]
    if corruption == "extra_column":
        column = next(key for key in verification["point_counts"] if key.startswith("runtime/"))
        verification["point_counts"][column] -= 1
    elif corruption == "final_metric":
        verification["final_metrics"]["train/loss"] += 0.01
    elif corruption == "ulp":
        verification["serialization_errors"]["train/loss"]["max_ulp"] = 3
    elif corruption == "tokens":
        verification["serialization_errors"]["tokens"]["max_absolute"] = 1
    (destination / "swanlab-upload-wave12.json").write_text(json.dumps(cloud) + "\n")
    rebuilt = {"analysis": {k: v for k, v in source_analysis.items() if k != "time"}, "histories": history}

    def remote_output(command, **kwargs):
        # The only fake is the frozen subprocess output, leaving receipt parsing and acceptance real.
        output = columns if command[0].endswith(".venv-logging/bin/python") else rebuilt
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(preparation.subprocess, "run", remote_output)
    if corruption == "none":
        result = preparation.result_evidence(root, frozen)
        assert result["completed_histories"] == history
    else:
        with pytest.raises(ValueError, match="cloud identity"):
            preparation.result_evidence(root, frozen)


def test_archive_write_failure_preserves_cpu_outcome_receipt(parent, tmp_path, monkeypatch):
    root, _, _ = parent
    candidate = tmp_path / "attribution"
    preparation.prepare(root, candidate)
    monkeypatch.setattr(preparation, "cpu_command", lambda: [PYTHON, "-c", "raise SystemExit(0)"])
    original_open = preparation.tarfile.open

    def fail_new_archive(name, mode="r", *args, **kwargs):
        if mode == "x":
            raise OSError("fixture archive creation failed")
        return original_open(name, mode, *args, **kwargs)

    monkeypatch.setattr(preparation.tarfile, "open", fail_new_archive)
    out = tmp_path / "cpu-failure"
    with pytest.raises(RuntimeError, match="CPU/archive verification failed"):
        preparation.cpu(root, candidate, out)
    receipt = read(out / "receipt.json")
    assert receipt["status"] == "fused_attribution_cpu_failed"
    assert receipt["exit_code"] == 0
    assert receipt["archive_sha256"] is None
    assert "archive creation failed" in receipt["error"]
