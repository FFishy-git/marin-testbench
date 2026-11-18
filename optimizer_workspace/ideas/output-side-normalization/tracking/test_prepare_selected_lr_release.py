# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check exact release preparation, parent preservation and archived CPU failures."""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import prepare_selected_lr_release as preparation
import pytest
from launch_wave5 import read, sha256
from selected_lr_validation import PREPARATION as PARENT_PREPARATION
from wave7_release import candidate_inventory


def record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


@pytest.fixture
def parent(tmp_path, monkeypatch):
    root, previous = tmp_path / "root", tmp_path / "original"
    experiment = previous / "experiment"
    (experiment / "tests").mkdir(parents=True)
    (experiment / "__init__.py").write_text("")
    (experiment / "train.py").write_text(
        "import argparse\nfrom pathlib import Path\n"
        "from .selected_lr_owner import require_selected_lr_training\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        '    parser.add_argument("--config", type=Path, required=True)\n'
        '    parser.add_argument("--out", type=Path, required=True)\n'
        "    args = parser.parse_args()\n"
        "    require_selected_lr_training(args)\n"
        "    numerical_body = args.config.read_bytes()\n"
        "    args.out.mkdir()\n"
        "    return numerical_body\n\n"
        "if __name__ == '__main__':\n    main()\n"
    )
    (experiment / "selected_lr_owner.py").write_text("def require_selected_lr_training(args):\n    return None\n")
    (experiment / "production_owner.py").write_text("HISTORICAL_OWNER = True\n")
    (experiment / "selected_lr_validation.py").write_text("UNCHANGED_VALIDATION = True\n")
    for arm in ("muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused"):
        record(experiment / f"configs/{arm}_0p5m_seed0.json", {"arm": arm, "lr": 0.064, "steps": 5120})
    record(experiment / "cluster.json", {"wave_name": "validation", "nodes": ["original"], "gpus_per_run": 8})
    pinned = previous / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    (pinned / "numerical.bin").write_bytes(b"unchanged original pinned source")
    (previous / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    record(previous / PARENT_PREPARATION, {"status": "original"})
    binding = root / "run-preparation/wave11-selected-lr-validation-binding.json"
    record(binding, {"status": "original_frozen"})
    artifacts = {}
    for name in ("cpu_receipt", "controller_cpu_receipt", "archive"):
        path = root / f"{name}.json"
        record(path, {"status": "preserved_original"})
        artifacts[name] = str(path)
        artifacts[f"{name}_sha256"] = sha256(path)
    frozen = {
        "candidate": str(previous),
        "files_sha256": candidate_inventory(previous),
        "helpers": str(root / "controllers/original/tracking"),
        "preparation_sha256": sha256(previous / PARENT_PREPARATION),
        "data_receipt_sha256": "original-data",
        **artifacts,
    }
    monkeypatch.setattr(preparation, "original", lambda root: frozen)
    return root, previous, frozen


def test_release_preparation_retains_original_denial_numerics_and_exact_four_changes(parent, tmp_path):
    root, previous, frozen = parent
    before = candidate_inventory(previous)
    candidate = tmp_path / "release-candidate"
    result = preparation.prepare(root, candidate)
    files = candidate_inventory(candidate)
    changed = {name for name in files.keys() | before.keys() if files.get(name) != before.get(name)}
    assert changed == {
        "experiment/train.py",
        "experiment/cluster.json",
        "experiment/selected_lr_release_owner.py",
        "experiment/tests/test_selected_lr_release_owner.py",
    }
    assert before == candidate_inventory(previous) == frozen["files_sha256"]
    for name in before.keys() - changed:
        assert (candidate / name).read_bytes() == (previous / name).read_bytes()
    assert (candidate / PARENT_PREPARATION).read_bytes() == (previous / PARENT_PREPARATION).read_bytes()
    assert (
        "from experiment import selected_lr_release_owner as owner"
        in (candidate / "experiment/tests/test_selected_lr_release_owner.py").read_text()
    )
    assert preparation.require_preparation(root, candidate) == result
    denied = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "experiment.train",
            "--config",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "not-created"),
            "--launch-release",
            str(root / "missing-release.json"),
        ],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert denied.returncode != 0
    assert not (tmp_path / "not-created").exists()
    assert not (root / "run-preparation/wave11-launch-binding.json").exists()
    assert not (root / "runs/wave11").exists()
    with pytest.raises(FileExistsError):
        preparation.prepare(root, candidate)


@pytest.mark.parametrize("fault", ("config", "train", "cluster", "owner"))
def test_release_preparation_rejects_rehashed_numerics_or_allowed_file_replacement(parent, tmp_path, fault):
    root, _, _ = parent
    candidate = tmp_path / "release-candidate"
    preparation.prepare(root, candidate)
    if fault == "config":
        (candidate / "experiment/configs/muont_0p5m_seed0.json").write_text('{"lr":999}\n')
    elif fault == "train":
        path = candidate / "experiment/train.py"
        path.write_text(path.read_text().replace("return numerical_body", "return b'changed numerics'"))
    elif fault == "cluster":
        path = candidate / "experiment/cluster.json"
        cluster = read(path)
        cluster["gpus_per_run"] = 1
        record(path, cluster)
    else:
        (candidate / "experiment/selected_lr_release_owner.py").write_text(
            "def require_selected_lr_release_owner(args):\n    return None\n"
        )
    path = candidate / preparation.PREPARATION
    prepared = read(path)
    prepared["files_sha256"] = candidate_inventory(candidate)
    record(path, prepared)
    with pytest.raises(ValueError, match="scope"):
        preparation.require_preparation(root, candidate)


@pytest.mark.parametrize("outcome", ("pass", "exit", "timeout", "source_drift"))
def test_release_cpu_preserves_exact_archive_and_withholds_failed_source(parent, tmp_path, monkeypatch, outcome):
    root, _, _ = parent
    candidate, out = tmp_path / "release-candidate", tmp_path / "cpu"
    preparation.prepare(root, candidate)

    def run(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        kwargs["stdout"].write("observed CPU boundary\n")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 600)
        if outcome == "source_drift":
            (candidate / "experiment/configs/muont_0p5m_seed0.json").write_text("{}\n")
        return subprocess.CompletedProcess(command, 1 if outcome == "exit" else 0)

    monkeypatch.setattr(preparation.subprocess, "run", run)
    if outcome == "pass":
        result = preparation.cpu(root, candidate, out)
        assert preparation.require_cpu(root, out / "receipt.json") == result
    else:
        with pytest.raises(RuntimeError, match="preserved"):
            preparation.cpu(root, candidate, out)
        with pytest.raises(ValueError):
            preparation.require_cpu(root, out / "receipt.json")
    receipt = read(out / "receipt.json")
    assert receipt["exit_code"] == {"pass": 0, "exit": 1, "timeout": 124, "source_drift": 0}[outcome]
    assert receipt["log_sha256"] == sha256(out / "candidate.log")
    assert receipt["archive_sha256"] == sha256(out / "source.tar")
    with tarfile.open(out / "source.tar") as bundle:
        names = set(bundle.getnames())
        assert names == set(receipt["files_sha256"]) | {
            preparation.PREPARATION,
            "artifacts/external-sources",
            "tracking/prepare_selected_lr_release.py",
            "tracking/selected_lr_release_owner.py",
            "tracking/test_selected_lr_release_owner.py",
        }
        assert len(bundle.getmembers()) == len(names)
    assert not (root / "run-preparation/wave11-launch-binding.json").exists()


def test_copied_release_owner_tests_run_standalone_in_temporary_candidate(parent, tmp_path):
    root, _, _ = parent
    candidate = tmp_path / "release-candidate"
    preparation.prepare(root, candidate)
    result = subprocess.run(
        preparation.cpu_command(),
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (root / "runs/wave11").exists()
