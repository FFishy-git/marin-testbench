# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU source-contract checks; no live GPU ownership or keeper manipulation."""

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import prepare_root_preparation_diagnostic as preparation
import pytest
import root_preparation_contract as contract
from selected_lr_release_owner import candidate_files, sha256

PARENT = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/candidates/fused-continuation-owner-fixture-20261118T0634"
)


@pytest.fixture
def parent(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    previous = tmp_path / "parent"
    shutil.copytree(PARENT, previous, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    binding = root / preparation.PARENT_BINDING
    binding.parent.mkdir()
    binding.write_text('{"fixture": "authenticated diagnostic"}\n')
    predecessor = {
        "status": "saved_ns_reconstructed_exact_outputs_restored_pending_live_root_handoff",
        "evidence_sha256": {},
        "restored_keeper": {"pid": 1268038},
    }
    frozen = {
        "candidate": str(previous),
        "files_sha256": candidate_files(previous),
        "data_receipt_sha256": "unchanged-data",
    }
    monkeypatch.setattr(preparation, "parent", lambda root: (frozen, predecessor))
    return root, previous, frozen


def test_candidate_changes_only_readiness_owner_trainer_and_tests(parent, tmp_path):
    root, previous, frozen = parent
    candidate = tmp_path / "candidate"
    prepared = preparation.prepare(root, candidate)
    assert preparation.require_preparation(root, candidate) == prepared
    assert candidate_files(previous) == frozen["files_sha256"]
    assert prepared["plan"]["tolerance"] == {"rtol": 1e-5, "atol": 1e-6}
    assert (candidate / contract.CONFIG).read_bytes() == (previous / contract.CONFIG).read_bytes()
    for name in (
        "experiment/optimizer.py",
        "experiment/block_toner.py",
        "experiment/toner_state.py",
        "experiment/records.py",
    ):
        assert (candidate / name).read_bytes() == (previous / name).read_bytes()
    train = (candidate / "experiment/train.py").read_text()
    ast.parse(train)
    assert train.index("restore_checkpoint(") < train.index("checked_preparation(") < train.index("ContinuationTrace(")
    assert train.count("checked_preparation(") == 1
    assert 'root_preparation_receipt["identity"] = identity' in train
    assert "atomic_json(receipt_path, root_preparation_receipt)" in train
    assert len(prepared["plan"]["commands"]) == 12
    assert (candidate / "experiment/prepare_restored_roots.py").read_bytes() == (
        Path(preparation.__file__).parent / "prepare_restored_roots.py"
    ).read_bytes()


def test_twelve_commands_keep_exact_full_schedule_and_separate_source_paths(parent, tmp_path):
    root, _, _ = parent
    candidate = tmp_path / "candidate"
    preparation.prepare(root, candidate)
    commands = contract.validation_commands(candidate, root, contract.locations(root)[3])
    assert len(commands) == 12
    for index, command in enumerate(commands):
        regime, protocol, mode = (
            contract.REGIMES[index // 6],
            contract.PROTOCOLS[index % 6 // 3],
            contract.MODES[index % 3],
        )
        assert ("--trace-continuation" in command) == (protocol == "trace")
        assert ("--prepare-restored-roots" in command) == (regime == "prepared" and mode == "resumed")
        assert ("--resume" in command) == (mode == "resumed")
        assert command[command.index("--max-steps") + 1] == ("22" if mode == "interrupted" else "24")
        assert command[command.index("--launch-release") + 1] == str(contract.locations(root)[0])
        assert command[command.index("--config") + 1] == str(candidate / contract.CONFIG)
    assert all("fused-continuation-binding-20261118.json" not in " ".join(command) for command in commands)


@pytest.mark.parametrize("fault", ("optimizer", "config", "trainer", "helper"))
def test_rehashed_scope_or_intervention_mutation_cannot_be_accepted(parent, tmp_path, fault):
    root, _, _ = parent
    candidate = tmp_path / "candidate"
    prepared = preparation.prepare(root, candidate)
    name = {
        "optimizer": "experiment/block_toner.py",
        "config": contract.CONFIG,
        "trainer": "experiment/train.py",
        "helper": "experiment/prepare_restored_roots.py",
    }[fault]
    path = candidate / name
    if fault == "config":
        config = json.loads(path.read_text())
        config["optimizer"]["ns_steps"] = 9
        path.write_text(json.dumps(config))
    else:
        path.write_bytes(path.read_bytes() + b"\n# substituted source\n")
    prepared["files_sha256"] = candidate_files(candidate)
    (candidate / contract.PREPARATION).write_text(json.dumps(prepared))
    with pytest.raises(ValueError):
        preparation.require_preparation(root, candidate)


def test_actual_entry_denies_unowned_launch_before_outputs_or_cuda(parent, tmp_path):
    root, _, _ = parent
    candidate = tmp_path / "candidate"
    preparation.prepare(root, candidate)
    target = tmp_path / "never-created"
    result = subprocess.run(
        [
            contract.PYTHON,
            "-B",
            "-m",
            "experiment.train",
            "--config",
            str(candidate / contract.CONFIG),
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
            "--prepare-restored-roots",
        ],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0 and not target.exists()
    assert "no canonical frozen execution binding" in result.stderr
    assert "no separate prepared execution owner" not in result.stderr


def test_predecessor_substitution_refuses_before_candidate_creation(tmp_path):
    root = tmp_path / "root"
    (root / "run-preparation").mkdir(parents=True)
    (root / preparation.REPLAY_OBSERVATION).write_text("{}")
    candidate = tmp_path / "never-created"
    with pytest.raises(ValueError, match="independently accepted"):
        preparation.prepare(root, candidate)
    assert not candidate.exists()


def test_candidate_test_root_allows_real_spawned_rank_workers(parent, tmp_path):
    root, _, _ = parent
    candidate = tmp_path / "candidate"
    preparation.prepare(root, candidate)
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    shutil.copy2(
        preparation.helper_sources()["root_diagnostic_pytest_guard.py"],
        helpers / "root_diagnostic_pytest_guard.py",
    )
    result = subprocess.run(
        [
            *preparation.cpu_command(candidate),
            "-k",
            "distributed_diagonal_moments_pool_all_rank_rows or "
            "distributed_output_statistics_pool_unequal_rank_counts_before_ema",
        ],
        cwd=candidate,
        env=os.environ
        | {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_OFFLINE": "1",
            "PYTHONPATH": os.pathsep.join((str(candidate), str(helpers))),
        },
        capture_output=True,
        text=True,
        timeout=55,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout and "170 deselected" in result.stdout


def test_helpers_retain_timeout_guard_and_no_duplicate_paths():
    sources = preparation.helper_sources()
    assert "root_diagnostic_pytest_guard.py" in sources
    assert "prepare_restored_roots.py" in sources
    assert len(sources) == len(set(sources))
    assert sha256(sources["prepare_restored_roots.py"]) == preparation.ROOT_HELPER_SHA256
    command = preparation.cpu_command(PARENT)
    assert "--timeout=60" in command and "--session-timeout=600" in command
    assert preparation.SAFE_MARKERS in command
