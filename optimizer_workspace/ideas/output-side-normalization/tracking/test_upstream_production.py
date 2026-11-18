# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep production preparation within ownership changes and preserve one-shot launch failures."""

import json
import os
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path

import pytest
import upstream_production as production
from launch_wave5 import read, sha256
from wave7_release import candidate_inventory


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def predecessor(tmp_path, monkeypatch):
    root, candidate = tmp_path / "root", tmp_path / "validated"
    (root / "run-preparation").mkdir(parents=True)
    experiment = candidate / "experiment"
    experiment.mkdir(parents=True)
    (experiment / "execution_owner.py").write_text(
        "from .reference import TASK\n\n"
        "def require_training_owner(args, config):\n"
        "    if args.launch_release is not None:\n"
        '        raise RuntimeError("Upstream full-budget production release has not been issued")\n'
    )
    (experiment / "train.py").write_text("def train():\n    return 42\n")
    (experiment / "tests").mkdir()
    (experiment / "tests/test_upstream_release.py").write_text("def test_pending():\n    assert True\n")
    record(experiment / "cluster.json", {"wave_name": "validated"})
    for arm in production.ARMS:
        record(experiment / f"configs/{arm}_0p5m_seed0.json", {"arm": arm, "steps": 5120, "lr": 0.016})
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    frozen = {
        "candidate": str(candidate),
        "files_sha256": candidate_inventory(candidate),
        "helpers": str(candidate / "tracking"),
        "helpers_sha256": {},
    }
    record(root / "run-preparation/upstream-full-model-corrected-binding.json", frozen)
    monkeypatch.setattr(production, "original", lambda root: frozen)
    return root, candidate, frozen


def test_owner_only_preparation_preserves_validated_numerics_and_rejects_later_drift(predecessor, tmp_path):
    root, previous, frozen = predecessor
    candidate = tmp_path / "production"
    before = candidate_inventory(previous)
    result = production.prepare(root, candidate)
    assert set(result["changed_files"]) == production.CHANGES
    assert candidate_inventory(previous) == before
    assert (candidate / "experiment/train.py").read_bytes() == (previous / "experiment/train.py").read_bytes()
    assert production.require_preparation(root, candidate)["files_sha256"] == result["files_sha256"]
    with pytest.raises(FileExistsError):
        production.prepare(root, candidate)
    (candidate / "experiment/train.py").write_text("def train():\n    return 99\n")
    with pytest.raises(ValueError, match="scope"):
        production.require_preparation(root, candidate)
    assert frozen["files_sha256"] == before


def test_incomplete_validation_prevents_release_without_creating_production_intent(predecessor, tmp_path, monkeypatch):
    root, _, _ = predecessor
    candidate = tmp_path / "production"
    production.prepare(root, candidate)
    record(tmp_path / "cpu.json", {"status": "fixture"})
    monkeypatch.setattr(production, "require_cpu", lambda *a: {"candidate": str(candidate)})

    def pending(*args):
        raise RuntimeError("Original full-model queue/controller has not completed")

    monkeypatch.setattr(production, "require_completed_validation", pending)
    with pytest.raises(RuntimeError, match="has not completed"):
        production.freeze(root, tmp_path / "cpu.json")
    assert not production.paths(root)[0].exists()
    assert not production.paths(root)[1].exists()
    assert not (root / "runs/wave9").exists()


def test_allocation_preflight_failure_preserves_unlaunched_state(tmp_path, monkeypatch):
    (tmp_path / "run-preparation").mkdir()
    monkeypatch.setattr(production, "binding", lambda root: {})
    monkeypatch.setattr(production, "require_completed_validation", lambda root: {})

    def unavailable(root, release, arm, phase):
        raise RuntimeError("allocation occupied")

    monkeypatch.setattr(production, "remote", unavailable)
    with pytest.raises(RuntimeError, match="occupied"):
        production.launch(tmp_path)
    assert not (tmp_path / "run-preparation/wave9-launch-state.json").exists()
    assert not (tmp_path / "runs/wave9").exists()


def test_worker_retains_failure_and_refuses_restoration_with_unverified_children(tmp_path, monkeypatch):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    release_path, _ = production.paths(tmp_path)
    record(release_path, {"status": "bound"})
    arm = production.ARMS[0]
    release = {"helpers": str(tmp_path)}
    command = production.worker_command(tmp_path, release, arm)
    path = production.worker_path(tmp_path, arm)
    record(
        path.with_suffix(".start.json"),
        {
            "pid": os.getpid(),
            "start_ticks": "123",
            "worker_marker": "owned",
            "release_binding_sha256": sha256(release_path),
        },
    )
    record(
        preparation / "wave9-launch-state.json",
        {
            "status": "launching",
            "binding_sha256": sha256(release_path),
            "worker_markers": {arm: "owned"},
        },
    )
    monkeypatch.setenv(production.WORKER_MARKER, "owned")
    monkeypatch.setattr(production, "binding", lambda root: release)
    monkeypatch.setattr(production, "observed_process", lambda pid: ("S", "123", command))
    monkeypatch.setattr(production, "node_guard", lambda *args: nullcontext([]))
    monkeypatch.setattr(production, "allocation", lambda *args: None)
    monkeypatch.setattr(production, "healthy_keeper", lambda: None)
    monkeypatch.setattr(production, "release_keeper", lambda state, path: state.update(keeper_release_attempted=True))
    restored = []
    monkeypatch.setattr(production, "node_action", lambda root, action: restored.append(action))

    def failed(root, release, state, path):
        state["cleanup_error"] = "live unverified child"
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(production, "run_child", failed)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        production.run_arm(tmp_path, arm)
    state = read(path)
    assert state["status"] == "needs_attention"
    assert "refused" in state["keeper_restore"]
    assert restored == []


@pytest.mark.parametrize("drift", ["helper", "candidate"])
def test_reconstruction_source_drift_preserves_unissued_release_paths(predecessor, tmp_path, monkeypatch, drift):
    root, _, frozen = predecessor
    candidate = tmp_path / "production"
    production.prepare(root, candidate)

    def passed(command, **kwargs):
        kwargs["stdout"].write("candidate CPU pass\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(production.subprocess, "run", passed)
    production.cpu(root, candidate, tmp_path / "cpu")
    helpers = tmp_path / "mutable"
    helpers.mkdir()
    for name in production.EXTRA_HELPERS:
        shutil.copy2(Path(production.__file__).parent / name, helpers / name)
    monkeypatch.setattr(production, "__file__", str(helpers / "upstream_production.py"))
    frozen_helpers = tmp_path / "original-helpers"
    frozen_helpers.mkdir()
    frozen["helpers"] = str(frozen_helpers)
    frozen["helpers_sha256"] = {}

    def reconstruction(root):
        changed = helpers / "upstream_production.py" if drift == "helper" else candidate / "experiment/train.py"
        changed.write_text(changed.read_text() + "\n# changed while rederiving\n")
        return {}

    monkeypatch.setattr(production, "require_completed_validation", reconstruction)
    with pytest.raises(ValueError):
        production.freeze(root, tmp_path / "cpu/receipt.json")
    release, installed = production.paths(root)
    assert not release.exists()
    assert not installed.exists()
    assert not (root / "run-preparation/source-wave9-launch-20261117.tar").exists()
    assert not (root / "runs/wave9").exists()


@pytest.mark.parametrize("occupancy", ["partial", "shared_gpu", "foreign"])
def test_allocation_refuses_inexact_keeper_before_release(tmp_path, monkeypatch, occupancy):
    keeper_file = tmp_path / "keeper.pid"
    keeper_file.write_text("100\n")
    generation = [(pid, str(pid), pid) for pid in range(100, 109)]
    monkeypatch.setattr(production, "keeper_generation", lambda pid: generation)
    rows = [f"{pid}, GPU-{pid}" for pid in range(101, 109)]
    if occupancy == "partial":
        rows.pop()
    elif occupancy == "shared_gpu":
        rows[-1] = "108, GPU-101"
    else:
        rows[-1] = "999, GPU-108"
    monkeypatch.setattr(production.subprocess, "check_output", lambda *args, **kwargs: "\n".join(rows))
    with pytest.raises(RuntimeError):
        production.healthy_keeper(keeper_file)
    assert keeper_file.read_text() == "100\n"
    rows[:] = [f"{pid}, GPU-{pid}" for pid in range(101, 109)]
    assert production.healthy_keeper(keeper_file)["generation"] == generation


@pytest.mark.parametrize("failure", ["spawn_failed", "exited"])
def test_startup_absence_becomes_terminal_without_retry(tmp_path, monkeypatch, failure):
    root = tmp_path
    release_path, _ = production.paths(root)
    record(release_path, {"status": "released"})
    arm = production.ARMS[0]
    release = {"helpers": str(tmp_path / "helpers")}
    path = production.worker_path(root, arm).with_suffix(".start.json")
    intent = {
        "status": failure,
        "arm": arm,
        "node": production.node_for(arm),
        "command": production.worker_command(root, release, arm),
        "worker_marker": "absent-wave9-fixture",
        "release_binding_sha256": sha256(release_path),
    }
    if failure == "exited":
        intent.update(pid=123456789, start_ticks="123")
    record(path, intent)
    record(
        root / "run-preparation/wave9-launch-state.json",
        {"status": "launching", "worker_markers": {arm: intent["worker_marker"]}},
    )
    monkeypatch.setattr(production, "observed_process", lambda pid: None)
    result = production.startup_observation(root, release, arm, path)
    assert result["status"] == "needs_attention"
    assert result["startup_absence_verified"] is True
    assert read(path) == intent
    assert not production.worker_path(root, arm).exists()
    assert not (root / "runs/wave9").exists()


@pytest.mark.parametrize("launch_status", ["launching", "needs_attention"])
def test_unacknowledged_spawn_remains_unknown_while_starter_generation_is_live(tmp_path, monkeypatch, launch_status):
    release_path, _ = production.paths(tmp_path)
    record(release_path, {"status": "released"})
    arm, release = production.ARMS[0], {"helpers": str(tmp_path / "helpers")}
    path = production.worker_path(tmp_path, arm).with_suffix(".start.json")
    record(
        path,
        {
            "status": "starting",
            "arm": arm,
            "node": production.node_for(arm),
            "command": production.worker_command(tmp_path, release, arm),
            "worker_marker": "pending-wave9-fixture",
            "release_binding_sha256": sha256(release_path),
            "starter_pid": 987654321,
            "starter_start_ticks": "456",
        },
    )
    record(
        tmp_path / "run-preparation/wave9-launch-state.json",
        {"status": launch_status, "worker_markers": {arm: "pending-wave9-fixture"}},
    )
    monkeypatch.setattr(production, "observed_process", lambda pid: ("S", "456", ["startup"]))
    assert production.startup_observation(tmp_path, release, arm, path)["status"] == "unknown"
    monkeypatch.setattr(production, "observed_process", lambda pid: None)
    assert production.startup_observation(tmp_path, release, arm, path)["status"] == "needs_attention"
    assert not production.worker_path(tmp_path, arm).exists()


def test_starter_exit_rereads_late_worker_acknowledgment_before_declaring_absence(tmp_path, monkeypatch):
    release_path, _ = production.paths(tmp_path)
    record(release_path, {"status": "released"})
    arm, release = production.ARMS[0], {"helpers": str(tmp_path / "helpers")}
    path = production.worker_path(tmp_path, arm).with_suffix(".start.json")
    intent = {
        "status": "starting",
        "arm": arm,
        "node": production.node_for(arm),
        "command": production.worker_command(tmp_path, release, arm),
        "worker_marker": "late-wave9-fixture",
        "release_binding_sha256": sha256(release_path),
        "starter_pid": 999999999,
        "starter_start_ticks": "456",
    }
    record(path, intent)
    record(
        tmp_path / "run-preparation/wave9-launch-state.json",
        {"status": "needs_attention", "worker_markers": {arm: intent["worker_marker"]}},
    )

    def observe(pid):
        if pid == intent["starter_pid"]:
            record(path, intent | {"pid": 999999998, "start_ticks": "789"})
            return None
        return "S", "789", intent["command"]

    monkeypatch.setattr(production, "observed_process", observe)
    result = production.startup_observation(tmp_path, release, arm, path)
    assert result["status"] == "unknown"
    assert "startup_absence_verified" not in result


@pytest.mark.parametrize("published", ["helpers", "archive"])
def test_interrupted_freeze_publication_finishes_identical_payload_only(tmp_path, published):
    preparation = tmp_path / "run-preparation"
    staging = preparation / "wave9-freeze-staging-fixture"
    staged_helpers = staging / "tracking"
    staged_helpers.mkdir(parents=True)
    helper = staged_helpers / "controller.py"
    helper.write_text("pass\n")
    staged_archive = staging / "source.tar"
    staged_archive.write_bytes(b"freeze archive")
    release_path, helpers = production.paths(tmp_path)
    archive = preparation / "source-wave9-launch-20261117.tar"
    release = {
        "root": str(tmp_path),
        "helpers": str(helpers),
        "archive": str(archive),
        "helpers_sha256": {"controller.py": sha256(helper)},
        "archive_sha256": sha256(staged_archive),
    }
    record(staging / "release.json", release)
    publication = {
        "status": "prepared_identical_freeze_publication",
        "root": str(tmp_path),
        "staging": str(staging),
        "release": release,
        "release_sha256": sha256(staging / "release.json"),
    }
    if published == "helpers":
        helpers.parent.mkdir(parents=True)
        staged_helpers.rename(helpers)
    else:
        staged_archive.rename(archive)
    production.publish_freeze(tmp_path, publication)
    assert read(release_path) == release
    assert sha256(helpers / "controller.py") == release["helpers_sha256"]["controller.py"]
    assert sha256(archive) == release["archive_sha256"]
    production.publish_freeze(tmp_path, publication)
    (helpers / "controller.py").write_text("changed\n")
    with pytest.raises(ValueError):
        production.publish_freeze(tmp_path, publication)
    assert read(release_path) == release
    assert not (tmp_path / "runs/wave9").exists()
