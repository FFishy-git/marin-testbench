# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check Wave-8 production ownership, failure-preserving snapshots and release identities offline."""

import copy
import json
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import pytest
import wave8_history
import wave8_release
from analyze_tuning import executed_sources
from compare_runs import WINDOW
from launch_wave5 import NODES, PYTHON, sha256
from validate_wave5 import ENVIRONMENT
from wave7_release import candidate_inventory


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def release_fixture(tmp_path, monkeypatch):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    candidate = tmp_path / "candidate"
    (candidate / "experiment").mkdir(parents=True)
    (candidate / "experiment/train.py").write_text("# Frozen training fixture\n")
    base = {
        "experiment": {"arm": "muont"},
        "training": {"steps": 5120, "tokens": 2684354560, "batch_tokens": 524288, "seed": 0},
        "data": {"training": "same"},
    }
    for arm in wave8_history.ARMS:
        config = copy.deepcopy(base)
        config["experiment"]["arm"] = arm
        record(candidate / f"experiment/configs/{arm}_0p5m_seed0.json", config)
    pinned = candidate / "artifacts/pinned-sources-materialized/Toner-fixture"
    module = pinned / "reproduce/marin/model.py"
    module.parent.mkdir(parents=True)
    module.write_text("# Frozen upstream fixture\n")
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    record(
        candidate / "experiment/upstream_manifest.json",
        {
            "revision": "fixture",
            "files": {"reproduce/marin/model.py": sha256(module)},
        },
    )
    record(tmp_path / "data/experiment_receipt.json", {"status": "verified"})
    archive = preparation / "candidate.tar"
    archive.write_bytes(b"candidate")
    old_helpers = tmp_path / "controllers/wave8-20261117/tracking"
    old_helpers.mkdir(parents=True)
    (old_helpers / "old.py").write_text("# Frozen old helper\n")
    candidate_path = preparation / "wave8-candidate.json"
    numerical = {
        "candidate": str(candidate),
        "helpers": str(old_helpers),
        "helpers_sha256": {"old.py": sha256(old_helpers / "old.py")},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "files_sha256": candidate_inventory(candidate),
        "source_sha256": {"experiment/train.py": sha256(candidate / "experiment/train.py")},
        "data_receipt_sha256": sha256(tmp_path / "data/experiment_receipt.json"),
        "environment": ENVIRONMENT,
        "required_gates": ["gpu"],
    }
    record(candidate_path, numerical)
    log = preparation / "gpu.log"
    log.write_text("successful GPU fixture\n")
    state_path = preparation / "wave8-validation-state.json"
    validation_path = preparation / "validation-wave8-gpu.json"
    record(
        validation_path,
        {
            "source_sha256": numerical["source_sha256"],
            "data_receipt_sha256": numerical["data_receipt_sha256"],
            "environment": ENVIRONMENT,
            "checks": {"gpu": True},
            "owner": {"controller_pid": 12, "controller_start_ticks": "34"},
            "evidence": {"gpu_commands": [{"exit_code": 0, "log": str(log), "log_sha256": sha256(log)}] * 13},
        },
    )
    record(
        state_path,
        {
            "status": "passed",
            "controller_pid": 12,
            "controller_start_ticks": "34",
            "keeper_restore": 0,
            "cleanup_exit_code": 0,
            "validation_sha256": sha256(validation_path),
            "binding_sha256": sha256(candidate_path),
        },
    )
    release_path, helpers = wave8_history.paths(tmp_path)
    helpers.mkdir(parents=True)
    for name in ("old.py", *wave8_history.EXTRA_HELPERS):
        (helpers / name).write_text("# Bound launch helper\n" if name != "old.py" else "# Frozen old helper\n")
    launch_archive = preparation / "source-wave8-launch-20261117.tar"
    launch_archive.write_bytes(b"launch")
    release = {
        "status": "frozen_validated_launch_release",
        "root": str(tmp_path),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "helpers_sha256": {name: sha256(helpers / name) for name in ("old.py", *wave8_history.EXTRA_HELPERS)},
        "archive": str(launch_archive),
        "archive_sha256": sha256(launch_archive),
        "candidate_binding_sha256": sha256(candidate_path),
        "validation_sha256": sha256(validation_path),
        "validation_state_sha256": sha256(state_path),
        "data_receipt_sha256": numerical["data_receipt_sha256"],
        "environment": ENVIRONMENT,
        "arms": wave8_history.ARMS,
        "nodes": NODES,
        "commands": wave8_history.commands(tmp_path, candidate),
        "monitor_interval_seconds": 900,
        "training_timeout_seconds": 21600,
    }
    record(release_path, release)
    monkeypatch.setattr(wave8_history, "__file__", str(helpers / "wave8_history.py"))
    return tmp_path, candidate, release


def test_launch_binding_requires_exact_numerical_and_helper_inventory(release_fixture):
    root, _, release = release_fixture
    assert wave8_history.binding(root)["commands"] == release["commands"]


@pytest.mark.parametrize("fault", ["helper", "archive", "data", "state", "command", "extra_helper"])
def test_launch_binding_refuses_changed_source_or_evidence(release_fixture, fault):
    root, _, release = release_fixture
    path, helpers = wave8_history.paths(root)
    if fault == "helper":
        (helpers / "wave8_release.py").write_text("# Changed\n")
    elif fault == "archive":
        Path(release["archive"]).write_bytes(b"changed")
    elif fault == "data":
        (root / "data/experiment_receipt.json").write_text('{"replacement":true}')
    elif fault == "state":
        state_path = root / "run-preparation/wave8-validation-state.json"
        state = json.loads(state_path.read_text())
        state["keeper_restore"] = False
        record(state_path, state)
        release["validation_state_sha256"] = sha256(state_path)
        record(path, release)
    elif fault == "command":
        release["commands"][NODES[0]].extend(["--max-steps", "24"])
        record(path, release)
    else:
        (helpers / "unexpected.py").write_text("# Unexpected\n")
        release["helpers_sha256"]["unexpected.py"] = sha256(helpers / "unexpected.py")
        record(path, release)
    with pytest.raises(ValueError):
        wave8_history.binding(root)


def test_helper_only_observation_remains_available_after_data_identity_changes(release_fixture):
    root, _, _ = release_fixture
    (root / "data/experiment_receipt.json").write_text('{"replacement":true}')
    assert wave8_history.helper_binding(root)["status"] == "frozen_validated_launch_release"
    with pytest.raises(ValueError):
        wave8_history.binding(root)


def test_worker_restores_only_after_verified_cleanup(tmp_path, monkeypatch):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    record(preparation / "wave8-launch-binding.json", {"status": "bound"})
    release = {}
    monkeypatch.setenv(wave8_release.WORKER_MARKER, "worker")
    monkeypatch.setattr(wave8_release, "binding", lambda root: release)
    monkeypatch.setattr(wave8_release, "started_worker", lambda *args: ("S", "12", ["worker"]))
    monkeypatch.setattr(wave8_release, "node_guard", lambda *args: nullcontext([]))
    monkeypatch.setattr(wave8_release, "allocation", lambda *args: None)
    monkeypatch.setattr(wave8_release, "release_keeper", lambda state, path: state.update(keeper_release_attempted=True))
    restored = []
    monkeypatch.setattr(wave8_release, "node_action", lambda root, action: restored.append(action))

    def failed_child(root, frozen, state, path):
        state["cleanup_error"] = "live marked child"
        raise RuntimeError("child cleanup failed")

    monkeypatch.setattr(wave8_release, "train_child", failed_child)
    with pytest.raises(RuntimeError, match="child cleanup failed"):
        wave8_release.run_arm(tmp_path, "muont")
    assert restored == []
    state = json.loads(wave8_release.worker_path(tmp_path, "muont").read_text())
    assert state["status"] == "needs_attention"
    assert state["keeper_restore"].startswith("refused:")


def test_completed_history_requires_all_evaluations_and_rank_checkpoints(release_fixture):
    root, candidate, release = release_fixture
    arm = wave8_history.ARMS[0]
    directory = root / "runs/wave8" / arm
    config = json.loads((candidate / f"experiment/configs/{arm}_0p5m_seed0.json").read_text())
    numerical = json.loads((root / "run-preparation/wave8-candidate.json").read_text())
    sources = executed_sources(candidate, numerical)
    manifest = {
        "config": config,
        "data_receipt_sha256": release["data_receipt_sha256"],
        "source_sha256": sources,
        "status": "complete",
        "completed_steps": 5120,
    }
    record(directory / "run.json", manifest)
    eval_steps = {1, *range(256, 5121, 256), 5120}
    eval_steps.add(22)
    rows = []
    for step in range(1, 5121):
        row = {"step": step, "tokens": step * 524288, "train_loss": 4.0}
        if step in eval_steps:
            row.update(c4_loss=4.1, selection_loss=4.0, heldout_losses={"heldout/dclm": 4.0})
        rows.append(row)
    (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    checkpoint = directory / "checkpoints/step_005120"
    record(checkpoint / "complete.json", {"step": 5120, "world": 8})
    for rank in range(8):
        (checkpoint / f"rank_{rank}.pt").write_bytes(b"nonempty fixture")
    assert len(eval_steps) == 22
    assert wave8_history.completed_run(root, arm, release)["complete"]
    (checkpoint / "rank_7.pt").unlink()
    with pytest.raises(FileNotFoundError):
        wave8_history.completed_run(root, arm, release)


def test_preflight_all_nodes_precedes_start_intents(tmp_path, monkeypatch):
    (tmp_path / "run-preparation").mkdir()
    release = {"helpers": str(tmp_path), "commands": {}}
    monkeypatch.setattr(wave8_release, "binding", lambda root: release)
    monkeypatch.setattr(wave8_release, "require_lr_complete", lambda root: None)
    monkeypatch.setattr(wave8_release, "require_numerical", lambda root: {})
    seen = []

    def inspect(root, frozen, arm, phase):
        seen.append((arm, phase))
        if arm == wave8_history.ARMS[-1]:
            raise RuntimeError("occupied final node")
        return {"status": "allocation_preflight_passed"}

    monkeypatch.setattr(wave8_release, "remote", inspect)
    with pytest.raises(RuntimeError, match="occupied"):
        wave8_release.launch(tmp_path)
    assert [phase for _, phase in seen] == ["inspect"] * 4
    assert not (tmp_path / "run-preparation/wave8-launch-state.json").exists()
    assert not (tmp_path / "runs/wave8").exists()


def test_training_marker_and_generation_are_persisted_before_owned_child_runs(tmp_path, monkeypatch):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    (tmp_path / "runs/wave8").mkdir(parents=True)
    state_path = preparation / "worker.json"
    script = """
import json, os, sys, time
from pathlib import Path
p=Path(sys.argv[1])
deadline=time.monotonic()+5
while time.monotonic()<deadline:
    state=json.loads(p.read_text())
    if state.get("child_start_ticks"):
        assert state["child_pid"]==os.getpid()
        assert state["child_marker"]==os.environ["OUTPUT_MARIN_DIAGNOSTIC_ID"]
        print("owned child acknowledged", flush=True)
        break
    time.sleep(.01)
else:
    raise RuntimeError("missing acknowledgment")
"""
    command = [PYTHON, "-c", script, str(state_path)]
    release = {"commands": {NODES[0]: command}, "candidate": str(tmp_path), "training_timeout_seconds": 10}
    state = {"arm": "muont", "node": NODES[0]}
    original = wave8_release.subprocess.Popen

    def spawn(*args, **kwargs):
        before = json.loads(state_path.read_text())
        assert before["child_marker"] == kwargs["env"]["OUTPUT_MARIN_DIAGNOSTIC_ID"]
        assert before["status"] == "starting"
        return original(*args, **kwargs)

    monkeypatch.setattr(wave8_release.subprocess, "Popen", spawn)
    assert wave8_release.train_child(tmp_path, release, state, state_path) == 0
    assert state["cleanup_exit_code"] == 0
    assert state["child_start_ticks"]
    assert "owned child acknowledged" in Path(state["log"]).read_text()


def test_snapshot_preserves_launch_inventory_and_unknown_token_charges(tmp_path, monkeypatch):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    source = tmp_path / "candidate"
    release = {"commands": wave8_history.commands(tmp_path, source)}
    monkeypatch.setattr(wave8_release, "helper_binding", lambda root: release)
    record(preparation / "wave8-launch-binding.json", {"status": "bound"})
    record(preparation / "wave8-launch-state.json", {"status": "launched"})
    arm = wave8_history.ARMS[0]
    entry = {
        "arm": arm,
        "node": NODES[0],
        "out": str(tmp_path / "runs/wave8" / arm),
        "pid": 123,
        "command": release["commands"][NODES[0]],
    }
    record(preparation / "wave8.json", {"expected_launches": 4, "launches": [entry]})

    def unavailable(*args):
        raise subprocess.TimeoutExpired("ssh", 30)

    monkeypatch.setattr(wave8_release, "remote", unavailable)
    result = wave8_release.snapshot(tmp_path)
    assert result["status"] == "active"
    assert result["all_workers_terminal"] is False
    assert all(run["observed_tokens"] is None for run in result["runs"])
    assert json.loads((preparation / "wave8.json").read_text())["launches"] == [entry]


def test_dead_worker_with_live_marked_children_is_not_terminal(tmp_path, monkeypatch):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    release_path, _ = wave8_history.paths(tmp_path)
    record(release_path, {"status": "frozen"})
    release = {}
    arm = wave8_history.ARMS[0]
    record(preparation / "wave8-launch-state.json", {"worker_markers": {arm: "worker"}})
    record(
        wave8_release.worker_path(tmp_path, arm),
        {
            "arm": arm,
            "node": NODES[0],
            "release_binding_sha256": sha256(release_path),
            "worker_marker": "worker",
            "worker_pid": 123,
            "worker_start_ticks": "45",
            "status": "complete",
            "child_marker": "child",
            "keeper_restore": 0,
        },
    )
    monkeypatch.setattr(wave8_release, "helper_binding", lambda root: release)
    monkeypatch.setattr(wave8_release, "local_node", lambda: NODES[0])
    monkeypatch.setattr(wave8_release, "observed_process", lambda pid: None)
    monkeypatch.setattr(wave8_release, "tagged_processes", lambda marker: {567: "89"})
    assert wave8_release.probe_arm(tmp_path, arm)["status"] == "live_orphaned_training"


def test_reused_worker_pid_preserves_live_child_receipt_as_nonterminal(tmp_path, monkeypatch):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    release_path, _ = wave8_history.paths(tmp_path)
    record(release_path, {"status": "frozen"})
    arm = wave8_history.ARMS[0]
    record(preparation / "wave8-launch-state.json", {"worker_markers": {arm: "worker"}})
    state = {
        "arm": arm,
        "node": NODES[0],
        "release_binding_sha256": sha256(release_path),
        "worker_marker": "worker",
        "worker_pid": 123,
        "worker_start_ticks": "old",
        "status": "running",
        "child_marker": "child",
    }
    record(wave8_release.worker_path(tmp_path, arm), state)
    monkeypatch.setattr(wave8_release, "helper_binding", lambda root: {})
    monkeypatch.setattr(wave8_release, "local_node", lambda: NODES[0])
    monkeypatch.setattr(wave8_release, "observed_process", lambda pid: ("S", "new", ["unrelated"]))
    monkeypatch.setattr(wave8_release, "tagged_processes", lambda marker: {567: "89"})
    result = wave8_release.probe_arm(tmp_path, arm)
    assert result["status"] == "live_orphaned_training"
    assert result["receipt"] == state
    assert result["descendants"] == {567: "89"}
    monkeypatch.setattr(wave8_release, "tagged_processes", lambda marker: {})
    assert wave8_release.probe_arm(tmp_path, arm)["status"] == "worker_generation_changed"


def test_worker_marker_discovery_uses_worker_environment_key():
    marker = f"wave8-worker-test-{os.getpid()}-{time.monotonic_ns()}"
    child = subprocess.Popen(
        [PYTHON, "-c", "import time; time.sleep(10)"],
        env={**os.environ, wave8_release.WORKER_MARKER: marker},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert child.pid in wave8_release.marked_workers(marker)
        assert child.pid not in wave8_release.tagged_processes(marker)
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_analysis_protocol_keeps_large_windows_and_combined_control_pair():
    source = Path(wave8_release.__file__).with_name("analyze_wave8.py").read_text()
    assert "WINDOW" in source
    assert WINDOW == 536870912
    assert '("outer_leftdiag_fused", ("outer_leftdiag_combined_fused",))' in source
