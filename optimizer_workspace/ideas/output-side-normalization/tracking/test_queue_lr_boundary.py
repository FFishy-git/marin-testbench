# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise one-shot handoff controllers without touching CUDA or active allocations."""

import importlib.util
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest
import queue_lr_boundary
from launch_wave5 import read, sha256
from prepare_lr_ceiling import STUDY as CEILING_STUDY


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def queue(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign/campaign.json"
    families = ["muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused"]
    record(campaign, {"root": str(tmp_path), "source": str(tmp_path), "families": families})
    target = campaign.parent / "queue"
    target.mkdir()
    record(target / "binding.json", {"status": "fixture"})
    record(
        campaign.parent / "execution-binding.json",
        {
            "helpers": str(tmp_path / "execution"),
            "plan": [
                {"family": family, "node": node}
                for family, node in zip(families, ("node0", "node1", "node2", "node3"), strict=True)
            ],
        },
    )
    release_path = tmp_path / "run-preparation/wave8-launch-binding.json"
    record(release_path, {"helpers": str(tmp_path / "predecessor")})
    frozen = {
        "execution_binding_sha256": sha256(campaign.parent / "execution-binding.json"),
        "predecessor_release_sha256": sha256(release_path),
    }
    monkeypatch.setattr(queue_lr_boundary, "binding", lambda campaign: frozen)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: queue_lr_boundary.NODE + "\n")
    observed = {
        "status": "active",
        "all_workers_terminal": False,
        "binding_sha256": frozen["predecessor_release_sha256"],
        "workers": [{"arm": arm, "status": "running"} for arm in queue_lr_boundary.ARMS],
    }
    calls = []
    spawns = []

    def run(command, **kwargs):
        calls.append(command)
        if "--out" in command:
            record(Path(command[command.index("--out") + 1]), observed)
        return subprocess.CompletedProcess(command, 0, "{}", "")

    def spawn(command, **kwargs):
        spawns.append((command, kwargs))
        return type("Child", (), {"pid": 99991})()

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    # Mocked spawn tests need a launch-capable fixture; denial tests override it.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(queue_lr_boundary, "observed_process", lambda pid: ("S", "1234", ["placeholder"]))
    monkeypatch.setattr(queue_lr_boundary, "require_completed", lambda root: {"status": "complete_rederived"})
    return campaign, observed, calls, spawns


def finish_predecessor(campaign, observed):
    observed.update(status="complete", all_workers_terminal=True)
    for worker in observed["workers"]:
        worker["status"] = "complete"
    root = Path(read(campaign)["root"])
    record(root / "run-preparation/swanlab-upload-wave8.json", {"status": "verified"})


def test_ceiling_queue_waits_on_original_wave9_handles_and_cloud_not_completed_wave8(queue, monkeypatch):
    campaign, observed, calls, spawns = queue
    root = Path(read(campaign)["root"])
    record(campaign, {**read(campaign), "study": CEILING_STUDY})
    release = root / "run-preparation/wave9-launch-binding.json"
    record(release, {"helpers": str(root / "wave9-predecessor")})
    frozen = {
        "execution_binding_sha256": sha256(campaign.parent / "execution-binding.json"),
        "predecessor_release_sha256": sha256(release),
    }
    monkeypatch.setattr(queue_lr_boundary, "binding", lambda path: frozen)
    observed["binding_sha256"] = frozen["predecessor_release_sha256"]
    observed["workers"] = [{"arm": arm, "status": "running"} for arm in queue_lr_boundary.handoff_arms(campaign)]
    record(root / "run-preparation/swanlab-upload-wave8.json", {"status": "verified"})
    assert queue_lr_boundary.advance(campaign)["status"] == "waiting_for_wave9"
    assert not spawns
    assert Path(calls[-1][1]).name == "upstream_production.py"
    observed.update(status="complete", all_workers_terminal=True)
    for worker in observed["workers"]:
        worker["status"] = "complete"
    assert queue_lr_boundary.advance(campaign)["status"] == "waiting_for_wave9_upload"
    assert not spawns
    record(root / "run-preparation/swanlab-upload-wave9.json", {"status": "verified"})
    verified = []
    monkeypatch.setattr(
        queue_lr_boundary,
        "require_completed",
        lambda path, *, wave: verified.append((path, wave)) or {"status": "complete_rederived"},
    )
    assert queue_lr_boundary.advance(campaign)["status"] == "validating"
    assert verified == [(root, "wave9")]
    assert len(spawns) == 1
    assert spawns[0][0] == queue_lr_boundary.command(campaign, "validate")
    assert Path(calls[-1][1]).name == "analyze_upstream_run.py"


def acknowledged_launch(campaign):
    execution = read(campaign.parent / "execution-binding.json")
    digest = sha256(campaign.parent / "execution-binding.json")
    workers = {
        assignment["family"]: {
            "status": "running",
            "family": assignment["family"],
            "node": assignment["node"],
            "binding_sha256": digest,
            "worker_marker": f"marker-{index}",
            "pid": index + 1000,
            "start_ticks": str(index + 5000),
            "command": [
                queue_lr_boundary.PYTHON,
                str(Path(execution["helpers"]) / "run_fixed_age_tuning.py"),
                "--campaign",
                str(campaign),
                "--phase",
                "worker",
                "--family",
                assignment["family"],
            ],
        }
        for index, assignment in enumerate(execution["plan"])
    }
    record(campaign.parent / "monitor/pid", 8888)
    return {
        "status": "launched",
        "binding_sha256": digest,
        "monitor_pid": 8888,
        "worker_markers": {family: worker["worker_marker"] for family, worker in workers.items()},
        "workers": workers,
    }


def test_active_wave_waits_without_controller_spawn_or_allocation_change(queue):
    campaign, _, _, spawns = queue
    assert queue_lr_boundary.advance(campaign)["status"] == "waiting_for_wave8"
    assert not spawns
    assert not (campaign.parent / "validation-state.json").exists()
    assert not (campaign.parent / "queue/validate.json").exists()


def test_unknown_observation_waits_without_restarting_predecessor(queue, monkeypatch):
    campaign, _, _, spawns = queue

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 600)

    monkeypatch.setattr(subprocess, "run", timeout)
    state = queue_lr_boundary.advance(campaign)
    assert state["status"] == "waiting_for_wave8"
    assert "TimeoutExpired" in state["observation_error"]
    assert not spawns


def test_completed_wave_waits_for_upload_then_spawns_exact_validator_only_once(queue, monkeypatch):
    campaign, observed, calls, spawns = queue
    finish_predecessor(campaign, observed)
    cloud = Path(read(campaign)["root"]) / "run-preparation/swanlab-upload-wave8.json"
    cloud.unlink()
    assert queue_lr_boundary.advance(campaign)["status"] == "waiting_for_wave8_upload"
    assert not spawns
    record(cloud, {"status": "verified"})
    assert queue_lr_boundary.advance(campaign)["status"] == "validating"
    assert len(spawns) == 1
    assert spawns[0][0] == queue_lr_boundary.command(campaign, "validate")
    assert spawns[0][1]["start_new_session"] is True
    assert queue_lr_boundary.MARKER in spawns[0][1]["env"]
    intent = read(campaign.parent / "queue/validate.json")
    assert intent["pid"] == 99991 and intent["start_ticks"] == "1234"
    monkeypatch.setattr(queue_lr_boundary, "validation_ready", lambda campaign: False)
    assert queue_lr_boundary.advance(campaign)["status"] == "validating"
    assert len(spawns) == 1
    assert any(Path(command[1]).name == "analyze_wave8.py" for command in calls)


@pytest.mark.parametrize("fault", ["predecessor_failed", "binding", "existing_validation"])
def test_bad_predecessor_or_unowned_execution_evidence_never_spawns_gpu_controller(queue, fault):
    campaign, observed, _, spawns = queue
    finish_predecessor(campaign, observed)
    if fault == "predecessor_failed":
        observed["status"] = "needs_attention"
    elif fault == "binding":
        observed["binding_sha256"] = "different"
    else:
        record(campaign.parent / "validation-state.json", {"status": "external"})
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        queue_lr_boundary.advance(campaign)
    assert read(campaign.parent / "queue/state.json")["status"] == "needs_attention"
    assert not spawns


def test_uncertain_spawn_preserves_intent_and_cannot_retry(queue, monkeypatch):
    campaign, observed, _, spawns = queue
    finish_predecessor(campaign, observed)
    monkeypatch.setattr(queue_lr_boundary, "observed_process", lambda pid: None)
    assert queue_lr_boundary.advance(campaign)["status"] == "validating"
    assert queue_lr_boundary.observe_controller(campaign, "validate") == "unknown"
    assert queue_lr_boundary.advance(campaign)["status"] == "validating"
    assert len(spawns) == 1


def test_spawning_exception_keeps_intent_and_needs_attention(queue, monkeypatch):
    campaign, observed, _, _ = queue
    finish_predecessor(campaign, observed)

    def fail(*args, **kwargs):
        raise OSError("spawn failure")

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(OSError):
        queue_lr_boundary.advance(campaign)
    assert (campaign.parent / "queue/validate.json").exists()
    assert read(campaign.parent / "queue/state.json")["status"] == "needs_attention"
    assert queue_lr_boundary.advance(campaign)["status"] == "needs_attention"


def test_passed_actual_validation_launches_once_then_requires_four_acknowledgments(queue, monkeypatch):
    campaign, observed, _, spawns = queue
    finish_predecessor(campaign, observed)
    queue_lr_boundary.advance(campaign)
    monkeypatch.setattr(queue_lr_boundary, "validation_ready", lambda campaign: True)
    assert queue_lr_boundary.advance(campaign)["status"] == "launching"
    assert len(spawns) == 2
    assert spawns[-1][0] == queue_lr_boundary.command(campaign, "launch")
    monkeypatch.setattr(queue_lr_boundary, "observe_controller", lambda *args: "running")
    assert queue_lr_boundary.advance(campaign)["status"] == "launching"
    assert len(spawns) == 2
    record(
        campaign.parent / "launch-state.json",
        acknowledged_launch(campaign),
    )
    monkeypatch.setattr(queue_lr_boundary, "observe_controller", lambda *args: "exited")
    assert queue_lr_boundary.advance(campaign)["status"] == "launched"
    assert len(spawns) == 2


@pytest.mark.parametrize("fault", ["missing", "partial", "binding"])
def test_exited_partial_launcher_is_not_retried_or_treated_as_complete(queue, monkeypatch, fault):
    campaign, observed, _, spawns = queue
    finish_predecessor(campaign, observed)
    queue_lr_boundary.advance(campaign)
    monkeypatch.setattr(queue_lr_boundary, "validation_ready", lambda campaign: True)
    queue_lr_boundary.advance(campaign)
    monkeypatch.setattr(queue_lr_boundary, "observe_controller", lambda *args: "exited")
    if fault != "missing":
        record(
            campaign.parent / "launch-state.json",
            {
                "status": "needs_attention" if fault == "partial" else "launched",
                "binding_sha256": (
                    "different" if fault == "binding" else sha256(campaign.parent / "execution-binding.json")
                ),
                "workers": {},
            },
        )
    with pytest.raises((RuntimeError, FileNotFoundError)):
        queue_lr_boundary.advance(campaign)
    assert read(campaign.parent / "queue/state.json")["status"] == "needs_attention"
    assert len(spawns) == 2


def test_reused_controller_pid_stops_handoff(queue):
    campaign, observed, _, spawns = queue
    finish_predecessor(campaign, observed)
    queue_lr_boundary.advance(campaign)
    with pytest.raises(RuntimeError, match="reused or argv changed"):
        queue_lr_boundary.advance(campaign)
    assert len(spawns) == 1


@pytest.mark.parametrize("fault", ["generation", "restore", "boolean_restore", "cleanup", "exit", "state"])
def test_validation_ready_refuses_incomplete_or_changed_gate_owner(queue, monkeypatch, fault):
    campaign, _, _, _ = queue
    record(campaign.parent / "queue/validate.json", {"pid": 99991, "start_ticks": "1234"})
    state = {
        "controller_pid": 99991,
        "controller_start_ticks": "1234",
        "status": "passed",
        "keeper_restore": 0,
        "cleanup_exit_code": 0,
        "exit_code": 0,
    }
    if fault == "generation":
        state["controller_start_ticks"] = "changed"
    elif fault == "restore":
        state["keeper_restore"] = 1
    elif fault == "boolean_restore":
        state["keeper_restore"] = False
    elif fault == "cleanup":
        state["cleanup_error"] = "live children"
    elif fault == "exit":
        state["exit_code"] = 1
    else:
        state["status"] = "needs_attention"
    record(campaign.parent / "validation-state.json", state)
    monkeypatch.setattr(queue_lr_boundary, "observe_controller", lambda *args: "exited")
    with pytest.raises(RuntimeError, match="did not pass"):
        queue_lr_boundary.validation_ready(campaign)


def test_validation_ready_rederives_actual_gates_after_controller_exit(queue, monkeypatch):
    campaign, _, calls, _ = queue
    record(campaign.parent / "queue/validate.json", {"pid": 99991, "start_ticks": "1234"})
    record(
        campaign.parent / "validation-state.json",
        {
            "controller_pid": 99991,
            "controller_start_ticks": "1234",
            "status": "passed",
            "keeper_restore": 0,
            "cleanup_exit_code": 0,
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(queue_lr_boundary, "observe_controller", lambda *args: "exited")
    assert queue_lr_boundary.validation_ready(campaign)
    assert "require_ready(campaign)" in calls[-1][2]
    assert "tagged_processes" in calls[-1][2]


def test_controller_observer_checks_current_generation_argv_and_marker(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign.json"
    intent = {
        "pid": os.getpid(),
        "start_ticks": "1234",
        "command": ["python", "validator"],
        "marker": "intentionally_not_in_environment",
    }
    record(tmp_path / "queue/validate.json", intent)
    monkeypatch.setattr(queue_lr_boundary, "observed_process", lambda pid: ("S", "1234", intent["command"]))
    with pytest.raises(RuntimeError, match="marker changed"):
        queue_lr_boundary.observe_controller(campaign, "validate")


@pytest.mark.parametrize("fault", ["marker", "generation", "node", "monitor"])
def test_exited_launcher_with_changed_acknowledgments_preserves_workers_without_retry(queue, monkeypatch, fault):
    campaign, observed, _, spawns = queue
    finish_predecessor(campaign, observed)
    queue_lr_boundary.advance(campaign)
    monkeypatch.setattr(queue_lr_boundary, "validation_ready", lambda campaign: True)
    queue_lr_boundary.advance(campaign)
    launch = acknowledged_launch(campaign)
    worker = launch["workers"]["muont"]
    if fault == "marker":
        worker["worker_marker"] = "different-generation"
    elif fault == "generation":
        worker["start_ticks"] = None
    elif fault == "node":
        worker["node"] = "different-node"
    else:
        record(campaign.parent / "monitor/pid", 7777)
    record(campaign.parent / "launch-state.json", launch)
    monkeypatch.setattr(queue_lr_boundary, "observe_controller", lambda *args: "exited")
    with pytest.raises(RuntimeError):
        queue_lr_boundary.advance(campaign)
    assert read(campaign.parent / "queue/state.json")["status"] == "needs_attention"
    assert read(campaign.parent / "launch-state.json") == launch
    assert len(spawns) == 2


def test_unowned_launch_directory_prevents_validator_or_launcher_spawn(queue, monkeypatch):
    campaign, observed, _, spawns = queue
    finish_predecessor(campaign, observed)
    (campaign.parent / "workers").mkdir()
    with pytest.raises(FileExistsError):
        queue_lr_boundary.advance(campaign)
    assert not spawns
    assert not (campaign.parent / "queue/validate.json").exists()


def test_reused_zombie_generation_is_not_a_controller_exit(queue, monkeypatch):
    campaign, _, _, _ = queue
    record(
        campaign.parent / "queue/validate.json",
        {"pid": 99991, "start_ticks": "1234", "command": ["python"], "marker": "marker"},
    )
    monkeypatch.setattr(queue_lr_boundary, "observed_process", lambda pid: ("Z", "different", []))
    with pytest.raises(RuntimeError):
        queue_lr_boundary.observe_controller(campaign, "validate")
    assert not (campaign.parent / "queue/launch.json").exists()


def test_controller_observer_accepts_real_marked_process_and_detects_its_exit(tmp_path):
    campaign = tmp_path / "campaign.json"
    command = [
        queue_lr_boundary.PYTHON,
        "-c",
        "import sys; print('ready', flush=True); sys.stdin.read()",
    ]
    with subprocess.Popen(
        command,
        env={**os.environ, queue_lr_boundary.MARKER: "test-owner"},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            assert child.stdout.readline().strip() == "ready"
            observed = queue_lr_boundary.observed_process(child.pid)
            record(
                tmp_path / "queue/validate.json",
                {"pid": child.pid, "start_ticks": observed[1], "command": command, "marker": "test-owner"},
            )
            assert queue_lr_boundary.observe_controller(campaign, "validate") == "running"
        finally:
            child.communicate(input="", timeout=10)
    assert queue_lr_boundary.observe_controller(campaign, "validate") == "exited"


@pytest.mark.parametrize("fault", [None, "helper", "archive", "campaign"])
@pytest.mark.parametrize("wave,study", [("wave8", queue_lr_boundary.STUDY), ("wave9", CEILING_STUDY)])
def test_real_freeze_roundtrip_binds_source_evidence_and_rejects_later_changes(
    tmp_path, monkeypatch, fault, wave, study
):
    campaign = tmp_path / "campaign/campaign.json"
    record(campaign, {"study": study, "root": str(tmp_path)})
    predecessor = tmp_path / f"run-preparation/{wave}-launch-binding.json"
    record(predecessor, {"status": "fixture"})
    for kind, module in (("validation", "run_fixed_age_validation.py"), ("execution", "run_fixed_age_tuning.py")):
        helpers = campaign.parent / f"{kind}-helpers"
        helpers.mkdir()
        (helpers / module).write_text("fixture_source = True\n")
        archive = campaign.parent / f"{kind}-source.tar"
        with tarfile.open(archive, "x") as stream:
            stream.add(helpers / module, arcname=f"tracking/{module}")
        release = {
            "campaign_sha256": sha256(campaign),
            "predecessor_release_sha256": sha256(predecessor),
            "helpers": str(helpers),
            "helpers_sha256": {module: sha256(helpers / module)},
            "archive": str(archive),
            "archive_sha256": sha256(archive),
        }
        if kind == "execution":
            release["validation_binding_sha256"] = sha256(campaign.parent / "validation-binding.json")
        record(campaign.parent / f"{kind}-binding.json", release)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "{}", ""))
    frozen = queue_lr_boundary.freeze(campaign)
    spec = importlib.util.spec_from_file_location(
        "frozen_boundary_queue_test", Path(frozen["helpers"]) / "queue_lr_boundary.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.binding(campaign)["status"] == f"frozen_waiting_for_{wave}"
    with tarfile.open(frozen["archive"]) as stream:
        for member in stream.getmembers():
            assert stream.extractfile(member).read() == (Path(frozen["helpers"]) / Path(member.name).name).read_bytes()
    with pytest.raises(FileExistsError):
        queue_lr_boundary.freeze(campaign)
    if fault:
        changed = (
            Path(frozen["helpers"]) / "queue_lr_boundary.sh"
            if fault == "helper"
            else Path(frozen["archive"]) if fault == "archive" else campaign
        )
        with changed.open("ab") as stream:
            stream.write(b"\n")
        with pytest.raises(ValueError):
            module.binding(campaign)
    assert not (campaign.parent / "queue/validate.json").exists()
    assert not (campaign.parent / "validation-state.json").exists()


def test_failed_queue_installation_preserves_intent_and_cannot_spawn_twice(queue, monkeypatch):
    campaign, _, _, spawns = queue
    monkeypatch.setattr(queue_lr_boundary, "binding", lambda path: {"helpers": str(campaign.parent / "queue/helpers")})

    def fail(*args, **kwargs):
        spawns.append(args)
        raise OSError("queue spawn failed")

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(OSError):
        queue_lr_boundary.install(campaign)
    intent = read(campaign.parent / "queue/installation.json")
    assert intent["status"] == "needs_attention"
    assert intent["binding_sha256"] == sha256(campaign.parent / "queue/binding.json")
    with pytest.raises(FileExistsError):
        queue_lr_boundary.install(campaign)
    assert len(spawns) == 1
    assert not (campaign.parent / "queue/validate.json").exists()


@pytest.mark.parametrize("phase", ["install", "validate", "launch"])
def test_cpu_hidden_queue_refuses_before_persisting_intent_or_spawning(queue, monkeypatch, phase):
    # The first ceiling campaign inherited the CPU-test mask and released a
    # keeper before all ranks failed CUDA initialization without training.
    campaign, _, _, spawns = queue
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    before = set((campaign.parent / "queue").iterdir())
    with pytest.raises(RuntimeError, match="CPU-only CUDA visibility"):
        if phase == "install":
            queue_lr_boundary.install(campaign)
        else:
            queue_lr_boundary.spawn(campaign, {}, phase)
    assert set((campaign.parent / "queue").iterdir()) == before
    assert not spawns
    assert not (campaign.parent / "validation-state.json").exists()
