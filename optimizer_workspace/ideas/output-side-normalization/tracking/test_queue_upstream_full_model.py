# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise one-shot full-model handoff through persisted ownership and result boundaries."""

import importlib.util
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest
import queue_upstream_full_model as queue
from launch_wave5 import read, sha256
from update_endeavor import upstream_full_model_queue_lines


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def bound_queue(tmp_path):
    root = tmp_path
    binding_path, helpers, _ = queue.paths(root)
    helpers.mkdir(parents=True)
    controller = helpers / "run_upstream_full_model.py"
    controller.write_text("import json\nprint(json.dumps({'status':'fixture_check'}))\n")
    probe = root / "run-preparation/probe"
    probe_helpers = probe / "queue/helpers"
    probe_helpers.mkdir(parents=True)
    probe_controller = probe_helpers / "queue_upstream_outer_gpu.py"
    probe_controller.write_text("import json\nprint(json.dumps({'status':'fixture_check'}))\n")
    record(probe / "binding.json", {"cpu_receipt": str(root / "cpu/receipt.json")})
    record(
        probe / "queue/binding.json",
        {
            "helpers": str(probe_helpers),
            "helpers_sha256": {probe_controller.name: sha256(probe_controller)},
            "cpu_receipt": str(root / "cpu/receipt.json"),
        },
    )
    validation = {
        "probe": str(probe),
        "helpers": str(helpers),
        "helpers_sha256": {controller.name: sha256(controller)},
    }
    record(binding_path, validation)
    saved = queue.freeze(root)
    spec = importlib.util.spec_from_file_location(
        "frozen_full_model_queue_test", Path(saved["helpers"]) / "queue_upstream_full_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return root, module, saved


def test_queue_freeze_preserves_validator_and_probe_then_rejects_modified_release(bound_queue):
    root, module, saved = bound_queue
    original = sha256(queue.paths(root)[0])
    assert module.binding(root)["command"] == queue.command(root)
    with tarfile.open(saved["archive"]) as bundle:
        for member in bundle.getmembers():
            assert bundle.extractfile(member).read() == (Path(saved["helpers"]) / Path(member.name).name).read_bytes()
    with pytest.raises(FileExistsError):
        module.freeze(root)
    assert sha256(queue.paths(root)[0]) == original
    assert not (root / "run-preparation/upstream-full-model-corrected-state.json").exists()
    (Path(saved["helpers"]) / "queue_upstream_full_model.sh").write_text("changed\n")
    with pytest.raises(ValueError, match="identity changed"):
        module.binding(root)


@pytest.fixture
def handoff(bound_queue, monkeypatch):
    root, module, frozen = bound_queue
    probe = Path(frozen["probe"])
    original = read(probe / "queue/binding.json")
    installation = {
        "status": "acknowledged",
        "binding_sha256": frozen["probe_queue_binding_sha256"],
        "pid": 123456,
        "start_ticks": "1234",
        "marker": "probe-owner",
        "command": [
            "bash",
            str(Path(original["helpers"]) / "queue_upstream_outer_gpu.sh"),
            str(root),
            original["cpu_receipt"],
            str(probe),
        ],
    }
    record(probe / "queue/installation.json", installation)
    record(
        probe / "queue/state.json",
        {"status": "waiting_for_upper_lr", "binding_sha256": frozen["probe_queue_binding_sha256"]},
    )
    live = {"status": "running"}
    launches = []
    reconstruction = {"error": None}
    original_run = subprocess.run

    def run(argv, **kwargs):
        if "-c" not in argv:
            return original_run(argv, **kwargs)
        if reconstruction["error"] is not None:
            raise reconstruction["error"]
        return subprocess.CompletedProcess(argv, 0, json.dumps({"rederived": {"source": "fixture"}}), "")

    def popen(argv, **kwargs):
        # Source checks are real subprocesses; only controller launch is faked.
        if argv == frozen["command"]:
            launches.append({"argv": argv, "kwargs": kwargs})
            return type("Child", (), {"pid": 987654})()
        return original_popen(argv, **kwargs)

    def observe(intent, marker):
        return live["status"] if marker == "OUTPUT_MARIN_UPSTREAM_PROBE_QUEUE" else "unknown"

    original_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: module.NODE + "\n")
    monkeypatch.setattr(module, "observe", observe)
    monkeypatch.setattr(module, "observed_process", lambda pid: None)
    return root, module, frozen, probe, live, launches, reconstruction


def test_live_original_probe_queue_blocks_allocation_handoff_even_when_result_is_saved(handoff):
    root, module, frozen, probe, _, launches, _ = handoff
    record(probe / "queue/state.json", {"status": "measured", "binding_sha256": frozen["probe_queue_binding_sha256"]})
    assert module.advance(root)["status"] == "waiting_for_probe"
    assert not launches
    assert not (queue.paths(root)[2] / "controller.json").exists()
    assert not (root / "run-preparation/upstream-full-model-corrected-state.json").exists()


@pytest.mark.parametrize("fault", ["failed", "unmeasured_exit", "changed_installation"])
def test_failed_or_misowned_probe_queue_stops_without_full_model_launch(handoff, fault):
    root, module, frozen, probe, live, launches, _ = handoff
    if fault == "changed_installation":
        installation = read(probe / "queue/installation.json")
        installation["command"] = ["unowned"]
        record(probe / "queue/installation.json", installation)
    else:
        live["status"] = "exited"
        record(
            probe / "queue/state.json",
            {
                "status": "needs_attention" if fault == "failed" else "waiting_for_upper_lr",
                "binding_sha256": frozen["probe_queue_binding_sha256"],
            },
        )
    with pytest.raises(RuntimeError):
        module.advance(root)
    assert read(queue.paths(root)[2] / "state.json")["status"] == "needs_attention"
    assert not launches


def test_completed_owned_probe_rederives_then_spawns_exact_validator_once(handoff):
    root, module, frozen, probe, live, launches, _ = handoff
    live["status"] = "exited"
    record(probe / "queue/state.json", {"status": "measured", "binding_sha256": frozen["probe_queue_binding_sha256"]})
    (probe / "queue/stopped_at").write_text("2026-11-17T08:00:00+08:00\n")
    assert module.advance(root)["status"] == "unknown_controller"
    assert launches[0]["argv"] == queue.command(root)
    assert read(queue.paths(root)[2] / "state.json")["predecessor"] == {"rederived": {"source": "fixture"}}
    assert read(queue.paths(root)[2] / "controller.json")["start_ticks"] is None
    assert module.advance(root)["status"] == "unknown_controller"
    assert len(launches) == 1
    with pytest.raises(FileExistsError):
        module.spawn(root, frozen)


def test_prerequisite_observation_timeout_waits_without_retrying_or_creating_controller(handoff):
    root, module, frozen, probe, live, launches, reconstruction = handoff
    live["status"] = "exited"
    record(probe / "queue/state.json", {"status": "measured", "binding_sha256": frozen["probe_queue_binding_sha256"]})
    (probe / "queue/stopped_at").write_text("2026-11-17T08:00:00+08:00\n")
    reconstruction["error"] = subprocess.TimeoutExpired(["prerequisite"], 1200)
    assert module.advance(root)["status"] == "waiting_for_probe"
    assert not launches
    assert not (queue.paths(root)[2] / "controller.json").exists()


def test_spawn_failure_persists_controller_intent_and_prohibits_retry(bound_queue, monkeypatch):
    root, module, frozen = bound_queue

    def failed_spawn(*args, **kwargs):
        raise OSError("controller unavailable")

    monkeypatch.setattr(subprocess, "Popen", failed_spawn)
    with pytest.raises(OSError):
        module.spawn(root, frozen)
    assert read(queue.paths(root)[2] / "controller.json")["status"] == "starting"
    with pytest.raises(FileExistsError):
        module.spawn(root, frozen)


def test_corrected_handoff_does_not_reuse_failed_original_queue_intent(bound_queue, monkeypatch):
    root, module, frozen = bound_queue
    original = root / "run-preparation/upstream-full-model-queue"
    record(original / "state.json", {"status": "needs_attention"})
    record(original / "controller.json", {"status": "starting", "command": ["original"]})
    preserved = {name: sha256(original / name) for name in ("state.json", "controller.json")}
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: type("Child", (), {"pid": 987654})())
    monkeypatch.setattr(module, "observed_process", lambda pid: None)
    intent = module.spawn(root, frozen)
    assert intent["command"] == queue.command(root)
    assert intent["start_ticks"] is None
    assert {name: sha256(original / name) for name in preserved} == preserved
    assert read(queue.paths(root)[2] / "controller.json")["command"] == intent["command"]


def test_real_marked_controller_observation_requires_generation_and_exact_argv(bound_queue, monkeypatch):
    _, module, _ = bound_queue
    argv = [queue.PYTHON, "-c", "import sys; print('ready',flush=True); sys.stdin.read()"]
    with subprocess.Popen(
        argv,
        env=os.environ | {queue.CONTROLLER_MARKER: "test-owner"},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as child:
        assert child.stdout.readline().strip() == "ready"
        intent = {
            "pid": child.pid,
            "start_ticks": module.observed_process(child.pid)[1],
            "command": argv,
            "marker": "test-owner",
        }
        try:
            assert module.observe(intent, queue.CONTROLLER_MARKER) == "running"
            with pytest.raises(RuntimeError):
                module.observe(intent | {"command": ["unowned"]}, queue.CONTROLLER_MARKER)
        finally:
            child.communicate(input="", timeout=10)
    assert module.observe(intent, queue.CONTROLLER_MARKER) == "exited"
    monkeypatch.setattr(module, "observed_process", lambda pid: ("Z", "reused", []))
    with pytest.raises(RuntimeError, match="PID reused"):
        module.observe(intent, queue.CONTROLLER_MARKER)


def test_failed_queue_installation_retains_intent_and_does_not_install_again(bound_queue, monkeypatch):
    root, module, _ = bound_queue
    launches = []
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: queue.NODE + "\n")
    original = subprocess.Popen

    def fail(argv, **kwargs):
        if argv[0] == "bash":
            launches.append(argv)
            raise OSError("queue unavailable")
        return original(argv, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(OSError):
        module.install(root)
    assert read(queue.paths(root)[2] / "installation.json")["status"] == "needs_attention"
    with pytest.raises(FileExistsError):
        module.install(root)
    assert len(launches) == 1


@pytest.mark.parametrize("fault", ["owner", "cleanup", "restore", "receipt"])
def test_terminal_controller_requires_owned_success_and_complete_restoration(bound_queue, fault):
    root, module, frozen = bound_queue
    preparation = root / "run-preparation"
    receipt = preparation / "validation-upstream-full-model-corrected-gpu.json"
    record(receipt, {"checks": {"fixture": True}})
    intent = {"pid": 1234, "start_ticks": "5678"}
    state = {
        "status": "passed",
        "controller_pid": 1234,
        "controller_start_ticks": "5678",
        "binding_sha256": frozen["validation_binding_sha256"],
        "validation_sha256": sha256(receipt),
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "child_marker": "unassigned-test-marker",
    }
    if fault == "owner":
        state["controller_pid"] = 9999
    elif fault == "receipt":
        state["validation_sha256"] = "changed"
    else:
        state["cleanup_exit_code" if fault == "cleanup" else "keeper_restore"] = 1
    record(preparation / "upstream-full-model-corrected-state.json", state)
    with pytest.raises(RuntimeError):
        module.completed(root, frozen, intent)


@pytest.mark.parametrize("fault", [None, "keeper_occupancy"])
def test_completed_validation_reconstructs_receipt_and_requires_current_keeper_occupancy(
    bound_queue, monkeypatch, fault
):
    root, module, frozen = bound_queue
    receipt = root / "run-preparation/validation-upstream-full-model-corrected-gpu.json"
    record(receipt, {"checks": {"fixture": True}})
    intent = {"pid": 1234, "start_ticks": "5678"}
    record(
        root / "run-preparation/upstream-full-model-corrected-state.json",
        {
            "status": "passed",
            "controller_pid": 1234,
            "controller_start_ticks": "5678",
            "binding_sha256": frozen["validation_binding_sha256"],
            "validation_sha256": sha256(receipt),
            "exit_code": 0,
            "cleanup_exit_code": 0,
            "keeper_restore": 0,
            "child_marker": "unassigned-test-marker",
        },
    )
    keeper_file = root / "keeper.pid"
    keeper_file.write_text("9000\n")
    generation = [(9000 + rank, str(rank), 9000 + rank) for rank in range(9)]
    monkeypatch.setattr(module, "keeper_generation", lambda pid: generation)
    monkeypatch.setattr(module, "observe", lambda intent, marker: "exited")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, json.dumps({"checks": {"fixture": True}}), ""),
    )
    occupied = list(range(9001, 9009))
    if fault:
        occupied[-1] = 9999
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "\n".join(map(str, occupied)))
    if fault:
        with pytest.raises(RuntimeError):
            module.completed(root, frozen, intent, keeper_file)
    else:
        completed = module.completed(root, frozen, intent, keeper_file)
        assert completed["receipt_sha256"] == sha256(receipt)
        assert completed["keeper_generation"] == generation


def test_final_reconstruction_timeout_preserves_original_controller_and_retries_only_observation(handoff, monkeypatch):
    root, module, frozen, _, _, launches, _ = handoff
    directory = queue.paths(root)[2]
    record(
        directory / "controller.json",
        {
            "command": frozen["command"],
            "validation_binding_sha256": frozen["validation_binding_sha256"],
            "queue_binding_sha256": sha256(directory / "binding.json"),
            "pid": 1234,
            "start_ticks": "5678",
        },
    )
    monkeypatch.setattr(module, "observe", lambda intent, marker: "exited")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(["reconstruction"], 1200)

    monkeypatch.setattr(module, "completed", timeout)
    assert module.advance(root)["status"] == "unknown_controller"
    assert module.advance(root)["status"] == "unknown_controller"
    assert not launches


def test_queue_ledger_reports_saved_wait_and_withholds_changed_controller_or_source(bound_queue):
    root, _, frozen = bound_queue
    directory = queue.paths(root)[2]
    digest = sha256(directory / "binding.json")
    record(directory / "state.json", {"status": "waiting_for_probe", "binding_sha256": digest})
    assert "Saved handoff `waiting_for_probe`" in "\n".join(
        upstream_full_model_queue_lines(root, directory / "binding.json")
    )
    record(
        directory / "controller.json",
        {
            "status": "starting",
            "command": ["unowned"],
            "queue_binding_sha256": digest,
            "validation_binding_sha256": frozen["validation_binding_sha256"],
        },
    )
    assert "controller intent differs" in "\n".join(upstream_full_model_queue_lines(root, directory / "binding.json"))
    (Path(frozen["helpers"]) / "queue_upstream_full_model.sh").write_text("changed\n")
    lines = "\n".join(upstream_full_model_queue_lines(root, directory / "binding.json"))
    assert "release identity differs" in lines
    assert "Saved handoff" not in lines
