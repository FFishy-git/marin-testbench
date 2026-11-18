# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise profiler cleanup and completed-wave handoff at real process and receipt boundaries."""

import json
import os
import signal
import subprocess
from pathlib import Path

import pytest
import run_selected_lr_stage_profile as controller
import selected_lr_stage_profile_handoff as handoff

ORPHAN_VALIDATOR = """
import os, signal, subprocess, sys
read_fd, write_fd = os.pipe()
command = [sys.executable, '-c',
           'import os,signal,sys;os.write(int(sys.argv[1]),b"R");signal.pause()',
           str(write_fd)]
rank = subprocess.Popen(command, pass_fds=(write_fd,), start_new_session=True)
os.close(write_fd)
assert os.read(read_fd, 1) == b'R'
os.close(read_fd)
print(rank.pid, flush=True)
sys.exit(7)
"""


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def test_profile_failed_validator_reaps_independent_rank_without_signaling_unmarked_peer(tmp_path, monkeypatch):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    candidate.mkdir()
    _, helpers, state_path, _, _ = controller.locations(root)
    helpers.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    (helpers / "validate_selected_lr_stage_profile.py").write_text(ORPHAN_VALIDATOR)
    peer = subprocess.Popen([controller.PYTHON, "-c", "import signal;signal.pause()"], start_new_session=True)
    peer_handle = controller.pidfd_open(peer.pid)
    state = {}

    def forbid_numeric_signal(*args):
        raise AssertionError("Cleanup must signal stable pidfds, never numeric process IDs")

    monkeypatch.setattr(os, "kill", forbid_numeric_signal)
    monkeypatch.setattr(os, "killpg", forbid_numeric_signal)
    try:
        code = controller.bounded_validator(root, {"candidate": str(candidate)}, state)
        saved = json.loads(state_path.read_text())
        rank_pid = int(Path(saved["log"]).read_text().strip())
        rank = controller.observed(rank_pid)
        assert code == 7
        assert saved["cleanup_exit_code"] == 7
        assert saved["marked_descendants_absent"] is True
        assert saved["command"] == controller.validator_command(root, candidate)
        assert saved["log_sha256"] == controller.sha256(Path(saved["log"]))
        assert str(rank_pid) in saved["child_generation"]
        assert rank is None or rank["status"] in ("Z", "X", "x")
        assert not controller.marker_processes(controller.CHILD_MARKER, saved["child_marker"])
        assert peer.poll() is None
    finally:
        signal.pidfd_send_signal(peer_handle, signal.SIGTERM)
        peer.wait(timeout=10)
        os.close(peer_handle)


def test_profile_successful_validator_preserves_zero_cleanup_and_original_handle(tmp_path):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    candidate.mkdir()
    _, helpers, state_path, _, _ = controller.locations(root)
    helpers.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    (helpers / "validate_selected_lr_stage_profile.py").write_text("print('bounded success')\n")
    state = {}
    assert controller.bounded_validator(root, {"candidate": str(candidate)}, state) == 0
    saved = json.loads(state_path.read_text())
    assert saved["cleanup_exit_code"] == 0
    assert saved["marked_descendants_absent"] is True
    assert saved["log_sha256"] == controller.sha256(Path(saved["log"]))
    assert controller.observed(saved["child_pid"]) is None


@pytest.mark.parametrize("drift", ("parent_exit", "other_parent", "reused_pid", "session", "argv"))
def test_profile_detached_acknowledgment_allows_parent_exit_without_adopting_replacements(drift):
    generation = {
        "pid": 2001,
        "parent": 2000,
        "start_ticks": "original",
        "group": 2001,
        "session": 2001,
        "command": ["python", "-P", "owned-controller.py"],
        "status": "S",
    }
    intent = {"status": "acknowledged", "pid": 2001, "start_ticks": "original", "generation": generation}
    own = generation | {"parent": 1, "status": "R"}
    if drift == "other_parent":
        own["parent"] = 999
    elif drift == "reused_pid":
        own["start_ticks"] = "replacement"
    elif drift == "session":
        own["session"] = 999
    elif drift == "argv":
        own["command"] = ["unissued-controller"]
    if drift == "parent_exit":
        controller.require_acknowledged_controller(intent, own)
        assert own["parent"] == 1 and intent["generation"]["parent"] == 2000
    else:
        with pytest.raises(RuntimeError):
            controller.require_acknowledged_controller(intent, own)


@pytest.fixture
def completed_wave(tmp_path, monkeypatch):
    root = tmp_path / "root"
    helpers = root / "controllers/wave12/tracking"
    release = {
        "arms": list(handoff.ARMS),
        "nodes": list(handoff.NODES),
        "helpers": str(helpers),
        "commands": {node: ["owned-command", node] for node in handoff.NODES},
    }
    release_path = root / "run-preparation/wave12-launch-binding.json"
    record(release_path, release)
    digest = handoff.sha256(release_path)
    comparisons = {
        arm: {
            "final_comparison_eligible": True,
            "training_windows": [{"index": index} for index in range(5)],
            "evaluations": {"selection_loss": {"shared_tokens": list(range(22))}},
        }
        for arm in handoff.ARMS[1:]
    }
    analysis = {"source_release_sha256": digest, "analyses": [{"comparisons": comparisons}]}
    record(root / "run-preparation/analysis-wave12.json", analysis | {"time": "2026-11-17T18:00:00+00:00"})
    histories = {
        arm: {
            "manifest_sha256": f"manifest-{arm}",
            "metrics_sha256": f"metrics-{arm}",
            "checkpoint_receipt_sha256": f"checkpoint-{arm}",
            "checkpoint_rank_bytes": [100] * 8,
        }
        for arm in handoff.ARMS
    }
    columns = {
        arm: {
            "point_counts": {"tokens": 5120, "train/loss": 5120},
            "final_metrics": {"tokens": 2684354560, "loss": 3.4},
        }
        for arm in handoff.ARMS
    }
    cloud = {
        "status": "verified",
        "visibility": "PRIVATE",
        "project": "siyuc3141/output-side-normalization-marin",
        "runs": {
            arm: {
                "status": "verified",
                **histories[arm],
                "verification": {
                    "state": "FINISHED",
                    "configuration_verified": True,
                    "token_axis_verified": True,
                    "integer_values_exact": True,
                    "csv_export_max_ulp": 2,
                    **columns[arm],
                    "serialization_errors": {
                        key: {"max_absolute": 0.0, "max_ulp": 0.0} for key in columns[arm]["point_counts"]
                    },
                },
            }
            for arm in handoff.ARMS
        },
    }
    record(root / "run-preparation/swanlab-upload-wave12.json", cloud)
    workers = {
        arm: {
            "status": "complete",
            "arm": arm,
            "node": node,
            "completed_run": histories[arm],
            "command": release["commands"][node],
            "release_binding_sha256": digest,
            "exit_code": 0,
            "cleanup_exit_code": 0,
            "keeper_restore": 0,
        }
        for arm, node in zip(handoff.ARMS, handoff.NODES, strict=True)
    }

    def run(command, **kwargs):
        if command[0] == "ssh":
            arm = command[-1].split("--arm ")[-1]
            payload = {"status": "complete", "receipt": workers[arm]}
        elif "expected_columns" in command[3]:
            payload = columns
        elif "completed_run" in command[3]:
            payload = {"analysis": analysis, "histories": histories}
        else:
            raise AssertionError(f"Unexpected external process: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    # Source-bound production acceptance is separately tested; this fixture models completed telemetry/SSH.
    monkeypatch.setattr(handoff, "verify_production_binding", lambda root, wave: (release, digest))
    monkeypatch.setattr(subprocess, "run", run)
    return root, analysis, cloud, workers


def test_profile_handoff_accepts_writer_timestamp_without_weakening_scientific_identity(completed_wave):
    root, analysis, _, _ = completed_wave
    accepted = handoff.require_completed_wave12(root)
    assert accepted["status"] == "wave12_completed_original_workers_restored_analyzed_cloud_verified"
    assert set(accepted["workers"]) == set(handoff.ARMS)
    assert accepted["analysis_sha256"] == handoff.sha256(root / "run-preparation/analysis-wave12.json")
    changed = json.loads(root.joinpath("run-preparation/analysis-wave12.json").read_text())
    changed["analyses"][0]["comparisons"][handoff.ARMS[1]]["training_windows"][4]["index"] = 999
    record(root / "run-preparation/analysis-wave12.json", changed)
    with pytest.raises(RuntimeError):
        handoff.require_completed_wave12(root)
    assert analysis["analyses"][0]["comparisons"][handoff.ARMS[1]]["training_windows"][4]["index"] == 4


@pytest.mark.parametrize("fault", ("partial_window", "missing_eval", "cloud_rows", "worker_cleanup", "checkpoint"))
def test_profile_handoff_refuses_incomplete_final_analysis_cloud_or_cleanup(completed_wave, fault):
    root, analysis, cloud, workers = completed_wave
    arm = handoff.ARMS[1]
    if fault == "partial_window":
        analysis["analyses"][0]["comparisons"][arm]["training_windows"].pop()
        record(root / "run-preparation/analysis-wave12.json", analysis)
    elif fault == "missing_eval":
        analysis["analyses"][0]["comparisons"][arm]["evaluations"]["selection_loss"]["shared_tokens"].pop()
        record(root / "run-preparation/analysis-wave12.json", analysis)
    elif fault == "cloud_rows":
        cloud["runs"][arm]["verification"]["point_counts"] = {"tokens": 5119, "train/loss": 5119}
        record(root / "run-preparation/swanlab-upload-wave12.json", cloud)
    elif fault == "worker_cleanup":
        workers[arm]["cleanup_exit_code"] = 1
    else:
        cloud["runs"][arm]["checkpoint_rank_bytes"] = [101] * 8
        record(root / "run-preparation/swanlab-upload-wave12.json", cloud)
    with pytest.raises(RuntimeError):
        handoff.require_completed_wave12(root)
