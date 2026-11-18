# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify release contracts using local receipts and a fake process boundary."""

import json
import subprocess

import launch_wave5 as controller
import pytest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def campaign(tmp_path, monkeypatch):
    task = tmp_path / "task"
    root = tmp_path / "results"
    candidate = tmp_path / "candidate"
    monkeypatch.setattr(controller, "TASK", task)
    for name in controller.HELPERS:
        path = task / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("helper")
    save(root / "data/experiment_receipt.json", {"verified": True})
    source = candidate / "experiment/train.py"
    source.parent.mkdir(parents=True)
    source.write_text("source")
    entries = [
        {"arm": arm, "batch": "0p5m", "seed": 0}
        for arm in ("inner_outer_mlpdiag", "inner_outer_mlpdiag_fused", "inner_outer_leftdiag_fused", "muont")
    ]
    cluster = {"nodes": controller.NODES, "wave_name": "wave5", "environment": {}, "wave": entries}
    save(candidate / "experiment/cluster.json", cluster)
    for entry in entries:
        save(candidate / "experiment/configs" / f"{entry['arm']}_0p5m_seed0.json", entry)
    files = {
        str(path.relative_to(candidate)): controller.sha256(path) for path in candidate.rglob("*") if path.is_file()
    }
    prep = root / "run-preparation"
    data_hash = controller.sha256(root / "data/experiment_receipt.json")
    save(
        prep / "wave5-candidate.json",
        {
            "candidate": str(candidate),
            "files_sha256": files,
            "data_receipt_sha256": data_hash,
        },
    )
    execution_log = prep / "gate.log"
    execution_log.write_text("gate passed")
    save(
        prep / "validation-wave5-gpu.json",
        {
            "checks": dict.fromkeys(controller.REQUIRED_GATES, True),
            "source_sha256": {"experiment/train.py": controller.sha256(source)},
            "data_receipt_sha256": data_hash,
            "config_sha256": {
                entry["arm"]: controller.sha256(candidate / "experiment/configs" / f"{entry['arm']}_0p5m_seed0.json")
                for entry in entries
            },
            "environment": {},
            "evidence": {
                "gpu_commands": [
                    {
                        "exit_code": 0,
                        "log": str(execution_log),
                        "log_sha256": controller.sha256(execution_log),
                    }
                ]
            },
        },
    )
    save(
        prep / "wave5-validation-state.json",
        {
            "status": "passed",
            "keeper_restore": 0,
            "validation_sha256": controller.sha256(prep / "validation-wave5-gpu.json"),
        },
    )
    for wave, nodes in (("wave4", controller.NODES), ("wave4-baseline", ("28.216.17.144",))):
        launches = []
        cloud_runs = {}
        for index, node in enumerate(nodes):
            name = f"arm{index}" if wave == "wave4" else "muont_seed2"
            out = root / "runs" / wave / name
            save(
                out / "run.json",
                {
                    "status": "complete",
                    "completed_steps": 5120,
                    "config": {"training": {"steps": 5120}},
                },
            )
            (out / "metrics_from_000000.jsonl").write_text("scalar history")
            save(out / "checkpoints/step_005120/complete.json", {"step": 5120, "world": 8})
            (out.parent / f"keeper-restore-{node}.log").write_text("keeper start exit 0")
            launches.append({"node": node, "out": str(out), "pid": 99999, "arm": "muont", "seed": 2, "batch": "0p5m"})
            cloud_runs[name] = {
                "status": "verified",
                "manifest_sha256": controller.sha256(out / "run.json"),
                "metrics_sha256": controller.sha256(out / "metrics_from_000000.jsonl"),
            }
        save(prep / f"{wave}.json", {"launches": launches})
        save(prep / f"swanlab-upload-{wave}.json", {"status": "verified", "runs": cloud_runs})
    controller.bind_queue(root)
    return root, candidate


def test_release_requires_current_verified_execution_and_history(tmp_path, monkeypatch):
    root, candidate = campaign(tmp_path, monkeypatch)
    assert controller.require_release(root)[0] == candidate
    log = root / "run-preparation/gate.log"
    log.write_text("gate failure")
    with pytest.raises(ValueError, match="execution failed"):
        controller.require_release(root)
    log.write_text("gate passed")
    history = root / "runs/wave4-baseline/muont_seed2/metrics_from_000000.jsonl"
    history.write_text("changed history")
    with pytest.raises(ValueError, match="history changed"):
        controller.require_release(root)


def test_launch_queue_waits_for_keeper_restoration_and_rejects_failure(tmp_path, monkeypatch):
    root, _ = campaign(tmp_path, monkeypatch)
    directory = root / "queue"
    save(directory / "latest-wave4.json", {"status": "complete"})
    save(directory / "latest-baseline.json", {"status": "complete"})
    path = root / "run-preparation/wave5-validation-state.json"
    state = json.loads(path.read_text())
    state.pop("keeper_restore")
    save(path, state)
    assert controller.queue_status(root, directory) == "waiting"
    state["keeper_restore"] = False
    save(path, state)
    assert controller.queue_status(root, directory) == "needs_attention"
    state["keeper_restore"] = 0
    save(path, state)
    assert controller.queue_status(root, directory) == "ready"
    (root / "runs/wave4-baseline/keeper-restore-28.216.17.144.log").unlink()
    assert controller.queue_status(root, directory) == "waiting"


def test_failed_watcher_install_does_not_leave_other_launches_unattempted(tmp_path, monkeypatch):
    root, _ = campaign(tmp_path, monkeypatch)
    report = {
        "launches": [
            {"node": node, "pid": 99999, "out": str(root / "runs/wave5" / f"arm{index}")}
            for index, node in enumerate(controller.NODES)
        ]
    }
    attempts = []

    def check_output(command, **kwargs):
        node = command[-2].removeprefix("root@")
        attempts.append(node)
        if node == controller.NODES[0]:
            raise subprocess.CalledProcessError(255, command)
        return json.dumps({"pid": 1234, "status": "watching"})

    monkeypatch.setattr(controller.subprocess, "check_output", check_output)
    errors = controller.install_watchers(root, report)
    assert attempts == list(controller.NODES)
    assert set(errors) == {controller.NODES[0]}
    assert all(entry["keeper_watcher_pid"] == 1234 for entry in report["launches"][1:])
    assert json.loads((root / "run-preparation/wave5.json").read_text())["launches"] == report["launches"]


@pytest.mark.parametrize("process_status", ["Ssl python training", "Ssl python reused"])
def test_node_release_refuses_live_or_reused_predecessor_without_stopping_keeper(tmp_path, monkeypatch, process_status):
    root, _ = campaign(tmp_path, monkeypatch)
    calls = []

    def check_output(command, **kwargs):
        assert command == ["hostname", "-I"]
        return "28.216.17.9"

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, process_status, "")

    monkeypatch.setattr(controller.subprocess, "check_output", check_output)
    monkeypatch.setattr(controller.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="live or its PID"):
        controller.node_action(root, "release")
    assert len(calls) == 1
    assert calls[0][0] == "ps"


def test_restore_refuses_unacknowledged_training_before_cuda_initialization(tmp_path, monkeypatch):
    root, _ = campaign(tmp_path, monkeypatch)

    def check_output(command, **kwargs):
        if command == ["hostname", "-I"]:
            return "28.216.17.9"
        if command == ["ps", "-eo", "args="]:
            return "python -m torch.distributed.run --out /results/runs/wave5/unacknowledged"
        raise AssertionError("Restoration must not proceed past live training")

    monkeypatch.setattr(controller.subprocess, "check_output", check_output)
    with pytest.raises(RuntimeError, match="unacknowledged"):
        controller.node_action(root, "restore")


def test_monitor_startup_failure_cannot_bypass_idle_allocation_restoration(tmp_path, monkeypatch):
    root, _ = campaign(tmp_path, monkeypatch)
    report = {"launches": [{"node": controller.NODES[0], "pid": 1234, "out": "run"}]}
    state = {"released_nodes": list(controller.NODES)}
    restored = []
    monkeypatch.setattr(controller, "install_watchers", lambda root, report: {})

    def fail_monitor(root, state):
        raise OSError("Monitor process creation failed")

    monkeypatch.setattr(controller, "install_monitor", fail_monitor)
    monkeypatch.setattr(controller, "remote", lambda root, node, action: restored.append((node, action)))
    controller.reconcile(root, state, report)
    assert restored == [(node, "restore") for node in controller.NODES[1:]]
    saved = json.loads((root / "run-preparation/wave5-launch-state.json").read_text())
    assert saved["status"] == "needs_attention"
    assert saved["monitoring_recovery_error"] == "Monitor process creation failed"
