# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise canonical wave-6 inventory and failure reporting through real saved histories."""

import json
import shlex
import subprocess

import pytest
import snapshot_wave6


def inventory(root, *, complete=True):
    preparation = root / "run-preparation"
    preparation.mkdir()
    entries = []
    for index, (node, arm) in enumerate(zip(snapshot_wave6.NODES, snapshot_wave6.ARMS, strict=True)):
        directory = root / "runs/wave6" / arm
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(
            json.dumps(
                {
                    "status": "complete" if complete else "failed",
                    "completed_steps": 1,
                    "config": {"training": {"steps": 1, "seed": 0, "batch_tokens": 524288}},
                }
            )
        )
        (directory / "metrics_from_000000.jsonl").write_text(
            json.dumps({"step": 1, "tokens": 524288, "train_loss": 3.0}) + "\n"
        )
        entries.append({"node": node, "arm": arm, "seed": 0, "batch": "0p5m", "out": str(directory), "pid": 100 + index})
        (directory.parent / f"{arm}.watcher.json").write_text(
            json.dumps(
                {
                    "pid": 200 + index,
                    "watcher_start_ticks": str(1000 + index),
                    "training_pid": 100 + index,
                    "out": str(directory),
                    "status": "restored",
                    "keeper_restore": 0,
                }
            )
        )
    path = preparation / "wave6.json"
    path.write_text(json.dumps({"expected_launches": 4, "launches": entries}))
    return path, entries


def test_partial_inventory_preserves_confirmed_failed_run(tmp_path, monkeypatch):
    path, entries = inventory(tmp_path, complete=False)
    path.write_text(json.dumps({"expected_launches": 4, "launches": entries[:1]}))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", ""))
    result = snapshot_wave6.observe(tmp_path)
    assert result["runs"][0]["status"] == "stopped_incomplete"
    assert result["status"] == "needs_attention"
    with pytest.raises(ValueError, match="partial"):
        snapshot_wave6.verify_inventory(tmp_path, require_full=True)


def test_canonical_inventory_observes_complete_then_rejects_duplicate_and_wrong_path(tmp_path, monkeypatch):
    path, entries = inventory(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", ""))
    snapshot_wave6.verify_inventory(tmp_path, require_full=True)
    result = snapshot_wave6.observe(tmp_path)
    assert result["status"] == "complete"
    assert [run["name"] for run in result["runs"]] == list(snapshot_wave6.ARMS)
    path.write_text(json.dumps({"expected_launches": 4, "launches": [*entries[:3], entries[0]]}))
    with pytest.raises(ValueError, match="duplicate"):
        snapshot_wave6.observe(tmp_path)
    entries[-1] = {**entries[-1], "out": str(tmp_path / "other" / snapshot_wave6.ARMS[-1])}
    path.write_text(json.dumps({"expected_launches": 4, "launches": entries}))
    with pytest.raises(ValueError, match="canonical"):
        snapshot_wave6.observe(tmp_path)


def test_completed_training_waits_for_live_restoration_then_reports_stopped_watcher(tmp_path, monkeypatch):
    _, entries = inventory(tmp_path)
    entry = entries[0]
    path = tmp_path / "runs/wave6/muont.watcher.json"
    state = json.loads(path.read_text())
    state["status"] = "restoring"
    state.pop("keeper_restore")
    path.write_text(json.dumps(state))

    def observed(command, **kwargs):
        if shlex.split(command[-1])[-1] == str(state["pid"]):
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "status": "S",
                        "start_ticks": state["watcher_start_ticks"],
                        "same_generation": True,
                        "arguments": [
                            "python",
                            "watch_wave6.py",
                            "--run",
                            str(path),
                            "--pid",
                            str(entry["pid"]),
                            "--out",
                            entry["out"],
                            "--root",
                            str(tmp_path),
                        ],
                    }
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(subprocess, "run", observed)
    pending = snapshot_wave6.observe(tmp_path)
    assert pending["status"] == "awaiting_keeper_restoration"
    assert pending["runs"][0]["keeper_watcher"]["status"] == "restoring"

    def stopped(command, **kwargs):
        if shlex.split(command[-1])[-1] == str(state["pid"]):
            return subprocess.CompletedProcess(command, 0, json.dumps({"status": "missing"}), "")
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(subprocess, "run", stopped)
    failed = snapshot_wave6.observe(tmp_path)
    assert failed["status"] == "needs_attention"
    assert failed["runs"][0]["status"] == "complete"
    state.update(status="restored", keeper_restore=0)
    path.write_text(json.dumps(state))
    assert snapshot_wave6.observe(tmp_path)["status"] == "complete"
    state["training_pid"] = entry["pid"] + 1
    path.write_text(json.dumps(state))
    assert snapshot_wave6.observe(tmp_path)["status"] == "needs_attention"


def test_watcher_network_uncertainty_never_proves_restoration(tmp_path, monkeypatch):
    inventory(tmp_path)
    path = tmp_path / "runs/wave6/muont.watcher.json"
    state = json.loads(path.read_text())
    state["status"] = "watching"
    path.write_text(json.dumps(state))

    def network_failure(command, **kwargs):
        if shlex.split(command[-1])[-1] == str(state["pid"]):
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(subprocess, "run", network_failure)
    result = snapshot_wave6.observe(tmp_path)
    assert result["status"] == "awaiting_keeper_restoration"
    assert result["runs"][0]["keeper_watcher"]["status"] == "unknown"


def test_watcher_matching_command_with_reused_pid_is_not_live_evidence(tmp_path, monkeypatch):
    _, entries = inventory(tmp_path)
    path = tmp_path / "runs/wave6/muont.watcher.json"
    state = json.loads(path.read_text())
    state["status"] = "watching"
    path.write_text(json.dumps(state))

    def reused(command, **kwargs):
        if shlex.split(command[-1])[-1] == str(state["pid"]):
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "status": "S",
                        "start_ticks": "other",
                        "same_generation": True,
                        "arguments": [
                            "python",
                            "watch_wave6.py",
                            "--run",
                            str(path),
                            "--pid",
                            str(entries[0]["pid"]),
                            "--out",
                            entries[0]["out"],
                            "--root",
                            str(tmp_path),
                        ],
                    }
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(subprocess, "run", reused)
    result = snapshot_wave6.observe(tmp_path)
    assert result["status"] == "needs_attention"
    assert result["runs"][0]["keeper_watcher"]["error"] == "Keeper watcher stopped or PID was reused"
