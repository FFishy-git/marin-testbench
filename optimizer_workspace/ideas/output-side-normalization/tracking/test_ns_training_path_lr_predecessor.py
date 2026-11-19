# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real original-owner waiting, canonical handoff and unchanged retained evidence."""

import os
import signal
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ns_training_path_lr_predecessor as predecessor
import ns_training_path_lr_predecessor_test_support as support
import pytest
import selected_lr_failure_handoff
from fused_attribution_validation_controller import acknowledged_process
from selected_lr_release_owner import read, sha256


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(selected_lr_failure_handoff, "MANAGER_LOCK", tmp_path / "cpu-manager.lock")
    manager = tmp_path / "cpu-manager.sh"
    manager.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(selected_lr_failure_handoff, "MANAGER_SOURCE", manager)
    monkeypatch.setattr(selected_lr_failure_handoff, "KEEPER", str(manager))
    return support.completed(tmp_path, monkeypatch)


def test_original_terminal_evidence_preserves_rejections_without_reading_tensors_or_issuing_owner(completed):
    root, restored = completed
    before = {path: sha256(path) for path in root.rglob("*.json")}
    result = predecessor.evidence(root)
    assert len(result["original_decisions"]["accepted_cases"]) == 17
    assert result["original_decisions"]["rejected_cases"] == ["original-case-0"]
    assert result["original_decisions"]["all_cases_accepted"] is False
    assert len(result["checkpoint_payloads"]) == 18 * 2 * 3 * 8
    assert result["restored_keeper"] == restored
    assert result["actual_lr_execution_authorized"] is False and result["production_authorized"] is False
    assert before == {path: sha256(path) for path in before}
    assert not (root / predecessor.SEAL).exists()


@pytest.mark.parametrize("owner", ("controller", "validator", "terminal_waiter"))
def test_live_original_owner_refuses_handoff_without_terminal_mutation(completed, monkeypatch, owner):
    root, _ = completed
    frozen, intent, state, waiter = predecessor.original(root)
    work = root / predecessor.reporting.FOLLOWUP_WORK
    helpers = Path(frozen["helpers"])
    contexts = {
        "controller": (
            intent["command"],
            helpers,
            predecessor.contract.CONTROLLER_MARKER,
            intent["marker"],
            frozen["controller_environment"],
        ),
        "validator": (
            state["command"],
            Path(frozen["candidate"]),
            predecessor.contract.CHILD_MARKER,
            state["child_marker"],
            frozen["environment"],
        ),
        "terminal_waiter": (
            [predecessor.contract.PYTHON, "-c", "import signal; signal.pause()"],
            work,
            predecessor.followup.MARKER,
            waiter["marker"],
            predecessor.followup.ENVIRONMENT,
        ),
    }
    command, cwd, marker_name, marker, environment = contexts[owner]
    child = subprocess.Popen(
        command, cwd=cwd, env=os.environ | environment | {marker_name: marker}, start_new_session=True
    )
    try:
        own = acknowledged_process(child, command, cwd, marker_name, marker, environment)
        if owner == "controller":
            intent.update(pid=own["pid"], start_ticks=own["start_ticks"], generation=own)
            state.update(controller_pid=own["pid"], controller_start_ticks=own["start_ticks"])
            state["child_generation"]["parent"] = own["pid"]
        elif owner == "validator":
            # The external /proc boundary retains the fixture controller as this child's original parent.
            own["parent"] = intent["pid"]
            state.update(child_pid=own["pid"], child_start_ticks=own["start_ticks"], child_generation=own)
        else:
            waiter.update(pid=own["pid"], start_ticks=own["start_ticks"], command=command, generation=own)
            waiter_state = read(work / "state.json")
            waiter_state.update(pid=own["pid"], start_ticks=own["start_ticks"])
            support.record(work / "state.json", waiter_state)
            monkeypatch.setattr(predecessor.followup, "command", lambda *_: command)
        support.record(helpers.parent / "controller.json", intent)
        support.record(predecessor.contract.locations(root)[2], state)
        support.record(work / "intent.json", waiter)
        monkeypatch.setattr(predecessor.reporting, "INTENT_SHA256", sha256(helpers.parent / "controller.json"))
        monkeypatch.setattr(predecessor.reporting, "FOLLOWUP_INTENT_SHA256", sha256(work / "intent.json"))
        observation = predecessor.readiness(root)
        assert observation["ready"] is False
        assert observation["owners"][owner]["status"] == "live_exact_generation"
        with pytest.raises(RuntimeError):
            predecessor.evidence(root)
        assert not (root / predecessor.SEAL).exists()
    finally:
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=10)


@pytest.mark.parametrize(
    "fault", ("claimed", "waiter_incomplete", "missing_case", "erased_rejection", "changed_history", "payload")
)
def test_original_handoff_refuses_incomplete_rebound_or_post_claim_evidence(completed, fault):
    root, _ = completed
    report_path = root / "accepted-report.json"
    report = read(report_path)
    result_path = predecessor.contract.locations(root)[4]
    if fault == "claimed":
        report["gpu_progress"]["terminal"]["status"] = "reconstruction_claimed"
    elif fault == "waiter_incomplete":
        report["terminal_followup"]["status"] = "reconstructing_terminal"
    elif fault in ("missing_case", "erased_rejection"):
        result = read(result_path)
        if fault == "missing_case":
            result["results"].pop()
            report["gpu_progress"]["decisions"]["cases"] = result["results"]
        else:
            result["rejected_cases"] = []
        support.record(result_path, result)
    elif fault == "changed_history":
        next(root.rglob("metrics_continuous.jsonl")).write_text('{"changed":"scalar history"}\n')
    else:
        intent = read(predecessor.contract.locations(root)[1].parent / "controller.json")
        _, attempt, _, _ = predecessor.controller.terminal_paths(root, intent)
        support.record(attempt, {"time": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()})
    support.record(report_path, report)
    support.rebind_terminal(root)
    with pytest.raises((ValueError, RuntimeError)):
        predecessor.evidence(root)
    assert not (root / predecessor.SEAL).exists()


def test_original_handoff_seal_is_one_shot_and_rejects_changed_retained_payload(completed, monkeypatch):
    root, restored = completed
    # Keeper observation is the external device/process boundary; use its original persisted identity.
    monkeypatch.setattr(predecessor.handoff, "keeper_observation", lambda: restored)
    saved = predecessor.seal(root)
    path = root / predecessor.SEAL
    digest = sha256(path)
    assert predecessor.require_seal(root, digest) == saved
    with pytest.raises(FileExistsError):
        predecessor.seal(root)
    payload = next(root.rglob("rank_7.pt"))
    payload.write_bytes(payload.read_bytes() + b" changed")
    with pytest.raises(ValueError):
        predecessor.require_seal(root, digest)
    assert sha256(path) == digest and saved["original_decisions"]["rejected_cases"] == ["original-case-0"]


def test_original_handoff_seal_refuses_rebound_restored_keeper_before_publication(completed, monkeypatch):
    root, restored = completed
    changed = restored | {"pid": restored["pid"] + 1}
    monkeypatch.setattr(predecessor.handoff, "keeper_observation", lambda: changed)
    with pytest.raises((RuntimeError, ValueError)):
        predecessor.seal(root)
    assert not (root / predecessor.SEAL).exists()
