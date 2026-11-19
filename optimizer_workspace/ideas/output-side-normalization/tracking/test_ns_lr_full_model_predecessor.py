# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require exact original lower-LR terminal receipts without GPU execution or tensor replay."""

import os
import signal
import subprocess
from pathlib import Path

import ns_lr_full_model_predecessor as predecessor
import ns_lr_full_model_predecessor_test_support as support
import pytest
from fused_attribution_validation_controller import acknowledged_process
from selected_lr_release_owner import read, sha256


@pytest.fixture
def completed(tmp_path, monkeypatch):
    return support.completed(tmp_path, monkeypatch)


def test_original_terminal_handoff_retains_rejected_cases_and_no_actual_lr_authority(completed):
    root, cpu, intent, state, terminal, previous = completed
    before = {path: sha256(path) for path in root.rglob("*.json")}
    result = predecessor.retained(root, cpu, intent, state, terminal, previous)
    assert len(result["lower_lr_decisions"]["accepted_cases"]) == 17
    assert result["lower_lr_decisions"]["rejected_cases"] == [predecessor.contract.specs()[0]["name"]]
    assert result["lower_lr_decisions"]["all_cases_accepted"] is False
    assert result["actual_lr_execution_authorized"] is False and result["production_authorized"] is False
    assert result["restored_keeper"] == terminal["keeper"]
    assert result["evidence_sha256"][predecessor.queue.TERMINAL] == sha256(root / predecessor.queue.TERMINAL)
    assert before == {path: sha256(path) for path in before}


@pytest.mark.parametrize("fault", ("none", "phase_ancestry", "phase_cleanup", "missing_phase", "terminal_origin"))
def test_complete_original_queue_pipeline_authenticates_phase_receipts_without_tensor_replay(
    completed, monkeypatch, fault
):
    root, cpu, _, state, terminal, previous = completed
    # Accepted prior CPU archives and the earlier NS diagnostic are external evidence boundaries.
    monkeypatch.setattr(predecessor.reporting.queue, "require_cpu", lambda *args: cpu)
    monkeypatch.setattr(predecessor.previous, "evidence", lambda root: previous)
    if fault == "phase_ancestry":
        state["phases"][0]["generation"]["parent"] = 999
    elif fault == "phase_cleanup":
        state["phases"][-1]["cleanup_exit_code"] = False
    elif fault == "missing_phase":
        state["phases"].pop()
    elif fault == "terminal_origin":
        terminal["verifier_sha256"] = "substituted-verifier"
        support.rebind_terminal(completed)
    support.numerical.record(root / predecessor.queue.WORK / "state.json", state)
    if fault == "none":
        result = predecessor.evidence(root)
        assert result["lower_lr_decisions"]["rejected_cases"] == [predecessor.contract.specs()[0]["name"]]
        assert result["actual_lr_execution_authorized"] is False
        assert len(result["evidence_sha256"]) >= 120
    else:
        with pytest.raises((RuntimeError, ValueError)):
            predecessor.evidence(root)


@pytest.mark.parametrize(
    "fault", ("ancestry", "bool_exit", "rejected_summary", "rebound_keeper", "changed_source", "coverage")
)
def test_original_terminal_handoff_refuses_rebound_receipts_even_with_refreshed_terminal_hashes(completed, fault):
    root, cpu, intent, state, terminal, previous = completed
    binding, _, state_path, _, result_path = predecessor.contract.locations(root)
    if fault in ("ancestry", "bool_exit", "rejected_summary"):
        result = read(result_path)
        if fault == "ancestry":
            result["gpu_commands"][0]["generation"]["parent"] = 999
            entry = result["gpu_commands"][0]
            support.numerical.record(
                Path(entry["log"]).with_suffix(".execution.json"),
                {key: value for key, value in entry.items() if key not in ("trial", "mode", "trajectory")},
            )
        elif fault == "bool_exit":
            result["gpu_commands"][0]["exit_code"] = False
        else:
            result["rejected_cases"] = []
        support.numerical.record(result_path, result)
    elif fault == "rebound_keeper":
        terminal["keeper"]["generation"]["100"]["start_ticks"] = "substituted"
    elif fault == "changed_source":
        candidate = Path(read(binding)["candidate"])
        config = candidate / predecessor.contract.specs()[0]["config_name"]
        config.write_bytes(config.read_bytes() + b"\n")
    else:
        terminal["evidence_sha256"].pop(str(state_path.relative_to(root)))
        support.numerical.record(root / predecessor.queue.TERMINAL, terminal)
        state["terminal_sha256"] = sha256(root / predecessor.queue.TERMINAL)
        support.numerical.record(root / predecessor.queue.WORK / "state.json", state)
    if fault != "coverage":
        support.rebind_terminal(completed)
    with pytest.raises(ValueError):
        predecessor.retained(root, cpu, intent, state, terminal, previous)


def test_live_original_cpu_queue_waits_without_tensor_or_terminal_output(tmp_path, monkeypatch):
    root, work = tmp_path, tmp_path / predecessor.queue.WORK
    work.mkdir(parents=True)
    marker = f"cpu-original-{os.getpid()}"
    command = [predecessor.contract.PYTHON, "-c", "import signal; signal.pause()"]
    environment = predecessor.queue.ENVIRONMENT | {predecessor.queue.MARKER: marker}
    child = subprocess.Popen(command, cwd=work, env=os.environ | environment, start_new_session=True)
    try:
        generation = acknowledged_process(
            child, command, work, predecessor.queue.MARKER, marker, predecessor.queue.ENVIRONMENT
        )
        support.numerical.record(work / "intent.json", {"generation": generation, "marker": marker})
        support.numerical.record(work / "state.json", {"status": "waiting", "phases": []})
        monkeypatch.setattr(predecessor.reporting, "INTENT_SHA256", sha256(work / "intent.json"))
        observation = predecessor.readiness(root)
        assert observation["ready"] is False
        assert observation["owners"]["original_queue"]["status"] == "live_exact_generation"
        with pytest.raises(RuntimeError):
            predecessor.evidence(root)
        assert not (root / predecessor.queue.TERMINAL).exists()
    finally:
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=10)
