# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain saved actual-LR queue failures and rejected cases in both journal reports."""

from datetime import UTC, datetime

import ns_lr_full_model_queue_reporting as reporting
import pytest
import update_endeavor as journal
from ns_lr_full_model_predecessor_test_support import generation
from watch_wave5 import write


@pytest.fixture
def queued(tmp_path, monkeypatch):
    work, cpu_path = tmp_path / reporting.queue.WORK, tmp_path / reporting.CPU
    work.mkdir(parents=True)
    cpu_path.parent.mkdir(parents=True)
    write(cpu_path, {"fixture": "external accepted CPU boundary"})
    monkeypatch.setattr(reporting, "CPU_SHA256", reporting.sha256(cpu_path))
    cpu = {"helpers_sha256": {reporting.queue.NEW_HELPERS[0]: "accepted-runner"}}
    command = reporting.queue.command(tmp_path, cpu_path)
    intent = {
        "status": "acknowledged",
        "pid": 111,
        "start_ticks": "211",
        "command": command,
        "marker": "original",
        "generation": generation(111, 1, command),
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": reporting.sha256(cpu_path),
        "runner_sha256": "accepted-runner",
        "parent_cpu_receipt_sha256": reporting.queue.PARENT_SHA256,
        "interval": 900,
    }
    write(work / "intent.json", intent)
    monkeypatch.setattr(reporting, "INTENT_SHA256", reporting.sha256(work / "intent.json"))
    state = {
        "status": "waiting_for_original_full_model_queue",
        "pid": 111,
        "start_ticks": "211",
        **{
            name: intent[name]
            for name in ("cpu_receipt_sha256", "parent_cpu_receipt_sha256", "runner_sha256", "interval")
        },
        "phases": [],
    }
    write(work / "state.json", state)
    monkeypatch.setattr(reporting.queue, "require_cpu", lambda *args: cpu)
    return tmp_path, work, cpu_path, state, intent


@pytest.mark.parametrize("fault", ("none", "changed_cpu", "replaced_generation", "changed_interval", "failed"))
def test_actual_lr_journal_retains_wait_or_failure_and_refuses_substituted_acknowledgment(queued, fault):
    root, work, cpu_path, state, intent = queued
    if fault == "changed_cpu":
        write(cpu_path, {"fixture": "substituted CPU evidence"})
    elif fault == "replaced_generation":
        intent.update(pid=222, start_ticks="322", generation=generation(222, 1, intent["command"]))
        state.update(pid=222, start_ticks="322")
        write(work / "intent.json", intent)
    elif fault == "changed_interval":
        state["interval"] = 60
    elif fault == "failed":
        state.update(status="needs_attention", error="Original lower-LR terminal failed; no relaunch")
    write(work / "state.json", state)
    text = "\n".join(reporting.lines(root, lambda value: value))
    if fault in ("changed_cpu", "replaced_generation", "changed_interval"):
        assert "evidence refused" in text and "0/4 completed phases" not in text
    else:
        assert "0/4 completed phases" in text
        assert state.get("error", state["status"]) in text
    summary = journal.research_summary(text + "\n" + journal.END, datetime.now(UTC))
    assert "NS Actual-LR Validation Queue" in summary
    assert not reporting.queue.contract.locations(root)[0].exists()


@pytest.mark.parametrize("fault", ("none", "changed_log", "changed_decision", "changed_verifier"))
def test_actual_lr_journal_retains_terminal_rejections_and_refuses_changed_evidence(queued, fault):
    root, work, _, state, _ = queued
    helpers = reporting.queue.contract.locations(root)[1]
    helpers.mkdir(parents=True)
    verifier = helpers / "ns_lr_full_model_gpu_controller.py"
    verifier.write_text("# accepted terminal verifier fixture\n")
    commands = reporting.queue.phases(root)
    commands.append(
        (
            "terminal",
            [
                reporting.queue.contract.PYTHON,
                "-P",
                str(verifier),
                "--root",
                str(root),
                "--phase",
                "terminal",
                "--out",
                str(root / reporting.queue.TERMINAL),
            ],
            helpers,
        )
    )
    for index, (phase, argv, cwd) in enumerate(commands):
        own = generation(300 + index, state["pid"], argv)
        log = work / f"{phase}.log"
        log.write_text("Retained external completed phase\n")
        state["phases"].append(
            {
                "phase": phase,
                "command": argv,
                "cwd": str(cwd),
                "spawn_status": "acknowledged",
                "pid": own["pid"],
                "start_ticks": own["start_ticks"],
                "generation": own,
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "marked_descendants_absent": True,
                "log": str(log),
                "log_sha256": reporting.sha256(log),
            }
        )
    terminal_path = root / reporting.queue.TERMINAL
    terminal = {
        "status": reporting.queue.controller.TERMINAL_STATUS,
        "binding_sha256": "original-issued-binding",
        "controller_pid_absent": 200,
        "verifier_sha256": reporting.sha256(verifier),
        "evidence_sha256": {},
        "accepted_cases": ["muont"],
        "rejected_cases": ["fused"],
        "all_cases_accepted": False,
    }
    write(terminal_path, terminal)
    state.update(
        status="actual_lr_terminal_reconstructed",
        terminal=str(terminal_path),
        terminal_sha256=reporting.sha256(terminal_path),
        issued_binding_sha256=terminal["binding_sha256"],
        issued_controller={"pid": 200},
        **{name: terminal[name] for name in reporting.DECISIONS},
    )
    if fault == "changed_log":
        (work / "freeze.log").write_text("Changed completed phase\n")
    elif fault == "changed_decision":
        state.update(accepted_cases=["muont", "fused"], rejected_cases=[], all_cases_accepted=True)
    elif fault == "changed_verifier":
        verifier.write_text("# substituted verifier fixture\n")
    write(work / "state.json", state)
    if fault == "none":
        assert reporting.evidence(root)[-1]["all_cases_accepted"] is False
        text = "\n".join(reporting.lines(root, lambda value: value))
        summary = journal.research_summary(text + "\n" + journal.END, datetime.now(UTC))
        assert "1 accepted and 1 rejected cases" in summary
        assert "Saved all-case predicate: `False`" in summary
        assert "No production or efficiency claim" in summary
    else:
        with pytest.raises(ValueError):
            reporting.evidence(root)
        assert "evidence refused" in "\n".join(reporting.lines(root, lambda value: value))
