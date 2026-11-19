# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep saved full-model queue failures and decisions in both journal reports."""

from datetime import UTC, datetime

import ns_full_model_queue_reporting as reporting
import pytest
import update_endeavor as journal
from watch_wave5 import write


@pytest.mark.parametrize(
    "fault", ("none", "changed_cpu", "changed_intent", "replaced_generation", "changed_interval", "failed")
)
def test_saved_full_model_queue_reports_wait_or_failure_and_refuses_changed_acknowledgment(tmp_path, monkeypatch, fault):
    work = tmp_path / reporting.queue.WORK
    work.mkdir(parents=True)
    cpu_path = tmp_path / reporting.CPU
    cpu_path.parent.mkdir(parents=True)
    write(cpu_path, {"fixture": "source-bound CPU boundary"})
    monkeypatch.setattr(reporting, "CPU_SHA256", reporting.sha256(cpu_path))
    cpu = {
        "helpers_sha256": {reporting.queue.NEW_HELPERS[0]: "accepted-runner"},
        "time": "2026-11-19T00:00:00+00:00",
    }
    command = reporting.queue.command(tmp_path, cpu_path)
    intent = {
        "status": "acknowledged",
        "pid": 111,
        "start_ticks": "111",
        "command": command,
        "marker": "original",
        "generation": {"pid": 111, "start_ticks": "111", "command": command, "group": 111, "session": 111},
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": reporting.sha256(cpu_path),
        "runner_sha256": "accepted-runner",
        "parent_cpu_receipt_sha256": reporting.queue.PARENT_SHA256,
        "interval": reporting.queue.INTERVAL,
    }
    state = {
        "status": "needs_attention" if fault == "failed" else "waiting_for_original_ns_queue",
        "pid": 111,
        "start_ticks": "111",
        **{
            name: intent[name]
            for name in ("cpu_receipt_sha256", "runner_sha256", "parent_cpu_receipt_sha256", "interval")
        },
        "phases": [],
    }
    if fault == "failed":
        state["error"] = "Original NS terminal failed; not relaunched"
    if fault == "changed_intent":
        intent["generation"]["command"] = ["mutable", "substituted"]
    if fault == "changed_interval":
        state["interval"] = 60
    write(work / "intent.json", intent)
    monkeypatch.setattr(reporting, "INTENT_SHA256", reporting.sha256(work / "intent.json"))
    write(work / "state.json", state)
    if fault == "changed_cpu":
        write(cpu_path, {"fixture": "substituted CPU boundary"})
    if fault == "replaced_generation":
        intent.update(pid=222, start_ticks="222")
        intent["generation"].update(pid=222, start_ticks="222", group=222, session=222)
        state.update(pid=222, start_ticks="222")
        write(work / "intent.json", intent)
        write(work / "state.json", state)
    monkeypatch.setattr(reporting.queue, "require_cpu", lambda *args: cpu)
    text = "\n".join(reporting.lines(tmp_path, lambda value: value))
    if fault in ("changed_cpu", "changed_intent", "replaced_generation", "changed_interval"):
        assert "evidence refused" in text
        assert "0/4 completed phases" not in text
    else:
        assert "0/4 completed phases" in text
        if fault == "failed":
            assert state["error"] in text
        else:
            assert "waiting_for_original_ns_queue" in text
    summary = journal.research_summary(text + "\n" + journal.END, datetime.now(UTC))
    assert "NS Full-Model Validation Queue" in summary
    assert "No production" in summary or "evidence refused" in summary
    assert not reporting.contract.locations(tmp_path)[0].exists()
