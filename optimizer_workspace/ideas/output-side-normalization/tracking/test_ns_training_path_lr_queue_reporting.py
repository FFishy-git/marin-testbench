# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep accepted queue status and retained rejections visible in both research journals."""

from datetime import UTC, datetime
from pathlib import Path

import ns_training_path_lr_queue_reporting as reporting
import pytest
import update_endeavor as journal
from ns_training_path_lr_gpu_test_support import generation
from selected_lr_release_owner import read, sha256
from watch_wave5 import write


@pytest.fixture
def queued(tmp_path, monkeypatch):
    work, cpu_path = tmp_path / reporting.queue.WORK, tmp_path / reporting.CPU
    work.mkdir(parents=True)
    cpu_path.parent.mkdir(parents=True)
    write(cpu_path, {"fixture": "accepted external CPU boundary"})
    monkeypatch.setattr(reporting, "CPU_SHA256", sha256(cpu_path))
    runner = sha256(Path(reporting.queue.__file__))
    cpu = {
        "parent_cpu_receipt_sha256": reporting.queue.PARENT_SHA256,
        "command": reporting.queue.cpu_command(cpu_path.parent),
        "environment": reporting.queue.ENVIRONMENT,
        "scope": reporting.queue.SCOPE,
        "helpers_sha256": {reporting.queue.NEW_HELPERS[0]: runner},
    }
    argv = reporting.queue.command(tmp_path, cpu_path)
    intent = {
        "status": "acknowledged",
        "pid": 111,
        "start_ticks": "1111",
        "command": argv,
        "marker": "original",
        "generation": generation(111, 1, argv),
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": sha256(cpu_path),
        "runner_sha256": runner,
        "parent_cpu_receipt_sha256": reporting.queue.PARENT_SHA256,
        "interval": 900,
        "scope": reporting.queue.SCOPE,
    }
    write(work / "intent.json", intent)
    monkeypatch.setattr(reporting, "INTENT_SHA256", sha256(work / "intent.json"))
    state = {
        "status": "waiting_for_original_training_path_owners",
        "pid": 111,
        "start_ticks": "1111",
        **{
            name: intent[name]
            for name in ("cpu_receipt_sha256", "parent_cpu_receipt_sha256", "runner_sha256", "interval", "scope")
        },
        "phases": [],
        "original_observations": 1,
        "original_observation": {
            "ready": False,
            "status": "waiting_for_original_training_path_owners",
            "owners": {"controller": {"status": "live_exact_generation"}},
        },
    }
    write(work / "state.json", state)
    write(work / "original-observation-000000.json", state["original_observation"])
    monkeypatch.setattr(reporting, "accepted_archive", lambda *args: cpu)
    return tmp_path, work, cpu_path, state, intent


@pytest.mark.parametrize(
    "fault", ("none", "changed_cpu", "replaced_generation", "changed_observation", "reobserved", "failed")
)
def test_training_path_lr_journal_retains_wait_failure_and_refuses_substituted_receipts(queued, fault):
    root, work, cpu_path, state, intent = queued
    if fault == "changed_cpu":
        write(cpu_path, {"fixture": "substituted evidence"})
    elif fault == "replaced_generation":
        intent.update(pid=222, start_ticks="1222", generation=generation(222, 1, intent["command"]))
        state.update(pid=222, start_ticks="1222")
        write(work / "intent.json", intent)
    elif fault == "changed_observation":
        write(work / "original-observation-000000.json", {"ready": True})
    elif fault == "reobserved":
        state["original_observation"].pop("time", None)
    elif fault == "failed":
        state.update(status="needs_attention", error="Retained predecessor failure; one-shot phase not retried")
    write(work / "state.json", state)
    text = "\n".join(reporting.lines(root, lambda value: value))
    if fault in ("changed_cpu", "replaced_generation", "changed_observation"):
        assert "evidence refused" in text and "0/6 completed phases" not in text
    else:
        assert "0/6 completed phases" in text and state.get("error", state["status"]) in text
    summary = journal.research_summary(text + "\n" + journal.END, datetime.now(UTC))
    assert "NS Actual-LR Training-Path Queue" in summary
    assert not reporting.queue.contract.locations(root)[0].exists()


@pytest.mark.parametrize("fault", ("none", "changed_log", "changed_decision", "changed_outcome"))
def test_training_path_lr_journal_retains_original_and_actual_lr_terminal_rejections(queued, fault):
    root, work, _, state, _ = queued
    state["original_decisions"] = {
        "accepted_cases": ["original-control"],
        "rejected_cases": ["original-fused"],
        "all_cases_accepted": False,
    }
    seal = root / reporting.queue.predecessor.SEAL
    write(seal, {"original_decisions": state["original_decisions"]})
    state["predecessor_seal_sha256"] = sha256(seal)
    state["issued_controller"] = {
        "binding_sha256": "issued-binding",
        "pid": 200,
        "start_ticks": "1200",
        "marker": "issued-controller",
    }
    state["issued_binding_sha256"] = "issued-binding"
    helpers = reporting.queue.contract.locations(root)[1]
    helpers.mkdir(parents=True)
    verifier = helpers / "ns_training_path_lr_gpu_controller.py"
    verifier.write_text("# accepted terminal external fixture\n")
    for index, phase in enumerate(reporting.PHASES):
        argv, cwd = reporting.phase_command(root, phase, state)
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
                "log_sha256": sha256(log),
            }
        )
    terminal, attempt, outcome, _ = reporting.queue.controller.terminal_paths(root, state["issued_controller"])
    write(attempt, {"status": "reconstruction_claimed"})
    result = {
        "status": reporting.queue.controller.TERMINAL_STATUS,
        "binding_sha256": state["issued_binding_sha256"],
        "controller_pid_absent": state["issued_controller"]["pid"],
        "controller_start_ticks": state["issued_controller"]["start_ticks"],
        "verifier_sha256": sha256(verifier),
        "predecessor_acceptance": {"seal_sha256": state["predecessor_seal_sha256"]},
        "production_authorized": False,
        "reconstruction_intent_sha256": sha256(attempt),
        "evidence_sha256": {},
        "accepted_cases": ["actual-lr-control"],
        "rejected_cases": ["actual-lr-fused"],
        "all_cases_accepted": False,
    }
    write(terminal, result)
    write(
        outcome,
        {
            "status": "complete",
            "output": str(terminal),
            "terminal_sha256": sha256(terminal),
            "reconstruction_intent_sha256": sha256(attempt),
        },
    )
    state.update(
        status="actual_lr_training_path_terminal_reconstructed",
        terminal=str(terminal),
        terminal_sha256=sha256(terminal),
        terminal_outcome_sha256=sha256(outcome),
        **{name: result[name] for name in reporting.DECISIONS},
    )
    if fault == "changed_log":
        (work / "freeze.log").write_text("Changed external execution log\n")
    elif fault == "changed_decision":
        state["rejected_cases"] = []
    elif fault == "changed_outcome":
        write(outcome, read(outcome) | {"status": "needs_attention"})
        state["terminal_outcome_sha256"] = sha256(outcome)
    write(work / "state.json", state)
    before = sha256(terminal)
    text = "\n".join(reporting.lines(root, lambda value: value))
    if fault == "none":
        assert "6/6 completed phases" in text and "1 accepted, 1 rejected" in text
        _, _, saved, retained = reporting.evidence(root)
        assert saved["original_decisions"]["rejected_cases"] == ["original-fused"]
        assert retained["rejected_cases"] == ["actual-lr-fused"] and retained["production_authorized"] is False
    else:
        assert "evidence refused" in text and "Saved canonical actual-LR terminal" not in text
    assert sha256(terminal) == before
