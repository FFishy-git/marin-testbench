# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise saved NS reporting without numerical execution or unverified launch claims."""

import copy
import tarfile
from datetime import UTC, datetime

import ns_budget_reporting as reporting
import pytest
import update_endeavor as journal
from watch_wave5 import write


def test_ns_saved_queue_report_denies_changed_intent_without_launch_or_tensor_execution(tmp_path, monkeypatch):
    root = reporting.owner.execution.replay.ROOT
    cpu, inherited, intent, _ = reporting.queue_evidence(root)
    work = tmp_path / reporting.queue.WORK
    work.mkdir(parents=True)
    write(work / "intent.json", intent | {"interval": 60})
    monkeypatch.setattr(
        reporting, "cpu_evidence", lambda root, path, *args: cpu if path == root / reporting.QUEUE_CPU else inherited
    )
    with pytest.raises(ValueError):
        reporting.queue_evidence(tmp_path)
    assert not reporting.owner.locations(tmp_path)[0].exists()


@pytest.fixture
def saved_diagnostic(tmp_path):
    binding, helpers, state_path, work = reporting.owner.locations(tmp_path)
    helpers.mkdir(parents=True)
    work.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    source = helpers / "ns_budget_gpu_controller.py"
    source.write_text("# saved report fixture\n")
    hashes = {source.name: reporting.sha256(source)}
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        bundle.add(source, arcname=f"tracking/{source.name}", recursive=False)
    keeper = {"pid": 900, "generation": {}, "devices": [["0", "GPU-cpu", "CPU fixture"]]}
    frozen = {
        "status": reporting.owner.STATUS,
        "root": str(tmp_path),
        "helpers": str(helpers),
        "work": str(work),
        "cpu_receipt": str(tmp_path / reporting.queue.PARENT),
        "cpu_receipt_sha256": reporting.queue.PARENT_SHA256,
        "helpers_sha256": hashes,
        "controller_command": reporting.owner.controller_command(tmp_path),
        "environment": reporting.owner.ENVIRONMENT,
        "worker_environment": reporting.owner.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": "GPU-cpu"},
        "devices": keeper["devices"],
        "workers": 2,
        "worker_timeout": reporting.owner.WORKER_TIMEOUT,
        "coordinates": [reporting.execution.coordinates(index) for index in (0, 1)],
        "archive": str(archive),
        "archive_sha256": reporting.sha256(archive),
        "predecessor": {"intervention_accepted": False, "restored_keeper": keeper},
    }
    write(binding, frozen)
    generation = {
        "pid": 100,
        "start_ticks": "100",
        "command": frozen["controller_command"],
        "group": 100,
        "session": 100,
    }
    intent = {
        "status": "acknowledged",
        "binding_sha256": reporting.sha256(binding),
        "command": frozen["controller_command"],
        "pid": 100,
        "start_ticks": "100",
        "generation": generation,
    }
    write(helpers.parent / "controller.json", intent)
    entries, processes = [], []
    analysis = {
        "all_ten_step_tpu_phases_reproduce_saved_outputs": False,
        "all_compiled_phases_match_pinned_eager": True,
        "all_graph_phases_match_compiled_call": True,
        "all_same_input_repeats_exact_finite": True,
        "arms": [
            {
                "mode": "tpu",
                "steps": steps,
                "source_pairs": [
                    {"within_tolerance": True, "relative_frobenius_error": 0.125},
                    {"within_tolerance": False, "relative_frobenius_error": 0.25},
                ],
            }
            for steps in (5, 10)
        ],
    }
    timings = [
        {"mode": "tpu", "steps": steps, "source": source, "call_index": call, "median_milliseconds": elapsed}
        for steps, source, call, elapsed in (
            (5, "continuous", 0, 5.0),
            (5, "resumed", 1, 4.0),
            (10, "continuous", 0, 10.0),
            (10, "resumed", 1, 12.0),
        )
    ]
    for index in (0, 1):
        log, report = work / f"process-{index}.log", work / f"process-{index}.json"
        log.write_text("CPU saved reporting fixture\n")
        write(
            report,
            {
                "analysis": analysis,
                "timings": timings,
                "coordinates": reporting.execution.coordinates(index),
                "outputs_sha256": "0" * 64,
            },
        )
        entry = {
            "index": index,
            "command": reporting.owner.worker_command(tmp_path, reporting.sha256(binding), index),
            "controller_pid": 100,
            "controller_start_ticks": "100",
            "pid": 200 + index,
            "start_ticks": str(200 + index),
            "exit_code": 0,
            "cleanup_exit_code": 0,
            "marked_descendants_absent": True,
            "report": str(report),
            "report_sha256": reporting.sha256(report),
            "log": str(log),
            "log_sha256": reporting.sha256(log),
        }
        entry["generation"] = {
            "pid": entry["pid"],
            "start_ticks": entry["start_ticks"],
            "parent": 100,
            "command": entry["command"],
        }
        entries.append(entry)
        processes.append(
            {
                "worker_index": index,
                "report_sha256": entry["report_sha256"],
                "outputs_sha256": "0" * 64,
                "analysis": copy.deepcopy(analysis),
                "timings": copy.deepcopy(timings),
            }
        )
    comparison = {
        "status": "ns_budget_diagnostic_independently_reconstructed_not_continuation_or_production",
        "executions": entries,
        "processes": processes,
        **analysis,
        "historical_controls_interpretable": False,
        "cross_process": {"all_cross_process_exact_finite": True},
    }
    write(work / "comparison.json", comparison)
    state = {
        "status": "diagnostic_complete",
        "binding_sha256": reporting.sha256(binding),
        "predecessor": frozen["predecessor"],
        "controller_pid": 100,
        "controller_start_ticks": "100",
        "executions": entries,
        "comparison_sha256": reporting.sha256(work / "comparison.json"),
        "keeper_release_complete": True,
        "keeper_restore": 0,
        "restored_keeper": keeper,
    }
    write(state_path, state)
    return {"helpers_sha256": hashes}, work


@pytest.mark.parametrize("fault", ("none", "report", "log", "source", "comparison", "predicate"))
def test_ns_saved_diagnostic_report_preserves_negative_result_and_denies_changed_evidence(
    tmp_path, saved_diagnostic, fault
):
    cpu, work = saved_diagnostic
    if fault == "report":
        report = reporting.read(work / "process-1.json")
        report["analysis"]["all_ten_step_tpu_phases_reproduce_saved_outputs"] = True
        write(work / "process-1.json", report)
    elif fault == "log":
        (work / "process-1.log").write_text("changed\n")
    elif fault == "source":
        (reporting.owner.locations(tmp_path)[1] / "ns_budget_gpu_controller.py").write_text("changed\n")
    elif fault in ("comparison", "predicate"):
        record = reporting.read(work / "comparison.json")
        record["historical_controls_interpretable"] = True
        write(work / "comparison.json", record)
        if fault == "predicate":
            state_path = reporting.owner.locations(tmp_path)[2]
            state = reporting.read(state_path)
            state["comparison_sha256"] = reporting.sha256(work / "comparison.json")
            write(state_path, state)
    if fault != "none":
        with pytest.raises(ValueError):
            reporting.diagnostic_evidence(tmp_path, cpu, {})
        return
    _, _, state, comparison, terminal = reporting.diagnostic_evidence(tmp_path, cpu, {})
    assert state["status"] == "diagnostic_complete" and comparison["historical_controls_interpretable"] is False
    assert terminal == {} and not (tmp_path / reporting.queue.TERMINAL).exists()


def test_ns_completed_report_keeps_failed_pairs_and_matches_timing_calls(tmp_path, saved_diagnostic):
    cpu, _ = saved_diagnostic
    comparison = reporting.diagnostic_evidence(tmp_path, cpu, {})[3]
    rows = reporting.completed_lines(comparison)
    assert "| 0 | `tpu` | 5 | 1/2 | 0.125 to 0.25 | 4 to 5 | 2 to 3 |" in rows
    assert "| 1 | `tpu` | 10 | 1/2 | 0.125 to 0.25 | 10 to 12 | 1 to 1 |" in rows
    comparison["processes"][0]["analysis"]["arms"][0]["source_pairs"][1]["relative_frobenius_error"] = None
    assert "| 0 | `tpu` | 5 | 1/2 | nonfinite/unavailable | 4 to 5 | 2 to 3 |" in reporting.completed_lines(comparison)


def test_ns_saved_report_renders_failure_and_survives_research_summary(tmp_path, monkeypatch):
    cpu, inherited, intent, state = reporting.queue_evidence(reporting.owner.execution.replay.ROOT)
    state.update(status="needs_attention", error="preserved failed freeze")
    monkeypatch.setattr(reporting, "queue_evidence", lambda root: (cpu, inherited, intent, state))
    monkeypatch.setattr(reporting, "diagnostic_evidence", lambda *args: ({}, {}, {}, {}, {}))
    (tmp_path / reporting.QUEUE_CPU).parent.mkdir(parents=True)
    (tmp_path / reporting.QUEUE_CPU).write_text("{}")
    lines = reporting.lines(tmp_path, lambda value: value)
    assert "preserved failed freeze" in "\n".join(lines)
    payload = "\n".join((journal.START, *lines, journal.END))
    summary = journal.research_summary(payload, datetime.now(UTC))
    assert "### NS Budget Diagnostic And Queue" in summary
    assert "preserved failed freeze" in summary and "900-second" in summary
