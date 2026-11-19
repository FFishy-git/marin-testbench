# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate saved diagnostic reporting with CUDA hidden and immutable execution untouched."""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import rational_continuation_reporting as continuation_reporting
import root_preparation_reporting as reporting
import root_stage_precision_reporting as precision_reporting
import root_start_gpu_reporting as start_reporting
import root_start_rational_reporting as rational_reporting
from root_preparation_controller import CPU_STATUS, cpu_command
from selected_lr_release_owner import read, sha256

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
ACCEPTED = ROOT / "run-preparation/root-executor-cpu/stable-source-20261118T1313"


@pytest.mark.parametrize(
    "fault", ("none", "boolean", "log", "source", "xml", "xml_coverage", "fixture", "archive", "parent")
)
def test_saved_executor_reporting_denies_substituted_evidence_without_mutating_original(tmp_path, fault):
    out = tmp_path / "executor"
    shutil.copytree(ACCEPTED, out)
    path = out / "receipt.json"
    receipt = read(path)
    original_hash = sha256(ACCEPTED / "receipt.json")
    receipt.update(helpers=str(out / "tracking"), cwd=str(out), command=cpu_command(out))
    for name, filename in (("log", "controller.log"), ("tests", "tests.xml"), ("archive", "source.tar")):
        receipt[name] = str(out / filename)
    if fault == "boolean":
        receipt["exit_code"] = False
    elif fault == "log":
        (out / "controller.log").write_text("changed\n")
    elif fault == "source":
        (out / "tracking/root_preparation_probe.py").write_text("# changed\n")
    elif fault == "xml":
        (out / "tests.xml").write_text("<invalid")
    elif fault == "xml_coverage":
        tests = out / "tests.xml"
        tests.write_text(tests.read_text().replace("test_root_preparation_controller", "missing_controller"))
        receipt["tests_sha256"] = sha256(tests)
    elif fault == "fixture":
        receipt["fixtures_sha256"].pop(next(iter(receipt["fixtures_sha256"])))
    elif fault == "archive":
        (out / "source.tar").write_bytes(b"changed")
    elif fault == "parent":
        receipt["candidate_cpu_receipt_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt))
    if fault == "none":
        value, counts = reporting.cpu_evidence(ROOT, path)
        assert value["status"] == CPU_STATUS and counts == {"tests": 133, "failures": 0, "errors": 0, "skipped": 0}
    else:
        with pytest.raises((ValueError, KeyError)):
            reporting.cpu_evidence(ROOT, path)
    assert sha256(ACCEPTED / "receipt.json") == original_hash


@pytest.mark.parametrize("name", ("hardened-20261118T1305", "fixture-closure-20261118T1306"))
def test_reporting_preserves_failed_archive_receipts_without_claiming_acceptance(name):
    receipt, counts = reporting.cpu_evidence(ROOT, ROOT / f"run-preparation/root-executor-cpu/{name}/receipt.json")
    assert receipt["status"] == "needs_attention"
    if name.startswith("hardened"):
        assert counts["errors"] == 15 and receipt["exit_code"] == 1
    else:
        assert counts["errors"] == 0 and receipt["exit_code"] == 0
        assert receipt["error"] == "Executor source changed during CPU verification"


def test_no_saved_receipts_produce_no_live_claims(tmp_path):
    assert reporting.executor_lines(tmp_path, str) == []


@pytest.mark.parametrize("fault", ("none", "coordinates", "command", "log_alias", "boolean", "cold_log_alias", "owner"))
def test_saved_gpu_reporting_requires_bound_command_coordinates_logs_and_owner(tmp_path, monkeypatch, fault):
    path = reporting.locations(ROOT)[4]
    receipt = read(path)
    receipt["gpu_commands"] = receipt["gpu_commands"][:1]
    assert len(receipt["gpu_commands"]) == 1 and len(receipt["cold_probes"]) == 2
    entry = receipt["gpu_commands"][0]
    if fault == "coordinates":
        entry["protocol"] = "trace"
    elif fault == "command":
        entry["command"] = receipt["commands"][1]
    elif fault == "boolean":
        entry["exit_code"] = False
    elif fault == "owner":
        receipt["owner"]["controller_start_ticks"] = "different-generation"
    elif fault in ("log_alias", "cold_log_alias"):
        selected = entry if fault == "log_alias" else receipt["cold_probes"][0]
        alias = tmp_path / "same-bytes.log"
        shutil.copy2(selected["log"], alias)
        selected["log"] = str(alias)
        assert sha256(alias) == selected["log_sha256"]
    substituted = tmp_path / "gpu.json"
    substituted.write_text(json.dumps(receipt))
    snapshot_hash = sha256(substituted)

    def read_snapshot(source):
        return read(substituted if source == path else source)

    monkeypatch.setattr(reporting, "read", read_snapshot)
    if fault == "none":
        frozen, intent, state, gpu = reporting.execution_evidence(ROOT)
        assert gpu["gpu_commands"][0]["command"] == frozen["commands"][0]
        assert gpu["owner"]["controller_pid"] == state["controller_pid"] == intent["pid"]
    else:
        with pytest.raises(ValueError):
            reporting.execution_evidence(ROOT)
    assert sha256(substituted) == snapshot_hash


@pytest.mark.parametrize("fault", ("none", "completed_without_result", "worker_generation", "cpu_receipt"))
def test_failed_reconstruction_reporting_keeps_original_failure_and_denies_false_completion(
    tmp_path, monkeypatch, fault
):
    work = ROOT / reporting.FAILED_RECONSTRUCTION_DIRECTORY
    intent, state = read(work / "intent.json"), read(work / "state.json")
    if fault == "completed_without_result":
        state.update(status="reconstructed", exit_code=0, result_status=reporting.FAILED_RECONSTRUCTION_STATUS)
        state["result_sha256"] = "0" * 64
    elif fault == "worker_generation":
        state["start_ticks"] = "different-generation"
    elif fault == "cpu_receipt":
        state["cpu_receipt_sha256"] = "0" * 64
    substituted = tmp_path / "saved-state.json"
    substituted.write_text(json.dumps(state))
    original = reporting.locations(ROOT)[4]
    original_hash = sha256(original)

    def read_snapshot(path):
        return read(substituted if path == work / "state.json" else path)

    monkeypatch.setattr(reporting, "read", read_snapshot)
    if fault == "none":
        acknowledgment, saved, result = reporting.failed_reconstruction_evidence(ROOT)
        assert saved == read(work / "state.json")
        assert bool(result) is (saved["status"] == "reconstructed")
        assert acknowledgment["pid"] == intent["pid"] == saved["pid"]
        assert read(original)["status"] == "needs_attention"
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            reporting.failed_reconstruction_evidence(ROOT)
    assert sha256(original) == original_hash


@pytest.mark.parametrize("fault", ("none", "original_failure", "verifier", "chronology", "predicate"))
def test_terminal_failed_reconstruction_reporting_requires_bound_result_and_keeps_failure(tmp_path, monkeypatch, fault):
    work = ROOT / reporting.FAILED_RECONSTRUCTION_DIRECTORY
    intent, state = read(work / "intent.json"), read(work / "state.json")
    _, _, failed, gpu = reporting.execution_evidence(ROOT)
    cpu = read(ROOT / reporting.FAILED_RECONSTRUCTION_CPU)
    result_path = tmp_path / "result.json"
    result_time = datetime.fromisoformat(intent["time"]) + timedelta(seconds=1)
    required = {
        reporting.locations(ROOT)[0],
        reporting.locations(ROOT)[2],
        reporting.locations(ROOT)[4],
        reporting.locations(ROOT)[1].parent / "controller.json",
        reporting.locations(ROOT)[1].parent / "controller.log",
        Path(failed["log"]),
    }
    result = {
        "status": reporting.FAILED_RECONSTRUCTION_STATUS,
        "original_status": failed["status"],
        "original_failure": gpu["error"],
        "terminal_time": failed["time"],
        "binding_sha256": state["binding_sha256"],
        "verifier_sha256": cpu["helpers_sha256"]["root_preparation_reconstruction.py"],
        "gpu_commands": gpu["gpu_commands"],
        "tolerance": gpu["tolerance"],
        "controller_pid_absent": failed["controller_pid"],
        "validator_pid_absent": failed["child_pid"],
        "marked_descendants_absent": True,
        "evidence_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "analysis": {"gates": {"prepared_all_rank_continuation": False}, "diagnostic_intervention_accepted": False},
        "time": result_time.isoformat(),
    }
    if fault == "original_failure":
        result["original_status"] = "diagnostic_complete"
    elif fault == "verifier":
        result["verifier_sha256"] = "0" * 64
    elif fault == "chronology":
        result["time"] = (result_time + timedelta(seconds=10)).isoformat()
    elif fault == "predicate":
        result["analysis"]["diagnostic_intervention_accepted"] = True
    result_path.write_text(json.dumps(result))
    state.update(
        status="reconstructed",
        result_status=reporting.FAILED_RECONSTRUCTION_STATUS,
        result_sha256=sha256(result_path),
        exit_code=0,
        time=(result_time + timedelta(seconds=1)).isoformat(),
        out=str(result_path),
    )
    intent["out"] = str(result_path)
    state_path, intent_path = tmp_path / "state.json", tmp_path / "intent.json"
    state_path.write_text(json.dumps(state))
    intent_path.write_text(json.dumps(intent))
    original_hash = sha256(reporting.locations(ROOT)[4])

    def read_snapshot(path):
        replaced = {work / "state.json": state_path, work / "intent.json": intent_path}
        return read(replaced.get(path, path))

    monkeypatch.setattr(reporting, "read", read_snapshot)
    monkeypatch.setattr(reporting, "FAILED_RECONSTRUCTION_RESULT", str(result_path))
    if fault == "none":
        _, saved, accepted = reporting.failed_reconstruction_evidence(ROOT)
        assert saved["status"] == "reconstructed"
        assert accepted["original_status"] == "needs_attention"
        assert accepted["analysis"]["diagnostic_intervention_accepted"] is False
    else:
        with pytest.raises(ValueError):
            reporting.failed_reconstruction_evidence(ROOT)
    assert sha256(reporting.locations(ROOT)[4]) == original_hash


@pytest.mark.parametrize("fault", ("none", "retargeted_owner", "duplicate_check", "false_launch"))
def test_root_stage_queue_reporting_preserves_waiting_state_and_denies_false_launch(tmp_path, monkeypatch, fault):
    path = ROOT / reporting.ROOT_STAGE_QUEUE / "state.json"
    state = read(path)
    if fault == "retargeted_owner":
        state["predecessor_pid"] = 42
    elif fault == "duplicate_check":
        state.update(
            status="preparing_diagnostic",
            commands=[{"phase": phase} for phase in ("seal-predecessor", "freeze", "check", "check")],
        )
    elif fault == "false_launch":
        state.update(status="diagnostic_controller_acknowledged", commands=[])
    snapshot = tmp_path / "queue-state.json"
    snapshot.write_text(json.dumps(state))
    original = sha256(path)

    def read_snapshot(source):
        return read(snapshot if source == path else source)

    monkeypatch.setattr(reporting, "read", read_snapshot)
    if fault == "none":
        intent, saved, diagnostic = reporting.root_stage_queue_evidence(ROOT)
        assert saved["pid"] == intent["pid"] == 1463430
        assert saved == read(path)
        if saved["status"] == "diagnostic_controller_acknowledged":
            assert len(saved["commands"]) == 4
            assert diagnostic["binding_sha256"] == saved["controller"]["binding_sha256"]
        else:
            assert saved["commands"] == []
            assert diagnostic == {}
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            reporting.root_stage_queue_evidence(ROOT)
    assert sha256(path) == original


@pytest.mark.parametrize("fault", ("none", "generation", "false_terminal", "false_precision"))
def test_precision_followup_reporting_denies_false_completion_and_keeps_original_owner(tmp_path, monkeypatch, fault):
    path = ROOT / precision_reporting.FOLLOW_WORK / "state.json"
    state = read(path)
    if fault == "generation":
        state["start_ticks"] = "different-generation"
    elif fault == "false_terminal":
        state.update(status="terminal_reconstructed_precision_denied", terminal_sha256="0" * 64)
    elif fault == "false_precision":
        state.update(status="terminal_and_precision_reconstructed", precision_sha256="0" * 64)
    snapshot = tmp_path / "followup-state.json"
    snapshot.write_text(json.dumps(state))
    original_hash = sha256(path)

    def read_snapshot(source):
        return read(snapshot if source == path else source)

    monkeypatch.setattr(precision_reporting, "read", read_snapshot)
    if fault == "none":
        cpu, follow_cpu, intent, saved, terminal, precision = precision_reporting.evidence(ROOT)
        assert cpu["exit_code"] == follow_cpu["exit_code"] == 0
        assert intent["pid"] == saved["pid"] == 1479257
        assert saved == read(path)
        if saved["status"] == "terminal_and_precision_reconstructed":
            assert len(saved["commands"]) == 2
            assert terminal["intermediates_interpretable"] is True
            assert precision["historical_interpretation_allowed"] is False
        else:
            assert saved["commands"] == []
            assert terminal == precision == {}
    else:
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            precision_reporting.evidence(ROOT)
    assert sha256(path) == original_hash


@pytest.mark.parametrize("fault", ("none", "false_accuracy", "false_crossed_exact", "missing_matrix"))
def test_rational_start_reporting_preserves_cpu_limits_and_rejects_rewritten_results(tmp_path, monkeypatch, fault):
    path = ROOT / rational_reporting.RESULT
    result = read(path)
    if fault == "false_accuracy":
        result["matrices"][0]["rational_O_vs_fp64"]["max_absolute_error"] = 0.0
    elif fault == "false_crossed_exact":
        result["provenance"]["d"]["crossed_cpu_starts"][0]["original_O"]["exact"] = False
    elif fault == "missing_matrix":
        result["matrices"].pop()
    snapshot = tmp_path / "rational-result.json"
    snapshot.write_text(json.dumps(result))
    original_hash = sha256(path)

    def read_snapshot(source):
        return read(snapshot if source == path else source)

    monkeypatch.setattr(rational_reporting, "read", read_snapshot)
    monkeypatch.setattr(rational_reporting, "sha256", lambda source: sha256(snapshot if source == path else source))
    if fault == "none":
        snapshot.write_bytes(path.read_bytes())
        cpu, saved = rational_reporting.evidence(ROOT)
        assert cpu["exit_code"] == 0
        assert len(saved["matrices"]) == 32
        assert all(row["rational_O_vs_fp64"]["within_tolerance"] for row in saved["matrices"])
        assert all(row["original_O"]["exact"] for row in saved["provenance"]["d"]["crossed_cpu_starts"])
    else:
        with pytest.raises(ValueError):
            rational_reporting.evidence(ROOT)
    assert sha256(path) == original_hash


@pytest.mark.parametrize("fault", ("none", "false_accuracy", "missing_matrix", "changed_reference"))
def test_independent_reference_reporting_preserves_complete_cpu_result_and_rejects_rewrites(
    tmp_path, monkeypatch, fault
):
    path = ROOT / rational_reporting.REFERENCE
    result = read(path)
    if fault == "false_accuracy":
        result["matrices"][0]["original_O_vs_log_fp64"]["within_tolerance"] = True
    elif fault == "missing_matrix":
        result["matrices"].pop()
    elif fault == "changed_reference":
        result["reference_equivalence_tolerance"]["rtol"] = 1e-5
    snapshot = tmp_path / "independent-reference.json"
    snapshot.write_text(json.dumps(result))
    original_hash = sha256(path)

    def read_snapshot(source):
        return read(snapshot if source == path else source)

    monkeypatch.setattr(rational_reporting, "read", read_snapshot)
    monkeypatch.setattr(rational_reporting, "sha256", lambda source: sha256(snapshot if source == path else source))
    if fault == "none":
        snapshot.write_bytes(path.read_bytes())
        cpu, saved = rational_reporting.independent_reference(ROOT)
        assert cpu["exit_code"] == 0
        assert len(saved["matrices"]) == 32
        assert all(row["rational_O_vs_log_fp64"]["within_tolerance"] for row in saved["matrices"])
        assert all(row["exact"] for group in ("d", "f") for row in saved["provenance"][group]["crossed_CPU_O"])
    else:
        with pytest.raises(ValueError):
            rational_reporting.independent_reference(ROOT)
    assert sha256(path) == original_hash


@pytest.mark.parametrize(
    "fault", ("none", "generation", "false_completion", "boolean_cleanup", "missing_cross", "swapped_report", "analysis")
)
def test_start_accuracy_reporting_requires_complete_bound_results_and_keeps_issued_evidence(
    tmp_path, monkeypatch, fault
):
    _, _, state_path, work = start_reporting.owner.locations(ROOT)
    comparison_path = work / "comparison.json"
    state, comparison = read(state_path), read(comparison_path)
    original_hashes = {path: sha256(path) for path in (state_path, comparison_path)}
    if fault == "generation":
        state["controller_start_ticks"] = "different-generation"
    elif fault == "false_completion":
        state["executions"].pop()
    elif fault == "boolean_cleanup":
        state["executions"][0]["cleanup_exit_code"] = False
    elif fault == "missing_cross":
        comparison["cross_process"].pop()
    elif fault == "swapped_report":
        state["executions"][0]["report"] = state["executions"][1]["report"]
    elif fault == "analysis":
        comparison["processes"][0]["analysis"]["O_fp64_accuracy"][0]["within_tolerance"] = True
    snapshots = {state_path: tmp_path / "state.json", comparison_path: tmp_path / "comparison.json"}
    snapshots[comparison_path].write_text(json.dumps(comparison))
    state["comparison_sha256"] = sha256(snapshots[comparison_path])
    snapshots[state_path].write_text(json.dumps(state))

    def read_snapshot(path):
        return read(snapshots.get(path, path))

    monkeypatch.setattr(start_reporting, "read", read_snapshot)
    monkeypatch.setattr(start_reporting, "sha256", lambda path: sha256(snapshots.get(path, path)))
    monkeypatch.setattr(start_reporting, "TERMINAL", str(tmp_path / "absent-terminal.json"))
    if fault == "none":
        cpu, frozen, intent, saved, result, terminal = start_reporting.evidence(ROOT)
        assert cpu["exit_code"] == 0
        assert saved["status"] == "diagnostic_complete" and len(saved["executions"]) == 2
        assert result["all_cross_process_exact_finite"] is True
        assert result["all_rational_O_within_fp64_tolerance"] is True
        assert intent["pid"] == saved["controller_pid"] and frozen["workers"] == 2
        assert terminal == {}
    else:
        with pytest.raises((ValueError, KeyError)):
            start_reporting.evidence(ROOT)
    assert original_hashes == {path: sha256(path) for path in original_hashes}


@pytest.mark.parametrize("fault", ("none", "restoration", "accuracy", "owner"))
def test_start_terminal_reporting_rejects_rewritten_receipt_without_reexecuting_original(tmp_path, monkeypatch, fault):
    path = ROOT / start_reporting.TERMINAL
    terminal = read(path)
    original_hash = sha256(path)
    if fault == "restoration":
        terminal["keeper"]["pid"] = 42
    elif fault == "accuracy":
        terminal["comparison"]["all_rational_O_within_fp64_tolerance"] = False
    elif fault == "owner":
        terminal["controller_pid_absent"] = 42
    snapshot = tmp_path / "terminal.json"
    snapshot.write_bytes(path.read_bytes() if fault == "none" else json.dumps(terminal).encode())

    def read_snapshot(source):
        return read(snapshot if source == path else source)

    monkeypatch.setattr(start_reporting, "read", read_snapshot)
    monkeypatch.setattr(start_reporting, "sha256", lambda source: sha256(snapshot if source == path else source))
    if fault == "none":
        _, _, _, saved, result, verified = start_reporting.evidence(ROOT)
        assert verified["comparison"] == {key: value for key, value in result.items() if key != "time"}
        assert verified["terminal_time"] == saved["time"]
        assert verified["controller_pid_absent"] == saved["controller_pid"]
    else:
        with pytest.raises(ValueError):
            start_reporting.evidence(ROOT)
    assert sha256(path) == original_hash


@pytest.fixture
def saved_continuation(tmp_path):
    binding, helpers, state_path, work, result_path = continuation_reporting.contract.locations(tmp_path)
    frozen = read(continuation_reporting.contract.locations(ROOT)[0])
    binding.parent.mkdir(parents=True)
    helpers.mkdir(parents=True)
    work.mkdir(parents=True)
    intent = {
        "status": "acknowledged",
        "marker": "fixture-controller",
        "binding_sha256": continuation_reporting.BINDING_SHA256,
        "command": frozen["controller_command"],
        "pid": 123,
        "start_ticks": "456",
        "generation": {"pid": 123, "start_ticks": "456", "command": frozen["controller_command"]},
    }
    state = {
        "status": "validating",
        "binding_sha256": continuation_reporting.BINDING_SHA256,
        "controller_pid": 123,
        "controller_start_ticks": "456",
        "allocation_predecessor": frozen["allocation_predecessor"],
        "spawn_status": "acknowledged",
        "child_marker": "fixture-validator",
        "command": frozen["validator_command"],
        "child_pid": 124,
        "child_start_ticks": "457",
        "child_generation": {
            "pid": 124,
            "start_ticks": "457",
            "parent": 123,
            "command": frozen["validator_command"],
        },
    }
    outcomes = []
    for coordinate, command in zip(continuation_reporting.contract.coordinates(), frozen["commands"], strict=True):
        variant, protocol, mode = coordinate
        log = work / f"{variant}_{protocol}_{mode}.log"
        log.write_text(f"{variant}/{protocol}/{mode}\n")
        outcomes.append(
            {
                "variant": variant,
                "protocol": protocol,
                "mode": mode,
                "command": command,
                "exit_code": 0,
                "log": str(log),
                "log_sha256": sha256(log),
            }
        )
    result = frozen["plan"] | {
        "status": "diagnosing",
        "binding_sha256": continuation_reporting.BINDING_SHA256,
        "owner": {"controller_pid": 123, "controller_start_ticks": "456"},
        "gpu_commands": outcomes,
    }
    intent_path = helpers.parent / "controller.json"
    binding.write_text(json.dumps(frozen))
    intent_path.write_text(json.dumps(intent))
    return tmp_path, frozen, state_path, result_path, state, result


@pytest.mark.parametrize("fault", ("none", "reordered", "boolean_exit", "log", "owner", "false_completion"))
def test_continuation_reporting_authenticates_partial_history_and_denies_false_success(saved_continuation, fault):
    root, frozen, state_path, result_path, state, result = saved_continuation
    result["gpu_commands"] = result["gpu_commands"][:2]
    if fault == "reordered":
        result["gpu_commands"].reverse()
    elif fault == "boolean_exit":
        result["gpu_commands"][0]["exit_code"] = False
    elif fault == "log":
        Path(result["gpu_commands"][0]["log"]).write_text("replaced\n")
    elif fault == "owner":
        state["child_generation"]["parent"] = 999
    elif fault == "false_completion":
        state["status"] = "diagnostic_complete"
    result_path.write_text(json.dumps(result))
    state_path.write_text(json.dumps(state))
    original = {path: sha256(path) for path in (state_path, result_path)}
    if fault == "none":
        intent, saved, partial = continuation_reporting.execution_evidence(root, frozen)
        assert saved["status"] == "validating"
        assert len(partial["gpu_commands"]) == 2
        assert partial["owner"]["controller_pid"] == intent["pid"]
    else:
        with pytest.raises(ValueError):
            continuation_reporting.execution_evidence(root, frozen)
    assert original == {path: sha256(path) for path in original}


@pytest.mark.parametrize("restoration", (0, False))
def test_continuation_reporting_requires_integer_restoration_for_completed_history(saved_continuation, restoration):
    root, frozen, state_path, result_path, state, result = saved_continuation
    result["status"] = "completed_rational_start_full_model_diagnostic_not_production"
    result_path.write_text(json.dumps(result))
    state.update(
        status="diagnostic_complete",
        exit_code=0,
        cleanup_exit_code=0,
        keeper_restore=restoration,
        keeper_release_complete=True,
        marked_descendants_absent=True,
        result_sha256=sha256(result_path),
    )
    state_path.write_text(json.dumps(state))
    if type(restoration) is int:
        _, saved, complete = continuation_reporting.execution_evidence(root, frozen)
        assert saved["status"] == "diagnostic_complete"
        assert len(complete["gpu_commands"]) == 12
        assert all(type(entry["exit_code"]) is int for entry in complete["gpu_commands"])
    else:
        with pytest.raises(ValueError):
            continuation_reporting.execution_evidence(root, frozen)


@pytest.mark.parametrize("substituted_generation", (False, True))
def test_continuation_observer_requires_real_process_generation_and_recognizes_exit(tmp_path, substituted_generation):
    command = [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"]
    marker = "test-continuation-observer"
    environment = os.environ | continuation_reporting.contract.ENVIRONMENT | {"TEST_CONTINUATION_OBSERVER": marker}
    with subprocess.Popen(
        command, cwd=tmp_path, env=environment, stdin=subprocess.PIPE, start_new_session=True
    ) as child:
        expected = continuation_reporting.handoff.observed(child.pid)
        assert expected is not None
        if substituted_generation:
            expected["start_ticks"] = "different-generation"
            with pytest.raises(RuntimeError):
                continuation_reporting.handle(expected, tmp_path, "TEST_CONTINUATION_OBSERVER", marker)
        else:
            live = continuation_reporting.handle(expected, tmp_path, "TEST_CONTINUATION_OBSERVER", marker)
            assert live["status"] == "live_exact_generation"
            assert live["observed"]["pid"] == child.pid
        child.communicate(timeout=10)
    exited = continuation_reporting.handle(expected, tmp_path, "TEST_CONTINUATION_OBSERVER", marker)
    assert exited["status"] == "absent"


@pytest.mark.parametrize("wrong_device", (False, True))
def test_continuation_observer_records_absent_owners_and_rejects_changed_physical_allocation(
    saved_continuation, monkeypatch, wrong_device
):
    root, frozen, state_path, result_path, state, result = saved_continuation
    binding, helpers, _, _, _ = continuation_reporting.contract.locations(root)
    helper = helpers / "observer-fixture.py"
    helper.write_text("# source-bound observation fixture\n")
    frozen["helpers_sha256"] = {helper.name: sha256(helper)}
    binding.write_text(json.dumps(frozen))
    monkeypatch.setattr(continuation_reporting, "BINDING_SHA256", sha256(binding))
    intent_path = helpers.parent / "controller.json"
    intent = read(intent_path)
    intent["binding_sha256"] = state["binding_sha256"] = result["binding_sha256"] = sha256(binding)
    result["gpu_commands"] = result["gpu_commands"][:1]
    intent_path.write_text(json.dumps(intent))
    state_path.write_text(json.dumps(state))
    result_path.write_text(json.dumps(result))
    proc = root / "proc"
    proc.mkdir()
    monkeypatch.setattr(continuation_reporting.handoff, "PROC_ROOT", proc)
    devices = frozen["allocation_predecessor"]["restored_keeper"]["devices"]

    def device_response(command, **kwargs):
        if command[1] == "--query-gpu=index,uuid,name":
            rows = [list(row) for row in devices]
            if wrong_device:
                rows[0][1] = "GPU-unexpected"
            return "\n".join(", ".join(row) for row in rows)
        return f"999, {devices[0][1]}\n"

    monkeypatch.setattr(continuation_reporting.subprocess, "check_output", device_response)
    if wrong_device:
        with pytest.raises(RuntimeError):
            continuation_reporting.live_observation(root)
    else:
        observed = continuation_reporting.live_observation(root)
        assert observed["retained_commands"] == 1
        assert {name: owner["status"] for name, owner in observed["owners"].items()} == {
            "controller": "absent",
            "validator": "absent",
        }
        assert observed["occupied"] == [{"pid": 999, "gpu_uuid": devices[0][1], "active_training_rank": False}]
        assert observed["active_ranks"] == []


@pytest.mark.parametrize("fault", ("none", "generation", "false_terminal", "source"))
def test_read_only_followup_reporting_requires_original_owner_and_preserves_pending_receipt(
    tmp_path, monkeypatch, fault
):
    source = ROOT / continuation_reporting.FOLLOWUP_WORK / "state.json"
    original_state = read(source)
    state = {
        key: original_state[key]
        for key in ("pid", "start_ticks", "runner_sha256", "cpu_receipt_sha256", "binding_sha256", "interval", "time")
    } | {"status": "waiting_for_original_owners"}
    if fault == "generation":
        state["start_ticks"] = "different-generation"
    elif fault == "false_terminal":
        state["status"] = "terminal_reconstructed"
    elif fault == "source":
        state["runner_sha256"] = "0" * 64
    snapshot = tmp_path / "state.json"
    snapshot.write_text(json.dumps(state))
    original = sha256(source)

    def read_snapshot(path):
        return read(snapshot if path == source else path)

    monkeypatch.setattr(continuation_reporting, "read", read_snapshot)
    if fault == "none":
        cpu, intent, saved, result = continuation_reporting.followup_evidence(ROOT)
        assert cpu["test_count"] == 8 and cpu["exit_code"] == 0
        assert intent["pid"] == saved["pid"] == 1676886
        assert saved["status"] == "waiting_for_original_owners"
        assert result == {}
    else:
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            continuation_reporting.followup_evidence(ROOT)
    assert sha256(source) == original
