# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise predecessor refusal and file-only sealing before localization CUDA ownership."""

import copy
import json
import os
import signal
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_compiler_localization_predecessor as predecessor
import ns_compiler_localization_predecessor_test_support as support
import prepare_ns_compiler_localization_predecessor as preparation
import pytest
import selected_lr_failure_handoff
from fused_attribution_validation_controller import acknowledged_process
from selected_lr_release_owner import read, sha256


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    manager = tmp_path / "cpu-manager.sh"
    manager.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(selected_lr_failure_handoff, "MANAGER_LOCK", tmp_path / "manager.lock")
    monkeypatch.setattr(selected_lr_failure_handoff, "MANAGER_SOURCE", manager)
    monkeypatch.setattr(selected_lr_failure_handoff, "KEEPER", str(manager))
    root = tmp_path / "root"
    root.mkdir()
    return support.completed(root, monkeypatch)


def test_localization_handoff_preserves_both_rejections_and_never_deserializes_payloads(completed):
    root, restored = completed
    before = {path: sha256(path) for path in root.rglob("*.pt")}
    saved = predecessor.evidence(root)
    assert saved["original_decisions"]["rejected_cases"] == ["original-case-0"]
    assert len(saved["actual_lr_decisions"]["accepted_cases"]) == 53
    assert len(saved["actual_lr_decisions"]["rejected_cases"]) == 1
    assert len(saved["original_checkpoint_payloads"]) == 18 * 2 * 3 * 8
    assert len(saved["actual_lr_checkpoint_payloads"]) == 54 * 2 * 3 * 8
    assert saved["restored_keeper"] == restored
    assert saved["gpu_execution_authorized"] is False and saved["production_authorized"] is False
    assert before == {path: sha256(path) for path in before}
    assert not (root / predecessor.SEAL).exists()


@pytest.mark.parametrize("owner", ("queue", "phase", "controller", "validator", "command"))
def test_live_actual_lr_handles_refuse_localization_without_terminal_mutation(completed, monkeypatch, owner):
    root, _ = completed
    queue = predecessor.queue
    _, helpers, state_path, _, result_path = queue.contract.locations(root)
    work = root / queue.WORK
    queued, intent = read(work / "state.json"), read(work / "intent.json")
    frozen, state = read(queue.contract.locations(root)[0]), read(state_path)
    command = [queue.contract.PYTHON, "-c", "import signal; signal.pause()"]
    contexts = {
        "queue": (work, queue.MARKER, intent["marker"], queue.ENVIRONMENT),
        "phase": (helpers, queue.PHASE_MARKER, "live-terminal-phase", queue.ENVIRONMENT),
        "controller": (
            helpers,
            queue.contract.CONTROLLER_MARKER,
            queued["issued_controller"]["marker"],
            queue.owner.CONTROLLER_ENVIRONMENT | {queue.owner.PARENT_PROOF_MARKER: "external-accepted-parent"},
        ),
        "validator": (
            Path(frozen["candidate"]),
            queue.contract.CHILD_MARKER,
            state["child_marker"],
            queue.contract.ENVIRONMENT,
        ),
        "command": (
            Path(frozen["candidate"]),
            queue.contract.COMMAND_MARKER,
            "live-command",
            queue.contract.ENVIRONMENT,
        ),
    }
    cwd, marker_name, marker, environment = contexts[owner]
    child = subprocess.Popen(
        command, cwd=cwd, env=os.environ | environment | {marker_name: marker}, start_new_session=True
    )
    try:
        own = acknowledged_process(child, command, cwd, marker_name, marker, environment)
        if owner == "queue":
            monkeypatch.setattr(queue, "command", lambda *_: command)
            support.bind_queue_generation(root, monkeypatch, own)
        elif owner == "phase":
            queued["active_phase"] = {
                "phase": "terminal",
                "spawn_status": "acknowledged",
                "pid": own["pid"],
                "generation": own,
                "cwd": str(cwd),
                "marker": marker,
            }
        elif owner == "controller":
            issued = queued["issued_controller"]
            issued.update(pid=own["pid"], start_ticks=own["start_ticks"], command=command, generation=own)
            support.record(helpers.parent / "controller.json", issued)
            frozen["controller_command"] = command
            support.record(queue.contract.locations(root)[0], frozen)
            queued["issued_binding_sha256"] = sha256(queue.contract.locations(root)[0])
            issued["binding_sha256"] = queued["issued_binding_sha256"]
            support.record(helpers.parent / "controller.json", issued)
            monkeypatch.setattr(queue.owner, "controller_command", lambda *_: command)
        elif owner == "validator":
            state["child_generation"] = own
            support.record(state_path, state)
        else:
            result = read(result_path)
            result["gpu_commands"][-1].update(generation=own, marker=marker)
            support.record(result_path, result)
        if owner != "queue":
            support.record(work / "state.json", queued)
        terminal = Path(queued["terminal"])
        digest = sha256(terminal)
        observation = predecessor.readiness(root)
        assert observation["ready"] is False
        assert any(item["status"] == "live_exact_generation" for item in observation["owners"].values())
        with pytest.raises(RuntimeError):
            predecessor.evidence(root)
        assert sha256(terminal) == digest and not (root / predecessor.SEAL).exists()
    finally:
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=10)


@pytest.mark.parametrize(
    "fault",
    (
        "missing_command",
        "missing_case",
        "erased_rejection",
        "nonboolean_gate",
        "missing_gate",
        "rank_coverage",
        "trajectory_endpoint",
        "changed_history",
        "changed_payload",
        "failed_outcome",
        "changed_claim",
        "changed_phase_log",
        "command_sidecar",
    ),
)
def test_localization_handoff_refuses_incomplete_or_rebound_completed_evidence(completed, fault):
    root, _ = completed
    queue = predecessor.queue
    _, _, _, work, result_path = queue.contract.locations(root)
    result = read(result_path)
    if fault == "missing_command":
        result["gpu_commands"].pop()
    elif fault == "missing_case":
        result["results"].pop()
    elif fault == "erased_rejection":
        result["rejected_cases"] = []
    elif fault == "nonboolean_gate":
        result["results"][0]["gates"]["boundary_all_rank_continuation"] = 1
    elif fault == "missing_gate":
        result["results"][0]["gates"].pop("scalar_loss_continuation")
    elif fault == "rank_coverage":
        result["results"][0]["trajectories"]["resumed"]["checkpoints"]["24"]["rank_bytes"].pop()
    elif fault == "trajectory_endpoint":
        result["results"][0]["trajectories"]["resumed"]["updates"] = 23
    elif fault == "changed_history":
        next(work.rglob("metrics_continuous.jsonl")).write_text('{"changed":true}\n')
    elif fault == "changed_payload":
        next(work.rglob("rank_7.pt")).write_bytes(b"changed after original reconstruction")
    elif fault in ("failed_outcome", "changed_claim"):
        queued = read(root / queue.WORK / "state.json")
        _, attempt, outcome, _ = queue.controller.terminal_paths(root, queued["issued_controller"])
        if fault == "failed_outcome":
            support.record(outcome, read(outcome) | {"status": "needs_attention"})
        else:
            support.record(attempt, read(attempt) | {"controller_pid": 999})
    elif fault == "changed_phase_log":
        (root / queue.WORK / "freeze.log").write_text("changed phase execution\n")
    else:
        log = Path(result["gpu_commands"][0]["log"])
        support.record(log.with_suffix(".execution.json"), read(log.with_suffix(".execution.json")) | {"exit_code": 7})
    support.record(result_path, result)
    support.rebind(root)
    with pytest.raises((ValueError, RuntimeError)):
        predecessor.evidence(root)
    assert not (root / predecessor.SEAL).exists()


def test_localization_seal_is_one_shot_and_checks_caller_bound_decisions_and_payloads(completed):
    root, _ = completed
    saved = predecessor.seal(root)
    path = root / predecessor.SEAL
    digest = sha256(path)
    assert predecessor.require_seal(root, digest) == saved
    with pytest.raises(FileExistsError):
        predecessor.seal(root)
    altered = copy.deepcopy(saved)
    altered["actual_lr_decisions"]["rejected_cases"] = []
    support.record(path, altered)
    with pytest.raises(ValueError):
        predecessor.require_seal(root, sha256(path))
    support.record(path, saved)
    digest = sha256(path)
    next(predecessor.queue.contract.locations(root)[3].rglob("rank_7.pt")).write_bytes(b"later changed payload")
    with pytest.raises(ValueError):
        predecessor.require_seal(root, digest)
    assert sha256(path) == digest


def test_localization_seal_refuses_changed_keeper_generation(completed, monkeypatch):
    root, restored = completed
    monkeypatch.setattr(predecessor.handoff, "keeper_observation", lambda: restored | {"pid": restored["pid"] + 1})
    with pytest.raises((RuntimeError, ValueError)):
        predecessor.seal(root)
    assert not (root / predecessor.SEAL).exists()


@pytest.mark.parametrize("fault", ("none", "archive", "rebound_tests", "mutable_import"))
def test_localization_handoff_archive_acceptance_rejects_rebound_or_mutable_sources(tmp_path, monkeypatch, fault):
    root = support.ROOT
    previous, actual = preparation.inherited(root)
    sources = preparation.helper_sources(previous, actual)
    out = tmp_path / "archive"

    def external_cpu(command, **kwargs):
        xml = ET.Element("testsuites")
        suite = ET.SubElement(
            xml, "testsuite", tests=str(preparation.TEST_COUNT), failures="0", errors="0", skipped="0", time="0"
        )
        for index in range(preparation.TEST_COUNT):
            ET.SubElement(
                suite, "testcase", classname=Path(preparation.NEW_HELPERS[2]).stem, name=f"external_archive_case_{index}"
            )
        ET.ElementTree(xml).write(Path(kwargs["cwd"]) / "tests.xml")
        kwargs["stdout"].write("External CPU runner boundary\n")
        return subprocess.CompletedProcess(command, 0)

    with monkeypatch.context() as runner:
        runner.setattr(preparation.subprocess, "run", external_cpu)
        for name, value in preparation.ENVIRONMENT.items():
            runner.setenv(name, value)
        cpu = preparation.cpu(root, out)
    assert cpu["helpers_sha256"] == {name: sha256(path) for name, path in sources.items()}
    assert cpu["inventory_sha256"]["experiment/fused_attribution_owner.py"] == (
        previous["inventory_sha256"]["experiment/fused_attribution_owner.py"]
    )
    if fault == "archive":
        Path(cpu["archive"]).write_bytes(b"replaced archive")
    elif fault == "rebound_tests":
        tests = Path(cpu["tests"])
        tests.write_text(tests.read_text().replace('failures="0"', 'failures="1"'))
        cpu["tests_sha256"] = sha256(tests)
        support.record(out / "receipt.json", cpu)
    script = (
        "import importlib.util,json,sys;from pathlib import Path\n"
        "import prepare_ns_compiler_localization_predecessor as p\n"
        "root,receipt,mutable=map(Path,sys.argv[1:4])\n"
        "if sys.argv[4]=='mutable_import':\n"
        " spec=importlib.util.spec_from_file_location('ns_compiler_localization_predecessor',mutable)\n"
        " module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)\n"
        "cpu=p.accepted(root,receipt)\n"
        "print(json.dumps({'status':cpu['status'],'helpers':len(cpu['helpers_sha256'])}))\n"
    )
    result = subprocess.run(
        [
            predecessor.queue.contract.PYTHON,
            "-B",
            "-P",
            "-c",
            script,
            str(root),
            str(out / "receipt.json"),
            str(Path(predecessor.__file__)),
            fault,
        ],
        cwd=out / "tracking",
        env=os.environ
        | preparation.ENVIRONMENT
        | {
            "PYTHONPATH": f"{out / 'tracking'}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    if fault == "none":
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["status"] == preparation.STATUS
    else:
        assert result.returncode != 0
