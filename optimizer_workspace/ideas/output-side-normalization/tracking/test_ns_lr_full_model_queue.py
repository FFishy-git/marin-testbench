# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise original-owner waiting and one-shot actual-LR queue/terminal lifecycle."""

import json
import os
import select
import signal
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_lr_full_model_predecessor_test_support as previous_support
import ns_lr_full_model_queue as queue
import ns_lr_full_model_queue_test_support as support
import pytest
from fused_attribution_validation_controller import acknowledged_process
from selected_lr_failure_handoff import pidfd_open
from selected_lr_release_owner import read, sha256
from watch_wave5 import write


@pytest.mark.parametrize("outcome", ("live", "active_phase", "changed_intent"))
def test_actual_lr_waits_for_real_original_queue_or_phase_and_preserves_acknowledgment(tmp_path, monkeypatch, outcome):
    work, intent = support.original_queue(tmp_path, monkeypatch)
    original = queue.predecessor.queue
    marker_name = original.MARKER if outcome != "active_phase" else original.PHASE_MARKER
    argv, marker = [queue.contract.PYTHON, "-c", "import signal; signal.pause()"], f"original-cpu-{os.getpid()}"
    child = subprocess.Popen(
        argv,
        cwd=work,
        env=os.environ | original.ENVIRONMENT | {marker_name: marker},
        start_new_session=True,
    )
    try:
        own = acknowledged_process(child, argv, work, marker_name, marker, original.ENVIRONMENT)
        if outcome == "active_phase":
            write(
                work / "state.json",
                {
                    "phases": [],
                    "active_phase": {
                        "phase": "freeze",
                        "generation": own,
                        "marker": marker,
                        "cwd": str(work),
                        "spawn_status": "acknowledged",
                    },
                },
            )
        else:
            intent.update(generation=own, marker=marker)
            write(work / "intent.json", intent)
            monkeypatch.setattr(queue.predecessor.reporting, "INTENT_SHA256", sha256(work / "intent.json"))
            if outcome == "changed_intent":
                write(work / "intent.json", intent | {"marker": "substituted"})
        before = sha256(work / "state.json")
        if outcome == "changed_intent":
            with pytest.raises(ValueError):
                queue.readiness(tmp_path)
        else:
            observed = queue.readiness(tmp_path)
            assert observed["ready"] is False and observed["status"] == "waiting_for_original_full_model_queue"
            assert any(item["status"] == "live_exact_generation" for item in observed["owners"].values())
        assert sha256(work / "state.json") == before
        assert not queue.contract.locations(tmp_path)[0].exists() and not (tmp_path / queue.TERMINAL).exists()
    finally:
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=10)


@pytest.mark.parametrize("outcome", ("ready", "failed"))
def test_actual_lr_requires_complete_original_terminal_and_retains_lower_lr_rejection(tmp_path, monkeypatch, outcome):
    root, cpu, _, state, _, previous = previous_support.completed(tmp_path, monkeypatch)
    binding = queue.contract.locations(root)[0]
    before = sha256(binding)
    monkeypatch.setattr(queue.predecessor.reporting.queue, "require_cpu", lambda *args: cpu)
    monkeypatch.setattr(queue.predecessor.previous, "evidence", lambda root: previous)
    if outcome == "failed":
        state["phases"].pop()
        write(root / queue.predecessor.queue.WORK / "state.json", state)
        with pytest.raises(ValueError):
            queue.readiness(root)
    else:
        observed = queue.readiness(root)
        assert observed["ready"] is True and observed["status"] == "original_lower_lr_terminal_ready"
        decisions = observed["predecessor"]["lower_lr_decisions"]
        assert len(decisions["accepted_cases"]) == 17 and len(decisions["rejected_cases"]) == 1
        assert decisions["all_cases_accepted"] is False
        assert observed["predecessor"]["production_authorized"] is False
    assert sha256(binding) == before and not (root / queue.TERMINAL).exists()


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached", "new_owner", "timeout", "spawn_failure"))
def test_actual_lr_queue_phase_cleans_real_descendants_without_killing_new_owner(tmp_path, monkeypatch, outcome):
    work, helpers = tmp_path / queue.WORK, tmp_path / "helpers"
    work.mkdir(parents=True)
    helpers.mkdir()
    program = support.phase_program(helpers)
    if outcome == "spawn_failure":
        monkeypatch.setattr(queue.subprocess, "Popen", support.fail_spawn)
    if outcome == "timeout":
        monkeypatch.setattr(queue, "PHASE_TIMEOUT", 0.25)
    state = {"status": "issuing_actual_lr_validation", "phases": []}
    argv = [sys.executable, str(program), str(tmp_path), outcome]
    child_pid = None
    try:
        if outcome in ("nonzero", "detached", "timeout", "spawn_failure"):
            with pytest.raises((RuntimeError, OSError, subprocess.TimeoutExpired)):
                queue.run_phase(tmp_path, "freeze", argv, helpers, state)
        else:
            result = queue.run_phase(tmp_path, "freeze", argv, helpers, state)
            assert result["exit_code"] == result["cleanup_exit_code"] == 0
        saved = read(work / "state.json")["active_phase"]
        assert saved["log_sha256"] == sha256(work / "freeze.log")
        if outcome == "spawn_failure":
            assert saved["spawn_status"] == "starting" and "pid" not in saved
        else:
            assert saved["marked_descendants_absent"] is True
            assert queue.handoff.observed(saved["pid"]) is None
            assert not queue.handoff.marker_processes(queue.PHASE_MARKER, saved["marker"])
        if outcome in ("detached", "new_owner"):
            child_pid = int((work / "detached.pid").read_text())
            observed = queue.handoff.observed(child_pid)
            if outcome == "new_owner":
                assert observed is not None and observed["status"] not in ("Z", "X", "x")
            else:
                assert observed is None or observed["status"] == "Z"
        before = sha256(work / "state.json")
        with pytest.raises(FileExistsError):
            queue.run_phase(tmp_path, "freeze", argv, helpers, state)
        assert sha256(work / "state.json") == before
    finally:
        if child_pid is not None:
            observed = queue.handoff.observed(child_pid)
            if observed is not None and observed["status"] not in ("Z", "X", "x"):
                descriptor = pidfd_open(child_pid)
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                    poll = select.poll()
                    poll.register(descriptor, select.POLLIN)
                    assert poll.poll(5000)
                finally:
                    os.close(descriptor)


def test_partial_actual_lr_freeze_is_preserved_and_cannot_relaunch(tmp_path, monkeypatch):
    work, helpers = tmp_path / queue.WORK, tmp_path / "helpers"
    work.mkdir(parents=True)
    helpers.mkdir()
    program = helpers / "freeze.py"
    program.write_text(
        "import json,os,select,sys\nfrom pathlib import Path\n"
        f"root=Path(sys.argv[1]);work=root/{queue.WORK!r};poll=select.poll()\n"
        "while json.loads((work/'state.json').read_text())['active_phase'].get('pid')!=os.getpid():poll.poll(1)\n"
        f"binding=root/{str(queue.contract.locations(tmp_path)[0].relative_to(tmp_path))!r}\n"
        "binding.parent.mkdir(parents=True,exist_ok=True);binding.write_text('{\"partial\":true}');sys.exit(7)\n"
    )
    monkeypatch.setattr(queue, "readiness", lambda root: {"ready": True})
    monkeypatch.setattr(queue, "phases", lambda root: [("freeze", [sys.executable, str(program), str(root)], helpers)])
    state = {"status": "original_lower_lr_terminal_ready", "phases": []}
    with pytest.raises(RuntimeError):
        queue.launch(tmp_path, state)
    binding = queue.contract.locations(tmp_path)[0]
    assert read(binding)["partial"] is True
    assert read(work / "state.json")["active_phase"]["exit_code"] == 7
    before = sha256(binding)
    with pytest.raises(FileExistsError):
        queue.launch(tmp_path, state)
    assert sha256(binding) == before and state["phases"] == []


@pytest.mark.parametrize("outcome", ("controller_live", "validator_live", "ready", "failed", "marked_rank"))
def test_actual_lr_queue_waits_for_issued_owner_absence_and_clean_complete_state(tmp_path, monkeypatch, outcome):
    state, path = support.issued_owner(tmp_path, monkeypatch, outcome)
    before = sha256(path)
    if outcome in ("failed", "marked_rank"):
        with pytest.raises(RuntimeError):
            queue.issued_readiness(tmp_path, state)
    else:
        observed = queue.issued_readiness(tmp_path, state)
        assert observed["ready"] is (outcome == "ready")
    assert sha256(path) == before and not (tmp_path / queue.TERMINAL).exists()


@pytest.mark.parametrize("changed", (False, True))
def test_actual_lr_queue_terminal_retains_rejection_and_refuses_changed_verifier(tmp_path, monkeypatch, changed):
    _, state = support.terminal_program(tmp_path, changed)
    monkeypatch.setattr(queue.contract, "PYTHON", sys.executable)
    monkeypatch.setattr(queue, "issued_readiness", lambda root, state: {"ready": True})
    if changed:
        with pytest.raises(ValueError):
            queue.terminal_phase(tmp_path, state)
        assert (tmp_path / queue.TERMINAL).is_file()
    else:
        result = queue.terminal_phase(tmp_path, state)
        assert result["status"] == "actual_lr_terminal_reconstructed"
        assert result["accepted_cases"] == ["accepted"] and result["rejected_cases"] == ["rejected"]
        assert result["all_cases_accepted"] is False
        assert result["phases"][-1]["exit_code"] == result["phases"][-1]["cleanup_exit_code"] == 0
    before = sha256(tmp_path / queue.TERMINAL)
    with pytest.raises(RuntimeError):
        queue.terminal_phase(tmp_path, state)
    assert sha256(tmp_path / queue.TERMINAL) == before


@pytest.fixture(scope="module")
def inherited():
    return read(support.ROOT / queue.PARENT)


@pytest.mark.parametrize("fault", ("none", "truncated_xml", "archive", "fixture", "mutable_import"))
def test_actual_lr_queue_cpu_archive_preserves_parent_and_refuses_forged_evidence(
    tmp_path, monkeypatch, inherited, fault
):
    root, out = tmp_path / "root", tmp_path / "cpu"
    root.mkdir()
    monkeypatch.setattr(queue, "parent", lambda root: inherited)
    for name, value in queue.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(queue.subprocess, "run", support.completed_cpu)
    receipt = queue.cpu(root, out)
    assert all(receipt["helpers_sha256"][name] == digest for name, digest in inherited["helpers_sha256"].items())
    assert len(receipt["helpers_sha256"]) == 156
    if fault == "truncated_xml":
        tree = ET.parse(out / "tests.xml")
        suite = tree.getroot().find("testsuite")
        suite.remove(suite.findall("testcase")[-1])
        suite.set("tests", str(queue.TEST_COUNT - 1))
        tree.write(out / "tests.xml")
        write(out / "receipt.json", receipt | {"tests_sha256": sha256(out / "tests.xml")})
    elif fault == "archive":
        with tarfile.open(out / "source.tar", "a") as bundle:
            bundle.add(out / "tracking" / queue.NEW_HELPERS[0], arcname="unaccepted.py")
        write(out / "receipt.json", receipt | {"archive_sha256": sha256(out / "source.tar")})
    elif fault == "fixture":
        name = next(name for name in receipt["inventory_sha256"] if name.startswith("artifacts/pinned-sources"))
        (out / name).write_bytes((out / name).read_bytes() + b"\n")
    if fault in ("none", "mutable_import"):
        monkeypatch.undo()
        probe = support.archive_probe(root, out)
        checked = subprocess.run(
            [
                sys.executable,
                "-P",
                str(probe),
                str(root),
                str(out / "receipt.json"),
                str(support.ROOT / queue.PARENT),
                str(root / "mutable_predecessor.py"),
                fault,
            ],
            cwd=out,
            env=support.process_environment(out / "tracking"),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if fault == "mutable_import":
            assert checked.returncode != 0
        else:
            assert checked.returncode == 0, checked.stderr
            assert json.loads(checked.stdout) == {"status": "archived_actual_lr_queue_roundtrip", "helpers": 156}
            monkeypatch.setattr(queue, "parent", lambda root: inherited)
            assert queue.require_cpu(root, out / "receipt.json") == receipt
    else:
        with pytest.raises(ValueError):
            queue.require_cpu(root, out / "receipt.json")


@pytest.mark.parametrize("outcome", ("zero", "spawn_failure"))
def test_actual_lr_queue_start_acknowledges_one_original_cpu_waiter(tmp_path, monkeypatch, outcome):
    work, helpers = tmp_path / queue.WORK, tmp_path / "helpers"
    work.parent.mkdir(parents=True)
    helpers.mkdir()
    program = helpers / "queue.py"
    program.write_text(
        "import json,os,select,sys\nfrom pathlib import Path\n"
        "intent=Path(sys.argv[1]);poll=select.poll()\n"
        "while json.loads(intent.read_text()).get('pid')!=os.getpid():poll.poll(1)\n"
    )
    cpu_path = tmp_path / "receipt.json"
    write(cpu_path, {})
    receipt = {"helpers_sha256": {queue.NEW_HELPERS[0]: "CPU-runner"}, "helpers": str(helpers)}
    monkeypatch.setattr(queue, "accepted", lambda *args: receipt)
    monkeypatch.setattr(
        queue, "readiness", lambda root: {"ready": False, "status": "waiting_for_original_full_model_queue"}
    )
    monkeypatch.setattr(queue, "local_node", lambda: queue.owner.NODE)
    argv = [sys.executable, str(program), str(work / "intent.json")]
    monkeypatch.setattr(queue, "command", lambda *args: argv)
    children, original = [], subprocess.Popen

    def spawn(*args, **kwargs):
        if outcome == "spawn_failure":
            return support.fail_spawn(*args, **kwargs)
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(queue.subprocess, "Popen", spawn)
    try:
        if outcome == "zero":
            intent = queue.start(tmp_path, cpu_path)
            assert intent["status"] == "acknowledged" and intent["generation"]["parent"] == os.getpid()
            assert intent["generation"]["session"] == intent["generation"]["group"] == intent["pid"]
            assert children[0].wait(timeout=10) == 0
        else:
            with pytest.raises(OSError):
                queue.start(tmp_path, cpu_path)
            assert read(work / "intent.json")["status"] == "needs_attention"
        with pytest.raises(RuntimeError):
            queue.start(tmp_path, cpu_path)
        assert not queue.contract.locations(tmp_path)[0].exists()
    finally:
        for child in children:
            queue.finish_marked_child(child, read(work / "intent.json")["marker"], {}, queue.MARKER)


def test_actual_lr_execute_publishes_waiting_state_and_900_second_interval_without_gpu_phase(tmp_path, monkeypatch):
    work, intent = support.original_queue(tmp_path, monkeypatch)
    original = queue.predecessor.queue
    argv, marker = [queue.contract.PYTHON, "-c", "import signal; signal.pause()"], "live-original-full-model"
    child = subprocess.Popen(
        argv,
        cwd=work,
        env=os.environ | original.ENVIRONMENT | {original.MARKER: marker},
        start_new_session=True,
    )
    try:
        own = acknowledged_process(child, argv, work, original.MARKER, marker, original.ENVIRONMENT)
        intent.update(generation=own, marker=marker)
        write(work / "intent.json", intent)
        monkeypatch.setattr(queue.predecessor.reporting, "INTENT_SHA256", sha256(work / "intent.json"))
        queued = tmp_path / queue.WORK
        queued.mkdir()
        cpu_path = tmp_path / "cpu.json"
        write(cpu_path, {})
        command = queue.command(tmp_path, cpu_path)
        current = {
            "pid": os.getpid(),
            "status": "R",
            "start_ticks": "CPU-original-waiter",
            "parent": os.getppid(),
            "group": os.getpid(),
            "session": os.getpid(),
            "command": command,
        }
        write(
            queued / "intent.json",
            {
                "status": "acknowledged",
                "pid": current["pid"],
                "start_ticks": current["start_ticks"],
                "generation": current,
                "command": command,
                "marker": "CPU-waiter",
                "cpu_receipt_sha256": sha256(cpu_path),
                "runner_sha256": sha256(Path(queue.__file__)),
                "parent_cpu_receipt_sha256": queue.PARENT_SHA256,
                "interval": 900,
            },
        )
        monkeypatch.setattr(queue, "accepted", lambda *args: {})
        monkeypatch.setattr(queue, "process_context", lambda *args: current)
        monkeypatch.setattr(queue, "local_node", lambda: queue.owner.NODE)
        delays = []

        def stopped_clock(interval):
            delays.append(interval)
            raise KeyboardInterrupt("CPU clock ends after real live-owner observation")

        with monkeypatch.context() as clock:
            clock.setattr(queue.time, "sleep", stopped_clock)
            with pytest.raises(KeyboardInterrupt):
                queue.execute(tmp_path, cpu_path)
        saved = read(queued / "state.json")
        assert delays == [900] and saved["phases"] == []
        assert saved["original_observation"]["owners"]["original_queue"]["status"] == "live_exact_generation"
        assert not queue.contract.locations(tmp_path)[0].exists()
        assert not (tmp_path / queue.TERMINAL).exists()
    finally:
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=10)
