# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise exact original-owner waiting and the one-shot actual-LR queue lifecycle."""

import json
import os
import select
import signal
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_lr_predecessor_test_support as previous_support
import ns_training_path_lr_queue as queue
import ns_training_path_lr_queue_test_support as support
import pytest
from fused_attribution_validation_controller import acknowledged_process
from selected_lr_failure_handoff import pidfd_open
from selected_lr_release_owner import read, sha256
from watch_wave5 import write


@pytest.mark.parametrize("role", ("controller", "validator", "terminal_waiter"))
def test_training_path_lr_waits_for_real_original_owner_and_preserves_terminal(tmp_path, monkeypatch, role):
    root, _ = previous_support.completed(tmp_path, monkeypatch)
    frozen, intent, state, waiter = queue.predecessor.original(root)
    helpers = Path(frozen["helpers"])
    if role == "controller":
        cwd, argv, marker, environment = helpers, intent["command"], intent["marker"], frozen["controller_environment"]
        marker_name = queue.predecessor.contract.CONTROLLER_MARKER
    elif role == "validator":
        cwd, argv, marker, environment = (
            Path(frozen["candidate"]),
            state["command"],
            state["child_marker"],
            frozen["environment"],
        )
        marker_name = queue.predecessor.contract.CHILD_MARKER
    else:
        cwd, argv = root / queue.predecessor.reporting.FOLLOWUP_WORK, [
            queue.contract.PYTHON,
            "-c",
            "import signal; signal.pause()",
        ]
        marker, environment, marker_name = (
            waiter["marker"],
            queue.predecessor.followup.ENVIRONMENT,
            queue.predecessor.followup.MARKER,
        )
    child = subprocess.Popen(argv, cwd=cwd, env=os.environ | environment | {marker_name: marker}, start_new_session=True)
    try:
        own = acknowledged_process(child, argv, cwd, marker_name, marker, environment)
        if role == "controller":
            intent.update(pid=own["pid"], start_ticks=own["start_ticks"], generation=own)
            write(helpers.parent / "controller.json", intent)
            monkeypatch.setattr(queue.predecessor.reporting, "INTENT_SHA256", sha256(helpers.parent / "controller.json"))
            state.update(controller_pid=own["pid"], controller_start_ticks=own["start_ticks"])
            state["child_generation"]["parent"] = own["pid"]
            write(queue.predecessor.contract.locations(root)[2], state)
        elif role == "validator":
            state.update(
                child_pid=own["pid"],
                child_start_ticks=own["start_ticks"],
                child_generation=own | {"parent": intent["pid"]},
            )
            write(queue.predecessor.contract.locations(root)[2], state)
        else:
            monkeypatch.setattr(queue.predecessor.followup, "command", lambda *args: argv)
            waiter.update(pid=own["pid"], start_ticks=own["start_ticks"], generation=own, command=argv)
            work = root / queue.predecessor.reporting.FOLLOWUP_WORK
            write(work / "intent.json", waiter)
            saved = read(work / "state.json")
            write(work / "state.json", saved | {"pid": own["pid"], "start_ticks": own["start_ticks"]})
            monkeypatch.setattr(queue.predecessor.reporting, "FOLLOWUP_INTENT_SHA256", sha256(work / "intent.json"))
        original = queue.predecessor.controller.terminal_paths(root, intent)[0]
        before = sha256(original) if original.is_file() else None
        observed = queue.readiness(root)
        assert observed["ready"] is False
        assert observed["owners"][role]["status"] == "live_exact_generation"
        assert not queue.contract.locations(root)[0].exists() and not (root / queue.predecessor.SEAL).exists()
        assert (sha256(original) if original.is_file() else None) == before
    finally:
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=10)


@pytest.mark.parametrize("outcome", ("ready", "failed"))
def test_training_path_lr_requires_complete_original_terminal_and_retains_rejections(tmp_path, monkeypatch, outcome):
    root, _ = previous_support.completed(tmp_path, monkeypatch)
    _, intent, _, _ = queue.predecessor.original(root)
    terminal, _, _, _ = queue.predecessor.controller.terminal_paths(root, intent)
    before = sha256(terminal)
    if outcome == "failed":
        report = read(root / "accepted-report.json")
        report["gpu_progress"]["terminal"]["status"] = "needs_attention"
        write(root / "accepted-report.json", report)
        with pytest.raises(RuntimeError):
            queue.readiness(root)
    else:
        observed = queue.readiness(root)
        assert observed["ready"] is True and observed["original_terminal_sha256"] == before
        assert len(observed["original_decisions"]["accepted_cases"]) == 17
        assert len(observed["original_decisions"]["rejected_cases"]) == 1
        assert observed["original_decisions"]["all_cases_accepted"] is False
    assert sha256(terminal) == before and not (root / queue.predecessor.SEAL).exists()


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached", "new_owner", "timeout", "spawn_failure"))
def test_training_path_lr_phase_cleans_descendants_and_preserves_independent_owner(tmp_path, monkeypatch, outcome):
    work, helpers = tmp_path / queue.WORK, tmp_path / "helpers"
    work.mkdir(parents=True)
    helpers.mkdir()
    program = support.phase_program(helpers)
    if outcome == "spawn_failure":
        monkeypatch.setattr(queue.subprocess, "Popen", support.fail_spawn)
    if outcome == "timeout":
        monkeypatch.setattr(queue, "PHASE_TIMEOUT", 0.25)
    state, child_pid = {"status": "issuing", "phases": []}, None
    try:
        argv = [sys.executable, str(program), str(tmp_path), outcome]
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
            assert saved["marked_descendants_absent"] is True and queue.handoff.observed(saved["pid"]) is None
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


def test_partial_training_path_lr_seal_is_retained_and_cannot_relaunch(tmp_path, monkeypatch):
    root, _ = previous_support.completed(tmp_path, monkeypatch)
    work, helpers = root / queue.WORK, root / "phase-helpers"
    work.mkdir(parents=True)
    helpers.mkdir()
    program = helpers / "seal.py"
    program.write_text(
        "import json,os,select,sys\nfrom pathlib import Path\n"
        f"root=Path(sys.argv[1]);work=root/{queue.WORK!r};poll=select.poll()\n"
        "while json.loads((work/'state.json').read_text())['active_phase'].get('pid')!=os.getpid():poll.poll(1)\n"
        f"(root/{queue.predecessor.SEAL!r}).write_text('{{\"partial\":true}}');sys.exit(7)\n"
    )
    monkeypatch.setattr(queue, "predecessor_command", lambda *args: ([sys.executable, str(program), str(root)], helpers))
    state = {"status": "original_terminal_ready", "phases": []}
    with pytest.raises(RuntimeError):
        queue.launch(root, state)
    seal = root / queue.predecessor.SEAL
    assert read(seal)["partial"] is True and read(work / "state.json")["active_phase"]["exit_code"] == 7
    before = sha256(seal)
    with pytest.raises(FileExistsError):
        queue.launch(root, state)
    assert sha256(seal) == before and state["phases"] == [] and not queue.contract.locations(root)[0].exists()


def test_training_path_lr_queue_refuses_retained_failed_phase_without_reissuing_seal(tmp_path, monkeypatch):
    root, _ = previous_support.completed(tmp_path, monkeypatch)
    work = root / queue.WORK
    work.mkdir(parents=True)
    state = {"status": "needs_attention", "phases": [{"phase": "seal", "exit_code": 7}]}
    write(work / "state.json", state)
    before = sha256(work / "state.json")
    with pytest.raises(RuntimeError):
        queue.launch(root, state)
    assert sha256(work / "state.json") == before
    assert not (root / queue.predecessor.SEAL).exists() and not queue.contract.locations(root)[0].exists()


def test_training_path_lr_launch_binds_original_seal_and_preserves_real_acknowledged_successor(tmp_path, monkeypatch):
    root, _ = previous_support.completed(tmp_path, monkeypatch)
    sealed = queue.predecessor.evidence(root)
    _, helpers = support.launch_program(root, sealed)
    work = root / queue.WORK
    work.mkdir(parents=True)
    monkeypatch.setattr(
        queue,
        "predecessor_command",
        lambda root, phase, seal_sha256=None: (
            [
                queue.contract.PYTHON,
                "-P",
                str(helpers / "prepare_ns_training_path_lr_predecessor.py"),
                "--root",
                str(root),
                "--phase",
                phase,
                *(["--seal-sha256", seal_sha256] if seal_sha256 else []),
            ],
            helpers,
        ),
    )
    state = {"status": "original_terminal_ready", "phases": []}
    try:
        result = queue.launch(root, state)
        assert result["status"] == "actual_lr_training_path_controller_acknowledged"
        assert [entry["phase"] for entry in result["phases"]] == ["seal", "check-seal", "freeze", "check", "start"]
        assert all(entry["exit_code"] == entry["cleanup_exit_code"] == 0 for entry in result["phases"])
        digest = sha256(root / queue.predecessor.SEAL)
        assert read(work / "freeze-argv.json")["seal_sha256"] == result["predecessor_seal_sha256"] == digest
        assert result["original_decisions"]["rejected_cases"] == sealed["original_decisions"]["rejected_cases"]
        own = queue.handoff.observed(result["issued_controller"]["pid"])
        assert own is not None and own["status"] not in ("Z", "X", "x")
        assert own["start_ticks"] == result["issued_controller"]["start_ticks"]
        assert read(queue.contract.locations(root)[0])["allocation_predecessor"] == sealed
        before = sha256(root / queue.predecessor.SEAL)
        with pytest.raises(FileExistsError):
            queue.launch(root, state)
        assert sha256(root / queue.predecessor.SEAL) == before
    finally:
        if (work / "owner.pid").is_file():
            pid = int((work / "owner.pid").read_text())
            descriptor = pidfd_open(pid)
            try:
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                poll = select.poll()
                poll.register(descriptor, select.POLLIN)
                assert poll.poll(5000)
            finally:
                os.close(descriptor)


@pytest.mark.parametrize(
    "outcome",
    ("controller_live", "validator_live", "ready", "failed", "cleanup_failed", "marked_rank", "substituted", "consumed"),
)
def test_training_path_lr_issued_terminal_requires_owner_absence_cleanup_and_unused_claim(
    tmp_path, monkeypatch, outcome
):
    state, path = support.issued_owner(tmp_path, monkeypatch, outcome)
    before = sha256(path)
    if outcome in ("failed", "cleanup_failed", "substituted", "consumed"):
        with pytest.raises((RuntimeError, FileExistsError)):
            queue.issued_readiness(tmp_path, state)
    else:
        observed = queue.issued_readiness(tmp_path, state)
        assert observed["ready"] is (outcome == "ready")
    assert sha256(path) == before
    assert not queue.controller.terminal_paths(tmp_path, state["issued_controller"])[0].exists()


@pytest.mark.parametrize("fault", ("none", "verifier", "outcome"))
def test_training_path_lr_canonical_terminal_retains_rejections_and_refuses_changed_evidence(
    tmp_path, monkeypatch, fault
):
    state, _ = support.issued_owner(tmp_path, monkeypatch, "ready")
    monkeypatch.undo()
    (tmp_path / queue.WORK).mkdir(parents=True)
    support.terminal_program(tmp_path, state, fault)
    if fault == "none":
        result = queue.terminal_phase(tmp_path, state)
        assert result["status"] == "actual_lr_training_path_terminal_reconstructed"
        assert result["accepted_cases"] == ["accepted"] and result["rejected_cases"] == ["rejected"]
        assert result["all_cases_accepted"] is False and result["original_decisions"]["rejected_cases"] == [
            "original-rejected"
        ]
        assert result["phases"][-1]["exit_code"] == result["phases"][-1]["cleanup_exit_code"] == 0
    else:
        with pytest.raises(ValueError):
            queue.terminal_phase(tmp_path, state)
    terminal, attempt, outcome, _ = queue.controller.terminal_paths(tmp_path, state["issued_controller"])
    assert terminal.is_file() and attempt.is_file() and outcome.is_file()
    before = sha256(terminal)
    with pytest.raises(FileExistsError):
        queue.terminal_phase(tmp_path, state)
    assert sha256(terminal) == before


@pytest.fixture(scope="module")
def inherited():
    return read(support.ROOT / queue.PARENT)


@pytest.mark.parametrize("fault", ("none", "truncated_xml", "archive", "fixture", "mutable_import"))
def test_training_path_lr_queue_archive_preserves_parent_and_rejects_forged_evidence(
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
    assert len(receipt["helpers_sha256"]) == 180
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
            assert json.loads(checked.stdout) == {"status": "archived_training_path_lr_queue_roundtrip", "helpers": 180}
    else:
        with pytest.raises(ValueError):
            queue.require_cpu(root, out / "receipt.json")


@pytest.mark.parametrize("outcome", ("zero", "spawn_failure"))
def test_training_path_lr_queue_start_acknowledges_one_original_hidden_cpu_waiter(tmp_path, monkeypatch, outcome):
    root, _ = previous_support.completed(tmp_path, monkeypatch)
    helpers = root / "queue-helpers"
    helpers.mkdir()
    work = root / queue.WORK
    program = helpers / "queue.py"
    program.write_text(
        "import json,os,select,sys\nfrom pathlib import Path\n"
        "intent=Path(sys.argv[1]);poll=select.poll()\n"
        "while json.loads(intent.read_text()).get('pid')!=os.getpid():poll.poll(1)\n"
    )
    cpu_path = root / "receipt.json"
    write(cpu_path, {})
    receipt = {"helpers_sha256": {queue.NEW_HELPERS[0]: "CPU-runner"}, "helpers": str(helpers)}
    monkeypatch.setattr(queue, "accepted", lambda *args: receipt)
    monkeypatch.setattr(queue, "local_node", lambda: queue.owner.NODE)
    argv = [sys.executable, str(program), str(work / "intent.json")]
    monkeypatch.setattr(queue, "command", lambda *args: argv)
    children, original = [], subprocess.Popen

    def spawn(*args, **kwargs):
        if args[0] != argv:
            return original(*args, **kwargs)
        if outcome == "spawn_failure":
            return support.fail_spawn(*args, **kwargs)
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(queue.subprocess, "Popen", spawn)
    try:
        if outcome == "zero":
            intent = queue.start(root, cpu_path)
            assert intent["status"] == "acknowledged" and intent["generation"]["parent"] == os.getpid()
            assert intent["generation"]["session"] == intent["generation"]["group"] == intent["pid"]
            assert children[0].wait(timeout=10) == 0
        else:
            with pytest.raises(OSError):
                queue.start(root, cpu_path)
            assert read(work / "intent.json")["status"] == "needs_attention"
        with pytest.raises(FileExistsError):
            queue.start(root, cpu_path)
        assert not queue.contract.locations(root)[0].exists()
    finally:
        for child in children:
            queue.finish_marked_child(child, read(work / "intent.json")["marker"], {}, queue.MARKER)


def test_training_path_lr_execute_retains_900_second_original_observation_without_gpu_phase(tmp_path, monkeypatch):
    root, _ = previous_support.completed(tmp_path, monkeypatch)
    work = root / queue.WORK
    work.mkdir(parents=True)
    cpu_path = root / "cpu.json"
    write(cpu_path, {})
    argv = queue.command(root, cpu_path)
    current = support.generation(os.getpid(), os.getppid(), argv)
    write(
        work / "intent.json",
        {
            "status": "acknowledged",
            "pid": current["pid"],
            "start_ticks": current["start_ticks"],
            "generation": current,
            "command": argv,
            "marker": "CPU-waiter",
            "cpu_receipt_sha256": sha256(cpu_path),
            "runner_sha256": sha256(Path(queue.__file__)),
            "parent_cpu_receipt_sha256": queue.PARENT_SHA256,
            "interval": 900,
            "scope": queue.SCOPE,
        },
    )
    monkeypatch.setattr(queue, "accepted", lambda *args: {})
    monkeypatch.setattr(queue, "process_context", lambda *args: current)
    monkeypatch.setattr(queue, "local_node", lambda: queue.owner.NODE)
    monkeypatch.setattr(
        queue,
        "readiness",
        lambda root: {
            "ready": False,
            "status": "waiting_for_original_training_path_owners",
            "owners": {"controller": {"status": "live_exact_generation"}},
        },
    )
    delays = []

    def stopped_clock(interval):
        delays.append(interval)
        raise KeyboardInterrupt("CPU clock ends after original-owner observation")

    with monkeypatch.context() as clock:
        clock.setattr(queue.time, "sleep", stopped_clock)
        with pytest.raises(KeyboardInterrupt):
            queue.execute(root, cpu_path)
    saved = read(work / "state.json")
    assert delays == [900] and saved["phases"] == [] and saved["original_observations"] == 1
    assert read(work / "original-observation-000000.json")["owners"]["controller"]["status"] == "live_exact_generation"
    assert not queue.contract.locations(root)[0].exists() and not (root / queue.predecessor.SEAL).exists()
