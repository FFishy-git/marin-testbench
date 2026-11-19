# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise original-owner waiting and real one-shot NS queue phase lifecycles."""

import copy
import os
import select
import signal
import sys

import ns_budget_queue as queue
import pytest
from selected_lr_failure_handoff import pidfd_open
from watch_wave5 import write


@pytest.mark.parametrize("outcome", ("live", "absent", "substituted", "zombie", "wrong_marker"))
def test_ns_queue_handle_waits_on_original_generation_and_refuses_changed_context(tmp_path, monkeypatch, outcome):
    proc, cwd = tmp_path / "proc", tmp_path / "helpers"
    proc.mkdir()
    cwd.mkdir()
    expected = {
        "pid": 111,
        "start_ticks": "1234",
        "parent": 1,
        "group": 111,
        "session": 111,
        "status": "S",
        "command": ["python", "original"],
    }
    monkeypatch.setattr(queue.owner.handoff, "PROC_ROOT", proc)
    if outcome != "absent":
        directory = proc / "111"
        directory.mkdir()
        ticks = "9999" if outcome == "substituted" else "1234"
        status = "Z" if outcome == "zombie" else "S"
        (directory / "stat").write_text(f"111 (fixture) {status} 1 111 111 " + "0 " * 15 + ticks + " 0\n")
        (directory / "cmdline").write_bytes(b"python\0original\0")
        environment = queue.owner.ENVIRONMENT | {queue.MARKER: "changed" if outcome == "wrong_marker" else "original"}
        (directory / "environ").write_bytes(b"\0".join(f"{key}={value}".encode() for key, value in environment.items()))
        (directory / "cwd").symlink_to(cwd)
    if outcome in ("substituted", "wrong_marker"):
        with pytest.raises(RuntimeError):
            queue.handle(expected, cwd, queue.MARKER, "original", queue.owner.ENVIRONMENT)
    else:
        result = queue.handle(expected, cwd, queue.MARKER, "original", queue.owner.ENVIRONMENT)
        assert (
            result["status"]
            == {
                "live": "live_exact_generation",
                "absent": "absent",
                "zombie": "exited_handle_present",
            }[outcome]
        )
    assert list(cwd.iterdir()) == []


@pytest.fixture
def original_owners(tmp_path, monkeypatch):
    binding, helpers, state_path, _, _ = queue.owner.rational.locations(tmp_path)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    source = helpers / "issued.py"
    source.write_text("# immutable fixture\n")
    controller = {
        "pid": 111,
        "start_ticks": "111",
        "group": 111,
        "session": 111,
        "command": ["python", "original-controller"],
    }
    validator = controller | {
        "pid": 222,
        "parent": 111,
        "group": 222,
        "session": 222,
        "start_ticks": "222",
        "command": ["python", "original-validator"],
    }
    waiter = controller | {
        "pid": 333,
        "group": 333,
        "session": 333,
        "start_ticks": "333",
        "command": ["python", "original-waiter"],
    }
    frozen = {
        "candidate": str(candidate),
        "helpers_sha256": {source.name: queue.sha256(source)},
        "controller_command": controller["command"],
        "validator_command": validator["command"],
    }
    write(binding, frozen)
    state = {
        "status": "validating",
        "binding_sha256": queue.sha256(binding),
        "controller_pid": 111,
        "controller_start_ticks": "111",
        "child_pid": 222,
        "child_start_ticks": "222",
        "child_generation": validator,
        "child_marker": "validator",
        "command": validator["command"],
    }
    intent = {
        "status": "acknowledged",
        "binding_sha256": queue.sha256(binding),
        "pid": 111,
        "start_ticks": "111",
        "command": controller["command"],
        "generation": controller,
        "marker": "controller",
    }
    write(state_path, state)
    write(helpers.parent / "controller.json", intent)
    follow = tmp_path / queue.owner.FOLLOWUP
    follow.mkdir()
    write(follow / "intent.json", {"status": "acknowledged", "generation": waiter, "marker": "waiter"})
    cpu_path = tmp_path / queue.owner.FOLLOWUP_CPU
    cpu_path.parent.mkdir(parents=True)
    cpu_path.write_text("{}")
    monkeypatch.setattr(queue.owner.preparation, "BINDING_SHA256", queue.sha256(binding))
    monkeypatch.setattr(queue.owner, "FOLLOWUP_INTENT_SHA256", queue.sha256(follow / "intent.json"))
    monkeypatch.setattr(queue.owner, "FOLLOWUP_CPU_SHA256", queue.sha256(cpu_path))
    monkeypatch.setattr(
        queue.owner, "process_context", lambda pid, *args: {111: controller, 222: validator, 333: waiter}[pid]
    )
    return {111: controller, 222: validator, 333: waiter}, state_path


@pytest.mark.parametrize("outcome", ("controller_live", "validator_live", "waiter_live", "ready", "failed"))
def test_ns_queue_readiness_requires_all_original_handles_absent_before_terminal_acceptance(
    tmp_path, monkeypatch, original_owners, outcome
):
    owners, state_path = original_owners
    live = {"controller_live": 111, "validator_live": 222, "waiter_live": 333}.get(outcome)
    monkeypatch.setattr(
        queue.owner.handoff, "observed", lambda pid: owners[pid] | {"status": "R"} if pid == live else None
    )
    previous = {"intervention_accepted": False, "restored_keeper": {"pid": 444}}

    def terminal_boundary(root):
        if outcome in ("controller_live", "validator_live", "waiter_live"):
            pytest.fail("NS queue reconstructed terminal while an original owner was live")
        if outcome == "failed":
            raise RuntimeError("Original terminal reconstruction failed")
        return copy.deepcopy(previous)

    monkeypatch.setattr(queue.owner, "predecessor", terminal_boundary)
    before = queue.sha256(state_path)
    if outcome == "failed":
        with pytest.raises(RuntimeError):
            queue.readiness(tmp_path)
    else:
        result = queue.readiness(tmp_path)
        assert result["ready"] is (outcome == "ready")
        if outcome == "ready":
            assert result["predecessor"]["intervention_accepted"] is False
        else:
            assert result["owners"][outcome.removesuffix("_live").replace("waiter", "terminal_waiter")]["status"] == (
                "live_exact_generation"
            )
    assert queue.sha256(state_path) == before
    assert not queue.owner.locations(tmp_path)[0].exists()


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached", "new_owner"))
def test_real_ns_queue_phase_retains_cleanup_and_leaves_separately_marked_owner_running(tmp_path, outcome):
    work, helpers = tmp_path / queue.WORK, tmp_path / "helpers"
    work.mkdir(parents=True)
    helpers.mkdir()
    source = helpers / "phase.py"
    source.write_text(
        "import argparse,json,os,select,signal,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--outcome');a=p.parse_args()\n"
        f"work=Path(a.root)/{queue.WORK!r};poll=select.poll()\n"
        "while json.loads((work/'state.json').read_text())['active_phase'].get('pid')!=os.getpid():poll.poll(1)\n"
        "if a.outcome in ('detached','new_owner'):\n"
        " r,w=os.pipe();env=dict(os.environ)\n"
        f" if a.outcome=='new_owner':env[{queue.owner.CONTROLLER_MARKER!r}]='separately-acknowledged'\n"
        " child=subprocess.Popen([sys.executable,'-c',"
        "'import os,signal,sys;os.write(int(sys.argv[1]),b\"R\");signal.pause()',str(w)],"
        "pass_fds=(w,),start_new_session=True,env=env)\n"
        " os.close(w);assert os.read(r,1)==b'R';os.close(r)\n"
        " (work/'detached.pid').write_text(str(child.pid))\n"
        "sys.exit(7 if a.outcome in ('nonzero','detached') else 0)\n"
    )
    state = {"status": "issuing_ns_diagnostic", "phases": []}
    argv = [sys.executable, str(source), "--root", str(tmp_path), "--outcome", outcome]
    descendant = None
    try:
        if outcome in ("nonzero", "detached"):
            with pytest.raises(RuntimeError):
                queue.run_phase(tmp_path, "freeze", argv, helpers, state)
        else:
            result = queue.run_phase(tmp_path, "freeze", argv, helpers, state)
            assert result["exit_code"] == result["cleanup_exit_code"] == 0
        saved = queue.owner.read(work / "state.json")["active_phase"]
        assert saved["exit_code"] == (7 if outcome in ("nonzero", "detached") else 0)
        assert saved["cleanup_exit_code"] == saved["exit_code"] and saved["marked_descendants_absent"]
        assert queue.owner.handoff.observed(saved["pid"]) is None
        assert not queue.owner.handoff.marker_processes(queue.owner.CONTROLLER_MARKER, saved["marker"])
        assert saved["log_sha256"] == queue.sha256(work / "freeze.log")
        if outcome in ("detached", "new_owner"):
            descendant = int((work / "detached.pid").read_text())
            actual = queue.owner.handoff.observed(descendant)
            if outcome == "new_owner":
                assert actual is not None and actual["status"] not in ("Z", "X", "x")
            else:
                assert actual is None or actual["status"] == "Z"
        before = queue.sha256(work / "state.json")
        with pytest.raises(FileExistsError):
            queue.run_phase(tmp_path, "freeze", argv, helpers, state)
        assert queue.sha256(work / "state.json") == before
        assert (work / "freeze.log").is_file()
    finally:
        if descendant is not None:
            actual = queue.owner.handoff.observed(descendant)
            if actual is not None and actual["status"] not in ("Z", "X", "x"):
                descriptor = pidfd_open(descendant)
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                    poll = select.poll()
                    poll.register(descriptor, select.POLLIN)
                    assert poll.poll(5000)
                finally:
                    os.close(descriptor)


def test_ns_queue_preserves_partial_freeze_and_refuses_second_launch(tmp_path, monkeypatch):
    work, helpers = tmp_path / queue.WORK, tmp_path / "helpers"
    work.mkdir(parents=True)
    helpers.mkdir()
    source = helpers / "freeze.py"
    source.write_text(
        "import json,os,select,sys\nfrom pathlib import Path\n"
        f"root=Path(sys.argv[1]);work=root/{queue.WORK!r};poll=select.poll()\n"
        "while json.loads((work/'state.json').read_text())['active_phase'].get('pid')!=os.getpid():poll.poll(1)\n"
        f"binding=root/{str(queue.owner.locations(tmp_path)[0].relative_to(tmp_path))!r}\n"
        "binding.parent.mkdir(parents=True,exist_ok=True);binding.write_text('{\"partial\":true}');sys.exit(7)\n"
    )
    monkeypatch.setattr(
        queue, "readiness", lambda root: {"ready": True, "predecessor": {"intervention_accepted": False}}
    )
    monkeypatch.setattr(
        queue, "phases", lambda root, inherited: [("freeze", [sys.executable, str(source), str(tmp_path)], helpers)]
    )
    state = {"status": "original_terminal_ready", "phases": []}
    with pytest.raises(RuntimeError):
        queue.launch(tmp_path, {}, state)
    binding = queue.owner.locations(tmp_path)[0]
    assert queue.owner.read(binding)["partial"] is True
    assert queue.owner.read(work / "state.json")["active_phase"]["exit_code"] == 7
    digest = queue.sha256(binding)
    with pytest.raises(FileExistsError):
        queue.launch(tmp_path, {}, state)
    assert queue.sha256(binding) == digest and state["phases"] == []


def test_ns_queue_dependency_closure_preserves_every_accepted_owner_helper():
    root = queue.owner.execution.replay.ROOT
    inherited, sources = queue.parent(root), queue.helper_sources(root)
    assert set(sources) == set(inherited["helpers_sha256"]) | set(queue.NEW_HELPERS)
    assert all(queue.sha256(sources[name]) == value for name, value in inherited["helpers_sha256"].items())


@pytest.mark.parametrize("outcome", ("controller_live", "worker_live", "ready", "failed"))
def test_ns_queue_waits_for_issued_owner_and_worker_absence_before_terminal(tmp_path, monkeypatch, outcome):
    binding, helpers, state_path, _ = queue.owner.locations(tmp_path)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    source = helpers / "ns_budget_gpu_controller.py"
    source.write_text("# issued controller fixture\n")
    hashes = {source.name: queue.sha256(source)}
    frozen = {
        "helpers_sha256": hashes,
        "controller_command": queue.owner.controller_command(tmp_path),
        "environment": queue.owner.ENVIRONMENT,
        "worker_environment": queue.owner.ENVIRONMENT,
        "cpu_receipt_sha256": queue.PARENT_SHA256,
    }
    write(binding, frozen)
    generation = {
        "pid": 111,
        "start_ticks": "111",
        "group": 111,
        "session": 111,
        "command": queue.owner.controller_command(tmp_path),
    }
    intent = {
        "pid": 111,
        "start_ticks": "111",
        "generation": generation,
        "binding_sha256": queue.sha256(binding),
        "marker": "issued-controller",
    }
    write(helpers.parent / "controller.json", intent)
    child = {
        "index": 0,
        "spawn_status": "acknowledged",
        "marker": "issued-worker",
        "generation": generation | {"pid": 222, "group": 222, "session": 222, "command": ["python", "worker"]},
    }
    saved = {
        "status": "needs_attention" if outcome == "failed" else "diagnostic_complete",
        "controller_pid": 111,
        "controller_start_ticks": "111",
        "binding_sha256": queue.sha256(binding),
        "executions": [child],
    }
    write(state_path, saved)
    monkeypatch.setattr(queue, "parent", lambda root: {"helpers_sha256": hashes})
    live = {"controller_live": 111, "worker_live": 222}.get(outcome)
    monkeypatch.setattr(
        queue.owner.handoff,
        "observed",
        lambda pid: (generation if pid == 111 else child["generation"]) | {"status": "R"} if pid == live else None,
    )
    monkeypatch.setattr(
        queue.owner, "process_context", lambda pid, *args: generation if pid == 111 else child["generation"]
    )
    state = {
        "issued_binding_sha256": queue.sha256(binding),
        "issued_controller": queue.owner.read(helpers.parent / "controller.json"),
    }
    before = queue.sha256(state_path)
    if outcome == "failed":
        with pytest.raises(RuntimeError):
            queue.issued_readiness(tmp_path, state)
    else:
        result = queue.issued_readiness(tmp_path, state)
        assert result["ready"] is (outcome == "ready")
    assert queue.sha256(state_path) == before and not (tmp_path / queue.TERMINAL).exists()


def test_real_ns_queue_terminal_retains_negative_interpretation_and_refuses_repeat(tmp_path, monkeypatch):
    _, helpers, _, _ = queue.owner.locations(tmp_path)
    helpers.mkdir(parents=True)
    work = tmp_path / queue.WORK
    work.mkdir(parents=True)
    binding = queue.owner.locations(tmp_path)[0]
    binding.parent.mkdir(parents=True, exist_ok=True)
    source = helpers / "ns_budget_gpu_controller.py"
    source.write_text(
        "import argparse,hashlib,json,os,select\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--phase');p.add_argument('--out');a=p.parse_args()\n"
        f"root=Path(a.root);work=root/{queue.WORK!r};poll=select.poll()\n"
        "while json.loads((work/'state.json').read_text())['active_phase'].get('pid')!=os.getpid():poll.poll(1)\n"
        "assert a.phase=='terminal' and os.environ['CUDA_VISIBLE_DEVICES']==''\n"
        "state=json.loads((work/'state.json').read_text())\n"
        "Path(a.out).write_text(json.dumps({'status':"
        "'ns_budget_terminal_independently_reconstructed_exact_restoration_verified',"
        "'binding_sha256':state['issued_binding_sha256'],"
        "'controller_pid_absent':state['issued_controller']['pid'],"
        "'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"
        "'evidence_sha256':{},'comparison':{'historical_controls_interpretable':False}}))\n"
    )
    write(binding, {"helpers_sha256": {source.name: queue.sha256(source)}})
    state = {
        "status": "issued_ns_terminal_ready",
        "phases": [],
        "issued_binding_sha256": queue.sha256(binding),
        "issued_controller": {"pid": 111},
    }
    monkeypatch.setattr(queue.owner, "PYTHON", sys.executable)
    monkeypatch.setattr(queue, "issued_readiness", lambda root, saved: {"ready": True})
    result = queue.terminal_phase(tmp_path, state)
    assert result["status"] == "ns_diagnostic_terminal_reconstructed"
    assert result["historical_controls_interpretable"] is False
    assert result["phases"][-1]["exit_code"] == result["phases"][-1]["cleanup_exit_code"] == 0
    digest = queue.sha256(tmp_path / queue.TERMINAL)
    with pytest.raises(FileExistsError):
        queue.terminal_phase(tmp_path, state)
    assert queue.sha256(tmp_path / queue.TERMINAL) == digest
