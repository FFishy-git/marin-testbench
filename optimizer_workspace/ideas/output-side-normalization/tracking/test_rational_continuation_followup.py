# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise original-owner readiness and one-shot read-only verifier execution."""

import sys

import pytest
import rational_continuation_followup as followup
from selected_lr_release_owner import read, sha256
from watch_wave5 import write


@pytest.mark.parametrize("outcome", ("live", "absent", "failed", "substituted", "zombie"))
def test_original_owner_readiness_waits_for_exact_handles_and_preserves_failure(tmp_path, monkeypatch, outcome):
    _, helpers, state_path, _, _ = followup.contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    commands = {"controller_command": ["python", "controller"], "validator_command": ["python", "validator"]}
    expected = {
        "pid": 111,
        "start_ticks": "1234",
        "parent": 1,
        "group": 111,
        "session": 111,
        "status": "S",
        "command": commands["controller_command"],
    }
    child = expected | {
        "pid": 222,
        "parent": 111,
        "group": 222,
        "session": 222,
        "command": commands["validator_command"],
    }
    intent = {
        "status": "acknowledged",
        "binding_sha256": followup.BINDING_SHA256,
        "pid": 111,
        "start_ticks": "1234",
        "command": commands["controller_command"],
        "generation": expected,
        "marker": "controller",
    }
    state = {
        "status": "needs_attention" if outcome == "failed" else "diagnostic_complete",
        "binding_sha256": followup.BINDING_SHA256,
        "controller_pid": 111,
        "controller_start_ticks": "1234",
        "command": commands["validator_command"],
        "child_generation": child,
        "child_marker": "validator",
    }
    write(helpers.parent / "controller.json", intent)
    write(state_path, state)
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(followup.handoff, "PROC_ROOT", proc)
    if outcome in ("live", "substituted", "zombie"):
        directory = proc / "111"
        directory.mkdir()
        ticks = "9999" if outcome == "substituted" else "1234"
        status = "Z" if outcome == "zombie" else "S"
        (directory / "stat").write_text(f"111 (fixture) {status} 1 111 111 " + "0 " * 15 + ticks + " 0\n")
        (directory / "cmdline").write_bytes(b"python\0controller\0")
        environment = followup.contract.ENVIRONMENT | {followup.contract.CONTROLLER_MARKER: "controller"}
        (directory / "environ").write_bytes(b"\0".join(f"{key}={value}".encode() for key, value in environment.items()))
        (directory / "cwd").symlink_to(helpers)
    before = sha256(state_path)
    if outcome in ("failed", "substituted"):
        with pytest.raises(RuntimeError):
            followup.readiness(tmp_path, commands | {"candidate": str(candidate)})
    else:
        observed = followup.readiness(tmp_path, commands | {"candidate": str(candidate)})
        assert observed["ready"] is (outcome == "absent")
        if outcome == "zombie":
            assert observed["owners"]["controller"]["status"] == "exited_handle_present"
        elif outcome == "live":
            assert observed["owners"]["controller"]["status"] == "live_exact_generation"
    assert sha256(state_path) == before


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached"))
def test_real_terminal_phase_retains_result_exit_cleanup_and_refuses_second_execution(tmp_path, monkeypatch, outcome):
    helpers, work = tmp_path / "helpers", tmp_path / "work"
    helpers.mkdir()
    work.mkdir()
    (tmp_path / "run-preparation").mkdir()
    source = helpers / "rational_continuation_controller.py"
    source.write_text(
        "import argparse,json,os,select,signal,subprocess,sys\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--phase');p.add_argument('--out')\n"
        "a=p.parse_args();state=Path(a.root)/'work/state.json';poll=select.poll()\n"
        "while json.loads(state.read_text()).get('terminal_pid')!=os.getpid():poll.poll(1)\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='' and a.phase=='terminal'\n"
        f"outcome={outcome!r}\n"
        "if outcome=='detached':\n"
        " r,w=os.pipe()\n"
        " child=subprocess.Popen([sys.executable,'-c',"
        "'import os,signal,sys;os.write(int(sys.argv[1]),b\"R\");signal.pause()',str(w)],"
        "pass_fds=(w,),start_new_session=True)\n"
        " os.close(w);assert os.read(r,1)==b'R';os.close(r)\n"
        " Path(a.root,'detached.pid').write_text(str(child.pid))\n"
        "if outcome=='nonzero':sys.exit(7)\n"
        "Path(a.out).write_text(json.dumps({'status':"
        "'rational_continuation_terminal_independently_reconstructed_exact_restoration',"
        "'analysis':{'diagnostic_intervention_accepted':False}}))\n"
    )
    monkeypatch.setattr(followup.contract, "PYTHON", sys.executable)
    state = {"status": "terminal_ready"}
    if outcome == "nonzero":
        with pytest.raises(RuntimeError):
            followup.terminal_phase(tmp_path, helpers, work, state)
    else:
        followup.terminal_phase(tmp_path, helpers, work, state)
        assert state["status"] == "terminal_reconstructed"
        assert state["intervention_accepted"] is False
        assert state["terminal_sha256"] == sha256(tmp_path / followup.TERMINAL)
    saved = read(work / "state.json")
    assert saved["terminal_exit_code"] == (7 if outcome == "nonzero" else 0)
    assert saved["terminal_cleanup_exit_code"] == saved["terminal_exit_code"]
    assert saved["terminal_descendants_absent"] is True
    assert followup.handoff.observed(saved["terminal_pid"]) is None
    assert not followup.handoff.marker_processes(followup.CHILD_MARKER, saved["terminal_marker"])
    assert saved["terminal_log_sha256"] == sha256(work / "terminal.log")
    before = sha256(work / "state.json")
    with pytest.raises(FileExistsError):
        followup.terminal_phase(tmp_path, helpers, work, state)
    assert sha256(work / "state.json") == before
    if outcome == "detached":
        process = followup.handoff.observed(int((tmp_path / "detached.pid").read_text()))
        assert process is None or process["status"] == "Z"
