# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test exact-owner waiting and real one-shot verifier cleanup on CPU."""

import sys
from pathlib import Path

import ns_training_path_followup as followup
import pytest
from selected_lr_release_owner import read, sha256
from watch_wave5 import write


@pytest.fixture
def original(tmp_path, monkeypatch):
    binding_path, helpers, state_path, _, result_path = followup.contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    commands = {"controller_command": ["python", "controller"], "validator_command": ["python", "validator"]}
    controller = {
        "pid": 111,
        "start_ticks": "1234",
        "parent": 1,
        "group": 111,
        "session": 111,
        "status": "S",
        "command": commands["controller_command"],
    }
    validator = controller | {
        "pid": 222,
        "parent": 111,
        "group": 222,
        "session": 222,
        "command": commands["validator_command"],
    }
    intent = {
        "status": "acknowledged",
        "binding_sha256": followup.BINDING_SHA256,
        "pid": controller["pid"],
        "start_ticks": controller["start_ticks"],
        "command": controller["command"],
        "generation": controller,
        "marker": "original-controller",
    }
    result = {
        "status": "ns_training_path_actual_config_gates_reconstructed_not_production",
        "binding_sha256": followup.BINDING_SHA256,
        "gpu_commands": [{"marker": "original-command"}],
        "results": [{"accepted": False}],
        "accepted_cases": [],
        "rejected_cases": ["rejected-case"],
        "all_cases_accepted": False,
    }
    write(result_path, result)
    state = {
        "status": "validation_complete",
        "binding_sha256": followup.BINDING_SHA256,
        "controller_pid": controller["pid"],
        "controller_start_ticks": controller["start_ticks"],
        "command": validator["command"],
        "child_generation": validator,
        "child_pid": validator["pid"],
        "child_start_ticks": validator["start_ticks"],
        "child_marker": "original-validator",
        "keeper_release_complete": True,
        "marked_descendants_absent": True,
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "restored_keeper": {"fixture": "original keeper"},
        "result_sha256": sha256(result_path),
        **{key: result[key] for key in ("accepted_cases", "rejected_cases", "all_cases_accepted")},
    }
    write(helpers.parent / "controller.json", intent)
    write(state_path, state)
    binding = commands | {
        "root": str(tmp_path),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "environment": followup.contract.ENVIRONMENT,
        "controller_environment": followup.ENVIRONMENT,
        "commands": [controller["command"]],
        "plan": {"trials": [{"fixture": "rejected-case"}]},
    }
    write(binding_path, binding)
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(followup.handoff, "PROC_ROOT", proc)
    return tmp_path, binding, state, proc


def put_process(proc, expected, cwd, environment, status="S", ticks=None):
    directory = proc / str(expected["pid"])
    directory.mkdir()
    (directory / "stat").write_text(
        f"{expected['pid']} (fixture) {status} {expected['parent']} {expected['group']} {expected['session']} "
        + "0 " * 15
        + (ticks or expected["start_ticks"])
        + " 0\n"
    )
    (directory / "cmdline").write_bytes(b"\0".join(value.encode() for value in expected["command"]) + b"\0")
    (directory / "environ").write_bytes(b"\0".join(f"{key}={value}".encode() for key, value in environment.items()))
    (directory / "cwd").symlink_to(cwd)


@pytest.mark.parametrize(
    "outcome", ("controller_live", "validator_live", "absent", "failed", "substituted", "zombie", "descendant")
)
def test_original_readiness_requires_actual_absence_and_retains_rejections(original, outcome):
    root, binding, state, proc = original
    _, helpers, state_path, _, result_path = followup.contract.locations(root)
    intent = read(helpers.parent / "controller.json")
    if outcome in ("controller_live", "substituted", "zombie"):
        put_process(
            proc,
            intent["generation"],
            helpers,
            binding["controller_environment"] | {followup.contract.CONTROLLER_MARKER: intent["marker"]},
            status="Z" if outcome == "zombie" else "S",
            ticks="9999" if outcome == "substituted" else None,
        )
    elif outcome == "validator_live":
        put_process(
            proc,
            state["child_generation"],
            Path(binding["candidate"]),
            binding["environment"] | {followup.contract.CHILD_MARKER: state["child_marker"]},
        )
    elif outcome == "descendant":
        child = intent["generation"] | {"pid": 333, "group": 333, "session": 333, "command": ["python", "detached"]}
        put_process(proc, child, root, {followup.contract.COMMAND_MARKER: "original-command"})
    elif outcome == "failed":
        state.update(status="needs_attention", error="original failure")
        write(state_path, state)
    before = {path: sha256(path) for path in (state_path, result_path)}
    if outcome in ("failed", "substituted"):
        with pytest.raises(RuntimeError):
            followup.readiness(root, binding)
    else:
        observation = followup.readiness(root, binding)
        assert observation["ready"] is (outcome == "absent")
        assert observation["case_decisions"] == 1
        if outcome == "zombie":
            assert observation["owners"]["controller"]["status"] == "exited_handle_present"
        elif outcome == "descendant":
            assert observation["status"] == "waiting_for_original_descendants"
            assert observation["marked_pids"] == [333]
    assert before == {path: sha256(path) for path in before}
    assert read(result_path)["rejected_cases"] == ["rejected-case"]


@pytest.mark.parametrize("existing", ("claimed", "failed", "completed"))
def test_existing_canonical_terminal_claim_is_never_repeated(original, existing):
    root, binding, _, _ = original
    helpers = Path(binding["helpers"])
    terminal, attempt, outcome, _ = followup.controller.terminal_paths(root, read(helpers.parent / "controller.json"))
    path = terminal if existing == "completed" else outcome if existing == "failed" else attempt
    write(path, {"status": existing})
    digest = sha256(path)
    with pytest.raises(FileExistsError):
        followup.readiness(root, binding)
    assert sha256(path) == digest


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached", "bad_outcome"))
def test_real_terminal_child_preserves_rejection_canonical_output_and_cleanup(original, monkeypatch, outcome):
    root, binding, _, _ = original
    helpers = Path(binding["helpers"])
    work = root / "followup"
    work.mkdir()
    monkeypatch.setattr(followup.handoff, "PROC_ROOT", Path("/proc"))
    source = helpers / "ns_training_path_gpu_controller.py"
    source.write_text(
        "import argparse,hashlib,json,os,select,signal,subprocess,sys\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--phase');p.add_argument('--out')\n"
        "a=p.parse_args();root=Path(a.root);state=root/'followup/state.json';poll=select.poll()\n"
        "while json.loads(state.read_text()).get('terminal_pid')!=os.getpid():poll.poll(1)\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='' and a.phase=='terminal'\n"
        f"outcome={outcome!r};binding_sha256={followup.BINDING_SHA256!r}\n"
        "attempt=Path(__file__).parent.parent/'terminal-intent.json'\n"
        "attempt.write_text(json.dumps({'status':'reconstruction_claimed'}))\n"
        "if outcome=='nonzero':sys.exit(7)\n"
        "if outcome=='detached':\n"
        " r,w=os.pipe()\n"
        " child=subprocess.Popen([sys.executable,'-c',"
        "'import os,signal,sys;os.write(int(sys.argv[1]),b\"R\");signal.pause()',str(w)],"
        "pass_fds=(w,),start_new_session=True)\n"
        " os.close(w);assert os.read(r,1)==b'R';os.close(r)\n"
        " (root/'detached.pid').write_text(str(child.pid))\n"
        "digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()\n"
        "receipt={'status':'ns_training_path_terminal_independently_reconstructed_exact_restoration_not_production',"
        "'binding_sha256':binding_sha256,'production_authorized':False,'accepted_cases':[],"
        "'rejected_cases':['rejected-case'],'all_cases_accepted':False,'reconstruction_intent_sha256':digest(attempt)}\n"
        "out=Path(a.out);out.write_text(json.dumps(receipt))\n"
        "finished={'status':'complete','output':str(out),'terminal_sha256':digest(out),"
        "'reconstruction_intent_sha256':digest(attempt)}\n"
        "if outcome=='bad_outcome':finished['terminal_sha256']='incorrect'\n"
        "attempt.with_name('terminal-outcome.json').write_text(json.dumps(finished))\n"
    )
    monkeypatch.setattr(followup.contract, "PYTHON", sys.executable)
    state = {"status": "terminal_ready"}
    if outcome in ("nonzero", "bad_outcome"):
        with pytest.raises(RuntimeError if outcome == "nonzero" else ValueError):
            followup.terminal_phase(root, binding, work, state)
    else:
        followup.terminal_phase(root, binding, work, state)
        assert state["status"] == "terminal_reconstructed"
        assert state["rejected_cases"] == ["rejected-case"]
        assert state["all_cases_accepted"] is False
        assert state["terminal_sha256"] == sha256(Path(state["terminal"]))
    saved = read(work / "state.json")
    assert saved["terminal_exit_code"] == (7 if outcome == "nonzero" else 0)
    assert saved["terminal_cleanup_exit_code"] == saved["terminal_exit_code"]
    assert saved["terminal_descendants_absent"] is True
    assert followup.handoff.observed(saved["terminal_pid"]) is None
    assert not followup.handoff.marker_processes(followup.CHILD_MARKER, saved["terminal_marker"])
    assert saved["terminal_log_sha256"] == sha256(work / "terminal.log")
    before = sha256(work / "state.json")
    with pytest.raises(FileExistsError):
        followup.terminal_phase(root, binding, work, state)
    assert sha256(work / "state.json") == before
    if outcome == "detached":
        observed = followup.handoff.observed(int((root / "detached.pid").read_text()))
        assert observed is None or observed["status"] == "Z"


@pytest.mark.parametrize("fault", ("result_hash", "decision", "bool_cleanup"))
def test_original_completion_mismatch_refuses_before_terminal_claim(original, fault):
    root, binding, state, _ = original
    helpers = Path(binding["helpers"])
    if fault == "result_hash":
        state["result_sha256"] = "incorrect"
    elif fault == "decision":
        state["all_cases_accepted"] = True
    else:
        state["cleanup_exit_code"] = False
    write(followup.contract.locations(root)[2], state)
    with pytest.raises(RuntimeError if fault == "bool_cleanup" else ValueError):
        followup.readiness(root, binding)
    terminal, attempt, outcome, _ = followup.controller.terminal_paths(root, read(helpers.parent / "controller.json"))
    assert not any(path.exists() for path in (terminal, attempt, outcome))


def test_terminal_spawn_failure_retains_attempt_without_claiming_original(original, monkeypatch):
    root, binding, _, _ = original
    monkeypatch.setattr(followup.contract, "PYTHON", str(root / "missing-python"))
    work = root / "followup"
    work.mkdir()
    state = {"status": "terminal_ready"}
    with pytest.raises(OSError):
        followup.terminal_phase(root, binding, work, state)
    saved = read(work / "state.json")
    assert saved["terminal_command"] == state["terminal_command"]
    assert saved["terminal_log_sha256"] == sha256(work / "terminal.log")
    helpers = Path(binding["helpers"])
    terminal, attempt, outcome, _ = followup.controller.terminal_paths(root, read(helpers.parent / "controller.json"))
    assert not any(path.exists() for path in (terminal, attempt, outcome))
    with pytest.raises(FileExistsError):
        followup.terminal_phase(root, binding, work, state)


def test_premature_terminal_refuses_without_consuming_original_claim(original):
    root, binding, state, proc = original
    helpers = Path(binding["helpers"])
    intent = read(helpers.parent / "controller.json")
    put_process(
        proc,
        intent["generation"],
        helpers,
        binding["controller_environment"] | {followup.contract.CONTROLLER_MARKER: intent["marker"]},
    )
    work = root / "followup"
    work.mkdir()
    with pytest.raises(RuntimeError):
        followup.terminal_phase(root, binding, work, state)
    terminal, attempt, outcome, _ = followup.controller.terminal_paths(root, intent)
    assert not any(path.exists() for path in (terminal, attempt, outcome, work / "terminal.log", work / "state.json"))
