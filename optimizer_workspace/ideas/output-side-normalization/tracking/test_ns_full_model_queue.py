# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise original queue waiting and real full-model phase/terminal lifecycle."""

import os
import select
import signal
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_full_model_queue as queue
import pytest
from selected_lr_failure_handoff import pidfd_open
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def generation(pid, command, cwd):
    return {
        "pid": pid,
        "parent": 1,
        "start_ticks": str(pid),
        "group": pid,
        "session": pid,
        "status": "S",
        "command": command,
        "cwd": str(cwd),
    }


@pytest.fixture
def original_queue(tmp_path, monkeypatch):
    work = tmp_path / queue.predecessor.QUEUE
    work.mkdir(parents=True)
    own = generation(111, ["python", "original-queue"], work)
    intent = {"status": "acknowledged", "generation": own, "marker": "original-queue"}
    write(work / "intent.json", intent)
    write(work / "state.json", {"status": "waiting", "phases": []})
    monkeypatch.setattr(queue.predecessor, "QUEUE_INTENT_SHA256", sha256(work / "intent.json"))
    return tmp_path, work, own


@pytest.mark.parametrize("outcome", ("live", "ready", "failed", "changed_intent", "active_phase"))
def test_original_queue_readiness_waits_before_terminal_and_preserves_failure(original_queue, monkeypatch, outcome):
    root, work, own = original_queue
    phases = {}
    if outcome == "active_phase":
        child = generation(222, ["python", "original-phase"], work)
        phases[222] = child
        write(
            work / "state.json",
            {
                "phases": [],
                "active_phase": {
                    "phase": "freeze",
                    "generation": child,
                    "marker": "phase",
                    "cwd": str(work),
                    "spawn_status": "acknowledged",
                },
            },
        )
    if outcome == "changed_intent":
        write(work / "intent.json", {"status": "acknowledged", "generation": own, "marker": "different"})
    live = {111: own} if outcome == "live" else phases
    monkeypatch.setattr(queue.handoff, "observed", lambda pid: live.get(pid))
    monkeypatch.setattr(queue.predecessor.previous, "process_context", lambda pid, *args: live[pid])
    previous = {"historical_controls_interpretable": True, "restored_keeper": {"pid": 333}}

    def terminal(root):
        if live:
            pytest.fail("Terminal evidence inspected while an original queue owner remains")
        if outcome == "failed":
            raise RuntimeError("Original NS terminal rejected")
        return previous

    monkeypatch.setattr(queue.predecessor, "evidence", terminal)
    before = sha256(work / "state.json")
    if outcome in ("failed", "changed_intent"):
        with pytest.raises((RuntimeError, ValueError)):
            queue.readiness(root)
    else:
        result = queue.readiness(root)
        assert result["ready"] is (outcome == "ready")
        if outcome == "ready":
            assert result["predecessor"] == previous
        else:
            assert any(item["status"] == "live_exact_generation" for item in result["owners"].values())
    assert sha256(work / "state.json") == before
    assert not queue.contract.locations(root)[0].exists()


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached", "new_owner", "spawn_failure"))
def test_real_full_model_queue_phase_keeps_cleanup_and_separately_owned_controller(tmp_path, monkeypatch, outcome):
    work, helpers = tmp_path / queue.WORK, tmp_path / "helpers"
    work.mkdir(parents=True)
    helpers.mkdir()
    program = helpers / "phase.py"
    program.write_text(
        "import json,os,select,signal,subprocess,sys\nfrom pathlib import Path\n"
        f"work=Path(sys.argv[1])/{queue.WORK!r};outcome=sys.argv[2];poll=select.poll()\n"
        "while json.loads((work/'state.json').read_text())['active_phase'].get('pid')!=os.getpid():poll.poll(1)\n"
        "if outcome in ('detached','new_owner'):\n"
        " r,w=os.pipe();env=dict(os.environ)\n"
        f" if outcome=='new_owner':env[{queue.PHASE_MARKER!r}]='separate-controller'\n"
        " child=subprocess.Popen([sys.executable,'-c',"
        "'import os,signal,sys;os.write(int(sys.argv[1]),b\"R\");signal.pause()',str(w)],"
        "env=env,pass_fds=(w,),start_new_session=True)\n"
        " os.close(w);assert os.read(r,1)==b'R';os.close(r)\n"
        " (work/'detached.pid').write_text(str(child.pid))\n"
        "sys.exit(7 if outcome in ('nonzero','detached') else 0)\n"
    )
    if outcome == "spawn_failure":

        def fail(*args, **kwargs):
            raise OSError("External phase spawn failed")

        monkeypatch.setattr(queue.subprocess, "Popen", fail)
    state = {"status": "issuing_full_model_validation", "phases": []}
    argv = [sys.executable, str(program), str(tmp_path), outcome]
    child_pid = None
    try:
        if outcome in ("nonzero", "detached", "spawn_failure"):
            with pytest.raises((RuntimeError, OSError)):
                queue.run_phase(tmp_path, "freeze", argv, helpers, state)
        else:
            result = queue.run_phase(tmp_path, "freeze", argv, helpers, state)
            assert result["exit_code"] == result["cleanup_exit_code"] == 0
        saved = read(work / "state.json")["active_phase"]
        assert saved["log_sha256"] == sha256(work / "freeze.log")
        if outcome == "spawn_failure":
            assert saved["spawn_status"] == "starting" and "pid" not in saved
        else:
            assert saved["marked_descendants_absent"]
            assert queue.handoff.observed(saved["pid"]) is None
            assert not queue.handoff.marker_processes(queue.PHASE_MARKER, saved["marker"])
        if outcome in ("new_owner", "detached"):
            child_pid = int((work / "detached.pid").read_text())
            actual = queue.handoff.observed(child_pid)
            if outcome == "new_owner":
                assert actual is not None and actual["status"] not in ("Z", "X", "x")
            else:
                assert actual is None or actual["status"] == "Z"
        digest = sha256(work / "state.json")
        with pytest.raises(FileExistsError):
            queue.run_phase(tmp_path, "freeze", argv, helpers, state)
        assert sha256(work / "state.json") == digest
    finally:
        if child_pid is not None:
            actual = queue.handoff.observed(child_pid)
            if actual is not None and actual["status"] not in ("Z", "X", "x"):
                descriptor = pidfd_open(child_pid)
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                    poll = select.poll()
                    poll.register(descriptor, select.POLLIN)
                    assert poll.poll(5000)
                finally:
                    os.close(descriptor)


def test_partial_full_model_freeze_remains_and_second_launch_is_denied(tmp_path, monkeypatch):
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
    state = {"status": "original_ns_terminal_ready", "phases": []}
    with pytest.raises(RuntimeError):
        queue.launch(tmp_path, state)
    binding = queue.contract.locations(tmp_path)[0]
    assert read(binding)["partial"]
    assert read(work / "state.json")["active_phase"]["exit_code"] == 7
    digest = sha256(binding)
    with pytest.raises(FileExistsError):
        queue.launch(tmp_path, state)
    assert sha256(binding) == digest and state["phases"] == []


@pytest.mark.parametrize("outcome", ("controller_live", "validator_live", "ready", "failed", "marked_rank"))
def test_issued_full_model_owner_waits_for_absence_and_complete_clean_evidence(tmp_path, monkeypatch, outcome):
    binding, helpers, state_path, _, _ = queue.contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True)
    program = helpers / "ns_full_model_gpu_controller.py"
    program.write_text("# issued fixture\n")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    frozen = {
        "candidate": str(candidate),
        "helpers_sha256": {program.name: sha256(program)},
        "controller_command": queue.owner.controller_command(tmp_path),
        "cpu_receipt_sha256": queue.PARENT_SHA256,
        "environment": queue.contract.ENVIRONMENT,
    }
    write(binding, frozen)
    own = generation(111, frozen["controller_command"], helpers)
    child = generation(222, queue.owner.validator_command(tmp_path, candidate), candidate)
    intent = {
        "status": "acknowledged",
        "pid": 111,
        "start_ticks": own["start_ticks"],
        "binding_sha256": sha256(binding),
        "generation": own,
        "marker": "controller",
    }
    write(helpers.parent / "controller.json", intent)
    saved = {
        "status": "needs_attention" if outcome == "failed" else "validation_complete",
        "controller_pid": 111,
        "controller_start_ticks": own["start_ticks"],
        "binding_sha256": sha256(binding),
        "spawn_status": "acknowledged",
        "child_generation": child,
        "child_marker": "validator",
    }
    write(state_path, saved)
    live = {"controller_live": {111: own}, "validator_live": {222: child}}.get(outcome, {})
    monkeypatch.setattr(queue.handoff, "observed", lambda pid: live.get(pid))
    monkeypatch.setattr(queue.predecessor.previous, "process_context", lambda pid, *args: live[pid])
    monkeypatch.setattr(queue.handoff, "marker_processes", lambda *args: {333: {}} if outcome == "marked_rank" else {})
    state = {"issued_binding_sha256": sha256(binding), "issued_controller": read(helpers.parent / "controller.json")}
    before = sha256(state_path)
    if outcome in ("failed", "marked_rank"):
        with pytest.raises(RuntimeError):
            queue.issued_readiness(tmp_path, state)
    else:
        result = queue.issued_readiness(tmp_path, state)
        assert result["ready"] is (outcome == "ready")
    assert sha256(state_path) == before and not (tmp_path / queue.TERMINAL).exists()


@pytest.mark.parametrize("changed", (False, True))
def test_real_full_model_terminal_retains_rejected_cases_and_refuses_changed_origin(tmp_path, monkeypatch, changed):
    binding, helpers, _, _, _ = queue.contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    work = tmp_path / queue.WORK
    work.mkdir(parents=True)
    program = helpers / "ns_full_model_gpu_controller.py"
    program.write_text(
        "import argparse,hashlib,json,os,select\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--phase');p.add_argument('--out');a=p.parse_args()\n"
        f"root=Path(a.root);work=root/{queue.WORK!r};poll=select.poll()\n"
        "while json.loads((work/'state.json').read_text())['active_phase'].get('pid')!=os.getpid():poll.poll(1)\n"
        "assert a.phase=='terminal' and os.environ['CUDA_VISIBLE_DEVICES']==''\n"
        "state=json.loads((work/'state.json').read_text())\n"
        "Path(a.out).write_text(json.dumps({'status':"
        "'ns_full_model_terminal_independently_reconstructed_exact_restoration',"
        "'binding_sha256':state['issued_binding_sha256'],"
        "'controller_pid_absent':state['issued_controller']['pid'],"
        f"'verifier_sha256':{'\"changed\"' if changed else 'hashlib.sha256(Path(__file__).read_bytes()).hexdigest()'},"
        "'evidence_sha256':{},'accepted_cases':['accepted'],'rejected_cases':['rejected'],"
        "'all_cases_accepted':False}))\n"
    )
    write(binding, {"helpers_sha256": {program.name: sha256(program)}})
    state = {
        "status": "full_model_terminal_ready",
        "phases": [],
        "issued_binding_sha256": sha256(binding),
        "issued_controller": {"pid": 111},
    }
    monkeypatch.setattr(queue.contract, "PYTHON", sys.executable)
    monkeypatch.setattr(queue, "issued_readiness", lambda root, saved: {"ready": True})
    if changed:
        with pytest.raises(ValueError):
            queue.terminal_phase(tmp_path, state)
        assert (tmp_path / queue.TERMINAL).is_file()
        assert read(work / "state.json")["active_phase"]["cleanup_exit_code"] == 0
    else:
        result = queue.terminal_phase(tmp_path, state)
        assert result["status"] == "full_model_terminal_reconstructed"
        assert result["accepted_cases"] == ["accepted"] and result["rejected_cases"] == ["rejected"]
        assert result["all_cases_accepted"] is False
        assert result["phases"][-1]["exit_code"] == result["phases"][-1]["cleanup_exit_code"] == 0
    digest = sha256(tmp_path / queue.TERMINAL)
    with pytest.raises(RuntimeError):
        queue.terminal_phase(tmp_path, state)
    assert sha256(tmp_path / queue.TERMINAL) == digest


@pytest.fixture(scope="module")
def inherited():
    return read(ROOT / queue.PARENT)


@pytest.mark.parametrize("fault", ("none", "truncated_xml", "archive", "fixture", "mutable_import"))
def test_real_queue_cpu_archive_retains_parent_and_rejects_rehashed_incomplete_evidence(
    tmp_path, monkeypatch, inherited, fault
):
    root, out = tmp_path / "root", tmp_path / "cpu"
    root.mkdir()
    monkeypatch.setattr(queue, "PARENT", str(ROOT / queue.PARENT))
    monkeypatch.setattr(queue, "parent", lambda root: inherited)
    for name, value in queue.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    def completed(command, **kwargs):
        xml = ET.Element("testsuites")
        suite = ET.SubElement(xml, "testsuite", tests=str(queue.TEST_COUNT), failures="0", errors="0", skipped="0")
        for index in range(queue.TEST_COUNT):
            ET.SubElement(suite, "testcase", classname="test_ns_full_model_queue", name=f"external_case_{index}")
        ET.ElementTree(xml).write(out / "tests.xml")
        kwargs["stdout"].write("External CPU test runner complete\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(queue.subprocess, "run", completed)
    receipt = queue.cpu(root, out)
    assert all(receipt["helpers_sha256"][name] == digest for name, digest in inherited["helpers_sha256"].items())
    assert len(receipt["helpers_sha256"]) == 136
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
        mutable = root / "mutable_predecessor.py"
        mutable.write_bytes((out / "tracking/ns_full_model_predecessor.py").read_bytes())
        probe = root / "archive_probe.py"
        probe.write_text(
            "import importlib.util,json,sys\nfrom pathlib import Path\n"
            "root,receipt,parent,mutable=map(Path,sys.argv[1:5]);fault=sys.argv[5]\n"
            "import ns_full_model_queue as queue\n"
            "queue.PARENT=str(parent);queue.parent=lambda root:queue.read(parent)\n"
            "if fault=='mutable_import':\n"
            " spec=importlib.util.spec_from_file_location('ns_full_model_predecessor',mutable)\n"
            " mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)\n"
            "cpu=queue.accepted(root,receipt)\n"
            "print(json.dumps({'status':'archived_queue_roundtrip','helpers':len(cpu['helpers_sha256'])}))\n"
        )
        checked = subprocess.run(
            [
                sys.executable,
                "-P",
                str(probe),
                str(root),
                str(out / "receipt.json"),
                str(ROOT / queue.PARENT),
                str(mutable),
                fault,
            ],
            cwd=out,
            env=os.environ
            | {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": f"{out / 'tracking'}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        if fault == "mutable_import":
            assert checked.returncode != 0
            return
        assert checked.returncode == 0, checked.stderr
        assert '"archived_queue_roundtrip"' in checked.stdout and '"helpers": 136' in checked.stdout
        monkeypatch.setattr(queue, "PARENT", str(ROOT / queue.PARENT))
        monkeypatch.setattr(queue, "parent", lambda root: inherited)
        assert queue.require_cpu(root, out / "receipt.json") == receipt
    else:
        with pytest.raises(ValueError):
            queue.require_cpu(root, out / "receipt.json")


@pytest.mark.parametrize("outcome", ("zero", "spawn_failure"))
def test_real_queue_start_acknowledges_original_owner_and_retains_failed_attempt(tmp_path, monkeypatch, outcome):
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
    cpu_path.write_text("{}")
    receipt = {"helpers_sha256": {queue.NEW_HELPERS[0]: "CPU-runner"}, "helpers": str(helpers)}
    monkeypatch.setattr(queue, "accepted", lambda *args: receipt)
    monkeypatch.setattr(queue, "readiness", lambda root: {"ready": False, "status": "waiting_for_original_ns_queue"})
    monkeypatch.setattr(queue, "local_node", lambda: queue.owner.NODE)
    argv = [sys.executable, str(program), str(work / "intent.json")]
    monkeypatch.setattr(queue, "command", lambda *args: argv)
    children, original = [], subprocess.Popen

    def spawn(*args, **kwargs):
        if outcome == "spawn_failure":
            raise OSError("External queue startup failed")
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(queue.subprocess, "Popen", spawn)
    try:
        if outcome == "zero":
            intent = queue.start(tmp_path, cpu_path)
            assert intent["status"] == "acknowledged"
            assert intent["generation"]["parent"] == os.getpid()
            assert intent["generation"]["session"] == intent["generation"]["group"] == intent["pid"]
            assert children[0].wait(timeout=10) == 0
        else:
            with pytest.raises(OSError):
                queue.start(tmp_path, cpu_path)
            intent = read(work / "intent.json")
            assert intent["status"] == "needs_attention" and "OSError" in intent["error"]
        with pytest.raises(RuntimeError):
            queue.start(tmp_path, cpu_path)
        assert not queue.contract.locations(tmp_path)[0].exists()
    finally:
        for child in children:
            queue.finish_marked_child(child, read(work / "intent.json")["marker"], {}, queue.MARKER)
