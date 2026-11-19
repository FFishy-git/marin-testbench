# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test isolated transition readiness and inherited one-shot successor lifecycle behavior."""

import ast
import os
import select
import signal
import subprocess
from pathlib import Path

import ns_training_path_transition_lr_queue as queue
import ns_training_path_transition_lr_queue_parent as parent
import ns_training_path_transition_lr_queue_test_support as support
import pytest
from selected_lr_failure_handoff import pidfd_open
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

EXCLUDED = {
    "test_training_path_lr_waits_for_real_original_owner_and_preserves_terminal",
    "test_training_path_lr_requires_complete_original_terminal_and_retains_rejections",
    "test_partial_training_path_lr_seal_is_retained_and_cannot_relaunch",
    "test_training_path_lr_queue_refuses_retained_failed_phase_without_reissuing_seal",
    "test_training_path_lr_launch_binds_original_seal_and_preserves_real_acknowledged_successor",
    "test_training_path_lr_queue_start_acknowledges_one_original_hidden_cpu_waiter",
}


def inherited_tests() -> bytes:
    tree = queue.transform(Path(__file__).with_name("test_ns_training_path_lr_queue.py"))
    tree.body = [
        node
        for node in tree.body
        if not (
            (isinstance(node, ast.FunctionDef) and node.name in EXCLUDED)
            or (
                isinstance(node, ast.Import)
                and any(alias.name == "ns_training_path_lr_predecessor_test_support" for alias in node.names)
            )
        )
    ]
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "test_training_path_lr_execute_retains_900_second_original_observation_without_gpu_phase"
        ):
            node.body[0] = ast.parse("root=tmp_path").body[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if node.value == 180:
                node.value = 198
            elif isinstance(node.value, str):
                node.value = node.value.replace(
                    "ns_training_path_transition_lr_parent", "ns_training_path_transition_lr_queue_parent"
                )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())


@pytest.fixture
def external_terminal(tmp_path, monkeypatch):
    helpers = tmp_path / "external-terminal"
    helpers.mkdir()
    receipt_path = tmp_path / parent.bridge.CPU
    receipt_path.parent.mkdir(parents=True)
    write(receipt_path, {"helpers": str(helpers)})
    observation = helpers / "observation.json"
    write(
        observation,
        {
            "status": "waiting_for_transition_owners",
            "ready": False,
            "owners": {"controller": {"status": "live_exact_generation"}},
            "marked_pids": [],
        },
    )
    (helpers / parent.bridge.CHECKER).write_text(
        "from pathlib import Path\n"
        "def accepted(root,path):\n"
        " assert Path(__file__).parent==Path(__file__).resolve().parent\n"
        " assert path.is_file()\n"
        " return {}\n"
    )
    (helpers / parent.bridge.VERIFIER).write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "def readiness(root):\n"
        " folder=Path(__file__).parent\n"
        " value=json.loads((folder/'observation.json').read_text())\n"
        " (folder/'called.json').write_text(json.dumps({'operation':'readiness','source':__file__}))\n"
        " if value.get('exit'):sys.exit(7)\n"
        " if value.get('substitute'):(folder/'observation.json').write_text('{}')\n"
        " if value.get('detached'):\n"
        "  import os,subprocess,select\n"
        "  child=subprocess.Popen([sys.executable,'-c',"
        "'import signal;signal.pause()'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        "  (folder/'detached.json').write_text(json.dumps({'pid':child.pid,"
        "  'marker':os.environ['OUTPUT_MARIN_NS_TRANSITION_LR_QUEUE_OBSERVER']}))\n"
        "  if value.get('timeout'):select.poll().poll(30000)\n"
        " return value\n"
        "def evidence(root):\n"
        " value=readiness(root)\n"
        " if not value['ready']:raise RuntimeError('Original owner remains live')\n"
        " return {'original_terminal':'canonical-terminal','original_terminal_sha256':'retained',"
        "'original_decisions':{'accepted_cases':['accepted'],'rejected_cases':['rejected'],"
        "'all_cases_accepted':False}}\n"
    )
    monkeypatch.setattr(
        parent.bridge,
        "cpu_sources",
        lambda root: {"helpers": str(helpers), "observation_sha256": sha256(observation)},
    )
    for name, value in queue.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    return tmp_path, helpers, observation


@pytest.mark.parametrize("operation", ("readiness", "evidence"))
@pytest.mark.parametrize("fault", ("none", "exit", "substitute"))
def test_external_terminal_observation_uses_separate_real_archive_and_refuses_drift(external_terminal, operation, fault):
    root, helpers, observation = external_terminal
    write(observation, read(observation) | {"ready": True, fault: fault != "none"})
    before = sha256(observation)
    if fault == "none":
        result = parent.inspect(root, operation)
        assert sha256(observation) == before
        if operation == "readiness":
            assert result["ready"] is True
        else:
            assert result["original_decisions"]["rejected_cases"] == ["rejected"]
            assert result["original_decisions"]["all_cases_accepted"] is False
    else:
        with pytest.raises(RuntimeError if fault == "exit" else ValueError):
            parent.inspect(root, operation)
    called = read(helpers / "called.json")
    assert Path(called["source"]) == helpers / parent.bridge.VERIFIER
    assert not (root / parent.SEAL).exists()
    assert not queue.contract.locations(root)[0].exists()


@pytest.mark.parametrize("outcome", ("live", "ready", "incomplete"))
def test_transition_queue_readiness_requires_external_terminal_and_retains_rejections(external_terminal, outcome):
    root, _, observation = external_terminal
    write(observation, read(observation) | {"ready": outcome != "live"})
    if outcome == "incomplete":
        write(observation, read(observation) | {"exit": True})
        with pytest.raises(RuntimeError):
            queue.readiness(root)
    else:
        result = queue.readiness(root)
        assert result["ready"] is (outcome == "ready")
        if outcome == "ready":
            assert result["original_decisions"]["rejected_cases"] == ["rejected"]
    assert not (root / parent.SEAL).exists()
    assert not queue.contract.locations(root)[0].exists()


@pytest.mark.parametrize("phase", ("seal", "check-seal"))
def test_transition_predecessor_command_targets_external_archive(external_terminal, monkeypatch, phase):
    root, helpers, _ = external_terminal
    monkeypatch.setattr(parent.bridge, "require_cpu", lambda root: {"helpers": str(helpers)})
    digest = "a" * 64 if phase == "check-seal" else None
    argv, cwd = queue.predecessor_command(root, phase, digest)
    assert cwd == helpers
    program = helpers / parent.bridge.CHECKER
    program.write_text(
        "import argparse,json,os\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--phase');"
        "p.add_argument('--cpu-receipt');p.add_argument('--seal-sha256');a=p.parse_args()\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES']==''\n"
        "(Path(a.root)/'external-phase.json').write_text(json.dumps(vars(a)))\n"
    )
    result = subprocess.run(argv, cwd=cwd, env=os.environ | queue.ENVIRONMENT, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    saved = read(root / "external-phase.json")
    assert saved == {
        "root": str(root),
        "phase": phase,
        "cpu_receipt": str(root / parent.bridge.CPU),
        "seal_sha256": digest,
    }


@pytest.mark.parametrize("outcome", ("live", "failed", "consumed"))
def test_transition_queue_cannot_launch_before_complete_external_terminal(external_terminal, outcome):
    root, _, observation = external_terminal
    work = root / queue.WORK
    work.mkdir(parents=True)
    state = {"status": "waiting", "phases": []}
    if outcome == "failed":
        write(observation, read(observation) | {"exit": True})
    elif outcome == "consumed":
        write(root / parent.SEAL, {"partial": True})
    with pytest.raises(FileExistsError if outcome == "consumed" else RuntimeError):
        queue.launch(root, state)
    assert state["phases"] == []
    assert not queue.contract.locations(root)[0].exists()


@pytest.mark.parametrize("outcome", ("success", "partial_seal", "retained_failure"))
def test_transition_queue_launch_binds_seal_and_preserves_independent_successor(external_terminal, monkeypatch, outcome):
    root, _, observation = external_terminal
    write(observation, read(observation) | {"ready": True})
    sealed = parent.evidence(root)
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
                str(helpers / parent.bridge.CHECKER),
                "--root",
                str(root),
                "--phase",
                phase,
                *(["--seal-sha256", seal_sha256] if seal_sha256 else []),
            ],
            helpers,
        ),
    )

    def checked_seal(root, digest):
        assert sha256(root / parent.SEAL) == digest
        return read(root / parent.SEAL)

    monkeypatch.setattr(parent, "require_seal", checked_seal)
    state = {"status": "waiting", "phases": []}
    if outcome == "retained_failure":
        state["phases"] = [{"phase": "seal", "exit_code": 7}]
        before = dict(state)
        with pytest.raises(RuntimeError):
            queue.launch(root, state)
        assert state == before and not (root / parent.SEAL).exists()
        return
    if outcome == "partial_seal":
        (helpers / parent.bridge.CHECKER).write_text(
            "import json,os,select,sys\nfrom pathlib import Path\n"
            f"root=Path(sys.argv[sys.argv.index('--root')+1]);work=root/{queue.WORK!r};poll=select.poll()\n"
            "while json.loads((work/'state.json').read_text())['active_phase'].get('pid')!=os.getpid():poll.poll(1)\n"
            f"(root/{parent.SEAL!r}).write_text('{{\"partial\":true}}');sys.exit(7)\n"
        )
    try:
        if outcome == "success":
            result = queue.launch(root, state)
            assert [entry["phase"] for entry in result["phases"]] == ["seal", "check-seal", "freeze", "check", "start"]
            assert all(entry["exit_code"] == entry["cleanup_exit_code"] == 0 for entry in result["phases"])
            digest = sha256(root / parent.SEAL)
            assert read(work / "freeze-argv.json")["seal_sha256"] == result["predecessor_seal_sha256"] == digest
            assert result["original_decisions"]["rejected_cases"] == ["rejected"]
            own = queue.handoff.observed(result["issued_controller"]["pid"])
            assert own is not None and own["status"] not in ("Z", "X", "x")
            assert own["start_ticks"] == result["issued_controller"]["start_ticks"]
            assert read(queue.contract.locations(root)[0])["allocation_predecessor"] == sealed
        else:
            with pytest.raises(RuntimeError):
                queue.launch(root, state)
            assert read(root / parent.SEAL)["partial"] is True
            assert read(work / "state.json")["active_phase"]["exit_code"] == 7
            assert not queue.contract.locations(root)[0].exists()
        before = sha256(root / parent.SEAL)
        with pytest.raises(FileExistsError):
            queue.launch(root, state)
        assert sha256(root / parent.SEAL) == before
    finally:
        if (work / "owner.pid").is_file():
            descriptor = pidfd_open(int((work / "owner.pid").read_text()))
            try:
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                poll = select.poll()
                poll.register(descriptor, select.POLLIN)
                assert poll.poll(5000)
            finally:
                os.close(descriptor)


@pytest.mark.parametrize("outcome", ("success", "spawn_failure"))
def test_transition_queue_start_acknowledges_only_one_hidden_cpu_waiter(external_terminal, monkeypatch, outcome):
    root, helpers, _ = external_terminal
    work = root / queue.WORK
    program = helpers / "queue.py"
    program.write_text(
        "import json,os,select,sys\nfrom pathlib import Path\n"
        "intent=Path(sys.argv[1]);poll=select.poll()\n"
        "while json.loads(intent.read_text()).get('pid')!=os.getpid():poll.poll(1)\n"
    )
    cpu = root / "receipt.json"
    write(cpu, {})
    receipt = {"helpers_sha256": {queue.NEW_HELPERS[0]: "CPU-runner"}, "helpers": str(helpers)}
    monkeypatch.setattr(queue, "accepted", lambda *args: receipt)
    monkeypatch.setattr(queue, "local_node", lambda: queue.owner.NODE)
    argv = [queue.contract.PYTHON, str(program), str(work / "intent.json")]
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
        if outcome == "success":
            intent = queue.start(root, cpu)
            assert intent["status"] == "acknowledged"
            assert intent["generation"]["parent"] == os.getpid()
            assert intent["generation"]["session"] == intent["generation"]["group"] == intent["pid"]
            assert children[0].wait(timeout=10) == 0
        else:
            with pytest.raises(OSError):
                queue.start(root, cpu)
            assert read(work / "intent.json")["status"] == "needs_attention"
        with pytest.raises(FileExistsError):
            queue.start(root, cpu)
        assert not queue.contract.locations(root)[0].exists()
    finally:
        for child in children:
            queue.finish_marked_child(child, read(work / "intent.json")["marker"], {}, queue.MARKER)


@pytest.mark.parametrize("outcome", ("live_descendant", "restored_keeper", "replaced_keeper"))
def test_transition_terminal_waits_for_controller_descendants_before_consuming_claim(tmp_path, monkeypatch, outcome):
    state, path = support.issued_owner(tmp_path, monkeypatch, "ready")
    monkeypatch.undo()
    before = sha256(path)
    marker = state["issued_controller"]["marker"]
    argv = [queue.contract.PYTHON, "-c", "import signal;signal.pause()"]
    child = subprocess.Popen(
        argv,
        cwd=tmp_path,
        env=os.environ | queue.ENVIRONMENT | {queue.contract.CONTROLLER_MARKER: marker},
        start_new_session=True,
    )
    try:
        own = queue.acknowledged_process(
            child, argv, tmp_path, queue.contract.CONTROLLER_MARKER, marker, queue.ENVIRONMENT
        )
        if outcome != "live_descendant":
            saved = read(path)
            expected = own if outcome == "restored_keeper" else own | {"start_ticks": "replaced"}
            saved["restored_keeper"] = {"generation": {str(child.pid): expected}}
            write(path, saved)
            before = sha256(path)
        observed = queue.issued_readiness(tmp_path, state)
        assert observed["ready"] is (outcome == "restored_keeper")
        if outcome != "restored_keeper":
            assert child.pid in observed["marked_pids"]
        assert sha256(path) == before
        assert all(
            not item.exists() for item in queue.controller.terminal_paths(tmp_path, state["issued_controller"])[:3]
        )
    finally:
        queue.finish_marked_child(child, marker, {}, queue.contract.CONTROLLER_MARKER)


@pytest.mark.parametrize("outcome", ("timeout", "complete"))
def test_external_observer_cleans_detached_nested_reader_without_touching_unrelated_owner(
    external_terminal, monkeypatch, outcome
):
    root, helpers, observation = external_terminal
    write(observation, read(observation) | {"detached": True, "timeout": outcome == "timeout"})
    monkeypatch.setattr(parent, "OBSERVER_TIMEOUT", 2)
    unrelated = subprocess.Popen([queue.contract.PYTHON, "-c", "import signal;signal.pause()"])
    try:
        if outcome == "timeout":
            with pytest.raises(subprocess.TimeoutExpired):
                parent.inspect(root, "readiness")
        else:
            assert parent.inspect(root, "readiness")["ready"] is False
        detached = read(helpers / "detached.json")
        assert not queue.handoff.marker_processes(parent.OBSERVER_MARKER, detached["marker"])
        observed = queue.handoff.observed(detached["pid"])
        assert observed is None or observed["status"] in ("Z", "X", "x")
        assert unrelated.poll() is None
        assert not queue.contract.locations(root)[0].exists()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)
        if (helpers / "detached.json").is_file():
            pid = read(helpers / "detached.json")["pid"]
            observed = queue.handoff.observed(pid)
            if observed is not None and observed["status"] not in ("Z", "X", "x"):
                descriptor = pidfd_open(pid)
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                finally:
                    os.close(descriptor)
