# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise source-bound actual-LR ownership, real cleanup and retained rejection."""

import fcntl
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import ns_lr_full_model_contract as contract
import ns_lr_full_model_execution_source as source
import ns_lr_full_model_gpu_controller as controller
import ns_lr_full_model_gpu_owner as owner
import ns_lr_full_model_gpu_test_support as support
import owned_output_handoff as handoff
import pytest
from fused_attribution_validation_test_support import keeper, record
from selected_lr_release_owner import candidate_files, read, sha256


@pytest.fixture(scope="module")
def accepted_parent():
    return read(support.ROOT / source.PARENT)


@pytest.fixture
def execution_candidate(tmp_path, monkeypatch, accepted_parent):
    return support.execution_candidate(tmp_path, monkeypatch, accepted_parent)


def test_owner_only_source_preserves_numerics_and_real_unowned_trainer_validator_create_no_output(execution_candidate):
    root, candidate, prepared = execution_candidate
    original = Path(prepared["parent_candidate"])
    before = candidate_files(original)
    assert source.require_preparation(root, candidate) == prepared
    assert {
        name
        for name in before.keys() | prepared["files_sha256"].keys()
        if before.get(name) != prepared["files_sha256"].get(name)
    } == {source.OWNER}
    command = contract.commands(candidate, root, contract.locations(root)[3])[0]
    commands = (
        [command[0], *command[5:]],
        [
            contract.PYTHON,
            "-P",
            str(Path(controller.validation.__file__)),
            "--root",
            str(root),
            "--candidate",
            str(candidate),
            "--phase",
            "execute",
        ],
    )
    for argv in commands:
        denied = subprocess.run(
            argv,
            cwd=candidate,
            env=os.environ
            | {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": f"{Path(source.__file__).parent}:{candidate}",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert denied.returncode != 0
        assert not contract.locations(root)[3].exists() and not contract.locations(root)[4].exists()
    assert candidate_files(original) == before and candidate_files(candidate) == prepared["files_sha256"]


@pytest.mark.parametrize("fault", ("trainer", "arithmetic", "configuration", "adapter"))
def test_rehashed_actual_lr_source_cannot_expand_accepted_owner_scope(execution_candidate, fault):
    root, candidate, prepared = execution_candidate
    path = (
        candidate
        / {
            "trainer": "experiment/train.py",
            "arithmetic": "experiment/block_toner.py",
            "configuration": contract.specs()[0]["config_name"],
            "adapter": source.OWNER,
        }[fault]
    )
    path.write_bytes(path.read_bytes() + b"\n")
    record(candidate / source.PREPARATION, prepared | {"files_sha256": candidate_files(candidate)})
    with pytest.raises(ValueError):
        source.require_preparation(root, candidate)


@pytest.mark.parametrize("fault", ("none", "keeper_generation"))
def test_actual_lr_handoff_requires_exact_original_terminal_keeper_before_transfer(tmp_path, monkeypatch, fault):
    accepted = keeper()
    expected = {"restored_keeper": accepted, "lower_lr_decisions": {"all_cases_accepted": False}}
    monkeypatch.setattr(owner.predecessor, "evidence", lambda root: expected)
    observed = json.loads(json.dumps(accepted))
    if fault == "keeper_generation":
        observed["generation"]["100"]["start_ticks"] = "substituted-restoration"
    monkeypatch.setattr(owner.handoff, "keeper_observation", lambda: observed)
    if fault == "none":
        assert owner.require_predecessor_keeper(tmp_path, expected) == accepted
    else:
        with pytest.raises(RuntimeError):
            owner.require_predecessor_keeper(tmp_path, expected)
    assert not contract.locations(tmp_path)[0].exists() and not contract.locations(tmp_path)[3].exists()


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached", "timeout", "spawn_failure"))
def test_actual_lr_validator_retains_real_outcomes_and_cleans_detached_processes(tmp_path, monkeypatch, outcome):
    _, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    state = {"controller_pid": os.getpid(), "controller_start_ticks": "CPU-controller"}
    record(state_path, state)
    detached = tmp_path / "detached.pid"
    program = support.validator_program(helpers)
    argv = [sys.executable, str(program), str(state_path), outcome, str(detached)]
    monkeypatch.setattr(owner, "validator_command", lambda root, candidate: argv)
    monkeypatch.setattr(owner, "TIMEOUT", 0.25 if outcome == "timeout" else 10)
    monkeypatch.setattr(contract, "ENVIRONMENT", contract.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": ""})
    if outcome == "spawn_failure":
        monkeypatch.setattr(controller.subprocess, "Popen", support.fail_spawn)
        with pytest.raises(OSError):
            controller.bounded_validator(tmp_path, {"candidate": str(tmp_path)}, state)
    elif outcome == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            controller.bounded_validator(tmp_path, {"candidate": str(tmp_path)}, state)
    else:
        assert controller.bounded_validator(tmp_path, {"candidate": str(tmp_path)}, state) == (
            0 if outcome == "zero" else 7
        )
    saved = read(state_path)
    assert saved["log_sha256"] == sha256(Path(saved["log"]))
    assert not handoff.marker_processes(contract.CHILD_MARKER, saved["child_marker"])
    if outcome == "spawn_failure":
        assert saved["spawn_status"] == "starting" and "child_pid" not in saved
    else:
        assert saved["spawn_status"] == "acknowledged" and saved["marked_descendants_absent"] is True
        assert handoff.observed(saved["child_pid"]) is None
    if outcome == "detached":
        observed = handoff.observed(int(detached.read_text()))
        assert observed is None or observed["status"] == "Z"


@pytest.mark.parametrize("fault", ("none", "launcher_parent", "rank_session", "rank_topology", "command", "marker"))
def test_actual_lr_rank_requires_original_launcher_session_topology_and_active_command(tmp_path, monkeypatch, fault):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    argv = [
        contract.PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "experiment.train",
        "--config",
        str(candidate / "config.json"),
        "--launch-release",
        str(contract.locations(tmp_path)[0]),
        "--max-steps",
        "24",
    ]
    state = {"active_command": argv, "child_pid": 830000, "child_marker": "validator-marker"}
    own = {
        "pid": os.getpid(),
        "parent": 830001,
        "group": os.getpid(),
        "session": os.getpid(),
        "command": [argv[0], "-u", *argv[5:]],
    }
    launcher = {"pid": 830001, "parent": 830000, "group": 830001, "session": 830001, "command": argv}
    processes = {own["pid"]: own, launcher["pid"]: launcher}
    monkeypatch.setattr(owner, "require_controller", lambda root, candidate: (state, {"commands": [argv]}))

    def context(pid, command, cwd, marker_name, marker, environment):
        actual = processes[pid]
        if actual["command"] != command or marker != "command-marker":
            raise RuntimeError("External CPU process executable/marker differs")
        return actual

    monkeypatch.setattr(owner, "process_context", context)
    monkeypatch.setattr(
        owner, "process_details", lambda pid: {"environment": {contract.CHILD_MARKER: "validator-marker"}}
    )
    for name, value in {
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
        "RANK": "3",
        "LOCAL_RANK": "3",
        contract.COMMAND_MARKER: "command-marker",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["train.py", *argv[7:]])
    if fault == "launcher_parent":
        launcher["parent"] = 840000
    elif fault == "rank_session":
        own["session"] = launcher["session"]
    elif fault == "rank_topology":
        monkeypatch.setenv("LOCAL_RANK", "4")
    elif fault == "command":
        state["active_command"] = [*argv[:-1], "5120"]
    elif fault == "marker":
        monkeypatch.delenv(contract.COMMAND_MARKER)
    args = SimpleNamespace(config=candidate / "config.json", launch_release=contract.locations(tmp_path)[0])
    if fault == "none":
        owner.require_rank(tmp_path, candidate, args)
        assert state["active_command"] == argv
    else:
        with pytest.raises(RuntimeError):
            owner.require_rank(tmp_path, candidate, args)


@pytest.mark.parametrize("fault", ("none", "unowned_pid", "wrong_uuid", "unreleased", "keeper_pointer", "competing"))
def test_actual_lr_physical_allocation_rejects_foreign_ranks_or_incomplete_transfer(tmp_path, monkeypatch, fault):
    root, candidate, state, frozen, occupied, marked = support.physical_owner(tmp_path, monkeypatch)
    if fault == "unowned_pid":
        occupied[0][0] = "840000"
    elif fault == "wrong_uuid":
        occupied[0][1] = "GPU-4"
    elif fault == "unreleased":
        state["keeper_release_complete"] = False
        record(contract.locations(root)[2], state)
    elif fault == "keeper_pointer":
        owner.handoff.KEEPER_PID.write_text("100\n")
    elif fault == "competing":
        marked[840000] = support.generation(840000, state["child_pid"], state["active_command"])
    if fault == "none":
        saved, binding = owner.require_controller(root, candidate)
        assert saved == state and binding == frozen
    else:
        with pytest.raises(RuntimeError):
            owner.require_controller(root, candidate)


@pytest.mark.parametrize("outcome", ("spawn_failure", "nonzero", "timeout"))
def test_actual_lr_controller_restores_locked_allocation_after_real_validator_failure(tmp_path, monkeypatch, outcome):
    restored, events, node_path = support.failed_execution(tmp_path, monkeypatch, outcome)
    with pytest.raises((OSError, RuntimeError, subprocess.TimeoutExpired)):
        controller.execute(tmp_path)
    saved = read(contract.locations(tmp_path)[2])
    assert saved["status"] == "needs_attention" and saved["keeper_restore"] == 0
    assert events == ["released", "restored"] and saved["restored_keeper"] == restored
    with node_path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("outcome", ("zero", "spawn_failure"))
def test_actual_lr_one_shot_start_records_original_detached_acknowledgment(tmp_path, monkeypatch, outcome):
    binding, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    record(binding, {"fixture": "CPU frozen source"})
    intent_path = helpers.parent / "controller.json"
    program = helpers / "controller_fixture.py"
    program.write_text(
        "import json,os,select,sys\nfrom pathlib import Path\n"
        "intent=Path(sys.argv[1]);poll=select.poll()\n"
        "while json.loads(intent.read_text()).get('pid')!=os.getpid():poll.poll(1)\n"
    )
    argv, accepted = [sys.executable, str(program), str(intent_path)], keeper()
    monkeypatch.setattr(owner, "require_binding", lambda root: {"allocation_predecessor": {"restored_keeper": accepted}})
    monkeypatch.setattr(owner, "controller_command", lambda root: argv)
    monkeypatch.setattr(owner, "require_predecessor_keeper", lambda *args: accepted)
    monkeypatch.setattr(controller, "local_node", lambda: owner.NODE)
    monkeypatch.setattr(controller, "idle_training", lambda: None)
    monkeypatch.setattr(controller, "keeper_guard", lambda: support.locked(tmp_path / "manager.lock"))
    monkeypatch.setattr(contract, "ENVIRONMENT", contract.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": ""})
    children, original_spawn = [], subprocess.Popen

    def spawn(*args, **kwargs):
        if outcome == "spawn_failure":
            return support.fail_spawn(*args, **kwargs)
        child = original_spawn(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(controller.subprocess, "Popen", spawn)
    try:
        if outcome == "zero":
            intent = controller.start(tmp_path)
            own = intent["generation"]
            assert intent["status"] == "acknowledged" and own["parent"] == os.getpid()
            assert own["session"] == own["group"] == own["pid"]
            assert children[0].wait(timeout=10) == 0 and handoff.observed(own["pid"]) is None
        else:
            with pytest.raises(OSError):
                controller.start(tmp_path)
            assert read(intent_path)["status"] == "needs_attention"
        assert not state_path.exists()
        with pytest.raises(RuntimeError):
            controller.start(tmp_path)
    finally:
        for child in children:
            controller.finish_marked_child(child, read(intent_path)["marker"], {}, contract.CONTROLLER_MARKER)


@pytest.mark.parametrize("fault", ("none", "decisions", "keeper", "live_validator"))
def test_actual_lr_terminal_retains_rejection_and_requires_exact_original_restoration(tmp_path, monkeypatch, fault):
    state, intent, result, restored = support.completed_terminal(tmp_path, monkeypatch)
    if fault == "decisions":
        state.update(
            accepted_cases=["finite-map-passed", "finite-map-rejected"], rejected_cases=[], all_cases_accepted=True
        )
        record(contract.locations(tmp_path)[2], state)
    elif fault == "keeper":
        observed = json.loads(json.dumps(restored))
        observed["generation"]["100"]["start_ticks"] = "different-restoration"
        monkeypatch.setattr(controller.handoff, "keeper_observation", lambda: observed)
    elif fault == "live_validator":
        monkeypatch.setattr(controller.handoff, "observed", lambda pid: {"pid": pid})
    out = tmp_path / "run-preparation/terminal.json"
    if fault == "none":
        receipt = controller.terminal(tmp_path, out)
        assert receipt["rejected_cases"] == result["rejected_cases"] and receipt["all_cases_accepted"] is False
        assert receipt["keeper"] == restored and receipt["controller_pid_absent"] == intent["pid"]
        with pytest.raises(RuntimeError):
            controller.terminal(tmp_path, out)
    else:
        with pytest.raises((RuntimeError, ValueError)):
            controller.terminal(tmp_path, out)
        assert not out.exists()


@pytest.mark.parametrize("fault", ("none", "truncated_xml", "archive", "fixture"))
def test_actual_lr_cpu_archive_rejects_rehashed_incomplete_or_changed_evidence(execution_candidate, monkeypatch, fault):
    root, candidate, prepared = execution_candidate
    for name, value in source.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(source.subprocess, "run", support.completed_cpu)
    out = root / "cpu"
    cpu = source.cpu(root, candidate, out)
    assert cpu["helpers_sha256"] == prepared["helpers_sha256"] and len(cpu["helpers_sha256"]) == 153
    receipt_path = out / "receipt.json"
    if fault == "truncated_xml":
        tree = ET.parse(out / "tests.xml")
        suite = tree.getroot().findall("testsuite")[-1]
        suite.remove(suite.findall("testcase")[-1])
        suite.set("tests", str(int(suite.attrib["tests"]) - 1))
        tree.write(out / "tests.xml")
        record(receipt_path, cpu | {"tests_sha256": sha256(out / "tests.xml")})
    elif fault == "archive":
        with tarfile.open(out / "source.tar", "a") as bundle:
            bundle.add(out / "tracking" / source.NEW_HELPERS[0], arcname="unaccepted-source.py")
        record(receipt_path, cpu | {"archive_sha256": sha256(out / "source.tar")})
    elif fault == "fixture":
        name = next(name for name in cpu["inventory_sha256"] if name.startswith("artifacts/") and "external" not in name)
        (out / name).write_bytes((out / name).read_bytes() + b"\n")
    if fault == "none":
        assert source.require_cpu(root, receipt_path) == cpu
    else:
        with pytest.raises(ValueError):
            source.require_cpu(root, receipt_path)


@pytest.mark.parametrize("fault", ("none", "changed_helper", "mutable_import"))
def test_actual_lr_real_freeze_and_fresh_adapter_require_exact_archived_import_origins(
    execution_candidate, accepted_parent, monkeypatch, fault
):
    root, candidate, prepared = execution_candidate
    for name, value in source.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(source.subprocess, "run", support.completed_cpu)
    out = root / "cpu"
    cpu = source.cpu(root, candidate, out)
    monkeypatch.undo()
    monkeypatch.setattr(source, "inherited", lambda root: accepted_parent)
    previous = {"status": "original external terminal fixture", "restored_keeper": keeper()}
    monkeypatch.setattr(controller.predecessor, "evidence", lambda root: previous)
    monkeypatch.setattr(owner, "require_predecessor_keeper", lambda root, expected: previous["restored_keeper"])
    monkeypatch.setattr(controller, "keeper_guard", lambda: support.locked(root / "manager.lock"))
    (root / "run-preparation").mkdir(exist_ok=True)
    frozen = controller.freeze(root, out / "receipt.json")
    binding, helpers, _, work, result = contract.locations(root)
    assert frozen["helpers_sha256"] == cpu["helpers_sha256"] == prepared["helpers_sha256"]
    assert len(frozen["commands"]) == 162
    with pytest.raises(FileExistsError):
        controller.freeze(root, out / "receipt.json")
    original_parent = helpers / "ns_lr_full_model_predecessor.py"
    mutable = root / "mutable_predecessor.py"
    mutable.write_bytes(original_parent.read_bytes())
    if fault == "changed_helper":
        original_parent.write_bytes(original_parent.read_bytes() + b"\n")
    probe = root / "freeze_probe.py"
    probe.write_text(
        "import importlib.util,json,sys\nfrom pathlib import Path\n"
        "root,candidate,parent,mutable=map(Path,sys.argv[1:5]);fault=sys.argv[5]\n"
        "import ns_lr_full_model_execution_source as source\n"
        "source.inherited=lambda root:source.read(parent)\n"
        "if fault=='mutable_import':\n"
        " spec=importlib.util.spec_from_file_location('ns_lr_full_model_predecessor',mutable)\n"
        " mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)\n"
        "import ns_lr_full_model_gpu_owner as owner\n"
        "owner.predecessor.evidence=lambda root:"
        "source.read(source.contract.locations(root)[0])['allocation_predecessor']\n"
        "frozen=owner.require_binding(root)\n"
        "sys.path.insert(0,str(candidate))\n"
        "from experiment.fused_attribution_owner import execution_owner\n"
        "expected,loaded=execution_owner()\n"
        "assert expected==root and loaded is owner\n"
        "assert Path(loaded.__file__).resolve()==source.contract.locations(root)[1]/'ns_lr_full_model_gpu_owner.py'\n"
        "print(json.dumps({'status':'frozen_actual_lr_adapter_roundtrip','commands':len(frozen['commands']),"
        "'helpers':len(frozen['helpers_sha256'])}))\n"
    )
    checked = subprocess.run(
        [
            sys.executable,
            "-P",
            str(probe),
            str(root),
            str(candidate),
            str(support.ROOT / source.PARENT),
            str(mutable),
            fault,
        ],
        cwd=candidate,
        env=os.environ
        | {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": f"{helpers}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    if fault == "none":
        assert checked.returncode == 0, checked.stderr
        assert json.loads(checked.stdout) == {
            "status": "frozen_actual_lr_adapter_roundtrip",
            "commands": 162,
            "helpers": 153,
        }
    else:
        assert checked.returncode != 0
    assert not work.exists() and not result.exists() and read(binding) == frozen
