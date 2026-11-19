# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real source transformations, owner ancestry and detached cleanup with hidden CUDA."""

import fcntl
import json
import os
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import ns_full_model_contract as contract
import ns_full_model_execution_source as source
import ns_full_model_gpu_controller as controller
import ns_full_model_gpu_owner as owner
import ns_full_model_predecessor as predecessor
import owned_output_handoff as handoff
import pytest
from fused_attribution_validation_test_support import keeper, record
from selected_lr_release_owner import candidate_files, read, sha256

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


@pytest.fixture(scope="module")
def accepted_execution_inputs():
    inherited = read(ROOT / source.PARENT)
    sources = source.helper_sources(ROOT)
    return inherited, sources


@pytest.fixture
def execution_candidate(tmp_path, monkeypatch, accepted_execution_inputs):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    inherited, sources = accepted_execution_inputs
    record(root / "data/experiment_receipt.json", {"fixture": "immutable CPU data"})
    monkeypatch.setattr(source, "parent", lambda root: inherited)
    monkeypatch.setattr(source, "helper_sources", lambda root: sources)
    prepared = source.prepare(root, candidate)
    return root, candidate, prepared


def test_owner_only_candidate_preserves_all_numerics_and_real_unowned_training_is_denied(execution_candidate):
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
    denied = subprocess.run(
        [command[0], *command[5:]],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert denied.returncode != 0
    assert not contract.locations(root)[3].exists() and not contract.locations(root)[4].exists()
    assert candidate_files(original) == before and candidate_files(candidate) == prepared["files_sha256"]


@pytest.mark.parametrize("fault", ("trainer", "arithmetic", "configuration", "adapter"))
def test_rehashed_execution_candidate_cannot_expand_accepted_owner_scope(execution_candidate, fault):
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


def test_original_live_queue_handle_denies_predecessor_before_receipts_or_transfer(tmp_path):
    record(tmp_path / predecessor.QUEUE / "intent.json", {"pid": os.getpid(), "marker": "original-live-cpu"})
    record(tmp_path / predecessor.QUEUE / "state.json", {"phases": []})
    with pytest.raises(RuntimeError, match="original queue remains present"):
        predecessor.evidence(tmp_path)
    assert not contract.locations(tmp_path)[0].exists() and not contract.locations(tmp_path)[3].exists()


def generation(pid, parent, command):
    return {
        "pid": pid,
        "parent": parent,
        "group": pid,
        "session": pid,
        "start_ticks": str(pid + 500),
        "command": command,
        "status": "S",
    }


@pytest.fixture
def completed_predecessor(tmp_path, monkeypatch):
    queue_path = tmp_path / predecessor.QUEUE
    binding, helpers, state_path, work = predecessor.previous.locations(tmp_path)
    helpers.mkdir(parents=True)
    helper = helpers / "ns_budget_gpu_controller.py"
    helper.write_text("# pinned CPU owner fixture\n")
    hashes = {helper.name: sha256(helper)}
    archive = tmp_path / "owner-source.tar"
    with tarfile.open(archive, "x") as bundle:
        bundle.add(helper, arcname=f"tracking/{helper.name}")
    queue_cpu = {
        "helpers": str(tmp_path / "accepted-queue/tracking"),
        "helpers_sha256": {"ns_budget_queue.py": "CPU-queue"},
    }
    owner_cpu = {"helpers": str(tmp_path / "accepted-owner/tracking"), "helpers_sha256": hashes}
    record(tmp_path / predecessor.QUEUE_CPU, queue_cpu)
    record(tmp_path / predecessor.OWNER_CPU, owner_cpu)
    monkeypatch.setattr(predecessor, "QUEUE_CPU_SHA256", sha256(tmp_path / predecessor.QUEUE_CPU))
    monkeypatch.setattr(predecessor, "OWNER_CPU_SHA256", sha256(tmp_path / predecessor.OWNER_CPU))
    monkeypatch.setattr(predecessor, "require_cpu_archive", lambda root, path, status, count: read(path))
    queue_command = [
        contract.PYTHON,
        "-P",
        f"{queue_cpu['helpers']}/ns_budget_queue.py",
        "--root",
        str(tmp_path),
        "--cpu-receipt",
        str(tmp_path / predecessor.QUEUE_CPU),
        "--phase",
        "execute",
    ]
    intent = {
        "status": "acknowledged",
        "pid": 21000,
        "start_ticks": "21500",
        "generation": generation(21000, 1, queue_command),
        "command": queue_command,
        "marker": "original-queue",
        "runner_sha256": "CPU-queue",
        "cpu_receipt_sha256": predecessor.QUEUE_CPU_SHA256,
        "parent_cpu_receipt_sha256": predecessor.OWNER_CPU_SHA256,
        "interval": 900,
    }
    record(queue_path / "intent.json", intent)
    monkeypatch.setattr(predecessor, "QUEUE_INTENT_SHA256", sha256(queue_path / "intent.json"))
    restored = keeper()
    frozen = {
        "status": predecessor.previous.STATUS,
        "root": str(tmp_path),
        "node": owner.NODE,
        "cpu_receipt": str(tmp_path / predecessor.OWNER_CPU),
        "cpu_receipt_sha256": predecessor.OWNER_CPU_SHA256,
        "helpers_sha256": hashes,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "predecessor": {"fixture": "original rational terminal"},
        "devices": restored["devices"],
    }
    record(binding, frozen)
    owner_command = predecessor.previous.controller_command(tmp_path)
    owner_intent = {
        "status": "acknowledged",
        "pid": 22000,
        "start_ticks": "22500",
        "generation": generation(22000, 1, owner_command),
        "command": owner_command,
        "binding_sha256": sha256(binding),
        "marker": "original-controller",
    }
    record(helpers.parent / "controller.json", owner_intent)
    (helpers.parent / "controller.log").write_text("CPU original owner complete\n")
    executions = []
    for index in range(2):
        argv = predecessor.previous.worker_command(tmp_path, sha256(binding), index)
        log, report = work / f"process-{index}.log", work / f"process-{index}.json"
        record(report, {"fixture": index})
        log.write_text("CPU retained worker\n")
        executions.append(
            {
                "index": index,
                "command": argv,
                "pid": 22001 + index,
                "start_ticks": str(22501 + index),
                "generation": generation(22001 + index, 22000, argv),
                "marker": f"original-worker-{index}",
                "controller_pid": 22000,
                "controller_start_ticks": "22500",
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "marked_descendants_absent": True,
                "log": str(log),
                "log_sha256": sha256(log),
                "report": str(report),
                "report_sha256": sha256(report),
            }
        )
    comparison = {"historical_controls_interpretable": True, "fixture": "complete comparison"}
    record(work / "comparison.json", comparison | {"time": "2026-11-19T00:00:01+00:00"})
    state = {
        "status": "diagnostic_complete",
        "binding_sha256": sha256(binding),
        "predecessor": frozen["predecessor"],
        "controller_pid": 22000,
        "controller_start_ticks": "22500",
        "keeper_release_attempted": True,
        "keeper_release_complete": True,
        "keeper_restore": 0,
        "restored_keeper": restored,
        "executions": executions,
        "active_worker": executions[-1],
        "comparison_sha256": sha256(work / "comparison.json"),
        "time": "2026-11-19T00:00:02+00:00",
    }
    record(state_path, state)
    terminal = {
        "status": "ns_budget_terminal_independently_reconstructed_exact_restoration_verified",
        "binding_sha256": sha256(binding),
        "terminal_time": state["time"],
        "controller_pid_absent": 22000,
        "worker_pids_absent": [22001, 22002],
        "verifier_sha256": hashes[helper.name],
        "comparison": comparison,
        "keeper": restored,
        "evidence_sha256": {str((work / "process-0.json").relative_to(tmp_path)): sha256(work / "process-0.json")},
        "time": "2026-11-19T00:00:03+00:00",
    }
    record(tmp_path / predecessor.TERMINAL, terminal)
    phases = []
    for index, phase in enumerate(("freeze", "check", "start", "terminal")):
        runner = Path(owner_cpu["helpers"]) / helper.name if phase == "freeze" else helper
        argv = [contract.PYTHON, "-P", str(runner), "--root", str(tmp_path), "--phase", phase]
        if phase == "freeze":
            argv.extend(("--cpu-receipt", str(tmp_path / predecessor.OWNER_CPU)))
        if phase == "terminal":
            argv.extend(("--out", str(tmp_path / predecessor.TERMINAL)))
        log = queue_path / f"{phase}.log"
        log.write_text(f"CPU completed {phase}\n")
        phases.append(
            {
                "phase": phase,
                "command": argv,
                "pid": 21001 + index,
                "start_ticks": str(21501 + index),
                "generation": generation(21001 + index, 21000, argv),
                "marker": f"original-{phase}",
                "spawn_status": "acknowledged",
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "marked_descendants_absent": True,
                "log": str(log),
                "log_sha256": sha256(log),
            }
        )
    queue_state = {
        "status": "ns_diagnostic_terminal_reconstructed",
        "pid": intent["pid"],
        "start_ticks": intent["start_ticks"],
        "runner_sha256": intent["runner_sha256"],
        "cpu_receipt_sha256": predecessor.QUEUE_CPU_SHA256,
        "parent_cpu_receipt_sha256": predecessor.OWNER_CPU_SHA256,
        "interval": 900,
        "phases": phases,
        "terminal_sha256": sha256(tmp_path / predecessor.TERMINAL),
        "issued_binding_sha256": sha256(binding),
        "historical_controls_interpretable": True,
        "time": "2026-11-19T00:00:04+00:00",
    }
    record(queue_path / "state.json", queue_state)
    (queue_path / "queue.log").write_text("CPU original queue complete\n")
    monkeypatch.setattr(predecessor.handoff, "observed", lambda pid: None)
    monkeypatch.setattr(predecessor.handoff, "marker_processes", lambda name, marker: {})
    return tmp_path, queue_state, state, terminal, restored


@pytest.mark.parametrize("fault", ("none", "queue_cleanup", "worker_parent", "control", "terminal_hash"))
def test_predecessor_positive_handoff_preserves_original_cleanup_and_negative_controls(completed_predecessor, fault):
    root, queue_state, state, terminal, restored = completed_predecessor
    if fault == "queue_cleanup":
        queue_state["phases"][-1]["cleanup_exit_code"] = 1
        record(root / predecessor.QUEUE / "state.json", queue_state)
    elif fault == "worker_parent":
        state["executions"][0]["generation"]["parent"] = 99999
        record(predecessor.previous.locations(root)[2], state)
    elif fault == "control":
        terminal["comparison"]["historical_controls_interpretable"] = False
        record(root / predecessor.TERMINAL, terminal)
    elif fault == "terminal_hash":
        (root / predecessor.QUEUE / "terminal.log").write_text("changed command log\n")
    if fault == "none":
        result = predecessor.evidence(root)
        assert result["restored_keeper"] == restored
        assert result["historical_controls_interpretable"] is True
        assert result["evidence_sha256"][predecessor.TERMINAL] == sha256(root / predecessor.TERMINAL)
    else:
        with pytest.raises((RuntimeError, ValueError)):
            predecessor.evidence(root)


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached", "timeout"))
def test_bounded_validator_retains_cleanup_and_removes_real_detached_descendants(tmp_path, monkeypatch, outcome):
    _, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    state = {"controller_pid": os.getpid(), "controller_start_ticks": "CPU-controller"}
    record(state_path, state)
    program = helpers / "validator_fixture.py"
    detached = tmp_path / "detached.pid"
    program.write_text(
        "import json,os,select,signal,subprocess,sys\n"
        "from pathlib import Path\n"
        "state=Path(sys.argv[1]);poll=select.poll()\n"
        "while json.loads(state.read_text()).get('child_pid')!=os.getpid():poll.poll(1)\n"
        "if sys.argv[2]=='timeout':signal.pause()\n"
        "if sys.argv[2]=='detached':\n"
        " r,w=os.pipe()\n"
        " child=subprocess.Popen([sys.executable,'-c',"
        "'import os,signal,sys;os.write(int(sys.argv[1]),b\"R\");signal.pause()',str(w)],"
        "pass_fds=(w,),start_new_session=True)\n"
        " os.close(w);assert os.read(r,1)==b'R';os.close(r)\n"
        " Path(sys.argv[3]).write_text(str(child.pid))\n"
        "sys.exit(0 if sys.argv[2]=='zero' else 7)\n"
    )
    argv = [sys.executable, str(program), str(state_path), outcome, str(detached)]
    monkeypatch.setattr(owner, "validator_command", lambda root, candidate: argv)
    monkeypatch.setattr(owner, "TIMEOUT", 0.25 if outcome == "timeout" else 10)
    monkeypatch.setattr(contract, "ENVIRONMENT", contract.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": ""})
    if outcome == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            controller.bounded_validator(tmp_path, {"candidate": str(tmp_path)}, state)
    else:
        assert controller.bounded_validator(tmp_path, {"candidate": str(tmp_path)}, state) == (
            0 if outcome == "zero" else 7
        )
    saved = read(state_path)
    assert saved["spawn_status"] == "acknowledged"
    assert saved["marked_descendants_absent"] is True
    assert saved["log_sha256"] == sha256(Path(saved["log"]))
    assert handoff.observed(saved["child_pid"]) is None
    assert not handoff.marker_processes(contract.CHILD_MARKER, saved["child_marker"])
    if outcome == "detached":
        process = handoff.observed(int(detached.read_text()))
        assert process is None or process["status"] == "Z"


@pytest.fixture
def rank_owner(tmp_path, monkeypatch):
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
    state = {"active_command": argv, "child_pid": 1000, "child_marker": "validator-marker"}
    frozen = {"commands": [argv]}
    own = {
        "pid": os.getpid(),
        "parent": 1001,
        "group": os.getpid(),
        "session": os.getpid(),
        "command": [argv[0], "-u", *argv[5:]],
    }
    launcher = {"pid": 1001, "parent": 1000, "group": 1001, "session": 1001, "command": argv}
    processes = {own["pid"]: own, launcher["pid"]: launcher}
    monkeypatch.setattr(owner, "require_controller", lambda root, candidate: (state, frozen))

    def process(pid, command, cwd, marker_name, marker, environment):
        actual = processes[pid]
        if actual["command"] != command or marker != "command-marker":
            raise RuntimeError("CPU process fixture executable/marker differs")
        return actual

    monkeypatch.setattr(owner, "process_context", process)
    monkeypatch.setattr(
        owner,
        "process_details",
        lambda pid: {"environment": {contract.CHILD_MARKER: state["child_marker"]}},
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
    return tmp_path, candidate, state, frozen, own, launcher


@pytest.mark.parametrize("fault", ("none", "launcher_parent", "rank_session", "rank_topology", "command", "marker"))
def test_rank_owner_requires_original_launcher_and_independent_torch_sessions(rank_owner, monkeypatch, fault):
    root, candidate, state, frozen, own, launcher = rank_owner
    if fault == "launcher_parent":
        launcher["parent"] = 2000
    elif fault == "rank_session":
        own["session"] = launcher["session"]
    elif fault == "rank_topology":
        monkeypatch.setenv("LOCAL_RANK", "4")
    elif fault == "command":
        state["active_command"] = [*state["active_command"][:-1], "5120"]
    elif fault == "marker":
        monkeypatch.delenv(contract.COMMAND_MARKER)
    args = SimpleNamespace(config=candidate / "config.json", launch_release=contract.locations(root)[0])
    if fault == "none":
        owner.require_rank(root, candidate, args)
        assert state["active_command"] == frozen["commands"][0]
    else:
        with pytest.raises(RuntimeError):
            owner.require_rank(root, candidate, args)


@pytest.fixture
def physical_owner(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    binding, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    record(binding, {"fixture": "frozen owner"})
    accepted = keeper()
    controller_generation = generation(31000, 1, owner.controller_command(tmp_path))
    validator_generation = generation(31001, 31000, owner.validator_command(tmp_path, candidate))
    launcher_generation = generation(31002, 31001, [contract.PYTHON, "CPU-launcher"])
    rank_generation = generation(31003, 31002, [contract.PYTHON, "-u", "-m", "experiment.train", "--config", "CPU"])
    intent = {
        "status": "acknowledged",
        "pid": 31000,
        "start_ticks": "31500",
        "generation": controller_generation,
        "command": controller_generation["command"],
        "binding_sha256": sha256(binding),
        "marker": "controller-marker",
    }
    record(helpers.parent / "controller.json", intent)
    compact = {
        "pid": accepted["pid"],
        "start_ticks": accepted["generation"]["100"]["start_ticks"],
        "generation": [[int(pid), item["start_ticks"], item["group"]] for pid, item in accepted["generation"].items()],
    }
    frozen = {
        "candidate": str(candidate),
        "allocation_predecessor": {"restored_keeper": accepted},
        "devices": accepted["devices"],
    }
    state = {
        "status": "validating",
        "spawn_status": "acknowledged",
        "binding_sha256": sha256(binding),
        "controller_pid": 31000,
        "controller_start_ticks": "31500",
        "allocation_predecessor": frozen["allocation_predecessor"],
        "keeper_release_complete": True,
        "keeper": compact,
        "command": validator_generation["command"],
        "child_pid": 31001,
        "child_start_ticks": "31501",
        "child_generation": validator_generation,
        "child_marker": "validator-marker",
        "active_command": launcher_generation["command"],
        "lock_descriptors": {"node": 8, "manager": 9},
    }
    record(state_path, state)
    processes = {item["pid"]: item for item in (controller_generation, validator_generation)}
    occupied = [["31003", "GPU-3"]]
    marked = {item["pid"]: item for item in (validator_generation, launcher_generation, rank_generation)}
    monkeypatch.setattr(owner, "require_binding", lambda root: frozen)
    monkeypatch.setattr(owner, "process_context", lambda pid, *args: processes[pid])
    monkeypatch.setattr(owner, "require_held_lock", lambda *args: None)
    monkeypatch.setattr(owner, "process_details", lambda pid: {"environment": {"LOCAL_RANK": "3"}})
    monkeypatch.setattr(owner.handoff, "marker_processes", lambda *args: marked)
    monkeypatch.setattr(owner.handoff, "KEEPER_PID", tmp_path / "keeper.pid")
    monkeypatch.chdir(candidate)
    for name, value in contract.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    def query(argv, **kwargs):
        if argv == ["hostname", "-I"]:
            return owner.NODE
        if argv[1] == "--query-gpu=index,uuid,name":
            return "\n".join(",".join(row) for row in frozen["devices"])
        if argv[1] == "--query-compute-apps=pid,gpu_uuid":
            return "\n".join(",".join(row) for row in occupied)
        raise AssertionError(argv)

    monkeypatch.setattr(owner.subprocess, "check_output", query)
    return tmp_path, candidate, state, frozen, occupied


@pytest.mark.parametrize("fault", ("none", "unowned_pid", "wrong_uuid", "keeper_pointer", "unreleased"))
def test_physical_owner_matches_only_original_rank_uuid_and_completed_keeper_transfer(physical_owner, fault):
    root, candidate, state, frozen, occupied = physical_owner
    if fault == "unowned_pid":
        occupied[0][0] = "40000"
    elif fault == "wrong_uuid":
        occupied[0][1] = "GPU-4"
    elif fault == "keeper_pointer":
        owner.handoff.KEEPER_PID.write_text("100\n")
    elif fault == "unreleased":
        state["keeper_release_complete"] = False
        record(contract.locations(root)[2], state)
    if fault == "none":
        result, binding = owner.require_controller(root, candidate)
        assert result == state and binding == frozen
    else:
        with pytest.raises(RuntimeError):
            owner.require_controller(root, candidate)


@pytest.mark.parametrize("outcome", ("spawn_failure", "nonzero"))
def test_controller_restores_real_locked_allocation_after_validator_execution_failure(tmp_path, monkeypatch, outcome):
    binding, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    record(binding, {"fixture": "CPU external binding"})
    accepted, restored = keeper(), keeper()
    restored["generation"]["100"]["start_ticks"] = "restored-generation"
    own = {
        "pid": os.getpid(),
        "parent": os.getppid(),
        "group": os.getpid(),
        "session": os.getpid(),
        "start_ticks": "CPU-controller",
        "command": owner.controller_command(tmp_path),
        "status": "R",
    }
    record(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "pid": own["pid"],
            "start_ticks": own["start_ticks"],
            "generation": own,
            "command": own["command"],
            "binding_sha256": sha256(binding),
            "marker": "CPU-controller",
        },
    )
    program = helpers / "failed_validator.py"
    program.write_text(
        "import json,select,sys,os\nfrom pathlib import Path\n"
        "state=Path(sys.argv[1]);poll=select.poll()\n"
        "while json.loads(state.read_text()).get('child_pid')!=os.getpid():poll.poll(1)\n"
        "sys.exit(7)\n"
    )
    argv = [sys.executable, str(program), str(state_path)]
    monkeypatch.setattr(owner, "validator_command", lambda root, candidate: argv)
    monkeypatch.setattr(contract, "ENVIRONMENT", contract.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": ""})
    monkeypatch.setattr(
        owner,
        "require_binding",
        lambda root: {
            "candidate": str(tmp_path),
            "allocation_predecessor": {"restored_keeper": accepted},
        },
    )
    monkeypatch.setattr(controller, "process_context", lambda *args: own)
    monkeypatch.setattr(controller, "local_node", lambda: owner.NODE)
    monkeypatch.setattr(controller.predecessor, "live", lambda *args: accepted)
    events, manager_path = [], tmp_path / "manager.lock"
    node_path = tmp_path / "node.lock"

    @contextmanager
    def manager():
        with manager_path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield stream.fileno()

    @contextmanager
    def node(*args):
        with node_path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield []

    def release(keeper, state, path, descriptor):
        assert keeper == accepted and os.readlink(f"/proc/{os.getpid()}/fd/{descriptor}") == str(manager_path)
        state.update(keeper_release_attempted=True, keeper_release_complete=True)
        events.append("released")

    def restore(descriptor):
        assert os.readlink(f"/proc/{os.getpid()}/fd/{descriptor}") == str(manager_path)
        events.append("restored")

    monkeypatch.setattr(controller, "keeper_guard", manager)
    monkeypatch.setattr(controller, "node_guard", node)
    monkeypatch.setattr(controller, "lock_descriptor", lambda path: 10)
    monkeypatch.setattr(controller, "idle_training", lambda: None)
    monkeypatch.setattr(controller, "release_accepted_keeper", release)
    monkeypatch.setattr(controller, "restore_accepted_keeper", restore)
    monkeypatch.setattr(controller.handoff, "keeper_observation", lambda: restored)
    monkeypatch.setattr(controller.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0))
    if outcome == "spawn_failure":

        def fail(*args, **kwargs):
            raise OSError("CPU validator spawn failure")

        monkeypatch.setattr(controller.subprocess, "Popen", fail)
    with pytest.raises((OSError, RuntimeError)):
        controller.execute(tmp_path)
    saved = read(state_path)
    assert saved["status"] == "needs_attention" and saved["keeper_restore"] == 0
    assert events == ["released", "restored"] and saved["restored_keeper"] == restored
    with node_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("fault", ("none", "truncated_xml", "archive", "fixture"))
def test_cpu_source_archive_roundtrip_rejects_rehashed_incomplete_or_changed_evidence(
    execution_candidate, monkeypatch, fault
):
    root, candidate, prepared = execution_candidate
    monkeypatch.setattr(source, "PARENT", str(ROOT / source.PARENT))
    for name, value in source.previous.parent.owner.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    out = root / "cpu"

    def completed_cpu(command, **kwargs):
        xml = ET.Element("testsuites")
        for module, count in source.TEST_CASE_COUNTS.items():
            suite = ET.SubElement(xml, "testsuite", tests=str(count), failures="0", errors="0", skipped="0")
            for index in range(count):
                ET.SubElement(suite, "testcase", classname=module, name=f"cpu_boundary_case_{index}")
        ET.ElementTree(xml).write(out / "tests.xml")
        kwargs["stdout"].write("External CPU test-runner fixture complete\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(source.subprocess, "run", completed_cpu)
    cpu = source.cpu(root, candidate, out)
    assert cpu["helpers_sha256"] == prepared["helpers_sha256"]
    assert len(cpu["helpers_sha256"]) == 133
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
        fixture = next(
            name for name in cpu["inventory_sha256"] if name.startswith("artifacts/") and "external" not in name
        )
        (out / fixture).write_bytes((out / fixture).read_bytes() + b"\n")
    if fault == "none":
        assert source.require_cpu(root, receipt_path) == cpu
    else:
        with pytest.raises(ValueError):
            source.require_cpu(root, receipt_path)


@pytest.mark.parametrize("outcome", ("zero", "spawn_failure"))
def test_start_records_real_detached_original_acknowledgment_and_retains_spawn_failure(tmp_path, monkeypatch, outcome):
    binding, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    record(binding, {"fixture": "CPU issued source"})
    accepted = keeper()
    program = helpers / "controller_fixture.py"
    intent_path = helpers.parent / "controller.json"
    program.write_text(
        "import json,os,select,sys\nfrom pathlib import Path\n"
        "intent=Path(sys.argv[1]);poll=select.poll()\n"
        "while json.loads(intent.read_text()).get('pid')!=os.getpid():poll.poll(1)\n"
    )
    argv = [sys.executable, str(program), str(intent_path)]
    monkeypatch.setattr(owner, "require_binding", lambda root: {"allocation_predecessor": {"restored_keeper": accepted}})
    monkeypatch.setattr(owner, "controller_command", lambda root: argv)
    monkeypatch.setattr(controller, "local_node", lambda: owner.NODE)
    monkeypatch.setattr(controller, "idle_training", lambda: None)
    monkeypatch.setattr(controller.predecessor, "live", lambda *args: accepted)
    monkeypatch.setattr(contract, "ENVIRONMENT", contract.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": ""})

    @contextmanager
    def manager():
        with (tmp_path / "manager.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield stream.fileno()

    monkeypatch.setattr(controller, "keeper_guard", manager)
    children, original_spawn = [], subprocess.Popen

    def spawn(*args, **kwargs):
        if outcome == "spawn_failure":
            raise OSError("CPU original controller spawn failed")
        child = original_spawn(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(controller.subprocess, "Popen", spawn)
    try:
        if outcome == "zero":
            intent = controller.start(tmp_path)
            assert intent["status"] == "acknowledged" and intent["command"] == argv
            own = intent["generation"]
            assert own["parent"] == os.getpid() and own["session"] == own["group"] == own["pid"]
            assert children[0].wait(timeout=10) == 0
            assert handoff.observed(own["pid"]) is None
        else:
            with pytest.raises(OSError):
                controller.start(tmp_path)
            intent = read(intent_path)
            assert intent["status"] == "needs_attention" and "OSError" in intent["error"]
        assert not state_path.exists()
        with pytest.raises(RuntimeError):
            controller.start(tmp_path)
    finally:
        for child in children:
            controller.finish_marked_child(child, read(intent_path)["marker"], {}, contract.CONTROLLER_MARKER)


@pytest.mark.parametrize("fault", ("none", "decisions", "keeper", "live_validator"))
def test_terminal_roundtrip_retains_numerical_rejection_and_requires_exact_restored_generation(
    tmp_path, monkeypatch, fault
):
    binding, helpers, state_path, _, result_path = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    restored = keeper()
    frozen = {"candidate": str(candidate), "devices": restored["devices"], "allocation_predecessor": {"fixture": "NS"}}
    record(binding, frozen)
    argv = owner.controller_command(tmp_path)
    intent = {
        "status": "acknowledged",
        "pid": 41000,
        "start_ticks": "41500",
        "command": argv,
        "generation": generation(41000, 1, argv),
        "marker": "terminal-controller",
        "binding_sha256": sha256(binding),
        "time": "2026-11-19T00:00:01+00:00",
    }
    record(helpers.parent / "controller.json", intent)
    for filename in ("controller.log", "validator.log"):
        (helpers.parent / filename).write_text("Original completed CPU owner fixture\n")
    result = {
        "accepted_cases": ["finite-map-passed"],
        "rejected_cases": ["finite-map-rejected"],
        "all_cases_accepted": False,
        "time": "2026-11-19T00:00:02+00:00",
    }
    record(result_path, result)
    child_argv = owner.validator_command(tmp_path, candidate)
    state = {
        "status": "validation_complete",
        "binding_sha256": sha256(binding),
        "controller_pid": intent["pid"],
        "controller_start_ticks": intent["start_ticks"],
        "child_pid": 41001,
        "child_start_ticks": "41501",
        "child_generation": generation(41001, 41000, child_argv),
        "child_marker": "terminal-validator",
        "command": child_argv,
        "spawn_status": "acknowledged",
        "keeper_release_attempted": True,
        "keeper_release_complete": True,
        "marked_descendants_absent": True,
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "restored_keeper": restored,
        "allocation_predecessor": frozen["allocation_predecessor"],
        "log": str(helpers.parent / "validator.log"),
        "log_sha256": sha256(helpers.parent / "validator.log"),
        "result_sha256": sha256(result_path),
        **{name: result[name] for name in ("accepted_cases", "rejected_cases", "all_cases_accepted")},
        "time": "2026-11-19T00:00:03+00:00",
    }
    if fault == "decisions":
        state.update(
            accepted_cases=["finite-map-passed", "finite-map-rejected"], rejected_cases=[], all_cases_accepted=True
        )
    record(state_path, state)
    observed_keeper = read(state_path)["restored_keeper"]
    if fault == "keeper":
        observed_keeper["generation"]["100"]["start_ticks"] = "different-restoration"
    monkeypatch.setattr(owner, "require_binding", lambda root: frozen)
    monkeypatch.setattr(controller, "local_node", lambda: owner.NODE)
    monkeypatch.setattr(controller.handoff, "observed", lambda pid: {"pid": pid} if fault == "live_validator" else None)
    monkeypatch.setattr(controller.handoff, "marker_processes", lambda *args: {})
    monkeypatch.setattr(controller.handoff, "keeper_observation", lambda: observed_keeper)
    monkeypatch.setattr(controller.validation, "reconstruct", lambda root, candidate: read(result_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    @contextmanager
    def manager():
        with (tmp_path / "manager.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield stream.fileno()

    monkeypatch.setattr(controller, "keeper_guard", manager)
    out = tmp_path / "run-preparation/terminal.json"
    if fault == "none":
        receipt = controller.terminal(tmp_path, out)
        assert receipt["rejected_cases"] == ["finite-map-rejected"] and receipt["all_cases_accepted"] is False
        assert receipt["keeper"] == restored
        assert receipt["evidence_sha256"][str(state_path.relative_to(tmp_path))] == sha256(state_path)
        with pytest.raises(RuntimeError):
            controller.terminal(tmp_path, out)
    else:
        with pytest.raises((RuntimeError, ValueError)):
            controller.terminal(tmp_path, out)
        assert not out.exists()


@pytest.mark.parametrize("fault", ("none", "changed_helper", "mutable_import"))
def test_real_freeze_and_fresh_adapter_import_require_exact_cpu_archive_and_frozen_module_origins(
    execution_candidate, monkeypatch, fault
):
    root, candidate, prepared = execution_candidate
    inherited = read(ROOT / source.PARENT)
    monkeypatch.setattr(source, "PARENT", str(ROOT / source.PARENT))
    for name, value in source.previous.parent.owner.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    out = root / "cpu"

    def completed_cpu(command, **kwargs):
        xml = ET.Element("testsuites")
        for module, count in source.TEST_CASE_COUNTS.items():
            suite = ET.SubElement(xml, "testsuite", tests=str(count), failures="0", errors="0", skipped="0")
            for index in range(count):
                ET.SubElement(suite, "testcase", classname=module, name=f"external_cpu_case_{index}")
        ET.ElementTree(xml).write(out / "tests.xml")
        kwargs["stdout"].write("External CPU test-runner fixture complete\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(source.subprocess, "run", completed_cpu)
    cpu = source.cpu(root, candidate, out)
    monkeypatch.undo()
    monkeypatch.setattr(source, "PARENT", str(ROOT / source.PARENT))
    monkeypatch.setattr(source, "parent", lambda root: inherited)
    previous = {"status": "CPU predecessor boundary", "restored_keeper": keeper()}
    monkeypatch.setattr(controller.predecessor, "evidence", lambda root: previous)
    monkeypatch.setattr(controller.predecessor, "live", lambda root, expected: previous["restored_keeper"])

    @contextmanager
    def manager():
        with (root / "manager.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield lock.fileno()

    monkeypatch.setattr(controller, "keeper_guard", manager)
    (root / "run-preparation").mkdir()
    frozen = controller.freeze(root, out / "receipt.json")
    binding, helpers, _, work, result = contract.locations(root)
    assert frozen["helpers_sha256"] == cpu["helpers_sha256"] == prepared["helpers_sha256"]
    assert frozen["commands"] == contract.commands(candidate, root, work)
    with pytest.raises(FileExistsError):
        controller.freeze(root, out / "receipt.json")
    parent_receipt = ROOT / source.PARENT
    original_parent = helpers / "ns_full_model_predecessor.py"
    mutable = root / "mutable_predecessor.py"
    mutable.write_bytes(original_parent.read_bytes())
    if fault == "changed_helper":
        original_parent.write_bytes(original_parent.read_bytes() + b"\n")
    probe = root / "freeze_probe.py"
    probe.write_text(
        "import importlib.util,json,sys\nfrom pathlib import Path\n"
        "root,candidate,parent,mutable=map(Path,sys.argv[1:5]);fault=sys.argv[5]\n"
        "import ns_full_model_execution_source as source\n"
        "source.PARENT=str(parent)\n"
        "source.parent=lambda root:source.read(parent)\n"
        "if fault=='mutable_import':\n"
        " spec=importlib.util.spec_from_file_location('ns_full_model_predecessor',mutable)\n"
        " mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)\n"
        "import ns_full_model_gpu_owner as owner\n"
        "owner.predecessor.evidence=lambda root:"
        "source.read(source.contract.locations(root)[0])['allocation_predecessor']\n"
        "frozen=owner.require_binding(root)\n"
        "sys.path.insert(0,str(candidate))\n"
        "from experiment.fused_attribution_owner import execution_owner\n"
        "expected,loaded=execution_owner()\n"
        "assert expected==root and loaded is owner\n"
        "assert Path(loaded.__file__).resolve()==source.contract.locations(root)[1]/'ns_full_model_gpu_owner.py'\n"
        "print(json.dumps({'status':'frozen_adapter_roundtrip','commands':len(frozen['commands']),"
        "'helpers':len(frozen['helpers_sha256'])}))\n"
    )
    command = [sys.executable, "-P", str(probe), str(root), str(candidate), str(parent_receipt), str(mutable), fault]
    checked = subprocess.run(
        command,
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
        assert json.loads(checked.stdout) == {"status": "frozen_adapter_roundtrip", "commands": 54, "helpers": 133}
    else:
        assert checked.returncode != 0
    assert not work.exists() and not result.exists()
    assert read(binding) == frozen
