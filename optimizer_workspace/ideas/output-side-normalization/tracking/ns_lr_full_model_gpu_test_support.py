# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External process/allocation fixtures for distinct actual-LR owner acceptance."""

import fcntl
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_lr_full_model_contract as contract
import ns_lr_full_model_execution_source as source
import ns_lr_full_model_gpu_controller as controller
import ns_lr_full_model_gpu_owner as owner
from fused_attribution_validation_test_support import keeper, record
from ns_lr_full_model_predecessor_test_support import generation
from selected_lr_release_owner import read, sha256

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def execution_candidate(tmp_path: Path, monkeypatch, previous: dict) -> tuple[Path, Path, dict]:
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    record(root / "data/experiment_receipt.json", {"fixture": "immutable actual-LR CPU data"})
    monkeypatch.setattr(source, "inherited", lambda root: previous)
    return root, candidate, source.prepare(root, candidate)


def completed_cpu(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    out = Path(kwargs["cwd"])
    xml = ET.Element("testsuites")
    for module, count in source.TEST_CASE_COUNTS.items():
        suite = ET.SubElement(xml, "testsuite", tests=str(count), failures="0", errors="0", skipped="0")
        for index in range(count):
            ET.SubElement(suite, "testcase", classname=module, name=f"external_cpu_boundary_{index}")
    ET.ElementTree(xml).write(out / "tests.xml")
    kwargs["stdout"].write("External CPU runner fixture complete\n")
    return subprocess.CompletedProcess(command, 0)


@contextmanager
def locked(path: Path):
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield stream.fileno()


def validator_program(helpers: Path) -> Path:
    path = helpers / "validator_fixture.py"
    path.write_text(
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
    return path


def physical_owner(tmp_path: Path, monkeypatch) -> tuple[Path, Path, dict, dict, list, dict]:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    binding, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    record(binding, {"fixture": "frozen actual-LR owner"})
    accepted = keeper()
    own = generation(810001, 1, owner.controller_command(tmp_path))
    validator = generation(810002, own["pid"], owner.validator_command(tmp_path, candidate))
    launcher = generation(810003, validator["pid"], [contract.PYTHON, "CPU-launcher"])
    rank = generation(810004, launcher["pid"], [contract.PYTHON, "-u", "-m", "experiment.train", "--config", "CPU"])
    intent = {
        "status": "acknowledged",
        "pid": own["pid"],
        "start_ticks": own["start_ticks"],
        "generation": own,
        "command": own["command"],
        "binding_sha256": sha256(binding),
        "marker": "original-controller-marker",
    }
    record(helpers.parent / "controller.json", intent)
    frozen = {
        "candidate": str(candidate),
        "allocation_predecessor": {"restored_keeper": accepted},
        "devices": accepted["devices"],
    }
    state = {
        "status": "validating",
        "spawn_status": "acknowledged",
        "binding_sha256": sha256(binding),
        "controller_pid": own["pid"],
        "controller_start_ticks": own["start_ticks"],
        "allocation_predecessor": frozen["allocation_predecessor"],
        "keeper_release_complete": True,
        "keeper": {
            "pid": accepted["pid"],
            "start_ticks": accepted["generation"]["100"]["start_ticks"],
            "generation": [
                [int(pid), item["start_ticks"], item["group"]] for pid, item in accepted["generation"].items()
            ],
        },
        "command": validator["command"],
        "child_pid": validator["pid"],
        "child_start_ticks": validator["start_ticks"],
        "child_generation": validator,
        "child_marker": "original-validator-marker",
        "active_command": launcher["command"],
        "lock_descriptors": {"node": 8, "manager": 9},
    }
    record(state_path, state)
    processes = {entry["pid"]: entry for entry in (own, validator)}
    marked = {entry["pid"]: entry for entry in (validator, launcher, rank)}
    occupied = [[str(rank["pid"]), "GPU-3"]]
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
    return tmp_path, candidate, state, frozen, occupied, marked


def completed_terminal(tmp_path: Path, monkeypatch) -> tuple[dict, dict, dict, dict]:
    binding, helpers, state_path, _, result_path = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    restored = keeper()
    frozen = {
        "candidate": str(candidate),
        "devices": restored["devices"],
        "allocation_predecessor": {"fixture": "original lower-LR terminal"},
    }
    record(binding, frozen)
    argv = owner.controller_command(tmp_path)
    own = generation(820001, 1, argv)
    intent = {
        "status": "acknowledged",
        "pid": own["pid"],
        "start_ticks": own["start_ticks"],
        "command": argv,
        "generation": own,
        "marker": "original-terminal-controller",
        "binding_sha256": sha256(binding),
        "time": "2026-11-19T00:00:01+00:00",
    }
    record(helpers.parent / "controller.json", intent)
    for name in ("controller.log", "validator.log"):
        (helpers.parent / name).write_text("Original completed external actual-LR fixture\n")
    result = {
        "accepted_cases": ["finite-map-passed"],
        "rejected_cases": ["finite-map-rejected"],
        "all_cases_accepted": False,
        "time": "2026-11-19T00:00:02+00:00",
    }
    record(result_path, result)
    child = generation(820002, own["pid"], owner.validator_command(tmp_path, candidate))
    state = {
        "status": "validation_complete",
        "binding_sha256": sha256(binding),
        "controller_pid": own["pid"],
        "controller_start_ticks": own["start_ticks"],
        "child_pid": child["pid"],
        "child_start_ticks": child["start_ticks"],
        "child_generation": child,
        "child_marker": "original-terminal-validator",
        "command": child["command"],
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
    record(state_path, state)
    monkeypatch.setattr(owner, "require_binding", lambda root: frozen)
    monkeypatch.setattr(controller, "local_node", lambda: owner.NODE)
    monkeypatch.setattr(controller.handoff, "observed", lambda pid: None)
    monkeypatch.setattr(controller.handoff, "marker_processes", lambda *args: {})
    monkeypatch.setattr(controller.handoff, "keeper_observation", lambda: restored)
    monkeypatch.setattr(controller.validation, "reconstruct", lambda root, candidate: read(result_path))
    monkeypatch.setattr(controller, "keeper_guard", lambda: locked(tmp_path / "manager.lock"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return state, intent, result, restored


def failed_execution(tmp_path: Path, monkeypatch, outcome: str) -> tuple[dict, list[str], Path]:
    binding, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    record(binding, {"fixture": "external actual-LR CPU binding"})
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
    program = validator_program(helpers)
    argv = [sys.executable, str(program), str(state_path), outcome, str(tmp_path / "detached.pid")]
    monkeypatch.setattr(owner, "validator_command", lambda root, candidate: argv)
    monkeypatch.setattr(owner, "TIMEOUT", 0.25 if outcome == "timeout" else 10)
    monkeypatch.setattr(contract, "ENVIRONMENT", contract.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": ""})
    monkeypatch.setattr(
        owner,
        "require_binding",
        lambda root: {"candidate": str(tmp_path), "allocation_predecessor": {"restored_keeper": accepted}},
    )
    monkeypatch.setattr(owner, "require_predecessor_keeper", lambda *args: accepted)
    monkeypatch.setattr(controller, "process_context", lambda *args: own)
    monkeypatch.setattr(controller, "local_node", lambda: owner.NODE)
    events, manager_path, node_path = [], tmp_path / "manager.lock", tmp_path / "node.lock"

    @contextmanager
    def node(*args):
        with locked(node_path):
            yield []

    def release(current, state, path, descriptor):
        assert current == accepted and os.readlink(f"/proc/{os.getpid()}/fd/{descriptor}") == str(manager_path)
        state.update(keeper_release_attempted=True, keeper_release_complete=True)
        events.append("released")

    def restore(descriptor):
        assert os.readlink(f"/proc/{os.getpid()}/fd/{descriptor}") == str(manager_path)
        events.append("restored")

    monkeypatch.setattr(controller, "keeper_guard", lambda: locked(manager_path))
    monkeypatch.setattr(controller, "node_guard", node)
    monkeypatch.setattr(controller, "lock_descriptor", lambda path: 10)
    monkeypatch.setattr(controller, "idle_training", lambda: None)
    monkeypatch.setattr(controller, "release_accepted_keeper", release)
    monkeypatch.setattr(controller, "restore_accepted_keeper", restore)
    monkeypatch.setattr(controller.handoff, "keeper_observation", lambda: restored)
    monkeypatch.setattr(controller.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0))
    if outcome == "spawn_failure":
        monkeypatch.setattr(controller.subprocess, "Popen", fail_spawn)
    return restored, events, node_path


def fail_spawn(*args, **kwargs):
    raise OSError("External CPU process spawn failure")
