# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only process and allocation boundaries for training-path lifecycle tests."""

import ctypes
import fcntl
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import ns_training_path_lr_contract as contract
import ns_training_path_lr_gpu_controller as controller
import ns_training_path_lr_gpu_owner as owner
from fused_attribution_validation_controller import acknowledged_process
from fused_attribution_validation_test_support import keeper, record
from selected_lr_release_owner import read, sha256


def generation(pid: int, parent: int, command: list[str]) -> dict:
    return {
        "pid": pid,
        "parent": parent,
        "group": pid,
        "session": pid,
        "start_ticks": str(pid + 1000),
        "command": command,
        "status": "R",
    }


@contextmanager
def locked(path: Path):
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield stream.fileno()


def validator_program(helpers: Path) -> Path:
    path = helpers / "validator_fixture.py"
    path.write_text(
        "import json,os,select,signal,subprocess,sys\nfrom pathlib import Path\n"
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


def cpu_validator_io(monkeypatch) -> None:
    """Keep real child processes CUDA-hidden without changing protocol globals."""
    spawn = subprocess.Popen

    def cpu_spawn(*args, **kwargs):
        kwargs["env"] = kwargs["env"] | {"CUDA_VISIBLE_DEVICES": ""}
        return spawn(*args, **kwargs)

    def acknowledge(child, command, cwd, marker_name, marker, environment):
        return acknowledged_process(
            child,
            command,
            cwd,
            marker_name,
            marker,
            environment | {"CUDA_VISIBLE_DEVICES": ""},
        )

    monkeypatch.setattr(controller.subprocess, "Popen", cpu_spawn)
    monkeypatch.setattr(controller, "acknowledged_process", acknowledge)


def cuda_driver(monkeypatch, devices: list[list[str]]) -> tuple:
    """Expose only device identity calls through a CPU fake of the driver library."""
    identifiers = [row[1] for row in devices]

    def count(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = len(identifiers)
        return 0

    def device(pointer, ordinal):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = ordinal + 100
        return 0

    def identifier(pointer, handle):
        value = ctypes.cast(pointer, ctypes.POINTER(owner.CUDAUUID)).contents
        value.bytes[:] = UUID(identifiers[handle - 100].removeprefix("GPU-")).bytes
        return 0

    driver = SimpleNamespace(
        cuInit=lambda flags: 0,
        cuDeviceGetCount=count,
        cuDeviceGet=device,
        cuDeviceGetUuid_v2=identifier,
    )

    def library(name):
        if name != "libcuda.so.1":
            raise AssertionError("CPU identity fixture cannot load another library")
        return driver

    monkeypatch.setattr(
        owner,
        "ctypes",
        SimpleNamespace(
            **{name: getattr(ctypes, name) for name in dir(ctypes) if name != "CDLL"},
            CDLL=library,
        ),
    )
    return identifiers, driver


def physical_owner(tmp_path: Path, monkeypatch) -> tuple:
    """Persist original ancestry, using only synthetic device/process query boundaries."""
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    binding, helpers, _, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    record(binding, {"fixture": "CPU training-path binding"})
    accepted = keeper()
    accepted["devices"] = [
        [str(index), f"GPU-{UUID(int=index + 1)}", row[2]] for index, row in enumerate(accepted["devices"])
    ]
    cuda_driver(monkeypatch, accepted["devices"])
    own = generation(os.getpid(), os.getppid(), owner.controller_command(tmp_path))
    validator = generation(810002, own["pid"], owner.validator_command(tmp_path, candidate))
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
        str(binding),
        "--max-steps",
        "24",
    ]
    launcher = generation(810003, validator["pid"], argv)
    rank = generation(810004, launcher["pid"], [contract.PYTHON, "-u", *argv[5:]])
    record(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "pid": own["pid"],
            "start_ticks": own["start_ticks"],
            "generation": own,
            "command": own["command"],
            "binding_sha256": sha256(binding),
            "marker": "controller-marker",
        },
    )
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
        "child_marker": "validator-marker",
        "active_command": argv,
        "lock_descriptors": {},
    }
    processes = {entry["pid"]: entry for entry in (own, validator)}
    marked = {entry["pid"]: entry for entry in (validator, launcher, rank)}
    occupied = [[str(rank["pid"]), frozen["devices"][3][1]]]
    monkeypatch.setattr(owner, "require_binding", lambda root: frozen)
    monkeypatch.setattr(owner, "process_context", lambda pid, *args: processes[pid])
    monkeypatch.setattr(owner, "process_details", lambda pid: {"environment": {"LOCAL_RANK": "3", "RANK": "3"}})
    monkeypatch.setattr(owner.handoff, "marker_processes", lambda *args: marked)
    monkeypatch.setattr(owner.handoff, "KEEPER_PID", tmp_path / "keeper.pid")
    monkeypatch.chdir(candidate)
    monkeypatch.setattr(
        owner,
        "os",
        SimpleNamespace(
            **{name: getattr(os, name) for name in dir(os) if name != "environ"},
            environ=dict(os.environ) | contract.ENVIRONMENT,
        ),
    )

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


def completed_terminal(tmp_path: Path, monkeypatch) -> tuple:
    binding, helpers, state_path, _, result_path = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    restored = keeper()
    frozen = {
        "candidate": str(candidate),
        "devices": restored["devices"],
        "allocation_predecessor": {"fixture": "preserved failed NS"},
        "predecessor_acceptance": {"fixture": "accepted receipt/verifier/seal"},
    }
    record(binding, frozen)
    own = generation(820001, 1, owner.controller_command(tmp_path))
    intent = {
        "status": "acknowledged",
        "pid": own["pid"],
        "start_ticks": own["start_ticks"],
        "generation": own,
        "command": own["command"],
        "marker": "terminal-controller",
        "binding_sha256": sha256(binding),
        "time": "2026-11-19T00:00:01+00:00",
    }
    record(helpers.parent / "controller.json", intent)
    for name in ("controller.log", "validator.log"):
        (helpers.parent / name).write_text("CPU training-path lifecycle fixture\n")
    result = {
        "accepted_cases": [],
        "rejected_cases": [spec["name"] for spec in contract.specs()],
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
        "child_marker": "terminal-validator",
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


def failed_execution(tmp_path: Path, monkeypatch, outcome: str) -> tuple:
    binding, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    record(binding, {"fixture": "CPU frozen training-path source"})
    accepted, restored = keeper(), keeper()
    restored["generation"]["100"]["start_ticks"] = "restored-generation"
    own = generation(os.getpid(), os.getppid(), owner.controller_command(tmp_path))
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
    cpu_validator_io(monkeypatch)
    monkeypatch.setattr(
        owner,
        "require_binding",
        lambda root: {
            "candidate": str(tmp_path),
            "allocation_predecessor": {"restored_keeper": accepted},
            "predecessor_acceptance": {"seal_sha256": "CPU-fixture-seal"},
            "cpu_receipt_sha256": "CPU-fixture-receipt",
            "authenticated_files": {"CPU-fixture": {"sha256": "CPU-fixture"}},
        },
    )
    monkeypatch.setattr(owner, "require_predecessor_keeper", lambda *args: accepted)
    monkeypatch.setattr(controller, "process_context", lambda *args: own)
    monkeypatch.setattr(controller, "local_node", lambda: owner.NODE)
    events = []
    manager_path = tmp_path / "manager.lock"
    node_path = tmp_path / "run-preparation" / f".wave7-node-{owner.NODE}.lock"

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
    monkeypatch.setattr(controller, "idle_training", lambda: None)
    monkeypatch.setattr(controller, "release_accepted_keeper", release)
    monkeypatch.setattr(controller, "restore_accepted_keeper", restore)
    monkeypatch.setattr(controller.handoff, "keeper_observation", lambda: restored)
    monkeypatch.setattr(controller.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0))
    if outcome == "spawn_failure":
        monkeypatch.setattr(controller.subprocess, "Popen", fail_spawn)
    return restored, events, node_path


def fail_spawn(*args, **kwargs):
    raise OSError("CPU child-spawn fixture failure")
