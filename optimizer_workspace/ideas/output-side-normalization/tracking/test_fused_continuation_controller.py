# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the continuation controller's real bounded child and PID-safe failure cleanup."""

import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import fused_continuation_controller as controller
import pytest
from fused_attribution_validation_contract import trial_specs
from fused_attribution_validation_test_support import record
from fused_continuation_test_support import candidate_fixture, imported_controller, write_process
from selected_lr_failure_handoff import pidfd_open

SYNTHETIC_VALIDATOR = """
import json, os, select, signal, subprocess, sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--root') + 1])
path = root / 'run-preparation/fused-continuation-state-20261118.json'
poll = select.poll()
while not path.exists() or json.loads(path.read_bytes()).get('child_pid') != os.getpid():
    poll.poll(1)
if os.environ.get('CONTINUATION_CPU_TIMEOUT') == '1':
    signal.pause()
if os.environ.get('CONTINUATION_CPU_ORPHAN') == '1':
    read_fd, write_fd = os.pipe()
    child = subprocess.Popen([sys.executable, '-c',
        'import os,signal,sys;os.write(int(sys.argv[1]),b"R");signal.pause()', str(write_fd)],
        pass_fds=(write_fd,), start_new_session=True)
    os.close(write_fd)
    assert os.read(read_fd, 1) == b'R'
    print(child.pid, flush=True)
    sys.exit(7)
print(json.dumps({'pid': os.getpid(), 'mask': os.environ['CUDA_VISIBLE_DEVICES']}), flush=True)
"""


@pytest.fixture
def accepted_executor(tmp_path, monkeypatch):
    root, candidate, source = tmp_path / "root", tmp_path / "candidate", tmp_path / "source"
    cpu_path = candidate_fixture(root, candidate, source, controller)
    real_run = subprocess.run
    execution = {"code": 0, "after_suite": None}

    def cpu_boundary(command, **kwargs):
        if command == controller.cpu_command():
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            kwargs["stdout"].write("synthetic external executor suite\n")
            if execution["after_suite"]:
                execution["after_suite"](Path(kwargs["cwd"]))
            return subprocess.CompletedProcess(command, execution["code"])
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", cpu_boundary)
    return SimpleNamespace(
        root=root, candidate=candidate, source=source, cpu=cpu_path, temporary=tmp_path, execution=execution
    )


@pytest.mark.parametrize("fault", ("none", "suite", "log_after_receipt", "fixture"))
def test_executor_cpu_acceptance_verifies_issued_archive_and_preserves_terminal_failures(
    accepted_executor, monkeypatch, fault
):
    item = accepted_executor
    output = item.temporary / "executor-cpu"
    if fault == "suite":
        item.execution["code"] = 7
    elif fault == "fixture":
        item.execution["after_suite"] = lambda out: (out / controller.fixture_names()[0]).write_text("{}\n")
    elif fault == "log_after_receipt":
        replace = Path.replace
        changed = []

        def during_save(source, target):
            result = replace(source, target)
            if Path(target) == output / "receipt.json" and not changed:
                changed.append(True)
                (output / "controller.log").write_text("replaced after provisional success\n")
            return result

        monkeypatch.setattr(Path, "replace", during_save)
    with imported_controller(item.source) as archived:
        if fault == "none":
            result = archived.controller_cpu(item.root, item.cpu, output)
            assert archived.require_controller_cpu(item.root, output / "receipt.json", item.cpu) == result
        else:
            with pytest.raises(RuntimeError):
                archived.controller_cpu(item.root, item.cpu, output)
            result = archived.read(output / "receipt.json")
            assert result["status"] == "needs_attention"
    assert result["exit_code"] == (7 if fault == "suite" else 0)
    assert result["archive_sha256"] == controller.sha256(Path(result["archive"]))
    assert result["cuda_visible_devices"] == ""
    assert not any(path.exists() for path in controller.locations(item.root)[::2])


@pytest.fixture
def frozen_executor(accepted_executor):
    item = accepted_executor
    output = item.temporary / "executor-cpu"
    with imported_controller(item.source) as archived:
        archived.controller_cpu(item.root, item.cpu, output)
        item.controller_cpu = output / "receipt.json"
        item.frozen = archived.freeze(item.root, item.cpu, item.controller_cpu)
    return item


def test_executor_freeze_roundtrips_through_own_fresh_archived_checker_and_refuses_reuse(frozen_executor):
    item = frozen_executor
    _, helpers, state, work, receipt = controller.locations(item.root)
    before = controller.candidate_files(item.candidate)
    command = [
        controller.PYTHON,
        "-P",
        "-B",
        str(helpers / "fused_continuation_controller.py"),
        "--root",
        str(item.root),
        "--phase",
        "check",
    ]
    checked = subprocess.run(
        command,
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(helpers), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["status"] == controller.FROZEN_STATUS
    assert len(item.frozen["commands"]) == 6
    assert controller.candidate_files(item.candidate) == before
    with imported_controller(item.source) as archived:
        with pytest.raises(FileExistsError):
            archived.freeze(item.root, item.cpu, item.controller_cpu)
    assert not state.exists() and not work.exists() and not receipt.exists()
    (helpers / "fused_continuation_validation.py").write_text("# replaced validator\n")
    checked = subprocess.run(
        command,
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(helpers), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert checked.returncode != 0 and not state.exists() and not work.exists()


def test_executor_start_never_retries_existing_intent_or_touches_keeper(frozen_executor, monkeypatch):
    item = frozen_executor
    _, helpers, state, work, receipt = controller.locations(item.root)
    original = {"status": "starting", "unknown": "original intent"}
    record(helpers.parent / "controller.json", original)

    def node(command, **kwargs):
        assert command == ["hostname", "-I"]
        return controller.NODE

    real_spawn = subprocess.Popen

    def refuse_spawn(command, **kwargs):
        if command[:2] == [controller.PYTHON, "-P"] and "prepare_fused_continuation.py" in command[2]:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            return real_spawn(command, **kwargs)
        raise AssertionError("Prior intent must refuse before execution or keeper observation")

    for name, value in controller.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(subprocess, "check_output", node)
    monkeypatch.setattr(subprocess, "Popen", refuse_spawn)
    with imported_controller(helpers) as archived:
        with pytest.raises(FileExistsError):
            archived.start(item.root)
    assert controller.read(helpers.parent / "controller.json") == original
    assert not state.exists() and not work.exists() and not receipt.exists()


def test_executor_start_acknowledges_exact_detached_process_without_releasing_keeper(frozen_executor, monkeypatch):
    item = frozen_executor
    root = item.root
    _, helpers, state, work, receipt = controller.locations(root)
    proc = item.temporary / "proc"
    proc.mkdir()
    pointer = item.temporary / "keeper.pid"
    pointer.write_text("100")
    accepted = item.frozen["allocation_predecessor"]["restored_keeper"]
    previous = item.frozen["allocation_predecessor"]
    for process in accepted["generation"].values():
        write_process(proc, process, root, {previous["controller_marker_name"]: previous["controller_marker"]})
    real_spawn = subprocess.Popen

    def devices(command, **kwargs):
        if command == ["hostname", "-I"]:
            return controller.NODE
        if command == ["ps", "-eo", "args="]:
            return ""
        if "--query-gpu=index,uuid,name" in command:
            return "\n".join(",".join(row) for row in accepted["devices"])
        if "--query-compute-apps=pid,gpu_uuid" in command:
            return "\n".join(",".join(row) for row in accepted["occupied"])
        raise AssertionError(f"Unexpected start observation: {command}")

    def spawn(command, **kwargs):
        if command[:2] == [controller.PYTHON, "-P"] and "prepare_fused_continuation.py" in command[2]:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            return real_spawn(command, **kwargs)
        assert command == controller.controller_command(root)
        assert kwargs["start_new_session"] is True and kwargs["cwd"] == helpers
        process = {
            "pid": 25001,
            "parent": os.getpid(),
            "status": "S",
            "group": 25001,
            "session": 25001,
            "start_ticks": "new-controller",
            "command": command,
        }
        write_process(proc, process, helpers, kwargs["env"])
        return SimpleNamespace(pid=25001, poll=lambda: None)

    def refuse_signal(*args):
        raise AssertionError("Starting a diagnostic controller must not release the keeper")

    for name, value in controller.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    with imported_controller(helpers) as archived:
        failure = sys.modules["selected_lr_failure_handoff"]
        monkeypatch.setattr(archived.handoff, "PROC_ROOT", proc)
        monkeypatch.setattr(archived.handoff, "KEEPER_PID", pointer)
        monkeypatch.setattr(failure, "KEEPER_PID", pointer)
        monkeypatch.setattr(failure, "MANAGER_LOCK", item.temporary / "manager.lock")
        monkeypatch.setattr(failure, "KEEPER", str(helpers / controller.MANAGER))
        monkeypatch.setattr(subprocess, "check_output", devices)
        monkeypatch.setattr(subprocess, "Popen", spawn)
        monkeypatch.setattr(signal, "pidfd_send_signal", refuse_signal)
        result = archived.start(root)
        assert archived.read(helpers.parent / "controller.json") == result
        assert result["status"] == "acknowledged" and result["pid"] == 25001
        assert result["start_ticks"] == "new-controller"
        assert result["generation"]["command"] == controller.controller_command(root)
        with pytest.raises(FileExistsError):
            archived.start(root)
    assert pointer.read_text() == "100"
    assert not state.exists() and not work.exists() and not receipt.exists()


@pytest.mark.parametrize("fault", ("validator", "rank_swap", "partial_release", "restore"))
def test_archived_executor_refuses_keeper_drift_and_restores_only_complete_owned_transfer(
    frozen_executor, monkeypatch, fault
):
    item = frozen_executor
    root = item.root
    binding, helpers, state_path, _, receipt = controller.locations(root)
    proc = item.temporary / "proc"
    proc.mkdir()
    accepted = copy.deepcopy(item.frozen["allocation_predecessor"]["restored_keeper"])
    occupied = copy.deepcopy(accepted["occupied"])
    if fault == "rank_swap":
        occupied[0][1], occupied[1][1] = occupied[1][1], occupied[0][1]
    pointer, manager_lock = item.temporary / "keeper.pid", item.temporary / "manager.lock"
    manager = helpers / controller.MANAGER
    marker = item.frozen["allocation_predecessor"]
    for process in accepted["generation"].values():
        write_process(proc, process, root, {marker["controller_marker_name"]: marker["controller_marker"]})
    pointer.write_text(str(accepted["pid"]))
    own = {
        "pid": os.getpid(),
        "parent": 1,
        "status": "R",
        "group": os.getpid(),
        "session": os.getpid(),
        "start_ticks": "cpu-original",
        "command": controller.controller_command(root),
    }
    environment = controller.ENVIRONMENT | {controller.CONTROLLER_MARKER: "owned-cpu-controller"}
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    write_process(proc, own, helpers, environment)
    record(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "pid": own["pid"],
            "start_ticks": own["start_ticks"],
            "generation": own | {"parent": 999},
            "command": own["command"],
            "marker": "owned-cpu-controller",
            "binding_sha256": controller.sha256(binding),
            "allocation_predecessor": item.frozen["allocation_predecessor"],
        },
    )
    events, fd_pids = [], {}
    real_run, real_spawn = subprocess.run, subprocess.Popen

    def devices(command, **kwargs):
        if command == ["hostname", "-I"]:
            return controller.NODE
        if command == ["ps", "-eo", "args="]:
            return ""
        if "--query-gpu=index,uuid,name" in command:
            return "\n".join(",".join(row) for row in accepted["devices"])
        if "--query-compute-apps=pid,gpu_uuid" in command:
            return "\n".join(",".join(row) for row in occupied)
        if "--query-compute-apps=pid" in command:
            return "\n".join(row[0] for row in occupied)
        raise AssertionError(f"Unanticipated device boundary: {command}")

    def syscall(number, pid, flags):
        handle, writer = os.pipe()
        os.write(writer, b"ready")
        os.close(writer)
        fd_pids[handle] = pid.value
        return handle

    def signal_fd(handle, signum, *args):
        assert signum == signal.SIGTERM
        pid = fd_pids[handle]
        events.append(("release", pid))
        if fault == "partial_release" and pid == 101:
            return
        shutil.rmtree(proc / str(pid))
        occupied[:] = [row for row in occupied if int(row[0]) != pid]

    class FailedValidator:
        def __init__(self, command, **kwargs):
            assert command == controller.validator_command(root, item.candidate)
            self.pid, self.returncode = 25001, None
            child = {
                "pid": self.pid,
                "parent": own["pid"],
                "status": "R",
                "group": self.pid,
                "session": self.pid,
                "start_ticks": "child-original",
                "command": command,
            }
            write_process(proc, child, item.candidate, kwargs["env"])
            events.append(("validator", self.pid))

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 7
            shutil.rmtree(proc / str(self.pid))
            return self.returncode

    def spawn(command, **kwargs):
        if command[:2] == [controller.PYTHON, "-P"] and "prepare_fused_continuation.py" in command[2]:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            return real_spawn(command, **kwargs)
        return FailedValidator(command, **kwargs)

    def run(command, **kwargs):
        if command == [controller.PYTHON, "-m", "experiment.bootstrap", "--check-only"]:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["bash", str(manager), "start"]:
            assert kwargs["pass_fds"] == (int(kwargs["env"]["GPU_RESOURCE_MANAGER_LOCK_FD"]),)
            events.append(("restore", 300))
            if fault == "restore":
                raise subprocess.CalledProcessError(7, command)
            for offset, previous in enumerate(accepted["generation"].values()):
                pid = 300 + offset
                restored = previous | {
                    "pid": pid,
                    "parent": 1 if offset == 0 else 300,
                    "group": pid,
                    "session": pid,
                    "start_ticks": f"restored-{pid}",
                }
                write_process(proc, restored, root, {})
                if offset:
                    occupied.append([str(pid), accepted["devices"][offset - 1][1]])
            pointer.write_text("300")
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, **kwargs)

    with imported_controller(helpers) as archived:
        failure = sys.modules["selected_lr_failure_handoff"]
        monkeypatch.setattr(archived.handoff, "PROC_ROOT", proc)
        monkeypatch.setattr(archived.handoff, "KEEPER_PID", pointer)
        monkeypatch.setattr(failure, "KEEPER_PID", pointer)
        monkeypatch.setattr(failure, "MANAGER_LOCK", manager_lock)
        monkeypatch.setattr(failure, "KEEPER", str(manager))
        monkeypatch.setattr(failure, "Path", lambda value: proc if value == "/proc" else Path(value))
        monkeypatch.setattr(failure.LIBC, "syscall", syscall)
        monkeypatch.setattr(subprocess, "check_output", devices)
        monkeypatch.setattr(subprocess, "Popen", spawn)
        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setattr(signal, "pidfd_send_signal", signal_fd)
        with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
            archived.validate(root)
    saved = controller.read(state_path)
    assert saved["status"] == "needs_attention" and not receipt.exists()
    if fault == "rank_swap":
        assert events == [] and pointer.read_text() == "100"
        assert "keeper_release_attempted" not in saved
    elif fault == "partial_release":
        assert saved["keeper_release_complete"] is False
        assert saved["keeper_restore"].startswith("refused:")
        assert not any(event[0] in ("validator", "restore") for event in events)
    else:
        assert saved["exit_code"] == saved["cleanup_exit_code"] == 7
        assert saved["keeper_release_complete"] is True
        assert events[-2:] == [("validator", 25001), ("restore", 300)]
        if fault == "restore":
            assert saved["keeper_restore"].startswith("CalledProcessError:")
        else:
            assert saved["keeper_restore"] == 0 and saved["restored_keeper"]["pid"] == 300
    assert not any(event == ("release", pid) for pid in (201, 202) for event in events)


@pytest.mark.parametrize("fault", ("none", "missing_owner", "wrong_fixture"))
def test_controller_archive_retains_direct_owner_dependencies_and_all_trial_fixtures(tmp_path, fault):
    root = tmp_path / "root"
    binding, helpers, _, _, _ = controller.parent_locations(root)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True)
    manager = helpers / controller.MANAGER
    sibling = Path(controller.__file__).with_name(controller.MANAGER)
    manager.write_bytes(sibling.read_bytes() if sibling.is_file() else b"#!/bin/bash\nexit 0\n")
    controller.write(binding, {"helpers": str(helpers), "helpers_sha256": {manager.name: controller.sha256(manager)}})
    sources = controller.helper_sources(root, Path(controller.__file__).parent)
    candidate = tmp_path / "candidate"
    configs = {}
    for spec in trial_specs():
        path = candidate / spec["config_name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"family": spec["family"], "lr": spec["matrix_lr"]}) + "\n")
        configs[spec["config_name"]] = controller.sha256(path)
    out = tmp_path / "cpu"
    copied = out / "tracking"
    copied.mkdir(parents=True)
    archive = out / "source.tar"
    expected = {}
    with tarfile.open(archive, "x") as bundle:
        for name, source in sources.items():
            target = copied / name
            target.write_bytes(source.read_bytes())
            expected[f"tracking/{name}"] = controller.sha256(target)
            if not (fault == "missing_owner" and name == "fused_attribution_validation_owner.py"):
                bundle.add(target, arcname=f"tracking/{name}", recursive=False)
        for name in configs:
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((candidate / name).read_bytes())
            if fault == "wrong_fixture" and name != controller.fixture_names()[0]:
                target.write_text("{}\n")
            bundle.add(target, arcname=name, recursive=False)
            expected[name] = configs[name]
    if fault == "none":
        controller.require_archive(archive, expected, alias=False)
        assert "fused_attribution_validation_owner.py" in sources
        assert len(configs) == 9
        assert {name: controller.sha256(out / name) for name in configs} == configs
    else:
        with pytest.raises(ValueError):
            controller.require_archive(archive, expected, alias=False)


@pytest.mark.parametrize("scenario", ("complete", "orphan", "timeout"))
def test_exact_continuation_child_ack_cleanup_preserves_unrelated_cpu_process(tmp_path, monkeypatch, scenario):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    candidate.mkdir()
    _, helpers, state_path, _, _ = controller.locations(root)
    helpers.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    (helpers / "fused_continuation_validation.py").write_text(SYNTHETIC_VALIDATOR)
    monkeypatch.setattr(controller, "ENVIRONMENT", controller.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": ""})
    monkeypatch.setenv("CONTINUATION_CPU_TIMEOUT", "1" if scenario == "timeout" else "0")
    monkeypatch.setenv("CONTINUATION_CPU_ORPHAN", "1" if scenario == "orphan" else "0")
    if scenario == "timeout":
        monkeypatch.setattr(controller, "TIMEOUT", 0.1)
    peer = subprocess.Popen([controller.PYTHON, "-c", "import signal;signal.pause()"], start_new_session=True)
    handle = pidfd_open(peer.pid)

    def forbid_numeric_signal(*args):
        raise AssertionError("Continuation cleanup must address exact pidfds")

    monkeypatch.setattr(os, "kill", forbid_numeric_signal)
    monkeypatch.setattr(os, "killpg", forbid_numeric_signal)
    try:
        state = {}
        if scenario == "timeout":
            with pytest.raises(subprocess.TimeoutExpired):
                controller.bounded_validator(root, {"candidate": str(candidate)}, state)
            code = -signal.SIGTERM
        else:
            code = controller.bounded_validator(root, {"candidate": str(candidate)}, state)
        saved = controller.read(state_path)
        assert code == {"complete": 0, "orphan": 7, "timeout": -signal.SIGTERM}[scenario]
        assert saved["spawn_status"] == "acknowledged" and saved["marked_descendants_absent"] is True
        assert saved["cleanup_exit_code"] == code
        assert saved["log_sha256"] == controller.sha256(helpers.parent / "validator.log")
        assert not controller.handoff.marker_processes(controller.CHILD_MARKER, saved["child_marker"])
        assert peer.poll() is None
        if scenario == "orphan":
            child_pid = int((helpers.parent / "validator.log").read_text())
            process = controller.handoff.observed(child_pid)
            assert str(child_pid) in saved["child_generation"]
            assert process is None or process["status"] in ("Z", "X", "x")
        elif scenario == "complete":
            assert json.loads((helpers.parent / "validator.log").read_text())["mask"] == ""
    finally:
        signal.pidfd_send_signal(handle, signal.SIGTERM)
        peer.wait(timeout=10)
        os.close(handle)
